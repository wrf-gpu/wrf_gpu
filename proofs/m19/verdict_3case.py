"""M19 3-case fail-fast verdict driver.

PRINCIPAL SPEC (binding)
------------------------
Validation = 3 cases, NOT 30 (the principal runs more after v0.1.0).
FAIL-FAST + final-confirmation:

  * Run CASE 1 first on BOTH L2 and L3.
  * If case 1 FAILS on EITHER level -> STOP. Do not run cases 2/3. Emit a
    probe-pointer JSON saying which field / lead / level failed and why.
  * If case 1 PASSES on both -> run cases 2 and 3 to CONFIRM.

This is the END-OF-CORE-DEV stamp, not a gate that slows core dev. The real
gate is the 24-72h leads (the +1h/+3h skill is already validated, see
proofs/coupled/task2_skill_signal_{1,3}h.json). CPU-WRF truth is REUSED from the
existing corpus -- this driver launches NO new WRF runs.

GPU SAFETY
----------
The actual GPU forecast call lives in ONE clearly-marked function,
``run_and_score_real`` (the "REAL FORECAST CALL" block). It is GUARDED behind
``--execute``. Without ``--execute`` (the default) the driver runs in DRY mode:
case discovery + scorer-wiring import-check + fail-fast control-flow simulation,
using synthetic scores -- NO GPU, NO forecast. The manager flips this on by
passing ``--execute`` once the model's 24h stability is confirmed and the GPU is
free.

USAGE
-----
  # Dry run (NO GPU) -- proves discovery + wiring + control flow:
  python proofs/m19/verdict_3case.py --dry-run-out proofs/m19/verdict_dryrun.json

  # Simulate a case-1 FAIL path (still NO GPU):
  python proofs/m19/verdict_3case.py --simulate-case1-fail

  # REAL verdict (manager flips this on; needs GPU + confirmed 24h stability):
  PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 \
    taskset -c 0-3 python proofs/m19/verdict_3case.py --execute \
      --leads 24 48 72 --out proofs/m19/verdict_result.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "cases_manifest.json"

# ---------------------------------------------------------------------------
# PASS/FAIL THRESHOLDS  -- TUNABLE.  The manager should review/adjust these.
# ---------------------------------------------------------------------------
# Rationale: the GPU model is already validated at +1h/+3h on the 20260521 pin:
#     +1h: T2 RMSE 1.33 K, U10 2.22 m/s, V10 3.69 m/s
#     +3h: T2 RMSE 1.83 K, U10 1.90 m/s, V10 2.75 m/s
# M19 is the FIRST 24-72h check. We do not yet know the true 24-72h CPU-WRF-vs-
# CPU-WRF spread, so these are deliberately GENEROUS "not-blown-up / still-in-the
# -right-ballpark" gates, not tight equivalence gates (tight TOST is M20/task#25).
# Two independent failure modes are checked:
#   (A) ABSOLUTE CEILING per field -- ~2x the +3h validated RMSE, with a small
#       additive slack for longer-lead error growth. If GPU RMSE exceeds this at
#       any scored lead, that field/lead FAILS.
#   (B) NON-FINITE / BLOW-UP -- any non-finite T2/U10/V10, or a field whose
#       GPU spatial std or |mean| exceeds a physical sanity ceiling, is an
#       instant FAIL (the model detonated).
# A field that is finite and under its ceiling at ALL scored leads PASSES; a
# level passes iff all three fields pass; a case passes iff all its levels pass.
RMSE_CEILING = {
    # field: (base_K_or_mps, per_24h_growth_slack)
    "T2": (4.0, 1.0),    # ceiling = 4.0 + 1.0*(lead_h/24)  -> 5/6/7 K at 24/48/72h
    "U10": (5.0, 1.0),   # ceiling = 5.0 + 1.0*(lead_h/24)  -> 6/7/8 m/s
    "V10": (6.0, 1.0),   # ceiling = 6.0 + 1.0*(lead_h/24)  -> 7/8/9 m/s
}
# Physical sanity ceilings (blow-up detector). A coupled near-surface field that
# exceeds these is non-physical -> instant FAIL even if RMSE happened to be low.
SANITY = {
    "T2": {"abs_mean_max": 330.0, "abs_mean_min": 250.0, "std_max": 30.0},
    "U10": {"abs_mean_max": 40.0, "std_max": 40.0},
    "V10": {"abs_mean_max": 40.0, "std_max": 40.0},
}
FIELDS = ("T2", "U10", "V10")
DEFAULT_LEADS = (24, 48, 72)  # the real gate; L3 (24h) caps at 24.


def rmse_ceiling(field: str, lead_h: int) -> float:
    base, slack = RMSE_CEILING[field]
    return base + slack * (lead_h / 24.0)


# ---------------------------------------------------------------------------
# Scoring helpers (pure numpy; identical metric defn to task2_skill_signal._scores)
# ---------------------------------------------------------------------------
def score_fields(gpu: dict[str, Any], wrf: dict[str, Any]) -> dict[str, dict]:
    import numpy as np
    out: dict[str, dict] = {}
    for f in FIELDS:
        g = np.asarray(gpu[f], dtype=np.float64)
        t = np.asarray(wrf[f], dtype=np.float64)
        diff = g - t
        out[f] = {
            "rmse": float(np.sqrt(np.mean(diff**2))),
            "bias": float(np.mean(diff)),
            "mae": float(np.mean(np.abs(diff))),
            "gpu_mean": float(np.mean(g)),
            "gpu_std": float(np.std(g)),
            "wrf_mean": float(np.mean(t)),
            "gpu_finite": bool(np.isfinite(g).all()),
        }
    return out


def evaluate_field(field: str, lead_h: int, s: dict) -> tuple[bool, list[str]]:
    """Return (pass, reasons-on-fail). Applies ceiling (A) + sanity (B)."""
    reasons: list[str] = []
    if not s.get("gpu_finite", True):
        reasons.append(f"{field}@{lead_h}h: GPU field non-finite (blow-up)")
    ceil = rmse_ceiling(field, lead_h)
    if s["rmse"] > ceil:
        reasons.append(
            f"{field}@{lead_h}h: RMSE {s['rmse']:.2f} > ceiling {ceil:.2f}"
        )
    san = SANITY[field]
    if "abs_mean_max" in san and abs(s["gpu_mean"]) > san["abs_mean_max"]:
        reasons.append(
            f"{field}@{lead_h}h: |GPU mean| {abs(s['gpu_mean']):.1f} > {san['abs_mean_max']} (non-physical)"
        )
    if "abs_mean_min" in san and s["gpu_mean"] < san["abs_mean_min"]:
        reasons.append(
            f"{field}@{lead_h}h: GPU mean {s['gpu_mean']:.1f} < {san['abs_mean_min']} (non-physical)"
        )
    if "std_max" in san and s["gpu_std"] > san["std_max"]:
        reasons.append(
            f"{field}@{lead_h}h: GPU std {s['gpu_std']:.1f} > {san['std_max']} (blow-up)"
        )
    return (len(reasons) == 0), reasons


def evaluate_level(level_scores: dict[int, dict]) -> dict:
    """level_scores: {lead_h: {field: scoredict}}. Returns verdict for one level."""
    field_pass = {f: True for f in FIELDS}
    reasons: list[str] = []
    for lead_h in sorted(level_scores):
        for f in FIELDS:
            ok, why = evaluate_field(f, lead_h, level_scores[lead_h][f])
            if not ok:
                field_pass[f] = False
                reasons.extend(why)
    passed = all(field_pass.values())
    return {"passed": passed, "field_pass": field_pass, "fail_reasons": reasons}


# ---------------------------------------------------------------------------
# REAL FORECAST CALL  -- the ONE block the manager flips on with --execute.
# ---------------------------------------------------------------------------
def run_and_score_real(level_spec: dict, leads: list[int], *,
                       segment_steps: int, dt_s: float, acoustic_substeps: int,
                       radiation_cadence_steps: int) -> dict[int, dict]:
    """Run the REAL segmented GPU forecast for one case/level and score it vs the
    corpus CPU-WRF at each requested lead.  *** LAUNCHES A GPU FORECAST. ***

    Returns {lead_h: {field: scoredict}}. Caps requested leads at the level's
    max_lead_h. Uses the SAME config as proofs/coupled/task2_skill_signal.py:
    the operational segmented entry + compute_m9_diagnostics, scored against
    wrfout_d02 at the valid time.
    """
    import jax  # noqa: WPS433  (deferred import so dry-run never touches JAX/GPU)
    import numpy as np
    from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
    from gpuwrf.io.gen2_wrfout_loader import read_wrfout_file
    from gpuwrf.runtime.operational_mode import (
        compute_m9_diagnostics,
        run_forecast_operational_segmented,
    )

    cfg = DailyPipelineConfig(
        run_id=level_spec["run_id"],
        run_root=Path(level_spec["run_root"]),
        domain=level_spec["domain"],
        dt_s=dt_s,
        acoustic_substeps=acoustic_substeps,
        radiation_cadence_steps=radiation_cadence_steps,
    )
    case, run_dir = _build_real_case(cfg)
    time_utc = case.run_start
    nl = dataclasses.replace(
        case.namelist,
        run_physics=True,
        run_boundary=True,
        disable_guards=True,
        radiation_cadence_steps=radiation_cadence_steps,
        time_utc=time_utc,
    )

    max_lead = int(level_spec["max_lead_h"])
    leads = [h for h in leads if h <= max_lead]

    results: dict[int, dict] = {}
    # Run the LONGEST lead once and re-score intermediate leads from their own
    # corpus files would require per-lead final states; the simplest honest path
    # is one forecast per scored lead (segmented entry is compile-cached so the
    # first lead pays compile, the rest reuse). The manager may optimise later by
    # snapshotting intermediate states; correctness-first here.
    for lead_h in leads:
        final_state = run_forecast_operational_segmented(
            case.state, nl, float(lead_h), segment_steps=segment_steps
        )
        jax.block_until_ready(final_state.theta)
        lead_seconds = float(lead_h) * 3600.0
        diags = compute_m9_diagnostics(final_state, nl, lead_seconds)
        gpu = {
            "T2": np.asarray(jax.device_get(diags.t2)),
            "U10": np.asarray(jax.device_get(diags.u10)),
            "V10": np.asarray(jax.device_get(diags.v10)),
        }
        valid = time_utc + timedelta(hours=lead_h)
        wrfout = run_dir / f"wrfout_{level_spec['domain']}_{valid:%Y-%m-%d_%H:%M:%S}"
        if not wrfout.is_file():
            raise FileNotFoundError(f"no CPU-WRF truth at {wrfout}")
        wrf = read_wrfout_file(wrfout, fields=FIELDS)["fields"]
        results[lead_h] = score_fields(gpu, {f: np.asarray(wrf[f]) for f in FIELDS})
    return results


# ---------------------------------------------------------------------------
# DRY scoring -- synthesizes scores to exercise control flow WITHOUT a GPU.
# ---------------------------------------------------------------------------
def run_and_score_dry(level_spec: dict, leads: list[int], *, force_fail: bool) -> dict[int, dict]:
    """Synthesize plausible scores (no GPU). If force_fail, inject a clear
    blow-up at the first lead so the fail-fast path is exercised."""
    max_lead = int(level_spec["max_lead_h"])
    leads = [h for h in leads if h <= max_lead]
    results: dict[int, dict] = {}
    for i, lead_h in enumerate(leads):
        # Plausible PASS-side numbers anchored on the validated +3h levels, grown
        # mildly with lead.  These are SYNTHETIC -- dry-run only.
        grow = 1.0 + 0.15 * (lead_h / 24.0)
        base = {
            "T2": {"rmse": 1.8 * grow, "bias": 0.9, "mae": 1.1 * grow,
                   "gpu_mean": 293.5, "gpu_std": 4.0, "wrf_mean": 293.0, "gpu_finite": True},
            "U10": {"rmse": 1.9 * grow, "bias": 0.3, "mae": 1.4 * grow,
                    "gpu_mean": 1.2, "gpu_std": 3.5, "wrf_mean": 1.0, "gpu_finite": True},
            "V10": {"rmse": 2.8 * grow, "bias": 1.6, "mae": 2.3 * grow,
                    "gpu_mean": -4.8, "gpu_std": 4.5, "wrf_mean": -6.0, "gpu_finite": True},
        }
        if force_fail and i == 0:
            # Inject an unambiguous blow-up in U10 at the first lead.
            base["U10"]["rmse"] = 250.0
            base["U10"]["gpu_mean"] = 180.0
            base["U10"]["gpu_std"] = 120.0
            base["U10"]["gpu_finite"] = False
        results[lead_h] = base
    return results


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def write_probe_pointer(out_dir: Path, case: dict, level_name: str,
                        level_verdict: dict, level_spec: dict) -> Path:
    payload = {
        "verdict": "FAIL",
        "stopped_at": "case1",
        "failed_case": case["case_id"],
        "failed_level": level_name,
        "init_utc": case["init_utc"],
        "run_dir": level_spec["run_dir"],
        "gpu_init_source": level_spec["gpu_init_source"],
        "fail_reasons": level_verdict["fail_reasons"],
        "field_pass": level_verdict["field_pass"],
        "probe_pointer": {
            "what": "First failing field(s)/lead(s) above; re-run the operational "
                    "segmented forecast for this init+level and inspect the M9 "
                    "diagnostics at the FIRST failing lead.",
            "how": f"PYTHONPATH=src python proofs/coupled/task2_skill_signal.py "
                   f"--entry segmented --hours <first-fail-lead>  (after pointing "
                   f"_build_real_case at run_id={level_spec['run_id']}, "
                   f"run_root={level_spec['run_root']}, domain={level_spec['domain']})",
            "note": "A non-finite / std blow-up points at dycore/coupler instability "
                    "at that lead; an over-ceiling-but-finite RMSE points at a skill "
                    "(physics/coupling) regression rather than a crash.",
        },
    }
    p = out_dir / "verdict_probe_pointer.json"
    p.write_text(json.dumps(payload, indent=2) + "\n")
    return p


def run_case(case: dict, leads: list[int], *, execute: bool, force_fail_level: str | None,
             score_kwargs: dict) -> dict:
    """Score every level of a case and return per-level verdicts."""
    level_results: dict[str, dict] = {}
    for level_name, level_spec in case["levels"].items():
        if execute:
            scores = run_and_score_real(level_spec, leads, **score_kwargs)
        else:
            ff = force_fail_level is not None and level_name == force_fail_level
            scores = run_and_score_dry(level_spec, leads, force_fail=ff)
        verdict = evaluate_level(scores)
        level_results[level_name] = {
            "scored_leads_h": sorted(scores),
            "scores": scores,
            "verdict": verdict,
        }
    case_passed = all(lr["verdict"]["passed"] for lr in level_results.values())
    return {"case_id": case["case_id"], "init_utc": case["init_utc"],
            "passed": case_passed, "levels": level_results}


def main() -> int:
    ap = argparse.ArgumentParser(description="M19 3-case fail-fast verdict driver")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--leads", type=int, nargs="+", default=list(DEFAULT_LEADS),
                    help="verification leads in hours (default 24 48 72; L3 caps at 24)")
    ap.add_argument("--execute", action="store_true",
                    help="LAUNCH REAL GPU FORECASTS. Default OFF (dry run, no GPU).")
    ap.add_argument("--out", type=Path, default=HERE / "verdict_result.json")
    ap.add_argument("--dry-run-out", type=Path, default=None,
                    help="if set, write the dry-run proof here and emit BOTH a "
                         "pass-path and a fail-path simulation")
    ap.add_argument("--simulate-case1-fail", action="store_true",
                    help="dry-run: inject a case-1 L3 blow-up to exercise fail-fast")
    # forecast knobs (only used with --execute)
    ap.add_argument("--segment-steps", type=int, default=180)
    ap.add_argument("--dt-s", type=float, default=10.0)
    ap.add_argument("--acoustic-substeps", type=int, default=10)
    ap.add_argument("--radiation-cadence-steps", type=int, default=180)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    cases = {c["case_id"]: c for c in manifest["cases"]}
    score_kwargs = dict(
        segment_steps=args.segment_steps, dt_s=args.dt_s,
        acoustic_substeps=args.acoustic_substeps,
        radiation_cadence_steps=args.radiation_cadence_steps,
    )

    out_dir = (args.dry_run_out.parent if args.dry_run_out else args.out.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    def run_fail_fast(force_fail_level: str | None) -> dict:
        """Execute the fail-fast control flow. Returns the full verdict payload."""
        t0 = time.time()
        timeline: list[str] = []
        case1 = cases["case1"]
        timeline.append("run case1 on BOTH levels (L2 + L3)")
        c1 = run_case(case1, args.leads, execute=args.execute,
                      force_fail_level=force_fail_level, score_kwargs=score_kwargs)
        payload: dict[str, Any] = {
            "mode": "EXECUTE(real-GPU)" if args.execute else "DRY(no-GPU,synthetic-scores)",
            "leads_h": args.leads,
            "thresholds": {
                "rmse_ceiling_formula": {f: f"{b} + {g}*(lead_h/24)"
                                         for f, (b, g) in RMSE_CEILING.items()},
                "sanity": SANITY,
                "TUNABLE": True,
                "anchor": "validated +3h RMSE T2 1.83 / U10 1.90 / V10 2.75; "
                          "ceilings ~2x +3h with lead-growth slack",
            },
            "results": {"case1": c1},
        }
        if not c1["passed"]:
            # FAIL-FAST: identify the first failing level, write probe-pointer, STOP.
            failed_level = next(ln for ln, lr in c1["levels"].items()
                                if not lr["verdict"]["passed"])
            timeline.append(f"case1 FAILED on level {failed_level} -> STOP "
                            f"(cases 2/3 NOT run), emit probe-pointer")
            probe = write_probe_pointer(out_dir, case1, failed_level,
                                        c1["levels"][failed_level]["verdict"],
                                        case1["levels"][failed_level])
            payload["verdict"] = "FAIL"
            payload["stopped_at"] = "case1"
            payload["probe_pointer_file"] = str(probe)
        else:
            timeline.append("case1 PASSED both levels -> run cases 2 and 3 to confirm")
            for cid in ("case2", "case3"):
                payload["results"][cid] = run_case(
                    cases[cid], args.leads, execute=args.execute,
                    force_fail_level=None, score_kwargs=score_kwargs)
            all_pass = all(payload["results"][cid]["passed"]
                           for cid in payload["results"])
            payload["verdict"] = "PASS" if all_pass else "FAIL"
            payload["stopped_at"] = None
            timeline.append(f"final verdict = {payload['verdict']}")
        payload["control_flow_timeline"] = timeline
        payload["wall_s"] = round(time.time() - t0, 2)
        payload["generated_utc"] = datetime.now(timezone.utc).isoformat()
        return payload

    if args.dry_run_out is not None:
        # DRY-RUN PROOF: prove discovery + wiring + BOTH control-flow branches.
        import importlib.util as _ilu
        wiring = {
            "manifest_loaded": True,
            "n_cases": len(cases),
            "case_ids": list(cases),
            "case1_has_L2_and_L3": set(cases["case1"]["levels"]) >= {"L2", "L3"},
            "all_gpu_init_sources_exist": all(
                Path(lv["gpu_init_source"]).is_file()
                for c in cases.values() for lv in c["levels"].values()),
            "all_truth_files_exist": all(
                Path(p).is_file()
                for c in cases.values() for lv in c["levels"].values()
                for p in lv["wrfout_d02_truth"].values()),
            "scorer_importable": _ilu.find_spec("gpuwrf.runtime.operational_mode") is not None,
            "scorer_module_path": "src/gpuwrf/runtime/operational_mode.py "
                                  "(run_forecast_operational_segmented + compute_m9_diagnostics)",
            "tost_scorer_present_on_branch": (HERE.parent / "m20" / "paired_tost_scorer.py").is_file(),
        }
        pass_path = run_fail_fast(force_fail_level=None)
        fail_path = run_fail_fast(force_fail_level="L3")
        proof = {
            "scope": "M19 harness DRY-RUN proof -- NO GPU, synthetic scores. Proves "
                     "case discovery, scorer-wiring import, corpus-path resolution, "
                     "and BOTH fail-fast control-flow branches.",
            "gpu_used": False,
            "wiring": wiring,
            "simulated_pass_path": {
                "verdict": pass_path["verdict"],
                "stopped_at": pass_path["stopped_at"],
                "cases_run": list(pass_path["results"]),
                "timeline": pass_path["control_flow_timeline"],
            },
            "simulated_fail_path": {
                "verdict": fail_path["verdict"],
                "stopped_at": fail_path["stopped_at"],
                "cases_run": list(fail_path["results"]),
                "timeline": fail_path["control_flow_timeline"],
                "probe_pointer_file": fail_path.get("probe_pointer_file"),
            },
            "thresholds": pass_path["thresholds"],
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }
        Path(args.dry_run_out).write_text(json.dumps(proof, indent=2) + "\n")
        print(f"wrote dry-run proof -> {args.dry_run_out}")
        print(f"  wiring all-green: "
              f"{all(v for k, v in wiring.items() if isinstance(v, bool))}")
        print(f"  PASS-path verdict={proof['simulated_pass_path']['verdict']} "
              f"cases_run={proof['simulated_pass_path']['cases_run']}")
        print(f"  FAIL-path verdict={proof['simulated_fail_path']['verdict']} "
              f"stopped_at={proof['simulated_fail_path']['stopped_at']} "
              f"cases_run={proof['simulated_fail_path']['cases_run']}")
        return 0

    # Normal single run (DRY unless --execute).
    payload = run_fail_fast(force_fail_level=("L3" if args.simulate_case1_fail else None))
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote verdict -> {args.out}")
    print(f"  mode={payload['mode']} verdict={payload['verdict']} "
          f"stopped_at={payload['stopped_at']}")
    for line in payload["control_flow_timeline"]:
        print(f"   - {line}")
    return 0 if payload.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
