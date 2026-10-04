"""v0.1.0 REAL-WORLD 3 km (d02) VALIDATION — GPU forecast vs nightly CPU-WRF.

MISSION (principal, binding)
----------------------------
Prove the Coriolis-corrected GPU 3 km (d02) forecast gives near-identical,
no-blow-up results vs the nightly CPU-WRF, on 3 distinct real corpus days
(Canary / Tenerife focus). Headline paper/release evidence for v0.1.0.

This driver runs the GPU d02 forecast via the existing ``d02_replay`` path
(``daily_pipeline._build_real_case`` -> ``operational_mode`` segmented entry),
EXACTLY the wiring proofs/m19/verdict_3case.py uses, but extends scoring to:

  * T2 / U10 / V10 / surface-precip RMSE + bias, both FULL-DOMAIN and the
    TENERIFE-region subset (the island, where it matters).
  * all_finite + physically-plausible (blow-up detector) over the full run.
  * persistence-baseline skill: does the GPU beat a "hold t=0" forecast?

REUSE-ONLY: launches NO new WRF runs. CPU-WRF truth is the corpus wrfout_d02
hourly history. GPU init = each run's t=0 wrfout_d02 snapshot (the replay case).

The base model is the Coriolis-corrected dycore (manager HEAD 5319b8d).

USAGE
-----
  PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
    taskset -c 0-3 python proofs/v010_validation/v010_d02_validate.py \
      --execute --out proofs/v010_validation/v010_d02_result.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "v010_cases_manifest.json"

FIELDS = ("T2", "U10", "V10", "PRECIP")
# Tenerife bounding box (island + immediate surroundings) in the d02 grid.
# d02 spans lat [27.44, 29.22] lon [-18.57, -13.68]; Tenerife ~28.0-28.6N,
# -16.95 to -16.10W. We take a generous island box so the subset is non-trivial
# but firmly over/around Tenerife (NOT the open-ocean far corners).
TENERIFE_BOX = {"lat_min": 27.9, "lat_max": 28.7, "lon_min": -17.0, "lon_max": -16.0}

# ---------------------------------------------------------------------------
# PASS/FAIL THRESHOLDS (same spirit as verdict_3case; generous "near-identical /
# not-blown-up" ceilings, NOT tight TOST — that is M20). Anchored on the
# validated +1h/+3h GPU preview (T2 1.33/1.83 K, U10 2.22/1.90, V10 3.69/2.75).
# ---------------------------------------------------------------------------
RMSE_CEILING = {  # ceiling(lead_h) = base + slack*(lead_h/24)
    "T2": (4.0, 1.0),     # 5/6/7 K at 24/48/72h
    "U10": (5.0, 1.0),    # 6/7/8 m/s
    "V10": (6.0, 1.0),    # 7/8/9 m/s
    # Precip is an accumulator (grows with lead); judged by qualitative
    # closeness + finiteness, not a hard RMSE ceiling (no validated anchor yet).
}
SANITY = {
    "T2": {"abs_mean_max": 330.0, "abs_mean_min": 250.0, "std_max": 30.0},
    "U10": {"abs_mean_max": 40.0, "std_max": 40.0},
    "V10": {"abs_mean_max": 40.0, "std_max": 40.0},
    "PRECIP": {"abs_mean_max": 2000.0, "std_max": 2000.0},  # mm, very loose
}
DEFAULT_LEADS = (6, 12, 24, 48, 72)  # L3 caps at 24; L2 goes to 72.


def rmse_ceiling(field: str, lead_h: int) -> float | None:
    if field not in RMSE_CEILING:
        return None
    base, slack = RMSE_CEILING[field]
    return base + slack * (lead_h / 24.0)


# ---------------------------------------------------------------------------
# Scoring (pure numpy, float64; identical metric to verdict_3case.score_fields)
# ---------------------------------------------------------------------------
def _score_pair(g: np.ndarray, t: np.ndarray) -> dict[str, Any]:
    g = np.asarray(g, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    diff = g - t
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "bias": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
        "gpu_mean": float(np.mean(g)),
        "gpu_std": float(np.std(g)),
        "wrf_mean": float(np.mean(t)),
        "wrf_std": float(np.std(t)),
        "gpu_finite": bool(np.isfinite(g).all()),
        "n_points": int(g.size),
    }


def score_region(gpu: dict[str, np.ndarray], wrf: dict[str, np.ndarray],
                 mask: np.ndarray | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in FIELDS:
        g = np.asarray(gpu[f], dtype=np.float64)
        t = np.asarray(wrf[f], dtype=np.float64)
        if mask is not None:
            g = g[mask]
            t = t[mask]
        out[f] = _score_pair(g, t)
    return out


def evaluate_field(field: str, lead_h: int, s: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not s.get("gpu_finite", True):
        reasons.append(f"{field}@{lead_h}h: GPU field non-finite (blow-up)")
    ceil = rmse_ceiling(field, lead_h)
    if ceil is not None and s["rmse"] > ceil:
        reasons.append(f"{field}@{lead_h}h: RMSE {s['rmse']:.2f} > ceiling {ceil:.2f}")
    san = SANITY.get(field, {})
    if "abs_mean_max" in san and abs(s["gpu_mean"]) > san["abs_mean_max"]:
        reasons.append(f"{field}@{lead_h}h: |GPU mean| {abs(s['gpu_mean']):.1f} > {san['abs_mean_max']}")
    if "abs_mean_min" in san and s["gpu_mean"] < san["abs_mean_min"]:
        reasons.append(f"{field}@{lead_h}h: GPU mean {s['gpu_mean']:.1f} < {san['abs_mean_min']}")
    if "std_max" in san and s["gpu_std"] > san["std_max"]:
        reasons.append(f"{field}@{lead_h}h: GPU std {s['gpu_std']:.1f} > {san['std_max']} (blow-up)")
    return (len(reasons) == 0), reasons


def evaluate_level(level_scores: dict[int, dict]) -> dict:
    """level_scores: {lead_h: {'full': {field: s}, 'tenerife': {...}}}.
    PASS judged on FULL-DOMAIN T2/U10/V10 (precip is reported, not gated)."""
    gate_fields = ("T2", "U10", "V10")
    field_pass = {f: True for f in gate_fields}
    reasons: list[str] = []
    for lead_h in sorted(level_scores):
        full = level_scores[lead_h]["full"]
        for f in gate_fields:
            ok, why = evaluate_field(f, lead_h, full[f])
            if not ok:
                field_pass[f] = False
                reasons.extend(why)
    return {"passed": all(field_pass.values()), "field_pass": field_pass,
            "fail_reasons": reasons}


# ---------------------------------------------------------------------------
# Tenerife mask from the d02 XLAT/XLONG of the init file.
# ---------------------------------------------------------------------------
def tenerife_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat = np.asarray(lat); lon = np.asarray(lon)
    return ((lat >= TENERIFE_BOX["lat_min"]) & (lat <= TENERIFE_BOX["lat_max"]) &
            (lon >= TENERIFE_BOX["lon_min"]) & (lon <= TENERIFE_BOX["lon_max"]))


# ---------------------------------------------------------------------------
# WRF truth + persistence helpers (CPU/numpy only).
# ---------------------------------------------------------------------------
def read_wrf_surface(path: Path) -> dict[str, np.ndarray]:
    """Read T2/U10/V10/total-precip from a wrfout_d02. PRECIP = RAINNC (mm)
    accumulated. Returns numpy float64 arrays."""
    from gpuwrf.io.gen2_wrfout_loader import read_wrfout_file
    want = ("T2", "U10", "V10", "RAINNC")
    d = read_wrfout_file(path, fields=want)["fields"]
    return {
        "T2": np.asarray(d["T2"], dtype=np.float64),
        "U10": np.asarray(d["U10"], dtype=np.float64),
        "V10": np.asarray(d["V10"], dtype=np.float64),
        "PRECIP": np.asarray(d["RAINNC"], dtype=np.float64),
    }


def read_wrf_latlon(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from gpuwrf.io.gen2_wrfout_loader import read_wrfout_file
    d = read_wrfout_file(path, fields=("XLAT", "XLONG"))["fields"]
    return np.asarray(d["XLAT"]), np.asarray(d["XLONG"])


# ---------------------------------------------------------------------------
# REAL FORECAST CALL — the GPU forecast for one case/level at the given leads.
# ---------------------------------------------------------------------------
def run_and_score_real(level_spec: dict, leads: list[int], *, segment_steps: int,
                       dt_s: float, acoustic_substeps: int,
                       radiation_cadence_steps: int) -> dict:
    """Run ONE segmented GPU forecast to the max lead, snapshotting + scoring at
    each intermediate lead boundary.

    EFFICIENCY: a separate forecast per lead would re-integrate from t=0 every
    time (6+12+24+48+72 = 162 forecast-hours for an L2). Instead this advances a
    SINGLE carry across the contiguous segment loop (the SAME ``_advance_chunk``
    that ``run_forecast_operational_segmented`` uses, with identical global step
    indices + radiation cadence -> bit-identical to running each lead standalone)
    and computes the M9 diagnostics whenever the global step index reaches a lead
    boundary. ``block_until_ready`` between segments still frees each segment's
    scratch, so peak memory stays bounded to one segment regardless of lead.
    """
    import jax
    import jax.numpy as jnp
    from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
    from gpuwrf.runtime.operational_mode import (
        _advance_chunk,
        _enforce_operational_precision,
        compute_m9_diagnostics,
    )
    from gpuwrf.runtime.operational_state import initial_operational_carry

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
        case.namelist, run_physics=True, run_boundary=True, disable_guards=True,
        radiation_cadence_steps=radiation_cadence_steps, time_utc=time_utc,
    )

    max_lead = int(level_spec["max_lead_h"])
    leads = sorted(h for h in leads if h <= max_lead)

    # Tenerife mask from the init file lat/lon.
    init_path = Path(level_spec["gpu_init_source"])
    lat, lon = read_wrf_latlon(init_path)
    mask = tenerife_mask(lat, lon)

    # GPU init precip baseline (rain_acc at t=0 of the replay state) — should be 0.
    gpu_init_precip = float(np.asarray(jax.device_get(
        case.state.rain_acc + case.state.snow_acc
        + case.state.graupel_acc + case.state.ice_acc)).mean())
    wrf_init = read_wrf_surface(init_path)
    wrf_init_precip = wrf_init["PRECIP"]

    cadence = int(nl.radiation_cadence_steps)
    seg = int(segment_steps) if segment_steps else cadence
    dt = float(nl.dt_s)
    # global step index at the END of each lead (lead_h hours -> n steps).
    lead_steps = {h: int(round(h * 3600.0 / dt)) for h in leads}

    carry = initial_operational_carry(
        _enforce_operational_precision(case.state, force_fp64=bool(nl.force_fp64))
    )

    results: dict[int, dict] = {}
    timings: dict[int, float] = {}
    t_run = time.time()
    start = 1  # global step index of the next step to take (1-based, == _segmented)
    for lead_h in leads:
        target = lead_steps[lead_h]  # advance until global step index == target
        while start <= target:
            n = min(seg, target - start + 1)
            carry = _advance_chunk(
                carry, nl, jnp.asarray(start, dtype=jnp.int32),
                n_steps=int(n), cadence=cadence,
            )
            jax.block_until_ready(carry.state.theta)
            start += n
        # snapshot + score at this lead boundary.
        lead_seconds = float(lead_h) * 3600.0
        diags = compute_m9_diagnostics(carry.state, nl, lead_seconds)
        gpu_precip = np.asarray(jax.device_get(
            carry.state.rain_acc + carry.state.snow_acc
            + carry.state.graupel_acc + carry.state.ice_acc), dtype=np.float64)
        gpu = {
            "T2": np.asarray(jax.device_get(diags.t2), dtype=np.float64),
            "U10": np.asarray(jax.device_get(diags.u10), dtype=np.float64),
            "V10": np.asarray(jax.device_get(diags.v10), dtype=np.float64),
            "PRECIP": gpu_precip - gpu_init_precip,
        }
        valid = time_utc + timedelta(hours=lead_h)
        wrfout = run_dir / f"wrfout_{level_spec['domain']}_{valid:%Y-%m-%d_%H:%M:%S}"
        if not wrfout.is_file():
            raise FileNotFoundError(f"no CPU-WRF truth at {wrfout}")
        wrf = dict(read_wrf_surface(wrfout))
        wrf["PRECIP"] = wrf["PRECIP"] - wrf_init_precip
        results[lead_h] = {
            "full": score_region(gpu, wrf, None),
            "tenerife": score_region(gpu, wrf, mask),
        }
        timings[lead_h] = round(time.time() - t_run, 1)

    return {
        "scores": results,
        "timings_cumulative_s": timings,
        "tenerife_n_points": int(mask.sum()),
        "gpu_init_precip_mean": gpu_init_precip,
        "wrf_init_precip_mean": float(wrf_init_precip.mean()),
        "namelist": case.metadata["namelist"],
    }


# ---------------------------------------------------------------------------
# Persistence baseline (CPU/numpy; reuse same metric). Hold t=0 WRF constant.
# ---------------------------------------------------------------------------
def persistence_for_level(level_spec: dict, leads: list[int]) -> dict[int, dict]:
    init_path = Path(level_spec["gpu_init_source"])
    init = read_wrf_surface(init_path)
    lat, lon = read_wrf_latlon(init_path)
    mask = tenerife_mask(lat, lon)
    time_utc = datetime.strptime(
        init_path.name.split(f"wrfout_{level_spec['domain']}_", 1)[-1],
        "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc)
    max_lead = int(level_spec["max_lead_h"])
    out: dict[int, dict] = {}
    for lead_h in [h for h in leads if h <= max_lead]:
        valid = time_utc + timedelta(hours=lead_h)
        truth_path = init_path.parent / f"wrfout_{level_spec['domain']}_{valid:%Y-%m-%d_%H:%M:%S}"
        if not truth_path.is_file():
            continue
        truth = read_wrf_surface(truth_path)
        # persistence precip forecast = init accumulator held constant; both sides
        # accumulate from zero so persistence precip = 0 everywhere vs truth-init.
        pers = dict(init)
        pers["PRECIP"] = np.zeros_like(init["PRECIP"])
        truth_acc = dict(truth)
        truth_acc["PRECIP"] = truth["PRECIP"] - init["PRECIP"]
        out[lead_h] = {
            "full": score_region(pers, truth_acc, None),
            "tenerife": score_region(pers, truth_acc, mask),
        }
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_case(case: dict, leads: list[int], score_kwargs: dict, *,
             with_persistence: bool) -> dict:
    level_results: dict[str, dict] = {}
    for level_name, level_spec in case["levels"].items():
        rs = run_and_score_real(level_spec, leads, **score_kwargs)
        verdict = evaluate_level(rs["scores"])
        entry = {
            "scored_leads_h": sorted(rs["scores"]),
            "scores": rs["scores"],
            "verdict": verdict,
            "timings_cumulative_s": rs["timings_cumulative_s"],
            "tenerife_n_points": rs["tenerife_n_points"],
            "gpu_init_precip_mean": rs["gpu_init_precip_mean"],
            "wrf_init_precip_mean": rs["wrf_init_precip_mean"],
            "namelist": rs["namelist"],
        }
        if with_persistence:
            pers = persistence_for_level(level_spec, leads)
            entry["persistence"] = pers
            entry["skill_vs_persistence"] = _skill(rs["scores"], pers)
        level_results[level_name] = entry
    case_passed = all(lr["verdict"]["passed"] for lr in level_results.values())
    return {"case_id": case["case_id"], "init_utc": case["init_utc"],
            "passed": case_passed, "levels": level_results}


def _skill(gpu_scores: dict, pers_scores: dict) -> dict:
    """skill = 1 - GPU_RMSE/persistence_RMSE per field/region; aggregate W/T/L."""
    out: dict[str, Any] = {}
    for region in ("full", "tenerife"):
        agg = {f: {"wins": 0, "ties": 0, "losses": 0, "skills": []}
               for f in FIELDS}
        per: list[dict] = []
        for lead_h in sorted(gpu_scores):
            if lead_h not in pers_scores:
                continue
            for f in FIELDS:
                g = gpu_scores[lead_h][region][f]["rmse"]
                p = pers_scores[lead_h][region][f]["rmse"]
                if p <= 0:
                    continue
                sk = 1.0 - g / p
                if sk > 0.02:
                    agg[f]["wins"] += 1; outcome = "gpu_wins"
                elif sk < -0.02:
                    agg[f]["losses"] += 1; outcome = "gpu_loss"
                else:
                    agg[f]["ties"] += 1; outcome = "tie"
                agg[f]["skills"].append(sk)
                per.append({"lead_h": lead_h, "field": f, "gpu_rmse": g,
                            "pers_rmse": p, "skill": sk, "outcome": outcome})
        summ = {}
        for f in FIELDS:
            sk = agg[f]["skills"]
            summ[f] = {
                "n": len(sk),
                "mean_skill": (float(np.mean(sk)) if sk else None),
                "wins": agg[f]["wins"], "ties": agg[f]["ties"],
                "losses": agg[f]["losses"],
            }
        out[region] = {"per_entry": per, "by_field": summ}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="v0.1.0 d02 GPU-vs-nightly-WRF validation")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--leads", type=int, nargs="+", default=list(DEFAULT_LEADS))
    ap.add_argument("--cases", type=str, nargs="+", default=None,
                    help="subset of case ids to run (default all)")
    ap.add_argument("--execute", action="store_true", help="LAUNCH GPU forecasts")
    ap.add_argument("--no-persistence", action="store_true")
    ap.add_argument("--out", type=Path, default=HERE / "v010_d02_result.json")
    ap.add_argument("--segment-steps", type=int, default=180)
    ap.add_argument("--dt-s", type=float, default=10.0)
    ap.add_argument("--acoustic-substeps", type=int, default=10)
    ap.add_argument("--radiation-cadence-steps", type=int, default=180)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    cases = {c["case_id"]: c for c in manifest["cases"]}
    if args.cases:
        cases = {k: v for k, v in cases.items() if k in args.cases}
    score_kwargs = dict(segment_steps=args.segment_steps, dt_s=args.dt_s,
                        acoustic_substeps=args.acoustic_substeps,
                        radiation_cadence_steps=args.radiation_cadence_steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if not args.execute:
        print("DRY: not executing. Pass --execute to launch GPU forecasts.")
        wiring = {
            "n_cases": len(cases),
            "case_ids": list(cases),
            "all_init_exist": all(Path(lv["gpu_init_source"]).is_file()
                                  for c in cases.values()
                                  for lv in c["levels"].values()),
        }
        print(json.dumps(wiring, indent=2))
        return 0

    t0 = time.time()
    payload: dict[str, Any] = {
        "mode": "EXECUTE(real-GPU)",
        "base_model": "Coriolis-corrected dycore (HEAD 5319b8d)",
        "leads_h": args.leads,
        "tenerife_box": TENERIFE_BOX,
        "rmse_ceiling_formula": {f: f"{b} + {g}*(lead_h/24)"
                                 for f, (b, g) in RMSE_CEILING.items()},
        "sanity": SANITY,
        "results": {},
    }
    for cid, case in cases.items():
        print(f"=== running {cid} ({case['init_utc']}) ===", flush=True)
        payload["results"][cid] = run_case(
            case, args.leads, score_kwargs,
            with_persistence=not args.no_persistence)
        # incremental write so a late crash still leaves partial proof.
        payload["wall_s"] = round(time.time() - t0, 1)
        payload["generated_utc"] = datetime.now(timezone.utc).isoformat()
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"    {cid} passed={payload['results'][cid]['passed']}", flush=True)

    all_pass = all(payload["results"][cid]["passed"] for cid in payload["results"])
    no_blowup = all(
        s["full"][f]["gpu_finite"]
        for cid in payload["results"]
        for lv in payload["results"][cid]["levels"].values()
        for s in lv["scores"].values()
        for f in FIELDS
    )
    payload["verdict"] = "D02_VALIDATED" if (all_pass and no_blowup) else "D02_VALIDATION_ISSUE"
    payload["all_pass"] = all_pass
    payload["no_blowup"] = no_blowup
    payload["wall_s"] = round(time.time() - t0, 1)
    payload["generated_utc"] = datetime.now(timezone.utc).isoformat()
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}  verdict={payload['verdict']}  wall={payload['wall_s']}s")
    return 0 if all_pass and no_blowup else 1


if __name__ == "__main__":
    raise SystemExit(main())
