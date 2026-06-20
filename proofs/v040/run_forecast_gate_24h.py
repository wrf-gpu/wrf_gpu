#!/usr/bin/env python
"""v0.4.0 — FULL 24h native-init FORECAST GATE (the milestone-closing proof).

Runs the native real-init -> 24h GPU forecast (native wrfbdy is the ONLY LBC; NO
CPU-WRF replay) for every usable case and scores each per-lead vs the CPU-WRF
wrfout, under the principal's **h2+ core-skill policy** (council-proven; binding
2026-06-03):

  - The native-init first forecast hour carries an INHERENT cold-start spin-up
    transient. CPU-WRF spins up the SAME (its t0 wrfout QKE/TKE_PBL/PBLH are zero,
    UST~1e-4 -> both codes equilibrate turbulence/surface stress over h1).
  - Core-field PASS (T2/U10/V10 blocking; PSFC/PBLH reported alongside) is judged
    over leads >= h2. The h1 lead is REPORTED as a non-blocking spin-up diagnostic.
  - Margins are the SAME frozen continuous_gate REGRESSION_MARGINS — only the
    lead-inclusion of the worst-over-leads envelope changes (NOT a tol relaxation).

This runner drives cases ONE AT A TIME (single GPU) and COMMITS the deliverable
after EACH case lands (hibernation-safe). The first case pays the ~19-min one-time
XLA cold compile; subsequent forecast hours run at ~2 s each.

  taskset -c 0-3 env PYTHONPATH=src:proofs/v040:. python \
      proofs/v040/run_forecast_gate_24h.py \
      --hours 24 --out proofs/v040/forecast_gate_24h_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT / "proofs" / "v040"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from gpuwrf.init.real_init import comparator as C  # noqa: E402
from gpuwrf.init.real_init.comparator import ForecastGateSpec  # noqa: E402
import s5_native_init_parity as s5  # noqa: E402

BACKFILL_ROOT = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output")


def _run_start(case_id: str) -> datetime:
    ymd = case_id[:8]
    return datetime(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8]), 18, 0, 0,
                    tzinfo=timezone.utc)


def _ref_dir_for(oc) -> Path | None:
    """CPU reference dir: oracle run_dir if it retains t0, else the backfill dir."""
    init_vt = s5._init_valid_time(oc.case_id)
    if (oc.run_dir / f"wrfout_d01_{init_vt}").is_file():
        return oc.run_dir
    cand = BACKFILL_ROOT / oc.case_id
    if (cand / f"wrfout_d01_{init_vt}").is_file():
        return cand
    return None


def _ref_source_for(oc, ref_dir: Path | None) -> str | None:
    if ref_dir is None:
        return None
    if Path(ref_dir).resolve() == Path(oc.run_dir).resolve():
        return "oracle_run_dir"
    return "backfill_or_override_output_dir"


def _scoreable_leads(ref_dir: Path, case_id: str, hours: int) -> list[int]:
    t0 = _run_start(case_id)
    return [h for h in range(1, hours + 1)
            if (ref_dir / f"wrfout_d01_{(t0 + timedelta(hours=h)):%Y-%m-%d_%H:%M:%S}").is_file()]


def _usable_cases():
    """All native-input (oracle) d01 cases with >=2 wps met_em frames AND a
    retained CPU-WRF t0 wrfout (oracle run_dir OR backfill output)."""
    all_cases = C.discover_oracle_cases(require_domains=("d01",), require_wrfbdy=True)
    usable = []
    for oc in all_cases:
        wps = s5._wps_dir_for(oc.case_id)
        if wps is None or len(s5._ordered_metem_paths(wps, "d01")) < 2:
            continue
        if _ref_dir_for(oc) is None:
            continue
        usable.append(oc)
    return usable


def _git_commit(report_path: Path, msg: str) -> None:
    try:
        subprocess.run(["git", "-C", str(ROOT), "add", str(report_path)],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", str(ROOT), "commit", "-m", msg],
                       check=True, capture_output=True)
        print(f"  [git] committed {report_path.name}: {msg}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"  [git] commit skipped/failed: {e.stderr.decode(errors='replace')[:200]}",
              flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--dt-s", type=float, default=60.0)
    ap.add_argument("--acoustic-substeps", type=int, default=4)
    ap.add_argument("--radiation-cadence-steps", type=int, default=30)
    ap.add_argument("--core-skill-first-lead", type=int, default=2)
    ap.add_argument("--out", default="proofs/v040/forecast_gate_24h_report.json")
    ap.add_argument("--output-root", default="/tmp/v040_forecast_gate_24h")
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--case-id", action="append", default=None,
                    help="Run only the requested case id(s); may be repeated.")
    ap.add_argument("--commit-each", action="store_true", default=False)
    args = ap.parse_args()

    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {0, 1, 2, 3})
        except OSError:
            pass

    from s5_forecast_gate_exec import (  # GPU body + report classification
        run_one_case_forecast_gate,
        summarize_case_classes,
    )

    spec = ForecastGateSpec()
    cases = _usable_cases()
    if args.case_id:
        requested = set(args.case_id)
        cases = [oc for oc in cases if oc.case_id in requested]
        missing = sorted(requested - {oc.case_id for oc in cases})
        if missing:
            raise SystemExit(f"requested case-id(s) not usable: {', '.join(missing)}")
    if args.max_cases is not None:
        cases = cases[: int(args.max_cases)]

    factory = s5.make_factory(lbc_intervals=1)
    out_path = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_root = Path(args.output_root)
    first = int(args.core_skill_first_lead)

    # case metadata (lead coverage) up front for the report header
    case_meta = []
    for oc in cases:
        rd = _ref_dir_for(oc)
        leads = _scoreable_leads(rd, oc.case_id, int(args.hours))
        case_meta.append({"case_id": oc.case_id,
                          "oracle_run_dir": str(oc.run_dir),
                          "metadata_run_dir": str(oc.run_dir),
                          "reference_dir": str(rd),
                          "reference_source": _ref_source_for(oc, rd),
                          "scoreable_cpu_leads": leads,
                          "h2plus_cpu_ref_leads": [l for l in leads if l >= first],
                          "has_h2plus_cpu_ref": any(l >= first for l in leads),
                          "has_min_h2plus_cpu_ref": len([l for l in leads if l >= first]) >= 2,
                          "namelist_source": str(oc.run_dir / "namelist.input"),
                          "wrfinput_source": str(oc.run_dir / "wrfinput_d01")})

    print(f"FULL 24h GATE: {len(cases)} cases hours={args.hours} dt={args.dt_s} "
          f"core_skill_first_lead=h{first}", flush=True)
    for m in case_meta:
        print(f"  {m['case_id']}: cpu_leads={m['scoreable_cpu_leads']} "
              f"h2+ref={m['has_h2plus_cpu_ref']}", flush=True)

    base_header = {
        "schema": "v0.4.0-forecast-gate-24h-2026-06-03",
        "executed": True,
        "gpu_bound": True,
        "owner": "S5/manager (single-GPU serialization point)",
        "no_replay": spec.no_replay,
        "metric": spec.metric,
        "core_fields_blocking": list(spec.core_fields),
        "diag_fields_descriptive": list(spec.diag_fields),
        "forecast_hours": int(args.hours),
        "forecast_dt_s": float(args.dt_s),
        "forecast_acoustic_substeps": int(args.acoustic_substeps),
        "continuous_gate_module": "proofs/m20/continuous_gate.py",
        "scoring_policy": {
            "core_skill_first_lead": first,
            "h1_non_blocking_spinup": True,
            "margins_unchanged": True,
            "basis": (
                "principal decision 2026-06-03 (council-proven): native-init h1 "
                "cold-start spin-up is inherent — CPU-WRF t0 wrfout QKE/TKE_PBL/"
                "PBLH are zero and UST~1e-4, so BOTH codes equilibrate over h1. "
                "Core PASS judged from h2+; h1 reported non-blocking. Frozen "
                "continuous_gate margins UNCHANGED (lead-inclusion only)."
            ),
            "honest_claim": (
                "native-init forecast matches CPU-WRF from h2+ after a documented "
                "first-hour cold-start transient; h1 not claimed as nowcast parity"
            ),
        },
        "case_lead_coverage": case_meta,
    }

    def _write_and_commit(records, final=False):
        class_summary = summarize_case_classes(
            records,
            core_fields=spec.core_fields,
            core_skill_first_lead=first,
            min_h2plus_core_leads=2,
        )
        verdict = class_summary["final_verdict_from_scored_cases_only"]
        scored_table = [
            row for row in class_summary["case_class_table"]
            if row["case_class"] in {"SCORED_PASS", "SCORED_FAIL"}
        ]
        all_stable = bool(scored_table) and all(row["stable_finite"] for row in scored_table)
        all_physical = bool(scored_table) and all(row["physical_range_ok"] for row in scored_table)
        all_core_pass = bool(scored_table) and all(
            row["case_class"] == "SCORED_PASS" for row in scored_table)
        result = dict(base_header)
        result.update({
            "verdict": verdict,
            "foundation_confirmed": verdict == "FOUNDATION_CONFIRMED",
            "complete": bool(final),
            "n_cases_attempted": len(records),
            "n_cases_scored": class_summary["n_scored_cases"],
            "n_cases_with_h2plus_core_skill": class_summary["n_scored_cases"],
            "all_scored_cases_stable_finite": all_stable,
            "all_scored_cases_physical": all_physical,
            "all_scored_cases_core_within_margin": all_core_pass,
            "case_class_summary": class_summary,
            "case_class_counts": class_summary["case_class_counts"],
            "case_class_table": class_summary["case_class_table"],
            "cases": records,
        })
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str) + "\n",
                            encoding="utf-8")
        tag = "FINAL" if final else f"after {len(records)} case(s)"
        msg = (f"[v040] 24h native-init forecast gate (h2+ policy) — "
               f"v0.4.0 standalone-forecast proof [{tag}: {verdict}]")
        if args.commit_each:
            _git_commit(out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path, msg)
        return verdict

    records = []
    for i, oc in enumerate(cases):
        ref_dir = _ref_dir_for(oc)
        init_label = s5._init_valid_time(oc.case_id)
        print(f"\n=== CASE {i+1}/{len(cases)}: {oc.case_id} (ref={ref_dir}) ===",
              flush=True)
        if ref_dir is None or not (ref_dir / f"wrfout_d01_{init_label}").is_file():
            records.append({"case_id": oc.case_id, "status": "NO_REFERENCE_T0_WRFOUT"})
            _write_and_commit(records)
            continue
        try:
            product = factory(oc, "d01")
            rec = run_one_case_forecast_gate(
                product,
                case_id=oc.case_id,
                reference_run_dir=ref_dir,
                run_start=_run_start(oc.case_id),
                init_vt_label=init_label,
                forecast_hours=int(args.hours),
                dt_s=float(args.dt_s),
                acoustic_substeps=int(args.acoustic_substeps),
                radiation_cadence_steps=int(args.radiation_cadence_steps),
                output_dir=out_root / f"{oc.case_id}_d01",
                core_fields=spec.core_fields,
                diag_fields=spec.diag_fields,
                metadata_run_dir=oc.run_dir,
                core_skill_first_lead=first,
            )
            records.append(rec)
            st = rec["stability"]
            print(f"  -> leads={rec['n_leads_run']}/{rec['forecast_hours_attempted']} "
                  f"stable={st['stable_finite']} physical={st['physical_range_ok']} "
                  f"blew_up_at={st['blew_up_at_hour']} "
                  f"core_within(h2+)={rec['core_within_margin']} "
                  f"h2+evidence={rec['core_has_h2plus_evidence']}", flush=True)
            for f, fs in rec["per_field_summary"].items():
                if fs.get("status") == "scored":
                    h1 = fs.get("h1_spinup") or {}
                    print(f"      {f:6s} h2+worst|bias|={fs['worst_abs_bias']:.4f} "
                          f"margin={fs['regression_margin']} within(h2+)={fs['within_margin']} "
                          f"blocking={fs['blocking']} | h1|bias|={h1.get('abs_bias')}",
                          flush=True)
        except Exception as e:  # honest: record the failure, keep going / stop
            import traceback
            tb = traceback.format_exc()
            print(f"  !! CASE FAILED: {e}\n{tb}", flush=True)
            records.append({"case_id": oc.case_id, "status": "ERROR",
                           "reference_run_dir": str(ref_dir),
                           "reference_output_dir": str(ref_dir),
                           "metadata_run_dir": str(oc.run_dir),
                           "error": str(e), "traceback": tb})
        _write_and_commit(records)

    verdict = _write_and_commit(records, final=True)
    print(f"\nFINAL VERDICT: {verdict}", flush=True)
    print(f"  wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
