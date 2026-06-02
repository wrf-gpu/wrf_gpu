#!/usr/bin/env python
"""v0.4.0 S6 — GPU foundation smoke for the native-init -> forecast gate.

Runs ONE case for a SHORT lead through the wired
``comparator.run_forecast_gate(execute=True)`` body (native IC + native wrfbdy ->
the validated operational forecast entry; NO CPU-WRF replay) and writes the smoke
verdict. ONE GPU job at a time.

  taskset -c 0-3 env PYTHONPATH=src:proofs/v040:. python \
      proofs/v040/run_forecast_gate_smoke.py --hours 6 --max-cases 1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT / "proofs" / "v040"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from gpuwrf.init.real_init import comparator as C  # noqa: E402
import s5_native_init_parity as s5  # noqa: E402


def _usable_cases():
    all_cases = C.discover_oracle_cases(require_domains=("d01",), require_wrfbdy=True)
    usable = []
    for oc in all_cases:
        wps = s5._wps_dir_for(oc.case_id)
        if wps is None or len(s5._ordered_metem_paths(wps, "d01")) < 2:
            continue
        init_vt = s5._init_valid_time(oc.case_id)
        if not (oc.run_dir / f"wrfout_d01_{init_vt}").is_file():
            continue
        usable.append(oc)
    return usable


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=6)
    ap.add_argument("--max-cases", type=int, default=1)
    ap.add_argument("--dt-s", type=float, default=60.0)
    ap.add_argument("--acoustic-substeps", type=int, default=4)
    ap.add_argument("--radiation-cadence-steps", type=int, default=30)
    ap.add_argument("--out", default="proofs/v040/s5_forecast_gate_report.json")
    ap.add_argument("--case-id", default=None)
    args = ap.parse_args()

    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {0, 1, 2, 3})
        except OSError:
            pass

    cases = _usable_cases()
    if args.case_id:
        cases = [c for c in cases if c.case_id == args.case_id] or cases
    cases = cases[: int(args.max_cases)]
    print(f"SMOKE cases: {[c.case_id for c in cases]} hours={args.hours} dt={args.dt_s}", flush=True)

    result = C.run_forecast_gate(
        s5.make_factory(lbc_intervals=1),
        cases=cases,
        execute=True,
        forecast_hours=int(args.hours),
        max_cases=int(args.max_cases),
        dt_s=float(args.dt_s),
        acoustic_substeps=int(args.acoustic_substeps),
        radiation_cadence_steps=int(args.radiation_cadence_steps),
        out_path=args.out,
    )
    print(f"VERDICT {result['verdict']} foundation_confirmed={result['foundation_confirmed']}", flush=True)
    print(f"  stable_finite={result['all_cases_stable_finite']} "
          f"physical={result['all_cases_physical']} "
          f"core_within_margin={result['all_cases_core_within_margin']}", flush=True)
    for rec in result["cases"]:
        if rec.get("status"):
            print(f"  {rec['case_id']}: {rec['status']}", flush=True)
            continue
        st = rec["stability"]
        print(f"  {rec['case_id']}: leads={rec['n_leads_run']}/{rec['forecast_hours_attempted']} "
              f"stable={st['stable_finite']} physical={st['physical_range_ok']} "
              f"blew_up_at_hour={st['blew_up_at_hour']} core_pass={rec['core_within_margin']}", flush=True)
        for f, fs in rec["per_field_summary"].items():
            if fs.get("status") == "scored":
                print(f"      {f:6s} worst|bias|={fs['worst_abs_bias']:.4f} "
                      f"worst_rmse={fs['worst_rmse']:.4f} margin={fs['regression_margin']} "
                      f"within={fs['within_margin']} blocking={fs['blocking']}", flush=True)
    print(f"  wrote {args.out}", flush=True)
    return 0 if result["foundation_confirmed"] else (2 if result["all_cases_stable_finite"] else 3)


if __name__ == "__main__":
    raise SystemExit(main())
