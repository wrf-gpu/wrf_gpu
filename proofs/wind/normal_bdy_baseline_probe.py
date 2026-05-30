"""Short-lead BASELINE repro of the normal-momentum boundary spike (no patch).

Dumps the raw staggered u/v at the perpendicular boundaries after a short GPU
forecast so the spike (and later, its elimination) can be measured fast.

USAGE
  PYTHONPATH=src XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 OMP_NUM_THREADS=2 \
    taskset -c 0-3 python proofs/wind/normal_bdy_baseline_probe.py --lead-h 0.5
"""
from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead-h", type=float, default=0.5)
    ap.add_argument("--relax-strength", type=float, default=None,
                    help="override NORMAL_BDY_RELAX_STRENGTH for the sweep")
    args = ap.parse_args()

    import jax
    from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
    from gpuwrf.runtime.operational_mode import run_forecast_operational_segmented
    if args.relax_strength is None and os.environ.get("WIND_RELAX_STRENGTH"):
        args.relax_strength = float(os.environ["WIND_RELAX_STRENGTH"])
    if args.relax_strength is not None:
        import gpuwrf.coupling.boundary_apply as BA
        BA.NORMAL_BDY_RELAX_STRENGTH = float(args.relax_strength)
        print(f"[relax_strength override = {args.relax_strength}]", flush=True)

    cfg = DailyPipelineConfig(
        run_id="20260509_18z_l2_72h_20260511T190519Z",
        run_root=Path("/mnt/data/canairy_meteo/runs/wrf_l2"), domain="d02",
        dt_s=10.0, acoustic_substeps=10, radiation_cadence_steps=180)
    case, run_dir = _build_real_case(cfg)
    nl = dataclasses.replace(case.namelist, run_physics=True, run_boundary=True,
                             disable_guards=True, radiation_cadence_steps=180,
                             time_utc=case.run_start)
    fs = run_forecast_operational_segmented(case.state, nl, float(args.lead_h), segment_steps=180)
    jax.block_until_ready(fs.theta)
    v = np.array(fs.v); u = np.array(fs.u)
    print(f"[BASELINE lead={args.lead_h}h]", flush=True)
    print(" staggered v[z=0] S rows0-4:", np.round(v[0, :5, :].mean(axis=1), 2))
    print(" staggered v[z=0] N rows-5..-1:", np.round(v[0, -5:, :].mean(axis=1), 2))
    print(" staggered u[z=0] W cols0-4:", np.round(u[0, :, :5].mean(axis=0), 2))
    print(" staggered u[z=0] E cols-5..-1:", np.round(u[0, :, -5:].mean(axis=0), 2))
    print(" finite:", bool(np.isfinite(v).all() and np.isfinite(u).all()),
          " v|max|=%.2f u|max|=%.2f theta_finite=%s"
          % (np.abs(v).max(), np.abs(u).max(), bool(np.isfinite(np.array(fs.theta)).all())), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
