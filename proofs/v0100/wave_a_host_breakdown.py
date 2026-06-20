"""Daily-wrapper HOST breakdown for one production-style forecast hour.

Uses the TRUSTED ``run_forecast_operational_segmented`` path (the same one the
segscan stability gate and the daily pipeline drive) to advance one forecast
hour, then times the host-side post-processing the daily pipeline performs each
hour:
  * forecast (GPU, 1h)
  * finite-summary full-State D2H (daily_pipeline.finite_summary -> np.asarray on
    every leaf)
  * M9 surface diagnostics (compute_m9_diagnostics via _surface_diagnostics_for_output)
  * output-pack D2H + NetCDF write (write_wrfout_netcdf)

This sizes the GPT#3 device-side-finite-summary lever (Phase 1) honestly.

Run:
  PYTHONPATH=src GPUWRF_CANAIRY_ROOT=<DATA_ROOT>/canairy_meteo OMP_NUM_THREADS=4 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_GPU_ALLOCATOR=cuda_malloc_async taskset -c 0-3 \
    python proofs/v0100/wave_a_host_breakdown.py --out wave_a_before_host.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import jax
import jax.numpy as jnp

from gpuwrf.integration.daily_pipeline import (
    _build_real_case,
    DailyPipelineConfig,
    finite_summary,
    _surface_diagnostics_for_output,
)
from gpuwrf.runtime.operational_mode import (
    _enforce_operational_precision,
    run_forecast_operational_segmented,
)
from gpuwrf.io.wrfout_writer import write_wrfout_netcdf

PROOF = Path("proofs/v0100")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="wave_a_before_host.json")
    ap.add_argument("--cadence", type=int, default=180)
    args = ap.parse_args()
    cadence = int(args.cadence)
    acoustic_unroll = int(os.environ.get("GPUWRF_ACOUSTIC_UNROLL", "1"))

    from gpuwrf.config import paths
    cfg = DailyPipelineConfig(
        hours=1, dt_s=10.0, acoustic_substeps=10,
        run_id="20260521_18z_l2_72h_20260522T133443Z",
        run_root=paths.wrf_l2_root(), domain="d02",
        radiation_cadence_steps=cadence,
    )
    case, run_dir = _build_real_case(cfg)
    nl = dataclasses.replace(
        case.namelist, run_physics=True, run_boundary=True, disable_guards=False,
        radiation_cadence_steps=cadence, time_utc=case.run_start,
    )
    state0 = _enforce_operational_precision(case.state, force_fp64=bool(nl.force_fp64))

    # Warm one segment first so the forecast clock is steady-state.
    _ = run_forecast_operational_segmented(state0, nl, 1.0, segment_steps=cadence)
    jax.block_until_ready(_.theta)

    # --- forecast 1h (GPU) ---
    t0 = time.perf_counter()
    state_out = run_forecast_operational_segmented(state0, nl, 1.0, segment_steps=cadence)
    jax.block_until_ready(state_out.theta)
    forecast_1h_s = time.perf_counter() - t0

    # --- finite summary (full-State D2H) ---
    t0 = time.perf_counter()
    fs = finite_summary(state_out)
    finite_summary_s = time.perf_counter() - t0

    # --- M9 surface diagnostics ---
    t0 = time.perf_counter()
    diags = _surface_diagnostics_for_output(
        state_out, case.namelist, case.run_start, lead_seconds=3600.0)
    leaf = None
    for attr in ("t2", "T2", "psfc", "PSFC"):
        if hasattr(diags, attr):
            leaf = getattr(diags, attr)
            break
    if leaf is not None:
        jax.block_until_ready(leaf)
    m9_s = time.perf_counter() - t0

    # --- output pack D2H + netcdf write ---
    valid_time = case.run_start + timedelta(hours=1)
    tmp_out = PROOF / "_tmp_wrfout_hostbreak"
    t0 = time.perf_counter()
    write_wrfout_netcdf(
        state_out, case.grid, case.namelist, tmp_out,
        valid_time=valid_time, lead_hours=1.0, run_start=case.run_start,
        diagnostics=diags,
    )
    output_s = time.perf_counter() - t0
    try:
        tmp_out.unlink()
    except Exception:
        pass

    total = forecast_1h_s + finite_summary_s + m9_s + output_s
    non_forecast = finite_summary_s + m9_s + output_s
    out = {
        "scope": "daily-wrapper host breakdown for one production forecast hour",
        "run_dir": str(run_dir),
        "device": str(jax.devices()[0]),
        "config": {"force_fp64": bool(nl.force_fp64), "acoustic_unroll": acoustic_unroll,
                   "radiation_cadence_steps": cadence},
        "forecast_1h_s": forecast_1h_s,
        "finite_summary_full_state_d2h_s": finite_summary_s,
        "finite_summary_all_finite": bool(fs["all_finite"]),
        "finite_summary_field_count": int(fs["field_count"]),
        "m9_diagnostics_s": m9_s,
        "output_pack_and_netcdf_s": output_s,
        "total_hour_wall_s": total,
        "non_forecast_host_s": non_forecast,
        "non_forecast_host_pct_of_hour": 100.0 * non_forecast / total if total else None,
        "finite_summary_pct_of_hour": 100.0 * finite_summary_s / total if total else None,
    }
    PROOF.mkdir(parents=True, exist_ok=True)
    fn = PROOF / args.out
    fn.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)
    print(f"wrote {fn}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
