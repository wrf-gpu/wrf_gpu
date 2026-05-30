"""Task 2 skill signal (M19 preview): gridded T2/U10/V10 RMSE/bias vs CPU WRF.

Runs the coupled GPU forecast (all physics, guards OFF, fp64) via the M9
DiagnosticsCarry to a verification time that exists in the corpus wrfout_d02,
extracts T2/U10/V10 at that time, and computes the FIRST gridded RMSE/bias vs the
corresponding CPU-WRF wrfout_d02 field.

This is a SINGLE-CASE GRIDDED preview of M19 -- NOT the final station-masked TOST
(that is a later step with B5's scorer). Honest scope: whole-grid statistics.

Run:
  PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \
    taskset -c 0-3 python proofs/coupled/task2_skill_signal.py --hours 1
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.io.gen2_wrfout_loader import read_wrfout_file
from gpuwrf.runtime.operational_mode import (
    compute_m9_diagnostics,
    run_forecast_operational,
)

PROOF = Path("proofs/coupled")


def _scores(jax_field: np.ndarray, wrf_field: np.ndarray) -> dict:
    jax_field = np.asarray(jax_field, dtype=np.float64)
    wrf_field = np.asarray(wrf_field, dtype=np.float64)
    diff = jax_field - wrf_field
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "bias": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
        "jax_mean": float(np.mean(jax_field)),
        "wrf_mean": float(np.mean(wrf_field)),
        "jax_min": float(np.min(jax_field)),
        "jax_max": float(np.max(jax_field)),
        "wrf_min": float(np.min(wrf_field)),
        "wrf_max": float(np.max(wrf_field)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=1.0,
                    help="verification lead in hours (must exist in wrfout_d02)")
    args = ap.parse_args()

    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, run_dir = _build_real_case(cfg)
    time_utc = case.run_start

    nl = dataclasses.replace(
        case.namelist,
        run_physics=True,
        run_boundary=True,
        disable_guards=True,
        radiation_cadence_steps=180,
        time_utc=time_utc,
    )

    hours = float(args.hours)
    steps = int(round(hours * 3600.0 / float(nl.dt_s)))

    print(f"=== Task 2 SKILL SIGNAL (+{hours}h = {steps} steps, guards OFF) ===",
          flush=True)
    print(f"  run_dir={run_dir}", flush=True)

    # Lean path: run the compiled operational scan (no per-step diagnostics
    # stacking -> small memory), then compute the M9 surface map ONCE on the final
    # state at the verification lead. (The full M9 DiagnosticsCarry stacks all 360
    # steps and needs >20GB; we only need the single verification slice, so the
    # M9 diagnostic kernels are applied once post-forecast on the final State.)
    final_state = run_forecast_operational(case.state, nl, hours)
    jax.block_until_ready(final_state.theta)
    lead_seconds = float(steps) * float(nl.dt_s)
    diags = compute_m9_diagnostics(final_state, nl, lead_seconds)
    t2_jax = np.asarray(jax.device_get(diags.t2))
    u10_jax = np.asarray(jax.device_get(diags.u10))
    v10_jax = np.asarray(jax.device_get(diags.v10))

    # CPU-WRF truth at the same valid time.
    from datetime import timedelta
    valid = time_utc + timedelta(hours=hours)
    wrfout = run_dir / f"wrfout_d02_{valid:%Y-%m-%d_%H:%M:%S}"
    if not wrfout.is_file():
        raise FileNotFoundError(f"no CPU-WRF truth at {wrfout}")
    truth = read_wrfout_file(wrfout, fields=("T2", "U10", "V10"))["fields"]
    t2_wrf = np.asarray(truth["T2"])
    u10_wrf = np.asarray(truth["U10"])
    v10_wrf = np.asarray(truth["V10"])

    assert t2_jax.shape == t2_wrf.shape, (t2_jax.shape, t2_wrf.shape)

    scores = {
        "T2": _scores(t2_jax, t2_wrf),
        "U10": _scores(u10_jax, u10_wrf),
        "V10": _scores(v10_jax, v10_wrf),
    }

    out = {
        "scope": "single-case GRIDDED preview of M19 (NOT station-masked TOST)",
        "run_dir": str(run_dir),
        "init_utc": str(time_utc),
        "valid_utc": str(valid),
        "lead_hours": hours,
        "steps": steps,
        "grid_shape": list(t2_jax.shape),
        "config": {
            "run_physics": True, "run_boundary": True, "disable_guards": True,
            "force_fp64": bool(nl.force_fp64), "top_lid": bool(nl.top_lid),
            "epssm": float(nl.epssm), "radiation_cadence_steps": 180,
        },
        "scores": scores,
        "jax_finite": {
            "t2": bool(np.isfinite(t2_jax).all()),
            "u10": bool(np.isfinite(u10_jax).all()),
            "v10": bool(np.isfinite(v10_jax).all()),
        },
    }
    fn = PROOF / f"task2_skill_signal_{int(round(hours))}h.json"
    fn.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {fn}", flush=True)
    for fld in ("T2", "U10", "V10"):
        s = scores[fld]
        print(f"  {fld:4s}: RMSE={s['rmse']:.3f} BIAS={s['bias']:+.3f} MAE={s['mae']:.3f} "
              f"| GPU mean={s['jax_mean']:.2f} WRF mean={s['wrf_mean']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
