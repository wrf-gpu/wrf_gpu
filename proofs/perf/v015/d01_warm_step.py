"""v0.15 warmed per-step WALL on the REAL Switzerland d01 128x128x44 case.

Measures the warmed per-step wall (marginal of two horizons to remove fixed
per-call setup) for:
  * coupled  (run_physics + run_boundary, force_fp64) -- the production step
  * dycore-only (run_physics=False, run_boundary=False, force_fp64) -- isolates dynamics
  * physics delta = coupled - dycore

This is the d01 analogue of roofline_costanalysis.py (which used the smaller d02
synthetic case). Settles where the 173 ms/step lives: dynamics vs physics.

Run (GPU lock required):
  scripts/with_gpu_lock.sh --label perf-fix -- \
    taskset -c 0-3 env PYTHONPATH=src JAX_ENABLE_X64=true OMP_NUM_THREADS=4 \
      MKL_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async \
      python proofs/perf/v015/d01_warm_step.py
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import jax

import gpuwrf.contracts.state as _stmod
from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import run_forecast_operational

PROBE = Path("<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")
OUT = Path("proofs/perf/v015")

_orig_asarray = _stmod.jnp.asarray


def _safe_asarray(x, dtype=None, **kw):
    try:
        return _orig_asarray(x, dtype=dtype, **kw) if dtype is not None else _orig_asarray(x, **kw)
    except (TypeError, ValueError):
        return x


_stmod.jnp.asarray = _safe_asarray


def _block(x):
    jax.block_until_ready(x)
    return x


def _time_warm(builder, nl, hours, reps=4):
    """Warmed wall per call (compile+1 warm, then min of reps)."""
    st = builder()
    _block(run_forecast_operational(st, nl, float(hours)))   # compile
    _block(run_forecast_operational(builder(), nl, float(hours)))  # warm
    samples = []
    for _ in range(reps):
        st = builder()
        _block(st)
        t0 = time.perf_counter()
        _block(run_forecast_operational(st, nl, float(hours)))
        samples.append(time.perf_counter() - t0)
    return samples


def main() -> int:
    cfg = DailyPipelineConfig(run_id="run_h36", run_root=PROBE, domain="d01", hours=1)
    case, run_dir = _build_real_case(cfg)
    nl = case.namelist
    ny, nx, nz = int(case.grid.ny), int(case.grid.nx), int(case.grid.nz)
    dt_s = float(nl.dt_s)

    base_nl = dataclasses.replace(nl, force_fp64=True, radiation_cadence_steps=100000)  # no radiation in window
    dyn_nl = dataclasses.replace(base_nl, run_physics=False, run_boundary=False)

    def builder():
        return _build_real_case(cfg)[0].state

    # Two horizons under the radiation cadence; marginal isolates 1 steady step.
    nA, nB = 10, 40
    hA, hB = nA * dt_s / 3600.0, nB * dt_s / 3600.0

    def per_step(nl, label):
        sA = _time_warm(builder, nl, hA)
        sB = _time_warm(builder, nl, hB)
        mA, mB = min(sA), min(sB)
        ms = (mB - mA) / (nB - nA) * 1000.0
        return {
            "label": label,
            "samples_A_s": sA, "samples_B_s": sB,
            "min_A_s": mA, "min_B_s": mB,
            "marginal_ms_per_step": ms,
            "fc_hour_s_at_dt": ms / 1000.0 * (3600.0 / dt_s),
        }

    coupled = per_step(base_nl, "coupled (phys+bdy, force_fp64)")
    dycore = per_step(dyn_nl, "dycore-only (force_fp64)")

    out = {
        "scope": "v0.15 warmed per-step WALL on REAL Switzerland d01 (173 ms/step grid)",
        "run_dir": str(run_dir),
        "device": str(jax.devices()[0]),
        "grid": {"ny": ny, "nx": nx, "nz": nz, "mass_cells": ny * nx * nz},
        "dt_s": dt_s,
        "horizons_steps": [nA, nB],
        "coupled_step": coupled,
        "dycore_only_step": dycore,
        "physics_delta_ms_per_step": coupled["marginal_ms_per_step"] - dycore["marginal_ms_per_step"],
        "dycore_share": dycore["marginal_ms_per_step"] / coupled["marginal_ms_per_step"]
        if coupled["marginal_ms_per_step"] else float("nan"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / "d01_warm_step.json"
    fn.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "grid": out["grid"], "dt_s": dt_s,
        "coupled_ms_per_step": coupled["marginal_ms_per_step"],
        "dycore_ms_per_step": dycore["marginal_ms_per_step"],
        "physics_delta_ms": out["physics_delta_ms_per_step"],
        "dycore_share": out["dycore_share"],
        "coupled_fc_hour_s": coupled["fc_hour_s_at_dt"],
    }, indent=2), flush=True)
    print(f"\nwrote {fn}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
