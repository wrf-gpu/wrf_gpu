"""v0.15 fp32-BouLac OPERATIONAL identity gate on the REAL Switzerland d01 case.

Runs N production 1h forecast calls (the exact operational path) twice -- once
with GPUWRF_MYNN_BOULAC_FP32=0 (fp64 reference) and once =1 -- and reports the
field deltas (max_abs / rmse / p99) on the frozen-tolerance hard fields
(T/U/V/W/QVAPOR + surface) so the principal can compare against the manifest
(proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json). Also records
the per-hour wall for the wall-clock win.

This is the cheap operational oracle (short forecast) before any 72h gate.

Run (GPU lock required), pick hours via --hours (default 2):
  scripts/with_gpu_lock.sh --label perf-fix -- \
    taskset -c 0-3 env PYTHONPATH=src JAX_ENABLE_X64=true OMP_NUM_THREADS=4 \
      MKL_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false TF_GPU_ALLOCATOR=cuda_malloc_async \
      python proofs/perf/v015/fp32_boulac_forecast_identity.py --hours 2
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import time
from pathlib import Path

import numpy as np

PROBE = Path("<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")
OUT = Path("proofs/perf/v015")

# frozen-tolerance hard fields -> State leaf names (prognostic subset measurable here)
HARD = {
    "T": "theta", "U": "u", "V": "v", "W": "w", "QVAPOR": "qv",
    "P": "p_total", "PH": "ph_total", "MU": "mu_total",
}


def _delta(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    d = np.abs(a - b)
    den = np.maximum(np.abs(a), 1e-30)
    return {
        "max_abs": float(d.max()),
        "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
        "p99_abs": float(np.percentile(d, 99)),
        "max_rel": float((d / den).max()),
    }


def _run(boulac_fp32: bool, hours: int):
    os.environ["GPUWRF_MYNN_BOULAC_FP32"] = "1" if boulac_fp32 else "0"
    # force a clean reimport of the physics module + the couplers that bound it
    import gpuwrf.physics.mynn_pbl as _m
    importlib.reload(_m)
    import gpuwrf.coupling.physics_couplers as _pc
    importlib.reload(_pc)
    import gpuwrf.integration.daily_pipeline as dp
    importlib.reload(dp)
    import jax

    cfg = dp.DailyPipelineConfig(run_id="run_h36", run_root=PROBE, domain="d01", hours=hours)
    case, _ = dp._build_real_case(cfg)
    nl = dataclasses.replace(case.namelist, force_fp64=True)
    state = case.state
    dt_s = float(nl.dt_s)
    steps_per_hour = int(round(3600.0 / dt_s))
    from gpuwrf.runtime.operational_mode import run_forecast_operational

    out_by_hour = {}
    walls = []
    # warm compile on hour 1, then time hour-by-hour (rewindow each hour like prod)
    cur = state
    for hour in range(1, hours + 1):
        lead0 = (hour - 1) * 3600.0
        nl_h = dataclasses.replace(nl, time_utc=case.run_start)
        t0 = time.perf_counter()
        cur = run_forecast_operational(cur, nl_h, 1.0)
        jax.block_until_ready(cur)
        walls.append(time.perf_counter() - t0)
        snap = {}
        for fld, leaf in HARD.items():
            v = getattr(cur, leaf, None)
            if v is not None:
                snap[fld] = np.asarray(v, np.float64)
        out_by_hour[hour] = snap
    return out_by_hour, walls, dt_s, steps_per_hour


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=2)
    args = ap.parse_args()

    ref, ref_walls, dt_s, sph = _run(False, args.hours)
    test, test_walls, _, _ = _run(True, args.hours)

    deltas = {}
    for hour in ref:
        deltas[f"hour{hour}"] = {
            fld: _delta(ref[hour][fld], test[hour][fld])
            for fld in ref[hour] if fld in test[hour]
        }

    out = {
        "scope": "v0.15 fp32-BouLac OPERATIONAL identity gate (real Switzerland d01)",
        "hours": args.hours, "dt_s": dt_s, "steps_per_hour": sph,
        "fp64_per_hour_wall_s": ref_walls,
        "fp32_per_hour_wall_s": test_walls,
        "steady_fp64_ms_per_step": (min(ref_walls[1:]) if len(ref_walls) > 1 else ref_walls[-1]) / sph * 1000.0,
        "steady_fp32_ms_per_step": (min(test_walls[1:]) if len(test_walls) > 1 else test_walls[-1]) / sph * 1000.0,
        "field_deltas_fp32_vs_fp64": deltas,
        "manifest": "proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json",
        "hard_fields": list(HARD),
    }
    sp64 = out["steady_fp64_ms_per_step"]; sp32 = out["steady_fp32_ms_per_step"]
    out["wall_speedup"] = sp64 / sp32 if sp32 else None
    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / "fp32_boulac_forecast_identity.json"
    fn.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "steady_fp64_ms": sp64, "steady_fp32_ms": sp32, "wall_speedup": out["wall_speedup"],
        "last_hour_deltas": deltas.get(f"hour{args.hours}", {}),
    }, indent=2), flush=True)
    print(f"\nwrote {fn}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
