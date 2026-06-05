"""Minimal warmed-per-step probe: ONE compiled non-radiation segment, timed warm.

Compiles a single ``_advance_chunk`` of N non-radiation steps on the L2 d02 init
(guards ON, fp64), warms it, then times warm cache-hit calls (min of repeats).
No segmented host loop, no second compile -- isolates the per-step launch cost so
the carry-split / unroll effect is measured cleanly and fast.

Run:
  PYTHONPATH=src GPUWRF_CANAIRY_ROOT=/mnt/data/canairy_meteo OMP_NUM_THREADS=4 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_GPU_ALLOCATOR=cuda_malloc_async GPUWRF_ACOUSTIC_UNROLL=2 taskset -c 0-3 \
    python proofs/v0100/wave_a_micro_timing.py --steps 90 --out micro_u2.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.config import paths
from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import _advance_chunk, _enforce_operational_precision
from gpuwrf.runtime.operational_state import initial_operational_carry

PROOF = Path("proofs/v0100")
L2_RUN_ID = "20260521_18z_l2_72h_20260522T133443Z"


def _peak_mb() -> float:
    try:
        return float(jax.devices()[0].memory_stats()["peak_bytes_in_use"]) / (1024.0**2)
    except Exception:
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=90)
    ap.add_argument("--cadence", type=int, default=180)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", type=str, default="wave_a_micro.json")
    args = ap.parse_args()
    steps = int(args.steps)
    cadence = int(args.cadence)
    unroll = int(os.environ.get("GPUWRF_ACOUSTIC_UNROLL", "1"))

    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10,
                              run_id=L2_RUN_ID, run_root=paths.wrf_l2_root(),
                              domain="d02", radiation_cadence_steps=cadence)
    case, _ = _build_real_case(cfg)
    nl = dataclasses.replace(case.namelist, run_physics=True, run_boundary=True,
                             disable_guards=False, radiation_cadence_steps=cadence,
                             time_utc=case.run_start)
    carry0 = initial_operational_carry(
        _enforce_operational_precision(case.state, force_fp64=bool(nl.force_fp64)))

    print(f"unroll={unroll} steps={steps} -- compiling", flush=True)
    t0 = time.perf_counter()
    c = _advance_chunk(carry0, nl, jnp.asarray(1, dtype=jnp.int32),
                       n_steps=steps, cadence=cadence)
    jax.block_until_ready(c.state.theta)
    compile_s = time.perf_counter() - t0
    peak = _peak_mb()
    print(f"compiled in {compile_s:.1f}s, warming/timing", flush=True)

    ms = []
    for r in range(int(args.repeats)):
        start = jnp.asarray(1 + (r + 1) * steps, dtype=jnp.int32)
        t0 = time.perf_counter()
        c2 = _advance_chunk(c, nl, start, n_steps=steps, cadence=cadence)
        jax.block_until_ready(c2.state.theta)
        ms.append((time.perf_counter() - t0) / steps * 1000.0)
        c = c2
    per_step_ms = float(min(ms))
    th = np.asarray(jax.device_get(c.state.theta))
    finite = bool(np.isfinite(th).all())

    out = {"unroll": unroll, "steps": steps, "compile_s": compile_s,
           "warmed_per_step_ms": per_step_ms, "samples_ms": ms,
           "peak_mb": peak, "theta_finite": finite}
    PROOF.mkdir(parents=True, exist_ok=True)
    (PROOF / args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
