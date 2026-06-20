#!/usr/bin/env python
"""v0.15 kernel probe — short warmed steady segment for nsys/ncu kernel stats.

Builds the Switzerland h36 case, warms a 0.25h (50-step, non-radiation) program,
then runs it once more inside an NVTX range (STEADY50). Under nsys this yields
launch counts, CUDA API overhead, kernel time, and gap structure for exactly 50
production steps; under ncu it provides deterministic kernels to sample.
"""
from __future__ import annotations

import time
from pathlib import Path

import jax

from gpuwrf.integration import daily_pipeline as dp

PROBE = Path("<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")

config = dp.DailyPipelineConfig(
    run_id="run_h36",
    hours=1,
    output_dir=Path("/tmp/v015_perf/nsys_steady"),
    proof_dir=Path("/tmp/v015_perf/nsys_steady/proofs"),
    run_root=PROBE,
    domain="d01",
)

case, run_dir = dp._build_real_case(config)
state = case.state
boundary_leaves = dp._capture_boundary_leaves(state, case.namelist)
window_s = dp._boundary_window_cadence_s(case.namelist)
record_s = float((case.metadata.get("boundary") or {}).get("interval_seconds") or window_s)

HOURS = 50.0 * 18.0 / 3600.0  # 50 steps


def call(st, start_s):
    if boundary_leaves:
        st = dp._rewindow_boundary_leaves(
            st, boundary_leaves, segment_start_s=start_s, record_cadence_s=record_s, window_s=window_s
        )
    return dp._default_forecast_fn(st, case.namelist, HOURS)


t0 = time.perf_counter()
state = call(state, 0.0)  # compile fp32-input graph + run
print(f"warm1 {time.perf_counter()-t0:.1f}s", flush=True)
t0 = time.perf_counter()
state = call(state, 50 * 18.0)  # compile fp64 graph + run
print(f"warm2 {time.perf_counter()-t0:.1f}s", flush=True)

t0 = time.perf_counter()
with jax.profiler.TraceAnnotation("STEADY50"):
    state = call(state, 100 * 18.0)
wall = time.perf_counter() - t0
print(f"STEADY50: wall={wall:.3f}s per_step_ms={wall/50*1000:.2f}", flush=True)
