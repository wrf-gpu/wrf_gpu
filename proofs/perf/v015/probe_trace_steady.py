#!/usr/bin/env python
"""v0.15 kernel probe — steady-state isolation + JAX profiler trace.

Protocol (mirrors the production per-hour path exactly, minus host overhead):
  hour1 call : compiles/loads the fp32-mixed-input graph (replay state)
  hour2 call : compiles/loads the all-fp64 graph (state fed back)
  hour3 call : TIMED — pure jit steady-state wall (200 steps, no host work)
  hour4 call : traced with jax.profiler (per-kernel attribution)
Also times the per-hour HOST overhead components once (finite_summary,
prepare_wrfout_payload, rewindow, land refresh) against the hour-3 state.

Artifacts: proofs/perf/v015/trace_steady.json + jax_trace/ (perfetto).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import jax

from gpuwrf.integration import daily_pipeline as dp

PROBE = Path("/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")
HERE = Path(__file__).resolve().parent
TRACE_DIR = HERE / "jax_trace"
PROOF = HERE / "trace_steady.json"

config = dp.DailyPipelineConfig(
    run_id="run_h36",
    hours=4,
    output_dir=Path("/tmp/v015_perf/trace_steady"),
    proof_dir=Path("/tmp/v015_perf/trace_steady/proofs"),
    run_root=PROBE,
    domain="d01",
)

case, run_dir = dp._build_real_case(config)
state = case.state
boundary_leaves = dp._capture_boundary_leaves(state, case.namelist)
window_s = dp._boundary_window_cadence_s(case.namelist)
record_s = float((case.metadata.get("boundary") or {}).get("interval_seconds") or window_s)


def hour_call(st, hour):
    if boundary_leaves:
        st = dp._rewindow_boundary_leaves(
            st,
            boundary_leaves,
            segment_start_s=(hour - 1) * 3600.0,
            record_cadence_s=record_s,
            window_s=window_s,
        )
    return dp._default_forecast_fn(st, case.namelist, 1.0)


walls = {}
for hour in (1, 2):
    t0 = time.perf_counter()
    state = hour_call(state, hour)
    walls[f"hour{hour}_warm_s"] = round(time.perf_counter() - t0, 3)
    print(f"hour{hour} (warm): {walls[f'hour{hour}_warm_s']}s", flush=True)

# hour 3: clean steady-state measurement (jit call only; rewindow timed apart)
t0 = time.perf_counter()
state3_in = (
    dp._rewindow_boundary_leaves(
        state, boundary_leaves, segment_start_s=2 * 3600.0, record_cadence_s=record_s, window_s=window_s
    )
    if boundary_leaves
    else state
)
rewindow_s = time.perf_counter() - t0
t0 = time.perf_counter()
state = dp._default_forecast_fn(state3_in, case.namelist, 1.0)
steady_s = time.perf_counter() - t0
walls["hour3_rewindow_s"] = round(rewindow_s, 4)
walls["hour3_steady_jit_s"] = round(steady_s, 3)
walls["steady_ms_per_step"] = round(steady_s / 200.0 * 1000.0, 2)
print(f"hour3 steady: {steady_s:.2f}s = {walls['steady_ms_per_step']} ms/step", flush=True)

# per-hour host overhead components measured once against the hour-3 state
t0 = time.perf_counter()
summary = dp.finite_summary(state)
walls["finite_summary_s"] = round(time.perf_counter() - t0, 3)
assert summary["all_finite"], "state went nonfinite"

from datetime import timedelta
from gpuwrf.io.wrfout_writer import prepare_wrfout_payload

valid_time = case.run_start + timedelta(hours=3)
t0 = time.perf_counter()
diag = dp._surface_diagnostics_for_output(state, case.namelist, case.run_start, lead_seconds=3 * 3600.0)
walls["surface_diagnostics_s"] = round(time.perf_counter() - t0, 3)
t0 = time.perf_counter()
payload = prepare_wrfout_payload(
    state, case.grid, case.namelist, Path("/tmp/v015_perf/wrfout_probe"),
    valid_time=valid_time, lead_hours=3.0, run_start=case.run_start,
    diagnostics=dp._merge_output_diagnostics(case.writer_diagnostics, diag),
)
walls["prepare_wrfout_payload_s"] = round(time.perf_counter() - t0, 3)
t0 = time.perf_counter()
state_land, land_rec = dp._refresh_hourly_land_state(state, run_dir, config.domain, 3, use_noahmp=False)
jax.block_until_ready(state_land.theta)
walls["land_refresh_s"] = round(time.perf_counter() - t0, 3)
state = state_land

# hour 4: profiler trace of one steady hour
state4_in = (
    dp._rewindow_boundary_leaves(
        state, boundary_leaves, segment_start_s=3 * 3600.0, record_cadence_s=record_s, window_s=window_s
    )
    if boundary_leaves
    else state
)
TRACE_DIR.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter()
with jax.profiler.trace(str(TRACE_DIR), create_perfetto_trace=True):
    out = dp._default_forecast_fn(state4_in, case.namelist, 1.0)
walls["hour4_traced_s"] = round(time.perf_counter() - t0, 3)
print(f"hour4 traced: {walls['hour4_traced_s']}s", flush=True)

payload_out = {
    "schema": "V015KernelProbeTraceSteady",
    "case": "Switzerland d01 reinit h36 replay, 128x128x44, dt=18s, force_fp64",
    "walls": walls,
    "trace_dir": str(TRACE_DIR),
}
PROOF.write_text(json.dumps(payload_out, indent=2) + "\n")
print(json.dumps(payload_out, indent=2), flush=True)
