"""Confirm: with run_physics=True, does _advance_chunk recompile per start_step?

If start=4 / start=7 are cache hits like the physics-off probe, the per-segment
recompile is NOT from start_step.  If they are ~compile-cost, the physics path
(radiation cond / adapters / lead_seconds) re-specializes on start_step.
Uses tiny n_steps=3, cadence=3 so radiation fires inside the segment.
"""
from __future__ import annotations
import dataclasses, time
import jax, jax.numpy as jnp
from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import _advance_chunk, _enforce_operational_precision
from gpuwrf.runtime.operational_state import initial_operational_carry

cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
case, _ = _build_real_case(cfg)
nl = dataclasses.replace(case.namelist, run_physics=True, run_boundary=True,
                         disable_guards=True, radiation_cadence_steps=3,
                         time_utc=case.run_start)
carry = initial_operational_carry(_enforce_operational_precision(case.state, force_fp64=True))

def timed(start):
    ss = jnp.asarray(start, dtype=jnp.int32)
    t0 = time.perf_counter()
    out = _advance_chunk(carry, nl, ss, n_steps=3, cadence=3)
    jax.block_until_ready(out.state.theta)
    return time.perf_counter() - t0

print("=== run_physics=True, vary start_step (int32 array) ===", flush=True)
for s in (1, 4, 7, 4):
    dt = timed(s)
    print(f"  start={s}: {dt:.1f}s  ({'COMPILE' if dt > 5 else 'cache hit'})", flush=True)
print("VERDICT: if start=4/7 are cache hits, recompile is NOT from start_step; "
      "the full-walk recompile must come from a DIFFERENT changing arg.", flush=True)
print("DONE", flush=True)
