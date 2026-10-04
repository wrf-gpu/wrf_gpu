"""Fast micro-probe: does _advance_chunk recompile when start_step changes?

Uses a TINY n_steps and run_physics=False (no radiation) so each compile is cheap,
isolating ONLY the question of whether varying start_step forces a recompile.
Counts XLA compiles via jax's compile log hook.
"""
from __future__ import annotations
import dataclasses, time
import jax, jax.numpy as jnp
from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import _advance_chunk, _enforce_operational_precision
from gpuwrf.runtime.operational_state import initial_operational_carry

cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
case, _ = _build_real_case(cfg)
# run_physics=False -> no radiation branch -> CHEAP compile, isolates start_step.
nl = dataclasses.replace(case.namelist, run_physics=False, run_boundary=True,
                         disable_guards=True, radiation_cadence_steps=180,
                         time_utc=case.run_start)
carry = initial_operational_carry(_enforce_operational_precision(case.state, force_fp64=True))

def timed(start, weak):
    if weak:
        ss = start  # python int -> weak-typed when traced
    else:
        ss = jnp.asarray(start, dtype=jnp.int32)
    t0 = time.perf_counter()
    out = _advance_chunk(carry, nl, ss, n_steps=3, cadence=180)
    jax.block_until_ready(out.state.theta)
    return time.perf_counter() - t0

print("=== int32 device array start_step (as production/probe does) ===", flush=True)
for s in (1, 4, 7, 4):
    dt = timed(s, weak=False)
    print(f"  start={s}: {dt:.1f}s  ({'likely COMPILE' if dt > 5 else 'cache hit'})", flush=True)

print("=== python-int start_step (weak typed) ===", flush=True)
for s in (1, 4, 7, 4):
    dt = timed(s, weak=True)
    print(f"  start={s}: {dt:.1f}s  ({'likely COMPILE' if dt > 5 else 'cache hit'})", flush=True)
print("DONE", flush=True)
