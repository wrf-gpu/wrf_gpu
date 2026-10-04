"""Diagnose whether _advance_chunk recompiles per segment (seg=90 vs 180).

Times the cold (compile+exec) and warm (exec only) segments, and a THIRD segment
with a different traced start_step, to confirm the production claim that equal-length
segments reuse ONE compiled executable.
"""
from __future__ import annotations
import argparse, dataclasses, time
import jax, jax.numpy as jnp
from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import _advance_chunk, _enforce_operational_precision
from gpuwrf.runtime.operational_state import initial_operational_carry

ap = argparse.ArgumentParser(); ap.add_argument("--seg", type=int, default=90)
args = ap.parse_args(); seg = args.seg
cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
case, _ = _build_real_case(cfg)
cad = 180
nl = dataclasses.replace(case.namelist, run_physics=True, run_boundary=True,
                         disable_guards=True, radiation_cadence_steps=cad, time_utc=case.run_start)
carry = initial_operational_carry(_enforce_operational_precision(case.state, force_fp64=True))

for i, start in enumerate([1, 1 + seg, 1 + 2 * seg]):
    t0 = time.perf_counter()
    carry = _advance_chunk(carry, nl, jnp.asarray(start, dtype=jnp.int32), n_steps=seg, cadence=cad)
    jax.block_until_ready(carry.state.theta)
    dt = time.perf_counter() - t0
    kind = "COLD(compile+exec)" if i == 0 else "WARM(exec only?)"
    print(f"seg#{i} start={start} n={seg}: {dt:.1f}s  [{kind}]", flush=True)
print("If seg#1 and seg#2 are ~equal and << seg#0, the executable is reused (NO recompile).")
print("If seg#1 ~= seg#0, it recompiles per segment.")
