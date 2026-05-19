#!/usr/bin/env python3
"""Build the M3 state, run the dummy loop, and emit audit proof objects."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
from jax import config

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import State, Tendencies
from gpuwrf.profiling.budget import compiled_text, kernel_launches_per_step, median_step_us, write_hlo, write_spacetime_budget
from gpuwrf.profiling.transfer_audit import block_until_ready, write_transfer_audit
from gpuwrf.timestep.dummy_loop import run_dummy_loop


config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "m3"
SCRATCH = ROOT / "data" / "scratch" / "m3"
N_STEPS = 1000
DT = 3.0


def main() -> int:
    """Runs the complete M3 audit command as an idempotent CLI."""

    ART.mkdir(parents=True, exist_ok=True)
    (ART / "hlo_dump").mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)

    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)
    tendencies = Tendencies.zeros(grid)

    lowered = run_dummy_loop.lower(state, tendencies, DT, N_STEPS)
    compiled = lowered.compile()
    hlo_text = compiled_text(compiled)
    write_hlo(ART / "hlo_dump" / "dummy_loop.txt", hlo_text, SCRATCH / "dummy_loop_full_hlo.txt")

    result = compiled(state, tendencies, DT)
    block_until_ready(result)

    trace_dir = SCRATCH / "transfer_trace"
    if trace_dir.exists():
        shutil.rmtree(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(trace_dir), create_perfetto_link=False):
        traced = compiled(state, tendencies, DT)
        block_until_ready(traced)

    audit = write_transfer_audit(ART / "transfer_audit.json", N_STEPS, trace_dir)

    def run_once():
        """Calls the already-compiled dummy loop for median timing samples."""

        return compiled(state, tendencies, DT)

    wall_us = median_step_us(run_once, N_STEPS, samples=100)
    budget = write_spacetime_budget(
        ART / "spacetime_budget.json",
        state,
        tendencies,
        halo_buffer_bytes=0,
        launches_per_step=kernel_launches_per_step(hlo_text),
        wall_time_per_step_us=wall_us,
    )

    summary = {"transfer_audit": audit, "spacetime_budget": budget}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
