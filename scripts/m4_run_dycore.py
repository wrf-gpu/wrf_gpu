#!/usr/bin/env python3
"""Run the M4 dycore and emit profile, transfer, budget, and HLO proof objects."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax  # noqa: E402
from jax import config  # noqa: E402

from gpuwrf.dynamics.step import run, step  # noqa: E402
from gpuwrf.profiling.budget import compiled_text, kernel_launches_per_step, median_step_us, write_hlo, write_spacetime_budget  # noqa: E402
from gpuwrf.profiling.transfer_audit import block_until_ready, visible_gpu_name, write_transfer_audit  # noqa: E402
from gpuwrf.validation.tier2 import density_current_state, make_ideal_grid  # noqa: E402


config.update("jax_enable_x64", True)

ART = ROOT / "artifacts" / "m4"
SCRATCH = ROOT / "data" / "scratch" / "m4"
N_STEPS = 100
DT = 2.0
N_ACOUSTIC = 4


def _write_json(path: Path, payload: dict) -> None:
    """Keeps proof-object JSON formatting stable across M4 scripts."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    """Runs the dycore proof command as an idempotent CLI."""

    ART.mkdir(parents=True, exist_ok=True)
    (ART / "hlo_dump").mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)

    grid = make_ideal_grid()
    state, tendencies = density_current_state(grid)

    step_hlo = compiled_text(step.lower(state, tendencies, grid, DT, n_acoustic=N_ACOUSTIC, debug=False).compile())
    write_hlo(ART / "hlo_dump" / "dycore_step_production.txt", step_hlo, SCRATCH / "dycore_step_production_full.txt")

    result = run(state, tendencies, grid, DT, N_STEPS, n_acoustic=N_ACOUSTIC, debug=False)
    block_until_ready(result)

    trace_dir = SCRATCH / "transfer_trace"
    if trace_dir.exists():
        shutil.rmtree(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(trace_dir), create_perfetto_link=False):
        traced = run(state, tendencies, grid, DT, N_STEPS, n_acoustic=N_ACOUSTIC, debug=False)
        block_until_ready(traced)

    audit = write_transfer_audit(ART / "transfer_audit.json", N_STEPS, trace_dir)

    def run_once():
        """Calls the cached dycore scan for median timing samples."""

        return run(state, tendencies, grid, DT, N_STEPS, n_acoustic=N_ACOUSTIC, debug=False)

    wall_us = median_step_us(run_once, N_STEPS, samples=20)
    launches = kernel_launches_per_step(step_hlo)
    budget = write_spacetime_budget(
        ART / "spacetime_budget.json",
        state,
        tendencies,
        halo_buffer_bytes=0,
        launches_per_step=launches,
        wall_time_per_step_us=wall_us,
    )
    budget["case"] = "m4-dycore-40x80x80-100-step"
    budget["notes"]["flops_per_cell_per_step"] = "not statically counted for split-explicit dycore; M5 profiler will replace this estimate"
    _write_json(ART / "spacetime_budget.json", budget)

    profile = {
        "benchmark": "m4_dycore_step",
        "backend": "jax",
        "hardware": "RTX 5090 32GB",
        "case": "m4-dycore-40x80x80",
        "wall_time_s": float(wall_us) * 1.0e-6,
        "kernel_launches": int(launches),
        "host_device_transfer_bytes": int(audit["host_to_device_bytes_post_init"]) + int(audit["device_to_host_bytes_post_init"]),
        "occupancy_pct": None,
        "registers_per_thread": None,
        "local_memory_bytes": None,
        "artifact_paths": [
            "artifacts/m4/hlo_dump/dycore_step_production.txt",
            "artifacts/m4/transfer_audit.json",
            "artifacts/m4/spacetime_budget.json",
        ],
        "jax_version": jax.__version__,
        "gpu_name": visible_gpu_name(),
        "iterations": N_STEPS,
        "n_acoustic": N_ACOUSTIC,
        "host_to_device_bytes_post_init": int(audit["host_to_device_bytes_post_init"]),
        "device_to_host_bytes_post_init": int(audit["device_to_host_bytes_post_init"]),
        "temporary_bytes_per_step": int(budget["temporary_bytes_per_step"]),
        "kernel_launches_per_dycore_step": int(launches),
        "profiler_limitation": "ncu/nsys perf counters not used in worker command; HLO-derived launch estimate plus JAX trace transfer audit",
    }
    _write_json(ART / "dycore_profile.json", profile)
    print(json.dumps({"dycore_profile": profile, "transfer_audit": audit, "spacetime_budget": budget}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
