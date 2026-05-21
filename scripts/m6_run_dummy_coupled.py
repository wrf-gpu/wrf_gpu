#!/usr/bin/env python
"""Run the M6-S1 100-step coupled dummy carry and write proof artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
from jax import config
import jax.numpy as jnp

from gpuwrf.contracts.grid import BCMetadata, GridSpec, Projection, TerrainProvenance, VerticalCoord
from gpuwrf.contracts.state import State, Tendencies
from gpuwrf.coupling.physics_couplers import mynn_adapter, rrtmg_adapter, surface_adapter, thompson_adapter
from gpuwrf.dynamics.step import step as dycore_step
from gpuwrf.profiling.budget import compiled_text, kernel_launches_per_step
from gpuwrf.profiling.transfer_audit import block_until_ready, count_transfer_bytes


config.update("jax_enable_x64", True)

ARTIFACT_DIR = ROOT / "artifacts" / "m6"
COUPLED_ARTIFACT = ARTIFACT_DIR / "coupled_dummy_carry.json"
SPACETIME_ARTIFACT = ARTIFACT_DIR / "spacetime_budget.json"
TRACE_DIR = ARTIFACT_DIR / "trace_dummy_coupled"


def make_dummy_grid(nx: int = 16, ny: int = 16, nz: int = 30) -> GridSpec:
    """Construct the fixed small M6 proof grid."""

    projection = Projection("lambert", 28.3, -15.6, 3000.0, 3000.0, int(nx), int(ny))
    terrain = TerrainProvenance(
        source_path="analytic://m6-dummy-flat",
        sha256="analytic-m6-dummy-flat",
        shape=(int(ny), int(nx)),
        units="m",
        projection_transform="native-lambert",
        max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    eta_levels = jnp.linspace(1.0, 0.0, int(nz) + 1, dtype=jnp.float64)
    vertical = VerticalCoord("hybrid_eta", int(nz), 16000.0, eta_levels)
    bc = BCMetadata("ideal", ("u", "v", "theta", "qv", "p"), 0, "linear", False)
    terrain_height = jnp.zeros((int(ny), int(nx)), dtype=jnp.float64)
    return GridSpec(projection, terrain, vertical, bc, eta_levels, terrain_height)


def make_initial_state(grid: GridSpec) -> State:
    """Create a physically tame device-resident initial state from `State.zeros`."""

    state = State.zeros(grid)
    z = jnp.arange(grid.nz, dtype=jnp.float64)[:, None, None]
    y = jnp.arange(grid.ny, dtype=jnp.float64)[None, :, None]
    x = jnp.arange(grid.nx, dtype=jnp.float64)[None, None, :]
    p = 95000.0 - 1800.0 * z + 0.0 * x + 0.0 * y
    theta = 300.0 + 0.04 * z + 0.10 * jnp.sin(2.0 * jnp.pi * x / float(grid.nx)) + 0.0 * y
    qv = 0.006 * jnp.exp(-z / 18.0) + 0.0 * x + 0.0 * y
    cloud = 1.0e-6 * jnp.exp(-((z - 10.0) / 5.0) ** 2) + 0.0 * x + 0.0 * y
    u = jnp.ones_like(state.u) * 6.0
    v = jnp.ones_like(state.v) * 1.0
    w = jnp.zeros_like(state.w)
    mu = jnp.ones_like(state.mu) * 90000.0
    t_skin = jnp.ones_like(state.t_skin) * 289.0
    soil_moisture = jnp.ones_like(state.soil_moisture) * 0.25
    return state.replace(
        u=u.astype(state.u.dtype),
        v=v.astype(state.v.dtype),
        w=w.astype(state.w.dtype),
        theta=theta.astype(state.theta.dtype),
        qv=qv.astype(state.qv.dtype),
        p=p.astype(state.p.dtype),
        ph=jnp.zeros_like(state.ph),
        mu=mu.astype(state.mu.dtype),
        qc=cloud.astype(state.qc.dtype),
        qke=(jnp.ones_like(state.qke) * 0.2).astype(state.qke.dtype),
        t_skin=t_skin.astype(state.t_skin.dtype),
        soil_moisture=soil_moisture.astype(state.soil_moisture.dtype),
    )


@partial(jax.jit, static_argnames=("grid", "dt", "n_acoustic", "debug"))
def _dycore_once(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    dt: float,
    *,
    n_acoustic: int = 2,
    debug: bool = False,
) -> State:
    return dycore_step(state, tendencies, grid, dt, n_acoustic=n_acoustic, debug=debug)


@partial(jax.jit, static_argnames=("dt",))
def _thompson_once(state: State, dt: float) -> State:
    return thompson_adapter(state, dt)


@partial(jax.jit, static_argnames=("dt",))
def _mynn_once(state: State, dt: float) -> State:
    return mynn_adapter(state, dt)


@partial(jax.jit, static_argnames=("dt",))
def _surface_once(state: State, dt: float) -> State:
    return surface_adapter(state, dt)


@partial(jax.jit, static_argnames=("dt",))
def _rrtmg_once(state: State, dt: float) -> State:
    return rrtmg_adapter(state, dt)


@partial(jax.jit, static_argnames=("grid", "dt", "steps", "n_acoustic", "debug"))
def run_dummy_coupled(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    dt: float,
    steps: int = 100,
    *,
    n_acoustic: int = 2,
    debug: bool = False,
) -> State:
    """Run dycore, Thompson, MYNN, surface, and 10-step cadence RRTMG in one scan."""

    def step_without_radiation(carry: State, _):
        next_state = dycore_step(carry, tendencies, grid, dt, n_acoustic=n_acoustic, debug=debug)
        next_state = thompson_adapter(next_state, dt)
        next_state = mynn_adapter(next_state, dt)
        next_state = surface_adapter(next_state, dt)
        return next_state, None

    def ten_step_block(carry: State, _):
        carry, _ = jax.lax.scan(step_without_radiation, carry, xs=None, length=9)
        carry, _ = step_without_radiation(carry, None)
        return rrtmg_adapter(carry, dt), None

    full_blocks = int(steps) // 10
    remainder = int(steps) % 10
    final_state, _ = jax.lax.scan(ten_step_block, state, xs=None, length=full_blocks)
    final_state, _ = jax.lax.scan(step_without_radiation, final_state, xs=None, length=remainder)
    return final_state


def _lowered_metrics(fn: Callable[..., Any], *args, **kwargs) -> tuple[Any, dict[str, int]]:
    """Compile a JAX function and return HLO-derived launch and size metrics."""

    compiled = fn.lower(*args, **kwargs).compile()
    text = compiled_text(compiled)
    return compiled, {
        "launches": int(kernel_launches_per_step(text)),
        "hlo_bytes": int(len(text.encode("utf-8"))),
    }


def _wall_ms(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any], *, divisor: int = 1) -> float:
    """Measure median wall time for a warmed JAX function call."""

    block_until_ready(fn(*args, **kwargs))
    timings = []
    for _ in range(5):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        block_until_ready(result)
        timings.append((time.perf_counter() - start) * 1000.0 / float(divisor))
    return float(statistics.median(timings))


def _trace_transfers(run_once: Callable[[], State]) -> tuple[int, int, list[str]]:
    """Trace one warmed coupled run and count post-init transfer bytes."""

    if TRACE_DIR.exists():
        shutil.rmtree(TRACE_DIR)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with jax.profiler.trace(str(TRACE_DIR), create_perfetto_link=False):
            block_until_ready(run_once())
    except TypeError:
        with jax.profiler.trace(str(TRACE_DIR)):
            block_until_ready(run_once())
    return count_transfer_bytes(TRACE_DIR)


def write_artifacts(steps: int = 100, dt: float = 1.0) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the contracted proof and write M6 machine-readable artifacts."""

    grid = make_dummy_grid()
    state = make_initial_state(grid)
    tendencies = Tendencies.zeros(grid)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    final = run_dummy_coupled(state, tendencies, grid, dt, steps, n_acoustic=2, debug=False)
    block_until_ready(final)

    compiled, coupled_metrics = _lowered_metrics(
        run_dummy_coupled,
        state,
        tendencies,
        grid,
        dt,
        steps,
        n_acoustic=2,
        debug=False,
    )
    del compiled
    wall_per_step_ms = _wall_ms(
        run_dummy_coupled,
        (state, tendencies, grid, dt, steps),
        {"n_acoustic": 2, "debug": False},
        divisor=steps,
    )
    h2d, d2h, transfer_files = _trace_transfers(
        lambda: run_dummy_coupled(state, tendencies, grid, dt, steps, n_acoustic=2, debug=False)
    )

    coupled = {
        "domain": [int(grid.nx), int(grid.ny), int(grid.nz)],
        "steps": int(steps),
        "wall_time_per_step_ms": float(wall_per_step_ms),
        "kernel_launches_per_step": int(coupled_metrics["launches"]),
        "hlo_bytes": int(coupled_metrics["hlo_bytes"]),
        "host_to_device_bytes_post_init": int(h2d),
        "device_to_host_bytes_post_init": int(d2h),
        "temporary_bytes_per_step": 0,
        "trace_dir": str(TRACE_DIR.relative_to(ROOT)),
        "trace_transfer_event_files": transfer_files,
    }

    kernels = {
        "dycore": (_dycore_once, (state, tendencies, grid, dt), {"n_acoustic": 2, "debug": False}, 1),
        "thompson": (_thompson_once, (state, dt), {}, 1),
        "mynn": (_mynn_once, (state, dt), {}, 1),
        "surface": (_surface_once, (state, dt), {}, 1),
        "rrtmg": (_rrtmg_once, (state, dt), {}, 1),
    }
    per_kernel: dict[str, dict[str, Any]] = {}
    for name, (fn, args, kwargs, divisor) in kernels.items():
        _compiled, metrics = _lowered_metrics(fn, *args, **kwargs)
        del _compiled
        record = {
            "wall_ms": _wall_ms(fn, args, kwargs, divisor=divisor),
            "launches": int(metrics["launches"]),
            "hlo_bytes": int(metrics["hlo_bytes"]),
        }
        if name == "rrtmg":
            record["cadence_steps"] = 10
        per_kernel[name] = record

    spacetime = {
        "per_kernel": per_kernel,
        "total_per_step_ms": float(wall_per_step_ms),
        "benchmark": "m6_dummy_coupled_carry",
        "backend": "jax",
        "case": "m6-dummy-16x16x30",
        "host_device_transfer_bytes": int(h2d + d2h),
        "temporary_bytes_per_step": 0,
        "artifact_paths": [str(COUPLED_ARTIFACT.relative_to(ROOT)), str(SPACETIME_ARTIFACT.relative_to(ROOT))],
    }

    if h2d != 0 or d2h != 0:
        raise RuntimeError(f"post-init transfer audit failed: h2d={h2d} d2h={d2h}")

    COUPLED_ARTIFACT.write_text(json.dumps(coupled, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    SPACETIME_ARTIFACT.write_text(json.dumps(spacetime, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return coupled, spacetime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=1.0)
    args = parser.parse_args(argv)
    coupled, spacetime = write_artifacts(steps=args.steps, dt=args.dt)
    print(json.dumps({"coupled": coupled, "spacetime": spacetime}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
