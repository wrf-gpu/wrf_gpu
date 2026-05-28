"""Hand-stripped M4 dycore timestep sibling for debug HLO identity checks."""

from __future__ import annotations

from functools import partial

import jax
from jax import config

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.halo import apply_halo
from gpuwrf.contracts.state import State, Tendencies

from .acoustic import forward_backward_acoustic
from .advection import compute_advection_tendencies, halo_spec
from .tendencies import add_scaled_tendencies


config.update("jax_enable_x64", True)


def _rk3_stage_stripped(origin: State, stage_state: State, base_tendencies: Tendencies, grid: GridSpec, dt_stage: float) -> State:
    """Duplicates the production RK stage with all debug calls physically absent."""

    tendencies = compute_advection_tendencies(stage_state, base_tendencies, grid)
    return add_scaled_tendencies(origin, tendencies, dt_stage)


def _rk3_step_stripped(state: State, base_tendencies: Tendencies, grid: GridSpec, dt: float, n_acoustic: int) -> State:
    """Duplicates production RK sequencing with no diagnostic branches."""

    s0 = apply_halo(state, halo_spec(grid))
    s1 = _rk3_stage_stripped(s0, s0, base_tendencies, grid, dt / 3.0)
    s1 = apply_halo(s1, halo_spec(grid))
    s2 = _rk3_stage_stripped(s0, s1, base_tendencies, grid, dt / 2.0)
    s2 = forward_backward_acoustic(s2, grid, dt / 2.0, n_acoustic)
    s2 = apply_halo(s2, halo_spec(grid))
    s3 = _rk3_stage_stripped(s0, s2, base_tendencies, grid, dt)
    s3 = forward_backward_acoustic(s3, grid, dt, n_acoustic)
    return apply_halo(s3, halo_spec(grid))


@partial(jax.jit, static_argnames=("grid", "dt", "n_acoustic"))
def step_debug_stripped(state: State, tendencies: Tendencies, grid: GridSpec, dt: float, *, n_acoustic: int = 4) -> State:
    """Runs one dycore step from source that contains no debug hook calls."""

    return _rk3_step_stripped(state, tendencies, grid, dt, n_acoustic)


@partial(jax.jit, static_argnames=("grid", "dt", "n_steps", "n_acoustic"))
def run_debug_stripped(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    dt: float,
    n_steps: int,
    *,
    n_acoustic: int = 4,
) -> State:
    """Runs the stripped sibling loop under one scan for source-level comparison."""

    def body(carry, _):
        """Keeps stripped scan structure aligned with production `run`."""

        return _rk3_step_stripped(carry, tendencies, grid, dt, n_acoustic), None

    next_state, _ = jax.lax.scan(body, state, xs=None, length=int(n_steps))
    return next_state
