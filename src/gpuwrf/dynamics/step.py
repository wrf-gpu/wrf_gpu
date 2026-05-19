"""Public M4 dycore timestep entry points with static debug branching."""

from __future__ import annotations

from functools import partial

import jax
from jax import config

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import State, Tendencies

from .rk3 import rk3_step


config.update("jax_enable_x64", True)


def _step_impl(state: State, tendencies: Tendencies, grid: GridSpec, dt: float, n_acoustic: int, debug: bool) -> State:
    """Keeps jitted public wrappers thin and shares the implementation with tests."""

    return rk3_step(state, tendencies, grid, dt, n_acoustic, debug)


@partial(jax.jit, static_argnames=("grid", "dt", "n_acoustic", "debug"))
def step(state: State, tendencies: Tendencies, grid: GridSpec, dt: float, *, n_acoustic: int = 4, debug: bool = False) -> State:
    """Runs one split-explicit dycore step as a static-parameter JAX program."""

    return _step_impl(state, tendencies, grid, dt, n_acoustic, debug)


@partial(jax.jit, static_argnames=("grid", "dt", "n_steps", "n_acoustic", "debug"))
def run(
    state: State,
    tendencies: Tendencies,
    grid: GridSpec,
    dt: float,
    n_steps: int,
    *,
    n_acoustic: int = 4,
    debug: bool = False,
) -> State:
    """Runs the whole dycore loop under one jitted scan."""

    def body(carry, _):
        """Holds the scan body to one dycore step without extra Python state."""

        return _step_impl(carry, tendencies, grid, dt, n_acoustic, debug), None

    next_state, _ = jax.lax.scan(body, state, xs=None, length=int(n_steps))
    return next_state


@partial(jax.jit, static_argnames=("grid", "dt", "n_acoustic"))
def step_stripped_reference(state: State, tendencies: Tendencies, grid: GridSpec, dt: float, *, n_acoustic: int = 4) -> State:
    """Provides an owned-path stripped production reference for HLO identity checks."""

    return _step_impl(state, tendencies, grid, dt, n_acoustic, False)
