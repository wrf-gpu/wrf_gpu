"""Wicker-Skamarock-style three-stage large timestep for M4."""

from __future__ import annotations

import jax.numpy as jnp

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.halo import apply_halo
from gpuwrf.contracts.state import State, Tendencies
from gpuwrf.debug.asserts import assert_finite, assert_physical_bounds
from gpuwrf.debug.snapshots import snapshot

from .acoustic import forward_backward_acoustic
from .advection import compute_advection_tendencies, halo_spec
from .tendencies import add_scaled_tendencies


def _assert_state(state: State, stage: str, debug: bool) -> State:
    """Collects debug assertions so stage code stays readable."""

    if not debug:
        return state
    return state.replace(
        u=assert_finite(state.u, f"{stage}.u", enabled=debug),
        v=assert_finite(state.v, f"{stage}.v", enabled=debug),
        w=assert_finite(state.w, f"{stage}.w", enabled=debug),
        theta=assert_physical_bounds(state.theta, 0.0, 500.0, f"{stage}.theta", enabled=debug),
        qv=assert_physical_bounds(state.qv, 0.0, 1.0, f"{stage}.qv", enabled=debug),
        p=assert_finite(state.p, f"{stage}.p", enabled=debug),
        ph=assert_finite(state.ph, f"{stage}.ph", enabled=debug),
        mu=assert_finite(state.mu, f"{stage}.mu", enabled=debug),
    )


def rk3_stage(state: State, base_tendencies: Tendencies, grid: GridSpec, dt_stage: float) -> State:
    """Encapsulates one tendency calculation plus Euler advance."""

    tendencies = compute_advection_tendencies(state, base_tendencies, grid)
    return add_scaled_tendencies(state, tendencies, dt_stage)


def rk3_step(state: State, base_tendencies: Tendencies, grid: GridSpec, dt: float, n_acoustic: int, debug: bool) -> State:
    """Runs the M4 three-stage large step with acoustic stages two and three."""

    s0 = apply_halo(state, halo_spec(grid))
    s1 = rk3_stage(s0, base_tendencies, grid, dt / 3.0)
    s1 = apply_halo(s1, halo_spec(grid))
    if debug:
        s1 = snapshot(_assert_state(s1, "rk1", debug), "rk1", enabled=debug)

    s2 = rk3_stage(s1, base_tendencies, grid, dt / 2.0)
    s2 = forward_backward_acoustic(s2, grid, dt / 2.0, n_acoustic)
    s2 = apply_halo(s2, halo_spec(grid))
    if debug:
        s2 = snapshot(_assert_state(s2, "rk2_acoustic", debug), "rk2_acoustic", enabled=debug)

    s3 = rk3_stage(s2, base_tendencies, grid, dt)
    s3 = forward_backward_acoustic(s3, grid, dt, n_acoustic)
    s3 = apply_halo(s3, halo_spec(grid))
    if debug:
        s3 = snapshot(_assert_state(s3, "rk3_acoustic", debug), "rk3_acoustic", enabled=debug)
    return s3


def rk3_scalar_decay(y0, dt: float, n_steps: int):
    """Provides a tiny RK3 reference ODE used only by the operator test."""

    def rhs(y):
        """Keeps the scalar decay equation local to the manufactured test helper."""

        return -y

    value = y0
    for _ in range(int(n_steps)):
        k1 = rhs(value)
        y1 = value + dt * k1
        k2 = rhs(y1)
        y2 = 0.75 * value + 0.25 * (y1 + dt * k2)
        k3 = rhs(y2)
        value = (value + 2.0 * (y2 + dt * k3)) / 3.0
    return jnp.asarray(value)
