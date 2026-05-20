"""Forward-backward acoustic substeps for the reduced M4 dycore."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.halo import apply_halo
from gpuwrf.contracts.state import State

from .advection import halo_spec


def _dx(grid: GridSpec) -> float:
    """Shares static x-spacing access with the acoustic finite differences."""

    return float(grid.projection.dx_m)


def _dy(grid: GridSpec) -> float:
    """Shares static y-spacing access with the acoustic finite differences."""

    return float(grid.projection.dy_m)


def _dz(grid: GridSpec) -> float:
    """Defines the M4 flat-column vertical spacing for acoustic differences."""

    return float(grid.vertical.top_pressure_pa) / float(grid.nz)


def divergence_cgrid(state: State, grid: GridSpec):
    """Computes mass-point velocity divergence from C-grid face velocities."""

    return (
        (state.u[:, :, 1:] - state.u[:, :, :-1]) / _dx(grid)
        + (state.v[:, 1:, :] - state.v[:, :-1, :]) / _dy(grid)
        + (state.w[1:, :, :] - state.w[:-1, :, :]) / _dz(grid)
    )


def _grad_x_to_u(p, grid: GridSpec):
    """Maps mass pressure to x-face pressure gradients for the acoustic update."""

    grad = (p - jnp.roll(p, 1, axis=2)) / _dx(grid)
    return jnp.concatenate((grad, grad[:, :, :1]), axis=2)


def _grad_y_to_v(p, grid: GridSpec):
    """Maps mass pressure to y-face pressure gradients for the acoustic update."""

    grad = (p - jnp.roll(p, 1, axis=1)) / _dy(grid)
    return jnp.concatenate((grad, grad[:, :1, :]), axis=1)


def _grad_z_to_w(p, grid: GridSpec):
    """Maps mass pressure to vertical-face gradients with rigid top and bottom."""

    interior = (p[1:, :, :] - p[:-1, :, :]) / _dz(grid)
    lower = p[:1, :, :] * 0.0
    upper = p[-1:, :, :] * 0.0
    return jnp.concatenate((lower, interior, upper), axis=0)


def acoustic_once(state: State, grid: GridSpec, dt_sub: float) -> State:
    """Performs one conservative forward-backward acoustic substep."""

    c2 = 1.0
    pressure_coupling = 1.0e-3
    div = divergence_cgrid(state, grid)
    p_next = state.p - c2 * dt_sub * div
    u_next = state.u - pressure_coupling * dt_sub * _grad_x_to_u(p_next, grid)
    v_next = state.v - pressure_coupling * dt_sub * _grad_y_to_v(p_next, grid)
    w_next = state.w - pressure_coupling * dt_sub * _grad_z_to_w(p_next, grid)
    ph_next = state.ph + dt_sub * w_next
    return state.replace(u=u_next, v=v_next, w=w_next, p=p_next, ph=ph_next)


def forward_backward_acoustic(state: State, grid: GridSpec, dt: float, n_acoustic: int) -> State:
    """Runs the static-count acoustic scan used inside RK stages two and three."""

    dt_sub = dt / float(n_acoustic)

    def body(carry, _):
        """Keeps halo and acoustic math fused inside one scan body."""

        haloed = apply_halo(carry, halo_spec(grid))
        return acoustic_once(haloed, grid, dt_sub), None

    next_state, _ = jax.lax.scan(body, state, xs=None, length=int(n_acoustic))
    return apply_halo(next_state, halo_spec(grid))
