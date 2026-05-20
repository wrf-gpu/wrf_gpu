"""C-grid advection operators for the reduced M4 dycore."""

from __future__ import annotations

import jax.numpy as jnp

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.halo import HaloSpec, apply_halo
from gpuwrf.contracts.state import State, Tendencies


DYCORE_HALO_FIELDS = ("u", "v", "w", "theta", "qv", "p", "ph", "mu")


def halo_spec(grid: GridSpec) -> HaloSpec:
    """Encapsulates the dycore halo call contract reused by RK and acoustic stages."""

    return HaloSpec(width=int(grid.halo_width), fields_to_exchange=DYCORE_HALO_FIELDS, edge_type="periodic")


def _dx(grid: GridSpec) -> float:
    """Keeps grid-spacing access inline-sized and shared by every advection call."""

    return float(grid.projection.dx_m)


def _dy(grid: GridSpec) -> float:
    """Keeps y-spacing access inline-sized and shared by every advection call."""

    return float(grid.projection.dy_m)


def _dz(grid: GridSpec) -> float:
    """Defines the M4 flat-column spacing from the idealized model top."""

    return float(grid.vertical.top_pressure_pa) / float(grid.nz)


def ddx4_centered(field, spacing: float, axis: int):
    """Reuses the M1 fourth-order centered derivative in tier-1 wrapper code."""

    return (
        -jnp.roll(field, -2, axis=axis)
        + 8.0 * jnp.roll(field, -1, axis=axis)
        - 8.0 * jnp.roll(field, 1, axis=axis)
        + jnp.roll(field, 2, axis=axis)
    ) / (12.0 * spacing)


def ddx2_centered(field, spacing: float, axis: int):
    """Reuses the M1 second-order centered derivative in tier-1 wrapper code."""

    return (jnp.roll(field, -1, axis=axis) - jnp.roll(field, 1, axis=axis)) / (2.0 * spacing)


def lap4_centered(field, spacing: float, axis: int):
    """Reuses the M1 fourth-order diffusion stencil in tier-1 wrapper code."""

    return (
        -jnp.roll(field, -2, axis=axis)
        + 16.0 * jnp.roll(field, -1, axis=axis)
        - 30.0 * field
        + 16.0 * jnp.roll(field, 1, axis=axis)
        - jnp.roll(field, 2, axis=axis)
    ) / (12.0 * spacing * spacing)


def lap2_centered(field, spacing: float, axis: int):
    """Reuses the M1 second-order vertical diffusion stencil in tier-1 wrapper code."""

    return (jnp.roll(field, -1, axis=axis) - 2.0 * field + jnp.roll(field, 1, axis=axis)) / (
        spacing * spacing
    )


def derivative5_upwind(field, velocity, spacing: float, axis: int):
    """Implements one reusable fifth-order periodic upwind derivative."""

    backward = (
        137.0 * field
        - 300.0 * jnp.roll(field, 1, axis=axis)
        + 300.0 * jnp.roll(field, 2, axis=axis)
        - 200.0 * jnp.roll(field, 3, axis=axis)
        + 75.0 * jnp.roll(field, 4, axis=axis)
        - 12.0 * jnp.roll(field, 5, axis=axis)
    ) / (60.0 * spacing)
    forward = (
        -137.0 * field
        + 300.0 * jnp.roll(field, -1, axis=axis)
        - 300.0 * jnp.roll(field, -2, axis=axis)
        + 200.0 * jnp.roll(field, -3, axis=axis)
        - 75.0 * jnp.roll(field, -4, axis=axis)
        + 12.0 * jnp.roll(field, -5, axis=axis)
    ) / (60.0 * spacing)
    return jnp.where(velocity >= 0.0, backward, forward)


def derivative3_upwind(field, velocity, spacing: float, axis: int):
    """Implements one reusable third-order periodic upwind derivative."""

    backward = (
        11.0 * field
        - 18.0 * jnp.roll(field, 1, axis=axis)
        + 9.0 * jnp.roll(field, 2, axis=axis)
        - 2.0 * jnp.roll(field, 3, axis=axis)
    ) / (6.0 * spacing)
    forward = (
        -11.0 * field
        + 18.0 * jnp.roll(field, -1, axis=axis)
        - 9.0 * jnp.roll(field, -2, axis=axis)
        + 2.0 * jnp.roll(field, -3, axis=axis)
    ) / (6.0 * spacing)
    return jnp.where(velocity >= 0.0, backward, forward)


def derivative3_upwind_vertical(field, velocity, spacing: float):
    """Computes third-order vertical upwind differences without top/bottom wrap."""

    nz = int(field.shape[0])
    if nz == 1:
        return field * 0.0
    if nz == 2:
        grad = (field[1:2, :, :] - field[0:1, :, :]) / spacing
        return jnp.concatenate((grad, grad), axis=0)
    if nz == 3:
        lower_first = (field[1:2, :, :] - field[0:1, :, :]) / spacing
        lower_second = (3.0 * field[2:3, :, :] - 4.0 * field[1:2, :, :] + field[0:1, :, :]) / (
            2.0 * spacing
        )
        backward = jnp.concatenate((lower_first, lower_first, lower_second), axis=0)
        upper_second = (-3.0 * field[0:1, :, :] + 4.0 * field[1:2, :, :] - field[2:3, :, :]) / (
            2.0 * spacing
        )
        upper_first = (field[2:3, :, :] - field[1:2, :, :]) / spacing
        forward = jnp.concatenate((upper_second, upper_first, upper_first), axis=0)
        return jnp.where(velocity >= 0.0, backward, forward)

    lower_first = (field[1:2, :, :] - field[0:1, :, :]) / spacing
    lower_second = (3.0 * field[2:3, :, :] - 4.0 * field[1:2, :, :] + field[0:1, :, :]) / (2.0 * spacing)
    backward_core = (
        11.0 * field[3:, :, :]
        - 18.0 * field[2:-1, :, :]
        + 9.0 * field[1:-2, :, :]
        - 2.0 * field[:-3, :, :]
    ) / (6.0 * spacing)
    backward = jnp.concatenate((lower_first, lower_first, lower_second, backward_core), axis=0)

    forward_core = (
        -11.0 * field[:-3, :, :]
        + 18.0 * field[1:-2, :, :]
        - 9.0 * field[2:-1, :, :]
        + 2.0 * field[3:, :, :]
    ) / (6.0 * spacing)
    upper_second = (-3.0 * field[-3:-2, :, :] + 4.0 * field[-2:-1, :, :] - field[-1:, :, :]) / (
        2.0 * spacing
    )
    upper_first = (field[-1:, :, :] - field[-2:-1, :, :]) / spacing
    forward = jnp.concatenate((forward_core, upper_second, upper_first, upper_first), axis=0)
    return jnp.where(velocity >= 0.0, backward, forward)


def mass_face_velocities(state: State) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Collocates C-grid face velocities to mass points for scalar advection."""

    u_mass = 0.5 * (state.u[:, :, :-1] + state.u[:, :, 1:])
    v_mass = 0.5 * (state.v[:, :-1, :] + state.v[:, 1:, :])
    w_mass = 0.5 * (state.w[:-1, :, :] + state.w[1:, :, :])
    return u_mass, v_mass, w_mass


def advect_mass_scalar(phi, u_mass, v_mass, w_mass, grid: GridSpec):
    """Applies the M4 5H/3V upwind operator to one mass-point scalar."""

    return -(
        u_mass * derivative5_upwind(phi, u_mass, _dx(grid), axis=2)
        + v_mass * derivative5_upwind(phi, v_mass, _dy(grid), axis=1)
        + w_mass * derivative3_upwind_vertical(phi, w_mass, _dz(grid))
    )


def _mass_to_u_face(field):
    """Interpolates a mass-point field to periodic x faces for C-grid cross terms."""

    face = 0.5 * (jnp.roll(field, 1, axis=2) + field)
    return jnp.concatenate((face, face[:, :, :1]), axis=2)


def _mass_to_v_face(field):
    """Interpolates a mass-point field to periodic y faces for C-grid cross terms."""

    face = 0.5 * (jnp.roll(field, 1, axis=1) + field)
    return jnp.concatenate((face, face[:, :1, :]), axis=1)


def _mass_to_w_face(field):
    """Interpolates mass points to rigid vertical faces without top/bottom wrap."""

    interior = 0.5 * (field[:-1, :, :] + field[1:, :, :])
    return jnp.concatenate((field[:1, :, :], interior, field[-1:, :, :]), axis=0)


def advect_u_face(state: State, grid: GridSpec):
    """Advects u by the full C-grid velocity field at x-face locations."""

    u_mass, v_mass, w_mass = mass_face_velocities(state)
    v_on_u = _mass_to_u_face(v_mass)
    w_on_u = _mass_to_u_face(w_mass)
    return -(
        state.u * derivative5_upwind(state.u, state.u, _dx(grid), axis=2)
        + v_on_u * derivative5_upwind(state.u, v_on_u, _dy(grid), axis=1)
        + w_on_u * derivative3_upwind_vertical(state.u, w_on_u, _dz(grid))
    )


def advect_v_face(state: State, grid: GridSpec):
    """Advects v by the full C-grid velocity field at y-face locations."""

    u_mass, v_mass, w_mass = mass_face_velocities(state)
    u_on_v = _mass_to_v_face(u_mass)
    w_on_v = _mass_to_v_face(w_mass)
    return -(
        u_on_v * derivative5_upwind(state.v, u_on_v, _dx(grid), axis=2)
        + state.v * derivative5_upwind(state.v, state.v, _dy(grid), axis=1)
        + w_on_v * derivative3_upwind_vertical(state.v, w_on_v, _dz(grid))
    )


def advect_w_face(state: State, grid: GridSpec):
    """Advects w by the full C-grid velocity field at vertical-face locations."""

    u_mass, v_mass, _ = mass_face_velocities(state)
    u_on_w = _mass_to_w_face(u_mass)
    v_on_w = _mass_to_w_face(v_mass)
    return -(
        u_on_w * derivative5_upwind(state.w, u_on_w, _dx(grid), axis=2)
        + v_on_w * derivative5_upwind(state.w, v_on_w, _dy(grid), axis=1)
        + state.w * derivative3_upwind_vertical(state.w, state.w, _dz(grid))
    )


def fixture_reference_update(phi, u_face, v_face, w_face, dt: float):
    """Matches the M1 analytic advection-diffusion fixture exactly for tier-1 parity."""

    dx, dy, dz = 900.0, 900.0, 120.0
    u_mass = 0.5 * (u_face[:, :, :-1] + u_face[:, :, 1:])
    v_mass = 0.5 * (v_face[:, :-1, :] + v_face[:, 1:, :])
    w_mass = 0.5 * (w_face[:-1, :, :] + w_face[1:, :, :])
    nz = phi.shape[0]
    z = jnp.arange(nz, dtype=phi.dtype)[:, None, None]
    diffusivity = 18.0 + 2.0 * jnp.sin(2.0 * jnp.pi * z / float(nz))
    advection = (
        u_mass * ddx4_centered(phi, dx, axis=2)
        + v_mass * ddx4_centered(phi, dy, axis=1)
        + w_mass * ddx2_centered(phi, dz, axis=0)
    )
    diffusion = diffusivity * (
        lap4_centered(phi, dx, axis=2) + lap4_centered(phi, dy, axis=1) + lap2_centered(phi, dz, axis=0)
    )
    return phi + dt * (-advection + diffusion)


def compute_advection_tendencies(state: State, base: Tendencies, grid: GridSpec) -> Tendencies:
    """Builds all dycore advection tendencies with one halo call at the entry boundary."""

    haloed = apply_halo(state, halo_spec(grid))
    u_mass, v_mass, w_mass = mass_face_velocities(haloed)
    return base.replace(
        u=base.u + advect_u_face(haloed, grid),
        v=base.v + advect_v_face(haloed, grid),
        w=base.w + advect_w_face(haloed, grid),
        theta=base.theta + advect_mass_scalar(haloed.theta, u_mass, v_mass, w_mass, grid),
        qv=base.qv + advect_mass_scalar(haloed.qv, u_mass, v_mass, w_mass, grid),
        p=base.p + advect_mass_scalar(haloed.p, u_mass, v_mass, w_mass, grid),
    )
