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
        + w_mass * derivative3_upwind(phi, w_mass, _dz(grid), axis=0)
    )


def advect_u_face(u, grid: GridSpec):
    """Advects u on its own face line; shared by dycore tendencies and tests."""

    return -(u * derivative5_upwind(u, u, _dx(grid), axis=2))


def advect_v_face(v, grid: GridSpec):
    """Advects v on its own face line; shared by dycore tendencies and tests."""

    return -(v * derivative5_upwind(v, v, _dy(grid), axis=1))


def advect_w_face(w, grid: GridSpec):
    """Advects w vertically on its own face line; shared by dycore tendencies and tests."""

    return -(w * derivative3_upwind(w, w, _dz(grid), axis=0))


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
        u=base.u + advect_u_face(haloed.u, grid),
        v=base.v + advect_v_face(haloed.v, grid),
        w=base.w + advect_w_face(haloed.w, grid),
        theta=base.theta + advect_mass_scalar(haloed.theta, u_mass, v_mass, w_mass, grid),
        qv=base.qv + advect_mass_scalar(haloed.qv, u_mass, v_mass, w_mass, grid),
        p=base.p + advect_mass_scalar(haloed.p, u_mass, v_mass, w_mass, grid),
    )
