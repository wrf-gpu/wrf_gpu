"""JAX implementation of the M2 analytic stencil fixture."""

from __future__ import annotations

import jax
from jax import config
import jax.numpy as jnp


config.update("jax_enable_x64", True)

DX = 900.0
DY = 900.0
DZ = 120.0
DT = 3.0


def _ddx4(field: jax.Array, spacing: float, axis: int) -> jax.Array:
    return (
        -jnp.roll(field, -2, axis=axis)
        + 8.0 * jnp.roll(field, -1, axis=axis)
        - 8.0 * jnp.roll(field, 1, axis=axis)
        + jnp.roll(field, 2, axis=axis)
    ) / (12.0 * spacing)


def _ddx2(field: jax.Array, spacing: float, axis: int) -> jax.Array:
    return (jnp.roll(field, -1, axis=axis) - jnp.roll(field, 1, axis=axis)) / (2.0 * spacing)


def _lap4(field: jax.Array, spacing: float, axis: int) -> jax.Array:
    return (
        -jnp.roll(field, -2, axis=axis)
        + 16.0 * jnp.roll(field, -1, axis=axis)
        - 30.0 * field
        + 16.0 * jnp.roll(field, 1, axis=axis)
        - jnp.roll(field, 2, axis=axis)
    ) / (12.0 * spacing * spacing)


def _lap2(field: jax.Array, spacing: float, axis: int) -> jax.Array:
    return (jnp.roll(field, -1, axis=axis) - 2.0 * field + jnp.roll(field, 1, axis=axis)) / (spacing * spacing)


@jax.jit
def stencil_advdiff(
    phi_initial: jax.Array,
    u_face: jax.Array,
    v_face: jax.Array,
    w_face: jax.Array,
) -> jax.Array:
    """Run one periodic fp64 advection-diffusion update."""

    u_mass = 0.5 * (u_face[:, :, :-1].astype(jnp.float64) + u_face[:, :, 1:].astype(jnp.float64))
    v_mass = 0.5 * (v_face[:, :-1, :].astype(jnp.float64) + v_face[:, 1:, :].astype(jnp.float64))
    w_mass = 0.5 * (w_face[:-1, :, :].astype(jnp.float64) + w_face[1:, :, :].astype(jnp.float64))

    nz = phi_initial.shape[0]
    z = jnp.arange(nz, dtype=jnp.float64)[:, None, None]
    diffusivity = 18.0 + 2.0 * jnp.sin(2.0 * jnp.pi * z / nz)

    advection = (
        u_mass * _ddx4(phi_initial, DX, axis=2)
        + v_mass * _ddx4(phi_initial, DY, axis=1)
        + w_mass * _ddx2(phi_initial, DZ, axis=0)
    )
    diffusion = diffusivity * (
        _lap4(phi_initial, DX, axis=2) + _lap4(phi_initial, DY, axis=1) + _lap2(phi_initial, DZ, axis=0)
    )
    return (phi_initial + DT * (-advection + diffusion)).astype(jnp.float64)
