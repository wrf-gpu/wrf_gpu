"""Opt-in halo-fed horizontal operators for x-sharded domains.

These helpers are not wired into the default dycore path. They are the small
operator bodies S3 uses to verify that an x-domain shard with refreshed halos
can reproduce the owned interior of the current single-domain WRF formulas.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.dynamics.explicit_diffusion import sixth_order_diffusion_tendency
from gpuwrf.dynamics.flux_advection import flux5_face_periodic


def trim_x_halo(field: jax.Array, *, halo_width: int) -> jax.Array:
    """Return the owned mass-grid x interval from a haloed local field."""

    h = int(halo_width)
    if h <= 0:
        return field
    return field[..., h:-h]


def trim_x_face_halo(field: jax.Array, *, halo_width: int) -> jax.Array:
    """Return the owned x-face interval, including the right boundary face."""

    h = int(halo_width)
    if h <= 0:
        return field
    return field[..., h:-h]


def sharded_flux5_face_periodic_x(
    field: jax.Array,
    vel: jax.Array,
    *,
    halo_width: int = 3,
) -> jax.Array:
    """Return owned x-face flux values from a haloed local scalar field."""

    if int(halo_width) < 3:
        raise ValueError("5th-order flux needs at least three x halo cells")
    return trim_x_halo(flux5_face_periodic(field, vel, axis=-1), halo_width=halo_width)


def sharded_sixth_order_diffusion_tendency(
    field: jax.Array,
    *,
    dt: float,
    diff_6th_factor: float,
    halo_width: int = 3,
    horizontal_only: bool = True,
    monotonic: bool = True,
) -> jax.Array:
    """Return owned sixth-order diffusion tendency from a haloed local field."""

    if int(halo_width) < 3:
        raise ValueError("6th-order diffusion needs at least three x halo cells")
    local = sixth_order_diffusion_tendency(
        field,
        dt=dt,
        diff_6th_factor=diff_6th_factor,
        horizontal_only=horizontal_only,
        monotonic=monotonic,
    )
    return trim_x_halo(local, halo_width=halo_width)


def sharded_x_staggered_divergence(
    u_face: jax.Array,
    *,
    rdx: float,
    halo_width: int = 1,
) -> jax.Array:
    """Return owned ``rdx * (u_east - u_west)`` on mass cells."""

    h = int(halo_width)
    if h < 0:
        raise ValueError("halo_width must be nonnegative")
    owned = int(u_face.shape[-1]) - 2 * h - 1
    if owned < 1:
        raise ValueError("haloed x-face field has no owned mass cells")
    faces = u_face[..., h : h + owned + 1]
    return float(rdx) * (faces[..., 1:] - faces[..., :-1])


def sharded_x_face_pair_3d_edge(
    field: jax.Array,
    *,
    halo_width: int,
    global_start: jax.Array | int,
    global_nx: int,
) -> tuple[jax.Array, jax.Array]:
    """Return acoustic edge-padded x-face pairs for owned faces.

    Internal shard faces use halo values. The physical west/east domain edges
    reproduce the existing acoustic edge-repeat convention.
    """

    h = int(halo_width)
    if h < 1:
        raise ValueError("x-face pair extraction needs at least one halo cell")
    owned = int(field.shape[-1]) - 2 * h
    if owned < 1:
        raise ValueError("haloed mass field has no owned x cells")
    left = field[:, :, h - 1 : h + owned]
    right = field[:, :, h : h + owned + 1]
    start = jnp.asarray(global_start, dtype=jnp.int32)
    is_first = start == 0
    is_last = start + owned == int(global_nx)
    left = left.at[:, :, 0].set(jnp.where(is_first, right[:, :, 0], left[:, :, 0]))
    right = right.at[:, :, -1].set(jnp.where(is_last, left[:, :, -1], right[:, :, -1]))
    return left, right


def sharded_x_face_pressure_dpn(
    p: jax.Array,
    *,
    fnm: jax.Array,
    fnp: jax.Array,
    cf1: jax.Array,
    cf2: jax.Array,
    cf3: jax.Array,
    halo_width: int,
    global_start: jax.Array | int,
    global_nx: int,
    top_lid: bool = False,
) -> jax.Array:
    """Return owned x-face pressure ``dpn`` used by acoustic horizontal PGF."""

    left, right = sharded_x_face_pair_3d_edge(
        p,
        halo_width=halo_width,
        global_start=global_start,
        global_nx=global_nx,
    )
    pair_sum = left + right
    _, ny, nx_face = pair_sum.shape
    bottom = 0.5 * (cf1 * pair_sum[0] + cf2 * pair_sum[1] + cf3 * pair_sum[2])
    interior = 0.5 * (
        fnm[1:, None, None] * pair_sum[1:, :, :]
        + fnp[1:, None, None] * pair_sum[:-1, :, :]
    )
    if bool(top_lid):
        top = 0.5 * (cf1 * pair_sum[-1, :, :] + cf2 * pair_sum[-2, :, :] + cf3 * pair_sum[-3, :, :])
    else:
        top = jnp.zeros((ny, nx_face), dtype=p.dtype)
    return jnp.concatenate([bottom[None, :, :], interior, top[None, :, :]], axis=0)


__all__ = [
    "sharded_flux5_face_periodic_x",
    "sharded_sixth_order_diffusion_tendency",
    "sharded_x_face_pair_3d_edge",
    "sharded_x_face_pressure_dpn",
    "sharded_x_staggered_divergence",
    "trim_x_face_halo",
    "trim_x_halo",
]
