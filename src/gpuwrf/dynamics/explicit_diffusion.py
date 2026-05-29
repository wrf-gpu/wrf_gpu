"""Explicit diffusion tendencies for the dry dycore (Block 1 stabiliser).

Two WRF-faithful paths, both returned as *uncoupled* tendencies (du/dt etc.) so
they add into the operational RK tendency convention (the roll-advection path
also produces uncoupled tendencies):

1. ``sixth_order_diffusion_tendency`` — WRF ``sixth_order_diffusion``
   (``module_big_step_utilities_em.F:6504-6920``) with the monotonic flux
   limiter (``diff_6th_opt=2``).  This is the operational d02 numerical filter
   (``diff_6th_opt=2``, ``diff_6th_factor=0.12``) that suppresses 2dx noise.

2. ``constant_k_diffusion_tendency`` — WRF ``horizontal_diffusion`` /
   ``vertical_diffusion`` with a constant eddy viscosity ``khdif=kvdif=K``
   (``module_big_step_utilities_em.F:2999-3234``).  The Straka et al. (1993)
   density-current reference solution is *defined* with constant ν = 75 m²/s on
   u, v, θ, so this is part of the test definition, not a masking clamp.

Periodic-x/-y only (the idealized + audit configuration); map factors are unity
for the idealized slab and treated as unity here.  These act as a documented
scoped restriction matching the F7-B gate configuration.
"""

from __future__ import annotations

import jax
from jax import config
import jax.numpy as jnp


config.update("jax_enable_x64", True)


def _dflux6(field: jax.Array, axis: int) -> tuple[jax.Array, jax.Array]:
    """WRF 6th-order diffusive flux pair (Xue eq. 3) at faces p0 (i) and p1 (i+1).

    ``dflux_p0 = 10*(f(i)-f(i-1)) - 5*(f(i+1)-f(i-2)) + (f(i+2)-f(i-3))`` located
    at the left face of cell ``i``; ``dflux_p1`` is the same shifted to ``i+1``.
    Returns ``(dflux_p0, dflux_p1, grad_p0, grad_p1)`` where the gradients
    ``f(i)-f(i-1)`` / ``f(i+1)-f(i)`` are used by the monotonic limiter.
    """

    fm3 = jnp.roll(field, 3, axis=axis)
    fm2 = jnp.roll(field, 2, axis=axis)
    fm1 = jnp.roll(field, 1, axis=axis)
    f0 = field
    fp1 = jnp.roll(field, -1, axis=axis)
    fp2 = jnp.roll(field, -2, axis=axis)
    fp3 = jnp.roll(field, -3, axis=axis)
    dflux_p0 = 10.0 * (f0 - fm1) - 5.0 * (fp1 - fm2) + (fp2 - fm3)
    dflux_p1 = 10.0 * (fp1 - f0) - 5.0 * (fp2 - fm1) + (fp3 - fm2)
    grad_p0 = f0 - fm1
    grad_p1 = fp1 - f0
    return dflux_p0, dflux_p1, grad_p0, grad_p1


def _sixth_axis(field: jax.Array, axis: int, coef: float, monotonic: bool) -> jax.Array:
    """Uncoupled 6th-order diffusion tendency along one axis (periodic).

    WRF coupled form: ``tend = coef*(mu_p1*dflux_p1 - mu_p0*dflux_p0)``, and the
    prognostic later divides by mass; with unit map factors and the perturbation
    update dividing by the same mass, the *uncoupled* tendency is
    ``coef*(dflux_p1 - dflux_p0)`` (mu cancels to first order for the column).
    """

    dflux_p0, dflux_p1, grad_p0, grad_p1 = _dflux6(field, axis)
    if monotonic:
        # diff_6th_opt=2: prohibit up-gradient diffusion (Xue eq. 10 variant).
        dflux_p0 = jnp.where(dflux_p0 * grad_p0 <= 0.0, 0.0, dflux_p0)
        dflux_p1 = jnp.where(dflux_p1 * grad_p1 <= 0.0, 0.0, dflux_p1)
    return float(coef) * (dflux_p1 - dflux_p0)


def sixth_order_diffusion_tendency(
    field: jax.Array,
    *,
    dt: float,
    diff_6th_factor: float,
    horizontal_only: bool = True,
    monotonic: bool = True,
) -> jax.Array:
    """Return the WRF 6th-order numerical-diffusion tendency for one 3-D field.

    Source: WRF ``module_big_step_utilities_em.F:6504-6920``.  The coefficient is
    ``diff_6th_coef = diff_6th_factor * 0.015625 / (2*dt)`` (``:6605``).  WRF
    applies the filter on the horizontal coordinate surfaces (x and y); the
    one-row idealized slab and the audit case are effectively x-only in the
    horizontal, with the y-axis a singleton (its roll-stencil contribution is
    zero on a 1-wide axis).
    """

    coef = float(diff_6th_factor) * 0.015625 / (2.0 * float(dt))
    tend = _sixth_axis(field, axis=2, coef=coef, monotonic=monotonic)
    if field.shape[1] > 1:
        tend = tend + _sixth_axis(field, axis=1, coef=coef, monotonic=monotonic)
    if not horizontal_only and field.shape[0] > 1:
        # WRF applies the 6th-order filter only on coordinate (horizontal)
        # surfaces; vertical 6th-order is not part of diff_6th_opt.  Kept off.
        pass
    return tend


def _laplacian_axis_periodic(field: jax.Array, axis: int, spacing: float) -> jax.Array:
    """Second-order periodic Laplacian d2f/dx2 along ``axis``."""

    return (
        jnp.roll(field, -1, axis=axis) - 2.0 * field + jnp.roll(field, 1, axis=axis)
    ) / (float(spacing) * float(spacing))


def _laplacian_z_rigid(field: jax.Array, spacing) -> jax.Array:
    """Second-order vertical Laplacian with zero-gradient (rigid) top/bottom.

    ``spacing`` may be a Python float or a traced JAX scalar (used inside jit).
    """

    nz = int(field.shape[0])
    if nz < 3:
        return jnp.zeros_like(field)
    sp2 = jnp.asarray(spacing, dtype=field.dtype) ** 2
    interior = (field[2:, :, :] - 2.0 * field[1:-1, :, :] + field[:-2, :, :]) / sp2
    lap = jnp.zeros_like(field)
    lap = lap.at[1:-1, :, :].set(interior)
    return lap


def constant_k_diffusion_tendency(
    field: jax.Array,
    *,
    k_m2_s: float,
    dx_m: float,
    dy_m: float,
    dz_m: float,
    horizontal: bool = True,
    vertical: bool = True,
) -> jax.Array:
    """Return a constant-viscosity (``K``) 2nd-order diffusion tendency.

    ``du/dt += K * (d2/dx2 + d2/dy2 + d2/dz2) field``.  Source: WRF
    ``horizontal_diffusion`` / ``vertical_diffusion`` with constant ``xkmhd=K``
    (``module_big_step_utilities_em.F:2999-3234``).  This is the Straka et al.
    (1993) ν = 75 m²/s definition (the reference solution is *defined* with it).
    Periodic horizontal, rigid vertical boundaries.
    """

    tend = jnp.zeros_like(field)
    if horizontal:
        tend = tend + float(k_m2_s) * _laplacian_axis_periodic(field, axis=2, spacing=dx_m)
        if field.shape[1] > 1:
            tend = tend + float(k_m2_s) * _laplacian_axis_periodic(field, axis=1, spacing=dy_m)
    if vertical and field.shape[0] > 1:
        tend = tend + float(k_m2_s) * _laplacian_z_rigid(field, spacing=dz_m)
    return tend


__all__ = [
    "sixth_order_diffusion_tendency",
    "constant_k_diffusion_tendency",
]
