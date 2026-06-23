"""JAX-native tridiagonal solvers for vertical implicit column updates."""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

import jax
from jax import config
import jax.numpy as jnp


configure_jax_x64()


R_D = 287.0
CP_D = 1004.0
GRAVITY_M_S2 = 9.80665


def build_epssm_column_coefficients(
    theta: jax.Array,
    dz_m: jax.Array,
    *,
    dt: float,
    epssm: float = 0.1,
    theta_coefficient: jax.Array | None = None,
) -> tuple[jax.Array, ...]:
    """Builds MPAS/WRF-family off-centered tridiagonal coefficients.

    The coefficient names follow MPAS-A
    ``mpas_atm_time_integration.F:1589-1651``: ``dtseps``, ``cofrz``,
    ``cofwr``, ``cofwz``, ``coftz``, ``cofwt``, and the tridiagonal
    ``a/b/c`` rows. This is the epssm-aware builder used by the production
    nonhydrostatic ADR-023 column path. Inputs are batched columns with the
    leading axis as mass levels.
    """

    theta = jnp.asarray(theta)
    theta_for_coftz = theta if theta_coefficient is None else jnp.asarray(theta_coefficient, dtype=theta.dtype)
    dz_m = jnp.asarray(dz_m, dtype=theta.dtype)
    dtseps = 0.5 * float(dt) * (1.0 + float(epssm))
    rcv = R_D / (CP_D - R_D)
    c2 = CP_D * rcv

    rdzw = 1.0 / dz_m
    dz_face = 0.5 * (dz_m[:-1, :, :] + dz_m[1:, :, :])
    theta_face = 0.5 * (theta_for_coftz[:-1, :, :] + theta_for_coftz[1:, :, :])

    cofrz = dtseps * rdzw
    cofwt = 0.5 * dtseps * rcv * GRAVITY_M_S2 / theta

    cofwz = jnp.zeros((theta.shape[0] + 1,) + theta.shape[1:], dtype=theta.dtype)
    cofwr = jnp.zeros_like(cofwz)
    coftz = jnp.zeros_like(cofwz)
    cofwz = cofwz.at[1:-1, :, :].set(dtseps * c2 / dz_face)
    cofwr = cofwr.at[1:-1, :, :].set(0.5 * dtseps * GRAVITY_M_S2)
    coftz = coftz.at[1:-1, :, :].set(dtseps * theta_face)

    a = jnp.zeros_like(cofwz)
    b = jnp.ones_like(cofwz)
    c = jnp.zeros_like(cofwz)
    a_interior = (
        -cofwz[1:-1, :, :] * coftz[:-2, :, :] * rdzw[:-1, :, :]
        + cofwr[1:-1, :, :] * cofrz[:-1, :, :]
        - cofwt[:-1, :, :] * coftz[:-2, :, :] * rdzw[:-1, :, :]
    )
    b_interior = (
        1.0
        + cofwz[1:-1, :, :]
        * (
            coftz[1:-1, :, :] * rdzw[1:, :, :]
            + coftz[1:-1, :, :] * rdzw[:-1, :, :]
        )
        - coftz[1:-1, :, :]
        * (cofwt[1:, :, :] * rdzw[1:, :, :] - cofwt[:-1, :, :] * rdzw[:-1, :, :])
        + cofwr[1:-1, :, :] * (cofrz[1:, :, :] - cofrz[:-1, :, :])
    )
    c_interior = (
        -cofwz[1:-1, :, :] * coftz[2:, :, :] * rdzw[1:, :, :]
        - cofwr[1:-1, :, :] * cofrz[1:, :, :]
        + cofwt[1:, :, :] * coftz[2:, :, :] * rdzw[1:, :, :]
    )
    a = a.at[1:-1, :, :].set(a_interior)
    b = b.at[1:-1, :, :].set(b_interior)
    c = c.at[1:-1, :, :].set(c_interior)
    return cofrz, cofwr, cofwz, coftz, cofwt, rdzw, a, b, c


def solve_tridiagonal_thomas(a: jax.Array, b: jax.Array, c: jax.Array, rhs: jax.Array) -> jax.Array:
    """Solves a batched tridiagonal system along the leading vertical axis.

    Inputs have shape ``(nz, ...)`` with lower diagonal ``a``, diagonal ``b``,
    upper diagonal ``c``, and right-hand side ``rhs``. Boundary rows are
    supplied by the caller. The recurrence is pure ``lax.scan`` so it remains
    XLA-resident inside acoustic timestep scans.
    """

    rhs = jnp.asarray(rhs)
    a = jnp.asarray(a, dtype=rhs.dtype)
    b = jnp.asarray(b, dtype=rhs.dtype)
    c = jnp.asarray(c, dtype=rhs.dtype)

    cp0 = c[0] / b[0]
    dp0 = rhs[0] / b[0]

    def forward(carry, entries):
        cp_prev, dp_prev = carry
        ai, bi, ci, di = entries
        denom = bi - ai * cp_prev
        cp_i = ci / denom
        dp_i = (di - ai * dp_prev) / denom
        return (cp_i, dp_i), (cp_i, dp_i)

    (_, _), (cp_tail, dp_tail) = jax.lax.scan(
        forward,
        (cp0, dp0),
        (a[1:], b[1:], c[1:], rhs[1:]),
        unroll=False,
    )
    cp = jnp.concatenate((cp0[None, ...], cp_tail), axis=0)
    dp = jnp.concatenate((dp0[None, ...], dp_tail), axis=0)

    x_last = dp[-1]

    def backward(x_next, entries):
        cp_i, dp_i = entries
        x_i = dp_i - cp_i * x_next
        return x_i, x_i

    _, x_rev = jax.lax.scan(
        backward,
        x_last,
        (cp[:-1][::-1], dp[:-1][::-1]),
        unroll=False,
    )
    return jnp.concatenate((x_rev[::-1], x_last[None, ...]), axis=0)


def solve_tridiagonal_xla(a: jax.Array, b: jax.Array, c: jax.Array, rhs: jax.Array) -> jax.Array:
    """Solves a leading-axis tridiagonal system through XLA's primitive."""

    rhs = jnp.asarray(rhs)
    moved_rhs = jnp.moveaxis(rhs, 0, -1)
    moved_a = jnp.moveaxis(jnp.asarray(a, dtype=rhs.dtype), 0, -1)
    moved_b = jnp.moveaxis(jnp.asarray(b, dtype=rhs.dtype), 0, -1)
    moved_c = jnp.moveaxis(jnp.asarray(c, dtype=rhs.dtype), 0, -1)
    solved = jax.lax.linalg.tridiagonal_solve(
        moved_a,
        moved_b,
        moved_c,
        moved_rhs[..., None],
    )[..., 0]
    return jnp.moveaxis(solved, -1, 0)


def solve_tridiagonal(a: jax.Array, b: jax.Array, c: jax.Array, rhs: jax.Array) -> jax.Array:
    """Default conservative-column solver.

    Thomas scan is selected for ADR-023 v0 because it has simple JAX semantics
    and a compact transfer-audit surface. ``solve_tridiagonal_xla`` remains
    available as the future cyclic-reduction / backend-primitive comparison.
    """

    return solve_tridiagonal_thomas(a, b, c, rhs)
