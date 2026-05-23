"""JAX-native tridiagonal solvers for vertical implicit column updates."""

from __future__ import annotations

import jax
from jax import config
import jax.numpy as jnp


config.update("jax_enable_x64", True)


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
