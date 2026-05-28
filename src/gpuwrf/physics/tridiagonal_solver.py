"""Reusable JAX tridiagonal solver for vertical implicit column updates."""

from __future__ import annotations

import jax
from jax import config
import jax.numpy as jnp


config.update("jax_enable_x64", True)


def solve_tridiagonal(a, b, c, d):
    """Solves a batched tridiagonal system along the last axis.

    The production path uses XLA's tridiagonal primitive so the MYNN column
    keeps the M5-S2 launch budget. `solve_tridiagonal_thomas_reference` below
    mirrors WRF MYNN's `tridiag2` Thomas algorithm at module_bl_mynn.F90 lines
    5318-5350 and is used by tests as the independent recurrence reference.
    """

    a = jnp.asarray(a, dtype=d.dtype)
    b = jnp.asarray(b, dtype=d.dtype)
    c = jnp.asarray(c, dtype=d.dtype)
    d = jnp.asarray(d)
    if d.ndim == b.ndim:
        return jax.lax.linalg.tridiagonal_solve(a, b, c, d[..., None])[..., 0]
    return jax.lax.linalg.tridiagonal_solve(a, b, c, d)


def solve_tridiagonal_thomas_reference(a, b, c, d):
    """Thomas-recurrence reference matching WRF `tridiag2` for tests."""

    a = jnp.asarray(a, dtype=d.dtype)
    b = jnp.asarray(b, dtype=d.dtype)
    c = jnp.asarray(c, dtype=d.dtype)
    d = jnp.asarray(d)

    cp0 = c[..., 0] / b[..., 0]
    dp0 = d[..., 0] / b[..., 0]

    def forward(carry, entries):
        cp_prev, dp_prev = carry
        ai, bi, ci, di = entries
        denom = bi - cp_prev * ai
        cp_i = ci / denom
        dp_i = (di - dp_prev * ai) / denom
        return (cp_i, dp_i), (cp_i, dp_i)

    (_, _), (cp_tail, dp_tail) = jax.lax.scan(
        forward,
        (cp0, dp0),
        (a[..., 1:].swapaxes(0, -1), b[..., 1:].swapaxes(0, -1), c[..., 1:].swapaxes(0, -1), d[..., 1:].swapaxes(0, -1)),
        unroll=False,
    )
    cp = jnp.concatenate((cp0[..., None], cp_tail.swapaxes(0, -1)), axis=-1)
    dp = jnp.concatenate((dp0[..., None], dp_tail.swapaxes(0, -1)), axis=-1)

    x_last = dp[..., -1]

    def backward(x_next, entries):
        cp_i, dp_i = entries
        x_i = dp_i - cp_i * x_next
        return x_i, x_i

    _, x_rev_tail = jax.lax.scan(
        backward,
        x_last,
        (cp[..., :-1].swapaxes(0, -1)[::-1], dp[..., :-1].swapaxes(0, -1)[::-1]),
        unroll=False,
    )
    return jnp.concatenate((x_rev_tail[::-1].swapaxes(0, -1), x_last[..., None]), axis=-1)
