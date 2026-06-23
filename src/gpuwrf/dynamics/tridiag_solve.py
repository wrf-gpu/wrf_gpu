"""WRF-shaped Thomas sweep helpers for savepoint parity comparisons."""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

import jax
from jax import config
import jax.numpy as jnp


configure_jax_x64()


def thomas_forward_scan(a: jax.Array, alpha: jax.Array, rhs: jax.Array) -> jax.Array:
    """Apply WRF ``advance_w`` forward sweep along the leading vertical axis.

    Source: WRF ``dyn_em/module_small_step_em.F:1533-1537``.
    """

    rhs = jnp.asarray(rhs)
    a = jnp.asarray(a, dtype=rhs.dtype)
    alpha = jnp.asarray(alpha, dtype=rhs.dtype)

    def step(prev_w, entries):
        a_k, alpha_k, rhs_k = entries
        w_k = (rhs_k - a_k * prev_w) * alpha_k
        return w_k, w_k

    _, tail = jax.lax.scan(step, rhs[0], (a[1:], alpha[1:], rhs[1:]), unroll=False)
    return jnp.concatenate((rhs[0][None, ...], tail), axis=0)


def thomas_back_scan(gamma: jax.Array, w_fwd: jax.Array) -> jax.Array:
    """Apply WRF ``advance_w`` back-substitution along the leading vertical axis.

    Source: WRF ``dyn_em/module_small_step_em.F:1546-1550``.
    """

    w_fwd = jnp.asarray(w_fwd)
    gamma = jnp.asarray(gamma, dtype=w_fwd.dtype)
    if w_fwd.shape[0] <= 2:
        return w_fwd

    def step(next_w, entries):
        gamma_k, w_k = entries
        solved = w_k - gamma_k * next_w
        return solved, solved

    _, interior_rev = jax.lax.scan(
        step,
        w_fwd[-1],
        (gamma[1:-1][::-1], w_fwd[1:-1][::-1]),
        unroll=False,
    )
    interior = interior_rev[::-1]
    return jnp.concatenate((w_fwd[0][None, ...], interior, w_fwd[-1][None, ...]), axis=0)


def thomas_solve_scan(a: jax.Array, alpha: jax.Array, gamma: jax.Array, rhs: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return WRF-shaped forward-sweep state and back-substituted solution."""

    fwd = thomas_forward_scan(a, alpha, rhs)
    return fwd, thomas_back_scan(gamma, fwd)
