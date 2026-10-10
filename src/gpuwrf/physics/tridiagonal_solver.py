"""Reusable JAX tridiagonal solver for vertical implicit column updates."""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

import jax
from jax import config
import jax.numpy as jnp


configure_jax_x64()


def _cpu_thomas_fallback() -> bool:
    """True when the active JAX backend is CPU (resolved at trace time).

    jaxlib's CPU LAPACK tridiagonal FFI (``lapack_*gtsv_ffi`` ->
    ``jax::TridiagonalSolver`` -> ``jax::ParallelBatchMap`` ->
    ``BlockingCounter::Wait``) deadlocks XLA:CPU's intra-op pool when the thunk
    executor runs at least pool-size independent ``tridiagonal_solve`` calls
    concurrently: each occupies a pool thread and then blocks waiting for
    batch sub-tasks queued on the same, now-exhausted pool (kernel finding
    E105/E119, eu-stack of a real Swiss d01 step). The CPU pool is sized from
    the process CPU affinity, so pinning fewer cores makes it *more* likely.

    The GPU path (cuSPARSE FFI / Pallas REAL kernel) is unaffected and must stay
    byte-identical, so the pure-JAX fallback is selected *only* on CPU. The
    backend is resolved at trace time (a Python-level branch: a ``jax.jit`` is
    traced -- and therefore made backend-specific -- before lowering), never at
    import time.
    """

    return jax.default_backend() == "cpu"


def solve_tridiagonal(a, b, c, d):
    """Solves a batched tridiagonal system along the last axis.

    On the GPU the production path uses XLA's tridiagonal primitive (cuSPARSE
    ``gtsv``) so the MYNN column keeps the M5-S2 launch budget, and it is
    **byte-identical to the historical behaviour**. On the CPU backend that
    primitive routes to jaxlib's LAPACK FFI, which deadlocks under the pinned
    one/two-core CPU harnesses (see :func:`_cpu_thomas_fallback`); there the
    solve is routed through the pure-JAX Thomas scan
    ``solve_tridiagonal_thomas_reference`` below, which mirrors WRF MYNN's
    ``tridiag2`` Thomas algorithm (module_bl_mynnedmf.F90:5318-5350) and is the
    same recurrence the GPU REAL kernel (``phys_mynn_tridiag``) implements.
    """

    a = jnp.asarray(a, dtype=d.dtype)
    b = jnp.asarray(b, dtype=d.dtype)
    c = jnp.asarray(c, dtype=d.dtype)
    d = jnp.asarray(d)
    if _cpu_thomas_fallback():
        return _solve_tridiagonal_cpu(a, b, c, d)
    if d.ndim == b.ndim:
        return jax.lax.linalg.tridiagonal_solve(a, b, c, d[..., None])[..., 0]
    return jax.lax.linalg.tridiagonal_solve(a, b, c, d)


def _solve_tridiagonal_cpu(a, b, c, d):
    """CPU-backend Thomas scan, same contract as XLA's tridiagonal primitive.

    ``d`` is either a single right-hand side (``d.ndim == b.ndim``) or a stack of
    right-hand sides (``d.ndim == b.ndim + 1``); the multi-RHS case is handled by
    mapping the single-RHS recurrence over the trailing axis.
    """

    if d.ndim == b.ndim:
        return solve_tridiagonal_thomas_reference(a, b, c, d)
    return jax.vmap(
        lambda rhs: solve_tridiagonal_thomas_reference(a, b, c, rhs),
        in_axes=-1,
        out_axes=-1,
    )(d)


def solve_tridiagonal_thomas_reference(a, b, c, d):
    """Thomas-recurrence tridiagonal solve along the last axis.

    Mirrors WRF MYNN's ``tridiag2`` (module_bl_mynnedmf.F90:5318-5350) and is
    used both as the independent recurrence reference for tests and as the
    CPU-backend production fallback for :func:`solve_tridiagonal`. Handles
    arbitrary leading batch dimensions (the level axis is moved to the front so
    one ``lax.scan`` runs the recurrence for every column at once).
    """

    a = jnp.asarray(a, dtype=d.dtype)
    b = jnp.asarray(b, dtype=d.dtype)
    c = jnp.asarray(c, dtype=d.dtype)
    d = jnp.asarray(d)

    a = jnp.moveaxis(a, -1, 0)
    b = jnp.moveaxis(b, -1, 0)
    c = jnp.moveaxis(c, -1, 0)
    d = jnp.moveaxis(d, -1, 0)

    cp0 = c[0] / b[0]
    dp0 = d[0] / b[0]

    def forward(carry, entries):
        cp_prev, dp_prev = carry
        ai, bi, ci, di = entries
        denom = bi - cp_prev * ai
        cp_i = ci / denom
        dp_i = (di - dp_prev * ai) / denom
        return (cp_i, dp_i), (cp_i, dp_i)

    (_, _), (cp_tail, dp_tail) = jax.lax.scan(
        forward, (cp0, dp0), (a[1:], b[1:], c[1:], d[1:]), unroll=False
    )
    cp = jnp.concatenate((cp0[None, ...], cp_tail), axis=0)
    dp = jnp.concatenate((dp0[None, ...], dp_tail), axis=0)

    x_last = dp[-1]

    def backward(x_next, entries):
        cp_i, dp_i = entries
        x_i = dp_i - cp_i * x_next
        return x_i, x_i

    _, x_rev = jax.lax.scan(
        backward, x_last, (cp[:-1][::-1], dp[:-1][::-1]), unroll=False
    )
    x = jnp.concatenate((x_rev[::-1], x_last[None, ...]), axis=0)
    return jnp.moveaxis(x, 0, -1)
