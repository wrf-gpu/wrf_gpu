"""WRF MYNN ``tridiag2`` Thomas sweep as one Pallas column kernel (REAL).

Literal operation order of module_bl_mynnedmf.F:5350-5382 (forward cp/dp
elimination, backward substitution) with PTX RN32 operations: no FMA
contraction and IEEE division, like the other WRF-REAL kernels -- bitwise
equal to the REAL Fortran recurrence on the GPU. It replaces XLA's
tridiagonal_solve on the native MYNN REAL path (cuSPARSE pcrGtsv: two launches
per solve and a cyclic-reduction elimination order WRF does not use).
One lane per column, levels unrolled, operands read and written in the
callers' (columns, levels) layout (no level-major copy).
Flag GPUWRF_MYNN_TRIDIAG_REAL (default off), read at trace time.
"""
from functools import partial
import math
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pt

from gpuwrf.kernels.dyn_rk_fp32 import _rn

BLOCK = 128


def enabled():
    return os.environ.get("GPUWRF_MYNN_TRIDIAG_REAL", "0") == "1"


def _thomas(a, b, c, d, x, *, nz, n, interpret):
    """Lanes beyond ``n`` load the last column; their stores are masked off."""
    cols = pl.program_id(0) * BLOCK + jnp.arange(BLOCK, dtype=jnp.int32)
    valid = cols < n
    safe = jnp.minimum(cols, n - 1)
    rn = partial(_rn, interpret=interpret)
    # cp(1)=c(1)/b(1); dp(1)=d(1)/b(1)
    cp = [rn("div", c[safe, 0], b[safe, 0])]
    dp = [rn("div", d[safe, 0], b[safe, 0])]
    for k in range(1, nz):
        ak = a[safe, k]
        m = rn("sub", b[safe, k], rn("mul", cp[-1], ak))       # m = b(k)-cp(k-1)*a(k)
        cp.append(rn("div", c[safe, k], m))                    # cp(k) = c(k)/m
        dp.append(rn("div", rn("sub", d[safe, k], rn("mul", dp[-1], ak)), m))  # (d(k)-dp(k-1)*a(k))/m
    xk = dp[nz - 1]                                            # x(kte) = dp(kte)
    pt.store(x.at[cols, nz - 1], xk, mask=valid)
    for k in range(nz - 2, -1, -1):
        xk = rn("sub", dp[k], rn("mul", cp[k], xk))            # x(k) = dp(k)-cp(k)*x(k+1)
        pt.store(x.at[cols, k], xk, mask=valid)


def solve_tridiagonal_real(a, b, c, d):
    """Batched WRF ``tridiag2`` along the last axis; REAL inputs, any leading shape."""

    a, b, c, d = jnp.broadcast_arrays(*(jnp.asarray(v) for v in (a, b, c, d)))
    if any(v.dtype != jnp.float32 for v in (a, b, c, d)):
        raise TypeError("WRF tridiag2 kernel requires REAL operands")
    shape = d.shape
    nz = shape[-1]
    n = math.prod(shape[:-1])
    interpret = jax.default_backend() == "cpu"
    x = pl.pallas_call(
        partial(_thomas, nz=nz, n=n, interpret=interpret), grid=(-(-n // BLOCK),),
        out_shape=jax.ShapeDtypeStruct((n, nz), jnp.float32), interpret=interpret,
        compiler_params=pt.CompilerParams(num_warps=4), name="mynn_tridiag2_real",
    )(*(v.reshape(n, nz) for v in (a, b, c, d)))
    return x.reshape(shape)
