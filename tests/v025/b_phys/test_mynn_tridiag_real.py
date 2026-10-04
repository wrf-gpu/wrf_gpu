"""BP56: Pallas WRF ``tridiag2`` sweep vs a literal REAL4 transcription of WRF.

module_bl_mynnedmf.F:5350-5382. NumPy float32 = IEEE RN without contraction (the
WRF REAL build); the CPU interpreter may contract mul+sub into FMA, so the CPU
bound is a few ulp; the asserted-GPU runner checks bitwise (PTX RN32 ops).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.kernels import phys_mynn_tridiag as T


def wrf_tridiag2(a, b, c, d, dtype=np.float32):
    a, b, c, d = (np.asarray(v, dtype) for v in (a, b, c, d))
    nz = d.shape[-1]
    cp, dp, x = np.empty_like(d), np.empty_like(d), np.empty_like(d)
    cp[..., 0] = c[..., 0] / b[..., 0]
    dp[..., 0] = d[..., 0] / b[..., 0]
    for k in range(1, nz):
        m = b[..., k] - cp[..., k - 1] * a[..., k]
        cp[..., k] = c[..., k] / m
        dp[..., k] = (d[..., k] - dp[..., k - 1] * a[..., k]) / m
    x[..., -1] = dp[..., -1]
    for k in range(nz - 2, -1, -1):
        x[..., k] = dp[..., k] - cp[..., k] * x[..., k + 1]
    return x


def check_against_wrf_real(got, a, b, c, d):
    """Kernel is as accurate as WRF's literal REAL tridiag2 (vs the same recurrence in f64)
    and within REAL rounding of it (the CPU interpreter may contract mul+sub into FMA)."""
    literal = wrf_tridiag2(a, b, c, d)
    exact = wrf_tridiag2(a, b, c, d, np.float64)
    scale = np.max(np.abs(exact), axis=-1, keepdims=True) + 1e-37
    err_kernel = np.max(np.abs(got - exact) / scale)
    err_literal = np.max(np.abs(literal - exact) / scale)
    assert got.dtype == np.float32 and got.shape == literal.shape
    assert err_kernel <= 2.0 * err_literal + 64 * np.finfo(np.float32).eps, (err_kernel, err_literal)
    assert np.max(np.abs(got - literal) / scale) <= 4.0 * err_literal + 64 * np.finfo(np.float32).eps


def mynn_like(seed, shape=(3, 70, 44)):
    """Implicit-diffusion structure: a, c <= 0, b = 1 + |a| + |c| (+ mass-flux skew), top row identity."""
    rng = np.random.default_rng(seed)
    lo = -rng.uniform(0.0, 40.0, shape).astype(np.float32)
    up = -rng.uniform(0.0, 40.0, shape).astype(np.float32)
    skew = rng.uniform(-0.3, 0.3, shape).astype(np.float32)
    lo[..., 0] = 0.0
    a, c = lo + skew, up - skew
    b = (np.float32(1.0) - lo - up).astype(np.float32)
    a[..., -1] = c[..., -1] = 0.0
    b[..., -1] = 1.0
    d = rng.normal(0.0, 1.0, shape).astype(np.float32) * np.float32(10.0) ** rng.integers(-8, 3, shape[:-1] + (1,))
    return a, b, c, d.astype(np.float32)


@pytest.mark.parametrize("seed", range(3))
def test_kernel_matches_literal_wrf_tridiag2(seed):
    a, b, c, d = mynn_like(seed)
    got = jax.jit(T.solve_tridiagonal_real)(a, b, c, d)
    check_against_wrf_real(np.asarray(got), a, b, c, d)


def test_odd_batch_padding_and_leading_shapes():
    a, b, c, d = mynn_like(7, shape=(131, 44))   # 131 columns -> 2 blocks, 125 padding lanes
    check_against_wrf_real(np.asarray(T.solve_tridiagonal_real(a, b, c, d)), a, b, c, d)
    with pytest.raises(TypeError):
        T.solve_tridiagonal_real(*(v.astype(np.float64) for v in (a, b, c, d)))


def test_mynn_seam_dispatches_only_real_solves(monkeypatch):
    from gpuwrf.physics import mynn_pbl as P

    calls = []
    real = T.solve_tridiagonal_real
    monkeypatch.setattr(T, "solve_tridiagonal_real", lambda *args: calls.append(1) or real(*args))
    a, b, c, d = mynn_like(3, shape=(5, 44))
    monkeypatch.setenv("GPUWRF_MYNN_TRIDIAG_REAL", "0")
    off = np.asarray(P._solve_tridiagonal(a, b, c, d))
    assert not calls
    monkeypatch.setenv("GPUWRF_MYNN_TRIDIAG_REAL", "1")
    on = np.asarray(P._solve_tridiagonal(a, b, c, d))
    assert calls == [1]
    P._solve_tridiagonal(*(jnp.asarray(v, jnp.float64) for v in (a, b, c, d)))  # legacy fp64 stays on XLA
    assert calls == [1]
    np.testing.assert_allclose(on, off, rtol=1e-4, atol=1e-30)   # cuSPARSE/XLA order vs WRF order
