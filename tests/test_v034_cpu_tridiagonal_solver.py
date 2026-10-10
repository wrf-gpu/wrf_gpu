"""v0.3.4 CPU tridiagonal-solver deadlock fix (lane o1-tridiag).

Regression gate for kernel findings E105/E119: jaxlib's **CPU** LAPACK
tridiagonal FFI (``lapack_*gtsv_ffi`` -> ``jax::TridiagonalSolver`` ->
``jax::ParallelBatchMap`` -> ``BlockingCounter::Wait``) deadlocks XLA:CPU's
affinity-sized intra-op pool when the thunk executor runs at least pool-size
independent ``tridiagonal_solve`` calls concurrently (each holds a pool thread
and then blocks for batch sub-tasks queued on the same exhausted pool).

The two production users were
``gpuwrf.physics.tridiagonal_solver.solve_tridiagonal`` (MYNN/BouLac) and
``gpuwrf.dynamics.vertical_implicit_solver.solve_tridiagonal_xla``. They now
route the solve through a pure-JAX Thomas scan **on the CPU backend only**; the
GPU lowering (cuSPARSE ``gtsv`` / Pallas REAL kernel) is untouched and
byte-identical.

Deletion sensitivity: :func:`test_cpu_lowering_avoids_lapack_gtsv_ffi` is red
whenever the CPU routing is removed, because the unfixed CPU lowering contains
the ``lapack_*gtsv_ffi`` custom call. The real Swiss d01 CPU step that
deadlocked before the fix is re-run (time-boxed) by
``proofs/o1_tridiag/run_repro.sh``.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.dynamics.vertical_implicit_solver import (
    build_epssm_column_coefficients,
    solve_tridiagonal_xla,
)
from gpuwrf.physics.tridiagonal_solver import solve_tridiagonal


@contextmanager
def _x64(enabled: bool):
    """Enable/restore jax_enable_x64 around a block (process-global flag)."""

    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", enabled)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old)


def _stripped_mlir(lowered) -> str:
    return str(lowered.as_text())


def _fresh(fn):
    """A new callable object so JAX's per-function trace cache cannot go stale.

    ``jax.jit(fn).trace(...)`` memoises its trace on the *function object*
    (jax._src.linear_util cache), so the same ``fn`` traced once on CPU and once
    under a flipped ``jax.default_backend`` would silently return the first
    trace. AGENTS.md section 4: tests that flip module globals must clear that
    function's JIT cache.
    """

    return lambda *args: fn(*args)


def _cpu_text(fn, specs) -> str:
    jax.clear_caches()
    return _stripped_mlir(jax.jit(_fresh(fn)).trace(*specs).lower())


def _cuda_text(fn, specs) -> str:
    with mock.patch.object(jax, "default_backend", lambda *a, **k: "cuda"):
        jax.clear_caches()
        return _stripped_mlir(
            jax.jit(_fresh(fn)).trace(*specs).lower(lowering_platforms=("cuda",))
        )


def _specs(shape, dtype, *, multi_rhs=0):
    """Specs for (a, b, c, d); ``multi_rhs>0`` stacks that many RHS in ``d``."""

    d_shape = shape + (multi_rhs,) if multi_rhs else shape
    return [
        jax.ShapeDtypeStruct(shape, dtype),
        jax.ShapeDtypeStruct(shape, dtype),
        jax.ShapeDtypeStruct(shape, dtype),
        jax.ShapeDtypeStruct(d_shape, dtype),
    ]


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_cpu_lowering_avoids_lapack_gtsv_ffi(dtype):
    # Deletion-sensitive: the pre-fix CPU lowering carried `lapack_<s|d>gtsv_ffi`.
    with _x64(dtype == jnp.float64):
        text = _cpu_text(solve_tridiagonal, _specs((6, 9), dtype))
    assert "gtsv" not in text, text
    assert "tridiagonal_solve" not in text, text
    # the pure-JAX Thomas recurrence lowers to a while loop (forward + backward scan)
    assert "stablehlo.while" in text


@pytest.mark.parametrize(
    "shape,multi_rhs",
    [((6, 9), 0), ((2, 3, 9), 0), ((6, 9), 2)],
    ids=["batch2d", "batch3d", "multi_rhs"],
)
def test_cpu_and_gpu_lowering_diverge_only_on_cpu(shape, multi_rhs):
    dtype = jnp.float32
    cpu = _cpu_text(solve_tridiagonal, _specs(shape, dtype, multi_rhs=multi_rhs))
    gpu = _cuda_text(solve_tridiagonal, _specs(shape, dtype, multi_rhs=multi_rhs))
    assert "gtsv" not in cpu
    assert "gtsv" in gpu, gpu  # OFF-path: the GPU keep-the-primitive guarantee


def test_gpu_backend_keeps_xla_primitive_in_jaxpr():
    dtype = jnp.float32
    specs = _specs((6, 9), dtype)
    jax.clear_caches()
    with mock.patch.object(jax, "default_backend", lambda *a, **k: "cuda"):
        jaxpr = str(jax.make_jaxpr(lambda a, b, c, d: solve_tridiagonal(a, b, c, d))(*specs))
    assert "tridiagonal_solve" in jaxpr
    # ... and the CPU backend does not lower it to the primitive
    cpu_jaxpr = str(jax.make_jaxpr(lambda a, b, c, d: solve_tridiagonal(a, b, c, d))(*specs))
    assert "tridiagonal_solve" not in cpu_jaxpr


def test_vertical_implicit_xla_helper_is_cpu_safe():
    dtype = jnp.float64
    with _x64(True):
        text = _cpu_text(solve_tridiagonal_xla, _specs((9, 6), dtype))
    assert "gtsv" not in text
    # and still uses the GPU primitive when the active backend is CUDA
    with _x64(True):
        gpu = _cuda_text(solve_tridiagonal_xla, _specs((9, 6), dtype))
    assert "gtsv" in gpu, gpu


def _dense_reference(a, b, c, d):
    """Dense reference for a batched (leading..., n) tridiagonal system."""

    a, b, c, d = (np.asarray(x, dtype=np.float64) for x in (a, b, c, d))
    lead = b.shape[:-1]
    n = b.shape[-1]
    flat = int(np.prod(lead)) if lead else 1
    a, b, c, d = (x.reshape(flat, n) for x in (a, b, c, d))
    out = np.empty_like(d)
    for k in range(flat):
        m = np.diag(b[k]) + np.diag(c[k][:-1], 1) + np.diag(a[k][1:], -1)
        out[k] = np.linalg.solve(m, d[k])
    return out.reshape(lead + (n,))


def _random_system(rng, shape, *, diagonally_dominant=True):
    n = shape[-1]
    lead = shape[:-1]
    flat = int(np.prod(lead)) if lead else 1
    b = rng.uniform(2.0, 3.0, (flat, n))
    a = rng.uniform(-0.4, 0.4, (flat, n))
    c = rng.uniform(-0.4, 0.4, (flat, n))
    a[:, 0] = 0.0
    c[:, -1] = 0.0
    if diagonally_dominant:
        b[:, 0] = b[:, -1] = 1.0
        a[:, 1] = 0.0
        c[:, -2] = 0.0
    d = rng.uniform(-1.0, 1.0, (flat, n))
    reshape = lambda x: x.reshape(shape)
    return reshape(a), reshape(b), reshape(c), reshape(d)


@pytest.mark.parametrize("shape", [(7,), (5, 7), (2, 3, 7)])
def test_cpu_thomas_matches_dense_reference(shape):
    rng = np.random.default_rng(20261010)
    with _x64(True):
        for dominant in (True, False):
            a, b, c, d = _random_system(rng, shape, diagonally_dominant=dominant)
            got = np.asarray(
                solve_tridiagonal(
                    jnp.asarray(a), jnp.asarray(b), jnp.asarray(c), jnp.asarray(d)
                )
            )
            want = _dense_reference(a, b, c, d)
            assert np.allclose(got, want, rtol=1e-12, atol=1e-12), np.abs(got - want).max()


def test_cpu_multi_rhs_matches_dense_reference():
    rng = np.random.default_rng(7)
    a, b, c, d = _random_system(rng, (4, 7))
    rhs = rng.uniform(-1.0, 1.0, (4, 7, 3))
    with _x64(True):
        got = np.asarray(
            solve_tridiagonal(
                jnp.asarray(a), jnp.asarray(b), jnp.asarray(c), jnp.asarray(rhs)
            )
        )
    want = np.stack(
        [_dense_reference(a, b, c, rhs[..., k]) for k in range(rhs.shape[-1])], axis=-1
    )
    assert got.shape == rhs.shape
    assert np.allclose(got, want, rtol=1e-12, atol=1e-12)


def test_cpu_solver_matches_lapack_primitive_small():
    """Parity with the (unfixable on CPU) LAPACK path on a small single call."""

    rng = np.random.default_rng(11)
    a, b, c, d = _random_system(rng, (4, 7))
    with _x64(True):
        thomas = np.asarray(
            solve_tridiagonal(
                jnp.asarray(a), jnp.asarray(b), jnp.asarray(c), jnp.asarray(d)
            )
        )
        lapack = np.asarray(
            jax.lax.linalg.tridiagonal_solve(
                jnp.asarray(a), jnp.asarray(b), jnp.asarray(c), jnp.asarray(d)[..., None]
            )[..., 0]
        )
    assert np.allclose(thomas, lapack, rtol=1e-12, atol=1e-12)


def test_real_structured_acoustic_system_matches_dense():
    """A real WRF-family (MPAS/WRF epssm) acoustic column, not a toy diagonal."""

    nz = 44
    nx, ny = 3, 4
    eta = np.linspace(1.0, 0.0, nz + 1)
    dz = np.linspace(250.0, 900.0, nz)[:, None, None] * np.ones((nz, nx, ny))
    theta = (300.0 + 40.0 * eta[:-1])[:, None, None] * np.ones((nz, nx, ny))
    with _x64(True):
        _, _, _, _, _, _, a, b, c = build_epssm_column_coefficients(
            jnp.asarray(theta), jnp.asarray(dz), dt=54.0, epssm=0.1
        )
        rng = np.random.default_rng(3)
        rhs = jnp.asarray(rng.uniform(-1.0, 1.0, (nz + 1, nx, ny)))
        got = np.asarray(solve_tridiagonal_xla(a, b, c, rhs))

    got_f = np.moveaxis(got, 0, -1).reshape(-1, nz + 1)
    aa = np.moveaxis(np.asarray(a), 0, -1).reshape(-1, nz + 1)
    bb = np.moveaxis(np.asarray(b), 0, -1).reshape(-1, nz + 1)
    cc = np.moveaxis(np.asarray(c), 0, -1).reshape(-1, nz + 1)
    dd = np.moveaxis(np.asarray(rhs), 0, -1).reshape(-1, nz + 1)
    want = np.empty_like(dd)
    for k in range(dd.shape[0]):
        m = np.diag(bb[k]) + np.diag(cc[k][:-1], 1) + np.diag(aa[k][1:], -1)
        want[k] = np.linalg.solve(m, dd[k])
    assert np.isfinite(got_f).all()
    assert np.allclose(got_f, want, rtol=1e-10, atol=1e-10)
