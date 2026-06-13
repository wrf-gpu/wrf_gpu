"""v0.15 kernel-final — MP column tiling (Task 3) CPU regression tests.

The BINDING identity gate is the GPU exact-output check
(proofs/perf/v015/km_bench/mp_tiling_identity.json): on GPU the tiled result
is bit-identical.  On CPU, batch-width SIMD codegen may differ by ~1 ulp (the
same documented caveat as MYNN's `_tiled_mynn_step`), so these tests assert
structure + tight value agreement, plus the activation contract (production
single-tile batches keep the untiled path).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest

from gpuwrf.physics import column_tiling as ct
from gpuwrf.physics import thompson_column as tc


def _make_state(shape, nz: int, seed: int = 0) -> tc.ThompsonColumnState:
    """Build a Thompson column state with horizontal ``shape`` (an int for a
    flat ``(ncol, nz)`` 2-D batch, or a tuple ``(ny, nx)`` for the PRODUCTION
    ``(ny, nx, nz)`` layout that `_thompson_column_from_state` actually emits)."""

    rng = np.random.default_rng(seed)
    lead = (shape,) if isinstance(shape, int) else tuple(shape)

    def mk(scale, base=0.0):
        return jnp.asarray(base + scale * rng.random(lead + (nz,)))

    return tc.ThompsonColumnState(
        qv=mk(8e-3, 1e-4), qc=mk(5e-4), qr=mk(2e-4), qi=mk(1e-4),
        qs=mk(3e-4), qg=mk(1e-4), Ni=mk(1e5), Nr=mk(1e5), Ns=mk(5e4), Ng=mk(2e4),
        T=mk(40.0, 250.0), p=mk(4e4, 5e4), rho=mk(0.6, 0.5),
        dz=mk(200.0, 100.0), w=mk(2.0, -1.0),
    )


@pytest.fixture()
def small_tiles(monkeypatch):
    monkeypatch.setattr(tc, "_MP_COLUMN_TILING", True)
    monkeypatch.setattr(tc, "_MP_COLUMN_TILE_COLS", 7)


def _assert_close(a, b, label):
    """~ulp agreement: tight rtol for real magnitudes, atol for the femto-scale
    precip noise floor (CPU SIMD batch-width ulps; GPU is bit-identical)."""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape, label
    assert np.allclose(a, b, rtol=1e-9, atol=1e-12), label


def test_tiled_matches_untiled_with_pad(small_tiles):
    """23 cols over 7-col tiles (4 tiles, 5 pad columns): values agree to ~ulp."""

    state = _make_state(23, 12)
    out_t, pr_t = tc._maybe_tiled_thompson_full(state, 18.0, False)
    out_u, pr_u = tc._step_thompson_column_full_impl(state, 18.0, False)
    for name in tc.ThompsonColumnState.__slots__:
        _assert_close(getattr(out_t, name), getattr(out_u, name), name)
    for key in pr_u:
        _assert_close(pr_t[key], pr_u[key], key)


def test_tiled_matches_untiled_exact_multiple(small_tiles):
    """21 cols over 7-col tiles (3 exact tiles, no pad)."""

    state = _make_state(21, 12, seed=1)
    out_t, pr_t = tc._maybe_tiled_thompson_full(state, 18.0, False)
    out_u, pr_u = tc._step_thompson_column_full_impl(state, 18.0, False)
    for name in tc.ThompsonColumnState.__slots__:
        _assert_close(getattr(out_t, name), getattr(out_u, name), name)
    for key in pr_u:
        _assert_close(pr_t[key], pr_u[key], key)


def test_single_tile_batch_keeps_untiled_path(monkeypatch):
    """ncol <= tile_cols (the production 16384-col case) must NOT tile: the
    untiled graph stays byte-for-byte untouched (Tier-S argument)."""

    calls = {"tiled": 0}
    real = ct.tiled_column_apply

    def spy(*args, **kwargs):
        calls["tiled"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(ct, "tiled_column_apply", spy)
    monkeypatch.setattr(tc, "_MP_COLUMN_TILING", True)
    monkeypatch.setattr(tc, "_MP_COLUMN_TILE_COLS", 64)
    state = _make_state(64, 12, seed=2)
    out_a, pr_a = tc._maybe_tiled_thompson_full(state, 18.0, False)
    out_b, pr_b = tc._step_thompson_column_full_impl(state, 18.0, False)
    assert calls["tiled"] == 0
    for name in tc.ThompsonColumnState.__slots__:
        assert np.asarray(getattr(out_a, name)).tobytes() == np.asarray(getattr(out_b, name)).tobytes(), name
    for key in pr_b:
        assert np.asarray(pr_a[key]).tobytes() == np.asarray(pr_b[key]).tobytes(), key


def test_tiling_disabled_by_knob(monkeypatch):
    monkeypatch.setattr(tc, "_MP_COLUMN_TILING", False)
    monkeypatch.setattr(tc, "_MP_COLUMN_TILE_COLS", 7)
    state = _make_state(23, 12, seed=3)
    out_a, pr_a = tc._maybe_tiled_thompson_full(state, 18.0, False)
    out_b, pr_b = tc._step_thompson_column_full_impl(state, 18.0, False)
    for name in tc.ThompsonColumnState.__slots__:
        assert np.asarray(getattr(out_a, name)).tobytes() == np.asarray(getattr(out_b, name)).tobytes(), name
    for key in pr_b:
        assert np.asarray(pr_a[key]).tobytes() == np.asarray(pr_b[key]).tobytes(), key


def test_tiled_matches_untiled_3d_production_layout(small_tiles):
    """PRODUCTION layout regression: the column view emitted by
    `_thompson_column_from_state` is ``(ny, nx, nz)`` 3-D, NOT a flat
    ``(ncol, nz)`` 2-D batch.  Tiling must flatten the leading (ny, nx) axes,
    scan over column tiles, and reshape back — value-identical per column AND
    shape-preserving.  Before the v0.15 ship-gate fix, the activation gate was
    ``ndim == 2`` so tiling NEVER engaged on this real shape (the VRAM cap was
    silently inert on every production forecast)."""

    ny, nx, nz = 5, 5, 12  # 25 cols over 7-col tiles -> 4 tiles, 3 pad columns
    state = _make_state((ny, nx), nz, seed=7)
    out_t, pr_t = tc._maybe_tiled_thompson_full(state, 18.0, False)
    out_u, pr_u = tc._step_thompson_column_full_impl(state, 18.0, False)
    for name in tc.ThompsonColumnState.__slots__:
        a = np.asarray(getattr(out_t, name))
        assert a.shape == (ny, nx, nz), f"{name} lost production (ny,nx,nz) shape: {a.shape}"
        _assert_close(getattr(out_t, name), getattr(out_u, name), name)
    for key in pr_u:
        a = np.asarray(pr_t[key])
        assert a.shape == (ny, nx), f"precip {key} lost (ny,nx) surface shape: {a.shape}"
        _assert_close(pr_t[key], pr_u[key], key)


def test_3d_single_tile_keeps_untiled_path(monkeypatch):
    """The production 128x128 = 16384-col grid is exactly one 16384-col tile and
    must NOT tile (Tier-S: untiled graph byte-for-byte untouched)."""

    calls = {"tiled": 0}
    real = ct.tiled_column_apply

    def spy(*args, **kwargs):
        calls["tiled"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(ct, "tiled_column_apply", spy)
    monkeypatch.setattr(tc, "_MP_COLUMN_TILING", True)
    monkeypatch.setattr(tc, "_MP_COLUMN_TILE_COLS", 64)
    state = _make_state((8, 8), 12, seed=9)  # 64 cols == tile_cols -> no tiling
    out_a, pr_a = tc._maybe_tiled_thompson_full(state, 18.0, False)
    out_b, pr_b = tc._step_thompson_column_full_impl(state, 18.0, False)
    assert calls["tiled"] == 0
    for name in tc.ThompsonColumnState.__slots__:
        assert np.asarray(getattr(out_a, name)).tobytes() == np.asarray(getattr(out_b, name)).tobytes(), name
    for key in pr_b:
        assert np.asarray(pr_a[key]).tobytes() == np.asarray(pr_b[key]).tobytes(), key


def test_pad_and_slice_leaf_contracts():
    arr = jnp.arange(10.0).reshape(5, 2)
    padded = ct.pad_columns_leaf(arr, 5, 7)
    assert padded.shape == (7, 2)
    assert np.array_equal(np.asarray(padded[5]), np.asarray(arr[-1]))
    assert np.array_equal(np.asarray(padded[6]), np.asarray(arr[-1]))
    # non-column leaf passes through
    passthrough = ct.pad_columns_leaf(jnp.ones((3, 2)), 5, 7)
    assert passthrough.shape == (3, 2)
    tile = ct.slice_columns_leaf(padded, jnp.asarray(3, dtype=jnp.int32), 4, 7)
    assert tile.shape == (4, 2)
    assert np.array_equal(np.asarray(tile[0]), np.asarray(padded[3]))
