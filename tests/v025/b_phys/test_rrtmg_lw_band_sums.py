"""BP57: LW band transfer with in-kernel cldprmc optics and g-summed fluxes.

Same WRF ``rtrnmc`` recurrence as the fused per-g-point kernel; the cloud
optical depth is formed per g-point from the global McICA mask and in-cloud
paths, and the g-point sum moves into the kernel (summation order only).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.kernels import rad_lw_transfer as T
from gpuwrf.physics import rrtmg_lw as lw
from gpuwrf.physics.rrtmg_tables import RRTMG_TABLES
from gpuwrf.validation.tier1_rrtmg import load_lw_fixture_state


def test_band_sums_match_per_gpoint_kernel():
    initial, _ = load_lw_fixture_state()
    base = lw._lw_solver_base(initial, RRTMG_TABLES, build_taumol=False)
    state, coef, secdiff, plank, plevel, psurf, cldf, taucld, _, _ = base
    glob = lw._lw_solver_base(initial, RRTMG_TABLES, build_taumol=False, cloud_global=True)
    cldf_global, clw, ciw, csw = glob[6]
    assert glob[7] is None
    cloud = jnp.any(cldf > .5, axis=(-1, -2))
    np.testing.assert_array_equal(cloud, jnp.any(cldf_global > .5, axis=-1))
    assert bool(jnp.any(cldf_global > .5))                     # the fixture has cloudy g-points
    native = lw._native_lw_tables()
    for band in range(16):
        tau, frac = lw._lw_taumol_band(band, coef, native, RRTMG_TABLES)
        scale = RRTMG_TABLES.lw_delwave[band] * jnp.pi * 1e4
        common = (secdiff[..., band], scale, RRTMG_TABLES.lw_gpoint_mask[band], plank[..., band],
                  plevel[..., band], psurf[..., band], cloud, True)
        ref = jax.jit(lambda *a: T.lw_band_fluxes(*a, interpret=True), static_argnums=12)(
            state, tau, frac, cldf[..., band, :], taucld[..., band, :], *common)
        got = jax.jit(lambda *a: T.lw_band_flux_sums(*a, gpoint_counts=lw._LW_GPOINT_COUNTS,
                                                    cloud=lw._native_lw_cloud_tables(),
                                                    interpret=True), static_argnums=12)(
            state, tau, frac, T.pack_lw_cloud(cldf_global > .5, clw, ciw, csw), jnp.int32(band), *common)
        for g, r in zip(got, ref[:4]):
            expect = np.asarray(jnp.sum(r, axis=-1))
            g = np.asarray(g)
            assert g.shape == expect.shape and g.dtype == expect.dtype
            scale_v = max(float(np.max(np.abs(expect))), 1e-30)
            assert float(np.max(np.abs(g - expect))) <= 4e-6 * scale_v, (band, float(np.max(np.abs(g - expect))), scale_v)


import pytest


@pytest.mark.parametrize("real_entry", [False, True])
def test_chunked_driver_band_sums_pass_wrf_tier1(monkeypatch, tmp_path, real_entry):
    from gpuwrf.physics import rrtmg_sw
    from gpuwrf.validation import tier1_rrtmg
    from gpuwrf.validation.tier1_rrtmg import run_tier1_lw

    calls = []
    real = T.lw_band_flux_sums
    monkeypatch.setattr(lw, "_FUSED_TRANSFER", True)
    monkeypatch.setattr(rrtmg_sw, "_FUSED_QUADRATURE", real_entry)   # both flags -> WRF-REAL entry
    assert lw._real_entry() is real_entry
    monkeypatch.setenv("GPUWRF_RRTMG_LW_BAND_SUMS", "1")
    monkeypatch.setattr(T, "lw_band_flux_sums", lambda *a, **k: calls.append(1) or real(*a, **{**k, "interpret": True}))
    # OOM tests reload lw, while this oracle retains its earlier imported jit.
    # Spy on the same fresh executable that the oracle will actually invoke.
    monkeypatch.setattr(tier1_rrtmg, "solve_rrtmg_lw_column", lw.solve_rrtmg_lw_column)
    lw.solve_rrtmg_lw_column.clear_cache()
    try:
        record = run_tier1_lw(tmp_path / "lw_band_sums.json")
        assert record["pass"], record
        assert calls                                               # the band-sum path ran (E114)
    finally:
        lw.solve_rrtmg_lw_column.clear_cache()
