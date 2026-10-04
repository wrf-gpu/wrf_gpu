"""Independent WRF per-g-point flux gate for the fused fp32 transfer sweep."""

import jax
import jax.numpy as jnp
import numpy as np
import os

from gpuwrf.kernels.rad_lw_transfer import lw_band_fluxes
from gpuwrf.physics import rrtmg_lw as lw
from gpuwrf.physics.rrtmg_tables import RRTMG_TABLES
from gpuwrf.validation.tier1_rrtmg import load_lw_fixture_state
from gpuwrf.validation.rrtmg_intermediate_oracles import ORACLE, validate_lw_rtrnmc_per_gpoint_flux


def test_fused_lw_transfer_wrf_fluxes():
    initial, _ = load_lw_fixture_state()
    state, coef, secdiff, plank, plevel, psurf, cldf, taucld, _, _ = lw._lw_solver_base(
        initial, RRTMG_TABLES, build_taumol=False)
    cloud = jnp.any(cldf > .5, axis=(-1, -2))
    native = lw._native_lw_tables()
    with np.load(ORACLE) as f:
        truth_dn = f['lw_rtrnmc_zfd_per_gpoint']
        truth_up = f['lw_rtrnmc_zfu_per_gpoint']
    interpret = os.environ.get('B_RAD_PALLAS_INTERPRET', '1') == '1'
    if not interpret:
        assert jax.devices()[0].platform == 'gpu', 'Device oracle must execute on GPU'
    for band in range(16):
        tau, frac = lw._lw_taumol_band(band, coef, native, RRTMG_TABLES)
        args = (state, tau, frac, cldf[..., band, :], taucld[..., band, :],
                secdiff[..., band], RRTMG_TABLES.lw_delwave[band] * jnp.pi * 1e4,
                RRTMG_TABLES.lw_gpoint_mask[band], plank[..., band],
                plevel[..., band], psurf[..., band], cloud, True)
        candidate = jax.jit(lambda *a: lw_band_fluxes(*a, interpret=interpret), static_argnums=12)(*args)
        dn, up, cd, cu, tfn = candidate
        n = lw._LW_GPOINT_COUNTS[band]
        result = validate_lw_rtrnmc_per_gpoint_flux(
            dn[..., :n], up[..., :n], truth_dn[..., :n, band], truth_up[..., :n, band], band + 1)
        assert result['pass'], result
        assert all(np.isfinite(np.asarray(v)).all() for v in (dn, up, cd, cu, tfn))


def test_fused_lw_full_wrf_oracle(monkeypatch, tmp_path):
    from gpuwrf.kernels import rad_lw_transfer
    from gpuwrf.validation.tier1_rrtmg import run_tier1_lw
    from gpuwrf.validation.rrtmg_intermediate_oracles import run_intermediate_validation

    interpret = os.environ.get('B_RAD_PALLAS_INTERPRET', '1') == '1'
    if not interpret:
        assert jax.devices()[0].platform == 'gpu'
    monkeypatch.setattr(lw, '_FUSED_TRANSFER', True)
    monkeypatch.setattr(rad_lw_transfer, 'lw_band_fluxes',
        lambda *a: lw_band_fluxes(*a, interpret=interpret))
    lw.solve_rrtmg_lw_column.clear_cache()
    try:
        assert run_tier1_lw(tmp_path / 'lw_fused.json')['pass']
        if interpret:
            record = run_intermediate_validation(tmp_path / 'fused_intermediate.json', tmp_path / 'fused_bands.json')
            assert record['pass'], record
    finally:
        lw.solve_rrtmg_lw_column.clear_cache()
