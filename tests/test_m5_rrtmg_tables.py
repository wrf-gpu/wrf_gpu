from __future__ import annotations

from pathlib import Path

import numpy as np

from gpuwrf.physics.rrtmg_constants import WRF_RRTMG_LW_BANDS, WRF_RRTMG_SW_BANDS
from gpuwrf.physics.rrtmg_tables import TABLE_ASSET, asset_sha256, load_rrtmg_tables


def test_rrtmg_table_asset_loads_with_wrf_band_counts():
    assert TABLE_ASSET.exists()
    assert TABLE_ASSET.stat().st_size > 1_000_000
    tables = load_rrtmg_tables()
    assert tables.sw_band_weights.shape == (WRF_RRTMG_SW_BANDS,)
    assert tables.lw_band_weights.shape == (WRF_RRTMG_LW_BANDS,)
    assert tables.sw_absorption_coefficients.shape[:2] == (WRF_RRTMG_SW_BANDS, 59)
    assert tables.sw_gpoint_mask.shape[0] == WRF_RRTMG_SW_BANDS
    assert tables.lw_absorption_coefficients.shape[0] == WRF_RRTMG_LW_BANDS
    assert tables.lw_gpoint_mask.shape[0] == WRF_RRTMG_LW_BANDS
    assert np.isclose(np.asarray(tables.sw_band_weights).sum(), 1.0)
    assert np.isclose(np.asarray(tables.lw_band_weights).sum(), 1.0)
    assert len(asset_sha256(Path(TABLE_ASSET))) == 64
    with np.load(TABLE_ASSET, allow_pickle=False) as loaded:
        assert loaded["sw_raw_payload_bytes"].size > 600_000
        assert loaded["lw_raw_payload_bytes"].size > 800_000
        sw_active = loaded["sw_absorption_coefficients"][loaded["sw_absorption_coefficients"] > 0.0]
        lw_mask = np.broadcast_to(loaded["lw_gpoint_mask"].astype(bool)[:, None, :], loaded["lw_absorption_coefficients"].shape)
        lw_active = loaded["lw_absorption_coefficients"][lw_mask]
        assert np.mean(np.isclose(sw_active, 0.0025)) < 0.2
        assert np.mean(np.isclose(lw_active, 0.003)) < 0.2
