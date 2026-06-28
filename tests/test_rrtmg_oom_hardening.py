from __future__ import annotations

import importlib
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")


def test_rrtmg_default_column_caps_are_tightened():
    import gpuwrf.physics.rrtmg_lw as lw
    import gpuwrf.physics.rrtmg_sw as sw

    assert lw._LW_COLUMN_TILE_COLS == 1024
    assert sw._SW_COLUMN_TILE_COLS == 1024
    assert lw._effective_lw_column_tile_cols(144801) == 1024
    assert sw._effective_sw_column_tile_cols(144801) == 1024
    assert lw._effective_lw_column_tile_cols(512) == 512
    assert sw._effective_sw_column_tile_cols(512) == 512


def test_rrtmg_m9_column_cap_can_be_scoped_per_call():
    import gpuwrf.physics.rrtmg_lw as lw
    import gpuwrf.physics.rrtmg_sw as sw

    assert lw._LW_COLUMN_TILE_COLS == 1024
    assert sw._SW_COLUMN_TILE_COLS == 1024
    assert lw._effective_lw_column_tile_cols(144801, column_tile_cols=512) == 512
    assert sw._effective_sw_column_tile_cols(144801, column_tile_cols=512) == 512


def test_rrtmg_common_tile_cap_env_overrides(monkeypatch):
    monkeypatch.setenv("GPUWRF_RRTMG_COLUMN_TILE_COLS", "1536")
    import gpuwrf.physics.rrtmg_lw as lw
    import gpuwrf.physics.rrtmg_sw as sw

    lw = importlib.reload(lw)
    sw = importlib.reload(sw)
    try:
        assert lw._LW_COLUMN_TILE_COLS == 1536
        assert sw._SW_COLUMN_TILE_COLS == 1536
    finally:
        monkeypatch.delenv("GPUWRF_RRTMG_COLUMN_TILE_COLS", raising=False)
        importlib.reload(lw)
        importlib.reload(sw)
