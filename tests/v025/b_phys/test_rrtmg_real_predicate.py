"""review-rrtmg32 A2: coupler REAL preparation and both solver REAL entries share one predicate."""
from __future__ import annotations

import itertools

import pytest

from gpuwrf.coupling import physics_couplers as C
from gpuwrf.physics import rrtmg_lw, rrtmg_sw


@pytest.mark.parametrize("lw,sw", list(itertools.product((False, True), repeat=2)))
def test_real_entry_requires_both_radiation_flags(monkeypatch, lw, sw):
    monkeypatch.setattr(rrtmg_lw, "_FUSED_TRANSFER", lw)
    monkeypatch.setattr(rrtmg_sw, "_FUSED_QUADRATURE", sw)
    expected = lw and sw
    assert rrtmg_lw._real_entry() is expected
    assert rrtmg_sw._real_entry() is expected
    assert bool(C._rrtmg_real_enabled()) is expected


@pytest.mark.parametrize("lw,sw,explicit,expected", [
    (True, True, False, 4096), (True, False, False, 1024), (False, False, False, 1024), (True, True, True, 1024),
])
def test_column_tile_default_follows_real_predicate(monkeypatch, lw, sw, explicit, expected):
    """BP49: 4096-column tiles only on the REAL radiation path; an explicit env tile wins."""

    monkeypatch.setattr(rrtmg_lw, "_FUSED_TRANSFER", lw)
    monkeypatch.setattr(rrtmg_sw, "_FUSED_QUADRATURE", sw)
    for module, prefix in ((rrtmg_lw, "_LW"), (rrtmg_sw, "_SW")):
        monkeypatch.setattr(module, f"{prefix}_COLUMN_TILE_COLS", 1024)
        monkeypatch.setattr(module, f"{prefix}_COLUMN_TILE_COLS_EXPLICIT", explicit)
    assert rrtmg_lw._effective_lw_column_tile_cols(31239) == expected
    assert rrtmg_sw._effective_sw_column_tile_cols(31239) == expected
    assert rrtmg_lw._effective_lw_column_tile_cols(31239, column_tile_cols=512) == 512
