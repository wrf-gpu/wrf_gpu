from __future__ import annotations

import math
from pathlib import Path

import yaml

from gpuwrf.physics import thompson_constants as c
from gpuwrf.physics.thompson_tables import TABLE_ASSET, THOMPSON_TABLES, asset_sha256


def test_thompson_constants_match_wrf_source_values():
    c.assert_finite_constants()
    assert c.T_0 == 273.15
    assert c.RHO_W == 1000.0
    assert c.RHO_I == 890.0
    assert c.NT_C == 100.0e6
    assert c.R1 == 1.0e-12
    assert c.R2 == 1.0e-6
    assert c.HGFR == 235.16
    assert c.RV == 461.5
    assert c.R_D == 287.04
    assert c.CP == 1004.0
    assert c.LSUB == 2.834e6
    assert c.LVAP0 == 2.5e6
    assert c.LFUS == c.LSUB - c.LVAP0
    assert math.isclose(c.D0I, (c.XM0I / c.AM_I) ** (1.0 / 3.0), rel_tol=0.0, abs_tol=1.0e-18)
    assert c.NU_C_MP8 == 12.0
    assert c.CRG3 == 6.0
    assert c.CRE9 == 4.0
    assert c.T1_QR_EV == 0.78
    assert c.T2_QR_EV > 0.0


def test_thompson_derived_constants_match_source_formulas():
    assert math.isclose(c.CIE2, c.BM_I + c.MU_I + 1.0, rel_tol=0.0, abs_tol=1.0e-15)
    assert math.isclose(c.CGE11, 0.5 * (c.BV_G_MP8 + 5.0 + 2.0 * c.MU_G_MP8), rel_tol=0.0, abs_tol=1.0e-15)
    assert math.isclose(c.CGG11, math.gamma(c.CGE11), rel_tol=1.0e-7)


def test_thompson_table_asset_is_pinned_in_manifest():
    manifest = yaml.safe_load(Path("fixtures/manifests/analytic-thompson-column-v1.yaml").read_text(encoding="utf-8"))
    entries = [entry for entry in manifest["files"] if entry["path"] == "data/fixtures/thompson-tables-v1.npz"]
    assert entries
    assert TABLE_ASSET.exists()
    assert entries[0]["checksum_sha256"] == asset_sha256()


def test_runtime_tables_have_expected_wrf_shapes():
    assert THOMPSON_TABLES.t_Efrw.shape == (100, 100)
    assert THOMPSON_TABLES.iaus.shape == (64, 55, 3)
    assert THOMPSON_TABLES.snow_sa.shape == (10,)
