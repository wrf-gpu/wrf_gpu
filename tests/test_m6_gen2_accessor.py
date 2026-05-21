from __future__ import annotations

from pathlib import Path

import numpy as np

from gpuwrf.io.gen2_accessor import Gen2Run, LazyNetCDFArray


RUN_PATH = Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260519_18z_l3_24h_20260520T025228Z")


def test_gen2_run_discovers_domains_grid_and_missing_d02_bdy():
    run = Gen2Run(RUN_PATH)

    assert run.domains == ["d01", "d02", "d03", "d04", "d05"]
    assert not (RUN_PATH / "wrfbdy_d02").exists()
    grid = run.grid("d02")
    assert grid.dx_m == 3000.0
    assert grid.dy_m == 3000.0
    assert grid.e_we == 160
    assert grid.e_sn == 67
    assert grid.e_vert == 45
    assert grid.mass_nx == 159
    assert grid.mass_ny == 66
    assert grid.mass_nz == 44
    assert grid.grid_proj == "lambert"
    assert grid.parent_id == 1
    assert grid.parent_grid_ratio == 3


def test_gen2_variables_and_lazy_load_are_read_only_device_cached():
    run = Gen2Run(RUN_PATH)

    variables = run.variables("d02")
    for name in ("U", "V", "T", "QVAPOR", "PH"):
        assert name in variables

    lazy = run.load("d02", "T", time=0, lazy=True)
    assert isinstance(lazy, LazyNetCDFArray)
    assert lazy.shape == (44, 66, 159)
    first = lazy.materialize()
    second = lazy.materialize()
    assert first is second
    assert tuple(first.shape) == (44, 66, 159)
    assert bool(np.isfinite(np.asarray(first)).all())


def test_gen2_manifest_shape_without_hashing(tmp_path):
    run = Gen2Run(RUN_PATH)
    manifest_path = tmp_path / "gen2_manifest.json"

    manifest = run.write_manifest(manifest_path, include_sha256=False)

    assert manifest["run_id"] == "20260519_18z_l3_24h_20260520T025228Z"
    assert manifest["path"] == str(RUN_PATH)
    assert manifest["no_write_audit"] is True
    assert {domain["id"] for domain in manifest["domains"]} == {"d01", "d02", "d03", "d04", "d05"}
    assert "d02" in manifest["variable_inventory"]
    assert "T" in manifest["variable_inventory"]["d02"]
    assert manifest_path.exists()
