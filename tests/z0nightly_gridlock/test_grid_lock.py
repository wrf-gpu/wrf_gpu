"""Grid lock for static per-grid surface-control files (host-only; synthetic wrfinput files)."""
import json

import numpy as np
import pytest

nc = pytest.importorskip("netCDF4")

from gpuwrf.surface_params import grid_lock as gl

DOMS = ("d03", "d04")
RULE = {"control": "z0bare", "class": 16}


def _wrfinput(path, seed):
    rng = np.random.default_rng(seed)
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("Time", 1); ds.createDimension("south_north", 4); ds.createDimension("west_east", 5)
        for k in gl.ATTRS:
            ds.setncattr(k, "MODIFIED_IGBP_MODIS_NOAH" if k == "MMINLU" else (np.int32(3) if k.isupper() and k in (
                "WEST-EAST_GRID_DIMENSION", "SOUTH-NORTH_GRID_DIMENSION", "MAP_PROJ", "GRID_ID", "PARENT_ID",
                "I_PARENT_START", "J_PARENT_START", "PARENT_GRID_RATIO", "NUM_LAND_CAT", "ISWATER", "ISLAKE",
                "ISICE", "ISURBAN", "ISOILWATER") else np.float32(1000.0)))
        for k in gl.ARRAYS:
            dt = "i4" if k == "IVGTYP" else "f4"
            v = ds.createVariable(k, dt, ("Time", "south_north", "west_east"))
            v[:] = rng.integers(1, 21, (1, 4, 5)) if k in ("IVGTYP", "LU_INDEX") else rng.random((1, 4, 5))


@pytest.fixture
def case(tmp_path):
    for i, d in enumerate(DOMS):
        _wrfinput(tmp_path / f"wrfinput_{d}", i)
    lock = gl.make_lock(tmp_path, DOMS, RULE)
    manifest = {"domains": {d: {} for d in DOMS}, "provenance": {gl.KEY: lock}}
    return tmp_path, manifest


def test_unlocked_file_is_unchanged_behaviour(tmp_path):
    assert gl.require_grid_lock({"domains": {}, "provenance": {}}, None, set()) is None


def test_same_grid_passes(case):
    path, manifest = case
    assert gl.require_grid_lock(manifest, path, {"d01", "d02", *DOMS})["status"] == "GRID_LOCK_PASS"


@pytest.mark.parametrize("field", ["LU_INDEX", "HGT", "XLAT", "LANDMASK"])
def test_any_grid_field_change_refuses(case, field):
    path, manifest = case
    with nc.Dataset(path / "wrfinput_d04", "a") as ds:
        ds[field][0, 1, 1] = ds[field][0, 1, 1] + 1
    with pytest.raises(gl.GridLockError, match=f"d04 grid differs.*{field}"):
        gl.require_grid_lock(manifest, path, {"d01", "d02", *DOMS})


def test_attribute_change_refuses(case):
    path, manifest = case
    with nc.Dataset(path / "wrfinput_d03", "a") as ds:
        ds.setncattr("DX", np.float32(999.0))
    with pytest.raises(gl.GridLockError, match="DX"):
        gl.require_grid_lock(manifest, path, {"d01", "d02", *DOMS})


def test_locked_domain_missing_from_run_refuses_no_silent_off(case):
    path, manifest = case
    with pytest.raises(gl.GridLockError, match="not in this run"):
        gl.require_grid_lock(manifest, path, {"d01", "d02", "d03"})


def test_locked_domain_missing_from_file_refuses(case):
    path, manifest = case
    manifest = {**manifest, "domains": {"d03": {}}}
    with pytest.raises(gl.GridLockError, match="without controls"):
        gl.require_grid_lock(manifest, path, {"d01", "d02", *DOMS})


def test_altered_lock_refuses(case):
    path, manifest = case
    lock = json.loads(json.dumps(manifest["provenance"][gl.KEY]))
    lock["rule"]["class"] = 7
    with pytest.raises(gl.GridLockError, match="digest mismatch"):
        gl.require_grid_lock({**manifest, "provenance": {gl.KEY: lock}}, path, {"d01", "d02", *DOMS})


def test_locked_file_without_input_dir_refuses(case):
    _path, manifest = case
    with pytest.raises(gl.GridLockError, match="input-dir"):
        gl.require_grid_lock(manifest, None, {"d01", "d02", *DOMS})


def test_missing_wrfinput_refuses(case):
    path, manifest = case
    (path / "wrfinput_d04").unlink()
    with pytest.raises(OSError):
        gl.require_grid_lock(manifest, path, {"d01", "d02", *DOMS})
