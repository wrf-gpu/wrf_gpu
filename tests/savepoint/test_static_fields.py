from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
from netCDF4 import Dataset
import numpy as np
import pytest

from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.io.land_state import load_prescribed_land_state
from gpuwrf.physics.noah_mp import roughness_from_prescribed_fields


RUN_PATH = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z")
WRFINPUT = RUN_PATH / "wrfinput_d02"


def _read_wrfinput_2d(name: str) -> np.ndarray:
    with Dataset(WRFINPUT, "r") as dataset:
        values = dataset.variables[name][:]
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    array = np.asarray(values)
    if array.shape[:1] == (1,):
        array = array[0]
    return array


def _optional_wrfinput_2d(name: str) -> np.ndarray | None:
    with Dataset(WRFINPUT, "r") as dataset:
        if name not in dataset.variables:
            return None
    return _read_wrfinput_2d(name)


def _assert_exact_numeric_match(name: str, candidate, reference) -> None:
    actual = np.asarray(candidate)
    expected = np.asarray(reference)
    assert actual.shape == expected.shape, name
    diff = np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    assert float(np.nanmax(np.abs(diff))) == 0.0, name
    assert np.array_equal(actual, expected), name


def _assert_exact_category_match(name: str, candidate, reference) -> None:
    actual = np.asarray(candidate).astype(np.int64)
    expected = np.asarray(reference).astype(np.int64)
    assert actual.shape == expected.shape, name
    assert int(np.max(np.abs(actual - expected))) == 0, name
    assert np.array_equal(actual, expected), name


def _cpu_state_with_lu_index(lu_index: np.ndarray) -> State:
    ny, nx = lu_index.shape
    grid = SimpleNamespace(nz=1, ny=int(ny), nx=int(nx))
    fields = {
        name: jnp.zeros(shape, dtype=jnp.int32 if name == "lu_index" else jnp.float64)
        for name, shape in _state_field_shapes(grid).items()
    }
    fields["lu_index"] = jnp.asarray(lu_index)
    return State(**fields)


def test_inv8_canary_static_fields_match_wrfinput() -> None:
    if not WRFINPUT.exists():
        pytest.skip("Canary 20260521 wrfinput_d02 fixture unavailable")

    run = Gen2Run(RUN_PATH)
    land = load_prescribed_land_state(run, domain="d02", time=0)

    lu_index = _read_wrfinput_2d("LU_INDEX")
    state = _cpu_state_with_lu_index(lu_index)
    _assert_exact_category_match("state.lu_index", state.lu_index, lu_index)
    _assert_exact_category_match("land.lu_index", land.lu_index, lu_index)
    assert state.lu_index.dtype == jnp.int32

    _assert_exact_numeric_match("HGT", run.load_wrfinput("d02", "HGT", lazy=False), _read_wrfinput_2d("HGT"))
    _assert_exact_numeric_match("LANDMASK", land.landmask, _read_wrfinput_2d("LANDMASK"))
    _assert_exact_numeric_match("XLAND", land.xland, _read_wrfinput_2d("XLAND"))
    _assert_exact_category_match("IVGTYP", land.ivgtyp, _read_wrfinput_2d("IVGTYP"))
    _assert_exact_category_match("ISLTYP", land.isltyp, _read_wrfinput_2d("ISLTYP"))

    znt = _optional_wrfinput_2d("ZNT")
    if znt is None:
        roughness_reference = roughness_from_prescribed_fields(
            _read_wrfinput_2d("XLAND"),
            _read_wrfinput_2d("LANDMASK"),
            vegfra=_optional_wrfinput_2d("VEGFRA"),
            cm=_optional_wrfinput_2d("CM"),
            lu_index=lu_index,
        )
    else:
        roughness_reference = jnp.clip(jnp.asarray(znt, dtype=jnp.float64), 1.0e-7, 10.0)
    _assert_exact_numeric_match("roughness_m", land.roughness_m, roughness_reference)
