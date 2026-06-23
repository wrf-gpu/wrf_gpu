from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
from netCDF4 import Dataset
import numpy as np
import pytest

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.noahmp_state import NoahMPLandState
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.coupling.noahclassic_surface_hook import NoahClassicLandState, NoahClassicRadiation
from gpuwrf.io.wrfout_writer import write_wrfout_netcdf
from gpuwrf.io.wrfrst_netcdf import (
    CARRY_ARRAY_FIELDS,
    SCHEMA_VERSION,
    STOCHASTIC_SEED_RESTART_VARIABLES,
    carry_extension_name,
    cumulus_extension_name,
    inspect_wrfrst_schema,
    noahclassic_land_extension_name,
    noahclassic_rad_extension_name,
    noahmp_land_extension_name,
    noahmp_rad_extension_name,
    read_wrfrst_carry,
    read_wrfrst_state,
    read_wrfrst_stochastic_seeds,
    state_extension_name,
    write_wrfrst_carry,
    write_wrfrst_state,
)
from gpuwrf.runtime.operational_state import initial_operational_carry


def _pattern(shape: tuple[int, ...], dtype, offset: int):
    if np.dtype(dtype) == np.dtype("int32"):
        values = (np.arange(int(np.prod(shape)), dtype=np.int32).reshape(shape) + offset) % 30
        return jnp.asarray(values, dtype=dtype)
    values = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)
    values = values / 997.0 + offset / 991.0 + 0.25
    return jnp.asarray(values, dtype=dtype)


def _state(grid: GridSpec, *, mp_physics: int = 8) -> State:
    return State(
        **{
            field: _pattern(shape, DEFAULT_DTYPES.dtype_for(field), index)
            for index, (field, shape) in enumerate(_state_field_shapes(grid, mp_physics=mp_physics).items(), start=1)
        }
    )


def _array(shape: tuple[int, ...], offset: int, *, dtype=jnp.float64):
    return _pattern(shape, dtype, offset)


def _noahmp_land(grid: GridSpec) -> NoahMPLandState:
    xy = (grid.ny, grid.nx)
    soil = (4, grid.ny, grid.nx)
    snow = (3, grid.ny, grid.nx)
    snso = (7, grid.ny, grid.nx)
    return NoahMPLandState(
        tslb=_array(soil, 101),
        smois=_array(soil, 102),
        sh2o=_array(soil, 103),
        smcwtd=_array(xy, 104),
        isnow=_array(xy, 105, dtype=jnp.int32),
        tsno=_array(snow, 106),
        snice=_array(snow, 107),
        snliq=_array(snow, 108),
        zsnso=_array(snso, 109),
        snowh=_array(xy, 110),
        sneqv=_array(xy, 111),
        sneqvo=_array(xy, 112),
        tauss=_array(xy, 113),
        albold=_array(xy, 114),
        tv=_array(xy, 115),
        tg=_array(xy, 116),
        tah=_array(xy, 117),
        eah=_array(xy, 118),
        canliq=_array(xy, 119),
        canice=_array(xy, 120),
        fwet=_array(xy, 121),
        lai=_array(xy, 122),
        sai=_array(xy, 123),
        cm=_array(xy, 124),
        ch=_array(xy, 125),
        t_skin=_array(xy, 126),
        qsfc=_array(xy, 127),
        znt=_array(xy, 128),
        emiss=_array(xy, 129),
        albedo=_array(xy, 130),
        sfcrunoff=_array(xy, 131),
        udrunoff=_array(xy, 132),
    )


def _noahclassic_land(grid: GridSpec) -> NoahClassicLandState:
    xy = (grid.ny, grid.nx)
    trailing_soil = (grid.ny, grid.nx, 4)
    return NoahClassicLandState(
        t1=_array(xy, 201),
        stc=_array(trailing_soil, 202),
        smc=_array(trailing_soil, 203),
        sh2o=_array(trailing_soil, 204),
        cmc=_array(xy, 205),
        sneqv=_array(xy, 206),
        snowh=_array(xy, 207),
        sncovr=_array(xy, 208),
        snotime1=_array(xy, 209),
        ribb=_array(xy, 210),
        flx4=_array(xy, 211),
        fvb=_array(xy, 212),
        fbur=_array(xy, 213),
        fgsn=_array(xy, 214),
        smcrel=_array(trailing_soil, 215),
        xlaidyn=_array(xy, 216),
        hfx=_array(xy, 217),
        qfx=_array(xy, 218),
        lh=_array(xy, 219),
        grdflx=_array(xy, 220),
    )


def _seed_arrays() -> dict[str, np.ndarray]:
    return {
        name: np.arange(8, dtype=np.int32) + offset * 100
        for offset, name in enumerate(STOCHASTIC_SEED_RESTART_VARIABLES, start=1)
    }


def _equal(left, right) -> bool:
    a = np.asarray(left)
    b = np.asarray(right)
    return bool(a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b))


def _assert_object_equal(left, right, fields: tuple[str, ...]) -> None:
    for field in fields:
        assert _equal(getattr(left, field), getattr(right, field)), field


def test_wrfrst_state_roundtrip_bit_identical_and_wrf_schema(tmp_path: Path) -> None:
    grid = GridSpec.canary_3km_template()
    state = _state(grid)
    path = tmp_path / "wrfrst_d01_2026-06-03_00:00:00"

    write_wrfrst_state(
        state,
        grid,
        {},
        path,
        valid_time="2026-06-03_00:10:00",
        run_start="2026-06-03_00:00:00",
        step_index=1,
    )
    restored, metadata = read_wrfrst_state(path)

    assert metadata["schema_version"] == SCHEMA_VERSION
    assert metadata["step_index"] == 1
    assert metadata["state_field_order"] == list(state.active_field_names())
    assert len(metadata["state_field_order"]) == 57  # v0.20 S1: -3 legacy p/ph/mu aliases
    for field in state.active_field_names():
        assert _equal(getattr(state, field), getattr(restored, field)), field

    schema = inspect_wrfrst_schema(path)
    assert "Times" in schema["variables"]
    for name in metadata["standard_restart_variables"]:
        assert name in schema["variables"], name
    for field in state.active_field_names():
        assert state_extension_name(field) in schema["variables"], field
    for name in ("QHAIL", "QNHAIL", "QVGRAUPEL", "QVHAIL", "QNWFA", "QNIFA", "HAILNC"):
        assert name not in schema["variables"], name
    for field in ("qh", "Nh", "qvolg", "qvolh", "nwfa", "nifa", "hail_acc"):
        assert state_extension_name(field) not in schema["variables"], field
    assert schema["variables"]["U"]["stagger"] == "X"
    assert schema["variables"]["V"]["stagger"] == "Y"
    assert schema["variables"]["W"]["stagger"] == "Z"
    assert schema["variables"]["T"]["dimensions"] == ["Time", "bottom_top", "south_north", "west_east"]


def test_wrfrst_hail_state_roundtrip_writes_hail_conditionals(tmp_path: Path) -> None:
    grid = GridSpec.canary_3km_template()
    state = _state(grid, mp_physics=24)
    path = tmp_path / "wrfrst_hail"

    write_wrfrst_state(
        state,
        grid,
        {},
        path,
        valid_time="2026-06-03_00:10:00",
        run_start="2026-06-03_00:00:00",
        step_index=4,
    )
    restored, metadata = read_wrfrst_state(path)
    schema = inspect_wrfrst_schema(path)

    assert metadata["state_field_order"] == list(state.active_field_names())
    assert len(metadata["state_field_order"]) == 62  # v0.20 S1: 57 base + 5 hail
    for field in ("qh", "Nh", "qvolg", "qvolh", "hail_acc"):
        assert getattr(restored, field) is not None, field
        assert state_extension_name(field) in schema["variables"], field
    for field in ("nwfa", "nifa"):
        assert getattr(restored, field) is None, field
        assert state_extension_name(field) not in schema["variables"], field
    for name in ("QHAIL", "QNHAIL", "QVGRAUPEL", "QVHAIL", "HAILNC"):
        assert name in schema["variables"], name
    for name in ("QNWFA", "QNIFA"):
        assert name not in schema["variables"], name
    for field in state.active_field_names():
        assert _equal(getattr(state, field), getattr(restored, field)), field


def test_wrfrst_carry_roundtrip_includes_promoted_scratch(tmp_path: Path) -> None:
    grid = GridSpec.canary_3km_template()
    state = _state(grid)
    carry = initial_operational_carry(state)
    carry = carry.replace(rthraten=jnp.ones_like(carry.rthraten) * 1.25e-5)
    path = tmp_path / "wrfrst_carry"

    write_wrfrst_carry(
        carry,
        grid,
        {},
        path,
        valid_time="2026-06-03_00:20:00",
        run_start="2026-06-03_00:00:00",
        step_index=2,
    )
    restored, metadata = read_wrfrst_carry(path)

    assert metadata["carry_present"] is True
    assert metadata["step_index"] == 2
    assert metadata["state_field_order"] == list(carry.state.active_field_names())
    for field in carry.state.active_field_names():
        assert _equal(getattr(carry.state, field), getattr(restored.state, field)), field
    for field in CARRY_ARRAY_FIELDS:
        assert _equal(getattr(carry, field), getattr(restored, field)), field
    schema = inspect_wrfrst_schema(path)
    for field in CARRY_ARRAY_FIELDS:
        assert carry_extension_name(field) in schema["variables"], field


def test_wrfrst_optional_nested_carry_roundtrip_and_wrf_land_schema(tmp_path: Path) -> None:
    grid = GridSpec.canary_3km_template()
    state = _state(grid)
    xy = (grid.ny, grid.nx)
    carry = initial_operational_carry(
        state,
        noahmp_land=_noahmp_land(grid),
        noahmp_rad=(_array(xy, 301), _array(xy, 302), _array(xy, 303)),
        cumulus_carry=(_array((grid.nz, grid.ny, grid.nx), 304), _array(xy, 305, dtype=jnp.int32)),
        noahclassic_land=_noahclassic_land(grid),
        noahclassic_rad=NoahClassicRadiation(_array(xy, 306), _array(xy, 307), _array(xy, 308)),
    )
    path = tmp_path / "wrfrst_full_carry"
    seeds = _seed_arrays()

    write_wrfrst_carry(
        carry,
        grid,
        {},
        path,
        valid_time="2026-06-03_00:30:00",
        run_start="2026-06-03_00:00:00",
        step_index=3,
        stochastic_seed_arrays=seeds,
    )
    restored, metadata = read_wrfrst_carry(path)
    restored_seeds = read_wrfrst_stochastic_seeds(path)

    assert metadata["optional_carry_kind"] == {
        "noahmp_land": "object",
        "noahmp_rad": "tuple",
        "cumulus_carry": "tuple",
        "noahclassic_land": "object",
        "noahclassic_rad": "tuple",
    }
    assert metadata["stochastic_seed_variables"] == list(STOCHASTIC_SEED_RESTART_VARIABLES)
    _assert_object_equal(carry.noahmp_land, restored.noahmp_land, tuple(NoahMPLandState.__slots__))
    for left, right in zip(carry.noahmp_rad, restored.noahmp_rad, strict=True):
        assert _equal(left, right)
    for left, right in zip(carry.cumulus_carry, restored.cumulus_carry, strict=True):
        assert _equal(left, right)
    _assert_object_equal(carry.noahclassic_land, restored.noahclassic_land, tuple(NoahClassicLandState._fields))
    _assert_object_equal(carry.noahclassic_rad, restored.noahclassic_rad, tuple(NoahClassicRadiation._fields))

    schema = inspect_wrfrst_schema(path)
    for name, dimension in {
        "snow_layers_stag": 3,
        "snso_layers_stag": 7,
        "seed_dim_stag": 8,
    }.items():
        assert schema["dimensions"][name] == dimension
    for name in ("TSLB", "SMOIS", "SH2O", "TSNO", "SNICE", "SNLIQ", "ZSNSO"):
        assert name in schema["variables"], name
    for name in STOCHASTIC_SEED_RESTART_VARIABLES:
        assert name in schema["variables"], name
        assert schema["variables"][name]["dimensions"] == ["Time", "seed_dim_stag"]
        assert schema["variables"][name]["dtype"] == "int32"
        np.testing.assert_array_equal(np.asarray(restored_seeds[name]), seeds[name])
    assert noahmp_land_extension_name("tsno") in schema["variables"]
    assert noahmp_rad_extension_name("soldn") in schema["variables"]
    assert cumulus_extension_name("w0avg") in schema["variables"]
    assert noahclassic_land_extension_name("stc") in schema["variables"]
    assert schema["variables"][noahclassic_land_extension_name("stc")]["dimensions"] == [
        "Time",
        "south_north",
        "west_east",
        "soil_layers_stag",
    ]
    assert noahclassic_rad_extension_name("cosz") in schema["variables"]


def test_wrfout_writes_ki3_snow_snso_and_seed_dimensions(tmp_path: Path) -> None:
    grid = GridSpec.canary_3km_template()
    state = _state(grid)
    diagnostics = {
        "ISEEDARR_SPPT": np.arange(8, dtype=np.int32),
        "ISEEDARR_SKEBS": np.arange(8, dtype=np.int32) + 10,
    }
    path = tmp_path / "wrfout_d01_2026-06-03_00:00:00"

    write_wrfout_netcdf(
        state,
        grid,
        {},
        path,
        valid_time="2026-06-03_00:00:00",
        lead_hours=0.0,
        run_start="2026-06-03_00:00:00",
        diagnostics=diagnostics,
        land_state=_noahmp_land(grid),
    )

    with Dataset(path, "r") as dataset:
        assert len(dataset.dimensions["snow_layers_stag"]) == 3
        assert len(dataset.dimensions["snso_layers_stag"]) == 7
        assert len(dataset.dimensions["seed_dim_stag"]) == 8
        assert dataset.variables["TSNO"].dimensions == ("Time", "snow_layers_stag", "south_north", "west_east")
        assert dataset.variables["ZSNSO"].dimensions == ("Time", "snso_layers_stag", "south_north", "west_east")
        assert dataset.variables["ISEEDARR_SPPT"].dimensions == ("Time", "seed_dim_stag")
        assert np.dtype(dataset.variables["ISEEDARR_SPPT"].dtype) == np.dtype("int32")
        np.testing.assert_array_equal(dataset.variables["ISEEDARR_SPPT"][0], diagnostics["ISEEDARR_SPPT"])


def test_wrfrst_missing_schema_fields_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "bad_wrfrst"
    with Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.createDimension("Time", None)
        dataset.createDimension("DateStrLen", 19)
        dataset.GPUWRF_WRFRST_SCHEMA_VERSION = SCHEMA_VERSION
        dataset.GPUWRF_STATE_FIELD_ORDER = "[]"
        dataset.GPUWRF_STANDARD_RESTART_VARIABLES = "[]"
        dataset.GPUWRF_UNSUPPORTED_REGISTRY_RESTART_FIELDS = "[]"
        dataset.GPUWRF_CARRY_PRESENT = 0

    with pytest.raises(ValueError, match="State field order"):
        read_wrfrst_state(path)
