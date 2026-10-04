from __future__ import annotations

from dataclasses import fields as dataclass_fields
from pathlib import Path

import jax.numpy as jnp
from netCDF4 import Dataset
import numpy as np
import pytest

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.noahmp_state import NoahMPLandState
from gpuwrf.contracts.precision import (
    DEFAULT_DTYPES, MYNN_DIAGNOSTIC_LEAVES, GWDO_DIAGNOSTIC_LEAVES, SURFACE_LAYER_CARRY_LEAVES,
)
from gpuwrf.contracts.state import SCALAR_BOUNDARY_OPTIONAL_LEAVES, State, _state_field_shapes
from gpuwrf.coupling.noahclassic_surface_hook import NoahClassicLandState, NoahClassicRadiation
from gpuwrf.io.wrfout_writer import bind_wrfout_domain_authority, write_wrfout_netcdf
from gpuwrf.io.restart import GWDO_DIAGNOSTICS_SCHEMA_VERSION
from gpuwrf.io.wrfrst_netcdf import (
    CARRY_ARRAY_FIELDS,
    OPTIONAL_CARRY_FIELDS,
    SCHEMA_VERSION,
    STOCHASTIC_SEED_RESTART_VARIABLES,
    UNSUPPORTED_CARRY_FIELDS,
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
from gpuwrf.runtime.operational_state import OperationalCarry, initial_operational_carry
from test_m7_netcdf_writer import authenticated_source_grid


REQUIRED_MYNN_LEAVES = ("el_pbl", "maxmf", "maxwidth", "ztop_plume")
OPTIONAL_GWDO_LEAVES = ("dtaux3d", "dtauy3d", "dusfcg", "dvsfcg")
assert MYNN_DIAGNOSTIC_LEAVES == REQUIRED_MYNN_LEAVES
assert GWDO_DIAGNOSTIC_LEAVES == OPTIONAL_GWDO_LEAVES
FROZEN_BASE_LEAF_COUNT = 62
REQUIRED_BASE_LEAF_COUNT = FROZEN_BASE_LEAF_COUNT + len(REQUIRED_MYNN_LEAVES)


def _pattern(shape: tuple[int, ...], dtype, offset: int):
    if np.dtype(dtype) == np.dtype("int32"):
        values = (np.arange(int(np.prod(shape)), dtype=np.int32).reshape(shape) + offset) % 30
        return jnp.asarray(values, dtype=dtype)
    values = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)
    values = values / 997.0 + offset / 991.0 + 0.25
    return jnp.asarray(values, dtype=dtype)


def _state(grid: GridSpec, *, mp_physics: int = 8) -> State:
    state = State(
        **{
            field: _pattern(shape, DEFAULT_DTYPES.dtype_for(field), index)
            for index, (field, shape) in enumerate(_state_field_shapes(grid, mp_physics=mp_physics).items(), start=1)
        }
    )

    assert len(State.__slots__) == 84
    assert State.__slots__[71:] == (SURFACE_LAYER_CARRY_LEAVES
                                  + REQUIRED_MYNN_LEAVES + OPTIONAL_GWDO_LEAVES)
    for name in REQUIRED_MYNN_LEAVES:
        value = getattr(state, name)
        shape = (grid.nz, grid.ny, grid.nx) if name == "el_pbl" else (grid.ny, grid.nx)
        assert value.shape == shape, name
        assert value.dtype == jnp.float32, name
        assert value.nbytes == 4 * int(np.prod(shape)), name
    for name in OPTIONAL_GWDO_LEAVES:
        assert getattr(state, name) is None, name
    return state


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
    return bool(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes() and np.array_equal(a, b))


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
    assert len(metadata["state_field_order"]) == REQUIRED_BASE_LEAF_COUNT == 66
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
    assert len(metadata["state_field_order"]) == REQUIRED_BASE_LEAF_COUNT + 5 == 71
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
        **{name: "none" for name in OPTIONAL_CARRY_FIELDS},
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
        domain="d01",
        domain_authority=bind_wrfout_domain_authority(
            "d01",
            authenticated_source_grid(grid, "d01"),
            grid,
        ),
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
        # The fixture targets the State-order guard after a valid current
        # GWDO marker; missing markers have a separate E78 fail-closed gate.
        dataset.GPUWRF_GWDO_DIAGNOSTICS_SCHEMA_VERSION = GWDO_DIAGNOSTICS_SCHEMA_VERSION
        dataset.GPUWRF_STATE_FIELD_ORDER = "[]"
        dataset.GPUWRF_STANDARD_RESTART_VARIABLES = "[]"
        dataset.GPUWRF_UNSUPPORTED_REGISTRY_RESTART_FIELDS = "[]"
        dataset.GPUWRF_CARRY_PRESENT = 0

    with pytest.raises(ValueError, match="State field order"):
        read_wrfrst_state(path)


# --- v0.25 M3R restart-gate repair: optional scalar wrfbdy leaves + exactness ---


def _boundary_state(grid: GridSpec, *, ntimes: int, scalar_leaves: tuple[str, ...] = ()) -> State:
    """State whose ``*_bdy`` leaves carry ``ntimes`` levels, plus the requested
    v0.22 optional scalar wrfbdy leaves (``qc_bdy`` .. ``Nr_bdy``); the others stay
    ``None`` exactly as ``State.__init__`` leaves them when wrfbdy lacks them."""

    shapes = _state_field_shapes(grid)
    fields: dict[str, object] = {}
    for index, (field, shape) in enumerate(shapes.items(), start=1):
        if field.endswith("_bdy"):
            shape = (ntimes, *shape[1:])
        fields[field] = _pattern(shape, DEFAULT_DTYPES.dtype_for(field), index)
    mass_bdy = (ntimes, *shapes["qv_bdy"][1:])
    for offset, leaf in enumerate(scalar_leaves, start=400):
        fields[leaf] = _pattern(mass_bdy, DEFAULT_DTYPES.dtype_for(leaf), offset)
    return State(**fields)


def _assert_state_roundtrip_exact(original: State, restored: State) -> None:
    assert restored.active_field_names() == original.active_field_names()
    for field in State.__slots__:
        left, right = getattr(original, field), getattr(restored, field)
        if left is None:
            assert right is None, field
        else:
            assert _equal(left, right), field


def test_wrfrst_optional_scalar_boundary_leaves_roundtrip_present_and_absent(tmp_path: Path) -> None:
    """The 2026-07-31 audit regression: the writer assumed every ``*_bdy`` slot holds
    an array with a time axis, but the optional scalar wrfbdy leaves are ``None``
    unless wrfbdy carried them. Both populations must round-trip exactly."""

    grid = GridSpec.canary_3km_template()
    kwargs = dict(valid_time="2026-06-03_00:30:00", run_start="2026-06-03_00:00:00")

    # (a) every optional scalar boundary leaf present, three boundary time levels.
    full = _boundary_state(grid, ntimes=3, scalar_leaves=SCALAR_BOUNDARY_OPTIONAL_LEAVES)
    assert len(full.active_field_names()) == REQUIRED_BASE_LEAF_COUNT + 7 == 73
    path_full = tmp_path / "wrfrst_scalar_bdy_full"
    write_wrfrst_state(full, grid, {}, path_full, step_index=6, **kwargs)
    restored_full, metadata_full = read_wrfrst_state(path_full)
    assert metadata_full["state_field_order"] == list(full.active_field_names())
    _assert_state_roundtrip_exact(full, restored_full)
    schema_full = inspect_wrfrst_schema(path_full)
    for leaf in ("u_bdy", "ph_bdy", "mu_bdy", *SCALAR_BOUNDARY_OPTIONAL_LEAVES):
        assert schema_full["dimensions"][f"gpuwrf_{leaf}_time"] == 3, leaf
        assert schema_full["variables"][state_extension_name(leaf)]["shape"][1] == 3, leaf

    # (b) a wrfbdy with only QCLOUD/QNICE strips: two present, five absent.
    subset = ("qc_bdy", "Ni_bdy")
    partial = _boundary_state(grid, ntimes=2, scalar_leaves=subset)
    assert len(partial.active_field_names()) == REQUIRED_BASE_LEAF_COUNT + 2 == 68
    path_partial = tmp_path / "wrfrst_scalar_bdy_partial"
    write_wrfrst_state(partial, grid, {}, path_partial, step_index=7, **kwargs)
    restored_partial, _ = read_wrfrst_state(path_partial)
    _assert_state_roundtrip_exact(partial, restored_partial)
    schema_partial = inspect_wrfrst_schema(path_partial)
    for leaf in SCALAR_BOUNDARY_OPTIONAL_LEAVES:
        present = leaf in subset
        assert (state_extension_name(leaf) in schema_partial["variables"]) is present, leaf
        assert (f"gpuwrf_{leaf}_time" in schema_partial["dimensions"]) is present, leaf
        assert (getattr(restored_partial, leaf) is not None) is present, leaf

    # (c) the same two populations through the full-carry writer/reader.
    for tag, state in (("full", full), ("partial", partial)):
        carry = initial_operational_carry(state)
        path_carry = tmp_path / f"wrfrst_scalar_bdy_carry_{tag}"
        write_wrfrst_carry(carry, grid, {}, path_carry, step_index=8, **kwargs)
        restored_carry, metadata_carry = read_wrfrst_carry(path_carry)
        assert metadata_carry["carry_present"] is True
        _assert_state_roundtrip_exact(carry.state, restored_carry.state)
        for field in CARRY_ARRAY_FIELDS:
            assert _equal(getattr(carry, field), getattr(restored_carry, field)), field


def test_wrfrst_exact_payload_shape_violations_fail_closed_before_any_file_exists(tmp_path: Path) -> None:
    """Exact ``GPUWRF_*`` variables are never broadcast/squeezed into shape: a leaf
    that could not be read back bit-identically is refused, and refused BEFORE the
    NetCDF file is created so no partial restart is ever left behind."""

    grid = GridSpec.canary_3km_template()
    kwargs = dict(valid_time="2026-06-03_00:30:00", run_start="2026-06-03_00:00:00", step_index=1)
    state = _boundary_state(grid, ntimes=2, scalar_leaves=("qc_bdy", "qr_bdy"))

    legacy_rank = state.replace(_cast=False, qc_bdy=jnp.asarray(state.qc_bdy)[:, :, 0])  # (t, side, z, side_index)
    path = tmp_path / "legacy_rank"
    with pytest.raises(ValueError, match="qc_bdy must be 5-D"):
        write_wrfrst_state(legacy_rank, grid, {}, path, **kwargs)
    assert not path.exists()

    narrow_width = state.replace(_cast=False, qr_bdy=jnp.asarray(state.qr_bdy)[:, :, :3])
    path = tmp_path / "narrow_width"
    with pytest.raises(ValueError, match="State leaves do not match their exact wrfrst shapes"):
        write_wrfrst_state(narrow_width, grid, {}, path, **kwargs)
    assert not path.exists()

    scalar_surface = state.replace(_cast=False, mavail=jnp.asarray(0.2))
    path = tmp_path / "scalar_surface"
    with pytest.raises(ValueError, match="mavail"):
        write_wrfrst_state(scalar_surface, grid, {}, path, **kwargs)
    assert not path.exists()

    carry = initial_operational_carry(state).replace(t_2ave=jnp.zeros((grid.nz, grid.ny, grid.nx - 1)))
    path = tmp_path / "carry_shape"
    with pytest.raises(ValueError, match="OperationalCarry leaves do not match"):
        write_wrfrst_carry(carry, grid, {}, path, **kwargs)
    assert not path.exists()


def test_wrfrst_exact_read_keeps_force_fp64_leaf_dtypes(tmp_path: Path) -> None:
    """``run_forecast_operational(force_fp64=True)`` upcasts the fp32-gated leaves
    (``u_bdy``, ``Nc``, ``qc_bdy`` ...) to float64 with ``replace(_cast=False)``.
    The exact reader must hand back the STORED dtype, not re-canonicalise through
    ``State.__init__`` (which would silently downcast them to the fp32 matrix)."""

    grid = GridSpec.canary_3km_template()
    base = _boundary_state(grid, ntimes=2, scalar_leaves=("qc_bdy",))
    forced = base.replace(
        _cast=False,
        u_bdy=jnp.asarray(base.u_bdy, dtype=jnp.float64),
        Nc=jnp.asarray(base.Nc, dtype=jnp.float64),
        qc_bdy=jnp.asarray(base.qc_bdy, dtype=jnp.float64),
    )
    assert np.asarray(base.u_bdy).dtype == np.float32  # the gated default this guards against
    kwargs = dict(valid_time="2026-06-03_00:30:00", run_start="2026-06-03_00:00:00", step_index=2)

    path = tmp_path / "wrfrst_force_fp64_state"
    write_wrfrst_state(forced, grid, {}, path, **kwargs)
    restored, _ = read_wrfrst_state(path)
    _assert_state_roundtrip_exact(forced, restored)
    for leaf in ("u_bdy", "Nc", "qc_bdy"):
        assert np.asarray(getattr(restored, leaf)).dtype == np.float64, leaf
    assert np.asarray(restored.lu_index).dtype == np.int32

    path = tmp_path / "wrfrst_force_fp64_carry"
    write_wrfrst_carry(initial_operational_carry(forced), grid, {}, path, **kwargs)
    restored_carry, _ = read_wrfrst_carry(path)
    _assert_state_roundtrip_exact(forced, restored_carry.state)


def test_wrfrst_carry_inventory_is_complete_and_new_groups_roundtrip(tmp_path: Path) -> None:
    """Every carry field has a declared v3 payload; optional land rad is exact."""
    carry_fields = {field.name for field in dataclass_fields(OperationalCarry)}
    covered = {"state", *CARRY_ARRAY_FIELDS, *OPTIONAL_CARRY_FIELDS}
    assert covered == carry_fields
    assert UNSUPPORTED_CARRY_FIELDS == ()
    grid = GridSpec.canary_3km_template()
    carry = initial_operational_carry(_state(grid)).replace(slab_rad=(jnp.zeros((grid.ny, grid.nx)),) * 2)
    path = tmp_path / "wrfrst_supported_carry"
    write_wrfrst_carry(
        carry, grid, {}, path, valid_time="2026-06-03_00:30:00", run_start="2026-06-03_00:00:00", step_index=1
    )
    restored, _ = read_wrfrst_carry(path)
    for left, right in zip(carry.slab_rad, restored.slab_rad, strict=True):
        assert _equal(left, right)
