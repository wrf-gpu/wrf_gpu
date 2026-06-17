"""WRF-compatible NetCDF restart (wrfrst) writer/reader for gpuwrf.

The file carries two layers:

* WRF-named restart variables with WRF dimensions, staggering and attributes.
* Exact ``GPUWRF_*`` extension variables for every ``State`` leaf and promoted
  operational scratch field.  The extensions make gpuwrf resume fail-closed and
  bitwise, while WRF tools can still inspect the standard restart surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

from gpuwrf.contracts.state import CONDITIONAL_STATE_LEAVES, State
from gpuwrf.io.wrfout_writer import (
    DATE_STR_LEN,
    MAPFAC_U_XY,
    MAPFAC_V_XY,
    SEED,
    SNOW,
    SNSO,
    SOIL,
    STOCHASTIC_SEED_VARIABLES,
    TIME_ONLY,
    U_XY,
    U_XYZ,
    V_XY,
    V_XYZ,
    WRFOUT_VARIABLE_SPECS,
    WrfoutVariableSpec,
    W_XYZ,
    XY,
    XYZ,
    Z_FULL,
    Z_HALF,
    Z_XYZ,
    _add_grid_coordinate_fields,
    _coerce_array,
    _coerce_datetime,
    _create_dimensions,
    _dimension_sizes,
    _grid_extent,
    _grid_or_state_array,
    _latlon_fields,
    _landmask,
    _lookup,
    _set_variable_attrs,
    _shape_for_dimensions,
    _write_global_attrs,
    _write_times,
    _wrf_time_string,
)
from gpuwrf.runtime.operational_state import OperationalCarry

try:
    from gpuwrf.contracts.noahmp_state import NoahMPLandState
except Exception:  # pragma: no cover - optional package surface
    NoahMPLandState = None  # type: ignore

try:
    from gpuwrf.coupling.noahclassic_surface_hook import NoahClassicLandState, NoahClassicRadiation
except Exception:  # pragma: no cover - optional package surface
    NoahClassicLandState = None  # type: ignore
    NoahClassicRadiation = None  # type: ignore


SCHEMA_VERSION = "v0.11.0-wrfrst-netcdf-2"
THETA_BASE_OFFSET_K = 300.0
STATE_EXTENSION_PREFIX = "GPUWRF_STATE_"
CARRY_EXTENSION_PREFIX = "GPUWRF_CARRY_"
NOAHMP_LAND_EXTENSION_PREFIX = "GPUWRF_NOAHMP_LAND_"
NOAHMP_RAD_EXTENSION_PREFIX = "GPUWRF_NOAHMP_RAD_"
CUMULUS_EXTENSION_PREFIX = "GPUWRF_CUMULUS_"
NOAHCLASSIC_LAND_EXTENSION_PREFIX = "GPUWRF_NOAHCLASSIC_LAND_"
NOAHCLASSIC_RAD_EXTENSION_PREFIX = "GPUWRF_NOAHCLASSIC_RAD_"

BDY_TIME = "gpuwrf_bdy_time"
BDY_SIDE = "gpuwrf_bdy_side"
BDY_WIDTH = "gpuwrf_bdy_width"
BDY_SIDE_INDEX = "gpuwrf_bdy_side_index"
BDY_SURFACE = "gpuwrf_bdy_surface"
SOIL_TRAILING = ("Time", "south_north", "west_east", "soil_layers_stag")

STATE_FIELD_ORDER: tuple[str, ...] = tuple(State.__slots__)


def _validate_state_field_order(field_order: tuple[str, ...]) -> None:
    if any(field not in STATE_FIELD_ORDER for field in field_order):
        raise ValueError("wrfrst State field order contains unknown leaves")
    if field_order != tuple(field for field in STATE_FIELD_ORDER if field in field_order):
        raise ValueError("wrfrst State field order does not match current State.__slots__")
    missing = tuple(field for field in STATE_FIELD_ORDER if field not in field_order)
    if any(field not in CONDITIONAL_STATE_LEAVES for field in missing):
        raise ValueError(
            "wrfrst State field order is missing non-conditional leaves: "
            f"{[field for field in missing if field not in CONDITIONAL_STATE_LEAVES]}"
        )


def _state_field_order_from_dataset(dataset: Dataset) -> tuple[str, ...]:
    field_order = tuple(json.loads(str(getattr(dataset, "GPUWRF_STATE_FIELD_ORDER", "[]"))))
    _validate_state_field_order(field_order)
    return field_order

WRF_STANDARD_RESTART_VARIABLES: tuple[str, ...] = (
    "XTIME",
    "ITIMESTEP",
    "XLAT",
    "XLONG",
    "XLAT_U",
    "XLONG_U",
    "XLAT_V",
    "XLONG_V",
    "HGT",
    "LANDMASK",
    "ZNU",
    "ZNW",
    "P_TOP",
    "MAPFAC_M",
    "MAPFAC_U",
    "MAPFAC_V",
    "F",
    "E",
    "SINALPHA",
    "COSALPHA",
    "U",
    "V",
    "W",
    "T",
    "P",
    "PB",
    "PH",
    "PHB",
    "MU",
    "MUB",
    "QVAPOR",
    "QCLOUD",
    "QRAIN",
    "QICE",
    "QSNOW",
    "QGRAUP",
    "QNICE",
    "QNRAIN",
    "QNSNOW",
    "QNGRAUPEL",
    "QNCLOUD",
    "QNCCN",
    # v0.17 ADR-032 graupel/hail substrate.
    "QHAIL",
    "QNHAIL",
    "QVGRAUPEL",
    "QVHAIL",
    # v0.16 aerosol-aware Thompson (mp=28).
    "QNWFA",
    "QNIFA",
    "QKE",
    "UST",
    "TSK",
    "XLAND",
    "MAVAIL",
    "ZNT",
    "LU_INDEX",
    "RAINC",
    "RAINNC",
    "SNOWNC",
    "GRAUPELNC",
    # v0.17 hail microphysics surface accumulator (WSM7/WDM7).
    "HAILNC",
)

_ALWAYS_WRITTEN_STANDARD_RESTART_VARIABLES: frozenset[str] = frozenset((
    "XTIME",
    "ITIMESTEP",
    "XLAT",
    "XLONG",
    "XLAT_U",
    "XLONG_U",
    "XLAT_V",
    "XLONG_V",
    "HGT",
    "LANDMASK",
    "ZNU",
    "ZNW",
    "P_TOP",
    "MAPFAC_M",
    "MAPFAC_U",
    "MAPFAC_V",
    "F",
    "E",
    "SINALPHA",
    "COSALPHA",
))

# WRF-named restart fields written when the matching optional carry is present.
NOAHMP_WRF_RESTART_VARIABLES: tuple[str, ...] = (
    "TSLB",
    "SMOIS",
    "SH2O",
    "TSNO",
    "SNICE",
    "SNLIQ",
    "ZSNSO",
    "SNOW",
    "SNOWH",
    "CANWAT",
    "SFROFF",
    "UDROFF",
    "ALBEDO",
    "EMISS",
)

STOCHASTIC_SEED_RESTART_VARIABLES: tuple[str, ...] = tuple(STOCHASTIC_SEED_VARIABLES)
OPTIONAL_WRF_RESTART_VARIABLES: tuple[str, ...] = (
    *NOAHMP_WRF_RESTART_VARIABLES,
    *STOCHASTIC_SEED_RESTART_VARIABLES,
)

# Registry restart fields that are still outside the supported Canary runtime
# options or cannot be represented without a native WRF reader.
DEFERRED_REGISTRY_RESTART_FIELDS: tuple[str, ...] = (
    "Native WRF wrfrst ingestion without GPUWRF exact extensions is not accepted",
    "Inactive WRF stochastic-perturbation seed arrays are written only when explicit seed state is provided",
    "WRF options outside the scan-wired Canary suite remain intentionally absent",
)

CARRY_ARRAY_FIELDS: tuple[str, ...] = (
    "t_2ave",
    "ww",
    "mudf",
    "muave",
    "muts",
    "ph_tend",
    "u_save",
    "v_save",
    "w_save",
    "t_save",
    "ph_save",
    "mu_save",
    "ww_save",
    "rthraten",
)

OPTIONAL_CARRY_FIELDS: tuple[str, ...] = (
    "noahmp_land",
    "noahmp_rad",
    "cumulus_carry",
    "noahclassic_land",
    "noahclassic_rad",
)


@dataclass(frozen=True)
class RestartVariableSpec:
    """WRF restart variable schema metadata."""

    name: str
    dimensions: tuple[str, ...]
    memory_order: str
    description: str
    units: str
    stagger: str = ""
    coordinates: str | None = None
    dtype: str | None = None


@dataclass(frozen=True)
class StandardRestartField:
    """A WRF-named restart variable generated from the gpuwrf State."""

    spec: RestartVariableSpec
    value: Callable[[State], Any]


def _spec(
    name: str,
    dimensions: tuple[str, ...],
    memory_order: str,
    description: str,
    units: str,
    *,
    stagger: str = "",
    coordinates: str | None = None,
    dtype: str | None = None,
) -> RestartVariableSpec:
    return RestartVariableSpec(
        name=name,
        dimensions=dimensions,
        memory_order=memory_order,
        description=description,
        units=units,
        stagger=stagger,
        coordinates=coordinates,
        dtype=dtype,
    )


def _from_wrfout(name: str, *, dtype: str | None = None) -> RestartVariableSpec:
    source = WRFOUT_VARIABLE_SPECS[name]
    return _spec(
        name,
        source.dimensions,
        source.memory_order,
        source.description,
        source.units,
        stagger=source.stagger,
        coordinates=source.coordinates,
        dtype=dtype,
    )


WRFRST_EXTRA_SPECS: dict[str, RestartVariableSpec] = {
    "ITIMESTEP": _spec("ITIMESTEP", TIME_ONLY, "0  ", "WRF restart integer timestep", "", dtype="i4"),
    "MAVAIL": _spec(
        "MAVAIL",
        XY,
        "XY ",
        "MOISTURE AVAILABILITY",
        "",
        coordinates="XLONG XLAT XTIME",
    ),
    "ZNT": _spec(
        "ZNT",
        XY,
        "XY ",
        "TIME-VARYING ROUGHNESS LENGTH",
        "m",
        coordinates="XLONG XLAT XTIME",
    ),
}


def _standard_spec(name: str) -> RestartVariableSpec:
    if name in WRFRST_EXTRA_SPECS:
        return WRFRST_EXTRA_SPECS[name]
    return _from_wrfout(name)


STANDARD_FIELD_SPECS: dict[str, RestartVariableSpec] = {
    name: _standard_spec(name)
    for name in WRF_STANDARD_RESTART_VARIABLES
    if name not in {"XTIME", "ITIMESTEP"}
}

NOAHMP_WRF_FIELD_SPECS: dict[str, RestartVariableSpec] = {
    name: _from_wrfout(name)
    for name in NOAHMP_WRF_RESTART_VARIABLES
}

STOCHASTIC_SEED_FIELD_SPECS: dict[str, RestartVariableSpec] = {
    name: _from_wrfout(name, dtype="i4")
    for name in STOCHASTIC_SEED_RESTART_VARIABLES
}


def _base_pair(total: Any, perturbation: Any) -> Any:
    return jnp.asarray(total) - jnp.asarray(perturbation)


STANDARD_RESTART_FIELDS: tuple[StandardRestartField, ...] = (
    StandardRestartField(STANDARD_FIELD_SPECS["U"], lambda state: state.u),
    StandardRestartField(STANDARD_FIELD_SPECS["V"], lambda state: state.v),
    StandardRestartField(STANDARD_FIELD_SPECS["W"], lambda state: state.w),
    StandardRestartField(STANDARD_FIELD_SPECS["T"], lambda state: jnp.asarray(state.theta) - THETA_BASE_OFFSET_K),
    StandardRestartField(STANDARD_FIELD_SPECS["P"], lambda state: state.p_perturbation),
    StandardRestartField(STANDARD_FIELD_SPECS["PB"], lambda state: _base_pair(state.p_total, state.p_perturbation)),
    StandardRestartField(STANDARD_FIELD_SPECS["PH"], lambda state: state.ph_perturbation),
    StandardRestartField(STANDARD_FIELD_SPECS["PHB"], lambda state: _base_pair(state.ph_total, state.ph_perturbation)),
    StandardRestartField(STANDARD_FIELD_SPECS["MU"], lambda state: state.mu_perturbation),
    StandardRestartField(STANDARD_FIELD_SPECS["MUB"], lambda state: _base_pair(state.mu_total, state.mu_perturbation)),
    StandardRestartField(STANDARD_FIELD_SPECS["QVAPOR"], lambda state: state.qv),
    StandardRestartField(STANDARD_FIELD_SPECS["QCLOUD"], lambda state: state.qc),
    StandardRestartField(STANDARD_FIELD_SPECS["QRAIN"], lambda state: state.qr),
    StandardRestartField(STANDARD_FIELD_SPECS["QICE"], lambda state: state.qi),
    StandardRestartField(STANDARD_FIELD_SPECS["QSNOW"], lambda state: state.qs),
    StandardRestartField(STANDARD_FIELD_SPECS["QGRAUP"], lambda state: state.qg),
    StandardRestartField(STANDARD_FIELD_SPECS["QNICE"], lambda state: state.Ni),
    StandardRestartField(STANDARD_FIELD_SPECS["QNRAIN"], lambda state: state.Nr),
    StandardRestartField(STANDARD_FIELD_SPECS["QNSNOW"], lambda state: state.Ns),
    StandardRestartField(STANDARD_FIELD_SPECS["QNGRAUPEL"], lambda state: state.Ng),
    StandardRestartField(STANDARD_FIELD_SPECS["QNCLOUD"], lambda state: state.Nc),
    StandardRestartField(STANDARD_FIELD_SPECS["QNCCN"], lambda state: state.Nn),
    # v0.17 ADR-032 graupel/hail substrate prognostics.
    StandardRestartField(STANDARD_FIELD_SPECS["QHAIL"], lambda state: state.qh),
    StandardRestartField(STANDARD_FIELD_SPECS["QNHAIL"], lambda state: state.Nh),
    StandardRestartField(STANDARD_FIELD_SPECS["QVGRAUPEL"], lambda state: state.qvolg),
    StandardRestartField(STANDARD_FIELD_SPECS["QVHAIL"], lambda state: state.qvolh),
    # v0.16 aerosol-aware Thompson (mp=28) prognostic aerosol numbers.
    StandardRestartField(STANDARD_FIELD_SPECS["QNWFA"], lambda state: state.nwfa),
    StandardRestartField(STANDARD_FIELD_SPECS["QNIFA"], lambda state: state.nifa),
    StandardRestartField(STANDARD_FIELD_SPECS["QKE"], lambda state: state.qke),
    StandardRestartField(STANDARD_FIELD_SPECS["UST"], lambda state: state.ustar),
    StandardRestartField(STANDARD_FIELD_SPECS["TSK"], lambda state: state.t_skin),
    StandardRestartField(STANDARD_FIELD_SPECS["XLAND"], lambda state: state.xland),
    StandardRestartField(STANDARD_FIELD_SPECS["MAVAIL"], lambda state: state.mavail),
    StandardRestartField(STANDARD_FIELD_SPECS["ZNT"], lambda state: state.roughness_m),
    StandardRestartField(STANDARD_FIELD_SPECS["LU_INDEX"], lambda state: state.lu_index),
    StandardRestartField(STANDARD_FIELD_SPECS["RAINC"], lambda state: state.rainc_acc),
    StandardRestartField(STANDARD_FIELD_SPECS["RAINNC"], lambda state: state.rain_acc),
    StandardRestartField(STANDARD_FIELD_SPECS["SNOWNC"], lambda state: jnp.asarray(state.snow_acc) + jnp.asarray(state.ice_acc)),
    StandardRestartField(STANDARD_FIELD_SPECS["GRAUPELNC"], lambda state: state.graupel_acc),
    # v0.17 hail microphysics: WRF HAILNC (accumulated grid-scale hail, mm).
    StandardRestartField(STANDARD_FIELD_SPECS["HAILNC"], lambda state: state.hail_acc),
)


STATE_EXACT_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "u": U_XYZ,
    "v": V_XYZ,
    "w": W_XYZ,
    "theta": XYZ,
    "qv": XYZ,
    "p": XYZ,
    "p_total": XYZ,
    "p_perturbation": XYZ,
    "ph": Z_XYZ,
    "ph_total": Z_XYZ,
    "ph_perturbation": Z_XYZ,
    "mu": XY,
    "mu_total": XY,
    "mu_perturbation": XY,
    "qc": XYZ,
    "qr": XYZ,
    "qi": XYZ,
    "qs": XYZ,
    "qg": XYZ,
    "Ni": XYZ,
    "Nr": XYZ,
    "Ns": XYZ,
    "Ng": XYZ,
    "qke": XYZ,
    "ustar": XY,
    "theta_flux": XY,
    "qv_flux": XY,
    "tau_u": XY,
    "tau_v": XY,
    "rhosfc": XY,
    "fltv": XY,
    "t_skin": XY,
    "soil_moisture": XY,
    "xland": XY,
    "lakemask": XY,
    "mavail": XY,
    "roughness_m": XY,
    "rain_acc": XY,
    "snow_acc": XY,
    "graupel_acc": XY,
    "ice_acc": XY,
    "lu_index": XY,
    "Nc": XYZ,
    "Nn": XYZ,
    "rainc_acc": XY,
    # v0.15 MYNN SGS-cloud leaves (closure-2.6 qsq + mym_condensation cloud).
    "qsq": XYZ,
    "qc_bl": XYZ,
    "qi_bl": XYZ,
    "cldfra_bl": XYZ,
    # v0.17 ADR-032 graupel/hail substrate (QHAIL/QNHAIL/QVGRAUPEL/QVHAIL).
    "qh": XYZ,
    "Nh": XYZ,
    "qvolg": XYZ,
    "qvolh": XYZ,
    # v0.16 aerosol-aware Thompson (mp=28) QNWFA/QNIFA prognostics.
    "nwfa": XYZ,
    "nifa": XYZ,
    # v0.17 hail surface-precip accumulator (HAILNC).
    "hail_acc": XY,
    "u_bdy": ("Time", "gpuwrf_u_bdy_time", BDY_SIDE, BDY_WIDTH, "bottom_top", BDY_SIDE_INDEX),
    "v_bdy": ("Time", "gpuwrf_v_bdy_time", BDY_SIDE, BDY_WIDTH, "bottom_top", BDY_SIDE_INDEX),
    "theta_bdy": ("Time", "gpuwrf_theta_bdy_time", BDY_SIDE, BDY_WIDTH, "bottom_top", BDY_SIDE_INDEX),
    "qv_bdy": ("Time", "gpuwrf_qv_bdy_time", BDY_SIDE, BDY_WIDTH, "bottom_top", BDY_SIDE_INDEX),
    "p_bdy": ("Time", "gpuwrf_p_bdy_time", BDY_SIDE, BDY_WIDTH, "bottom_top", BDY_SIDE_INDEX),
    "pb_bdy": ("Time", "gpuwrf_pb_bdy_time", BDY_SIDE, BDY_WIDTH, "bottom_top", BDY_SIDE_INDEX),
    "ph_bdy": ("Time", "gpuwrf_ph_bdy_time", BDY_SIDE, BDY_WIDTH, "bottom_top_stag", BDY_SIDE_INDEX),
    "w_bdy": ("Time", "gpuwrf_w_bdy_time", BDY_SIDE, BDY_WIDTH, "bottom_top_stag", BDY_SIDE_INDEX),
    "phb_bdy": ("Time", "gpuwrf_phb_bdy_time", BDY_SIDE, BDY_WIDTH, "bottom_top_stag", BDY_SIDE_INDEX),
    "mu_bdy": ("Time", "gpuwrf_mu_bdy_time", BDY_SIDE, BDY_WIDTH, BDY_SURFACE, BDY_SIDE_INDEX),
    "mub_bdy": ("Time", "gpuwrf_mub_bdy_time", BDY_SIDE, BDY_WIDTH, BDY_SURFACE, BDY_SIDE_INDEX),
}

CARRY_EXACT_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "t_2ave": XYZ,
    "ww": W_XYZ,
    "mudf": XY,
    "muave": XY,
    "muts": XY,
    "ph_tend": Z_XYZ,
    "u_save": U_XYZ,
    "v_save": V_XYZ,
    "w_save": W_XYZ,
    "t_save": XYZ,
    "ph_save": Z_XYZ,
    "mu_save": XY,
    "ww_save": W_XYZ,
    "rthraten": XYZ,
}

NOAHMP_LAND_EXACT_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "tslb": SOIL,
    "smois": SOIL,
    "sh2o": SOIL,
    "smcwtd": XY,
    "isnow": XY,
    "tsno": SNOW,
    "snice": SNOW,
    "snliq": SNOW,
    "zsnso": SNSO,
    "snowh": XY,
    "sneqv": XY,
    "sneqvo": XY,
    "tauss": XY,
    "albold": XY,
    "tv": XY,
    "tg": XY,
    "tah": XY,
    "eah": XY,
    "canliq": XY,
    "canice": XY,
    "fwet": XY,
    "lai": XY,
    "sai": XY,
    "cm": XY,
    "ch": XY,
    "t_skin": XY,
    "qsfc": XY,
    "znt": XY,
    "emiss": XY,
    "albedo": XY,
    "sfcrunoff": XY,
    "udrunoff": XY,
}

NOAHCLASSIC_LAND_EXACT_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "t1": XY,
    "stc": SOIL_TRAILING,
    "smc": SOIL_TRAILING,
    "sh2o": SOIL_TRAILING,
    "cmc": XY,
    "sneqv": XY,
    "snowh": XY,
    "sncovr": XY,
    "snotime1": XY,
    "ribb": XY,
    "flx4": XY,
    "fvb": XY,
    "fbur": XY,
    "fgsn": XY,
    "smcrel": SOIL_TRAILING,
    "xlaidyn": XY,
    "hfx": XY,
    "qfx": XY,
    "lh": XY,
    "grdflx": XY,
}

RAD_FIELD_NAMES: tuple[str, ...] = ("soldn", "lwdn", "cosz")


def write_wrfrst_state(
    state: State,
    grid: Any,
    namelist: Mapping[str, Any] | Any | None,
    path: str | Path,
    *,
    valid_time: datetime | date | str,
    run_start: datetime | date | str,
    step_index: int,
    lead_hours: float | None = None,
    stochastic_seed_arrays: Mapping[str, Any] | None = None,
) -> Path:
    """Write one WRF-style NetCDF restart containing an exact gpuwrf State."""

    return _write_wrfrst(
        state=state,
        carry=None,
        grid=grid,
        namelist=namelist,
        path=path,
        valid_time=valid_time,
        run_start=run_start,
        step_index=int(step_index),
        lead_hours=lead_hours,
        stochastic_seed_arrays=stochastic_seed_arrays,
    )


def write_wrfrst_carry(
    carry: OperationalCarry,
    grid: Any,
    namelist: Mapping[str, Any] | Any | None,
    path: str | Path,
    *,
    valid_time: datetime | date | str,
    run_start: datetime | date | str,
    step_index: int,
    lead_hours: float | None = None,
    stochastic_seed_arrays: Mapping[str, Any] | None = None,
) -> Path:
    """Write a restart with exact State plus promoted operational scan carry."""

    return _write_wrfrst(
        state=carry.state,
        carry=carry,
        grid=grid,
        namelist=namelist,
        path=path,
        valid_time=valid_time,
        run_start=run_start,
        step_index=int(step_index),
        lead_hours=lead_hours,
        stochastic_seed_arrays=stochastic_seed_arrays,
    )


def read_wrfrst_state(path: str | Path) -> tuple[State, dict[str, Any]]:
    """Read an exact gpuwrf State from a gpuwrf-authored NetCDF wrfrst."""

    target = Path(path)
    with Dataset(target, "r") as dataset:
        _validate_common_schema(dataset, require_carry=False)
        field_order = _state_field_order_from_dataset(dataset)
        expected_shapes = _expected_state_shapes_from_dataset(dataset)
        fields: dict[str, Any] = {}
        for leaf in field_order:
            var_name = state_extension_name(leaf)
            fields[leaf] = jnp.asarray(_read_exact_variable(dataset, var_name, expected_shapes[leaf]))
        metadata = _read_metadata(dataset)
    return State(**fields), metadata


def read_wrfrst_carry(path: str | Path) -> tuple[OperationalCarry, dict[str, Any]]:
    """Read an exact gpuwrf OperationalCarry from a gpuwrf-authored wrfrst."""

    target = Path(path)
    with Dataset(target, "r") as dataset:
        _validate_common_schema(dataset, require_carry=True)
        field_order = _state_field_order_from_dataset(dataset)
        expected_shapes = _expected_state_shapes_from_dataset(dataset)
        fields: dict[str, Any] = {}
        for leaf in field_order:
            fields[leaf] = jnp.asarray(_read_exact_variable(dataset, state_extension_name(leaf), expected_shapes[leaf]))
        state = State(**fields)
        carry_fields: dict[str, Any] = {"state": state}
        for name in CARRY_ARRAY_FIELDS:
            shape = _shape_for_dimensions(CARRY_EXACT_DIMENSIONS[name], _dataset_dimension_sizes(dataset))
            carry_fields[name] = jnp.asarray(_read_exact_variable(dataset, carry_extension_name(name), shape))
        carry_fields.update(_read_optional_carry_groups(dataset))
        metadata = _read_metadata(dataset)
    return OperationalCarry(**carry_fields), metadata


def read_wrfrst_stochastic_seeds(path: str | Path) -> dict[str, jnp.ndarray]:
    """Read WRF stochastic-physics restart seed arrays from a gpuwrf wrfrst."""

    target = Path(path)
    with Dataset(target, "r") as dataset:
        _validate_common_schema(dataset, require_carry=False)
        return _read_stochastic_seed_arrays(dataset)


def inspect_wrfrst_schema(path: str | Path) -> dict[str, Any]:
    """Return a compact schema summary used by proof generation."""

    with Dataset(path, "r") as dataset:
        return {
            "dimensions": {name: int(len(dim)) for name, dim in dataset.dimensions.items()},
            "variables": {
                name: {
                    "dimensions": list(var.dimensions),
                    "shape": [int(dim) for dim in var.shape],
                    "dtype": str(np.dtype(var.dtype)),
                    "FieldType": int(getattr(var, "FieldType", -1)),
                    "MemoryOrder": str(getattr(var, "MemoryOrder", "")),
                    "stagger": str(getattr(var, "stagger", "")),
                    "units": str(getattr(var, "units", "")),
                }
                for name, var in dataset.variables.items()
            },
            "global_attrs": {name: _json_safe_attr(dataset.getncattr(name)) for name in dataset.ncattrs()},
        }


def state_extension_name(leaf: str) -> str:
    return STATE_EXTENSION_PREFIX + _extension_token(leaf)


def carry_extension_name(field: str) -> str:
    return CARRY_EXTENSION_PREFIX + _extension_token(field)


def noahmp_land_extension_name(field: str) -> str:
    return NOAHMP_LAND_EXTENSION_PREFIX + _extension_token(field)


def noahmp_rad_extension_name(field: str) -> str:
    return NOAHMP_RAD_EXTENSION_PREFIX + _extension_token(field)


def cumulus_extension_name(field: str) -> str:
    return CUMULUS_EXTENSION_PREFIX + _extension_token(field)


def noahclassic_land_extension_name(field: str) -> str:
    return NOAHCLASSIC_LAND_EXTENSION_PREFIX + _extension_token(field)


def noahclassic_rad_extension_name(field: str) -> str:
    return NOAHCLASSIC_RAD_EXTENSION_PREFIX + _extension_token(field)


def _extension_token(name: str) -> str:
    return "".join(ch.upper() if ch.isalnum() else "_" for ch in name)


def _write_wrfrst(
    *,
    state: State,
    carry: OperationalCarry | None,
    grid: Any,
    namelist: Mapping[str, Any] | Any | None,
    path: str | Path,
    valid_time: datetime | date | str,
    run_start: datetime | date | str,
    step_index: int,
    lead_hours: float | None,
    stochastic_seed_arrays: Mapping[str, Any] | None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    run_start_dt = _coerce_datetime(run_start)
    valid_dt = _coerce_datetime(valid_time)
    if lead_hours is None:
        lead_hours = (valid_dt - run_start_dt).total_seconds() / 3600.0
    nx, ny, nz = _grid_extent(grid)
    dimensions = _restart_dimension_sizes(state=state, nx=nx, ny=ny, nz=nz, namelist=namelist)
    coordinate_fields = _grid_coordinate_payload(state, grid, namelist, dimensions)
    seed_arrays = _normalize_stochastic_seed_arrays(stochastic_seed_arrays)

    with Dataset(target, "w", format="NETCDF4") as dataset:
        _create_restart_dimensions(dataset, dimensions)
        _write_global_attrs(dataset, grid, namelist, dimensions, run_start_dt, valid_dt)
        _write_restart_global_attrs(dataset, state, carry, step_index, seed_arrays)
        _write_times(dataset, valid_dt)
        _write_xtime_restart(dataset, run_start_dt, lead_hours)
        _write_itimestep(dataset, step_index)

        for name in ("XLAT", "XLONG", "XLAT_U", "XLONG_U", "XLAT_V", "XLONG_V", "HGT", "LANDMASK"):
            _write_restart_variable(dataset, STANDARD_FIELD_SPECS[name], coordinate_fields[name], dimensions)
        for name in ("ZNU", "ZNW", "P_TOP", "MAPFAC_M", "MAPFAC_U", "MAPFAC_V", "F", "E", "SINALPHA", "COSALPHA"):
            if name in coordinate_fields:
                _write_restart_variable(dataset, STANDARD_FIELD_SPECS[name], coordinate_fields[name], dimensions)

        for field in STANDARD_RESTART_FIELDS:
            value = field.value(state)
            if value is None:
                continue
            _write_restart_variable(dataset, field.spec, value, dimensions)

        if carry is not None and carry.noahmp_land is not None:
            _write_noahmp_wrf_restart_variables(dataset, carry.noahmp_land, dimensions)
        _write_stochastic_seed_variables(dataset, seed_arrays, dimensions)

        for leaf in state.active_field_names():
            _write_exact_state_variable(dataset, leaf, getattr(state, leaf), dimensions)

        if carry is not None:
            for name in CARRY_ARRAY_FIELDS:
                _write_exact_carry_variable(dataset, name, getattr(carry, name), dimensions)
            _write_optional_carry_variables(dataset, carry, dimensions)
    return target


def _restart_dimension_sizes(
    *,
    state: State,
    nx: int,
    ny: int,
    nz: int,
    namelist: Mapping[str, Any] | Any | None,
) -> dict[str, int | None]:
    dimensions = _dimension_sizes(nx=nx, ny=ny, nz=nz, namelist=namelist)
    bdy_shape = np.asarray(state.u_bdy).shape
    if len(bdy_shape) != 5:
        raise ValueError(f"u_bdy must be 5-D, got {bdy_shape}")
    dimensions.update(
        {
            BDY_TIME: int(bdy_shape[0]),
            BDY_SIDE: int(bdy_shape[1]),
            BDY_WIDTH: int(bdy_shape[2]),
            BDY_SIDE_INDEX: int(bdy_shape[-1]),
            BDY_SURFACE: 1,
        }
    )
    for leaf in STATE_FIELD_ORDER:
        if leaf.endswith("_bdy"):
            dimensions[_bdy_time_dimension(leaf)] = int(np.asarray(getattr(state, leaf)).shape[0])
    return dimensions


def _create_restart_dimensions(dataset: Dataset, dimensions: Mapping[str, int | None]) -> None:
    _create_dimensions(dataset, dimensions)
    for name in (BDY_TIME, BDY_SIDE, BDY_WIDTH, BDY_SIDE_INDEX, BDY_SURFACE):
        dataset.createDimension(name, dimensions[name])
    for leaf in STATE_FIELD_ORDER:
        if leaf.endswith("_bdy"):
            name = _bdy_time_dimension(leaf)
            if name not in dataset.dimensions:
                dataset.createDimension(name, dimensions[name])


def _bdy_time_dimension(leaf: str) -> str:
    return f"gpuwrf_{leaf}_time"


def _write_restart_global_attrs(
    dataset: Dataset,
    state: State,
    carry: OperationalCarry | None,
    step_index: int,
    stochastic_seed_arrays: Mapping[str, np.ndarray],
) -> None:
    active_state_order = state.active_field_names()
    active_standard_variables = [
        name for name in WRF_STANDARD_RESTART_VARIABLES
        if name in _ALWAYS_WRITTEN_STANDARD_RESTART_VARIABLES
    ]
    active_standard_variables.extend(
        field.spec.name
        for field in STANDARD_RESTART_FIELDS
        if field.value(state) is not None
    )
    dataset.TITLE = "OUTPUT FROM GPUWRF WRF-COMPATIBLE NETCDF RESTART"
    dataset.RESTART_STATUS = "RESTART"
    dataset.GPUWRF_WRFRST_SCHEMA_VERSION = SCHEMA_VERSION
    dataset.GPUWRF_STATE_FIELD_ORDER = json.dumps(list(active_state_order), separators=(",", ":"))
    dataset.GPUWRF_STATE_FIELD_COUNT = np.int32(len(active_state_order))
    dataset.GPUWRF_STANDARD_RESTART_VARIABLES = json.dumps(active_standard_variables, separators=(",", ":"))
    dataset.GPUWRF_OPTIONAL_WRF_RESTART_VARIABLES = json.dumps(list(OPTIONAL_WRF_RESTART_VARIABLES), separators=(",", ":"))
    dataset.GPUWRF_STOCHASTIC_SEED_VARIABLE_ORDER = json.dumps(
        list(stochastic_seed_arrays),
        separators=(",", ":"),
    )
    dataset.GPUWRF_EXACT_STATE_VARIABLE_PREFIX = STATE_EXTENSION_PREFIX
    dataset.GPUWRF_CARRY_FIELD_ORDER = json.dumps(list(CARRY_ARRAY_FIELDS), separators=(",", ":"))
    dataset.GPUWRF_CARRY_PRESENT = np.int32(1 if carry is not None else 0)
    dataset.GPUWRF_OPTIONAL_CARRY_FIELD_ORDER = json.dumps(
        _optional_carry_field_order(carry),
        sort_keys=True,
        separators=(",", ":"),
    )
    dataset.GPUWRF_OPTIONAL_CARRY_KIND = json.dumps(
        _optional_carry_kind(carry),
        sort_keys=True,
        separators=(",", ":"),
    )
    dataset.GPUWRF_UNSUPPORTED_REGISTRY_RESTART_FIELDS = json.dumps(
        list(DEFERRED_REGISTRY_RESTART_FIELDS),
        separators=(",", ":"),
    )
    dataset.GPUWRF_BDY_SIDE_ORDER = "W,E,S,N"
    dataset.GPUWRF_STEP_INDEX = np.int32(step_index)
    dataset.history = "Created by gpuwrf.io.wrfrst_netcdf"
    dataset.setncattr(
        "GPUWRF_STATE_LEAF_DTYPES",
        json.dumps(
            {leaf: str(np.asarray(getattr(state, leaf)).dtype) for leaf in active_state_order},
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _write_xtime_restart(dataset: Dataset, run_start: datetime, lead_hours: float) -> None:
    source = WRFOUT_VARIABLE_SPECS["XTIME"]
    spec = _spec(
        "XTIME",
        source.dimensions,
        source.memory_order,
        source.description,
        source.units,
        dtype="f8",
    )
    variable = dataset.createVariable("XTIME", "f8", spec.dimensions)
    _set_variable_attrs(variable, _as_wrfout_spec(spec, dtype="f8"))
    variable[0] = np.float64(float(lead_hours) * 60.0)
    label = run_start.strftime("%Y-%m-%d %H:%M:%S")
    variable.description = f"minutes since {label}"
    variable.units = f"minutes since {label}"


def _write_itimestep(dataset: Dataset, step_index: int) -> None:
    spec = WRFRST_EXTRA_SPECS["ITIMESTEP"]
    variable = dataset.createVariable("ITIMESTEP", "i4", spec.dimensions)
    _set_variable_attrs(variable, _as_wrfout_spec(spec, dtype="i4"))
    variable[0] = np.int32(step_index)


def _grid_coordinate_payload(
    state: State,
    grid: Any,
    namelist: Mapping[str, Any] | Any | None,
    dimensions: Mapping[str, int | None],
) -> dict[str, np.ndarray]:
    shape_xy = _shape_for_dimensions(XY, dimensions)
    shape_u_xy = _shape_for_dimensions(U_XY, dimensions)
    shape_v_xy = _shape_for_dimensions(V_XY, dimensions)
    shape_soil = _shape_for_dimensions(SOIL, dimensions)
    del shape_soil

    xlat, xlong = _latlon_fields(state, grid, namelist, shape_xy)
    xlat_u, xlong_u = _latlon_fields(state, grid, namelist, shape_u_xy, suffix="_u")
    xlat_v, xlong_v = _latlon_fields(state, grid, namelist, shape_v_xy, suffix="_v")
    landmask = _landmask(state, shape_xy)
    fields: dict[str, np.ndarray] = {
        "XLAT": xlat,
        "XLONG": xlong,
        "XLAT_U": xlat_u,
        "XLONG_U": xlong_u,
        "XLAT_V": xlat_v,
        "XLONG_V": xlong_v,
        "HGT": _grid_or_state_array(state, grid, ("HGT", "hgt", "terrain_height"), shape_xy),
        "LANDMASK": landmask,
    }
    _add_grid_coordinate_fields(
        fields,
        grid,
        dimensions,
        shape_xy=shape_xy,
        shape_mapu=_shape_for_dimensions(MAPFAC_U_XY, dimensions),
        shape_mapv=_shape_for_dimensions(MAPFAC_V_XY, dimensions),
        landmask=landmask,
    )
    return fields


def _write_restart_variable(
    dataset: Dataset,
    spec: RestartVariableSpec,
    data: Any,
    dimensions: Mapping[str, int | None],
) -> None:
    expected_shape = _shape_for_dimensions(spec.dimensions, dimensions)
    array = _coerce_preserving_dtype(spec.name, data, expected_shape)
    dtype = spec.dtype or _netcdf_dtype(array.dtype)
    variable = dataset.createVariable(spec.name, dtype, spec.dimensions)
    _set_variable_attrs(variable, _as_wrfout_spec(spec, dtype=dtype))
    _assign_variable(variable, spec.dimensions, array)


def _write_exact_state_variable(
    dataset: Dataset,
    leaf: str,
    data: Any,
    dimensions: Mapping[str, int | None],
) -> None:
    dims = STATE_EXACT_DIMENSIONS[leaf]
    stagger = _stagger_for_dimensions(dims)
    spec = _spec(
        state_extension_name(leaf),
        dims,
        _memory_order_for_dimensions(dims),
        f"gpuwrf exact State leaf {leaf}",
        _units_for_leaf(leaf),
        stagger=stagger,
        coordinates=_coordinates_for_dimensions(dims),
    )
    _write_restart_variable(dataset, spec, data, dimensions)
    dataset.variables[spec.name].gpuwrf_state_leaf = leaf


def _write_exact_carry_variable(
    dataset: Dataset,
    name: str,
    data: Any,
    dimensions: Mapping[str, int | None],
) -> None:
    dims = CARRY_EXACT_DIMENSIONS[name]
    spec = _spec(
        carry_extension_name(name),
        dims,
        _memory_order_for_dimensions(dims),
        f"gpuwrf exact OperationalCarry field {name}",
        _units_for_carry(name),
        stagger=_stagger_for_dimensions(dims),
        coordinates=_coordinates_for_dimensions(dims),
    )
    _write_restart_variable(dataset, spec, data, dimensions)
    dataset.variables[spec.name].gpuwrf_carry_field = name


def _write_noahmp_wrf_restart_variables(
    dataset: Dataset,
    land_state: Any,
    dimensions: Mapping[str, int | None],
) -> None:
    """Write WRF-named Noah-MP land/snow restart variables from the land carry."""

    values: dict[str, Any] = {
        "TSLB": getattr(land_state, "tslb"),
        "SMOIS": getattr(land_state, "smois"),
        "SH2O": getattr(land_state, "sh2o"),
        "TSNO": getattr(land_state, "tsno"),
        "SNICE": getattr(land_state, "snice"),
        "SNLIQ": getattr(land_state, "snliq"),
        "ZSNSO": getattr(land_state, "zsnso"),
        "SNOW": getattr(land_state, "sneqv"),
        "SNOWH": getattr(land_state, "snowh"),
        "CANWAT": jnp.asarray(getattr(land_state, "canliq")) + jnp.asarray(getattr(land_state, "canice")),
        "SFROFF": jnp.asarray(getattr(land_state, "sfcrunoff")) * 1.0e3,
        "UDROFF": jnp.asarray(getattr(land_state, "udrunoff")) * 1.0e3,
        "ALBEDO": getattr(land_state, "albedo"),
        "EMISS": getattr(land_state, "emiss"),
    }
    for name in NOAHMP_WRF_RESTART_VARIABLES:
        _write_restart_variable(dataset, NOAHMP_WRF_FIELD_SPECS[name], values[name], dimensions)


def _normalize_stochastic_seed_arrays(seed_arrays: Mapping[str, Any] | None) -> dict[str, np.ndarray]:
    if seed_arrays is None:
        return {}
    unknown = sorted(set(seed_arrays) - set(STOCHASTIC_SEED_RESTART_VARIABLES))
    if unknown:
        raise ValueError(f"unsupported stochastic seed restart variables: {unknown}")
    return {
        name: np.asarray(seed_arrays[name], dtype=np.int32)
        for name in STOCHASTIC_SEED_RESTART_VARIABLES
        if name in seed_arrays and seed_arrays[name] is not None
    }


def _write_stochastic_seed_variables(
    dataset: Dataset,
    seed_arrays: Mapping[str, np.ndarray],
    dimensions: Mapping[str, int | None],
) -> None:
    for name in seed_arrays:
        _write_restart_variable(dataset, STOCHASTIC_SEED_FIELD_SPECS[name], seed_arrays[name], dimensions)
        dataset.variables[name].gpuwrf_stochastic_seed_variable = name


def _write_optional_carry_variables(
    dataset: Dataset,
    carry: OperationalCarry,
    dimensions: Mapping[str, int | None],
) -> None:
    if carry.noahmp_land is not None:
        for field in _object_field_order(carry.noahmp_land):
            _write_optional_variable(
                dataset,
                group="noahmp_land",
                field=field,
                variable_name=noahmp_land_extension_name(field),
                data=getattr(carry.noahmp_land, field),
                dimensions=dimensions,
                dims=NOAHMP_LAND_EXACT_DIMENSIONS.get(field),
            )
    if carry.noahmp_rad is not None:
        for field, data in zip(RAD_FIELD_NAMES, tuple(carry.noahmp_rad), strict=True):
            _write_optional_variable(
                dataset,
                group="noahmp_rad",
                field=field,
                variable_name=noahmp_rad_extension_name(field),
                data=data,
                dimensions=dimensions,
                dims=XY,
            )
    for field, data in _cumulus_items(carry.cumulus_carry):
        _write_optional_variable(
            dataset,
            group="cumulus_carry",
            field=field,
            variable_name=cumulus_extension_name(field),
            data=data,
            dimensions=dimensions,
        )
    if carry.noahclassic_land is not None:
        for field in _object_field_order(carry.noahclassic_land):
            _write_optional_variable(
                dataset,
                group="noahclassic_land",
                field=field,
                variable_name=noahclassic_land_extension_name(field),
                data=getattr(carry.noahclassic_land, field),
                dimensions=dimensions,
                dims=NOAHCLASSIC_LAND_EXACT_DIMENSIONS.get(field),
            )
    if carry.noahclassic_rad is not None:
        for field, data in zip(RAD_FIELD_NAMES, tuple(carry.noahclassic_rad), strict=True):
            _write_optional_variable(
                dataset,
                group="noahclassic_rad",
                field=field,
                variable_name=noahclassic_rad_extension_name(field),
                data=data,
                dimensions=dimensions,
                dims=XY,
            )


def _write_optional_variable(
    dataset: Dataset,
    *,
    group: str,
    field: str,
    variable_name: str,
    data: Any,
    dimensions: Mapping[str, int | None],
    dims: tuple[str, ...] | None = None,
) -> None:
    dims = dims or _dimensions_for_array(data, dimensions)
    spec = _spec(
        variable_name,
        dims,
        _memory_order_for_dimensions(dims),
        f"gpuwrf exact optional OperationalCarry {group}.{field}",
        _units_for_optional(group, field),
        stagger=_stagger_for_dimensions(dims),
        coordinates=_coordinates_for_dimensions(dims),
    )
    _write_restart_variable(dataset, spec, data, dimensions)
    variable = dataset.variables[spec.name]
    variable.gpuwrf_optional_carry_group = group
    variable.gpuwrf_optional_carry_field = field


def _optional_carry_field_order(carry: OperationalCarry | None) -> dict[str, list[str]]:
    if carry is None:
        return {name: [] for name in OPTIONAL_CARRY_FIELDS}
    return {
        "noahmp_land": list(_object_field_order(carry.noahmp_land)) if carry.noahmp_land is not None else [],
        "noahmp_rad": list(RAD_FIELD_NAMES) if carry.noahmp_rad is not None else [],
        "cumulus_carry": [name for name, _ in _cumulus_items(carry.cumulus_carry)],
        "noahclassic_land": list(_object_field_order(carry.noahclassic_land)) if carry.noahclassic_land is not None else [],
        "noahclassic_rad": list(RAD_FIELD_NAMES) if carry.noahclassic_rad is not None else [],
    }


def _optional_carry_kind(carry: OperationalCarry | None) -> dict[str, str]:
    if carry is None:
        return {name: "none" for name in OPTIONAL_CARRY_FIELDS}
    cumulus_kind = "none"
    if carry.cumulus_carry is not None:
        cumulus_kind = "tuple" if isinstance(carry.cumulus_carry, tuple) else "array"
    return {
        "noahmp_land": "object" if carry.noahmp_land is not None else "none",
        "noahmp_rad": "tuple" if carry.noahmp_rad is not None else "none",
        "cumulus_carry": cumulus_kind,
        "noahclassic_land": "object" if carry.noahclassic_land is not None else "none",
        "noahclassic_rad": "tuple" if carry.noahclassic_rad is not None else "none",
    }


def _object_field_order(obj: Any) -> tuple[str, ...]:
    if obj is None:
        return ()
    slots = getattr(obj, "__slots__", ())
    if slots:
        return tuple(str(name) for name in slots)
    fields = getattr(obj, "_fields", ())
    if fields:
        return tuple(str(name) for name in fields)
    raise TypeError(f"unsupported optional carry object {type(obj).__name__}")


def _cumulus_items(carry: Any) -> list[tuple[str, Any]]:
    if carry is None:
        return []
    if isinstance(carry, tuple):
        if len(carry) == 2:
            names = ("w0avg", "nca")
        else:
            names = tuple(f"field_{idx}" for idx in range(len(carry)))
        return list(zip(names, carry, strict=True))
    return [("cldefi", carry)]


def _dimensions_for_array(data: Any, dimensions: Mapping[str, int | None]) -> tuple[str, ...]:
    shape = np.asarray(data).shape
    candidates = (XY, XYZ, U_XYZ, V_XYZ, W_XYZ, Z_XYZ, SOIL, SOIL_TRAILING, SNOW, SNSO, SEED, TIME_ONLY)
    for dims in candidates:
        if _shape_for_dimensions(dims, dimensions) == shape:
            return dims
    raise ValueError(f"optional carry array shape {shape} does not match supported wrfrst dimensions")


def _read_optional_carry_groups(dataset: Dataset) -> dict[str, Any]:
    """Read optional exact carry groups declared in the restart manifest."""

    field_order = _optional_carry_manifest(dataset)
    kind = _optional_carry_kind_manifest(dataset)
    carry_fields: dict[str, Any] = {name: None for name in OPTIONAL_CARRY_FIELDS}

    noahmp_land_fields = tuple(field_order["noahmp_land"])
    if noahmp_land_fields:
        if NoahMPLandState is None:
            raise ValueError("wrfrst contains Noah-MP land carry but NoahMPLandState is unavailable")
        expected = tuple(NoahMPLandState.__slots__)
        if noahmp_land_fields != expected:
            raise ValueError("wrfrst Noah-MP land field order does not match NoahMPLandState.__slots__")
        carry_fields["noahmp_land"] = NoahMPLandState(
            **{
                field: jnp.asarray(_read_optional_variable(dataset, noahmp_land_extension_name(field)))
                for field in noahmp_land_fields
            }
        )

    noahmp_rad_fields = tuple(field_order["noahmp_rad"])
    if noahmp_rad_fields:
        if noahmp_rad_fields != RAD_FIELD_NAMES:
            raise ValueError("wrfrst Noah-MP radiation field order does not match soldn/lwdn/cosz")
        carry_fields["noahmp_rad"] = tuple(
            jnp.asarray(_read_optional_variable(dataset, noahmp_rad_extension_name(field)))
            for field in noahmp_rad_fields
        )

    cumulus_fields = tuple(field_order["cumulus_carry"])
    if cumulus_fields:
        values = tuple(
            jnp.asarray(_read_optional_variable(dataset, cumulus_extension_name(field)))
            for field in cumulus_fields
        )
        if kind["cumulus_carry"] == "array":
            if len(values) != 1:
                raise ValueError("wrfrst array cumulus carry must contain exactly one field")
            carry_fields["cumulus_carry"] = values[0]
        elif kind["cumulus_carry"] == "tuple":
            carry_fields["cumulus_carry"] = values
        else:
            raise ValueError(f"unsupported wrfrst cumulus carry kind {kind['cumulus_carry']!r}")

    noahclassic_land_fields = tuple(field_order["noahclassic_land"])
    if noahclassic_land_fields:
        if NoahClassicLandState is None:
            raise ValueError("wrfrst contains Noah-classic land carry but NoahClassicLandState is unavailable")
        expected = tuple(NoahClassicLandState._fields)
        if noahclassic_land_fields != expected:
            raise ValueError("wrfrst Noah-classic land field order does not match NoahClassicLandState._fields")
        carry_fields["noahclassic_land"] = NoahClassicLandState(
            **{
                field: jnp.asarray(_read_optional_variable(dataset, noahclassic_land_extension_name(field)))
                for field in noahclassic_land_fields
            }
        )

    noahclassic_rad_fields = tuple(field_order["noahclassic_rad"])
    if noahclassic_rad_fields:
        if noahclassic_rad_fields != RAD_FIELD_NAMES:
            raise ValueError("wrfrst Noah-classic radiation field order does not match soldn/lwdn/cosz")
        values = tuple(
            jnp.asarray(_read_optional_variable(dataset, noahclassic_rad_extension_name(field)))
            for field in noahclassic_rad_fields
        )
        carry_fields["noahclassic_rad"] = NoahClassicRadiation(*values) if NoahClassicRadiation is not None else values

    return carry_fields


def _read_optional_variable(dataset: Dataset, name: str) -> np.ndarray:
    if name not in dataset.variables:
        raise ValueError(f"wrfrst missing optional carry variable {name}")
    variable = dataset.variables[name]
    return np.asarray(variable[0, ...] if variable.dimensions and variable.dimensions[0] == "Time" else variable[...])


def _read_stochastic_seed_arrays(dataset: Dataset) -> dict[str, jnp.ndarray]:
    seeds: dict[str, jnp.ndarray] = {}
    for name in _stochastic_seed_manifest(dataset):
        variable = dataset.variables[name]
        seeds[name] = jnp.asarray(variable[0, ...], dtype=jnp.int32)
    return seeds


def _stochastic_seed_manifest(dataset: Dataset) -> list[str]:
    raw = getattr(dataset, "GPUWRF_STOCHASTIC_SEED_VARIABLE_ORDER", "[]")
    payload = json.loads(str(raw))
    if not isinstance(payload, list) or not all(isinstance(name, str) for name in payload):
        raise ValueError("wrfrst stochastic seed manifest must be a string list")
    unsupported = sorted(set(payload) - set(STOCHASTIC_SEED_RESTART_VARIABLES))
    if unsupported:
        raise ValueError(f"wrfrst stochastic seed manifest has unsupported variables: {unsupported}")
    duplicates = sorted({name for name in payload if payload.count(name) > 1})
    if duplicates:
        raise ValueError(f"wrfrst stochastic seed manifest has duplicate variables: {duplicates}")
    return list(payload)


def _optional_carry_manifest(dataset: Dataset) -> dict[str, list[str]]:
    raw = getattr(dataset, "GPUWRF_OPTIONAL_CARRY_FIELD_ORDER", None)
    if raw is None:
        raise ValueError("wrfrst missing optional carry field-order manifest")
    payload = json.loads(str(raw))
    if not isinstance(payload, dict):
        raise ValueError("wrfrst optional carry field-order manifest is not an object")
    missing = [name for name in OPTIONAL_CARRY_FIELDS if name not in payload]
    if missing:
        raise ValueError(f"wrfrst optional carry manifest missing groups: {missing}")
    extra = sorted(set(payload) - set(OPTIONAL_CARRY_FIELDS))
    if extra:
        raise ValueError(f"wrfrst optional carry manifest has unsupported groups: {extra}")
    manifest: dict[str, list[str]] = {}
    for group in OPTIONAL_CARRY_FIELDS:
        fields = payload[group]
        if not isinstance(fields, list) or not all(isinstance(field, str) for field in fields):
            raise ValueError(f"wrfrst optional carry manifest group {group!r} must be a string list")
        manifest[group] = list(fields)
    return manifest


def _optional_carry_kind_manifest(dataset: Dataset) -> dict[str, str]:
    raw = getattr(dataset, "GPUWRF_OPTIONAL_CARRY_KIND", None)
    if raw is None:
        raise ValueError("wrfrst missing optional carry kind manifest")
    payload = json.loads(str(raw))
    if not isinstance(payload, dict):
        raise ValueError("wrfrst optional carry kind manifest is not an object")
    missing = [name for name in OPTIONAL_CARRY_FIELDS if name not in payload]
    if missing:
        raise ValueError(f"wrfrst optional carry kind manifest missing groups: {missing}")
    allowed = {"none", "object", "tuple", "array"}
    kinds: dict[str, str] = {}
    for group in OPTIONAL_CARRY_FIELDS:
        value = payload[group]
        if value not in allowed:
            raise ValueError(f"wrfrst optional carry kind {group}={value!r} is unsupported")
        kinds[group] = value
    return kinds


def _optional_variable_name(group: str, field: str) -> str:
    if group == "noahmp_land":
        return noahmp_land_extension_name(field)
    if group == "noahmp_rad":
        return noahmp_rad_extension_name(field)
    if group == "cumulus_carry":
        return cumulus_extension_name(field)
    if group == "noahclassic_land":
        return noahclassic_land_extension_name(field)
    if group == "noahclassic_rad":
        return noahclassic_rad_extension_name(field)
    raise ValueError(f"unsupported optional carry group {group!r}")


def _units_for_optional(group: str, field: str) -> str:
    if group == "noahmp_land":
        return {
            "tslb": "K",
            "tsno": "K",
            "snice": "kg m-2",
            "snliq": "kg m-2",
            "zsnso": "m",
            "snowh": "m",
            "sneqv": "kg m-2",
            "sneqvo": "kg m-2",
            "sfcrunoff": "m",
            "udrunoff": "m",
            "albedo": "-",
            "emiss": "",
        }.get(field, _units_for_leaf(field))
    if group in {"noahmp_rad", "noahclassic_rad"}:
        return "W m-2" if field in {"soldn", "lwdn"} else ""
    if group == "noahclassic_land":
        return {
            "t1": "K",
            "stc": "K",
            "smc": "m3 m-3",
            "sh2o": "m3 m-3",
            "cmc": "m",
            "sneqv": "m",
            "snowh": "m",
            "hfx": "W m-2",
            "qfx": "kg m-2 s-1",
            "lh": "W m-2",
            "grdflx": "W m-2",
        }.get(field, _units_for_leaf(field))
    return _units_for_leaf(field)


def _assign_variable(variable: Any, dimensions: tuple[str, ...], array: np.ndarray) -> None:
    if dimensions and dimensions[0] == "Time":
        variable[0, ...] = array
    else:
        variable[...] = array


def _coerce_preserving_dtype(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape == (1, *shape):
        array = array[0]
    if array.shape == shape:
        return np.array(array, copy=False)
    if array.shape == ():
        return np.full(shape, array.item(), dtype=array.dtype)
    try:
        return np.broadcast_to(array, shape).astype(array.dtype, copy=False)
    except ValueError as exc:
        raise ValueError(f"{name} shape {array.shape} cannot be written to WRF shape {shape}") from exc


def _netcdf_dtype(dtype: np.dtype[Any]) -> str:
    dtype = np.dtype(dtype)
    if dtype == np.dtype("float64"):
        return "f8"
    if dtype == np.dtype("float32"):
        return "f4"
    if dtype == np.dtype("int64"):
        return "i8"
    if dtype == np.dtype("int32"):
        return "i4"
    if dtype == np.dtype("int16"):
        return "i2"
    if dtype == np.dtype("int8"):
        return "i1"
    if dtype == np.dtype("uint8"):
        return "u1"
    raise TypeError(f"unsupported NetCDF dtype {dtype}")


def _as_wrfout_spec(spec: RestartVariableSpec, *, dtype: str) -> WrfoutVariableSpec:
    return WrfoutVariableSpec(
        name=spec.name,
        dimensions=spec.dimensions,
        memory_order=spec.memory_order,
        description=spec.description,
        units=spec.units,
        stagger=spec.stagger,
        coordinates=spec.coordinates,
        dtype=dtype,
    )


def _memory_order_for_dimensions(dimensions: tuple[str, ...]) -> str:
    dims = tuple(dim for dim in dimensions if dim != "Time")
    if dims == ("south_north", "west_east"):
        return "XY "
    if dims == ("south_north", "west_east_stag"):
        return "XY "
    if dims == ("south_north_stag", "west_east"):
        return "XY "
    if len(dims) == 3 and (
        dims[0] in {"bottom_top", "bottom_top_stag", "soil_layers_stag", "snow_layers_stag", "snso_layers_stag"}
        or dims[-1] in {"soil_layers_stag", "snow_layers_stag", "snso_layers_stag"}
    ):
        return "XYZ"
    if len(dims) >= 5 and BDY_SIDE in dims:
        return "XYZ"
    if len(dims) == 1:
        return "Z  "
    return "0  "


def _stagger_for_dimensions(dimensions: tuple[str, ...]) -> str:
    if "west_east_stag" in dimensions:
        return "X"
    if "south_north_stag" in dimensions:
        return "Y"
    if any(dim in dimensions for dim in ("bottom_top_stag", "soil_layers_stag", "snow_layers_stag", "snso_layers_stag", "seed_dim_stag")):
        return "Z"
    return ""


def _coordinates_for_dimensions(dimensions: tuple[str, ...]) -> str | None:
    if BDY_SIDE in dimensions:
        return None
    if "west_east_stag" in dimensions:
        return "XLONG_U XLAT_U XTIME"
    if "south_north_stag" in dimensions:
        return "XLONG_V XLAT_V XTIME"
    if "south_north" in dimensions and "west_east" in dimensions:
        return "XLONG XLAT XTIME"
    return None


def _units_for_leaf(leaf: str) -> str:
    if leaf in {"u", "v", "w"}:
        return "m s-1"
    if leaf in {"theta", "t_skin"}:
        return "K"
    if leaf in {"qv", "qc", "qr", "qi", "qs", "qg", "qh", "soil_moisture"}:
        return "kg kg-1" if leaf != "soil_moisture" else "m3 m-3"
    if leaf in {"qvolg", "qvolh"}:
        # v0.17 ADR-032 predicted-density particle volume.
        return "m3 kg-1"
    if leaf in {"p", "p_total", "p_perturbation", "mu", "mu_total", "mu_perturbation"}:
        return "Pa"
    if leaf in {"ph", "ph_total", "ph_perturbation"}:
        return "m2 s-2"
    if leaf in {"Ni", "Nr", "Ns", "Ng", "Nc", "Nn", "Nh", "nwfa", "nifa"}:
        return "kg-1"
    if leaf == "qke":
        return "m2 s-2"
    if leaf == "qsq":
        return "kg2 kg-2"
    if leaf in {"qc_bl", "qi_bl"}:
        return "kg kg-1"
    if leaf == "cldfra_bl":
        return ""
    if leaf in {"ustar", "roughness_m"}:
        return "m" if leaf == "roughness_m" else "m s-1"
    if leaf in {"theta_flux", "fltv"}:
        return "K m s-1"
    if leaf == "qv_flux":
        return "kg kg-1 m s-1"
    if leaf in {"tau_u", "tau_v"}:
        return "m2 s-2"
    if leaf == "rhosfc":
        return "kg m-3"
    if leaf in {"rain_acc", "snow_acc", "graupel_acc", "ice_acc", "rainc_acc", "hail_acc"}:
        return "mm"
    if leaf.endswith("_bdy"):
        return "WRF lateral boundary tendency/history units"
    return ""


def _units_for_carry(name: str) -> str:
    if name in {"t_2ave", "t_save", "rthraten"}:
        return "K" if name != "rthraten" else "K s-1"
    if name in {"ww", "ww_save"}:
        return "Pa s-1"
    if name in {"mudf", "muave", "muts", "mu_save"}:
        return "Pa"
    if name in {"ph_tend", "ph_save"}:
        return "m2 s-2"
    if name in {"u_save", "v_save", "w_save"}:
        return "m s-1"
    return ""


def _validate_common_schema(dataset: Dataset, *, require_carry: bool) -> None:
    schema = str(getattr(dataset, "GPUWRF_WRFRST_SCHEMA_VERSION", ""))
    if schema != SCHEMA_VERSION:
        raise ValueError(f"unsupported wrfrst schema {schema!r}; expected {SCHEMA_VERSION!r}")
    field_order = _state_field_order_from_dataset(dataset)
    standard_manifest = json.loads(str(getattr(dataset, "GPUWRF_STANDARD_RESTART_VARIABLES", "[]")))
    missing_standard = [
        name for name in standard_manifest
        if name != "Times" and name not in dataset.variables
    ]
    if missing_standard:
        raise ValueError(f"wrfrst missing WRF-standard restart variables: {missing_standard}")
    missing_state = [
        state_extension_name(leaf)
        for leaf in field_order
        if state_extension_name(leaf) not in dataset.variables
    ]
    if missing_state:
        raise ValueError(f"wrfrst missing exact gpuwrf State variables: {missing_state}")
    carry_present = int(getattr(dataset, "GPUWRF_CARRY_PRESENT", 0))
    if require_carry and carry_present != 1:
        raise ValueError("wrfrst does not contain exact gpuwrf carry variables")
    if require_carry:
        missing_carry = [
            carry_extension_name(name)
            for name in CARRY_ARRAY_FIELDS
            if carry_extension_name(name) not in dataset.variables
        ]
        if missing_carry:
            raise ValueError(f"wrfrst missing exact gpuwrf carry variables: {missing_carry}")
    for name in _stochastic_seed_manifest(dataset):
        if name not in dataset.variables:
            raise ValueError(f"wrfrst missing stochastic seed variable {name}")
        variable = dataset.variables[name]
        if tuple(variable.dimensions) != SEED:
            raise ValueError(f"wrfrst stochastic seed variable {name} dimensions {variable.dimensions} != {SEED}")
        if np.dtype(variable.dtype) != np.dtype("int32"):
            raise ValueError(f"wrfrst stochastic seed variable {name} dtype {np.dtype(variable.dtype)} != int32")
    optional_order = _optional_carry_manifest(dataset)
    optional_kind = _optional_carry_kind_manifest(dataset)
    for group, fields in optional_order.items():
        if fields and optional_kind[group] == "none":
            raise ValueError(f"wrfrst optional carry group {group} declares fields but kind=none")
        if not fields and optional_kind[group] != "none":
            raise ValueError(f"wrfrst optional carry group {group} declares kind={optional_kind[group]!r} but no fields")
        missing_optional = [
            _optional_variable_name(group, field)
            for field in fields
            if _optional_variable_name(group, field) not in dataset.variables
        ]
        if missing_optional:
            raise ValueError(f"wrfrst missing exact optional carry variables: {missing_optional}")


def _expected_state_shapes_from_dataset(dataset: Dataset) -> dict[str, tuple[int, ...]]:
    dimensions = _dataset_dimension_sizes(dataset)
    field_order = _state_field_order_from_dataset(dataset)
    return {
        leaf: _shape_for_dimensions(STATE_EXACT_DIMENSIONS[leaf], dimensions)
        for leaf in field_order
    }


def _dataset_dimension_sizes(dataset: Dataset) -> dict[str, int | None]:
    return {name: int(len(dim)) for name, dim in dataset.dimensions.items()}


def _read_exact_variable(dataset: Dataset, name: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    if name not in dataset.variables:
        raise ValueError(f"wrfrst missing variable {name}")
    variable = dataset.variables[name]
    data = np.asarray(variable[0, ...] if variable.dimensions and variable.dimensions[0] == "Time" else variable[...])
    if data.shape != expected_shape:
        raise ValueError(f"{name} shape {data.shape} != expected {expected_shape}")
    return data


def _read_metadata(dataset: Dataset) -> dict[str, Any]:
    valid_text = "".join(ch.decode("ascii") for ch in np.asarray(dataset.variables["Times"][0], dtype="S1"))
    return {
        "schema_version": str(getattr(dataset, "GPUWRF_WRFRST_SCHEMA_VERSION")),
        "state_field_order": json.loads(str(getattr(dataset, "GPUWRF_STATE_FIELD_ORDER"))),
        "standard_restart_variables": json.loads(str(getattr(dataset, "GPUWRF_STANDARD_RESTART_VARIABLES"))),
        "optional_wrf_restart_variables": json.loads(str(getattr(dataset, "GPUWRF_OPTIONAL_WRF_RESTART_VARIABLES", "[]"))),
        "stochastic_seed_variables": _stochastic_seed_manifest(dataset),
        "optional_carry_field_order": _optional_carry_manifest(dataset),
        "optional_carry_kind": _optional_carry_kind_manifest(dataset),
        "deferred_registry_restart_fields": json.loads(str(getattr(dataset, "GPUWRF_UNSUPPORTED_REGISTRY_RESTART_FIELDS"))),
        "carry_present": bool(int(getattr(dataset, "GPUWRF_CARRY_PRESENT", 0))),
        "step_index": int(getattr(dataset, "GPUWRF_STEP_INDEX", -1)),
        "valid_time": valid_text,
        "xtime_minutes": float(np.asarray(dataset.variables["XTIME"][0])),
    }


def _json_safe_attr(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


__all__ = [
    "CARRY_ARRAY_FIELDS",
    "DEFERRED_REGISTRY_RESTART_FIELDS",
    "SCHEMA_VERSION",
    "STATE_FIELD_ORDER",
    "WRF_STANDARD_RESTART_VARIABLES",
    "carry_extension_name",
    "cumulus_extension_name",
    "inspect_wrfrst_schema",
    "noahclassic_land_extension_name",
    "noahclassic_rad_extension_name",
    "noahmp_land_extension_name",
    "noahmp_rad_extension_name",
    "read_wrfrst_carry",
    "read_wrfrst_state",
    "read_wrfrst_stochastic_seeds",
    "state_extension_name",
    "STOCHASTIC_SEED_RESTART_VARIABLES",
    "write_wrfrst_carry",
    "write_wrfrst_state",
]
