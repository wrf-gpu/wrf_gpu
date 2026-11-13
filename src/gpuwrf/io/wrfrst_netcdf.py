"""GPUWRF NetCDF checkpoint with WRF-convention inspection fields.

The file carries two layers:

* WRF-named restart variables with WRF dimensions, staggering and attributes.
* Exact ``GPUWRF_*`` extension variables for every ``State`` leaf and promoted
  operational scratch field.  The extensions make gpuwrf resume fail-closed and
  bitwise, while WRF tools can inspect the WRF-convention fields. This is not a
  complete native CPU-WRF restart: previous time levels and unsupported Registry
  fields are absent. GPUWRF readers use only the exact extension payload.
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields, replace as dataclass_replace
from datetime import date, datetime
import json
import jax
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

import jax.numpy as jnp
import numpy as np
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.pbl_epsilon_schema import (EPSILON_PBL_SCHEMA_VERSION,epsilon_pbl_active,epsilon_pbl_config,validate_epsilon_state,validate_epsilon_history)

from gpuwrf.contracts.ice_diagnostics import ICE_DIAGNOSTIC_LEAVES, ice_diagnostic_leaves
from gpuwrf.contracts.state import CONDITIONAL_STATE_LEAVES, State, mynn_sfc_wspd_enabled
from gpuwrf.contracts.precision import (
    EXTRA_SCALAR_BOUNDARY_LEAVES, GWDO_DIAGNOSTIC_LEAVES, LES_TKE_LEAVES,
    MP_RE_DIAGNOSTIC_LEAVES, SMS_CARRY_LEAVES, EPSILON_PBL_LEAVES, les_tke_active,
    MYNN_CLOUDPDF_LEAVES, mynn_cloudpdf_leaves,
)
from gpuwrf.io.wrfout_writer import (
    DATE_STR_LEN,
    RVOVRD,
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
    RESTART_WRFOUT_VARIABLE_SPECS as WRFOUT_VARIABLE_SPECS,
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
from gpuwrf.io.wrfout_theta_convention import parse_use_theta_m, thm_source
from gpuwrf.io.restart_schema import ssib_active
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


from gpuwrf.contracts.cammg_fields import STATE_FIELDS as CAMMG_STATE_FIELDS, MASS_FIELDS as CAMMG_MASS_FIELDS, INTERFACE_FIELDS as CAMMG_INTERFACE_FIELDS, SURFACE_FIELDS as CAMMG_SURFACE_FIELDS
from gpuwrf.physics.cammg.restart import validate as validate_cammg

SCHEMA_VERSION = "v0.25-wrfrst-netcdf-5"
LEGACY_SCHEMA_VERSION = "v0.11.0-wrfrst-netcdf-2"
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

from gpuwrf.io.restart_schema import (
    CAMUW_SCHEMA_VERSION, effective_radii_required, effective_radius_fields, physics_option, validate_camuw, validate_held_state,
    LAKE_SCHEMA_VERSION,validate_lake,
    TEMF_SCHEMA_VERSION,validate_temf,
    QNSE_SCHEMA_VERSION,validate_qnse,
    GODDARD_RADIATION_SCHEMA_VERSION,
)


def _validate_state_field_order(field_order: tuple[str, ...], *, namelist=None) -> None:
    if any(field not in STATE_FIELD_ORDER for field in field_order):
        raise ValueError("wrfrst State field order contains unknown leaves")
    if field_order != tuple(field for field in STATE_FIELD_ORDER if field in field_order):
        raise ValueError("wrfrst State field order does not match current State.__slots__")
    missing = tuple(field for field in STATE_FIELD_ORDER if field not in field_order)
    if physics_option(namelist, "mp_physics", 8) == 13 and "rimi" in missing:
        raise ValueError("wrfrst lacks held SBU RIMI (E78)")
    if physics_option(namelist, "bl_pbl_physics", 5) == 12 and "exch_tke" in missing:
        raise ValueError("wrfrst lacks held GBM EXCH_TKE (E78)")
    if mynn_sfc_wspd_enabled() and "sfc_wspd" in missing:
        raise ValueError("wrfrst State schema lacks held MYNN sfc_wspd (E78)")
    if effective_radii_required(namelist) and any(field in effective_radius_fields(namelist) for field in missing):
        raise ValueError("MP_RE wrfrst is missing held Thompson effective radii (E78)")
    if les_tke_active(namelist) and any(field in LES_TKE_LEAVES for field in missing):
        raise ValueError("km_opt=2 wrfrst is missing the prognostic LES TKE (WRF tke_2) (E78)")
    if any(field in missing for field in mynn_cloudpdf_leaves(namelist)):
        raise ValueError("bl_mynn_cloudpdf wrfrst is missing held MYNN TSQ/COV/SH3D (E78)")
    if physics_option(namelist, "km_opt") == 5 and any(f in SMS_CARRY_LEAVES for f in missing):
        raise ValueError("km_opt=5 wrfrst is missing held SMS carry (E78)")
    if any(name in missing for name in ice_diagnostic_leaves(physics_option(namelist, "mp_physics", 8))):
        raise ValueError("wrfrst is missing held ice diagnostics (E78)")
    optional = (*CONDITIONAL_STATE_LEAVES, *GWDO_DIAGNOSTIC_LEAVES, *MP_RE_DIAGNOSTIC_LEAVES, *LES_TKE_LEAVES, *SMS_CARRY_LEAVES, *EPSILON_PBL_LEAVES, *MYNN_CLOUDPDF_LEAVES)
    if any(field not in optional for field in missing):
        raise ValueError(
            "wrfrst State field order is missing non-conditional leaves: "
            f"{[field for field in missing if field not in optional]}"
        )


def _state_field_order_from_dataset(dataset: Dataset) -> tuple[str, ...]:
    field_order = tuple(json.loads(str(getattr(dataset, "GPUWRF_STATE_FIELD_ORDER", "[]"))))
    required = tuple(json.loads(str(getattr(dataset, "GPUWRF_EXTRA_SCALAR_BOUNDARY_FIELDS", "[]"))))
    if any(name not in field_order for name in required):
        raise ValueError("wrfrst is missing live-child extra scalar boundary records (E78)")
    config = json.loads(str(getattr(dataset, "GPUWRF_MP_RE_CONFIG", "{}")))
    config.update(json.loads(str(getattr(dataset, "GPUWRF_CAMUW_PBL_CONFIG", "{}"))))
    config.update(json.loads(str(getattr(dataset, "GPUWRF_LES_TKE_CONFIG", "{}"))))
    config.update(json.loads(str(getattr(dataset,"GPUWRF_EPSILON_PBL_CONFIG","{}"))))
    config.update(json.loads(str(getattr(dataset, "GPUWRF_MYNN_CLOUDPDF_CONFIG", "{}"))))
    _validate_state_field_order(field_order, namelist=config)
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

STOCHASTIC_SEED_RESTART_VARIABLES: tuple[str, ...] = (*STOCHASTIC_SEED_VARIABLES, "ISEEDARR_RAND_PERTURB")
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
    "slab_land", "slab_rad", "px_land", "px_rad", "base_state",
    "radiation_diagnostics", "cumulus_tendencies", "census", "h_diabatic", "history_diagnostics",
    "land_history", "energy_accumulators",
    "noahmp_precipitation",
    "o3rad",
    "ruc_land", "ruc_rad",
    "cam_abs",  # ra_lw_physics=3 held REAL absorptivities (WRF abstot/absnxt/emstot)
    "camuw_pbl",  # bl_pbl_physics=9 held REAL diffusion/stress/TKE
    "fdda",  # grid_fdda=1 auxinput10 pair + held R*NDGDTEN (WRF restart fdda3d / RUNDGDTEN..)
    "urban_slucm",  # sf_urban_physics=1 SLUCM state (WRF *_URB2D/3D + C*_SFCDIF restart 'r' fields)
    'lake',
    "obs",  # obs_nudge_opt=1 rolling VAROBS/RKO + saved weighted innovations
    'regional_streams',
    "precipitation_buckets",  # prec_acc_dt: exact REAL interval accumulations
    "nwp_diagnostics", "radar_reflectivity",
    "solar_interpolation",  # swint_opt=1 exact REAL parameters and last surface fluxes
    'temf',
    "urban_bep",  # compact BEP/BEM material/indoor/water state and multilayer PBL sources
    "stochastic",
    'feedback_history',
    "windfarm_power",
    "tracer_tendencies",  # held passive-scalar PBL tendencies
    'ocean', 'hurricane_ustm',
    "ssib_land", "ssib_rad",
    "qnse",  # sf/bl=4 INOUT exchange/surface memory and MF cloud condensate
    'osas_rng_key',
    'cuplus',
    'noahmp_mmf',
    'gfdl_radiation',
    "sst_skin",  # sst_skin=1 exact bulk SST / SSTSK / DTW
    "noahmp_irrigation",  # exact WRF event amounts, counters and cumulative water/evaporation
    "noahmp_carbon",  # exact six evolving carbon pools for DVEG 2/5/6
    'noahmp_tile_drain',  # exact accumulated QTDRAIN for opt_tdrn=1/2
    "noahmp_groundwater",  # opt_run1 positive-depth ZWT/WA/WT (distinct from MMF)
    "cu_forcing",  # cu 3/5/6/16/93 held WRF RTHFTEN/RQVFTEN pair
    "aerod",  # aer_opt=1 nested runs: held REAL Tegen per-layer AOD (6, z, y, x)
    "taod5502d",  # aer_opt=3 (mp28): held REAL TAOD5502D (y, x)
    "mynn_budget",  # optional history only, schema 1, exact resident groups
    "afwa_diagnostics",  # afwa_severe_opt=1: exact REAL WSPD10MAX maximum since the last history write
)
PYTREE_CARRY_FIELDS = OPTIONAL_CARRY_FIELDS[5:]
# Pytree groups appended after the current schema: files written before them
# carry no manifest entry and resume with the group absent.
_LATE_PYTREE_CARRY_FIELDS: tuple[str, ...] = ("h_diabatic", "history_diagnostics", "ruc_land", "ruc_rad", "cam_abs", "camuw_pbl",
                                               "fdda", "urban_slucm", "lake", "obs", "regional_streams", "precipitation_buckets", "nwp_diagnostics", "radar_reflectivity", "solar_interpolation", "temf", "urban_bep", "stochastic", "feedback_history", "windfarm_power", "tracer_tendencies", "ocean", "hurricane_ustm", "ssib_land", "ssib_rad", "qnse", "osas_rng_key", "cuplus", "noahmp_mmf", "gfdl_radiation", "sst_skin", "noahmp_irrigation", "noahmp_carbon", "noahmp_groundwater", "noahmp_tile_drain",
                                               "cu_forcing", "aerod", "taod5502d", "mynn_budget", "afwa_diagnostics")

# ``OperationalCarry`` fields this schema has no variables for.  A populated
# (non-``None``) unsupported field is REFUSED at write time instead of being
# silently dropped, so a resume can never lose carry state without a loud error.
# Derived from the dataclass so a future carry field lands here automatically.
UNSUPPORTED_CARRY_FIELDS: tuple[str, ...] = tuple(
    field.name
    for field in dataclass_fields(OperationalCarry)
    if field.name != "state" and field.name not in CARRY_ARRAY_FIELDS and field.name not in OPTIONAL_CARRY_FIELDS
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


def _dry_theta(state: State) -> np.ndarray:
    # Registry.EM_COMMON:209-211 distinguishes dry T from prognostic THM_2.
    # State.theta carries full moist theta_m under both use_theta_m options.
    return np.asarray(state.theta) / (1.0 + RVOVRD * np.asarray(state.qv))


# Only the current time level is available; never fabricate WRF's *_1 fields.
WRF_CURRENT_SLOT_ALIASES = {"U": "U_2", "V": "V_2", "W": "W_2",
                            "PH": "PH_2", "MU": "MU_2"}
THM_CURRENT_SLOT_SPEC = dataclass_replace(_from_wrfout("THM"), name="THM_2")


STANDARD_RESTART_FIELDS: tuple[StandardRestartField, ...] = (
    StandardRestartField(STANDARD_FIELD_SPECS["U"], lambda state: state.u),
    StandardRestartField(STANDARD_FIELD_SPECS["V"], lambda state: state.v),
    StandardRestartField(STANDARD_FIELD_SPECS["W"], lambda state: state.w),
    StandardRestartField(STANDARD_FIELD_SPECS["T"], lambda state: _dry_theta(state) - THETA_BASE_OFFSET_K),
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
    **{name: XYZ for name in CAMMG_MASS_FIELDS},
    **{name: ("Time", "bottom_top_stag", "south_north", "west_east") for name in CAMMG_INTERFACE_FIELDS},
    **{name: XY for name in CAMMG_SURFACE_FIELDS},
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
    "qir": XYZ,
    "qib": XYZ,
    "qi2": XYZ,
    "Ni2": XYZ,
    **{name: XYZ for name in ("qvoli", "qaoli", "qvoli2", "qaoli2", "qi3", "Ni3", "qvoli3", "qaoli3")},
    "qir2": XYZ,
    "qib2": XYZ,
    "qzi": XYZ,
    "mp_th_old": XYZ,
    "mp_qv_old": XYZ,
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
    "qc_bdy": ("Time", "gpuwrf_qc_bdy_time", BDY_SIDE, BDY_WIDTH, "bottom_top", BDY_SIDE_INDEX),
    "qr_bdy": ("Time", "gpuwrf_qr_bdy_time", BDY_SIDE, BDY_WIDTH, "bottom_top", BDY_SIDE_INDEX),
    "qi_bdy": ("Time", "gpuwrf_qi_bdy_time", BDY_SIDE, BDY_WIDTH, "bottom_top", BDY_SIDE_INDEX),
    "qs_bdy": ("Time", "gpuwrf_qs_bdy_time", BDY_SIDE, BDY_WIDTH, "bottom_top", BDY_SIDE_INDEX),
    "qg_bdy": ("Time", "gpuwrf_qg_bdy_time", BDY_SIDE, BDY_WIDTH, "bottom_top", BDY_SIDE_INDEX),
    "Ni_bdy": ("Time", "gpuwrf_Ni_bdy_time", BDY_SIDE, BDY_WIDTH, "bottom_top", BDY_SIDE_INDEX),
    "Nr_bdy": ("Time", "gpuwrf_Nr_bdy_time", BDY_SIDE, BDY_WIDTH, "bottom_top", BDY_SIDE_INDEX),
    **{name: ("Time", f"gpuwrf_{name}_time", BDY_SIDE, BDY_WIDTH, "bottom_top", BDY_SIDE_INDEX)
       for name in EXTRA_SCALAR_BOUNDARY_LEAVES},
    # B39 MYNN surface-layer carry (WRF restart MOL/HFX/QFX/QSFC/PBLH).
    "mol": XY,
    "hfx": XY,
    "qfx": XY,
    "qsfc": XY,
    "pblh": XY,
    "el_pbl": XYZ,
    "maxmf": XY,
    "maxwidth": XY,
    "ztop_plume": XY,
    "sfc_wspd": XY,
    "re_cloud": XYZ,
    "re_ice": XYZ,
    "re_snow": XYZ,
    "tke": XYZ,
    # Opt-in bl_mynn_cloudpdf=0/1 previous-step MYNN state (WRF restart TSQ/COV/SH3D, "r").
    **{name: XYZ for name in MYNN_CLOUDPDF_LEAVES},
    **{name: XY for name in SMS_CARRY_LEAVES},
    "dtaux3d": XYZ,
    "dtauy3d": XYZ,
    "dusfcg": XY,
    "dvsfcg": XY,
    **{name:(('Time',) if name=='epsilon_pbl_step' else XYZ) for name in EPSILON_PBL_LEAVES},
    "tracers": ("Time", "gpuwrf_tracer_count", "bottom_top", "south_north", "west_east"),
    "tracer_budget": ("Time", "gpuwrf_tracer_count", "gpuwrf_tracer_budget_columns"),
    "tracer_bdy": ("Time", "gpuwrf_tracer_count", "gpuwrf_tracer_bdy_time", BDY_SIDE,
                   "gpuwrf_tracer_bdy_width", "bottom_top", "gpuwrf_tracer_bdy_side_index"),
    "qt": XYZ,
    "qrimef": XYZ,
    "f_ice_phy": XYZ,
    "f_rain_phy": XYZ,
    "f_rimef_phy": XYZ,
    "ferrier_sr": XY,
    **{name: XYZ for name in ("acphysc", "acphyse", "acphysd", "acphyss", "acphysm", "acphysf",
                              "re_cloud_gsfc", "re_rain_gsfc", "re_ice_gsfc", "re_snow_gsfc",
                              "re_graupel_gsfc", "re_hail_gsfc")},
    "goddard_sr": XY,
    "phys_tot": XYZ,
    "nwfa2d": XY,
    "nifa2d": XY,
    "rimi": XYZ,
    "exch_tke": XYZ,
    "thompson_sr": XY,
    "thompson_rainnc": XY,
    "thompson_snownc": XY,
    "ntu_qdcn": XYZ,
    "ntu_qtcn": XYZ,
    "ntu_qccn": XYZ,
    "ntu_qrcn": XYZ,
    "ntu_qnin": XYZ,
    "ntu_fi": XYZ,
    "ntu_fs": XYZ,
    "ntu_vi": XYZ,
    "ntu_vs": XYZ,
    "ntu_vg": XYZ,
    "ntu_ai": XYZ,
    "ntu_as": XYZ,
    "ntu_ag": XYZ,
    "ntu_ah": XYZ,
    "ntu_i3m": XYZ,
    "ntu_rainnc": XY,
    "ntu_snownc": XY,
    "ntu_sr": XY,

    **{name: XYZ for name in ICE_DIAGNOSTIC_LEAVES},
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
    "glacier_qsnow": XY,
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
    feedback_schema: dict[str, Any] | None = None,
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
        feedback_schema=feedback_schema,
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
        saved_physics=json.loads(str(getattr(dataset, "GPUWRF_MP_RE_CONFIG", "{}")))
        saved_physics.update(json.loads(str(getattr(dataset, "GPUWRF_CAMUW_PBL_CONFIG", "{}"))))
        validate_held_state(fields, saved_physics)
        validate_cammg(fields,json.loads(str(getattr(dataset,"GPUWRF_MP_RE_CONFIG","{}"))),
                       getattr(dataset,"GPUWRF_CAMMG_SCHEMA_VERSION",None))
        validate_epsilon_state(fields,json.loads(str(getattr(dataset,'GPUWRF_EPSILON_PBL_CONFIG','{}'))),
                               getattr(dataset,'GPUWRF_EPSILON_PBL_SCHEMA_VERSION',None))
        _validate_tracer_dataset(dataset, fields, None)
    return _state_from_exact_fields(fields), metadata


def read_wrfrst_carry(path: str | Path, *, namelist=None, feedback_schema=None) -> tuple[OperationalCarry, dict[str, Any]]:
    """Read exact carry; legacy files without PBL selection need a namelist."""

    target = Path(path)
    with Dataset(target, "r") as dataset:
        _validate_common_schema(dataset, require_carry=True)
        from gpuwrf.io.mynn_cloudpdf_schema import validate as validate_cloudpdf
        validate_cloudpdf(None, json.loads(str(getattr(dataset, "GPUWRF_MYNN_CLOUDPDF_CONFIG", "{}"))), namelist)
        from gpuwrf.io.restart_schema import validate_cycling_restart
        validate_cycling_restart(namelist, getattr(dataset, "GPUWRF_CYCLING_SCHEMA_VERSION", None))
        # Read/validate active feedback on the host before State or optional
        # leaves reach jnp.asarray. Reuse these buffers during restoration.
        feedback_groups=_read_feedback_restart_groups(dataset,expected=feedback_schema)
        _validate_epsilon_history_dataset(dataset)
        field_order = _state_field_order_from_dataset(dataset)
        expected_shapes = _expected_state_shapes_from_dataset(dataset)
        fields: dict[str, Any] = {}
        for leaf in field_order:
            fields[leaf] = jnp.asarray(_read_exact_variable(dataset, state_extension_name(leaf), expected_shapes[leaf]))
        state = _state_from_exact_fields(fields)
        saved_physics=json.loads(str(getattr(dataset, "GPUWRF_MP_RE_CONFIG", "{}")))
        saved_physics.update(json.loads(str(getattr(dataset, "GPUWRF_CAMUW_PBL_CONFIG", "{}"))))
        validate_held_state(fields, saved_physics)
        validate_cammg(fields,json.loads(str(getattr(dataset,"GPUWRF_MP_RE_CONFIG","{}"))),
                       getattr(dataset,"GPUWRF_CAMMG_SCHEMA_VERSION",None))
        validate_epsilon_state(fields,json.loads(str(getattr(dataset,'GPUWRF_EPSILON_PBL_CONFIG','{}'))),
                               getattr(dataset,'GPUWRF_EPSILON_PBL_SCHEMA_VERSION',None))
        if namelist is not None:
            validate_held_state(fields, namelist)
            validate_cammg(fields,namelist,getattr(dataset,"GPUWRF_CAMMG_SCHEMA_VERSION",None))
            validate_epsilon_state(fields,namelist,getattr(dataset,'GPUWRF_EPSILON_PBL_SCHEMA_VERSION',None))
        carry_fields: dict[str, Any] = {"state": state}
        for name in CARRY_ARRAY_FIELDS:
            shape = _shape_for_dimensions(CARRY_EXACT_DIMENSIONS[name], _dataset_dimension_sizes(dataset))
            carry_fields[name] = jnp.asarray(_read_exact_variable(dataset, carry_extension_name(name), shape))
        carry_fields.update(_read_optional_carry_groups(dataset, namelist=namelist,
                                                       feedback_groups=feedback_groups,ssib_state_fields=fields))
        _validate_tracer_dataset(dataset, fields, carry_fields.get("tracer_tendencies"), namelist=namelist)
        metadata = _read_metadata(dataset)
    return OperationalCarry(**carry_fields), metadata


def _validate_tracer_dataset(dataset, fields, tendencies, *, namelist=None):
    from gpuwrf.tracers.restart import validate_restart
    raw = getattr(dataset, "GPUWRF_TRACER_CONFIG", None)
    try:
        config = None if raw is None else json.loads(str(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError("tracer restart configuration is malformed") from exc
    validate_restart(fields, tendencies, config, getattr(dataset, "GPUWRF_TRACER_SCHEMA_VERSION", None),
                     fields["qv"].shape, requested=namelist)


def _state_from_exact_fields(fields: Mapping[str, Any]) -> State:
    """Rebuild the State exactly as stored.

    ``State.__init__`` re-canonicalises leaves (``_as_dtype`` to the fp32-gated
    default matrix, ``None`` -> zeros), which would silently DOWNCAST the fp64
    lateral-boundary/number leaves an ``force_fp64`` operational run carries.  A
    restart is an exact snapshot, so the leaves are assigned verbatim through the
    pytree inverse (``tree_unflatten``); the manifest already guarantees every
    non-conditional leaf is present and inactive conditional leaves stay ``None``.
    """

    return State.tree_unflatten(None, tuple(fields.get(name) for name in State.__slots__))


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
    feedback_schema: dict[str, Any] | None = None,
) -> Path:
    target = Path(path)
    from gpuwrf.io.cuplus_schema import validate as validate_cuplus,SCHEMA_VERSION as CUPLUS_SCHEMA_VERSION
    validate_cuplus(None if carry is None else carry.cuplus,namelist,state.theta.shape,
                   CUPLUS_SCHEMA_VERSION if carry is not None and carry.cuplus is not None else None)
    from gpuwrf.io.cu_forcing_schema import validate as validate_cu_forcing,SCHEMA_VERSION as CU_FORCING_SCHEMA_VERSION
    validate_cu_forcing(None if carry is None else carry.cu_forcing,namelist,state.theta.shape,
                        CU_FORCING_SCHEMA_VERSION if carry is not None and carry.cu_forcing is not None else None)
    target.parent.mkdir(parents=True, exist_ok=True)
    run_start_dt = _coerce_datetime(run_start)
    from gpuwrf.io.feedback_restart import configuration,validate
    feedback_config=configuration(None if carry is None else carry.base_state,
        None if carry is None else carry.feedback_history,state.qv.shape,state.t_skin.shape)
    validate(None if carry is None else carry.base_state,None if carry is None else carry.feedback_history,
             state.qv.shape,state.t_skin.shape,feedback_config,expected=feedback_schema)
    from gpuwrf.io.windfarm_schema import configuration as wind_config, validate_restart as validate_wind
    windfarm_config = wind_config(namelist)
    validate_wind(None if carry is None else carry.windfarm_power,windfarm_config,state.t_skin.shape,requested=namelist)
    validate_held_state({name: getattr(state, name) for name in state.active_field_names()}, namelist)
    validate_cammg(state,namelist,1 if physics_option(namelist,"mp_physics",8)==11 else None)
    from gpuwrf.io.glacier_schema import validate as validate_glacier,field_order as glacier_field_order,marker as glacier_marker
    glacier_land=None if carry is None else carry.noahmp_land
    glacier_order=() if glacier_land is None else glacier_field_order(glacier_land)
    validate_glacier(None if glacier_land is None else {k:getattr(glacier_land,k) for k in glacier_order},
        glacier_order,glacier_marker(glacier_land),namelist=namelist)
    from gpuwrf.io.restart_schema import validate_urban_bep,validate_urban_radiation
    validate_urban_bep(None if carry is None else carry.urban_bep,namelist)
    validate_urban_radiation(None if carry is None else carry.radiation_diagnostics,namelist)
    validate_epsilon_state(state,namelist,EPSILON_PBL_SCHEMA_VERSION)
    if carry is not None:
        validate_epsilon_history(carry.history_diagnostics,namelist,state.t_skin.shape)
        validate_camuw(carry.camuw_pbl, namelist, CAMUW_SCHEMA_VERSION)
        from gpuwrf.io.restart_schema import validate_goddard_radiation
        validate_goddard_radiation(carry.radiation_diagnostics,namelist,
            GODDARD_RADIATION_SCHEMA_VERSION if type(carry.radiation_diagnostics).__name__=='GoddardRadiationDiagnostics' else None)
        from gpuwrf.io.flg_gfdl_schema import validate as validate_legacy,marker
        validate_legacy(carry.radiation_diagnostics,carry.gfdl_radiation,namelist,
            marker(carry.radiation_diagnostics,carry.gfdl_radiation),state_shape=state.qv.shape)
        validate_temf(carry.temf,namelist,TEMF_SCHEMA_VERSION)
        from gpuwrf.io.restart_schema import validate_sas_held
        validate_sas_held(carry.cumulus_tendencies, namelist, carry.osas_rng_key, carry.cumulus_carry)
        validate_lake(carry.lake,namelist,LAKE_SCHEMA_VERSION)
        from gpuwrf.io.ocean_schema import validate as validate_ocean
        validate_ocean(carry.ocean,carry.hurricane_ustm,namelist,1,shape=state.t_skin.shape,history=carry.history_diagnostics,require_history=True)
        from gpuwrf.io.restart_schema import validate_ssib
        validate_ssib(carry.ssib_land,carry.ssib_rad,namelist,1,carry.state.t_skin.shape,
        driver_fields=dict(wspd=carry.state.sfc_wspd,history=carry.history_diagnostics,precipitation=carry.noahmp_precipitation))
        validate_qnse(carry.qnse,namelist,QNSE_SCHEMA_VERSION,history=carry.history_diagnostics,expected_shape=state.theta.shape)
        from gpuwrf.io.restart_schema import validate_observations
        validate_observations(carry.obs,namelist)
        from gpuwrf.diagnostics.precipitation import configured_interval, validate_restart_buckets
        validate_restart_buckets(carry.precipitation_buckets, configured_interval(namelist), 1, carry.state.t_skin.shape)
        from gpuwrf.diagnostics.online import configuration, validate_restart
        validate_restart(carry.nwp_diagnostics,carry.radar_reflectivity,configuration(namelist),1,carry.state.qv.shape)
    from gpuwrf.diagnostics.mynn_budget import configuration as mynn_configuration,validate_restart as validate_mynn_restart
    validate_mynn_restart(None if carry is None else carry.mynn_budget,mynn_configuration(namelist),1,state.theta.shape)
    from gpuwrf.diagnostics.afwa import configuration as afwa_configuration, validate_restart_afwa
    validate_restart_afwa(None if carry is None else carry.afwa_diagnostics,
                          afwa_configuration(namelist)["afwa_severe_opt"], 1, state.t_skin.shape,
                          history=None if carry is None else carry.history_diagnostics, check_history=True)
    from gpuwrf.io.restart_schema import validate_spectral_fdda
    validate_spectral_fdda(None if carry is None else carry.fdda, namelist)
    from gpuwrf.diagnostics.solar_config import validate_options as solar_options, validate_restart as validate_solar_restart
    solar_config = solar_options(namelist)
    from gpuwrf.diagnostics.radiation import validate_restart as validate_rad_diag
    validate_rad_diag(None if carry is None else carry.radiation_diagnostics,
                      namelist, 1, state.theta.shape)
    validate_solar_restart(None if carry is None else carry.solar_interpolation, solar_config, 1,
                           state.t_skin.shape, requested=namelist)
    from gpuwrf.io.stochastic_schema import validate_stochastic, seed_arrays as stochastic_seeds
    validate_stochastic(None if carry is None else carry.stochastic, namelist)
    from gpuwrf.tracers.restart import configuration as tracer_configuration, validate_restart as validate_tracer_restart
    tracer_config = tracer_configuration(namelist)
    validate_tracer_restart({name: getattr(state, name) for name in state.active_field_names()},
                           None if carry is None else carry.tracer_tendencies, tracer_config, 1, state.qv.shape)
    from gpuwrf.io.noahmp_mmf import validate_restart as validate_mmf
    mmf = None if carry is None else carry.noahmp_mmf
    validate_mmf(mmf, namelist, 1 if mmf is not None else None, state.t_skin.shape)
    from gpuwrf.io.noahmp_mmf import validate_land as validate_mmf_land
    validate_mmf_land(mmf,None if carry is None else carry.noahmp_land,state.t_skin.shape)
    from gpuwrf.coupling.sst_skin_hook import validate_restart as validate_skin
    skin_config = int(_lookup(namelist, "sst_skin", 0))
    validate_skin(None if carry is None else carry.sst_skin, skin_config, 1, state.t_skin.shape)
    from gpuwrf.io.noahmp_options import irrigation_configuration
    from gpuwrf.io.noahmp_irrigation import validate_restart as validate_irrigation
    irrigation_config=irrigation_configuration(namelist)
    validate_irrigation(None if carry is None else carry.noahmp_irrigation,irrigation_config,1,state.t_skin.shape)
    from gpuwrf.io.noahmp_carbon import validate_restart as validate_carbon
    validate_carbon(None if carry is None else carry.noahmp_carbon,namelist,1,state.t_skin.shape)
    from gpuwrf.io.noahmp_groundwater import validate_restart as validate_groundwater
    validate_groundwater(None if carry is None else carry.noahmp_groundwater,namelist,1,state.t_skin.shape)
    from gpuwrf.io.noahmp_hydrology import validate_tile_restart
    validate_tile_restart(None if carry is None else carry.noahmp_tile_drain,namelist,1,state.t_skin.shape)
    valid_dt = _coerce_datetime(valid_time)
    if lead_hours is None:
        lead_hours = (valid_dt - run_start_dt).total_seconds() / 3600.0
    nx, ny, nz = _grid_extent(grid)
    dimensions = _restart_dimension_sizes(state=state, nx=nx, ny=ny, nz=nz, namelist=namelist)
    _validate_exact_shapes(
        "State",
        {leaf: getattr(state, leaf) for leaf in state.active_field_names()},
        STATE_EXACT_DIMENSIONS,
        dimensions,
    )
    if carry is not None:
        _reject_unsupported_carry(carry)
        _validate_exact_shapes(
            "OperationalCarry",
            {name: getattr(carry, name) for name in CARRY_ARRAY_FIELDS},
            CARRY_EXACT_DIMENSIONS,
            dimensions,
        )
    coordinate_fields = _grid_coordinate_payload(state, grid, namelist, dimensions)
    if carry is not None and carry.stochastic is not None:
        native_seeds = stochastic_seeds(carry.stochastic)
        if stochastic_seed_arrays is not None and any(
                name not in native_seeds or not np.array_equal(value,native_seeds[name])
                for name,value in stochastic_seed_arrays.items()):
            raise ValueError("explicit restart seeds differ from the live stochastic carry")
        stochastic_seed_arrays = native_seeds
    seed_arrays = _normalize_stochastic_seed_arrays(stochastic_seed_arrays)

    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    with Dataset(temporary, "w", format="NETCDF4") as dataset:
        _create_restart_dimensions(dataset, dimensions)
        _write_global_attrs(dataset, grid, namelist, dimensions, run_start_dt, valid_dt)
        _write_restart_global_attrs(dataset, state, carry, step_index, seed_arrays,namelist=namelist)
        if physics_option(namelist, "cycling", 0):
            from gpuwrf.io.restart_schema import CYCLING_SCHEMA_VERSION
            dataset.GPUWRF_CYCLING_SCHEMA_VERSION = CYCLING_SCHEMA_VERSION
        if feedback_config is not None:
            dataset.GPUWRF_FEEDBACK_SCHEMA=json.dumps(feedback_config,sort_keys=True)
        from gpuwrf.physics.rrtmg_mp_re import mp_re_config
        dataset.GPUWRF_MP_RE_CONFIG = json.dumps(mp_re_config(namelist), sort_keys=True)
        dataset.GPUWRF_EXTRA_SCALAR_BOUNDARY_FIELDS = json.dumps(
            [name for name in EXTRA_SCALAR_BOUNDARY_LEAVES if getattr(state, name) is not None])
        from gpuwrf.io.mynn_cloudpdf_schema import selection as cloudpdf_selection
        if cloudpdf_selection(namelist) != 2:
            dataset.GPUWRF_MYNN_CLOUDPDF_CONFIG = json.dumps({"bl_pbl_physics": 5,
                "bl_mynn_cloudpdf": cloudpdf_selection(namelist)})
        dataset.GPUWRF_LES_TKE_CONFIG = json.dumps({"diff_opt": int(_lookup(namelist, "diff_opt", 0) or 0),
                                                     "km_opt": int(_lookup(namelist, "km_opt", 0) or 0)},
                                                    sort_keys=True)
        dataset.GPUWRF_SAS_CU_PHYSICS = physics_option(namelist, "cu_physics")
        dataset.GPUWRF_OSAS_DOMAIN_ID = physics_option(namelist, "osas_domain_id", 1)
        dataset.GPUWRF_CAMUW_PBL_CONFIG = json.dumps({"bl_pbl_physics": physics_option(namelist, "bl_pbl_physics")})
        if carry is not None and (carry.ocean is not None or carry.hurricane_ustm is not None):
            from gpuwrf.io.ocean_schema import configuration
            dataset.GPUWRF_OCEAN_SCHEMA_VERSION=1
            dataset.GPUWRF_OCEAN_CONFIG=json.dumps(configuration(namelist))
        dataset.GPUWRF_CAMUW_PBL_SCHEMA_VERSION = CAMUW_SCHEMA_VERSION
        dataset.GPUWRF_TEMF_SCHEMA_VERSION = TEMF_SCHEMA_VERSION
        dataset.GPUWRF_WINDFARM_CONFIG = json.dumps(windfarm_config)
        dataset.GPUWRF_SSIB_ACTIVE = int(ssib_active(namelist))
        if carry is not None and carry.ssib_land is not None:
            dataset.GPUWRF_SSIB_SCHEMA_VERSION = 1
        dataset.GPUWRF_QNSE_CONFIG = json.dumps({key:physics_option(namelist,key) for key in
                                                ('bl_pbl_physics','sf_sfclay_physics','mfshconv')})
        dataset.GPUWRF_QNSE_SCHEMA_VERSION = QNSE_SCHEMA_VERSION
        from gpuwrf.io.glacier_schema import marker,required
        dataset.GPUWRF_GLACIER_REQUIRED = int(required(namelist))
        if carry is not None and marker(carry.noahmp_land) is not None:
            dataset.GPUWRF_GLACIER_SCHEMA_VERSION = marker(carry.noahmp_land)
        if carry is not None and carry.lake is not None:
            dataset.GPUWRF_LAKE_SCHEMA_VERSION = LAKE_SCHEMA_VERSION
            dataset.GPUWRF_LAKE_CONFIG = json.dumps({'sf_lake_physics':physics_option(namelist,'sf_lake_physics')})
        from gpuwrf.assimilation.fdda_spectral import SpectralFddaCarry
        if carry is not None and isinstance(carry.fdda, SpectralFddaCarry):
            dataset.GPUWRF_SPECTRAL_FDDA_SCHEMA_VERSION = 1
        from gpuwrf.assimilation.fdda_surface import SurfaceGridFddaCarry
        if carry is not None and isinstance(carry.fdda,SurfaceGridFddaCarry):
            dataset.GPUWRF_SURFACE_FDDA_SCHEMA_VERSION = 1
        dataset.GPUWRF_OBS_NUDGE_OPT = int(carry is not None and carry.obs is not None)
        from gpuwrf.diagnostics.precipitation import configured_interval
        dataset.GPUWRF_PRECIPITATION_BUCKETS_CONFIG = configured_interval(namelist)
        dataset.GPUWRF_PRECIPITATION_BUCKETS_SCHEMA_VERSION = 1
        from gpuwrf.diagnostics.online import configuration
        dataset.GPUWRF_ONLINE_DIAGNOSTICS_CONFIG=json.dumps(configuration(namelist),sort_keys=True)
        dataset.GPUWRF_ONLINE_DIAGNOSTICS_SCHEMA_VERSION=1
        mynn_config=mynn_configuration(namelist)
        if mynn_config!=mynn_configuration({}):
            dataset.GPUWRF_MYNN_BUDGET_CONFIG=json.dumps(mynn_config,sort_keys=True)
        if carry is not None and carry.mynn_budget is not None:
            dataset.GPUWRF_MYNN_BUDGET_SCHEMA_VERSION=1
        if carry is not None and carry.afwa_diagnostics is not None:
            dataset.GPUWRF_AFWA_SEVERE_OPT = 1
            dataset.GPUWRF_AFWA_DIAGNOSTICS_SCHEMA_VERSION = 1
            dataset.GPUWRF_AFWA_HISTORY_INTERVAL = float(namelist.diagnostic_history_interval)
        dataset.GPUWRF_SOLAR_INTERPOLATION_CONFIG=json.dumps(solar_config,sort_keys=True)
        dataset.GPUWRF_SOLAR_INTERPOLATION_SCHEMA_VERSION=1
        if getattr(namelist, "rad_diag", 0):
            dataset.GPUWRF_RAD_DIAG_SCHEMA_VERSION = 1
            dataset.GPUWRF_RAD_DIAG = 1
        if carry is not None and carry.stochastic is not None:
            dataset.GPUWRF_STOCHASTIC_SCHEMA_VERSION = 1
            from gpuwrf.io.stochastic_config import restart_controls_digest
            _stoch_cfg = (namelist.get("stochastic_config") if isinstance(namelist, Mapping)
                          else getattr(namelist, "stochastic_config", None))
            if _stoch_cfg is None:
                raise ValueError("wrfrst stochastic carry needs its stochastic controls to be written (E78)")
            dataset.GPUWRF_STOCHASTIC_CONTROLS = restart_controls_digest(_stoch_cfg)
        dataset.GPUWRF_TRACER_CONFIG=json.dumps(tracer_config,sort_keys=True)
        dataset.GPUWRF_TRACER_SCHEMA_VERSION=1
        if carry is not None and carry.cuplus is not None:
            dataset.GPUWRF_CUPLUS_SCHEMA_VERSION = CUPLUS_SCHEMA_VERSION
            dataset.GPUWRF_CUPLUS_CONFIG = json.dumps({'cu_physics':physics_option(namelist,'cu_physics')})
        if carry is not None and carry.cu_forcing is not None:
            dataset.GPUWRF_CU_FORCING_SCHEMA_VERSION = CU_FORCING_SCHEMA_VERSION
            dataset.GPUWRF_CU_FORCING_CONFIG = json.dumps({'cu_physics':physics_option(namelist,'cu_physics')})
        use_theta_m = parse_use_theta_m(_lookup(namelist, "use_theta_m", 1))
        dataset.USE_THETA_M = np.int32(use_theta_m)
        if carry is not None and type(carry.radiation_diagnostics).__name__=='GoddardRadiationDiagnostics':
            dataset.GPUWRF_GODDARD_RADIATION_SCHEMA_VERSION=np.int32(GODDARD_RADIATION_SCHEMA_VERSION)
            dataset.GPUWRF_GODDARD_RADIATION_CONFIG=json.dumps({'ra_lw_physics':5,'ra_sw_physics':5})
        if carry is not None:
            from gpuwrf.io.flg_gfdl_schema import marker
            legacy=marker(carry.radiation_diagnostics,carry.gfdl_radiation)
            if legacy is not None:dataset.GPUWRF_FLG_GFDL_SCHEMA=json.dumps(legacy)
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
            if field.spec.name in WRF_CURRENT_SLOT_ALIASES:
                alias = dataclass_replace(field.spec, name=WRF_CURRENT_SLOT_ALIASES[field.spec.name])
                _write_restart_variable(dataset, alias, value, dimensions)

        theta_current = thm_source(np.asarray(state.theta), _dry_theta(state), use_theta_m)
        _write_restart_variable(dataset, THM_CURRENT_SLOT_SPEC,
                                theta_current - THETA_BASE_OFFSET_K, dimensions)

        if carry is not None and carry.noahmp_land is not None:
            _write_noahmp_wrf_restart_variables(dataset, carry.noahmp_land, dimensions)
        if carry is not None and carry.ocean is not None:
            for key,value in carry.ocean._asdict().items():
                variable=dataset.createVariable(key.upper(),'f4',('Time','south_north','west_east'))
                variable[0]=np.asarray(value,dtype=np.float32)
        if carry is not None and carry.hurricane_ustm is not None:
            variable=dataset.createVariable('USTM','f4',('Time','south_north','west_east'))
            variable[0]=np.asarray(carry.hurricane_ustm,dtype=np.float32)
        if carry is not None and carry.sst_skin is not None:
            for field, value, description, units in (
                ("SST", carry.sst_skin.bulk_sst, "SEA SURFACE TEMPERATURE", "K"),
                ("SSTSK", carry.sst_skin.sstsk, "SKIN SEA SURFACE TEMPERATURE", "K"),
                ("DTW", carry.sst_skin.dtw, "WARM LAYER TEMP DIFF", "C"),
            ):
                _write_restart_variable(dataset, _spec(field, XY, "XY", description, units), value, dimensions)
        if carry is not None and carry.noahmp_irrigation is not None:
            from gpuwrf.io.wrfout_writer import RESTART_WRFOUT_VARIABLE_SPECS as WRFOUT_VARIABLE_SPECS
            for field,value in carry.noahmp_irrigation._asdict().items():
                if field=="loss_fallback":
                    continue
                key=field.upper()
                if key in WRFOUT_VARIABLE_SPECS:
                    spec=_from_wrfout(key)
                else:
                    spec=_spec(key,XY,"XY","Irrigation water amount for the event","m")
                _write_restart_variable(dataset,spec,value,dimensions)
        if carry is not None and carry.noahmp_carbon is not None:
            from gpuwrf.io.wrfout_writer import RESTART_WRFOUT_VARIABLE_SPECS as WRFOUT_VARIABLE_SPECS
            for field,value in carry.noahmp_carbon._asdict().items():
                _write_restart_variable(dataset,WRFOUT_VARIABLE_SPECS[field.upper()],value,dimensions)
            dataset.GPUWRF_NOAHMP_CARBON_SCHEMA_VERSION=1
        if carry is not None and carry.noahmp_groundwater is not None:
            from gpuwrf.io.wrfout_writer import RESTART_WRFOUT_VARIABLE_SPECS as WRFOUT_VARIABLE_SPECS
            for field,value in carry.noahmp_groundwater._asdict().items():
                _write_restart_variable(dataset,WRFOUT_VARIABLE_SPECS[field.upper()],value,dimensions)
            dataset.GPUWRF_NOAHMP_GROUNDWATER_SCHEMA_VERSION=1
        if carry is not None and carry.noahmp_tile_drain is not None:
            from gpuwrf.io.wrfout_writer import RESTART_WRFOUT_VARIABLE_SPECS as WRFOUT_VARIABLE_SPECS
            _write_restart_variable(dataset,WRFOUT_VARIABLE_SPECS['QTDRAIN'],carry.noahmp_tile_drain,dimensions)
            dataset.GPUWRF_NOAHMP_TILE_DRAIN_SCHEMA_VERSION=1
        if carry is not None and carry.lake is not None:
            from gpuwrf.coupling.lake_surface_hook import lake_history_fields
            from gpuwrf.io.lake_schema import LAKE_HISTORY_FIELDS
            bundle = _lookup(namelist,'lake_bundle')
            if bundle is None:
                raise ValueError('lake restart requires lake_bundle for WRF fields')
            fields = lake_history_fields(carry.lake,bundle)
            from gpuwrf.io.wrfout_writer import RESTART_WRFOUT_VARIABLE_SPECS as WRFOUT_VARIABLE_SPECS
            for name in LAKE_HISTORY_FIELDS:
                _write_restart_variable(dataset,WRFOUT_VARIABLE_SPECS[name],np.asarray(fields[name]),dimensions)
        _write_stochastic_seed_variables(dataset, seed_arrays, dimensions)
        if skin_config:
            dataset.GPUWRF_SST_SKIN_CONFIG = skin_config
            dataset.GPUWRF_SST_SKIN_SCHEMA_VERSION = 1
        if irrigation_config[0]:
            dataset.GPUWRF_NOAHMP_IRRIGATION_CONFIG=json.dumps(irrigation_config)
            dataset.GPUWRF_NOAHMP_IRRIGATION_SCHEMA_VERSION=1
        from gpuwrf.io.noahmp_options import restart_configuration
        noahmp_config=restart_configuration(namelist)
        if noahmp_config is not None:
            dataset.GPUWRF_NOAHMP_OPTIONS=json.dumps(noahmp_config)
            from gpuwrf.io.noahmp_restart_identity import bound_table_identity
            dataset.GPUWRF_NOAHMP_BOUND_TABLES=bound_table_identity(namelist)
        from gpuwrf.surface_params.controls import restart_controls_digest
        if namelist is not None and restart_controls_digest(namelist) is not None:  # Track P (absent = released file)
            dataset.GPUWRF_SURFACE_CONTROLS_DIGEST=restart_controls_digest(namelist)

        for leaf in state.active_field_names():
            _write_exact_state_variable(dataset, leaf, getattr(state, leaf), dimensions)

        if carry is not None:
            for name in CARRY_ARRAY_FIELDS:
                _write_exact_carry_variable(dataset, name, getattr(carry, name), dimensions)
            _write_optional_carry_variables(dataset, carry, dimensions,
                **({'namelist': namelist} if carry.noahmp_mmf is not None else {}))
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(target)
    from gpuwrf.runtime.restart_store import _sync_dir
    _sync_dir(target.parent)
    return target


def _restart_dimension_sizes(
    *,
    state: State,
    nx: int,
    ny: int,
    nz: int,
    namelist: Mapping[str, Any] | Any | None,
) -> dict[str, int | None]:
    dimensions = _dimension_sizes(nx=nx, ny=ny, nz=nz, namelist=namelist, history_schema=False)
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
    for leaf in _active_boundary_leaves(state):
        shape = np.asarray(getattr(state, leaf)).shape
        if len(shape) != 5:
            raise ValueError(f"{leaf} must be 5-D (time, side, bdy_width, z, side_index), got {shape}")
        dimensions[_bdy_time_dimension(leaf)] = int(shape[0])
    if getattr(state, "tracers", None) is not None:
        dimensions["gpuwrf_tracer_count"] = int(state.tracers.shape[0])
    if getattr(state, "tracer_budget", None) is not None:
        dimensions["gpuwrf_tracer_budget_columns"] = 4
    if getattr(state, "tracer_bdy", None) is not None:
        dimensions.update(gpuwrf_tracer_bdy_time=int(state.tracer_bdy.shape[1]),
                          gpuwrf_tracer_bdy_width=int(state.tracer_bdy.shape[3]),
                          gpuwrf_tracer_bdy_side_index=int(state.tracer_bdy.shape[5]))
    return dimensions


def _active_boundary_leaves(state: State) -> tuple[str, ...]:
    """Lateral-boundary leaves that currently hold an array.

    The v0.22 optional scalar wrfbdy leaves (``qc_bdy`` .. ``Nr_bdy``) are ``None``
    unless the wrfbdy file carried them; a ``None`` leaf has no time axis and no
    restart variable, exactly like the other conditional leaves.
    """

    return tuple(leaf for leaf in state.active_field_names() if leaf.endswith("_bdy") and leaf != "tracer_bdy")


def _validate_exact_shapes(
    label: str,
    arrays: Mapping[str, Any],
    exact_dimensions: Mapping[str, tuple[str, ...]],
    dimensions: Mapping[str, int | None],
) -> None:
    """Refuse a write whose exact ``GPUWRF_*`` payload cannot round-trip bit-identically.

    Runs before the file is created: an exact variable is never broadcast,
    squeezed or filled, so a wrong-shaped leaf is a schema violation, not data.
    """

    mismatched = {
        name: (np.asarray(value).shape, _shape_for_dimensions(exact_dimensions[name], dimensions))
        for name, value in arrays.items()
        if np.asarray(value).shape != _shape_for_dimensions(exact_dimensions[name], dimensions)
    }
    if mismatched:
        raise ValueError(f"{label} leaves do not match their exact wrfrst shapes (actual, expected): {mismatched}")


def _reject_unsupported_carry(carry: OperationalCarry) -> None:
    populated = [name for name in UNSUPPORTED_CARRY_FIELDS if getattr(carry, name) is not None]
    if populated:
        raise ValueError(f"wrfrst has no variables for populated OperationalCarry fields {populated}; refusing to drop them")


def _create_restart_dimensions(dataset: Dataset, dimensions: Mapping[str, int | None]) -> None:
    _create_dimensions(dataset, dimensions)
    for name in (BDY_TIME, BDY_SIDE, BDY_WIDTH, BDY_SIDE_INDEX, BDY_SURFACE):
        dataset.createDimension(name, dimensions[name])
    for name in dimensions:
        if name.startswith("gpuwrf_tracer_") and name not in dataset.dimensions:
            dataset.createDimension(name, dimensions[name])
    for leaf in STATE_FIELD_ORDER:
        name = _bdy_time_dimension(leaf)
        if leaf.endswith("_bdy") and name in dimensions and name not in dataset.dimensions:
            dataset.createDimension(name, dimensions[name])


def _bdy_time_dimension(leaf: str) -> str:
    return f"gpuwrf_{leaf}_time"


def _write_restart_global_attrs(
    dataset: Dataset,
    state: State,
    carry: OperationalCarry | None,
    step_index: int,
    stochastic_seed_arrays: Mapping[str, np.ndarray],
    namelist=None,
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
    active_standard_variables.extend(WRF_CURRENT_SLOT_ALIASES.values())
    if carry is not None and carry.sst_skin is not None:
        active_standard_variables.extend(("SST","SSTSK","DTW"))
    if carry is not None and carry.noahmp_irrigation is not None:
        active_standard_variables.extend(name.upper() for name in carry.noahmp_irrigation._fields
                                         if name!="loss_fallback")
    if carry is not None and carry.noahmp_carbon is not None:
        active_standard_variables.extend(name.upper() for name in carry.noahmp_carbon._fields)
    if carry is not None and carry.noahmp_groundwater is not None:
        active_standard_variables.extend(name.upper() for name in carry.noahmp_groundwater._fields)
    if carry is not None and carry.noahmp_tile_drain is not None:active_standard_variables.append('QTDRAIN')
    active_standard_variables.append("THM_2")
    dataset.TITLE = "OUTPUT FROM GPUWRF NETCDF CHECKPOINT WITH WRF-CONVENTION FIELDS"
    dataset.GPUWRF_RESTART_SCOPE = "GPUWRF_ONLY: incomplete native CPU-WRF restart inventory"
    dataset.RESTART_STATUS = "RESTART"
    dataset.GPUWRF_WRFRST_SCHEMA_VERSION = SCHEMA_VERSION
    if physics_option(namelist,"mp_physics",8)==11:
        dataset.GPUWRF_CAMMG_SCHEMA_VERSION=1
    if epsilon_pbl_active(namelist):
        dataset.GPUWRF_EPSILON_PBL_SCHEMA_VERSION=EPSILON_PBL_SCHEMA_VERSION
        dataset.GPUWRF_EPSILON_PBL_CONFIG=json.dumps(epsilon_pbl_config(namelist),separators=(',',':'))
    from gpuwrf.io.restart import RADIATION_DIAGNOSTICS_SCHEMA_VERSION, GWDO_DIAGNOSTICS_SCHEMA_VERSION
    dataset.GPUWRF_RADIATION_DIAGNOSTICS_SCHEMA_VERSION = RADIATION_DIAGNOSTICS_SCHEMA_VERSION
    dataset.GPUWRF_GWDO_DIAGNOSTICS_SCHEMA_VERSION = GWDO_DIAGNOSTICS_SCHEMA_VERSION
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
    *,
    exact: bool = False,
) -> None:
    expected_shape = _shape_for_dimensions(spec.dimensions, dimensions)
    if exact:
        array = _exact_array(spec.name, data, expected_shape)
    else:
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
    _write_restart_variable(dataset, spec, data, dimensions, exact=True)
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
    _write_restart_variable(dataset, spec, data, dimensions, exact=True)
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
    *, namelist=None,
) -> None:
    import base64
    import pickle
    import jax
    from gpuwrf.io.restart_schema import validate_urban_bep
    if carry.noahmp_mmf is not None:
        from gpuwrf.io.noahmp_mmf import configuration
        dataset.GPUWRF_NOAHMP_MMF_SCHEMA_VERSION = np.int32(1)
        config = configuration(namelist)
        if not config['noahmp_mmf_enabled'] or 'dt_s' not in config:
            raise ValueError('MMF wrfrst needs its domain namelist and DT (E78)')
        dataset.GPUWRF_NOAHMP_MMF_CONFIG = json.dumps(config)
    if carry.urban_bep is not None:
        option=3 if hasattr(carry.urban_bep.state,'temperature') else 2
        from gpuwrf.io.restart_schema import _bem_extensions,_urban_context
        live=_bem_extensions(_urban_context(namelist)[0],None) if namelist is not None and option==3 else None
        selection=validate_urban_bep(carry.urban_bep,{'sf_urban_physics':option},
            grid_shape=(dimensions['south_north'],dimensions['west_east']),nz=dimensions['bottom_top'],extensions=live)
        dataset.GPUWRF_URBAN_BEP_OPTION=np.int32(option)
        if selection:
            # Bound only when active, so PV/green-roof-off restart files keep their attributes.
            dataset.GPUWRF_URBAN_BEM_EXTENSIONS=json.dumps(list(selection))
    for group in PYTREE_CARRY_FIELDS:
        value = getattr(carry, group)
        if value is None:
            continue
        if group == "regional_streams":
            from gpuwrf.streams.device import SCHEMA_VERSION, validate_schema
            validate_schema(value, SCHEMA_VERSION)
            dataset.GPUWRF_REGIONAL_STREAMS_SCHEMA_VERSION = SCHEMA_VERSION
        if group == "urban_slucm" and getattr(value,"inputs",None) is not None:
            dataset.GPUWRF_URBAN_INPUTS_SCHEMA_VERSION = 1
            inputs = value.inputs
            dataset.GPUWRF_URBAN_INPUTS_CONFIG = json.dumps({"required":True,
                "ahe":inputs.ahe is not None or inputs.hourly is not None,
                "diurnal":inputs.longitude is not None,"morphology":inputs.morphology is not None,
                "clock":inputs.clock is not None})
        leaves, structure = jax.tree.flatten(value)
        dataset.setncattr("GPUWRF_TREE_" + group.upper(), base64.b64encode(pickle.dumps(structure, protocol=5)).decode("ascii"))
        for index, leaf in enumerate(leaves):
            data = np.asarray(leaf)
            field = f"leaf_{index:03d}"
            dims = []
            for axis, extent in enumerate(data.shape):
                dimension = f"gpuwrf_{group}_{index}_{axis}"
                dataset.createDimension(dimension, extent)
                dims.append(dimension)
            boolean = group == 'noahmp_mmf' and data.dtype == np.bool_
            variable = dataset.createVariable(_optional_variable_name(group, field),
                                               'i1' if boolean else data.dtype, tuple(dims))
            if boolean:
                variable.GPUWRF_BOOLEAN = np.int32(1)
            variable[...] = data
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
    _write_restart_variable(dataset, spec, data, dimensions, exact=True)
    variable = dataset.variables[spec.name]
    variable.gpuwrf_optional_carry_group = group
    variable.gpuwrf_optional_carry_field = field


def _optional_carry_field_order(carry: OperationalCarry | None) -> dict[str, list[str]]:
    if carry is None:
        return {name: [] for name in OPTIONAL_CARRY_FIELDS}
    return {
        **{group: [f"leaf_{index:03d}" for index, _ in enumerate(jax.tree.leaves(getattr(carry, group)))] for group in PYTREE_CARRY_FIELDS},
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
        **{group: "pytree" if getattr(carry, group) is not None else "none" for group in PYTREE_CARRY_FIELDS},
        "noahmp_land": "object" if carry.noahmp_land is not None else "none",
        "noahmp_rad": "tuple" if carry.noahmp_rad is not None else "none",
        "cumulus_carry": cumulus_kind,
        "noahclassic_land": "object" if carry.noahclassic_land is not None else "none",
        "noahclassic_rad": "tuple" if carry.noahclassic_rad is not None else "none",
    }


def _object_field_order(obj: Any) -> tuple[str, ...]:
    if NoahMPLandState is not None and isinstance(obj,NoahMPLandState):
        from gpuwrf.io.glacier_schema import field_order
        return field_order(obj)
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


def _validate_epsilon_history_dataset(dataset):
    """Host-only EPS admission before State/device reconstruction."""
    config=json.loads(str(getattr(dataset,'GPUWRF_EPSILON_PBL_CONFIG','{}')))
    if not epsilon_pbl_active(config): return
    import base64
    import pickle
    import jax
    kind=_optional_carry_kind_manifest(dataset)
    if kind['history_diagnostics']!='pytree':
        raise ValueError('EEPS/KEPS full restart missing surface history carry (E78)')
    fields=_optional_carry_manifest(dataset)['history_diagnostics']
    structure=pickle.loads(base64.b64decode(dataset.getncattr('GPUWRF_TREE_HISTORY_DIAGNOSTICS')))
    leaves=[np.asarray(dataset.variables[_optional_variable_name('history_diagnostics',field)][...]) for field in fields]
    history=jax.tree.unflatten(structure,leaves)
    validate_epsilon_history(history,config,_expected_state_shapes_from_dataset(dataset)['t_skin'])


def _read_feedback_restart_groups(dataset, *, expected=None):
    """Preflight active ALB/history with NumPy leaves, before any device put."""
    import base64
    import pickle
    import jax
    kinds=_optional_carry_kind_manifest(dataset)
    orders=_optional_carry_manifest(dataset)
    structures={}
    for group in ('base_state','feedback_history'):
        if kinds[group]=='pytree':
            structures[group]=pickle.loads(base64.b64decode(dataset.getncattr('GPUWRF_TREE_'+group.upper())))
    raw=getattr(dataset,'GPUWRF_FEEDBACK_SCHEMA',None)
    prototype=(jax.tree.unflatten(structures['base_state'],[object()]*structures['base_state'].num_leaves)
               if 'base_state' in structures else None)
    if (raw is None and expected is None and kinds['feedback_history']=='none'
            and getattr(prototype,'alb',None) is None):
        return {}  # Keep inactive/default restoration unchanged.
    groups={}
    for group in ('base_state','feedback_history'):
        if kinds[group]=='none':
            groups[group]=None
            continue
        fields=orders[group]
        structure=structures.get(group)
        if (structure is None or structure.num_leaves!=len(fields)
                or fields!=[f'leaf_{i:03d}' for i in range(len(fields))]):
            raise ValueError('two-way wrfrst pytree schema differs (E78)')
        arrays=[np.asarray(dataset.variables[_optional_variable_name(group,field)][...]) for field in fields]
        groups[group]=jax.tree.unflatten(structure,arrays)
    from gpuwrf.io.feedback_restart import validate
    shape=tuple(len(dataset.dimensions[n]) for n in ('bottom_top','south_north','west_east'))
    validate(groups['base_state'],groups['feedback_history'],shape,shape[1:],
             None if raw is None else json.loads(str(raw)),expected=expected)
    return groups


def _read_optional_carry_groups(dataset: Dataset, *, namelist=None, feedback_groups=None, ssib_state_fields=None) -> dict[str, Any]:
    """Read optional exact carry groups declared in the restart manifest."""

    field_order = _optional_carry_manifest(dataset)
    kind = _optional_carry_kind_manifest(dataset)
    if kind["radiation_diagnostics"] != "none":
        from gpuwrf.io.restart import RADIATION_DIAGNOSTICS_SCHEMA_VERSION
        if getattr(dataset, "GPUWRF_RADIATION_DIAGNOSTICS_SCHEMA_VERSION", None) != RADIATION_DIAGNOSTICS_SCHEMA_VERSION:
            raise ValueError("wrfrst radiation diagnostics schema predates held cloud_fraction (E78)")
    carry_fields: dict[str, Any] = {name: None for name in OPTIONAL_CARRY_FIELDS}
    import base64
    import pickle
    import jax
    for group in PYTREE_CARRY_FIELDS:
        if feedback_groups is not None and group in feedback_groups:
            carry_fields[group]=jax.tree.map(jnp.asarray,feedback_groups[group])
            continue
        if kind[group] == "none":
            continue
        if kind[group] != "pytree":
            raise ValueError(f"wrfrst {group} must have kind=pytree")
        fields = field_order[group]
        if fields != [f"leaf_{index:03d}" for index in range(len(fields))]:
            raise ValueError(f"wrfrst {group} leaf order differs from schema")
        structure = pickle.loads(base64.b64decode(dataset.getncattr("GPUWRF_TREE_" + group.upper())))
        if structure.num_leaves != len(fields):
            raise ValueError(f"wrfrst {group} tree/leaf counts differ")
        leaves = []
        for field in fields:
            variable = dataset.variables[_optional_variable_name(group, field)]
            value = np.asarray(variable[...])
            if group == 'noahmp_mmf' and int(getattr(variable, 'GPUWRF_BOOLEAN', 0)):
                if value.dtype != np.int8 or np.any((value != 0) & (value != 1)):
                    raise ValueError('MMF restart eligibility encoding differs (E78)')
                value = value.astype(np.bool_)
            leaves.append(value)
        value = jax.tree.unflatten(structure, leaves)
        if group == 'noahmp_mmf':
            from gpuwrf.io.noahmp_mmf import validate_restart as validate_mmf
            raw_config = getattr(dataset, 'GPUWRF_NOAHMP_MMF_CONFIG', None)
            validate_mmf(value, None if raw_config is None else json.loads(str(raw_config)),
                         getattr(dataset, 'GPUWRF_NOAHMP_MMF_SCHEMA_VERSION', None))
        # Validate Goddard's stored precision before JAX can silently narrow a
        # malformed float64 leaf when the process has x64 disabled (E78).
        # Preserve exact stored ocean/USTM and required LH until their dtype
        # and grid shape have been checked; x64-OFF must not hide malformed f64.
        raw_ocean_group = group in ('ocean','hurricane_ustm','history_diagnostics')
        carry_fields[group] = (value if raw_ocean_group or group == 'cu_forcing'
                               or type(value).__name__ in ('GoddardRadiationDiagnostics','CuPlusCarry')
                               else jax.tree.map(jnp.asarray,value))
    urban_bundle = getattr(namelist,"urban_bundle",None)
    urban_inputs_version = getattr(dataset,"GPUWRF_URBAN_INPUTS_SCHEMA_VERSION",None)
    if urban_inputs_version is not None or getattr(urban_bundle,"requires_inputs",False):
        from gpuwrf.io.urban_schema import validate_urban_inputs
        saved = getattr(dataset,"GPUWRF_URBAN_INPUTS_CONFIG",None)
        if saved is None:
            raise ValueError("wrfrst urban inputs configuration missing (E78)")
        shape = tuple(len(dataset.dimensions[name]) for name in ("south_north","west_east"))
        validate_urban_inputs(carry_fields["urban_slucm"],json.loads(str(saved)),urban_inputs_version,shape)
        if namelist is not None:
            validate_urban_inputs(carry_fields["urban_slucm"],namelist,urban_inputs_version,shape)
    from gpuwrf.diagnostics.radiation import validate_restart as validate_rad_diag
    rad_shape = tuple(len(dataset.dimensions[key]) for key in ("bottom_top", "south_north", "west_east"))
    saved_rad_config = {"rad_diag": int(getattr(dataset, "GPUWRF_RAD_DIAG", 0))}
    validate_rad_diag(carry_fields["radiation_diagnostics"], saved_rad_config,
                      getattr(dataset, "GPUWRF_RAD_DIAG_SCHEMA_VERSION", None), rad_shape)
    if namelist is not None:
        validate_rad_diag(carry_fields["radiation_diagnostics"], namelist,
                          getattr(dataset, "GPUWRF_RAD_DIAG_SCHEMA_VERSION", None), rad_shape)
    from gpuwrf.io.cuplus_schema import validate as validate_cuplus
    saved_cuplus=getattr(dataset,'GPUWRF_CUPLUS_CONFIG',None)
    cuplus_shape=tuple(len(dataset.dimensions[key]) for key in ('bottom_top','south_north','west_east'))
    saved_cuplus_config=json.loads(str(saved_cuplus)) if saved_cuplus is not None else (namelist or {})
    validate_cuplus(carry_fields['cuplus'],saved_cuplus_config,cuplus_shape,
                   getattr(dataset,'GPUWRF_CUPLUS_SCHEMA_VERSION',None))
    if namelist is not None:
        validate_cuplus(carry_fields['cuplus'],namelist,cuplus_shape,
                       getattr(dataset,'GPUWRF_CUPLUS_SCHEMA_VERSION',None))
    if carry_fields['cuplus'] is not None:
        carry_fields['cuplus']=jax.tree.map(jnp.asarray,carry_fields['cuplus'])
    from gpuwrf.io.cu_forcing_schema import validate as validate_cu_forcing
    saved_cu_forcing=getattr(dataset,'GPUWRF_CU_FORCING_CONFIG',None)
    # Without the group, the always-written cumulus selector still binds the file (E78).
    saved_cu_forcing_config=(json.loads(str(saved_cu_forcing)) if saved_cu_forcing is not None
                             else {'cu_physics':int(getattr(dataset,'GPUWRF_SAS_CU_PHYSICS',0))})
    validate_cu_forcing(carry_fields['cu_forcing'],saved_cu_forcing_config,cuplus_shape,
                        getattr(dataset,'GPUWRF_CU_FORCING_SCHEMA_VERSION',None))
    if namelist is not None:
        validate_cu_forcing(carry_fields['cu_forcing'],namelist,cuplus_shape,
                            getattr(dataset,'GPUWRF_CU_FORCING_SCHEMA_VERSION',None))
    if carry_fields['cu_forcing'] is not None:
        carry_fields['cu_forcing']=jax.tree.map(jnp.asarray,carry_fields['cu_forcing'])
    if carry_fields["regional_streams"] is not None or hasattr(dataset, "GPUWRF_REGIONAL_STREAMS_SCHEMA_VERSION"):
        from gpuwrf.streams.restart import validate_restart
        validate_restart(carry_fields["regional_streams"], getattr(dataset, "GPUWRF_REGIONAL_STREAMS_SCHEMA_VERSION", None))
    from gpuwrf.io.noahmp_mmf import validate_restart as validate_mmf
    raw_config = getattr(dataset, 'GPUWRF_NOAHMP_MMF_CONFIG', None)
    mmf_version = getattr(dataset, 'GPUWRF_NOAHMP_MMF_SCHEMA_VERSION', None)
    validate_mmf(carry_fields['noahmp_mmf'],
                 None if raw_config is None else json.loads(str(raw_config)), mmf_version,
                 tuple(len(dataset.dimensions[n]) for n in ('south_north', 'west_east')))
    if namelist is not None:
        validate_mmf(carry_fields['noahmp_mmf'], namelist, mmf_version)
    from gpuwrf.io.restart_schema import validate_spectral_fdda
    validate_spectral_fdda(carry_fields["fdda"], namelist,
                          required=hasattr(dataset, "GPUWRF_SPECTRAL_FDDA_SCHEMA_VERSION"))
    from gpuwrf.io.stochastic_schema import validate_stochastic
    if carry_fields["stochastic"] is not None and not hasattr(dataset,"GPUWRF_STOCHASTIC_SCHEMA_VERSION"):
        raise ValueError("wrfrst holds stochastic state without its schema marker (E78)")
    if hasattr(dataset,"GPUWRF_STOCHASTIC_SCHEMA_VERSION") and namelist is not None and (
            namelist.get("stochastic_config") if isinstance(namelist,Mapping)
            else getattr(namelist,"stochastic_config",None)) is None:
        raise ValueError("wrfrst is an active stochastic restart but the request has stochastic physics off (E78)")
    validate_stochastic(carry_fields["stochastic"], namelist,
                        required=hasattr(dataset,"GPUWRF_STOCHASTIC_SCHEMA_VERSION"))
    if (hasattr(dataset,"GPUWRF_STOCHASTIC_SCHEMA_VERSION")
            and dataset.GPUWRF_STOCHASTIC_SCHEMA_VERSION != 1):
        raise ValueError("wrfrst stochastic schema mismatch (E78)")
    if hasattr(dataset,"GPUWRF_STOCHASTIC_SCHEMA_VERSION"):
        saved = getattr(dataset,"GPUWRF_STOCHASTIC_CONTROLS",None)
        if saved is None:
            raise ValueError("wrfrst stochastic restart lacks its bound controls digest (E78)")
        _cfg = (namelist.get("stochastic_config") if isinstance(namelist,Mapping)
                else getattr(namelist,"stochastic_config",None)) if namelist is not None else None
        if _cfg is not None:
            from gpuwrf.io.stochastic_config import restart_controls_digest
            if restart_controls_digest(_cfg) != str(saved):
                raise ValueError("wrfrst stochastic controls (nens/seeds/scales) differ from the restart (E78)")
    from gpuwrf.assimilation.fdda_surface import validate_surface_restart,SurfaceGridFddaCarry
    surface_version=getattr(dataset,'GPUWRF_SURFACE_FDDA_SCHEMA_VERSION',None)
    if isinstance(carry_fields['fdda'],SurfaceGridFddaCarry) and surface_version!=1:
        raise ValueError('wrfrst surface FDDA schema mismatch (E78)')
    validate_surface_restart(carry_fields['fdda'],namelist,required=surface_version is not None)
    if surface_version is not None and surface_version!=1:
        raise ValueError('wrfrst surface FDDA schema mismatch (E78)')
    if (hasattr(dataset, "GPUWRF_SPECTRAL_FDDA_SCHEMA_VERSION")
            and dataset.GPUWRF_SPECTRAL_FDDA_SCHEMA_VERSION != 1):
        raise ValueError("wrfrst spectral FDDA schema mismatch (E78)")
    from gpuwrf.io.restart_schema import validate_urban_bep,validate_urban_radiation
    urban_option=getattr(dataset,'GPUWRF_URBAN_BEP_OPTION',None)
    urban_config=namelist if namelist is not None else {'sf_urban_physics':int(urban_option or 0)}
    urban_grid=(len(dataset.dimensions['south_north']),len(dataset.dimensions['west_east']))
    saved_extensions=getattr(dataset,'GPUWRF_URBAN_BEM_EXTENSIONS',None)
    bound_extensions=(json.loads(str(saved_extensions)) if saved_extensions is not None
                      else [] if urban_option is not None and int(urban_option)==3 else None)
    validate_urban_bep(carry_fields['urban_bep'],urban_config,grid_shape=urban_grid,
                       nz=len(dataset.dimensions['bottom_top']),extensions=bound_extensions)
    validate_urban_radiation(carry_fields['radiation_diagnostics'],urban_config,grid_shape=urban_grid)
    from gpuwrf.io.windfarm_schema import validate_restart as validate_wind
    saved_wind = getattr(dataset,'GPUWRF_WINDFARM_CONFIG',None)
    validate_wind(carry_fields['windfarm_power'],json.loads(str(saved_wind)) if saved_wind is not None else None,
                  tuple(len(dataset.dimensions[a]) for a in ('south_north','west_east')),requested=namelist)
    from gpuwrf.io.restart_schema import validate_ssib
    ssib_shape=tuple(np.asarray(dataset.variables[state_extension_name('t_skin')][0]).shape)
    validate_ssib(carry_fields['ssib_land'],carry_fields['ssib_rad'],namelist,
        getattr(dataset,'GPUWRF_SSIB_SCHEMA_VERSION',None),ssib_shape,
        active=bool(getattr(dataset,'GPUWRF_SSIB_ACTIVE',False)) or ssib_active(namelist),
        driver_fields=dict(wspd=(ssib_state_fields or {}).get('sfc_wspd'),history=carry_fields.get('history_diagnostics'),precipitation=carry_fields.get('noahmp_precipitation')))
    saved_config = getattr(dataset, "GPUWRF_CAMUW_PBL_CONFIG", None)
    validate_temf(carry_fields['temf'],
                  json.loads(str(saved_config)) if saved_config is not None else namelist,
                  getattr(dataset,'GPUWRF_TEMF_SCHEMA_VERSION',None))
    if namelist is not None:
        validate_temf(carry_fields['temf'],namelist,getattr(dataset,'GPUWRF_TEMF_SCHEMA_VERSION',None))
    from gpuwrf.io.ocean_schema import validate as validate_ocean
    ocean_config=getattr(dataset,'GPUWRF_OCEAN_CONFIG',None)
    ocean_saved=json.loads(str(ocean_config)) if ocean_config is not None else namelist
    validate_ocean(carry_fields['ocean'],carry_fields['hurricane_ustm'],ocean_saved,getattr(dataset,'GPUWRF_OCEAN_SCHEMA_VERSION',None),shape=urban_grid,history=carry_fields.get('history_diagnostics'),require_history=True)
    if namelist is not None:
        validate_ocean(carry_fields['ocean'],carry_fields['hurricane_ustm'],namelist,getattr(dataset,'GPUWRF_OCEAN_SCHEMA_VERSION',None),shape=urban_grid,saved=ocean_saved,history=carry_fields.get('history_diagnostics'),require_history=True)
    raw_ocean_groups = ('ocean','hurricane_ustm') if kind['ocean'] != 'none' else ('hurricane_ustm',)
    for group in raw_ocean_groups:
        carry_fields[group] = jax.tree.map(jnp.asarray,carry_fields[group])
    qnse_config=getattr(dataset,'GPUWRF_QNSE_CONFIG',None)
    qnse_namelist=json.loads(str(qnse_config)) if qnse_config is not None else (
        json.loads(str(saved_config)) if saved_config is not None else namelist)
    qnse_shape=(len(dataset.dimensions['bottom_top']),*urban_grid)
    validate_qnse(carry_fields['qnse'],qnse_namelist,getattr(dataset,'GPUWRF_QNSE_SCHEMA_VERSION',None),
                  history=carry_fields['history_diagnostics'],expected_shape=qnse_shape)
    if namelist is not None:
        validate_qnse(carry_fields['qnse'],namelist,getattr(dataset,'GPUWRF_QNSE_SCHEMA_VERSION',None),
                      history=carry_fields['history_diagnostics'],expected_shape=qnse_shape)
        if qnse_config is not None and physics_option(namelist,'bl_pbl_physics') in (2,4):
            if any(physics_option(qnse_namelist,key)!=physics_option(namelist,key) for key in
                   ('bl_pbl_physics','sf_sfclay_physics','mfshconv')):
                raise ValueError('wrfrst QNSE configuration differs (E78)')
    carry_fields['history_diagnostics'] = jax.tree.map(jnp.asarray,carry_fields['history_diagnostics'])
    lake_config = getattr(dataset,'GPUWRF_LAKE_CONFIG',None)
    validate_lake(carry_fields['lake'],json.loads(str(lake_config)) if lake_config is not None else namelist,
                  getattr(dataset,'GPUWRF_LAKE_SCHEMA_VERSION',None))
    if namelist is not None:
        validate_lake(carry_fields['lake'],namelist,getattr(dataset,'GPUWRF_LAKE_SCHEMA_VERSION',None))
    if saved_config is None and namelist is None:
        # Older wrfrst files did not store PBL selection. Absence is not proof
        # the scheme was inactive: require the caller's resolved configuration.
        raise ValueError("wrfrst CAM-UW selection is unknown (E78); supply namelist for a legacy file")
    validate_camuw(carry_fields["camuw_pbl"],
                   json.loads(str(saved_config)) if saved_config is not None else namelist,
                   getattr(dataset, "GPUWRF_CAMUW_PBL_SCHEMA_VERSION", None))
    if namelist is not None:
        validate_camuw(carry_fields["camuw_pbl"], namelist,
                       getattr(dataset, "GPUWRF_CAMUW_PBL_SCHEMA_VERSION", None))
    from gpuwrf.io.restart_schema import validate_goddard_radiation
    goddard_config=getattr(dataset,'GPUWRF_GODDARD_RADIATION_CONFIG',None)
    goddard_schema=getattr(dataset,'GPUWRF_GODDARD_RADIATION_SCHEMA_VERSION',None)
    from gpuwrf.io.flg_gfdl_schema import validate as validate_legacy
    legacy=getattr(dataset,'GPUWRF_FLG_GFDL_SCHEMA',None)
    validate_legacy(carry_fields['radiation_diagnostics'],carry_fields['gfdl_radiation'],namelist,
        json.loads(str(legacy)) if legacy is not None else None,
        state_shape=tuple(len(dataset.dimensions[name]) for name in ('bottom_top','south_north','west_east')))
    validate_goddard_radiation(carry_fields['radiation_diagnostics'],
        json.loads(str(goddard_config)) if goddard_config is not None else
        json.loads(str(getattr(dataset,'GPUWRF_MP_RE_CONFIG','{}'))),goddard_schema,
        state_shape=tuple(len(dataset.dimensions[name]) for name in ('bottom_top','south_north','west_east')))
    if namelist is not None:
        validate_goddard_radiation(carry_fields['radiation_diagnostics'],namelist,goddard_schema)
    validate_goddard_radiation(carry_fields['radiation_diagnostics'],
        json.loads(str(getattr(dataset,'GPUWRF_MP_RE_CONFIG','{}'))),goddard_schema)
    if type(carry_fields['radiation_diagnostics']).__name__=='GoddardRadiationDiagnostics':
        carry_fields['radiation_diagnostics']=jax.tree.map(jnp.asarray,carry_fields['radiation_diagnostics'])
    from gpuwrf.io.restart_schema import validate_observations
    validate_observations(carry_fields["obs"],namelist)
    if int(getattr(dataset,"GPUWRF_OBS_NUDGE_OPT",0)) and carry_fields["obs"] is None:
        raise ValueError("wrfrst is missing held observation VAROBS/RKO/SAVWT (E78)")
    from gpuwrf.diagnostics.precipitation import configured_interval, validate_restart_buckets
    saved_interval = float(getattr(dataset, "GPUWRF_PRECIPITATION_BUCKETS_CONFIG", 0.0))
    bucket_schema = getattr(dataset, "GPUWRF_PRECIPITATION_BUCKETS_SCHEMA_VERSION", None)
    surface_shape = tuple(len(dataset.dimensions[name]) for name in ("south_north", "west_east"))
    validate_restart_buckets(carry_fields["precipitation_buckets"], saved_interval, bucket_schema, surface_shape)
    from gpuwrf.diagnostics.online import configuration, validate_restart
    diagnostic_config=json.loads(str(getattr(dataset,"GPUWRF_ONLINE_DIAGNOSTICS_CONFIG",'{}')))
    diagnostic_config=configuration(diagnostic_config)
    diagnostic_version=getattr(dataset,"GPUWRF_ONLINE_DIAGNOSTICS_SCHEMA_VERSION",None)
    mass_shape=(len(dataset.dimensions["bottom_top"]),*surface_shape)
    validate_restart(carry_fields["nwp_diagnostics"],carry_fields["radar_reflectivity"],
                     diagnostic_config,diagnostic_version,mass_shape)
    if namelist is not None:
        requested=configuration(namelist)
        validate_restart(carry_fields["nwp_diagnostics"],carry_fields["radar_reflectivity"],requested,
                         diagnostic_version,mass_shape)
        if (requested["nwp_diagnostics"] or requested["do_radar_ref"]) and requested!=diagnostic_config:
            raise ValueError("restart online diagnostic cadence/selection differs (E78)")
    from gpuwrf.diagnostics.mynn_budget import configuration as mynn_configuration,validate_restart as validate_mynn_restart
    mynn_raw=getattr(dataset,'GPUWRF_MYNN_BUDGET_CONFIG',None)
    mynn_config=mynn_configuration({} if mynn_raw is None else json.loads(str(mynn_raw)))
    mynn_marker=getattr(dataset,'GPUWRF_MYNN_BUDGET_SCHEMA_VERSION',None)
    validate_mynn_restart(carry_fields['mynn_budget'],mynn_config,mynn_marker,mass_shape)
    if namelist is not None:
        requested=mynn_configuration(namelist)
        validate_mynn_restart(carry_fields['mynn_budget'],requested,mynn_marker,mass_shape)
        if requested!=mynn_config:raise ValueError('wrfrst MYNN budget/EDMF selection differs (E78)')
    from gpuwrf.diagnostics.afwa import configuration as afwa_configuration, validate_restart_afwa
    afwa_saved = int(getattr(dataset, "GPUWRF_AFWA_SEVERE_OPT", 0))
    afwa_marker = getattr(dataset, "GPUWRF_AFWA_DIAGNOSTICS_SCHEMA_VERSION", None)
    afwa_marker = None if afwa_marker is None else int(afwa_marker)
    afwa_saved_interval = getattr(dataset, "GPUWRF_AFWA_HISTORY_INTERVAL", None)
    if carry_fields["afwa_diagnostics"] is not None and afwa_saved_interval is None:
        raise ValueError("wrfrst AFWA WSPD10MAX lacks its history cadence (E78)")
    from gpuwrf.diagnostics.afwa import history_interval as afwa_interval
    validate_restart_afwa(carry_fields["afwa_diagnostics"], afwa_saved, afwa_marker, mass_shape[-2:],
                          history=carry_fields.get("history_diagnostics"), check_history=True,
                          saved_interval=afwa_saved_interval,
                          requested_interval=None if namelist is None else afwa_interval(namelist))
    if namelist is not None and afwa_configuration(namelist)["afwa_severe_opt"] != afwa_saved:
        raise ValueError("wrfrst AFWA WSPD10MAX selection differs (E78)")
    from gpuwrf.diagnostics.solar_config import validate_restart as validate_solar_restart
    solar_raw = getattr(dataset, "GPUWRF_SOLAR_INTERPOLATION_CONFIG", None)
    try:
        solar_config = None if solar_raw is None else json.loads(str(solar_raw))
    except (TypeError, ValueError) as error:
        raise ValueError("restart Solar interpolation config is malformed (E78)") from error
    validate_solar_restart(carry_fields["solar_interpolation"], solar_config,
                           getattr(dataset, "GPUWRF_SOLAR_INTERPOLATION_SCHEMA_VERSION", None),
                           surface_shape, requested=namelist)
    from gpuwrf.coupling.sst_skin_hook import validate_restart as validate_skin
    skin_config = int(getattr(dataset, "GPUWRF_SST_SKIN_CONFIG", 0))
    skin_version = getattr(dataset, "GPUWRF_SST_SKIN_SCHEMA_VERSION", None)
    validate_skin(carry_fields["sst_skin"], skin_config, skin_version, surface_shape)
    if namelist is not None:
        validate_skin(carry_fields["sst_skin"], int(_lookup(namelist, "sst_skin", 0)), skin_version, surface_shape)
    from gpuwrf.io.noahmp_options import irrigation_configuration
    from gpuwrf.io.noahmp_irrigation import validate_restart as validate_irrigation
    irrigation_raw=getattr(dataset,"GPUWRF_NOAHMP_IRRIGATION_CONFIG","[0,0]")
    try:
        irrigation_config=json.loads(str(irrigation_raw))
    except (TypeError,ValueError) as error:
        raise ValueError("Noah-MP irrigation restart configuration is malformed (E78)") from error
    validate_irrigation(carry_fields["noahmp_irrigation"],irrigation_config,
        getattr(dataset,"GPUWRF_NOAHMP_IRRIGATION_SCHEMA_VERSION",None),surface_shape,
        requested=irrigation_configuration(namelist) if namelist is not None else None)
    from gpuwrf.io.noahmp_options import validate_option_restart
    try:
        noahmp_raw=getattr(dataset,"GPUWRF_NOAHMP_OPTIONS",None)
        noahmp_config=None if noahmp_raw is None else json.loads(str(noahmp_raw))
    except (TypeError,ValueError) as error:
        raise ValueError("Noah-MP restart options are malformed (E78)") from error
    validate_option_restart(noahmp_config,namelist)
    from gpuwrf.io.noahmp_restart_identity import validate_bound_table_restart
    validate_bound_table_restart(getattr(dataset,'GPUWRF_NOAHMP_BOUND_TABLES',None),namelist)
    from gpuwrf.surface_params.controls import validate_controls_restart
    validate_controls_restart(getattr(dataset,'GPUWRF_SURFACE_CONTROLS_DIGEST',None),namelist)
    from gpuwrf.io.noahmp_carbon import validate_restart as validate_carbon
    # Options are encoded even when a caller does not supply a requested namelist.
    from types import SimpleNamespace
    from gpuwrf.io.noahmp_options import NoahMPOptions
    saved_namelist=SimpleNamespace(noahmp_options=None if noahmp_config is None else NoahMPOptions(**noahmp_config))
    validate_carbon(carry_fields["noahmp_carbon"],saved_namelist,getattr(dataset,"GPUWRF_NOAHMP_CARBON_SCHEMA_VERSION",None),surface_shape)
    from gpuwrf.io.noahmp_groundwater import validate_restart as validate_groundwater
    validate_groundwater(carry_fields["noahmp_groundwater"],saved_namelist,
                        getattr(dataset,"GPUWRF_NOAHMP_GROUNDWATER_SCHEMA_VERSION",None),surface_shape)
    from gpuwrf.io.noahmp_hydrology import validate_tile_restart
    validate_tile_restart(carry_fields['noahmp_tile_drain'],saved_namelist,getattr(dataset,'GPUWRF_NOAHMP_TILE_DRAIN_SCHEMA_VERSION',None),surface_shape)
    if namelist is not None:
        validate_carbon(carry_fields["noahmp_carbon"],namelist,getattr(dataset,"GPUWRF_NOAHMP_CARBON_SCHEMA_VERSION",None),surface_shape)
        validate_groundwater(carry_fields["noahmp_groundwater"],namelist,
                            getattr(dataset,"GPUWRF_NOAHMP_GROUNDWATER_SCHEMA_VERSION",None),surface_shape)
        validate_tile_restart(carry_fields['noahmp_tile_drain'],namelist,getattr(dataset,'GPUWRF_NOAHMP_TILE_DRAIN_SCHEMA_VERSION',None),surface_shape)
    if namelist is not None:
        interval = configured_interval(namelist)
        validate_restart_buckets(carry_fields["precipitation_buckets"], interval, bucket_schema)
        if interval != saved_interval:
            raise ValueError("restart prec_acc_dt differs from requested interval (E78)")
    if carry_fields["land_history"] is not None or carry_fields["energy_accumulators"] is not None:
        # Files written before the packed history families hold per-name dicts.
        from gpuwrf.runtime.history_accumulators import ENERGY_ACCUMULATORS, LAND_FLUX_FIELDS, as_packed
        carry_fields["land_history"] = as_packed(carry_fields["land_history"], LAND_FLUX_FIELDS)
        carry_fields["energy_accumulators"] = as_packed(carry_fields["energy_accumulators"], ENERGY_ACCUMULATORS)

    noahmp_land_fields = tuple(field_order["noahmp_land"])
    if noahmp_land_fields:
        if NoahMPLandState is None:
            raise ValueError("wrfrst contains Noah-MP land carry but NoahMPLandState is unavailable")
        expected = tuple(NoahMPLandState.__slots__)
        if noahmp_land_fields not in (expected,expected[:-1]):
            raise ValueError("wrfrst Noah-MP land field order does not match NoahMPLandState.__slots__")
        carry_fields["noahmp_land"] = NoahMPLandState(
            **{
                field: jnp.asarray(_read_optional_variable(dataset, noahmp_land_extension_name(field)))
                for field in noahmp_land_fields
            }
        )
        from gpuwrf.io.glacier_schema import validate,required
        land=carry_fields['noahmp_land']
        validate({k:getattr(land,k) for k in noahmp_land_fields},noahmp_land_fields,
            getattr(dataset,'GPUWRF_GLACIER_SCHEMA_VERSION',None),
            active=bool(getattr(dataset,'GPUWRF_GLACIER_REQUIRED',False)) or required(namelist))

    if not noahmp_land_fields:
        from gpuwrf.io.glacier_schema import validate,required
        validate(None,(),getattr(dataset,'GPUWRF_GLACIER_SCHEMA_VERSION',None),
            active=bool(getattr(dataset,'GPUWRF_GLACIER_REQUIRED',False)) or required(namelist))
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

    from gpuwrf.io.restart_schema import validate_sas_held
    saved_cu = int(getattr(dataset, "GPUWRF_SAS_CU_PHYSICS", 0))
    validate_sas_held(carry_fields['cumulus_tendencies'], {'cu_physics': saved_cu,
        'osas_domain_id':int(getattr(dataset,'GPUWRF_OSAS_DOMAIN_ID',1))}, carry_fields['osas_rng_key'], carry_fields['cumulus_carry'])
    if namelist is not None:
        validate_sas_held(carry_fields['cumulus_tendencies'], namelist, carry_fields['osas_rng_key'], carry_fields['cumulus_carry'])

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

    from gpuwrf.io.noahmp_mmf import validate_land as validate_mmf_land
    validate_mmf_land(carry_fields['noahmp_mmf'],carry_fields['noahmp_land'],
                      tuple(len(dataset.dimensions[n]) for n in ('south_north','west_east')))
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
    if str(dataset.GPUWRF_WRFRST_SCHEMA_VERSION) == LEGACY_SCHEMA_VERSION:
        for group in PYTREE_CARRY_FIELDS:
            payload.setdefault(group, [])
    for group in _LATE_PYTREE_CARRY_FIELDS:
        payload.setdefault(group, [])
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
    if str(dataset.GPUWRF_WRFRST_SCHEMA_VERSION) == LEGACY_SCHEMA_VERSION:
        for group in PYTREE_CARRY_FIELDS:
            payload.setdefault(group, "none")
    for group in _LATE_PYTREE_CARRY_FIELDS:
        payload.setdefault(group, "none")
    missing = [name for name in OPTIONAL_CARRY_FIELDS if name not in payload]
    if missing:
        raise ValueError(f"wrfrst optional carry kind manifest missing groups: {missing}")
    allowed = {"none", "object", "tuple", "array", "pytree"}
    kinds: dict[str, str] = {}
    for group in OPTIONAL_CARRY_FIELDS:
        value = payload[group]
        if value not in allowed:
            raise ValueError(f"wrfrst optional carry kind {group}={value!r} is unsupported")
        kinds[group] = value
    return kinds


def _optional_variable_name(group: str, field: str) -> str:
    if group in PYTREE_CARRY_FIELDS:
        return "GPUWRF_" + group.upper() + "_" + field.upper()
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


def _exact_array(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    """Host copy of an exact ``GPUWRF_*`` payload: the stored shape IS the leaf shape."""

    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} != exact wrfrst shape {shape}")
    return array


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
    if leaf.startswith("ntu_"):
        if leaf in ("ntu_rainnc", "ntu_snownc"): return "mm"
        if leaf == "ntu_sr": return "1"
        from gpuwrf.io.microphysics_history import DIAGNOSTIC_METADATA
        return DIAGNOSTIC_METADATA[leaf[4:].upper()][1]
    if leaf in {"u", "v", "w"}:
        return "m s-1"
    if leaf in {"theta", "t_skin", "mp_th_old"}:
        return "K"
    if leaf in {"qv", "qc", "qr", "qi", "qs", "qg", "qh", "soil_moisture", "qi2", "qi3", "qir", "qir2", "mp_qv_old"}:
        return "kg kg-1" if leaf != "soil_moisture" else "m3 m-3"
    if leaf in {"qvoli", "qaoli", "qvoli2", "qaoli2", "qvoli3", "qaoli3"}:
        return "m(3) kg(-1)"
    if leaf in {"qvolg", "qvolh", "qib", "qib2", "qzi"}:
        # v0.17 ADR-032 predicted-density particle volume.
        return "m3 kg-1"
    if leaf in {"p", "p_total", "p_perturbation", "mu", "mu_total", "mu_perturbation"}:
        return "Pa"
    if leaf in {"ph", "ph_total", "ph_perturbation"}:
        return "m2 s-2"
    if leaf in {"Ni", "Nr", "Ns", "Ng", "Nc", "Nn", "Nh", "nwfa", "nifa", "Ni2", "Ni3"}:
        return "kg-1"
    if leaf in {"nwfa2d", "nifa2d"}:
        return "kg-1 s-1"
    if leaf in {"thompson_rainnc", "thompson_snownc"}:
        return "mm"
    if leaf == "phys_tot":
        return "K s-1"
    if leaf.startswith("acphys"):
        return "K"
    if leaf.endswith("_gsfc") and leaf.startswith("re_"):
        return "micron"
    if leaf == "qke":
        return "m2 s-2"
    if leaf in ('pek_adv','tke_adv'): return 'm2 s-2'
    if leaf in ('epsilon_pbl','pep_adv','diss_adv'): return 'm2 s-3'
    if leaf in ('tpe_pbl','tpe_adv'): return 'K2'
    if leaf == "qsq":
        return "kg2 kg-2"
    # Registry.EM_COMMON:1125/1127/1128 units of the opt-in bl_mynn_cloudpdf state.
    if leaf == "tsq":
        return "K2"
    if leaf == "cov":
        return "K kg/kg"
    if leaf == "sh3d":
        return ""
    if leaf in {"qc_bl", "qi_bl"}:
        return "kg kg-1"
    if leaf in ("dtaux3d", "dtauy3d"):
        return "m s-2"
    if leaf in ("dusfcg", "dvsfcg"):
        return "Pa"
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
    if leaf == "mol":
        return "K"
    if leaf == "hfx":
        return "W m-2"
    if leaf == "qfx":
        return "kg m-2 s-1"
    if leaf == "qsfc":
        return "kg kg-1"
    if leaf == "pblh":
        return "m"
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
    from gpuwrf.io.restart import _validate_gwdo_diagnostics_schema
    _validate_gwdo_diagnostics_schema({
        "gwdo_diagnostics_schema_version": getattr(dataset, "GPUWRF_GWDO_DIAGNOSTICS_SCHEMA_VERSION", None)
    })
    if schema not in (SCHEMA_VERSION, LEGACY_SCHEMA_VERSION):
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
    if schema == LEGACY_SCHEMA_VERSION and optional_order.get("noahmp_land"):
        raise ValueError("wrfrst schema predates held Noah-MP precipitation (E78)")
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
    "OPTIONAL_CARRY_FIELDS",
    "UNSUPPORTED_CARRY_FIELDS",
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
