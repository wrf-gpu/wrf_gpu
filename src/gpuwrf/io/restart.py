"""``wrfrst``-equivalent restart for the FULL operational physics carry (P0-5b).

WRF's restart file (``wrfrst``) is a *bit-continuation* snapshot: reloading it and
continuing the integration reproduces the uninterrupted run exactly, because it
serializes not just the prognostic state but every piece of physics carry the
driver holds between steps (held radiation tendencies, accumulators, sub-step
scratch, land-surface memory, ...). ``runtime.checkpoint`` already round-trips the
prognostic :class:`State` (+ optional runtime/land sub-trees), but the operational
driver advances an :class:`~gpuwrf.runtime.operational_state.OperationalCarry`, and
the parts of that carry OUTSIDE ``State`` are exactly the WRF restart-relevant
physics memory:

* the WRF small-step scratch family ``t_2ave/ww/mudf/muave/muts/ph_tend`` and the
  RK/acoustic transition ``*_save`` arrays (``runtime.operational_state``);
* ``rthraten`` — the HELD radiative potential-temperature tendency (K/s) WRF
  refreshes once per ``radt`` and ADDS every dynamics step in between. Dropping it
  at a restart that lands mid-radiation-interval would silently lose up to one
  ``radt`` of radiative heating — a real (small) forecast discontinuity;
* ``noahmp_land`` — the prognostic Noah-MP land carry (soil/snow/canopy memory),
  and ``noahmp_rad`` — the held SOLDN/LWDN/COSZ surface-radiation forcing.

This module is the I/O-lane restart schema that serializes that COMPLETE carry so a
resumed forecast bit-continues. It does NOT touch ``runtime/checkpoint.py``
(runtime lane owns that); it is a strict superset built on the same fail-closed,
host-numpy, schema-versioned discipline.

Serialization discipline (mirrors ``runtime.checkpoint``):
* every array leaf is copied to a host ``numpy`` array (process-independent pickle,
  no device dependency — the save/load round-trip is CPU-only and GPU-free);
* each pytree is stored as an explicit ``{field_name: array}`` dict with the
  recorded ``__slots__``/field order, so a schema change FAILS CLOSED on read
  rather than silently mis-reconstructing;
* ``format``/``format_version`` gate the payload.
"""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

from dataclasses import fields as dataclass_fields, is_dataclass, replace
from pathlib import Path
import pickle
import os
import tempfile
from typing import Any

import jax
from jax import config
import numpy as np

from gpuwrf.contracts.ice_diagnostics import ice_diagnostic_leaves
from gpuwrf.contracts.state import CONDITIONAL_STATE_LEAVES, State, mynn_sfc_wspd_enabled
from gpuwrf.contracts.precision import GWDO_DIAGNOSTIC_LEAVES, LES_TKE_LEAVES, MP_RE_DIAGNOSTIC_LEAVES, SMS_CARRY_LEAVES, EPSILON_PBL_LEAVES, les_tke_active, MYNN_CLOUDPDF_LEAVES, mynn_cloudpdf_leaves
from gpuwrf.io.pbl_epsilon_schema import EPSILON_PBL_SCHEMA_VERSION,validate_epsilon_state,validate_epsilon_history
from gpuwrf.runtime.operational_state import OperationalCarry
from gpuwrf.io.restart_schema import (
    CAMUW_SCHEMA_VERSION, effective_radii_required, effective_radius_fields, physics_option, validate_camuw, validate_held_state,
    LAKE_SCHEMA_VERSION,validate_lake,
    TEMF_SCHEMA_VERSION,validate_temf,
    QNSE_SCHEMA_VERSION,validate_qnse,
    GODDARD_RADIATION_SCHEMA_VERSION,
)

try:  # Noah-MP land/static live in the v0.2.0 land package; optional so a
    # pure-dycore checkout (Noah-MP off) can still read/write a land-less restart.
    from gpuwrf.contracts.noahmp_state import NoahMPLandState, NoahMPStatic
except Exception:  # pragma: no cover
    NoahMPLandState = None  # type: ignore
    NoahMPStatic = None  # type: ignore


configure_jax_x64()

FORMAT = "gpuwrf-operational-restart"
# v1: State + small-step scratch + Noah-MP land/radiation.
# v2: every current OperationalCarry field, including cadence-held physics,
# radiation diagnostics, optional land/base-state trees and census counters.
# v3: resident full-history accumulators. v4: held GWDO stress diagnostics;
# pre-GWDO-schema checkpoints fail closed (including old inactive domains).
# v5: held Noah-MP precipitation. v6: held precipitation SR channel for Noah-classic FFROZP
# (pre-v6 Noah-classic checkpoints fail closed; t-calleraudit F24).
FORMAT_VERSION = 6
SUPPORTED_FORMAT_VERSIONS = (1, 2, 3, 4, 5, 6)
# E78: a populated pre-cloud-fraction diagnostics tuple must not acquire None
# from the new NamedTuple default during unpickling and silently lose CLDFRA.
RADIATION_DIAGNOSTICS_SCHEMA_VERSION = 1
GWDO_DIAGNOSTICS_SCHEMA_VERSION = 2

# OperationalCarry leaves that are NOT themselves nested pytrees handled
# specially below (state / noahmp_land / noahmp_rad). These are the WRF
# small-step scratch + held radiation arrays carried between steps.
_CARRY_SCRATCH_FIELDS: tuple[str, ...] = (
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

# Nested carry groups this format serializes (each as an explicit field dict).
_SUPPORTED_CARRY_GROUPS: tuple[str, ...] = ("noahmp_land", "noahmp_rad")

# Fields absent from the v1 payload. v2 stores each as an exact host pytree.
_V1_UNSUPPORTED_CARRY_FIELDS: tuple[str, ...] = tuple(
    field.name
    for field in dataclass_fields(OperationalCarry)
    if field.name != "state"
    and field.name not in _CARRY_SCRATCH_FIELDS
    and field.name not in _SUPPORTED_CARRY_GROUPS
)
UNSUPPORTED_CARRY_FIELDS: tuple[str, ...] = ()
_CARRY_FIELD_ORDER = tuple(field.name for field in dataclass_fields(OperationalCarry))


def _reject_unsupported_carry(carry: OperationalCarry) -> None:
    populated = [name for name in UNSUPPORTED_CARRY_FIELDS if getattr(carry, name) is not None]
    if populated:
        raise ValueError(
            f"restart format {FORMAT!r} v{FORMAT_VERSION} has no payload for populated "
            f"OperationalCarry fields {populated}; refusing to drop them"
        )


def _hostify(leaf: Any) -> np.ndarray:
    return np.asarray(leaf)


def _hostify_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(_hostify, tree)


def _device_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda leaf: jax.device_put(leaf), tree)


def _hostify_namelist(namelist: Any, grid: Any) -> Any:
    """Host-copy the namelist pytree, re-pinning ``grid`` by reference (matches
    ``runtime.checkpoint`` so a namelist carrying a grid handle round-trips)."""
    from gpuwrf.physics.thompson_hail.tables import checkpoint_split

    # mp38 bound tables (~0.97 GB) are stored as a content reference only.
    namelist, hail_ref = checkpoint_split(namelist)
    host = _hostify_tree(namelist)
    if is_dataclass(host) and hasattr(host, "grid"):
        host = replace(host, grid=grid)
    return host if hail_ref is None else replace(host, thompson_hail_tables=hail_ref)


def _restore_namelist(namelist: Any, grid: Any) -> Any:
    from gpuwrf.physics.thompson_hail.tables import checkpoint_restore, checkpoint_split

    namelist, hail_ref = checkpoint_split(namelist)
    device = _device_tree(namelist)
    if is_dataclass(device) and hasattr(device, "grid"):
        device = replace(device, grid=grid)
    return checkpoint_restore(device, hail_ref)


def _state_fields(state: State) -> dict[str, np.ndarray]:
    return {name: _hostify(getattr(state, name)) for name in state.active_field_names()}


def _validate_state_field_order(recorded: tuple[str, ...], *, namelist=None) -> None:
    expected = tuple(State.__slots__)
    if any(field not in expected for field in recorded):
        raise ValueError("restart State field order contains unknown leaves")
    if recorded != tuple(field for field in expected if field in recorded):
        raise ValueError("restart State field order does not match current State schema")
    missing = tuple(field for field in expected if field not in recorded)
    # E78: preserve BP90's guard before the later MP_RE schema extension.
    # OFF checkpoints may omit each optional diagnostic; ON must restore it.
    if mynn_sfc_wspd_enabled() and "sfc_wspd" in missing:
        raise ValueError("restart State schema lacks held MYNN sfc_wspd (E78)")
    if effective_radii_required(namelist) and any(field in effective_radius_fields(namelist) for field in missing):
        raise ValueError("MP_RE restart is missing held Thompson effective radii (E78)")
    if les_tke_active(namelist) and any(field in LES_TKE_LEAVES for field in missing):
        raise ValueError("km_opt=2 restart is missing the prognostic LES TKE (WRF tke_2) (E78)")
    if any(field in missing for field in mynn_cloudpdf_leaves(namelist)):
        raise ValueError("bl_mynn_cloudpdf restart is missing held MYNN TSQ/COV/SH3D (E78)")
    if physics_option(namelist, 'km_opt') == 5 and any(field in SMS_CARRY_LEAVES for field in missing):
        raise ValueError('km_opt=5 restart is missing held SMS carry (E78)')
    if any(name in missing for name in ice_diagnostic_leaves(physics_option(namelist, "mp_physics", 8))):
        raise ValueError("restart is missing held ice diagnostics (E78)")
    optional = (*CONDITIONAL_STATE_LEAVES, *GWDO_DIAGNOSTIC_LEAVES, *MP_RE_DIAGNOSTIC_LEAVES, *LES_TKE_LEAVES, *SMS_CARRY_LEAVES, *EPSILON_PBL_LEAVES, *MYNN_CLOUDPDF_LEAVES)
    if any(field not in optional for field in missing):
        raise ValueError(
            "restart State fields are missing non-conditional leaves: "
            f"{[field for field in missing if field not in optional]}"
        )


def _land_fields(land: Any) -> dict[str, np.ndarray]:
    if NoahMPLandState is None:
        raise ValueError("NoahMPLandState unavailable; cannot serialize land carry")
    from gpuwrf.io.glacier_schema import field_order
    return {name: _hostify(getattr(land, name)) for name in field_order(land)}


def _carry_to_payload(carry: OperationalCarry, *, mynn_config=None) -> dict[str, Any]:
    """Explicit field-named, host-numpy serialization of the FULL carry."""
    validate_epsilon_history(carry.history_diagnostics,
        {'bl_pbl_physics':16 if carry.state.epsilon_pbl is not None else 0},carry.state.t_skin.shape)
    if getattr(carry, "timeseries", None) is not None:
        raise ValueError("tslist stream checkpoint/resume is unsupported; refusing a buffered station stream")

    payload: dict[str, Any] = {
        "state_field_order": list(carry.state.active_field_names()),
        "state_fields": _state_fields(carry.state),
        "scratch_field_order": list(_CARRY_SCRATCH_FIELDS),
        "scratch_fields": {name: _hostify(getattr(carry, name)) for name in _CARRY_SCRATCH_FIELDS},
        # Noah-MP land carry (None when Noah-MP off).
        "noahmp_land_field_order": None,
        "noahmp_land_fields": None,
        # Held surface-radiation forcing (SOLDN, LWDN, COSZ) — a tuple of arrays.
        "noahmp_rad": None,
    }
    if carry.noahmp_land is not None:
        payload["noahmp_land_fields"] = _land_fields(carry.noahmp_land)
        payload["noahmp_land_field_order"] = list(payload['noahmp_land_fields'])
        from gpuwrf.io.glacier_schema import marker
        payload['glacier_schema_version']=marker(carry.noahmp_land)
    if carry.noahmp_rad is not None:
        payload["noahmp_rad"] = [_hostify(component) for component in carry.noahmp_rad]
    payload["carry_field_order"] = list(_CARRY_FIELD_ORDER)
    payload["extra_fields"] = {
        name: _hostify_tree(getattr(carry, name)) for name in _V1_UNSUPPORTED_CARRY_FIELDS
    }
    if carry.mynn_budget is not None or mynn_config is not None:
        from gpuwrf.diagnostics.mynn_budget import configuration,validate_restart
        config=configuration({}) if mynn_config is None else configuration(mynn_config)
        validate_restart(carry.mynn_budget,config,1,carry.state.theta.shape)
        if config!=configuration({}):payload['mynn_budget_config']=config
        if carry.mynn_budget is not None:payload['mynn_budget_schema_version']=1
    if carry.noahmp_mmf is not None:
        from gpuwrf.io.noahmp_mmf import validate_restart, validate_land
        validate_restart(payload['extra_fields']['noahmp_mmf'], version=1,
                         shape=carry.state.t_skin.shape)
        validate_land(payload['extra_fields']['noahmp_mmf'],payload['noahmp_land_fields'],carry.state.t_skin.shape)
        payload['noahmp_mmf_schema_version'] = 1
    payload["radiation_diagnostics_schema_version"] = RADIATION_DIAGNOSTICS_SCHEMA_VERSION
    if getattr(carry.radiation_diagnostics, "rad_diag", None) is not None:
        payload["rad_diag_schema_version"] = 1
    from gpuwrf.io.topographic_shadow_schema import marker as shadow_marker
    shadow_version = shadow_marker(carry.radiation_diagnostics)
    if shadow_version is not None:
        payload["topographic_shadow_schema_version"] = shadow_version
    from gpuwrf.io.flg_gfdl_schema import marker
    legacy=marker(carry.radiation_diagnostics,carry.gfdl_radiation)
    if legacy is not None:payload['flg_gfdl_schema']=legacy
    if type(carry.radiation_diagnostics).__name__=='GoddardRadiationDiagnostics':
        payload['goddard_radiation_schema_version']=GODDARD_RADIATION_SCHEMA_VERSION
    payload["gwdo_diagnostics_schema_version"] = GWDO_DIAGNOSTICS_SCHEMA_VERSION
    payload["camuw_pbl_schema_version"] = CAMUW_SCHEMA_VERSION
    payload['cuplus_schema_version'] = 1 if carry.cuplus is not None else None
    from gpuwrf.io.cu_forcing_schema import SCHEMA_VERSION as CU_FORCING_SCHEMA_VERSION
    payload['cu_forcing_schema_version'] = CU_FORCING_SCHEMA_VERSION if carry.cu_forcing is not None else None
    payload['temf_schema_version'] = TEMF_SCHEMA_VERSION
    payload['lake_schema_version'] = LAKE_SCHEMA_VERSION
    if carry.state.epsilon_pbl is not None:
        payload['epsilon_pbl_schema_version']=EPSILON_PBL_SCHEMA_VERSION
    payload['ocean_schema_version']=1
    # Full configuration is attached by write_restart, alongside the source namelist.
    if getattr(carry, "regional_streams", None) is not None:
        from gpuwrf.streams.device import SCHEMA_VERSION, validate_schema
        validate_schema(carry.regional_streams, SCHEMA_VERSION)
        payload["regional_streams_schema_version"] = SCHEMA_VERSION
    payload["precipitation_buckets_schema_version"] = 1
    if carry.afwa_diagnostics is not None:
        payload["afwa_diagnostics_schema_version"] = 1
    payload["online_diagnostics_schema_version"] = 1
    payload["solar_interpolation_schema_version"] = 1
    from gpuwrf.io.feedback_restart import configuration
    extras, state = payload['extra_fields'], payload['state_fields']
    feedback_config=configuration(extras.get('base_state'),extras.get('feedback_history'),
                                  state['qv'].shape,state['t_skin'].shape)
    if feedback_config is not None:
        payload['feedback_schema']=feedback_config
    payload["windfarm_config"] = None  # filled from resolved namelist by write_restart
    payload["urban_inputs_schema_version"] = 1 if getattr(carry.urban_slucm,"inputs",None) is not None else None
    if getattr(carry.state, "tracers", None) is not None or getattr(carry, "tracer_tendencies", None) is not None:
        payload["tracer_schema_version"] = 1
    if carry.ssib_land is not None:
        payload["ssib_schema_version"] = 1
    payload['qnse_schema_version'] = QNSE_SCHEMA_VERSION
    payload["sst_skin_schema_version"] = 1
    payload["noahmp_irrigation_schema_version"] = 1
    payload["noahmp_carbon_schema_version"] = 1
    payload["noahmp_groundwater_schema_version"] = 1
    payload['noahmp_tile_drain_schema_version']=1
    return payload


def _validate_radiation_diagnostics_schema(payload: dict[str, Any], *, namelist=None,
                                           topographic_shadow_required=None) -> None:
    from gpuwrf.diagnostics.radiation import validate_restart as validate_rad_diag
    diagnostics = payload.get("extra_fields", {}).get("radiation_diagnostics")
    # The carry serializer is also used without a requested namelist. Presence
    # still requires the schema marker; _read_payload checks the saved selector.
    config = namelist if namelist is not None else {"rad_diag": int(getattr(diagnostics, "rad_diag", None) is not None)}
    validate_rad_diag(diagnostics, config, payload.get("rad_diag_schema_version"),
                      getattr(payload.get("state_fields", {}).get("theta"), "shape", None))
    from gpuwrf.io.flg_gfdl_schema import validate
    extra=payload.get('extra_fields',{})
    validate(extra.get('radiation_diagnostics'),extra.get('gfdl_radiation'),namelist,
        payload.get('flg_gfdl_schema'),state_shape=getattr(payload.get('state_fields',{}).get('qv'),'shape',None))
    from gpuwrf.io.restart_schema import validate_goddard_radiation
    validate_goddard_radiation(payload.get('extra_fields',{}).get('radiation_diagnostics'),namelist,
                              schema_version=payload.get('goddard_radiation_schema_version'),
                              state_shape=getattr(payload.get('state_fields',{}).get('qv'),'shape',None))
    from gpuwrf.io.topographic_shadow_schema import validate as validate_shadow
    validate_shadow(payload.get("extra_fields", {}).get("radiation_diagnostics"),
        payload.get("topographic_shadow_schema_version"),
        getattr(payload.get("state_fields", {}).get("t_skin"), "shape", None),
        namelist=namelist, required=topographic_shadow_required)
    if payload.get("extra_fields", {}).get("radiation_diagnostics") is None:
        return  # Legacy dycore-only restarts have no held radiation to lose.
    if payload.get("radiation_diagnostics_schema_version") != RADIATION_DIAGNOSTICS_SCHEMA_VERSION:
        raise ValueError("restart radiation diagnostics schema predates held cloud_fraction (E78)")


def _validate_gwdo_diagnostics_schema(payload: dict[str, Any]) -> None:
    # An old checkpoint can omit these optional leaves even on an active GWDO
    # domain. They cannot be recomputed from its post-RK state: WRF used the
    # preceding pre-PBL inputs. Refuse rather than silently emit zeros (E78).
    if payload.get("gwdo_diagnostics_schema_version") != GWDO_DIAGNOSTICS_SCHEMA_VERSION:
        raise ValueError("restart GWDO diagnostics schema predates held DTAUX3D/DTAUY3D and DUSFCG/DVSFCG (E78)")


def write_restart(
    carry: OperationalCarry,
    namelist: Any,
    grid: Any,
    step_index: int,
    path: str | Path,
    *,
    noahmp_static: Any | None = None,
    extra_metadata: dict[str, Any] | None = None,
    feedback_schema: dict[str, Any] | None = None,
) -> Path:
    """Write a full-carry ``wrfrst``-equivalent restart file.

    Serializes the COMPLETE :class:`OperationalCarry` (prognostic ``State`` + WRF
    small-step scratch + held ``rthraten`` + Noah-MP land carry + held
    ``noahmp_rad``) plus ``grid``, ``namelist`` and ``step_index``. ``noahmp_static``
    (the read-only per-run Noah-MP inputs / parameter tables) is optionally stored
    so a resumed run can rebuild the land driver without re-reading ``wrfinput``;
    it is host-copied like everything else. All arrays land as host numpy, so the
    write is GPU-free.
    """

    _reject_unsupported_carry(carry)
    validate_epsilon_state(carry.state,namelist,EPSILON_PBL_SCHEMA_VERSION)
    from gpuwrf.io.urban_schema import validate_urban_inputs
    validate_urban_inputs(carry.urban_slucm,namelist,1,carry.state.t_skin.shape)
    from gpuwrf.io.cuplus_schema import validate as validate_cuplus
    validate_cuplus(carry.cuplus,namelist,carry.state.theta.shape,1 if carry.cuplus is not None else None)
    from gpuwrf.io.cu_forcing_schema import validate as validate_cu_forcing,SCHEMA_VERSION as CU_FORCING_SCHEMA_VERSION
    validate_cu_forcing(carry.cu_forcing,namelist,carry.state.theta.shape,
                        CU_FORCING_SCHEMA_VERSION if carry.cu_forcing is not None else None)
    from gpuwrf.io.noahmp_mmf import validate_restart as validate_mmf
    validate_mmf(carry.noahmp_mmf, namelist, 1 if carry.noahmp_mmf is not None else None,
                 carry.state.t_skin.shape)
    validate_held_state({name: getattr(carry.state, name) for name in carry.state.active_field_names()}, namelist)
    validate_camuw(carry.camuw_pbl, namelist, CAMUW_SCHEMA_VERSION)
    from gpuwrf.io.restart_schema import validate_goddard_radiation
    validate_goddard_radiation(carry.radiation_diagnostics,namelist,
        GODDARD_RADIATION_SCHEMA_VERSION if type(carry.radiation_diagnostics).__name__=='GoddardRadiationDiagnostics' else None)
    from gpuwrf.io.flg_gfdl_schema import validate,marker
    validate(carry.radiation_diagnostics,carry.gfdl_radiation,namelist,
             marker(carry.radiation_diagnostics,carry.gfdl_radiation),state_shape=carry.state.qv.shape)
    validate_temf(carry.temf,namelist,TEMF_SCHEMA_VERSION)
    from gpuwrf.io.restart_schema import validate_sas_held
    validate_sas_held(carry.cumulus_tendencies, namelist, carry.osas_rng_key, carry.cumulus_carry)
    validate_lake(carry.lake,namelist,LAKE_SCHEMA_VERSION)
    from gpuwrf.io.ocean_schema import validate as validate_ocean,configuration as ocean_configuration
    validate_ocean(carry.ocean,carry.hurricane_ustm,namelist,1,shape=carry.state.t_skin.shape,history=carry.history_diagnostics,require_history=True)
    from gpuwrf.io.restart_schema import validate_ssib
    validate_ssib(carry.ssib_land,carry.ssib_rad,namelist,1,carry.state.t_skin.shape,
        driver_fields=dict(wspd=carry.state.sfc_wspd,history=carry.history_diagnostics,precipitation=carry.noahmp_precipitation))
    validate_qnse(carry.qnse,namelist,QNSE_SCHEMA_VERSION,history=carry.history_diagnostics,expected_shape=carry.state.theta.shape)
    from gpuwrf.io.restart_schema import validate_spectral_fdda
    validate_spectral_fdda(carry.fdda, namelist)

    from gpuwrf.io.restart_schema import validate_observations
    validate_observations(carry.obs, namelist)
    from gpuwrf.diagnostics.precipitation import configured_interval, validate_restart_buckets
    validate_restart_buckets(carry.precipitation_buckets, configured_interval(namelist), 1, carry.state.t_skin.shape)
    from gpuwrf.diagnostics.online import configuration, validate_restart
    validate_restart(carry.nwp_diagnostics,carry.radar_reflectivity,configuration(namelist),1,carry.state.qv.shape)
    from gpuwrf.diagnostics.afwa import configuration as afwa_configuration, validate_restart_afwa
    validate_restart_afwa(carry.afwa_diagnostics, afwa_configuration(namelist)["afwa_severe_opt"], 1,
                          carry.state.t_skin.shape, history=carry.history_diagnostics, check_history=True)
    from gpuwrf.diagnostics.solar_config import validate_options as solar_options, validate_restart as validate_solar_restart
    solar_config = solar_options(namelist)
    validate_solar_restart(carry.solar_interpolation, solar_config, 1, carry.state.t_skin.shape, requested=namelist)
    from gpuwrf.io.restart_schema import validate_urban_bep,validate_urban_radiation
    validate_urban_bep(carry.urban_bep,namelist)
    validate_urban_radiation(carry.radiation_diagnostics,namelist)
    from gpuwrf.tracers.restart import STATE_FIELDS, configuration as tracer_configuration, validate_restart as validate_tracer_restart
    tracer_config = tracer_configuration(namelist)
    validate_tracer_restart({name: getattr(carry.state, name) for name in STATE_FIELDS},
                           carry.tracer_tendencies, tracer_config, 1, carry.state.qv.shape)
    from gpuwrf.coupling.sst_skin_hook import validate_restart as validate_skin
    validate_skin(carry.sst_skin, getattr(namelist, "sst_skin", 0), 1, carry.state.t_skin.shape)
    from gpuwrf.io.noahmp_options import irrigation_configuration
    from gpuwrf.io.noahmp_irrigation import validate_restart as validate_irrigation
    irrigation_config=irrigation_configuration(namelist)
    validate_irrigation(carry.noahmp_irrigation,irrigation_config,1,carry.state.t_skin.shape)
    from gpuwrf.io.noahmp_carbon import validate_restart as validate_carbon
    validate_carbon(carry.noahmp_carbon,namelist,1,carry.state.t_skin.shape)
    from gpuwrf.io.noahmp_groundwater import validate_restart as validate_groundwater
    validate_groundwater(carry.noahmp_groundwater,namelist,1,carry.state.t_skin.shape)
    from gpuwrf.io.noahmp_hydrology import validate_tile_restart
    validate_tile_restart(carry.noahmp_tile_drain,namelist,1,carry.state.t_skin.shape)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    host_grid = _hostify_tree(grid)
    from gpuwrf.io.mynn_budget_options import configuration as mynn_configuration
    payload: dict[str, Any] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "carry_type": "gpuwrf.runtime.operational_state.OperationalCarry",
        "carry": _carry_to_payload(carry,mynn_config=mynn_configuration(namelist)),
        "namelist": _hostify_namelist(namelist, host_grid),
        "grid": host_grid,
        "step_index": int(step_index),
        "noahmp_static": None if noahmp_static is None else _hostify_static(noahmp_static),
        "metadata": dict(extra_metadata) if extra_metadata else {},
    }
    if physics_option(namelist, "cycling", 0):
        from gpuwrf.io.restart_schema import CYCLING_SCHEMA_VERSION
        payload["cycling_schema_version"] = CYCLING_SCHEMA_VERSION
    from gpuwrf.io.noahmp_options import restart_configuration
    payload["noahmp_options"]=restart_configuration(namelist)
    from gpuwrf.io.noahmp_restart_identity import bound_table_identity
    payload["noahmp_bound_tables"]=bound_table_identity(namelist)
    from gpuwrf.surface_params.controls import restart_controls_digest
    payload["surface_controls_digest"]=restart_controls_digest(namelist)
    from gpuwrf.runtime.restart_parameters import parameter_digests
    parameters = parameter_digests(namelist)
    if parameters:
        payload["runtime_parameter_digests"] = parameters
    payload["carry"]["solar_interpolation_config"] = solar_config
    from gpuwrf.io.glacier_schema import validate as validate_glacier
    validate_glacier(payload['carry'].get('noahmp_land_fields'),payload['carry'].get('noahmp_land_field_order') or (),
        payload['carry'].get('glacier_schema_version'),namelist=namelist)
    from gpuwrf.io.feedback_restart import validate_payload
    validate_payload(payload['carry'],expected=feedback_schema)
    from gpuwrf.io.windfarm_schema import configuration as wind_config, validate_restart as validate_wind
    payload['carry']['windfarm_config'] = wind_config(namelist)
    validate_wind(carry.windfarm_power,payload['carry']['windfarm_config'],carry.state.t_skin.shape,requested=namelist)
    payload["carry"]["tracer_config"] = tracer_config
    payload["carry"]["ocean_config"] = ocean_configuration(namelist)
    payload["carry"]["sst_skin_config"] = int(getattr(namelist, "sst_skin", 0))
    payload["carry"]["noahmp_irrigation_config"]=irrigation_config
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".", suffix=".tmp", delete=False) as handle:
        tmp = Path(handle.name)
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(target)
    directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return target


def _hostify_static(static: Any) -> dict[str, Any]:
    """Host-copy NoahMPStatic field-by-field (arrays hostified; non-arrays kept).

    ``parameters`` (the table bundle) and scalars like ``dx_m`` are kept as-is; the
    field-named dict keeps the read fail-closed against a NoahMPStatic schema change.
    """
    if NoahMPStatic is None:
        raise ValueError("NoahMPStatic unavailable; cannot serialize Noah-MP static inputs")
    fields: dict[str, Any] = {}
    for name in NoahMPStatic.__slots__:
        value = getattr(static, name)
        fields[name] = _hostify_tree(value) if name not in ("parameters","glacier") else value
    return {"field_order": list(NoahMPStatic.__slots__), "fields": fields}


def _read_payload(path: str | Path, *, namelist=None, feedback_schema=None) -> dict[str, Any]:
    source = Path(path)
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("format") != FORMAT:
        raise ValueError(f"unsupported restart format {payload.get('format')!r} in {source}")
    version = int(payload.get("format_version", -1))
    if version not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            f"unsupported restart version {payload.get('format_version')} in {source}; "
            f"supported {SUPPORTED_FORMAT_VERSIONS}"
        )
    from gpuwrf.runtime.restart_parameters import validate_parameters
    validate_parameters(payload.get("runtime_parameter_digests", {}), payload["namelist"])
    if namelist is not None:
        validate_parameters(payload.get("runtime_parameter_digests", {}), namelist)

    from gpuwrf.io.restart_schema import validate_cycling_restart
    validate_cycling_restart(payload["namelist"], payload.get("cycling_schema_version"))
    if namelist is not None:
        validate_cycling_restart(namelist, payload.get("cycling_schema_version"))

    carry = payload["carry"]
    from gpuwrf.io.noahmp_mmf import validate_restart as validate_mmf
    validate_mmf(carry.get('extra_fields', {}).get('noahmp_mmf'), payload['namelist'],
                 carry.get('noahmp_mmf_schema_version'), carry['state_fields']['t_skin'].shape)
    from gpuwrf.io.noahmp_mmf import validate_land as validate_mmf_land
    validate_mmf_land(carry.get('extra_fields', {}).get('noahmp_mmf'),carry.get('noahmp_land_fields'),
                      carry['state_fields']['t_skin'].shape)
    if namelist is not None:
        validate_mmf(carry.get('extra_fields', {}).get('noahmp_mmf'), namelist,
                     carry.get('noahmp_mmf_schema_version'), carry['state_fields']['t_skin'].shape)
    from gpuwrf.io.feedback_restart import validate_payload
    validate_payload(carry,expected=feedback_schema)
    validate_epsilon_state(carry['state_fields'],payload['namelist'],carry.get('epsilon_pbl_schema_version'))
    validate_epsilon_history(carry.get('extra_fields',{}).get('history_diagnostics'),payload['namelist'],
                             carry['state_fields']['t_skin'].shape)
    from gpuwrf.io.windfarm_schema import validate_restart as validate_wind
    validate_wind(carry.get('extra_fields',{}).get('windfarm_power'),carry.get('windfarm_config'),
                  carry['state_fields']['t_skin'].shape,requested=namelist if namelist is not None else payload['namelist'])
    _validate_radiation_diagnostics_schema(carry, namelist=payload.get("namelist"))
    if namelist is not None:
        _validate_radiation_diagnostics_schema(carry, namelist=namelist)
    from gpuwrf.io.restart_schema import validate_goddard_radiation
    validate_goddard_radiation(carry.get('extra_fields',{}).get('radiation_diagnostics'),payload['namelist'],
                              carry.get('goddard_radiation_schema_version'))
    if version < 5 and carry.get("noahmp_land_fields") is not None:
        raise ValueError("restart predates held Noah-MP precipitation (E78); cannot resume land physics")
    if version < 6 and carry.get("extra_fields", {}).get("noahclassic_land") is not None:
        raise ValueError("restart predates the held precipitation SR channel (E78); cannot resume Noah-classic")
    if version < 3:
        from gpuwrf.runtime.history_accumulators import full_history_enabled
        if full_history_enabled() and carry.get("noahmp_land_fields") is not None:
            raise ValueError("restart predates resident full-history accumulators; cannot resume full history")
    _validate_gwdo_diagnostics_schema(carry)
    from gpuwrf.io.urban_schema import validate_urban_inputs
    validate_urban_inputs(carry.get("extra_fields",{}).get("urban_slucm"),payload["namelist"],
        carry.get("urban_inputs_schema_version"),carry["state_fields"]["t_skin"].shape)
    if namelist is not None:
        validate_urban_inputs(carry.get("extra_fields",{}).get("urban_slucm"),namelist,
            carry.get("urban_inputs_schema_version"),carry["state_fields"]["t_skin"].shape)
    validate_camuw(carry.get("extra_fields", {}).get("camuw_pbl"), payload["namelist"],
                   carry.get("camuw_pbl_schema_version"))
    validate_lake(carry.get('extra_fields',{}).get('lake'),payload['namelist'],carry.get('lake_schema_version'))
    from gpuwrf.io.ocean_schema import validate as validate_ocean
    extra=carry.get('extra_fields',{})
    validate_ocean(extra.get('ocean'),extra.get('hurricane_ustm'),payload['namelist'],carry.get('ocean_schema_version'),shape=carry['state_fields']['t_skin'].shape,history=extra.get('history_diagnostics'),require_history=True)
    if namelist is not None:
        validate_ocean(extra.get('ocean'),extra.get('hurricane_ustm'),namelist,carry.get('ocean_schema_version'),shape=carry['state_fields']['t_skin'].shape,saved=carry.get('ocean_config',payload['namelist']),history=extra.get('history_diagnostics'),require_history=True)
    validate_temf(carry.get('extra_fields',{}).get('temf'),payload['namelist'],carry.get('temf_schema_version'))
    from gpuwrf.io.restart_schema import validate_ssib
    extra=carry.get('extra_fields',{})
    validate_ssib(extra.get('ssib_land'),extra.get('ssib_rad'),payload['namelist'],
                  carry.get('ssib_schema_version'),carry['state_fields']['t_skin'].shape,
                  driver_fields=dict(wspd=carry['state_fields'].get('sfc_wspd'),history=extra.get('history_diagnostics'),precipitation=extra.get('noahmp_precipitation')))
    validate_qnse(carry.get('extra_fields',{}).get('qnse'),payload['namelist'],carry.get('qnse_schema_version'),
                  history=carry.get('extra_fields',{}).get('history_diagnostics'),expected_shape=carry['state_fields']['theta'].shape)
    from gpuwrf.io.restart_schema import validate_sas_held
    validate_sas_held(carry.get('extra_fields', {}).get('cumulus_tendencies'), payload['namelist'], carry.get('extra_fields', {}).get('osas_rng_key'), carry.get('extra_fields', {}).get('cumulus_carry'))
    from gpuwrf.io.restart_schema import validate_spectral_fdda
    validate_spectral_fdda(carry.get("extra_fields", {}).get("fdda"), payload["namelist"])

    from gpuwrf.io.restart_schema import validate_observations
    validate_observations(carry.get("extra_fields", {}).get("obs"),payload["namelist"])
    from gpuwrf.diagnostics.precipitation import configured_interval, validate_restart_buckets
    validate_restart_buckets(carry.get("extra_fields", {}).get("precipitation_buckets"),
                            configured_interval(payload["namelist"]),
                            carry.get("precipitation_buckets_schema_version"), carry["state_fields"]["t_skin"].shape)
    from gpuwrf.diagnostics.online import configuration, validate_restart
    validate_restart(carry.get("extra_fields",{}).get("nwp_diagnostics"),
                     carry.get("extra_fields",{}).get("radar_reflectivity"),configuration(payload["namelist"]),
                     carry.get("online_diagnostics_schema_version"),carry["state_fields"]["qv"].shape)
    from gpuwrf.diagnostics.afwa import (configuration as afwa_configuration, history_interval as afwa_interval,
                                         validate_restart_afwa)
    for afwa_namelist in (payload["namelist"],) + (() if namelist is None else (namelist,)):
        # The maximum's window is the saved history cadence: a changed cadence is refused (critic FIX 6).
        validate_restart_afwa(carry.get("extra_fields", {}).get("afwa_diagnostics"),
                              afwa_configuration(afwa_namelist)["afwa_severe_opt"],
                              carry.get("afwa_diagnostics_schema_version"), carry["state_fields"]["t_skin"].shape,
                              history=carry.get("extra_fields", {}).get("history_diagnostics"), check_history=True,
                              saved_interval=afwa_interval(payload["namelist"]),
                              requested_interval=afwa_interval(afwa_namelist))
    from gpuwrf.diagnostics.mynn_budget import validate_payload as validate_mynn_payload
    validate_mynn_payload(carry,payload['namelist'])
    if namelist is not None:validate_mynn_payload(carry,namelist)
    from gpuwrf.io.mynn_cloudpdf_schema import validate as validate_cloudpdf
    validate_cloudpdf(carry["state_fields"], payload["namelist"], namelist)
    from gpuwrf.diagnostics.solar_config import validate_restart as validate_solar_restart
    validate_solar_restart(carry.get("extra_fields", {}).get("solar_interpolation"),
                           carry.get("solar_interpolation_config"),
                           carry.get("solar_interpolation_schema_version"),
                           carry["state_fields"]["t_skin"].shape, requested=payload["namelist"])
    if namelist is not None:
        validate_solar_restart(carry.get("extra_fields", {}).get("solar_interpolation"),
                               carry.get("solar_interpolation_config"),
                               carry.get("solar_interpolation_schema_version"),
                               carry["state_fields"]["t_skin"].shape, requested=namelist)
    from gpuwrf.tracers.restart import configuration as tracer_configuration, validate_restart as validate_tracer_restart
    tracer_config = carry.get("tracer_config")
    validate_tracer_restart(carry["state_fields"], carry.get("extra_fields", {}).get("tracer_tendencies"),
                           tracer_config, carry.get("tracer_schema_version"), carry["state_fields"]["qv"].shape,
                           requested=payload["namelist"])
    if namelist is not None:
        validate_tracer_restart(carry["state_fields"], carry.get("extra_fields", {}).get("tracer_tendencies"),
                               tracer_config, carry.get("tracer_schema_version"), carry["state_fields"]["qv"].shape,
                               requested=namelist)
    from gpuwrf.coupling.sst_skin_hook import validate_restart as validate_skin
    skin_group = carry.get("extra_fields", {}).get("sst_skin")
    saved_skin = carry.get("sst_skin_config", 0)
    validate_skin(skin_group, saved_skin, carry.get("sst_skin_schema_version"), carry["state_fields"]["t_skin"].shape)
    validate_skin(skin_group, getattr(payload["namelist"], "sst_skin", 0), carry.get("sst_skin_schema_version"),
                  carry["state_fields"]["t_skin"].shape)
    if namelist is not None:
        validate_skin(skin_group, getattr(namelist, "sst_skin", 0), carry.get("sst_skin_schema_version"),
                      carry["state_fields"]["t_skin"].shape)
    if version >= 2:
        from gpuwrf.io.noahmp_groundwater import validate_restart as validate_groundwater
        from gpuwrf.io.noahmp_hydrology import validate_tile_restart
        validate_tile_restart(carry.get('extra_fields',{}).get('noahmp_tile_drain'),payload['namelist'],
            carry.get('noahmp_tile_drain_schema_version'),carry['state_fields']['t_skin'].shape)
        if namelist is not None:
            validate_tile_restart(carry.get('extra_fields',{}).get('noahmp_tile_drain'),namelist,
                carry.get('noahmp_tile_drain_schema_version'),carry['state_fields']['t_skin'].shape)
        validate_groundwater(carry.get("extra_fields",{}).get("noahmp_groundwater"),payload["namelist"],
                            carry.get("noahmp_groundwater_schema_version"),carry["state_fields"]["t_skin"].shape)
        if namelist is not None:
            validate_groundwater(carry.get("extra_fields",{}).get("noahmp_groundwater"),namelist,
                                carry.get("noahmp_groundwater_schema_version"),carry["state_fields"]["t_skin"].shape)
        from gpuwrf.io.noahmp_carbon import validate_restart as validate_carbon
        validate_carbon(carry.get("extra_fields",{}).get("noahmp_carbon"),payload["namelist"],
                        carry.get("noahmp_carbon_schema_version"),carry["state_fields"]["t_skin"].shape)
        if namelist is not None:
            validate_carbon(carry.get("extra_fields",{}).get("noahmp_carbon"),namelist,
                            carry.get("noahmp_carbon_schema_version"),carry["state_fields"]["t_skin"].shape)
        from gpuwrf.io.noahmp_options import validate_option_restart
        validate_option_restart(payload.get("noahmp_options"),payload["namelist"])
        if namelist is not None:validate_option_restart(payload.get("noahmp_options"),namelist)
        from gpuwrf.io.noahmp_restart_identity import validate_bound_table_restart
        validate_bound_table_restart(payload.get("noahmp_bound_tables"),payload.get("namelist"))
        if namelist is not None:validate_bound_table_restart(payload.get("noahmp_bound_tables"),namelist)
        from gpuwrf.surface_params.controls import validate_controls_restart
        validate_controls_restart(payload.get("surface_controls_digest"),namelist)
        from gpuwrf.io.noahmp_options import irrigation_configuration
        from gpuwrf.io.noahmp_irrigation import validate_restart as validate_irrigation
        group=carry.get("extra_fields",{}).get("noahmp_irrigation")
        cfg=carry.get("noahmp_irrigation_config",(0,0))
        validate_irrigation(group,cfg,carry.get("noahmp_irrigation_schema_version"),carry["state_fields"]["t_skin"].shape,
                            requested=irrigation_configuration(payload["namelist"]))
        if namelist is not None:
            validate_irrigation(group,cfg,carry.get("noahmp_irrigation_schema_version"),carry["state_fields"]["t_skin"].shape,
                                requested=irrigation_configuration(namelist))
        recorded_carry = tuple(carry.get("carry_field_order", ()))
        # Only the newly appended, inactive group can be absent in older files.
        expected_carry = tuple(name for name in _CARRY_FIELD_ORDER
                               if name not in ("camuw_pbl","lake","obs","regional_streams", "precipitation_buckets", "nwp_diagnostics", "radar_reflectivity", "solar_interpolation", "temf", "urban_bep", "feedback_history", "windfarm_power", "tracer_tendencies", "ocean", "hurricane_ustm", "qnse", "osas_rng_key", "cuplus", "noahmp_mmf", "gfdl_radiation", "sst_skin", "noahmp_irrigation", "noahmp_carbon", "noahmp_groundwater", "noahmp_tile_drain", "cu_forcing", "aerod", "taod5502d", "mynn_budget", "afwa_diagnostics") or name in recorded_carry)
        if recorded_carry != expected_carry:
            raise ValueError("restart OperationalCarry field order does not match current schema")
        expected_extra = set(_V1_UNSUPPORTED_CARRY_FIELDS)
        if "camuw_pbl" not in recorded_carry:
            expected_extra.remove("camuw_pbl")
        if 'lake' not in recorded_carry:
            expected_extra.remove('lake')
        if "obs" not in recorded_carry:
            expected_extra.remove("obs")
        if 'regional_streams' not in recorded_carry:
            expected_extra.remove('regional_streams')
        if "precipitation_buckets" not in recorded_carry:
            expected_extra.remove("precipitation_buckets")
        for group in ("nwp_diagnostics","radar_reflectivity","solar_interpolation","windfarm_power", "tracer_tendencies", 'sst_skin', 'noahmp_irrigation', 'noahmp_carbon', 'noahmp_groundwater', 'noahmp_tile_drain', 'cu_forcing', 'aerod', 'taod5502d', 'mynn_budget', 'afwa_diagnostics'):
            if group not in recorded_carry:
                expected_extra.remove(group)
        if 'temf' not in recorded_carry:
            expected_extra.remove('temf')
        if 'urban_bep' not in recorded_carry:
            expected_extra.remove('urban_bep')
        if 'feedback_history' not in recorded_carry:
            expected_extra.remove('feedback_history')
        for group in ('ocean','hurricane_ustm'):
            if group not in recorded_carry:
                expected_extra.remove(group)
        if 'qnse' not in recorded_carry:
            expected_extra.remove('qnse')
        if 'osas_rng_key' not in recorded_carry:
            expected_extra.remove('osas_rng_key')
        if 'cuplus' not in recorded_carry:
            expected_extra.remove('cuplus')
        if 'noahmp_mmf' not in recorded_carry:
            expected_extra.remove('noahmp_mmf')
        if 'gfdl_radiation' not in recorded_carry:
            expected_extra.remove('gfdl_radiation')
        if set(carry.get("extra_fields", {})) != expected_extra:
            raise ValueError("restart OperationalCarry extra fields do not match current schema")
    from gpuwrf.io.restart_schema import validate_urban_bep,validate_urban_radiation
    from gpuwrf.io.cuplus_schema import validate as validate_cuplus
    validate_cuplus(carry.get('extra_fields',{}).get('cuplus'),payload['namelist'],
                   carry['state_fields']['theta'].shape,carry.get('cuplus_schema_version'))
    if namelist is not None:
        validate_cuplus(carry.get('extra_fields',{}).get('cuplus'),namelist,
                       carry['state_fields']['theta'].shape,carry.get('cuplus_schema_version'))
    from gpuwrf.io.cu_forcing_schema import validate as validate_cu_forcing
    validate_cu_forcing(carry.get('extra_fields',{}).get('cu_forcing'),payload['namelist'],
                        carry['state_fields']['theta'].shape,carry.get('cu_forcing_schema_version'))
    if namelist is not None:
        validate_cu_forcing(carry.get('extra_fields',{}).get('cu_forcing'),namelist,
                            carry['state_fields']['theta'].shape,carry.get('cu_forcing_schema_version'))
    validate_urban_bep(carry.get('extra_fields',{}).get('urban_bep'),payload['namelist'])
    validate_urban_radiation(carry.get('extra_fields',{}).get('radiation_diagnostics'),payload['namelist'])
    # Fail-closed schema checks (exact field-order + field-set match), so a State /
    # scratch / land schema drift raises instead of mis-reconstructing the carry.
    recorded_state_order = tuple(carry.get("state_field_order", ()))
    _validate_state_field_order(recorded_state_order, namelist=payload["namelist"])
    if set(carry.get("state_fields", {})) != set(recorded_state_order):
        raise ValueError("restart State fields do not match current State schema")
    validate_held_state(carry["state_fields"], payload["namelist"])
    if tuple(carry.get("scratch_field_order", ())) != _CARRY_SCRATCH_FIELDS:
        raise ValueError("restart carry-scratch field order does not match current schema")
    if set(carry.get("scratch_fields", {})) != set(_CARRY_SCRATCH_FIELDS):
        raise ValueError("restart carry-scratch fields do not match current schema")
    land_fields = carry.get("noahmp_land_fields")
    if land_fields is not None:
        if NoahMPLandState is None:
            raise ValueError("restart carries Noah-MP land state but NoahMPLandState is unavailable")
        from gpuwrf.io.glacier_schema import validate
        validate(land_fields,carry.get('noahmp_land_field_order',()),carry.get('glacier_schema_version'),namelist=payload['namelist'])
    else:
        from gpuwrf.io.glacier_schema import validate
        validate(None,(),carry.get('glacier_schema_version'),namelist=payload['namelist'])
    return payload


def _restore_carry(carry_payload: dict[str, Any]) -> OperationalCarry:
    from gpuwrf.diagnostics.mynn_budget import validate_payload as validate_mynn_payload
    validate_mynn_payload(carry_payload)
    from gpuwrf.io.feedback_restart import validate_payload
    validate_payload(carry_payload)
    validate_epsilon_history(carry_payload.get('extra_fields',{}).get('history_diagnostics'),
        {'bl_pbl_physics':16 if carry_payload['state_fields'].get('epsilon_pbl') is not None else 0},
        carry_payload['state_fields']['t_skin'].shape)
    # Verbatim pytree rebuild (``tree_unflatten``), NOT ``State(**fields)``: the
    # constructor re-canonicalises leaves to the fp32-gated default matrix and
    # would silently DOWNCAST the fp64 boundary/number leaves a ``force_fp64``
    # operational run carries.  The fail-closed field-order check above already
    # guarantees every non-conditional leaf is present; conditional leaves absent
    # from the payload stay ``None`` exactly as they were on the written State.
    stored = {
        name: jax.device_put(carry_payload["state_fields"][name])
        for name in carry_payload["state_field_order"]
    }
    state = State.tree_unflatten(None, tuple(stored.get(name) for name in State.__slots__))
    scratch = {
        name: jax.device_put(carry_payload["scratch_fields"][name])
        for name in _CARRY_SCRATCH_FIELDS
    }
    noahmp_land = None
    if carry_payload.get("noahmp_land_fields") is not None:
        noahmp_land = NoahMPLandState(
            **{
                name: jax.device_put(carry_payload["noahmp_land_fields"].get(name))
                for name in NoahMPLandState.__slots__
            }
        )
    noahmp_rad = None
    if carry_payload.get("noahmp_rad") is not None:
        noahmp_rad = tuple(jax.device_put(component) for component in carry_payload["noahmp_rad"])
    extra = {name: _device_tree(value) for name, value in carry_payload.get("extra_fields", {}).items()}
    if extra.get("regional_streams") is not None or carry_payload.get("regional_streams_schema_version") is not None:
        from gpuwrf.streams.restart import validate_restart
        validate_restart(extra.get("regional_streams"), carry_payload.get("regional_streams_schema_version"))
    if extra.get("land_history") is not None or extra.get("energy_accumulators") is not None:
        # Checkpoints written before the packed history families hold per-name dicts.
        from gpuwrf.runtime.history_accumulators import ENERGY_ACCUMULATORS, LAND_FLUX_FIELDS, as_packed
        extra["land_history"] = as_packed(extra.get("land_history"), LAND_FLUX_FIELDS)
        extra["energy_accumulators"] = as_packed(extra.get("energy_accumulators"), ENERGY_ACCUMULATORS)
    return OperationalCarry(
        state=state,
        noahmp_land=noahmp_land,
        noahmp_rad=noahmp_rad,
        **scratch,
        **extra,
    )


def read_restart(path: str | Path, *, namelist=None, feedback_schema=None) -> tuple[OperationalCarry, Any, Any, int]:
    """Read a full-carry restart and reconstruct the carry pytree on device.

    Returns ``(carry, namelist, grid, step_index)``. The carry is the COMPLETE
    :class:`OperationalCarry` (State + scratch + held rthraten + Noah-MP land/rad)
    so the operational driver can resume the scan with no re-seeding.
    An optional requested ``namelist`` additionally checks the exact Solar
    selection and radiation clocks against the saved configuration.
    """
    payload = _read_payload(path, namelist=namelist,feedback_schema=feedback_schema)
    carry = _restore_carry(payload["carry"])
    grid = _device_tree(payload["grid"])
    namelist = _restore_namelist(payload["namelist"], grid)
    return carry, namelist, grid, int(payload["step_index"])


def read_restart_metadata(path: str | Path) -> dict[str, Any]:
    """Read only the (host) provenance/metadata block + schema header.

    Lets a manager/inspector check a restart's step index, version and carry
    coverage WITHOUT materializing arrays on a device (GPU-free).
    """
    payload = _read_payload(path)
    carry = payload["carry"]
    return {
        "format": payload["format"],
        "format_version": int(payload["format_version"]),
        "step_index": int(payload["step_index"]),
        "has_noahmp_land": carry.get("noahmp_land_fields") is not None,
        "has_noahmp_rad": carry.get("noahmp_rad") is not None,
        "has_noahmp_static": payload.get("noahmp_static") is not None,
        "state_field_count": len(carry.get("state_fields", {})),
        "scratch_field_count": len(carry.get("scratch_fields", {})),
        "metadata": payload.get("metadata", {}),
    }


__all__ = [
    "FORMAT",
    "FORMAT_VERSION",
    "SUPPORTED_FORMAT_VERSIONS",
    "UNSUPPORTED_CARRY_FIELDS",
    "read_restart",
    "read_restart_metadata",
    "write_restart",
]
