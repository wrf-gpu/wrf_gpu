"""Pickle checkpoints for M7 restart-continuity probes."""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

from dataclasses import is_dataclass, replace
from pathlib import Path
import pickle
from typing import Any

import jax
from jax import config
import numpy as np

from gpuwrf.contracts.state import SCALAR_BOUNDARY_OPTIONAL_LEAVES, EXTRA_SCALAR_BOUNDARY_SPECIES, P3_STATE_LEAVES, State
from gpuwrf.contracts.precision import GWDO_DIAGNOSTIC_LEAVES, LES_TKE_LEAVES, MP_RE_DIAGNOSTIC_LEAVES, SMS_CARRY_LEAVES, FERRIER_CARRY_LEAVES, GODDARD4ICE_CARRY_LEAVES, GODDARD4ICE_HISTORY_LEAVES, THOMPSON_HAIL_SURFACE_LEAVES, EPSILON_PBL_LEAVES, JENSEN_STATE_LEAVES, NTU_STATE_LEAVES, ICE_DIAGNOSTIC_LEAVES, les_tke_active, MYNN_CLOUDPDF_LEAVES, mynn_cloudpdf_leaves
from gpuwrf.io.pbl_epsilon_schema import EPSILON_PBL_SCHEMA_VERSION, epsilon_pbl_active, validate_epsilon_state
from gpuwrf.contracts.cammg_fields import STATE_FIELDS as CAMMG_STATE_FIELDS
from gpuwrf.physics.cammg.restart import validate as validate_cammg
from gpuwrf.io.restart_schema import validate_held_state,QNSE_SCHEMA_VERSION,validate_qnse
from gpuwrf.contracts.precision import WRITER_PROFILE_LEAVES

try:  # NoahMPLandState lives in the v0.2.0 land package; import is optional so a
    # pure-dycore checkout without the noahmp package can still read/write v1.
    from gpuwrf.contracts.noahmp_state import NoahMPLandState
except Exception:  # pragma: no cover
    NoahMPLandState = None  # type: ignore


configure_jax_x64()

# v2 (ADR-NOAHMP-INTERFACES.md §5): adds the optional prognostic Noah-MP land
# carry + a scope-options guard. v1 checkpoints (no land state) stay READABLE and
# load with ``noahmp_land_state = None`` -> driver cold-inits land from wrfinput.
# v3 (v0.6.0 S0 State-leaf materialization): appends the additive physics State
# leaves Nc/Nn/rainc_acc. A v1/v2 checkpoint recorded the pre-v0.6.0 leaf order
# (no Nc/Nn/rainc_acc); it stays READABLE because the new leaves are APPEND-ONLY
# -- the reader backfills any missing additive leaf with zeros (cold-start), and
# requires the recorded order to be a prefix of the current schema (fail-closed
# on any non-prefix divergence). A v3 writer always records the full order.
FORMAT_VERSION = 3
SUPPORTED_FORMAT_VERSIONS = (1, 2, 3)
# This State-only checkpoint format has its own held-GWDO schema marker.
# Inactive legacy checkpoints remain readable; active ones need all four leaves.
GWDO_CHECKPOINT_SCHEMA_VERSION = 1

# State leaves added after the original (v1/v2) checkpoint schema. The reader
# backfills any of these absent from an older checkpoint with zeros, so old
# restarts cold-start the new physics fields rather than failing closed.
# v0.6.0 added Nc/Nn/rainc_acc; v0.15 the MYNN SGS-cloud leaves; v0.17 ADR-032
# the graupel/hail substrate qh/Nh/qvolg/qvolh; v0.16 the aerosol-aware Thompson
# leaves nwfa/nifa; v0.17 hail microphysics the hail_acc surface accumulator;
# v0.22 added optional standalone wrfbdy scalar boundary leaves. All are
# append-only/optional leaves that legitimately cold-start absent from older
# checkpoints.
ADDITIVE_STATE_LEAVES_SINCE_V2 = (
    "Nc",
    "Nn",
    "rainc_acc",
    "qsq",
    "qc_bl",
    "qi_bl",
    "cldfra_bl",
    "qh",
    "Nh",
    "qvolg",
    "qvolh",
    "nwfa",
    "nifa",
    "hail_acc",
    *SCALAR_BOUNDARY_OPTIONAL_LEAVES,
    *(f"{name}_bdy" for name in EXTRA_SCALAR_BOUNDARY_SPECIES),
)

# The frozen Noah-MP scope-options the land carry is valid under (tables.py mirror).
NOAHMP_SCOPE_OPTIONS = {
    "dveg": 4, "opt_crs": 1, "opt_btr": 1, "opt_run": 3, "opt_sfc": 1,
    "opt_frz": 1, "opt_inf": 1, "opt_rad": 3, "opt_alb": 2, "opt_snf": 1,
    "opt_tbot": 2, "opt_stc": 1,
}


def _hostify_tree(tree: Any) -> Any:
    """Copy pytree array leaves to NumPy arrays for process-independent pickle."""

    return jax.tree_util.tree_map(lambda leaf: np.asarray(leaf), tree)


def _device_tree(tree: Any) -> Any:
    """Place checkpoint pytree leaves on the default JAX device after load."""

    return jax.tree_util.tree_map(lambda leaf: jax.device_put(leaf), tree)


def _hostify_namelist(namelist: Any, grid: Any) -> Any:
    from gpuwrf.physics.thompson_hail.tables import checkpoint_split

    # mp38 bound tables (~0.97 GB) are stored as a content reference only.
    namelist, hail_ref = checkpoint_split(namelist)
    host_namelist = _hostify_tree(namelist)
    if is_dataclass(host_namelist) and hasattr(host_namelist, "grid"):
        host_namelist = replace(host_namelist, grid=grid)
    return host_namelist if hail_ref is None else replace(host_namelist, thompson_hail_tables=hail_ref)


def _restore_namelist(namelist: Any, grid: Any) -> Any:
    from gpuwrf.physics.thompson_hail.tables import checkpoint_restore, checkpoint_split

    namelist, hail_ref = checkpoint_split(namelist)
    device_namelist = _device_tree(namelist)
    if is_dataclass(device_namelist) and hasattr(device_namelist, "grid"):
        device_namelist = replace(device_namelist, grid=grid)
    return checkpoint_restore(device_namelist, hail_ref)


def _state_to_host_fields(state: State) -> dict[str, np.ndarray]:
    return {field: np.asarray(getattr(state, field)) for field in state.active_field_names()}


def _validate_state_field_order(recorded: tuple[str, ...]) -> None:
    expected = tuple(State.__slots__)
    if any(field not in expected for field in recorded):
        raise ValueError("checkpoint State field order contains unknown leaves")
    if recorded != tuple(field for field in expected if field in recorded):
        raise ValueError("checkpoint State field order does not match current State schema")
    missing = tuple(field for field in expected if field not in recorded)
    # Held fields may be absent only when inactive, never additive cold starts.
    optional = (*ADDITIVE_STATE_LEAVES_SINCE_V2, *GWDO_DIAGNOSTIC_LEAVES, *LES_TKE_LEAVES,
                *WRITER_PROFILE_LEAVES,
                "sfc_wspd", *MP_RE_DIAGNOSTIC_LEAVES, *SMS_CARRY_LEAVES, *P3_STATE_LEAVES, *JENSEN_STATE_LEAVES, *FERRIER_CARRY_LEAVES, *GODDARD4ICE_CARRY_LEAVES, *GODDARD4ICE_HISTORY_LEAVES, *THOMPSON_HAIL_SURFACE_LEAVES, *EPSILON_PBL_LEAVES, "tracers", "tracer_bdy", "tracer_budget", *NTU_STATE_LEAVES, *CAMMG_STATE_FIELDS, *ICE_DIAGNOSTIC_LEAVES, *MYNN_CLOUDPDF_LEAVES)
    if any(leaf not in optional for leaf in missing):
        raise ValueError(
            "checkpoint State field order is missing non-additive leaves: "
            f"{[leaf for leaf in missing if leaf not in optional]}"
        )


def _validate_gwdo_checkpoint_schema(payload: dict[str, Any]) -> None:
    namelist = payload.get("namelist")
    gwd_opt = namelist.get("gwd_opt", 0) if isinstance(namelist, dict) else getattr(namelist, "gwd_opt", 0)
    fields = payload["state_fields"]
    active = int(gwd_opt) == 1 or any(fields.get(name) is not None for name in GWDO_DIAGNOSTIC_LEAVES)
    if not active:
        return
    # Held stresses came from the preceding pre-PBL inputs, so silently
    # cold-starting them from the post-RK checkpoint cannot preserve them (E78).
    if payload.get("gwdo_checkpoint_schema_version") != GWDO_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint GWDO diagnostics schema predates held stresses (E78)")
    if any(fields.get(name) is None for name in GWDO_DIAGNOSTIC_LEAVES):
        raise ValueError("checkpoint active GWDO is missing held stress diagnostics (E78)")


def _land_field_order() -> tuple[str, ...]:
    if NoahMPLandState is None:
        raise ValueError("NoahMPLandState unavailable; cannot serialize land carry")
    return tuple(NoahMPLandState.__slots__)


def _land_to_host_fields(land_state) -> dict[str, np.ndarray]:
    from gpuwrf.io.glacier_schema import field_order
    return {field: np.asarray(getattr(land_state, field)) for field in field_order(land_state)}


def write_checkpoint(
    state: State,
    namelist: Any,
    grid: Any,
    step_index: int,
    path: str | Path,
    *,
    runtime_state: Any | None = None,
    land_state: Any | None = None,
    scope_options: dict | None = None,
) -> Path:
    """Write a restart checkpoint containing State, namelist, grid, and step index.

    The State payload is stored as an explicit field dictionary so schema drift
    fails closed instead of silently relying on pickle internals. When
    ``land_state`` is given, the prognostic Noah-MP land carry is written under
    ``noahmp_land_state`` with the same fail-closed field-order discipline (v2);
    omitting it yields a payload a v1 reader can consume (the land keys are simply
    absent / None).
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    host_grid = _hostify_tree(grid)
    validate_epsilon_state(state,namelist,EPSILON_PBL_SCHEMA_VERSION)
    payload = {
        "format": "gpuwrf-runtime-checkpoint",
        "format_version": FORMAT_VERSION,
        "state_type": "gpuwrf.contracts.state.State",
        "state_field_order": list(state.active_field_names()),
        "state_field_count": len(state.active_field_names()),
        "state_fields": _state_to_host_fields(state),
        "gwdo_checkpoint_schema_version": GWDO_CHECKPOINT_SCHEMA_VERSION,
        "ocean_schema_version": 1,
        "namelist": _hostify_namelist(namelist, host_grid),
        "grid": host_grid,
        "step_index": int(step_index),
        "runtime_state": None if runtime_state is None else _hostify_tree(runtime_state),
        "tracer_schema_version": 1,
        "qnse_schema_version": QNSE_SCHEMA_VERSION,
        # --- v2 Noah-MP land carry (ADR §5). None when not coupled. ---
        "noahmp_land_state": None,
        "noahmp_land_field_order": None,
        "noahmp_format": None,
    }
    from gpuwrf.io.topographic_shadow_schema import marker as shadow_marker, validate as validate_shadow
    shadow_diag = getattr(runtime_state, "radiation_diagnostics", None)
    shadow_version = shadow_marker(shadow_diag)
    if shadow_version is not None:
        payload["topographic_shadow_schema_version"] = shadow_version
    validate_shadow(getattr(payload["runtime_state"], "radiation_diagnostics", None), shadow_version,
                    state.t_skin.shape, namelist=namelist)
    if epsilon_pbl_active(namelist):
        payload['epsilon_pbl_schema_version']=EPSILON_PBL_SCHEMA_VERSION
    from gpuwrf.tracers.restart import configuration as tracer_configuration
    payload["tracer_config"] = tracer_configuration(namelist)
    _validate_tracer_checkpoint(payload)
    from collections.abc import Mapping
    mmf_runtime = payload['runtime_state']
    mmf_group = (mmf_runtime.get('noahmp_mmf') if isinstance(mmf_runtime, Mapping)
                 else getattr(mmf_runtime, 'noahmp_mmf', None))
    if mmf_group is not None:
        payload['noahmp_mmf_schema_version'] = 1
    _validate_gwdo_checkpoint_schema(payload)
    validate_epsilon_state(payload['state_fields'],namelist,payload.get('epsilon_pbl_schema_version'))
    from gpuwrf.io.restart_schema import physics_option
    if physics_option(namelist,"mp_physics",8)==11:
        payload["cammg_schema_version"]=1
    validate_cammg(payload["state_fields"],namelist,payload.get("cammg_schema_version"))
    validate_held_state(payload["state_fields"], namelist)
    from gpuwrf.physics.milbrandt2mom.restart import validate_checkpoint_radar
    from gpuwrf.io.restart_schema import physics_option
    if physics_option(namelist,'mp_physics',8)==9:
        payload['milbrandt_radar_schema_version']=1
    validate_checkpoint_radar(payload)
    validate_qnse(getattr(runtime_state,'qnse',None),namelist,QNSE_SCHEMA_VERSION,
                  history=getattr(runtime_state,'history_diagnostics',None),expected_shape=state.theta.shape)
    from gpuwrf.io.restart_schema import validate_sas_held
    validate_sas_held(getattr(runtime_state, "cumulus_tendencies", None), namelist, getattr(runtime_state, "osas_rng_key", None), getattr(runtime_state, "cumulus_carry", None))
    from gpuwrf.io.flg_gfdl_schema import validate as validate_legacy,marker
    diagnostics=getattr(runtime_state,'radiation_diagnostics',None);gfdl=getattr(runtime_state,'gfdl_radiation',None)
    from gpuwrf.diagnostics.radiation import validate_restart as validate_rad_diag
    if getattr(diagnostics, "rad_diag", None) is not None:
        payload["rad_diag_schema_version"] = 1
    validate_rad_diag(diagnostics, namelist, payload.get("rad_diag_schema_version"), state.theta.shape)
    legacy=marker(diagnostics,gfdl)
    if legacy is not None:payload['flg_gfdl_schema']=legacy
    validate_legacy(diagnostics,gfdl,namelist,legacy,state_shape=state.qv.shape)
    from gpuwrf.coupling.sst_skin_hook import validate_restart as validate_skin
    validate_skin(getattr(runtime_state, "sst_skin", None), getattr(namelist, "sst_skin", 0),
                  1, state.t_skin.shape)
    payload["sst_skin_schema_version"] = 1
    from gpuwrf.io.noahmp_options import irrigation_configuration
    from gpuwrf.io.noahmp_irrigation import validate_restart as validate_irrigation
    validate_irrigation(getattr(runtime_state,"noahmp_irrigation",None),irrigation_configuration(namelist),1,state.t_skin.shape)
    payload["noahmp_irrigation_schema_version"]=1
    from gpuwrf.io.noahmp_carbon import validate_restart as validate_carbon
    validate_carbon(getattr(runtime_state,"noahmp_carbon",None),namelist,1,state.t_skin.shape)
    payload["noahmp_carbon_schema_version"]=1
    from gpuwrf.io.noahmp_groundwater import validate_restart as validate_groundwater
    validate_groundwater(getattr(runtime_state,"noahmp_groundwater",None),namelist,1,state.t_skin.shape)
    payload["noahmp_groundwater_schema_version"]=1
    from gpuwrf.io.noahmp_hydrology import validate_tile_restart
    validate_tile_restart(getattr(runtime_state,'noahmp_tile_drain',None),namelist,1,state.t_skin.shape)
    payload['noahmp_tile_drain_schema_version']=1
    from gpuwrf.io.cu_forcing_schema import validate as validate_cu_forcing,SCHEMA_VERSION as CU_FORCING_SCHEMA_VERSION
    cu_forcing=getattr(runtime_state,'cu_forcing',None)
    validate_cu_forcing(cu_forcing,namelist,state.theta.shape,CU_FORCING_SCHEMA_VERSION if cu_forcing is not None else None)
    payload['cu_forcing_schema_version']=CU_FORCING_SCHEMA_VERSION if cu_forcing is not None else None
    from gpuwrf.io.noahmp_options import restart_configuration
    payload["noahmp_options"]=restart_configuration(namelist)
    from gpuwrf.io.noahmp_restart_identity import bound_table_identity
    payload["noahmp_bound_tables"]=bound_table_identity(namelist)
    from gpuwrf.surface_params.controls import restart_controls_digest
    payload["surface_controls_digest"]=restart_controls_digest(namelist)
    _validate_spectral_fdda_checkpoint(payload)
    from gpuwrf.io.ocean_schema import validate as validate_ocean
    runtime=payload.get("runtime_state")
    validate_ocean(getattr(runtime,"ocean",None),getattr(runtime,"hurricane_ustm",None),payload.get("namelist"),payload.get("ocean_schema_version"),shape=payload["state_fields"]["t_skin"].shape,history=getattr(runtime,"history_diagnostics",None),require_history=True)

    from gpuwrf.io.windfarm_schema import configuration as wind_config, validate_restart as validate_wind
    payload['windfarm_config'] = wind_config(namelist)
    validate_wind(getattr(runtime_state,'windfarm_power',None),payload['windfarm_config'],state.t_skin.shape,requested=namelist)

    from gpuwrf.coupling.obs_rk import observation_nudging_active
    if observation_nudging_active(namelist):
        from gpuwrf.io.restart_schema import validate_observations
        validate_observations(getattr(runtime_state,'obs',None),namelist)
    if land_state is not None:
        order = _land_field_order()
        payload["noahmp_land_state"] = _land_to_host_fields(land_state)
        payload["noahmp_land_field_order"] = list(payload['noahmp_land_state'])
        from gpuwrf.io.glacier_schema import marker,validate
        payload['glacier_schema_version']=marker(land_state)
        validate(payload['noahmp_land_state'],payload['noahmp_land_field_order'],payload['glacier_schema_version'],namelist=namelist)
        payload["noahmp_format"] = {
            "nsoil": 4, "nsnow": 3,
            "scope_options": dict(scope_options) if scope_options else {
                **NOAHMP_SCOPE_OPTIONS,**({key:value for key,value in payload["noahmp_options"].items()
                                         if key in NOAHMP_SCOPE_OPTIONS or key.startswith("opt_")}
                                        if payload["noahmp_options"] is not None else {})},
        }
    else:
        from gpuwrf.io.glacier_schema import validate
        validate(None,(),None,namelist=namelist)
    tmp = target.with_name(f"{target.name}.tmp")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(target)
    return target


def _read_payload(path: str | Path, *, namelist=None) -> dict[str, Any]:
    source = Path(path)
    with source.open("rb") as handle:
        payload = pickle.load(handle)

    if payload.get("format") != "gpuwrf-runtime-checkpoint":
        raise ValueError(f"unsupported checkpoint format in {source}")
    version = int(payload.get("format_version", -1))
    if version not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            f"unsupported checkpoint version {payload.get('format_version')} in {source}; "
            f"supported {SUPPORTED_FORMAT_VERSIONS}"
        )

    recorded = tuple(payload.get("state_field_order", ()))
    _validate_state_field_order(recorded)
    fields = payload.get("state_fields")
    if not isinstance(fields, dict) or set(fields) != set(recorded):
        raise ValueError("checkpoint State fields do not match the recorded field order")
    _validate_gwdo_checkpoint_schema(payload)
    validate_epsilon_state(fields,payload.get('namelist'),payload.get('epsilon_pbl_schema_version'))
    if les_tke_active(payload.get("namelist")) and any(fields.get(name) is None for name in LES_TKE_LEAVES):
        raise ValueError("km_opt=2 checkpoint is missing the prognostic LES TKE (WRF tke_2) (E78)")
    from gpuwrf.io.mynn_cloudpdf_schema import validate as validate_cloudpdf
    validate_cloudpdf(fields, payload.get("namelist"), namelist)
    validate_cammg(fields,payload.get("namelist"),payload.get("cammg_schema_version"))
    if namelist is not None:
        validate_cammg(fields,namelist,payload.get("cammg_schema_version"))
    validate_held_state(fields, payload.get("namelist"))
    from gpuwrf.physics.milbrandt2mom.restart import validate_checkpoint_radar
    validate_checkpoint_radar(payload)
    _validate_tracer_checkpoint(payload, requested=namelist)
    validate_qnse(getattr(payload.get('runtime_state'),'qnse',None),payload.get('namelist'),payload.get('qnse_schema_version'),
                  history=getattr(payload.get('runtime_state'),'history_diagnostics',None),expected_shape=fields['theta'].shape)
    from gpuwrf.io.restart_schema import validate_sas_held
    validate_sas_held(getattr(payload.get("runtime_state"), "cumulus_tendencies", None), payload.get("namelist"), getattr(payload.get("runtime_state"), "osas_rng_key", None), getattr(payload.get("runtime_state"), "cumulus_carry", None))
    from gpuwrf.io.flg_gfdl_schema import validate as validate_legacy
    runtime=payload.get('runtime_state')
    from gpuwrf.diagnostics.radiation import validate_restart as validate_rad_diag
    for config in (payload.get("namelist"), namelist):
        if config is not None:
            validate_rad_diag(getattr(runtime, "radiation_diagnostics", None), config,
                              payload.get("rad_diag_schema_version"), fields["theta"].shape)
    from gpuwrf.io.topographic_shadow_schema import validate as validate_shadow
    validate_shadow(getattr(runtime, "radiation_diagnostics", None),
                    payload.get("topographic_shadow_schema_version"), fields["t_skin"].shape,
                    namelist=payload.get("namelist"))
    if namelist is not None:
        validate_shadow(getattr(runtime, "radiation_diagnostics", None),
                        payload.get("topographic_shadow_schema_version"), fields["t_skin"].shape,
                        namelist=namelist)
    validate_legacy(getattr(runtime,'radiation_diagnostics',None),getattr(runtime,'gfdl_radiation',None),
        payload.get('namelist'),payload.get('flg_gfdl_schema'),state_shape=fields['qv'].shape)
    if namelist is not None:
        validate_legacy(getattr(runtime,'radiation_diagnostics',None),getattr(runtime,'gfdl_radiation',None),
            namelist,payload.get('flg_gfdl_schema'),state_shape=fields['qv'].shape)
    from gpuwrf.coupling.sst_skin_hook import validate_restart as validate_skin
    validate_skin(getattr(payload.get("runtime_state"), "sst_skin", None),
                  getattr(payload.get("namelist"), "sst_skin", 0), payload.get("sst_skin_schema_version"),
                  fields["t_skin"].shape)
    from gpuwrf.io.noahmp_options import irrigation_configuration
    from gpuwrf.io.noahmp_irrigation import validate_restart as validate_irrigation
    validate_irrigation(getattr(payload.get("runtime_state"),"noahmp_irrigation",None),
        irrigation_configuration(payload.get("namelist")),payload.get("noahmp_irrigation_schema_version"),fields["t_skin"].shape)
    from gpuwrf.io.noahmp_carbon import validate_restart as validate_carbon
    validate_carbon(getattr(payload.get("runtime_state"),"noahmp_carbon",None),payload.get("namelist"),
                    payload.get("noahmp_carbon_schema_version"),fields["t_skin"].shape)
    from gpuwrf.io.noahmp_groundwater import validate_restart as validate_groundwater
    validate_groundwater(getattr(payload.get("runtime_state"),"noahmp_groundwater",None),payload.get("namelist"),
                        payload.get("noahmp_groundwater_schema_version"),fields["t_skin"].shape)
    from gpuwrf.io.noahmp_hydrology import validate_tile_restart
    validate_tile_restart(getattr(payload.get('runtime_state'),'noahmp_tile_drain',None),payload.get('namelist'),
        payload.get('noahmp_tile_drain_schema_version'),fields['t_skin'].shape)
    from gpuwrf.io.cu_forcing_schema import validate as validate_cu_forcing
    validate_cu_forcing(getattr(payload.get('runtime_state'),'cu_forcing',None),payload.get('namelist'),
        fields['theta'].shape,payload.get('cu_forcing_schema_version'))
    _validate_spectral_fdda_checkpoint(payload, requested=namelist)
    if namelist is not None:
        # Value-blind step programs: the saved stochastic controls must equal the requested ones (critic FIX finding 5).
        from collections.abc import Mapping
        from gpuwrf.io.stochastic_config import restart_controls_digest
        def _cfg(source):
            return (source.get("stochastic_config") if isinstance(source, Mapping)
                    else getattr(source, "stochastic_config", None))
        if restart_controls_digest(_cfg(payload.get("namelist"))) != restart_controls_digest(_cfg(namelist)):
            raise ValueError("checkpoint stochastic controls (selection/nens/seeds/scales) differ from the request (E78)")
    from gpuwrf.io.ocean_schema import validate as validate_ocean
    runtime=payload.get("runtime_state")
    validate_ocean(getattr(runtime,"ocean",None),getattr(runtime,"hurricane_ustm",None),payload.get("namelist"),payload.get("ocean_schema_version"),shape=payload["state_fields"]["t_skin"].shape,history=getattr(runtime,"history_diagnostics",None),require_history=True)

    from gpuwrf.io.windfarm_schema import validate_restart as validate_wind
    validate_wind(getattr(payload.get('runtime_state'),'windfarm_power',None),payload.get('windfarm_config'),
                  fields['t_skin'].shape,requested=payload['namelist'])
    from gpuwrf.io.noahmp_options import validate_option_restart
    validate_option_restart(payload.get("noahmp_options"),payload.get("namelist"))
    from gpuwrf.io.noahmp_restart_identity import validate_bound_table_restart
    validate_bound_table_restart(payload.get("noahmp_bound_tables"),payload.get("namelist"))
    from gpuwrf.surface_params.controls import validate_controls_restart
    validate_controls_restart(payload.get("surface_controls_digest"),namelist)
    if namelist is not None:
        # Authenticate the requested selectors as well as the saved bundle.
        # Equal table bytes do not imply equal DVEG/RUN/irrigation controls.
        validate_option_restart(payload.get("noahmp_options"),namelist)
        from collections.abc import Mapping
        requested_skin = namelist.get('sst_skin',0) if isinstance(namelist,Mapping) else getattr(namelist,'sst_skin',0)
        runtime=payload.get('runtime_state')
        validate_skin(getattr(runtime,'sst_skin',None),requested_skin,
                      payload.get('sst_skin_schema_version'),fields['t_skin'].shape)
        validate_irrigation(getattr(runtime,'noahmp_irrigation',None),
            irrigation_configuration(payload.get('namelist')),payload.get('noahmp_irrigation_schema_version'),
            fields['t_skin'].shape,requested=irrigation_configuration(namelist))
        validate_carbon(getattr(runtime,'noahmp_carbon',None),namelist,
                        payload.get('noahmp_carbon_schema_version'),fields['t_skin'].shape)
        validate_groundwater(getattr(runtime,'noahmp_groundwater',None),namelist,
                            payload.get('noahmp_groundwater_schema_version'),fields['t_skin'].shape)
        validate_tile_restart(getattr(runtime,'noahmp_tile_drain',None),namelist,
                              payload.get('noahmp_tile_drain_schema_version'),fields['t_skin'].shape)
        validate_cu_forcing(getattr(runtime,'cu_forcing',None),namelist,
                            fields['theta'].shape,payload.get('cu_forcing_schema_version'))
        validate_bound_table_restart(payload.get("noahmp_bound_tables"),namelist)

    from gpuwrf.coupling.obs_rk import observation_nudging_active
    if observation_nudging_active(payload.get("namelist")):
        from gpuwrf.io.restart_schema import validate_observations
        validate_observations(getattr(payload.get('runtime_state'),'obs',None),payload['namelist'])

    # v2 land carry: exact-match the field order (fail-closed, like State). A v1
    # checkpoint or a v2 with no land carry simply has noahmp_land_state == None.
    land_fields = payload.get("noahmp_land_state")
    if land_fields is not None:
        land_order = tuple(payload.get("noahmp_land_field_order", ()))
        if NoahMPLandState is None:
            raise ValueError("checkpoint carries Noah-MP land state but NoahMPLandState is unavailable")
        from gpuwrf.io.glacier_schema import validate
        validate(land_fields,land_order,payload.get('glacier_schema_version'),namelist=payload.get('namelist'))
    else:
        from gpuwrf.io.glacier_schema import validate
        validate(None,(),payload.get('glacier_schema_version'),namelist=payload.get('namelist'))
    return payload


def _validate_tracer_checkpoint(payload, *, requested=None):
    from collections.abc import Mapping
    from gpuwrf.tracers.restart import STATE_FIELDS, validate_restart
    runtime = payload.get("runtime_state")
    tendencies = runtime.get("tracer_tendencies") if isinstance(runtime, Mapping) else getattr(runtime, "tracer_tendencies", None)
    fields = payload["state_fields"]
    validate_restart(fields, tendencies, payload.get("tracer_config"), payload.get("tracer_schema_version"),
                     fields["qv"].shape, requested=payload.get("namelist"))
    if requested is not None:
        validate_restart(fields, tendencies, payload.get("tracer_config"), payload.get("tracer_schema_version"),
                         fields["qv"].shape, requested=requested)
    # OperationalCarry embeds State too. Both views must identify one snapshot.
    embedded = runtime.get("state") if isinstance(runtime, Mapping) else getattr(runtime, "state", None)
    if embedded is not None:
        for key in STATE_FIELDS:
            value = embedded.get(key) if isinstance(embedded, Mapping) else getattr(embedded, key, None)
            original = fields.get(key)
            if (value is None) != (original is None) or (value is not None and (
                    np.asarray(value).dtype != np.asarray(original).dtype
                    or not np.array_equal(value, original))):
                raise ValueError("tracer checkpoint State and runtime carry disagree")


def _refuse_tracer_runtime_discard(payload):
    from gpuwrf.tracers.restart import configuration, count
    if count(configuration(payload.get("tracer_config"))):
        raise ValueError("tracer checkpoint requires read_checkpoint_with_runtime_state to preserve tendencies")


def _validate_spectral_fdda_checkpoint(payload, *, requested=None):
    from collections.abc import Mapping
    from gpuwrf.io.restart_schema import validate_spectral_fdda
    runtime = payload.get("runtime_state")
    from gpuwrf.io.noahmp_mmf import validate_restart as validate_mmf
    mmf = runtime.get('noahmp_mmf') if isinstance(runtime, Mapping) else getattr(runtime, 'noahmp_mmf', None)
    # The full optional group is in runtime_state, never reconstructed from soil.
    validate_mmf(mmf, payload.get('namelist'), payload.get('noahmp_mmf_schema_version'))
    if requested is not None:
        validate_mmf(mmf,requested,payload.get('noahmp_mmf_schema_version'))
    from gpuwrf.io.noahmp_mmf import validate_land as validate_mmf_land
    land = runtime.get('noahmp_land') if isinstance(runtime,Mapping) else getattr(runtime,'noahmp_land',None)
    validate_mmf_land(mmf,land,payload['state_fields']['t_skin'].shape)
    group = runtime.get("fdda") if isinstance(runtime, Mapping) else getattr(runtime, "fdda", None)
    validate_spectral_fdda(group, payload.get("namelist"))
    from gpuwrf.io.stochastic_schema import validate_stochastic
    stoch = runtime.get("stochastic") if isinstance(runtime, Mapping) else getattr(runtime,"stochastic",None)
    validate_stochastic(stoch,payload.get("namelist"))


def read_checkpoint(path: str | Path, *, namelist=None) -> tuple[State, Any, Any, int]:
    """Read a checkpoint and reconstruct JAX pytrees on the default device."""

    payload = _read_payload(path, namelist=namelist)
    _refuse_tracer_runtime_discard(payload)
    _refuse_mmf_runtime_discard(payload)
    _refuse_land_option_runtime_discard(payload)
    from gpuwrf.io.restart_schema import physics_option
    if physics_option(payload.get("namelist"), "mp_physics", 8) in (50,51,52,53):
        raise ValueError("P3 held radar requires read_checkpoint_with_runtime_state (E78)")
    return _restore_checkpoint_payload(payload)


def _restore_checkpoint_payload(payload: dict[str, Any]) -> tuple[State, Any, Any, int]:
    # Construct from the RECORDED leaves only. Append-only additive leaves absent
    # from an older (v1/v2) checkpoint default to None in ``State.__init__``, which
    # backfills them with zeros at the matrix dtype (cold-start the new physics
    # fields). Inactive GWDO leaves stay None; active payloads were schema-checked
    # above rather than silently cold-starting held diagnostics.
    fields = payload["state_fields"]
    state = State(**{field: jax.device_put(value) for field, value in fields.items()})
    from gpuwrf.io.restart_schema import physics_option
    if physics_option(payload.get('namelist'),'mp_physics',8)==9:
        # Constructor defaults are legacy storage policy; restore the exact
        # recorded REAL carry rather than widening Milbrandt number fields.
        state=state.replace(_cast=False,**{field:jax.device_put(value) for field,value in fields.items()})
    grid = _device_tree(payload["grid"])
    namelist = _restore_namelist(payload["namelist"], grid)
    return state, namelist, grid, int(payload["step_index"])


def read_checkpoint_with_runtime_state(path: str | Path, *, namelist=None) -> tuple[State, Any, Any, int, Any | None]:
    """Read a checkpoint including optional operational reproducibility state."""

    payload = _read_payload(path, namelist=namelist)
    state, namelist, grid, step_index = _restore_checkpoint_payload(payload)
    runtime_state = payload.get("runtime_state")
    if runtime_state is not None:
        runtime_state = _device_tree(runtime_state)
    return state, namelist, grid, step_index, runtime_state


def _restore_land_state(payload: dict[str, Any]):
    """Rebuild the NoahMPLandState carry (or None for a v1 / land-less payload)."""
    land_fields = payload.get("noahmp_land_state")
    if land_fields is None:
        return None
    order = tuple(NoahMPLandState.__slots__)
    return NoahMPLandState(**{f: jax.device_put(land_fields.get(f)) for f in order})


def read_checkpoint_with_land_state(
    path: str | Path,
    *, namelist=None,
) -> tuple[State, Any, Any, int, Any]:
    """Read a checkpoint and the prognostic Noah-MP land carry (ADR §5).

    Returns ``(state, namelist, grid, step_index, land_state)`` where
    ``land_state`` is the restored ``NoahMPLandState`` for a v2 checkpoint, or
    ``None`` for a v1 (or land-less v2) checkpoint — in which case the caller
    cold-initialises Noah-MP land state from ``wrfinput``.
    """
    payload = _read_payload(path, namelist=namelist)
    _refuse_tracer_runtime_discard(payload)
    _refuse_mmf_runtime_discard(payload)
    _refuse_land_option_runtime_discard(payload)
    state, namelist, grid, step_index = _restore_checkpoint_payload(payload)
    land_state = _restore_land_state(payload)
    return state, namelist, grid, step_index, land_state


def _refuse_mmf_runtime_discard(payload):
    from gpuwrf.io.noahmp_mmf import configuration
    if configuration(payload.get('namelist'))['noahmp_mmf_enabled']:
        raise ValueError('MMF checkpoint requires read_checkpoint_with_runtime_state (E78)')


def _refuse_land_option_runtime_discard(payload):
    """Reduced readers cannot represent these evolving full-carry groups."""
    from collections.abc import Mapping
    runtime=payload.get('runtime_state')
    get=runtime.get if isinstance(runtime,Mapping) else lambda name:getattr(runtime,name,None)
    for name in ('sst_skin','noahmp_irrigation','noahmp_carbon','noahmp_groundwater','noahmp_tile_drain','cu_forcing'):
        if get(name) is not None:
            raise ValueError(f'{name} checkpoint requires read_checkpoint_with_runtime_state (E78)')


__all__ = [
    "read_checkpoint",
    "read_checkpoint_with_runtime_state",
    "read_checkpoint_with_land_state",
    "write_checkpoint",
]
