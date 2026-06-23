"""Pure shared M6B6 coupled timestep core.

This module composes the shared dycore core with physics and Gen2 lateral-boundary replay.

WRF ordering anchors:
- ``solve_em.F:1437-1704`` documents non-timesplit physics before RK updates.
- ``solve_em.F:1699-1935`` prepares and calls physics drivers for RK step 1.
- ``solve_em.F:2034-2285`` applies specified lateral-boundary tendencies.
- ``solve_em.F:3065-4363`` runs acoustic small steps inside each RK stage.
- ``solve_em.F:6765`` closes the RK predictor-corrector loop.
"""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

from dataclasses import dataclass
from typing import Any

import jax
from jax import config
import jax.numpy as jnp

from gpuwrf.contracts.state import State
from gpuwrf.coupling.boundary_apply import BoundaryConfig, apply_lateral_boundaries
from gpuwrf.coupling.physics_couplers import mynn_adapter, rrtmg_adapter, thompson_adapter
from gpuwrf.dynamics.core.acoustic import AcousticCoreState, FULL_STATE_FIELDS
from gpuwrf.dynamics.core.dycore import DycoreCoreConfig, dycore_timestep_core
from gpuwrf.contracts.grid import DycoreMetrics


configure_jax_x64()


PHYSICS_TENDENCY_FIELDS = (
    "theta_phys_tend",
    "qv_phys_tend",
    "qc_phys_tend",
    "qr_phys_tend",
    "qi_phys_tend",
    "qs_phys_tend",
    "qg_phys_tend",
    "qke_phys_tend",
    "u_phys_tend",
    "v_phys_tend",
    "w_phys_tend",
    "mu_bdy_tend",
)

COUPLED_STATE_FIELDS = FULL_STATE_FIELDS + PHYSICS_TENDENCY_FIELDS

NAMELIST_PHYSICS_BOUNDARY_ON = {
    "mp_physics": 8,
    "bl_pbl_physics": 5,
    "ra_lw_physics": 4,
    "ra_sw_physics": 4,
    "cu_physics": 0,
    "sf_sfclay_physics": 0,
    "sf_surface_physics": 0,
    "specified": True,
}


@dataclass(frozen=True)
class CoupledCoreConfig:
    """Static shared core config for M6B6 coupled-step composition."""

    dt: float
    dx: float
    dy: float
    acoustic_substeps: int = 10
    rk_order: int = 3
    epssm: float = 0.1
    top_lid: bool = False
    physics_enabled: bool = True
    boundary_enabled: bool = True
    theta_base_offset: float = 300.0
    pressure_base_offset: float = 90000.0
    boundary_config: BoundaryConfig | None = None
    periodic_x: bool = True
    specified: bool = False
    nested: bool = False

    def dycore_config(self) -> DycoreCoreConfig:
        return DycoreCoreConfig(
            dt=float(self.dt),
            dx=float(self.dx),
            dy=float(self.dy),
            acoustic_substeps=int(self.acoustic_substeps),
            rk_order=int(self.rk_order),
            epssm=float(self.epssm),
            top_lid=bool(self.top_lid),
            physics_enabled=False,
            boundary_enabled=False,
            periodic_x=bool(self.periodic_x),
            specified=bool(self.specified),
            nested=bool(self.nested),
        )


def _zeros(shape: tuple[int, ...], dtype=jnp.float64):
    return jnp.zeros(shape, dtype=dtype)


def _extra(extras: dict[str, Any], name: str, shape: tuple[int, ...], default: float, dtype=jnp.float64):
    if name in extras:
        return jnp.asarray(extras[name], dtype=dtype)
    return jnp.ones(shape, dtype=dtype) * default


def _boundary_leaf(extras: dict[str, Any], name: str, z_len: int, side_len: int, default: float):
    if name in extras:
        return jnp.asarray(extras[name], dtype=jnp.float64)
    return jnp.ones((2, 4, z_len, side_len), dtype=jnp.float64) * default


def _state_from_acoustic(
    acoustic: dict[str, Any],
    extras: dict[str, Any],
    cfg: CoupledCoreConfig,
    *,
    physical_theta: bool,
) -> State:
    theta = jnp.asarray(acoustic["theta"], dtype=jnp.float64)
    p = jnp.asarray(acoustic["p"], dtype=jnp.float64)
    ph = jnp.asarray(acoustic["ph"], dtype=jnp.float64)
    mu = jnp.asarray(acoustic["mu"], dtype=jnp.float64)
    nz, ny, nx = theta.shape
    side_len = max(nx + 1, ny + 1)
    surface = (ny, nx)
    mass = (nz, ny, nx)
    theta_state = theta + float(cfg.theta_base_offset) if physical_theta else theta
    p_state = p + float(cfg.pressure_base_offset) if physical_theta else p
    return State(
        u=jnp.asarray(acoustic["u"], dtype=jnp.float64),
        v=jnp.asarray(acoustic["v"], dtype=jnp.float64),
        w=jnp.asarray(acoustic["w"], dtype=jnp.float64),
        theta=theta_state,
        qv=_extra(extras, "qv", mass, 0.010),
        p=p_state,
        ph=ph,
        mu=mu,
        p_total=p_state,
        p_perturbation=p,
        ph_total=ph,
        ph_perturbation=ph,
        mu_total=mu,
        mu_perturbation=mu,
        qc=_extra(extras, "qc", mass, 0.0),
        qr=_extra(extras, "qr", mass, 0.0),
        qi=_extra(extras, "qi", mass, 0.0),
        qs=_extra(extras, "qs", mass, 0.0),
        qg=_extra(extras, "qg", mass, 0.0),
        Ni=_extra(extras, "Ni", mass, 1.0e5),
        Nr=_extra(extras, "Nr", mass, 1.0e5),
        Ns=_extra(extras, "Ns", mass, 0.0),
        Ng=_extra(extras, "Ng", mass, 0.0),
        qke=_extra(extras, "qke", mass, 0.20),
        ustar=_extra(extras, "ustar", surface, 0.30),
        theta_flux=_extra(extras, "theta_flux", surface, 0.0),
        qv_flux=_extra(extras, "qv_flux", surface, 0.0),
        tau_u=_extra(extras, "tau_u", surface, 0.0),
        tau_v=_extra(extras, "tau_v", surface, 0.0),
        rhosfc=_extra(extras, "rhosfc", surface, 1.0),
        fltv=_extra(extras, "fltv", surface, 0.0),
        t_skin=_extra(extras, "t_skin", surface, 295.0),
        soil_moisture=_extra(extras, "soil_moisture", surface, 0.20),
        xland=_extra(extras, "xland", surface, 1.0),
        lakemask=_extra(extras, "lakemask", surface, 0.0),
        mavail=_extra(extras, "mavail", surface, 0.20),
        roughness_m=_extra(extras, "roughness_m", surface, 0.05),
        rain_acc=_extra(extras, "rain_acc", surface, 0.0),
        snow_acc=_extra(extras, "snow_acc", surface, 0.0),
        graupel_acc=_extra(extras, "graupel_acc", surface, 0.0),
        ice_acc=_extra(extras, "ice_acc", surface, 0.0),
        u_bdy=_boundary_leaf(extras, "u_bdy", nz, side_len, 0.0),
        v_bdy=_boundary_leaf(extras, "v_bdy", nz, side_len, 0.0),
        theta_bdy=_boundary_leaf(extras, "theta_bdy", nz, side_len, 0.0),
        qv_bdy=_boundary_leaf(extras, "qv_bdy", nz, side_len, 0.010),
        ph_bdy=_boundary_leaf(extras, "ph_bdy", nz + 1, side_len, 0.0),
        mu_bdy=_boundary_leaf(extras, "mu_bdy", 1, side_len, 0.0),
    )


def _adaptive_boundary_config(acoustic: dict[str, Any], cfg: CoupledCoreConfig) -> BoundaryConfig:
    if cfg.boundary_config is not None:
        return cfg.boundary_config
    theta = jnp.asarray(acoustic["theta"])
    _, ny, nx = theta.shape
    relax_zone = max(1, min(5, (min(int(ny), int(nx)) - 1) // 2))
    return BoundaryConfig(spec_bdy_width=5, spec_zone=1, relax_zone=relax_zone, update_cadence_s=3600.0, spec_exp=0.0)


def _physics_update(acoustic: dict[str, Any], extras: dict[str, Any], cfg: CoupledCoreConfig) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    physical_before = _state_from_acoustic(acoustic, extras, cfg, physical_theta=True)
    physical_after = thompson_adapter(physical_before, float(cfg.dt))
    physical_after = mynn_adapter(physical_after, float(cfg.dt), None)
    physical_after = rrtmg_adapter(physical_after, float(cfg.dt), None)
    inv_dt = 1.0 / float(cfg.dt)
    updated = dict(acoustic)
    updated["theta"] = jnp.asarray(acoustic["theta"]) + (physical_after.theta - physical_before.theta)
    updated["u"] = physical_after.u
    updated["v"] = physical_after.v
    updated["w"] = physical_after.w
    tendencies = {
        "theta_phys_tend": (physical_after.theta - physical_before.theta) * inv_dt,
        "qv_phys_tend": (physical_after.qv - physical_before.qv) * inv_dt,
        "qc_phys_tend": (physical_after.qc - physical_before.qc) * inv_dt,
        "qr_phys_tend": (physical_after.qr - physical_before.qr) * inv_dt,
        "qi_phys_tend": (physical_after.qi - physical_before.qi) * inv_dt,
        "qs_phys_tend": (physical_after.qs - physical_before.qs) * inv_dt,
        "qg_phys_tend": (physical_after.qg - physical_before.qg) * inv_dt,
        "qke_phys_tend": (physical_after.qke - physical_before.qke) * inv_dt,
        "u_phys_tend": (physical_after.u - physical_before.u) * inv_dt,
        "v_phys_tend": (physical_after.v - physical_before.v) * inv_dt,
        "w_phys_tend": (physical_after.w - physical_before.w) * inv_dt,
    }
    return updated, tendencies


def _boundary_update(acoustic: dict[str, Any], extras: dict[str, Any], cfg: CoupledCoreConfig, step_index: int | jax.Array) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    state_before = _state_from_acoustic(acoustic, extras, cfg, physical_theta=False)
    lead_seconds = jnp.maximum(jnp.asarray(step_index, dtype=jnp.float64) - 1.0, 0.0) * float(cfg.dt)
    state_after = apply_lateral_boundaries(
        state_before,
        lead_seconds,
        float(cfg.dt),
        _adaptive_boundary_config(acoustic, cfg),
    )
    inv_dt = 1.0 / float(cfg.dt)
    updated = dict(acoustic)
    for name in ("u", "v", "theta", "ph", "mu"):
        updated[name] = getattr(state_after, name)
    tendencies = {"mu_bdy_tend": (state_after.mu - state_before.mu) * inv_dt}
    return updated, tendencies


def coupled_timestep_core(
    state: AcousticCoreState,
    metrics: DycoreMetrics,
    cfg: CoupledCoreConfig,
    *,
    extras: dict[str, Any] | None = None,
    step_index: int = 1,
) -> dict[str, jax.Array]:
    """Run one WRF-ordered coupled timestep."""

    if int(cfg.rk_order) != 3:
        raise ValueError("M6B coupled core requires RK3")
    if not bool(cfg.physics_enabled) or not bool(cfg.boundary_enabled):
        raise ValueError("M6B coupled core requires physics and boundary enabled")
    extra = dict(extras or {})
    _rk, dycore = dycore_timestep_core(state, metrics, cfg.dycore_config())
    after_physics, physics_tendencies = _physics_update(dycore, extra, cfg)
    after_boundary, boundary_tendencies = _boundary_update(after_physics, extra, cfg, step_index)
    snapshot = {name: jnp.asarray(after_boundary[name]) for name in FULL_STATE_FIELDS}
    snapshot.update(physics_tendencies)
    snapshot.update(boundary_tendencies)
    return snapshot


def coupled_timesteps_core(
    state: AcousticCoreState,
    metrics: DycoreMetrics,
    cfg: CoupledCoreConfig,
    *,
    steps: int,
    extras: dict[str, Any] | None = None,
) -> list[dict[str, jax.Array]]:
    """Run repeated coupled timesteps."""

    current = state
    snapshots: list[dict[str, jax.Array]] = []
    for step in range(1, int(steps) + 1):
        snapshot = coupled_timestep_core(current, metrics, cfg, extras=extras, step_index=step)
        snapshots.append(snapshot)
        current = current.replace(**{name: snapshot[name] for name in FULL_STATE_FIELDS})
    return snapshots


__all__ = [
    "COUPLED_STATE_FIELDS",
    "NAMELIST_PHYSICS_BOUNDARY_ON",
    "PHYSICS_TENDENCY_FIELDS",
    "CoupledCoreConfig",
    "coupled_timestep_core",
    "coupled_timesteps_core",
    "CoupledStepConfig",
    "coupled_timestep_wrf",
    "coupled_timesteps_wrf",
]


CoupledStepConfig = CoupledCoreConfig
coupled_timestep_wrf = coupled_timestep_core
coupled_timesteps_wrf = coupled_timesteps_core
