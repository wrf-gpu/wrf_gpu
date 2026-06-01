"""GPU-resident operational forecast loop for M6 perf-design.

This module is deliberately separate from the M6B validation savepoint ladder.
It runs timestep/RK/acoustic loops inside one JAX entry point and leaves debug
snapshots/sanitizers out of the compiled path.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
from jax import config
import jax.numpy as jnp

from gpuwrf.contracts.grid import DycoreMetrics, GridSpec
from gpuwrf.contracts.state import BaseState, State, Tendencies
from gpuwrf.contracts.precision import DEFAULT_DTYPES, STATE_FIELD_ORDER
from gpuwrf.contracts.halo import apply_halo
from gpuwrf.coupling.boundary_apply import (
    BoundaryConfig,
    DEFAULT_BOUNDARY_CONFIG,
    apply_lateral_boundaries,
    interpolate_boundary_leaf,
    normal_bdy_work_target_u,
    normal_bdy_work_target_v,
    nested_ph_relax_tendency,
    nested_w_relax_tendency,
    _full_ring_target_from_leaf,
)
from gpuwrf.coupling.physics_couplers import (
    mynn_adapter,
    rrtmg_radiation_diagnostics,
    rrtmg_theta_tendency,
    surface_adapter,
    surface_layer_diagnostics,
    thompson_adapter,
)
from gpuwrf.dynamics.advection import compute_advection_tendencies, halo_spec
from gpuwrf.dynamics.explicit_diffusion import (
    constant_k_diffusion_tendency,
    conservative_constant_k_diffusion_tendency,
    sixth_order_diffusion_tendency,
    wrf_deformation_momentum_tendency,
)
from gpuwrf.dynamics.flux_advection import (
    advect_scalar_flux,
    advect_u_flux,
    advect_v_flux,
    advect_w_flux,
    couple_velocities_periodic,
)
from gpuwrf.dynamics.acoustic_wrf import (
    CPOVCV,
    _inverse_density_from_theta_pressure,
    calc_coef_w_wrf_coefficients,
    diagnose_pressure_al_alt,
    horizontal_pressure_gradient,
    moisture_coupling_factors,
)
from gpuwrf.dynamics.core.acoustic import AcousticCoreConfig, AcousticCoreState, acoustic_substep_core
from gpuwrf.dynamics.core.advance_w import GRAVITY_M_S2, dry_cqw, pg_buoy_w_dry
from gpuwrf.dynamics.core.calc_p_rho import CalcPRhoStep0, calc_p_rho_wrf
from gpuwrf.dynamics.core.rhs_ph import rhs_ph_wrf
from gpuwrf.dynamics.core.coupled import CoupledCoreConfig, coupled_timestep_core
from gpuwrf.dynamics.core.rk_addtend_dry import (
    DryPhysicsTendencies,
    large_step_coriolis,
    large_step_horizontal_pgf,
    rk_addtend_dry,
)
from gpuwrf.dynamics.core.small_step_finish import small_step_finish_wrf
from gpuwrf.dynamics.core.small_step_prep import SmallStepPrepState, small_step_prep_wrf
from gpuwrf.dynamics.tendencies import add_scaled_tendencies
from gpuwrf.runtime.operational_state import OperationalCarry, initial_operational_carry


config.update("jax_enable_x64", True)

_THETA_LIMITER_MIN_K = 0.0
_THETA_LIMITER_MAX_K = 500.0


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class OperationalNamelist:
    """Static runtime controls plus resident metric/tendency leaves.

    ``grid`` and scalar controls are static cache keys. ``tendencies`` and
    ``metrics`` are device leaves so no host/device transfer is needed in the
    timestep loop. M6b promotes WRF small-step scratch fields into the resident
    production carry; see ``runtime.operational_state`` for the evidence table.
    """

    grid: GridSpec
    tendencies: Tendencies
    metrics: DycoreMetrics
    dt_s: float = 10.0
    acoustic_substeps: int = 10
    rk_order: int = 3
    epssm: float = 0.1
    top_lid: bool = False
    run_physics: bool = True
    run_boundary: bool = True
    radiation_cadence_steps: int = 60
    boundary_config: BoundaryConfig = DEFAULT_BOUNDARY_CONFIG
    use_vertical_solver: bool = True
    disable_guards: bool = False
    # WRF damping (Gen2 d02 namelist: w_damping=1, damp_opt=3, zdamp=5000, dampcoef=0.2).
    # Defaults OFF so the bare acoustic core (Sprint A) behaviour is unchanged unless
    # the caller explicitly enables WRF damping for the operational dt.
    w_damping: int = 0
    damp_opt: int = 0
    dampcoef: float = 0.0
    zdamp: float = 5000.0
    diff_opt: int = 0
    km_opt: int = 0
    khdif: float = 0.0
    kvdif: float = 0.0
    diff_6th_opt: int = 0
    diff_6th_factor: float = 0.12
    # Constant eddy viscosity (Straka ν=75) on u, v, theta when > 0.
    const_nu_m2_s: float = 0.0
    # Use WRF flux-form mass-coupled scalar advection (Block 2) for theta.
    use_flux_advection: bool = False
    # Force pure fp64 (Sprint F7-B is fp64-correctness-only; idealized cases set it).
    force_fp64: bool = False
    # Use the WRF deformation-tensor momentum diffusion (diff_opt=2/km_opt=1) for
    # u/v/w instead of the scalar flux-divergence Laplacian.  Theta always keeps
    # the conservative scalar flux-divergence (WRF horizontal_diffusion_s).  Only
    # active when const_nu_m2_s > 0.  Sprint U (P0-2).
    use_deformation_momentum_diffusion: bool = False
    # Model-init UTC instant (recomp B3 hook). Static aux (datetime / ISO string /
    # None). When set, the RRTMG radiation adapter is driven by the actual forecast
    # clock (time_utc + lead_seconds) inside the scan, so the diurnal SW cycle
    # evolves over the run; None keeps the adapter's legacy fixed-time behaviour.
    time_utc: object = None

    @classmethod
    def from_grid(
        cls,
        grid: GridSpec,
        *,
        tendencies: Tendencies | None = None,
        metrics: DycoreMetrics | None = None,
        dt_s: float = 10.0,
        acoustic_substeps: int = 10,
        radiation_cadence_steps: int = 60,
        boundary_config: BoundaryConfig = DEFAULT_BOUNDARY_CONFIG,
        use_vertical_solver: bool = True,
        disable_guards: bool = False,
        epssm: float = 0.1,
        top_lid: bool = False,
        w_damping: int = 0,
        damp_opt: int = 0,
        dampcoef: float = 0.0,
        zdamp: float = 5000.0,
        diff_opt: int = 0,
        km_opt: int = 0,
        khdif: float = 0.0,
        kvdif: float = 0.0,
        diff_6th_opt: int = 0,
        diff_6th_factor: float = 0.12,
        const_nu_m2_s: float = 0.0,
        use_flux_advection: bool = False,
        force_fp64: bool = False,
        use_deformation_momentum_diffusion: bool = False,
        time_utc: object = None,
    ) -> "OperationalNamelist":
        """Build a namelist using resident zero tendencies and flat metrics."""

        if tendencies is None:
            tendencies = Tendencies.zeros(grid)
        if metrics is None:
            metrics = DycoreMetrics.flat(
                ny=grid.ny,
                nx=grid.nx,
                nz=grid.nz,
                eta_levels=grid.vertical.eta_levels,
                top_pressure_pa=grid.vertical.top_pressure_pa,
                provenance="operational-flat-from-grid",
            )
        return cls(
            grid=grid,
            tendencies=tendencies,
            metrics=metrics,
            dt_s=dt_s,
            acoustic_substeps=acoustic_substeps,
            epssm=epssm,
            top_lid=top_lid,
            radiation_cadence_steps=radiation_cadence_steps,
            boundary_config=boundary_config,
            use_vertical_solver=use_vertical_solver,
            disable_guards=disable_guards,
            w_damping=w_damping,
            damp_opt=damp_opt,
            dampcoef=dampcoef,
            zdamp=zdamp,
            diff_opt=diff_opt,
            km_opt=km_opt,
            khdif=khdif,
            kvdif=kvdif,
            diff_6th_opt=diff_6th_opt,
            diff_6th_factor=diff_6th_factor,
            const_nu_m2_s=const_nu_m2_s,
            use_flux_advection=use_flux_advection,
            force_fp64=force_fp64,
            use_deformation_momentum_diffusion=use_deformation_momentum_diffusion,
            time_utc=time_utc,
        )

    def tree_flatten(self):
        children = (self.tendencies, self.metrics)
        aux = (
            self.grid,
            float(self.dt_s),
            int(self.acoustic_substeps),
            int(self.rk_order),
            float(self.epssm),
            bool(self.top_lid),
            bool(self.run_physics),
            bool(self.run_boundary),
            int(self.radiation_cadence_steps),
            self.boundary_config,
            bool(self.use_vertical_solver),
            bool(self.disable_guards),
            int(self.w_damping),
            int(self.damp_opt),
            float(self.dampcoef),
            float(self.zdamp),
            int(self.diff_opt),
            int(self.km_opt),
            float(self.khdif),
            float(self.kvdif),
            int(self.diff_6th_opt),
            float(self.diff_6th_factor),
            float(self.const_nu_m2_s),
            bool(self.use_flux_advection),
            bool(self.force_fp64),
            bool(self.use_deformation_momentum_diffusion),
            self.time_utc,
        )
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        tendencies, metrics = children
        (
            grid,
            dt_s,
            acoustic_substeps,
            rk_order,
            epssm,
            top_lid,
            run_physics,
            run_boundary,
            radiation_cadence_steps,
            boundary_config,
            use_vertical_solver,
            disable_guards,
            w_damping,
            damp_opt,
            dampcoef,
            zdamp,
            diff_opt,
            km_opt,
            khdif,
            kvdif,
            diff_6th_opt,
            diff_6th_factor,
            const_nu_m2_s,
            use_flux_advection,
            force_fp64,
            use_deformation_momentum_diffusion,
            time_utc,
        ) = aux
        return cls(
            grid=grid,
            tendencies=tendencies,
            metrics=metrics,
            dt_s=dt_s,
            acoustic_substeps=acoustic_substeps,
            rk_order=rk_order,
            epssm=epssm,
            top_lid=top_lid,
            run_physics=run_physics,
            run_boundary=run_boundary,
            radiation_cadence_steps=radiation_cadence_steps,
            boundary_config=boundary_config,
            use_vertical_solver=use_vertical_solver,
            disable_guards=disable_guards,
            w_damping=w_damping,
            damp_opt=damp_opt,
            dampcoef=dampcoef,
            zdamp=zdamp,
            diff_opt=diff_opt,
            km_opt=km_opt,
            khdif=khdif,
            kvdif=kvdif,
            diff_6th_opt=diff_6th_opt,
            diff_6th_factor=diff_6th_factor,
            const_nu_m2_s=const_nu_m2_s,
            use_flux_advection=use_flux_advection,
            force_fp64=force_fp64,
            use_deformation_momentum_diffusion=use_deformation_momentum_diffusion,
            time_utc=time_utc,
        )


@dataclass(frozen=True)
class _RKStageDescriptor:
    """Static WRF RK/acoustic cadence descriptor from ``solve_em.F:1472-1483``."""

    rk_step: int
    dt_rk: float
    dts_rk: float
    number_of_small_timesteps: int


def _steps_for_hours(hours: float, dt_s: float) -> int:
    raw = float(hours) * 3600.0 / float(dt_s)
    rounded = int(round(raw))
    if abs(raw - rounded) > 1.0e-9:
        raise ValueError(f"forecast length {hours}h is not an integer number of dt={dt_s}s steps")
    return rounded


def _enforce_operational_precision(state: State, *, force_fp64: bool = False) -> State:
    if bool(force_fp64):
        # Sprint F7-B is fp64-correctness-only: idealized cases and any caller
        # that sets force_fp64 keep every prognostic in float64.  The fp32-gated
        # operational matrix (ADR-007) is a perf decision deferred to F7-perf.
        updates = {
            field: getattr(state, field).astype(jnp.float64) for field in STATE_FIELD_ORDER
        }
        # _cast=False so the fp64 upcast is NOT canonicalised back to each
        # field's loaded dtype.  Real-case states arrive mixed-precision
        # (DEFAULT_DTYPES perf matrix: theta/u/v fp32, w/mu/ph fp64); without
        # this the force_fp64 path is a silent no-op (Sprint U P0-1).
        return state.replace(_cast=False, **updates)
    updates = {
        field: getattr(state, field).astype(DEFAULT_DTYPES.dtype_for(field))
        for field in STATE_FIELD_ORDER
    }
    return state.replace(**updates)


def _theta_base_offset(theta: jax.Array) -> jax.Array:
    """Return the WRF perturbation-theta offset for operational Gen2 states."""

    return jnp.asarray(300.0, dtype=theta.dtype)


def _u_face_average_2d(field: jax.Array) -> jax.Array:
    west = field[:, :1]
    east = field[:, -1:]
    interior = 0.5 * (field[:, :-1] + field[:, 1:])
    return jnp.concatenate((west, interior, east), axis=1)


def _v_face_average_2d(field: jax.Array) -> jax.Array:
    south = field[:1, :]
    north = field[-1:, :]
    interior = 0.5 * (field[:-1, :] + field[1:, :])
    return jnp.concatenate((south, interior, north), axis=0)


def _base_mu(state: State) -> jax.Array:
    return jnp.asarray(state.mu_total) - jnp.asarray(state.mu_perturbation)


def _valid_mixing_ratio(candidate: jax.Array, origin: jax.Array, upper: float = 0.05) -> jax.Array:
    """Keep nonfinite RK moisture excursions out of the physics boundary."""

    candidate = jnp.asarray(candidate)
    origin = jnp.asarray(origin, dtype=candidate.dtype)
    valid = jnp.isfinite(candidate) & (candidate >= 0.0) & (candidate <= float(upper))
    return jnp.where(valid, candidate, origin)


def _finite_or_origin(candidate: jax.Array, origin: jax.Array) -> jax.Array:
    """Reject nonfinite boundary replay values without clipping finite dynamics."""

    candidate = jnp.asarray(candidate)
    origin = jnp.asarray(origin, dtype=candidate.dtype)
    return jnp.where(jnp.isfinite(candidate), candidate, origin)


def _theta_mass_weights(theta: jax.Array, mu_total: jax.Array) -> jax.Array:
    """Broadcast positive column dry mass onto theta mass points."""

    theta = jnp.asarray(theta)
    mass_2d = jnp.asarray(mu_total, dtype=theta.dtype)
    mass_2d = jnp.where(jnp.isfinite(mass_2d) & (mass_2d > 0.0), mass_2d, 0.0)
    return jnp.broadcast_to(mass_2d[None, :, :], theta.shape)


def _theta_level_monotonic_bounds(
    origin: jax.Array,
    *,
    minimum_k: float = _THETA_LIMITER_MIN_K,
    maximum_k: float = _THETA_LIMITER_MAX_K,
) -> tuple[jax.Array, jax.Array]:
    """Return per-level monotonicity bounds for positive-definite theta advection."""

    origin = jnp.asarray(origin, dtype=jnp.float64)
    safe = jnp.where(jnp.isfinite(origin), origin, 0.5 * (float(minimum_k) + float(maximum_k)))
    lower = jnp.min(safe, axis=(1, 2), keepdims=True)
    upper = jnp.max(safe, axis=(1, 2), keepdims=True)
    lower = jnp.maximum(lower, float(minimum_k))
    upper = jnp.minimum(jnp.maximum(upper, lower), float(maximum_k))
    return lower, upper


def _first_limited_cell_xyz(mask: jax.Array) -> jax.Array:
    """Return first limited mass-cell coordinate as ``[x, y, z]`` or ``[-1, -1, -1]``."""

    flat = jnp.ravel(mask)
    count = jnp.sum(flat.astype(jnp.int32))
    flat_index = jnp.argmax(flat.astype(jnp.int32))
    ny = int(mask.shape[1])
    nx = int(mask.shape[2])
    z = flat_index // (ny * nx)
    rem = flat_index - z * ny * nx
    y = rem // nx
    x = rem - y * nx
    xyz = jnp.stack((x, y, z)).astype(jnp.int32)
    missing = jnp.full((3,), -1, dtype=jnp.int32)
    return jnp.where(count > 0, xyz, missing)


def _empty_theta_limiter_diagnostics(theta: jax.Array) -> dict[str, jax.Array]:
    """Build the INV-10 diagnostic record used when the limiter is inactive."""

    dtype = jnp.asarray(theta).dtype
    return {
        "theta_limited_cell_count": jnp.asarray(0, dtype=jnp.int32),
        "theta_first_limited_cell_xyz": jnp.full((3,), -1, dtype=jnp.int32),
        "theta_mass_before": jnp.asarray(0.0, dtype=dtype),
        "theta_mass_after": jnp.asarray(0.0, dtype=dtype),
        "theta_mass_residual": jnp.asarray(0.0, dtype=dtype),
    }


def _positive_definite_theta_increment_limiter(
    candidate: jax.Array,
    origin: jax.Array,
    mass: jax.Array,
    *,
    minimum_k: float = _THETA_LIMITER_MIN_K,
    maximum_k: float = _THETA_LIMITER_MAX_K,
    lower_bound: jax.Array | None = None,
    upper_bound: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Limit theta increments to a positive finite interval while conserving mass.

    Offending cells keep the RK direction but receive a smaller increment.  The
    removed mass-weighted theta increment is then redistributed over cells with
    available room, so feasible updates preserve the raw dycore scalar integral.
    """

    output_dtype = jnp.asarray(candidate).dtype
    candidate64 = jnp.asarray(candidate, dtype=jnp.float64)
    origin64 = jnp.asarray(origin, dtype=jnp.float64)
    mass64 = jnp.asarray(mass, dtype=jnp.float64)
    lower = jnp.asarray(float(minimum_k), dtype=jnp.float64)
    upper = jnp.asarray(float(maximum_k), dtype=jnp.float64)
    if lower_bound is not None:
        lower = jnp.maximum(lower, jnp.asarray(lower_bound, dtype=jnp.float64))
    if upper_bound is not None:
        upper = jnp.minimum(upper, jnp.asarray(upper_bound, dtype=jnp.float64))
    upper = jnp.maximum(upper, lower)
    midpoint = 0.5 * (lower + upper)

    safe_origin = jnp.where(jnp.isfinite(origin64), origin64, midpoint)
    safe_origin = jnp.minimum(jnp.maximum(safe_origin, lower), upper)
    finite_candidate = jnp.where(jnp.isfinite(candidate64), candidate64, safe_origin)
    raw_delta = finite_candidate - safe_origin

    over_upper = finite_candidate > upper
    under_lower = finite_candidate < lower
    invalid = ~jnp.isfinite(candidate64)
    limited_mask = invalid | over_upper | under_lower

    positive_delta = raw_delta > 0.0
    negative_delta = raw_delta < 0.0
    upper_alpha = (upper - safe_origin) / jnp.where(positive_delta, raw_delta, 1.0)
    lower_alpha = (lower - safe_origin) / jnp.where(negative_delta, raw_delta, -1.0)
    alpha = jnp.where(positive_delta, upper_alpha, jnp.where(negative_delta, lower_alpha, 1.0))
    alpha = jnp.where(limited_mask, jnp.minimum(jnp.maximum(alpha, 0.0), 1.0), 1.0)
    limited0 = safe_origin + alpha * raw_delta

    target_mass = jnp.sum(finite_candidate * mass64)
    mass0 = jnp.sum(limited0 * mass64)
    residual = target_mass - mass0
    add_room = upper - limited0
    subtract_room = limited0 - lower
    room = jnp.where(residual >= 0.0, add_room, subtract_room)
    capacity = jnp.sum(room * mass64)
    fraction = jnp.where(capacity > 0.0, jnp.minimum(jnp.abs(residual) / capacity, 1.0), 0.0)
    limited = limited0 + jnp.sign(residual) * fraction * room
    limited = jnp.minimum(jnp.maximum(limited, lower), upper)
    limited = limited.astype(output_dtype)

    after_mass = jnp.sum(limited.astype(jnp.float64) * mass64)
    diagnostics = {
        "theta_limited_cell_count": jnp.sum(limited_mask.astype(jnp.int32)),
        "theta_first_limited_cell_xyz": _first_limited_cell_xyz(limited_mask),
        "theta_mass_before": target_mass.astype(output_dtype),
        "theta_mass_after": after_mass.astype(output_dtype),
        "theta_mass_residual": (after_mass - target_mass).astype(output_dtype),
    }
    return limited, diagnostics


def _limit_guarded_mass_state(candidate: State, origin: State) -> State:
    """Keep finite positive dry mass without changing theta after physics/boundary."""

    mu_base = jnp.asarray(origin.mu_total) - jnp.asarray(origin.mu_perturbation)
    mu_perturbation = _finite_or_origin(candidate.mu_perturbation, origin.mu_perturbation)
    candidate_mu_total = mu_base + mu_perturbation
    valid_mu = jnp.isfinite(candidate_mu_total) & (candidate_mu_total >= 1.0)
    mu_perturbation = jnp.where(valid_mu, mu_perturbation, origin.mu_perturbation)
    mu_total = jnp.maximum(mu_base + mu_perturbation, jnp.asarray(1.0, dtype=mu_perturbation.dtype))
    mu_perturbation = mu_total - mu_base
    return candidate.replace(mu=mu_total, mu_total=mu_total, mu_perturbation=mu_perturbation)


def _limit_guarded_dynamics_state_with_diagnostics(candidate: State, origin: State) -> tuple[State, dict[str, jax.Array]]:
    """Apply the dycore theta safety net and dry-mass guard after one RK3 step.

    GUARDS-MUST-NOT-BE-LOAD-BEARING FIX (2026-06-01, operational-path-divergence
    sprint).  Previously this passed the per-level domain-MIN/MAX monotonic bounds
    (``_theta_level_monotonic_bounds(origin.theta)``) into the increment limiter and
    then mass-conservatively REDISTRIBUTED the clamped-away increment over the
    column.  On the operational d02/d03 path that made the guard LOAD-BEARING: over
    the cooling open ocean the coldest columns hit the per-level minimum, the
    suppressed cooling was treated as "removed mass" and pumped back as warming, so
    the integration drifted +3.3 K warm in the lowest levels over 6 h relative to the
    guards-off path that the v0.1.0 D02_VALIDATED proof used (and that matches
    CPU-WRF).  Root cause + isolation experiment: PERHOUR(guards-on) warm-drifts
    +3.3 K; PH_GUARDOFF (only difference = guards) collapses to the validated
    -0.1 K; see ``.agent/reviews/2026-06-01-opus-operational-path-divergence.md`` and
    ``proofs/v010_validation/path_divergence_case3.json``.

    The fix drops the tight per-level monotonic bounds so the limiter uses ONLY the
    WIDE physical envelope ``[_THETA_LIMITER_MIN_K, _THETA_LIMITER_MAX_K]`` =
    ``[0, 500] K`` plus the non-finite trap.  For any physically reasonable theta the
    envelope never fires (``limited_mask`` all-False), so the increment limiter is a
    strict identity AND its mass-redistribution residual is ~0 — i.e. it becomes a
    genuine non-load-bearing safety net that catches only NaN/Inf and true blow-ups,
    leaving the physical trajectory bit-equivalent to the guards-off integration.
    The idealized warm-bubble/Straka gates already run ``disable_guards=True`` so this
    path is a no-op for them; the change only affects the operational guards-on path.
    """

    mass = _theta_mass_weights(candidate.theta, candidate.mu_total)
    theta, diagnostics = _positive_definite_theta_increment_limiter(
        candidate.theta,
        origin.theta,
        mass,
    )
    limited = _limit_guarded_mass_state(candidate.replace(theta=theta), origin)
    return limited, diagnostics


def _limit_guarded_dynamics_state(candidate: State, origin: State) -> State:
    """Keep finite bounded dynamics from RK3 while preserving positive dry mass."""

    limited, _diagnostics = _limit_guarded_dynamics_state_with_diagnostics(candidate, origin)
    return limited


def _limit_theta_by_level(theta: jax.Array, origin_theta: jax.Array) -> jax.Array:
    """Back-compat thin envelope clip for diagnostic harness leaf-level interface.

    M11 removed the production [200K, 450K] envelope limiter in favor of the
    positive-definite increment limiter inside _limit_guarded_dynamics_state.
    The diagnostic harness still wants a leaf-level clip with origin fallback
    for instrumentation purposes; this preserves the old signature without
    changing production semantics (production calls the full state limiter).
    """
    lower_bound = jnp.asarray(200.0, dtype=theta.dtype)
    upper_bound = jnp.asarray(450.0, dtype=theta.dtype)
    in_envelope = jnp.isfinite(theta) & (theta >= lower_bound) & (theta <= upper_bound)
    return jnp.where(in_envelope, theta, jnp.clip(origin_theta, lower_bound, upper_bound))


def _with_save_family(carry: OperationalCarry, state: State, ww: jax.Array | None = None) -> OperationalCarry:
    """Update WRF ``*_save`` transition fields in resident operational carry."""

    ww_value = carry.ww if ww is None else ww
    mu_base = _base_mu(state)
    return carry.replace(
        state=state,
        muave=jnp.zeros_like(state.mu_perturbation),
        muts=mu_base,
        u_save=state.u,
        v_save=state.v,
        w_save=state.w,
        t_save=state.theta,
        ph_save=state.ph,
        mu_save=state.mu_perturbation,
        ww_save=ww_value,
    )


def _m6b_acoustic_tendencies(tendencies: Tendencies, base: Tendencies) -> Tendencies:
    """Legacy diagnostic import shim; no longer suppresses V tendencies."""

    del base
    return tendencies


def _horizontal_pressure_gradient_tendencies(state: State, namelist: OperationalNamelist) -> tuple[jax.Array, jax.Array]:
    """Compute WRF-shaped velocity PGF tendencies for operational RK u/v."""

    pressure, al, alt = diagnose_pressure_al_alt(state, None, namelist.metrics)
    cqu, cqv = moisture_coupling_factors(state)
    du_dt, dv_dt, _, _ = horizontal_pressure_gradient(
        state,
        None,
        namelist.metrics,
        pressure,
        al,
        alt,
        cqu,
        cqv,
        dx_m=namelist.grid.projection.dx_m,
        dy_m=namelist.grid.projection.dy_m,
        non_hydrostatic=True,
        top_lid=bool(namelist.top_lid),
    )
    return du_dt, dv_dt


def _acoustic_core_state(carry: OperationalCarry, namelist: OperationalNamelist) -> AcousticCoreState:
    state = carry.state
    theta_offset = _theta_base_offset(state.theta)
    theta_pert = (state.theta - theta_offset).astype(jnp.float64)
    theta_save_pert = (carry.t_save - theta_offset).astype(jnp.float64)
    theta_ave_pert = (carry.t_2ave - theta_offset).astype(jnp.float64)
    mu_base = _base_mu(state)
    mu_total = mu_base + state.mu_perturbation
    metrics = namelist.metrics
    # Real advance_w inputs for the legacy non-prep helper path so the WRF
    # implicit-w solve receives finite, consistent coefficients (matches the
    # production prep-path semantics): real c2a from the dry EOS, real dry cqw,
    # base pressure/geopotential, and terrain ht = phb(sfc)/g.
    p_base = (state.p_total - state.p_perturbation).astype(jnp.float64)
    ph_base = (state.ph_total - state.ph_perturbation).astype(jnp.float64)
    alt = _inverse_density_from_theta_pressure(
        state.theta.astype(jnp.float64), state.p_total.astype(jnp.float64)
    )
    c2a = CPOVCV * (p_base + state.p_perturbation.astype(jnp.float64)) / jnp.maximum(
        jnp.abs(alt), jnp.asarray(1.0e-12, dtype=alt.dtype)
    )
    nz = int(state.theta.shape[0])
    ny = int(state.theta.shape[1])
    nx = int(state.theta.shape[2])
    return AcousticCoreState(
        ww=carry.ww,
        ww_1=carry.ww_save,
        u=state.u,
        u_1=carry.u_save,
        v=state.v,
        v_1=carry.v_save,
        w=state.w,
        mu=state.mu_perturbation,
        mut=mu_base,
        muave=carry.muave,
        muts=carry.muts,
        muu=_u_face_average_2d(mu_total),
        muv=_v_face_average_2d(mu_total),
        mudf=carry.mudf,
        theta=theta_pert,
        theta_1=theta_save_pert,
        theta_ave=theta_ave_pert,
        theta_tend=namelist.tendencies.theta,
        mu_tend=namelist.tendencies.mu,
        ph_tend=carry.ph_tend,
        ph=state.ph_perturbation,
        p=state.p_perturbation,
        t_2ave=theta_ave_pert,
        dnw=metrics.dnw,
        fnm=metrics.fnm,
        fnp=metrics.fnp,
        rdnw=metrics.rdnw,
        c1h=metrics.c1h,
        c2h=metrics.c2h,
        msfuy=metrics.msfuy,
        msfvx_inv=1.0 / metrics.msfvx,
        msftx=metrics.msftx,
        msfty=metrics.msfty,
        coef_mut=mu_base,
        al=jnp.zeros_like(state.p_perturbation),
        alt=alt,
        p_base=p_base,
        ph_base=ph_base,
        cqu=jnp.ones_like(state.u, dtype=jnp.float64),
        cqv=jnp.ones_like(state.v, dtype=jnp.float64),
        msfux=metrics.msfux,
        msfvx=metrics.msfvx,
        msfvy=metrics.msfvy,
        cf1=metrics.cf1,
        cf2=metrics.cf2,
        cf3=metrics.cf3,
        c2a=c2a,
        cqw=dry_cqw(nz, ny, nx, dtype=jnp.float64),
        c1f=metrics.c1f,
        c2f=metrics.c2f,
        rdn=metrics.rdn,
        phb=ph_base,
        ph_1=carry.ph_save.astype(jnp.float64) - ph_base,
        ht=ph_base[0, :, :] / GRAVITY_M_S2,
        pm1=state.p_perturbation.astype(jnp.float64),
        ru_m=jnp.zeros_like(state.u, dtype=jnp.float64),
        rv_m=jnp.zeros_like(state.v, dtype=jnp.float64),
        ww_m=jnp.zeros_like(carry.ww),
        # Physical perturbation w from the carry save family (WRF w_save) for the
        # damp_opt=3 implicit Rayleigh damping in advance_w.
        w_save=carry.w_save.astype(jnp.float64),
    )


def _acoustic_core_state_from_prep(
    carry: OperationalCarry,
    prep: SmallStepPrepState,
    pressure: CalcPRhoStep0,
    namelist: OperationalNamelist,
    tendencies: Tendencies,
    *,
    lead_seconds=None,
) -> AcousticCoreState:
    """Build the acoustic work-state directly from WRF ``small_step_prep``."""

    state = prep.entry_state
    theta_pert = (state.theta - prep.theta_offset).astype(jnp.float64)
    ph_base = state.ph_total - state.ph_perturbation
    # F7H: WRF builds the large-step vertical PGF/buoyancy ``rw_tend`` ONCE per RK
    # stage in rk_tendency (module_em.F:1361-1368) by calling pg_buoy_w with the
    # stage diagnostic ``grid%p`` and the stage perturbation dry mass
    # ``mu' = mut - mub``.  In WRF that ``grid%p`` is the FULL-perturbation
    # ``calc_p_rho_phi`` diagnostic (module_big_step_utilities_em.F:1029,1083-1087)
    # built from the FULL ``ph'``, ``mu'`` and ``theta'`` — NOT the small-step
    # work-delta pressure.  Its ``rdn*(p[k]-p[k-1])`` interior PGF term
    # hydrostatically balances the ``-c1f*mu'`` weight of the perturbation column,
    # so the net interior forcing on a near-balanced thermal stays small.
    #
    # The previous F7G code fed ``pressure.p`` = ``calc_p_rho_wrf(prep)``, which is
    # built from ``prep.ph_work`` (= ph_ref - ph_cur ~ 0) and ``prep.mu_work``
    # (~0) — the small-step WORK-DELTA pressure, near zero and carrying NONE of the
    # ph'/mu' hydrostatic structure.  The PGF term then could not cancel the
    # ``-c1f*mu'`` weight, leaving a net forcing ~ g*c1f*mu' that grows as mu'
    # grows (w runaway).  Trace: proofs/f7h/full_p_compare.json (interior net
    # work_p >> full_p).  Fix = feed pg_buoy_w the full-perturbation grid%p via the
    # F7F-fixed diagnose_pressure_al_alt (the JAX calc_p_rho_phi), exactly as WRF
    # rk_tendency does.  ``pressure.p`` (work-delta) still correctly seeds the
    # substep ``p``/``pm1`` smdiv memory below.
    nz_stage = int(prep.theta_work.shape[0])
    ny_stage = int(prep.theta_work.shape[1])
    nx_stage = int(prep.theta_work.shape[2])
    mu_prime_stage = prep.mut - prep.mub  # stage perturbation dry mass mu' (WRF grid%mu_2)
    stage_base = BaseState(
        pb=prep.pb,
        phb=ph_base,
        mub=prep.mub,
        t0=jnp.asarray(prep.theta_offset),
        theta_base=jnp.full_like(state.theta, prep.theta_offset),
    )
    grid_p_full, _stage_al_full, _stage_alt_full = diagnose_pressure_al_alt(
        state, stage_base, namelist.metrics
    )
    rw_tend_stage = pg_buoy_w_dry(
        grid_p_full,
        mu_prime_stage,
        c1f=namelist.metrics.c1f,
        rdnw=namelist.metrics.rdnw,
        rdn=namelist.metrics.rdn,
        msfty=namelist.metrics.msfty,
        gravity=GRAVITY_M_S2,
    )
    # F7J item 2: WRF ``rk_tendency`` builds ``rw_tend`` as ``advect_w(w)`` (the
    # large-step vertical+horizontal advection of coupled w) THEN ``pg_buoy_w``
    # ADDS the vertical PGF/buoyancy (module_em.F:1011-1067 then :1361-1368).
    # ``tendencies.w`` is the COUPLED large-step w advection from
    # ``_augment_large_step_tendencies`` (``tendencies.w * mass_f``); fold it into
    # the stage ``rw_tend`` so the WRF assembly order is preserved.  Without #1
    # below it does not stabilise the mode (F7I wadv_fix_probe), but it is
    # WRF-correct and required together with the geopotential RHS.
    rw_tend_stage = rw_tend_stage + tendencies.w

    # F7J item 1 (PRIME): the large-step geopotential-equation RHS ``rhs_ph`` was
    # stubbed (``carry.ph_tend`` stayed 0; ``accumulate_ph_tend`` never wired in),
    # so the w/phi acoustic restoring loop never closed and the warm-bubble
    # buoyancy pumped without saturating.  WRF computes it once per RK stage in
    # ``rk_tendency`` (module_em.F:1254-1266 -> rhs_ph,
    # module_big_step_utilities_em.F:1365-2232) using the STAGE explicit omega
    # ``wwE = grid%ww`` and the STAGE geopotential perturbation ``ph``.  This is
    # the large-step (frozen-during-acoustic-loop) half of the geopotential
    # tendency; ``advance_w_wrf`` adds the small-step half (omega/ph_1 evolution).
    ph_tend_stage = rhs_ph_wrf(
        u=state.u,
        v=state.v,
        ww=carry.ww,
        ph=state.ph_perturbation,
        phb=ph_base,
        w=state.w,
        mut=prep.mut,
        muu=prep.muu,
        muv=prep.muv,
        c1f=namelist.metrics.c1f,
        c2f=namelist.metrics.c2f,
        fnm=namelist.metrics.fnm,
        fnp=namelist.metrics.fnp,
        rdnw=namelist.metrics.rdnw,
        rdx=1.0 / float(namelist.grid.projection.dx_m),
        rdy=1.0 / float(namelist.grid.projection.dy_m),
        msfty=namelist.metrics.msfty,
        non_hydrostatic=True,
        gravity=GRAVITY_M_S2,
    )

    # WIND-FIX: stage-constant coupled WORK-array boundary targets for the NORMAL
    # momentum, consumed by ``advance_uv_wrf`` inside the acoustic loop.  Built
    # once per RK stage from the time-interpolated decoupled wrfbdy leaf so that
    # ``small_step_finish_wrf`` reconstructs the boundary velocity ``u_bdy``:
    #     u = (msf*u_work + u_save*mass_cur)/mass_stage
    #  => u_work_bdy = (u_bdy*mass_stage - u_save*mass_cur)/msf .
    # Only staged when the real-case lateral boundary is active; ``None`` keeps the
    # idealized / replay / bare-core paths on the unmodified PGF advance.
    u_work_bdy = None
    v_work_bdy = None
    if bool(namelist.run_boundary) and lead_seconds is not None:
        c1h = namelist.metrics.c1h[:, None, None]
        c2h = namelist.metrics.c2h[:, None, None]
        mass_u_cur = c1h * prep.muu[None, :, :] + c2h
        mass_u_stage = c1h * prep.muus[None, :, :] + c2h
        mass_v_cur = c1h * prep.muv[None, :, :] + c2h
        mass_v_stage = c1h * prep.muvs[None, :, :] + c2h
        cadence = float(namelist.boundary_config.update_cadence_s)
        u_bdy_strip = interpolate_boundary_leaf(state.u_bdy, lead_seconds, cadence)
        v_bdy_strip = interpolate_boundary_leaf(state.v_bdy, lead_seconds, cadence)
        u_work_bdy = normal_bdy_work_target_u(
            u_bdy_strip, prep.u_save, mass_u_cur, mass_u_stage, namelist.metrics.msfuy,
            config=namelist.boundary_config,
        )
        v_work_bdy = normal_bdy_work_target_v(
            v_bdy_strip, prep.v_save, mass_v_cur, mass_v_stage, namelist.metrics.msfvx,
            config=namelist.boundary_config,
        )

    # P0-6 (2026-06-01): NESTED-child ph'/w boundary forcing (d03 T2 Exner bias).
    # Active ONLY for the nested replay path (run_boundary, lateral boundary active,
    # AND boundary_config.force_geopotential == False -- the d03 case).  For d02
    # self-replay (force_geopotential=True) and idealized/bare-core (lead_seconds
    # None / run_boundary False) these stay None and the additions are skipped, so
    # those paths are byte-for-byte unchanged.
    #
    # WRF cadence (solve_em.F:940 relax_bdy_dry once per stage -> rk_addtend_dry
    # folds ph_tendf/msfty into ph_tend, rw_tendf/msfty into rw_tend; the in-loop
    # advance_w consumes ph_tend/rw_tend every substep; spec_bdyupdate_ph pins the
    # spec_zone row of ph_2 after advance_w):
    #   * relax zone -> add the mass-coupled relax tendency to ph_tend_stage /
    #     rw_tend_stage here (so it flows through advance_w coupled with w);
    #   * spec zone  -> stage the full-ring parent ph' target + ph_save for the
    #     in-loop spec_bdyupdate_ph applied inside acoustic_substep_core.
    ph_bdy_target_full = None
    ph_save_for_spec = None
    if (
        bool(namelist.run_boundary)
        and lead_seconds is not None
        and not bool(namelist.boundary_config.force_geopotential)
    ):
        cfg_b = namelist.boundary_config
        cadence = float(cfg_b.update_cadence_s)
        ph_bdy_strip = interpolate_boundary_leaf(state.ph_bdy, lead_seconds, cadence)
        # relax-zone ph' tendency (mass-coupled, /msfty) -> add into ph_tend_stage.
        if bool(getattr(cfg_b, "nested_ph_relax", True)):
            ph_relax = nested_ph_relax_tendency(
                state.ph_perturbation,
                ph_bdy_strip,
                prep.mut,
                namelist.metrics.msfty,
                namelist.metrics.c1f,
                namelist.metrics.c2f,
                float(namelist.dt_s),
                cfg_b,
            )
            ph_tend_stage = ph_tend_stage + ph_relax
        # relax-zone w tendency (nested only) -> add into rw_tend_stage.  Default
        # OFF: the parent 3km w leaf interpolated to the 1km child is a poor target
        # and pumps interior vertical motion (d03 short-run hour-1 with w-relax ON:
        # interior theta' +11.6 K; the pressure collapse is delivered by ph-relax).
        if bool(getattr(cfg_b, "nested_w_relax", False)):
            w_bdy_strip = interpolate_boundary_leaf(state.w_bdy, lead_seconds, cadence)
            w_relax = nested_w_relax_tendency(
                state.w,
                w_bdy_strip,
                prep.mut,
                namelist.metrics.msfty,
                namelist.metrics.c1f,
                namelist.metrics.c2f,
                float(namelist.dt_s),
                cfg_b,
            )
            rw_tend_stage = rw_tend_stage + w_relax
        # spec-zone (outer row) ph' target for the in-loop spec_bdyupdate_ph.
        if bool(getattr(cfg_b, "nested_ph_spec", True)):
            nzp1 = int(state.ph_perturbation.shape[0])
            ny_f = int(state.ph_perturbation.shape[1])
            nx_f = int(state.ph_perturbation.shape[2])
            ph_bdy_target_full = _full_ring_target_from_leaf(
                ph_bdy_strip, nzp1, ny_f, nx_f, state.ph_perturbation.dtype
            )
            ph_save_for_spec = prep.ph_save

    return AcousticCoreState(
        ww=carry.ww,
        ww_1=prep.ww_save,
        u=prep.u_work,
        u_1=prep.u_save,
        v=prep.v_work,
        v_1=prep.v_save,
        w=prep.w_work,
        mu=prep.mu_save + prep.mu_work,
        mut=prep.mut,
        # F7G: stage-entry small-step mass-WORK average is ZERO; advance_mu_t
        # (module_small_step_em.F:1102-1108) rebuilds it from actual small-step
        # mass evolution.  For a fixed-mass mu'=0 thermal it stays zero.
        muave=jnp.zeros_like(prep.mu_work),
        muts=prep.muts,
        muu=prep.muu,
        muv=prep.muv,
        mudf=carry.mudf,
        theta=theta_pert,
        theta_1=prep.t_save,
        # F7G: stage-entry small-step WORK-theta average is ZERO (the coupled work
        # theta t_2 is zero at a fresh RK stage on a fixed-mass rest thermal); the
        # WRF advance_w t_2ave half-step (module_small_step_em.F:1341-1344) builds
        # it up from actual small-step evolution.  Seeding the full initialized
        # theta here was the double-count bug (gpt-council-findings.md §3.5).
        theta_ave=jnp.zeros_like(prep.theta_work),
        # Large-step coupled theta / mu tendencies from rk_tendency+rk_addtend_dry
        # (advection + diffusion), consumed by advance_mu_t (t_2 += msfty*dts*t_tend).
        theta_tend=tendencies.theta,
        mu_tend=tendencies.mu,
        # F7J: real WRF rhs_ph large-step geopotential tendency (was stub=0).
        ph_tend=ph_tend_stage,
        ph=prep.ph_work,
        p=pressure.p,
        t_2ave=jnp.zeros_like(prep.theta_work),
        dnw=namelist.metrics.dnw,
        fnm=namelist.metrics.fnm,
        fnp=namelist.metrics.fnp,
        rdnw=namelist.metrics.rdnw,
        c1h=namelist.metrics.c1h,
        c2h=namelist.metrics.c2h,
        msfuy=namelist.metrics.msfuy,
        msfvx_inv=1.0 / namelist.metrics.msfvx,
        msftx=namelist.metrics.msftx,
        msfty=namelist.metrics.msfty,
        coef_mut=prep.muts,
        u_tend=tendencies.u,
        v_tend=tendencies.v,
        p_base=prep.pb,
        ph_base=ph_base,
        al=pressure.al,
        alt=prep.alt,
        cqu=prep.cqu,
        cqv=prep.cqv,
        msfux=namelist.metrics.msfux,
        msfvx=namelist.metrics.msfvx,
        msfvy=namelist.metrics.msfvy,
        cf1=namelist.metrics.cf1,
        cf2=namelist.metrics.cf2,
        cf3=namelist.metrics.cf3,
        theta_work_reference=prep.theta_1,
        # Initialise the coupled-theta work leaf so the lax.scan carry structure
        # is invariant across substeps (advance_mu_t fills it each substep).
        theta_coupled_work=prep.theta_work,
        c2a=prep.c2a,
        cqw=dry_cqw(
            int(prep.theta_work.shape[0]),
            int(prep.theta_work.shape[1]),
            int(prep.theta_work.shape[2]),
            dtype=prep.theta_work.dtype,
        ),
        c1f=namelist.metrics.c1f,
        c2f=namelist.metrics.c2f,
        rdn=namelist.metrics.rdn,
        phb=state.ph_total - state.ph_perturbation,
        ph_1=prep.ph_1,
        # Terrain height ht = phb(surface)/g (WRF advance_w lower BC :1417-1429).
        ht=(state.ph_total - state.ph_perturbation)[0, :, :] / GRAVITY_M_S2,
        pm1=pressure.pm1,
        ru_m=jnp.zeros_like(prep.u_work),
        rv_m=jnp.zeros_like(prep.v_work),
        ww_m=jnp.zeros_like(carry.ww),
        # F7G: the once-per-RK-stage pg_buoy_w tendency from the stage grid%p/mu'
        # (computed above), carried UNCHANGED through all acoustic substeps.  The
        # legacy per-substep ``p_buoy`` recompute is disabled (None).
        p_buoy=None,
        rw_tend_pg_buoy=rw_tend_stage,
        # Uncoupled physical perturbation w saved by small_step_prep (WRF :272);
        # consumed by the damp_opt=3 implicit Rayleigh w-damping in advance_w.
        w_save=prep.w_save,
        # WIND-FIX: NORMAL-momentum boundary work targets (None unless real-case
        # boundary is active); see advance_uv_wrf / boundary_apply.apply_normal_bdy_work.
        u_work_bdy=u_work_bdy,
        v_work_bdy=v_work_bdy,
        # P0-6: NESTED ph' spec-zone in-loop target + stage-entry ph_save (None
        # unless the nested force_geopotential=False boundary is active).
        ph_bdy_target=ph_bdy_target_full,
        ph_save_for_spec=ph_save_for_spec,
    )


def _carry_from_acoustic_core(acoustic: AcousticCoreState, template: State, theta_offset: jax.Array) -> OperationalCarry:
    theta = acoustic.theta + theta_offset
    p_total = template.p_total - template.p_perturbation + acoustic.p
    ph_total = template.ph_total - template.ph_perturbation + acoustic.ph
    mu_base = template.mu_total - template.mu_perturbation
    # ``acoustic_substep_core`` returns ``advanced["mu"]``: the total physical
    # perturbation needed by ``advance_mu_t`` to preserve ``mu_save`` on the next
    # small step.  ``muts`` remains the WRF work array ``mut + mu_work``.
    mu_perturbation = acoustic.mu
    mu_total = mu_base + mu_perturbation
    next_state = template.replace(
        u=acoustic.u,
        v=acoustic.v,
        w=acoustic.w,
        theta=theta,
        p=p_total,
        p_total=p_total,
        p_perturbation=acoustic.p,
        ph=ph_total,
        ph_total=ph_total,
        ph_perturbation=acoustic.ph,
        mu=mu_total,
        mu_total=mu_total,
        mu_perturbation=mu_perturbation,
    )
    return OperationalCarry(
        state=next_state,
        t_2ave=acoustic.t_2ave + theta_offset,
        ww=acoustic.ww,
        mudf=acoustic.mudf,
        muave=acoustic.muave,
        muts=acoustic.muts,
        ph_tend=acoustic.ph_tend,
        u_save=next_state.u,
        v_save=next_state.v,
        w_save=next_state.w,
        t_save=next_state.theta,
        ph_save=next_state.ph,
        mu_save=acoustic.mu,
        ww_save=acoustic.ww,
        # rthraten is a physics-layer held tendency refreshed in the physics
        # chain (rrtmg cadence), not the dycore acoustic core; this legacy
        # single-substep constructor (test-only path) carries no radiation.
        rthraten=jnp.zeros_like(next_state.theta),
    )


def _refresh_grid_p_from_finished(next_state: State, prep: SmallStepPrepState, namelist: OperationalNamelist) -> State:
    """Recompute WRF ``grid%p`` from the finished physical ``ph'`` and ``theta``.

    WRF closes every RK step by calling ``calc_p_rho_phi`` (solve_em.F:6180,
    :7542) which rebuilds the diagnostic perturbation pressure ``grid%p`` (and
    ``al``) from the updated geopotential ``ph`` and theta
    (module_big_step_utilities_em.F:1029, :1083-1087).  The next RK stage's
    large-step horizontal PGF and once-per-stage ``pg_buoy_w`` then act on THAT
    refreshed pressure.

    The JAX operational path previously carried ``p_perturbation`` =
    ``calc_p_rho_step`` work pressure (a delta-from-reference, O(1-10 Pa) for a
    near-balanced thermal), which is NOT the WRF ``grid%p`` diagnostic
    (O(1e3-1e4 Pa) once ``ph'`` evolves).  Feeding that stale O(1) pressure to
    the next stage suppressed the restoring vertical/horizontal PGF, leaving a
    near-constant net vertical force -> w runaway (see proofs/f7h, GPT bughunt
    §2).  This refresh restores the WRF closing diagnostic.  The acoustic substep
    still uses ``calc_p_rho_step`` for its own work-array pressure + smdiv memory.
    """

    base = BaseState(
        pb=prep.pb,
        phb=next_state.ph_total - next_state.ph_perturbation,
        mub=prep.mub,
        t0=jnp.asarray(prep.theta_offset),
        theta_base=jnp.full_like(next_state.theta, prep.theta_offset),
    )
    p_pert, _al, _alt = diagnose_pressure_al_alt(next_state, base, namelist.metrics)
    p_base = next_state.p_total - next_state.p_perturbation
    p_total = p_base + p_pert
    return next_state.replace(
        p=p_total, p_total=p_total, p_perturbation=p_pert,
    )


def _carry_from_finished_stage(
    carry: OperationalCarry,
    prep: SmallStepPrepState,
    acoustic: AcousticCoreState,
    namelist: OperationalNamelist | None = None,
) -> OperationalCarry:
    next_state = small_step_finish_wrf(prep, acoustic)
    if namelist is not None:
        next_state = _refresh_grid_p_from_finished(next_state, prep, namelist)
    ww = acoustic.ww + prep.ww_save
    return carry.replace(
        state=next_state,
        t_2ave=acoustic.t_2ave + prep.theta_offset,
        ww=ww,
        mudf=acoustic.mudf,
        muave=acoustic.muave,
        muts=acoustic.muts,
        ph_tend=acoustic.ph_tend,
        u_save=prep.u_save,
        v_save=prep.v_save,
        w_save=prep.w_save,
        t_save=prep.t_save + prep.theta_offset,
        ph_save=prep.ph_save,
        mu_save=prep.mu_save,
        ww_save=prep.ww_save,
    )


def _operational_acoustic_substep_core(carry: OperationalCarry, namelist: OperationalNamelist, dt_sub: float) -> OperationalCarry:
    """Run one operational acoustic substep through the shared core."""

    state = apply_halo(carry.state, halo_spec(namelist.grid))
    theta_offset = _theta_base_offset(state.theta)
    acoustic = _acoustic_core_state(carry.replace(state=state), namelist)
    # WRF solve_em.F:2409-2717 builds the vertical-solve coefficients for the
    # acoustic small step before solve_em.F:3065 enters the recurrence.
    a, alpha, gamma = calc_coef_w_wrf_coefficients(
        acoustic.mut,
        namelist.metrics,
        dt=float(dt_sub),
        epssm=float(namelist.epssm),
        top_lid=bool(namelist.top_lid),
        cqw=acoustic.cqw,
        c2a=acoustic.c2a,
    )
    next_acoustic = acoustic_substep_core(
        acoustic,
        a=a,
        alpha=alpha,
        gamma=gamma,
        cfg=AcousticCoreConfig(
            dt=float(dt_sub),
            dx=float(namelist.grid.projection.dx_m),
            dy=float(namelist.grid.projection.dy_m),
            epssm=float(namelist.epssm),
            top_lid=bool(namelist.top_lid),
            w_damping=int(namelist.w_damping),
            damp_opt=int(namelist.damp_opt),
            dampcoef=float(namelist.dampcoef),
            zdamp=float(namelist.zdamp),
        ),
        cqw=acoustic.cqw,
    )
    return _carry_from_acoustic_core(next_acoustic, state, theta_offset)


def _acoustic_scan(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    *,
    stage: _RKStageDescriptor,
    prep: SmallStepPrepState,
    pressure: CalcPRhoStep0,
    tendencies: Tendencies,
    lead_seconds=None,
) -> OperationalCarry:
    acoustic = _acoustic_core_state_from_prep(
        carry, prep, pressure, namelist, tendencies, lead_seconds=lead_seconds
    )
    if bool(namelist.use_vertical_solver):
        # WRF calc_coef_w uses the FULL dry mass ``mut`` (solve_em.F:2676-2681),
        # real ``c2a`` from small_step_prep, and the real dry ``cqw``.
        cqw_field = dry_cqw(
            int(prep.theta_work.shape[0]),
            int(prep.theta_work.shape[1]),
            int(prep.theta_work.shape[2]),
            dtype=prep.theta_work.dtype,
        )
        a, alpha, gamma = calc_coef_w_wrf_coefficients(
            prep.mut,
            namelist.metrics,
            dt=float(stage.dts_rk),
            epssm=float(namelist.epssm),
            top_lid=bool(namelist.top_lid),
            cqw=cqw_field,
            c2a=prep.c2a,
        )

        stage_cfg = AcousticCoreConfig(
            dt=float(stage.dts_rk),
            dx=float(namelist.grid.projection.dx_m),
            dy=float(namelist.grid.projection.dy_m),
            epssm=float(namelist.epssm),
            top_lid=bool(namelist.top_lid),
            w_damping=int(namelist.w_damping),
            damp_opt=int(namelist.damp_opt),
            dampcoef=float(namelist.dampcoef),
            zdamp=float(namelist.zdamp),
            # WIND-FIX: full model dt so the in-loop normal-momentum relaxation
            # weight is scaled to a per-substep increment.
            dt_full=float(namelist.dt_s),
        )

        def body(scan_acoustic: AcousticCoreState, _):
            return acoustic_substep_core(
                scan_acoustic,
                a=a,
                alpha=alpha,
                gamma=gamma,
                cfg=stage_cfg,
                cqw=cqw_field,
            ), None

        acoustic, _ = jax.lax.scan(body, acoustic, xs=None, length=int(stage.number_of_small_timesteps))
        next_carry = _carry_from_finished_stage(carry, prep, acoustic, namelist)
        return next_carry.replace(state=apply_halo(next_carry.state, halo_spec(namelist.grid)))

    del tendencies
    return _with_save_family(carry, carry.state)


def _augment_large_step_tendencies(
    haloed: State, tendencies: Tendencies, namelist: OperationalNamelist, *, rk_step: int = 3
) -> Tendencies:
    """Add WRF explicit diffusion + flux-form scalar advection to the large step.

    All contributions are returned as *uncoupled* tendencies to match the
    operational RK convention (``add_scaled_tendencies`` adds them uncoupled,
    then ``small_step_prep`` couples).  Sources:
    * 6th-order monotonic filter -- ``module_big_step_utilities_em.F:6504-6920``.
    * constant-K diffusion (Straka ν) -- ``:2999-3234``.
    * flux-form theta advection -- ``module_advect_em.F:3029-4359`` (h=5/v=3).
    """

    metrics = namelist.metrics
    grid = namelist.grid
    dx = float(grid.projection.dx_m)
    dy = float(grid.projection.dy_m)
    # mean physical dz from the geopotential column (for the const-K vertical term).
    ph = haloed.ph_total
    dz = jnp.maximum(jnp.mean((ph[1:] - ph[:-1]) / GRAVITY_M_S2), jnp.asarray(1.0, dtype=ph.dtype))

    # All large-step tendencies are built *coupled* (mass-weighted) so they net
    # correctly with the coupled small-step work arrays consumed by advance_uv /
    # advance_mu_t / advance_w (u_work = mass*u etc.).  WRF rk_tendency works in
    # the coupled ru/rv/rw/t_tend space (module_em.F:855-1388); advance_uv adds
    # ``u += dts*ru_tend`` to the coupled u (module_small_step_em.F:805), and
    # advance_mu_t adds ``t_2 += msfty*dts*t_tend`` to the coupled theta
    # (module_small_step_em.F theta update).  Face dry-air masses below match the
    # coupling in small_step_prep_wrf.
    mu_total = haloed.mu_total
    muu = _u_face_average_2d(mu_total)
    muv = _v_face_average_2d(mu_total)
    mass_u = metrics.c1h[:, None, None] * muu[None, :, :] + metrics.c2h[:, None, None]
    mass_v = metrics.c1h[:, None, None] * muv[None, :, :] + metrics.c2h[:, None, None]
    mass_h = metrics.c1h[:, None, None] * mu_total[None, :, :] + metrics.c2h[:, None, None]
    mass_f = metrics.c1f[:, None, None] * mu_total[None, :, :] + metrics.c2f[:, None, None]

    # Advection from compute_advection_tendencies is an UNCOUPLED velocity/scalar
    # acceleration; couple it by the field-specific face mass so it lives in the
    # same coupled tendency space as the PGF and the small-step work arrays.
    u_t = tendencies.u * mass_u
    v_t = tendencies.v * mass_v
    w_t = tendencies.w * mass_f
    th_t = tendencies.theta * mass_h

    if bool(namelist.use_flux_advection):
        # WRF flux-form mass-coupled advection (h=5/v=3).  The *_flux helpers
        # return the COUPLED tendency d(mu*field)/dt, so they replace the
        # primitive coupled products built above (not add to them).
        vel = couple_velocities_periodic(
            haloed.u,
            haloed.v,
            mu_total,
            c1h=metrics.c1h,
            c2h=metrics.c2h,
            dnw=metrics.dnw,
            rdx=1.0 / dx,
            rdy=1.0 / dy,
        )
        # --- momentum: WRF advect_u/advect_v/advect_w (conservative flux form) ---
        # The previous JAX path advanced momentum with the *advective* (non-
        # conservative) primitive form u*du/dx (advection.py advect_u_face),
        # which does not conserve momentum and lets the Straka cold-front outflow
        # pile up instead of propagating (front crawls ~5 m/s while head |w| runs
        # away).  WRF advances coupled momentum with mass-flux-form advect_u/v/w
        # (module_advect_em.F:126/1530/4364).  Confirmed against pristine WRF
        # v4.7.1 em_grav2d_x ground truth (proofs/m9/wrf_em_grav2d_x_front_*):
        # WRF max|w| saturates ~22 m/s and the front reaches ~4.25 km by 300 s,
        # while the primitive JAX path detonates ~270-300 s with a stalled front.
        u_t = namelist.tendencies.u * mass_u + advect_u_flux(
            haloed.u, vel, rdx=1.0 / dx, rdy=1.0 / dy,
            rdzw=metrics.rdnw, fzm=metrics.fnm, fzp=metrics.fnp,
        )
        v_t = namelist.tendencies.v * mass_v + advect_v_flux(
            haloed.v, vel, rdx=1.0 / dx, rdy=1.0 / dy,
            rdzw=metrics.rdnw, fzm=metrics.fnm, fzp=metrics.fnp,
        )
        w_t = namelist.tendencies.w * mass_f + advect_w_flux(
            haloed.w, vel, rdx=1.0 / dx, rdy=1.0 / dy,
            rdn=metrics.rdn, fzm=metrics.fnm, fzp=metrics.fnp,
            top_lid=bool(namelist.top_lid),
        )
        # --- scalar theta: WRF advect_scalar (h=5/v=3) ---
        theta_offset = _theta_base_offset(haloed.theta)
        coupled_tend = advect_scalar_flux(
            haloed.theta - theta_offset,
            vel,
            mut=mu_total,
            c1=metrics.c1h,
            rdx=1.0 / dx,
            rdy=1.0 / dy,
            rdzw=metrics.rdnw,
            fzm=metrics.fnm,
            fzp=metrics.fnp,
        )
        # tendencies.theta carries the base zero; replace the advective theta part
        # with the flux-form coupled tendency.
        th_t = namelist.tendencies.theta * mass_h + coupled_tend

    if int(namelist.diff_6th_opt) != 0:
        f = float(namelist.diff_6th_factor)
        dt_diff = float(namelist.dt_s)
        u_t = u_t + mass_u * sixth_order_diffusion_tendency(haloed.u, dt=dt_diff, diff_6th_factor=f)
        v_t = v_t + mass_v * sixth_order_diffusion_tendency(haloed.v, dt=dt_diff, diff_6th_factor=f)
        w_t = w_t + mass_f * sixth_order_diffusion_tendency(haloed.w, dt=dt_diff, diff_6th_factor=f)
        th_t = th_t + mass_h * sixth_order_diffusion_tendency(haloed.theta, dt=dt_diff, diff_6th_factor=f)

    nu = float(namelist.const_nu_m2_s)
    if nu > 0.0:
        # WRF diff_opt=2 / km_opt=1 constant-K diffusion on u, v, w AND theta
        # (Straka ν=75).  Plain K∇² form (F7L baseline).  NOTE (F7M): WRF actually
        # diffuses MOMENTUM with the deformation stress tensor — factor-2 diagonal
        # (D11=2 du/dx, D33=2 dw/dz) plus du/dz<->dw/dx cross terms
        # (module_diffusion_em.F cal_deform_and_div :41-47, horizontal/
        # vertical_diffusion_{u,w}_2 :3118-4784, cal_titau_* :5331-5744).  F7M
        # implemented that deformation form (constant_k_deformation_momentum_
        # tendency) and verified it ~2-3x stronger than this Laplacian, but it left
        # the Straka 180s trace byte-identical and still detonated at 240s — the
        # residual is NOT diffusion-controlled (it is the touchdown horizontal-
        # spreading coupling; see proofs/f7m/wrf_vs_jax_straka_front.json).  The
        # deformation operator carries a half-cell cross-term stagger approximation
        # and did not help, so the plain WRF-faithful K∇² baseline is retained
        # pending the touchdown root-cause fix.
        # F7N: use the mass-CONSERVATIVE flux-divergence form d/dx_j(mass*K*d./dx_j)
        # (WRF horizontal_diffusion_s/vertical_diffusion, module_diffusion_em.F:
        # 2999-3018) instead of the non-conservative mass*K*∇² form.  The latter
        # leaked the dry-column mass integral at the sharp Straka cold front
        # (relative drift ~3.4e-8 over 900 s once the touchdown 2Δz fix let Straka
        # run to completion).  The conservative helper already carries the field
        # face mass, so it is NOT multiplied by mass again.  mass_u/mass_v are the
        # u/v face masses (u-face x-diffusion uses the u-face mass; conserves the
        # mass-weighted momentum integral to the same order as WRF).
        #
        # Sprint U (P0-2): theta ALWAYS uses the conservative scalar flux-divergence
        # (WRF horizontal_diffusion_s).  MOMENTUM (u, v, w) optionally uses the WRF
        # deformation-tensor operator (diff_opt=2/km_opt=1, the factor-2 diagonal +
        # du/dz<->dw/dx cross terms) when use_deformation_momentum_diffusion is set;
        # otherwise it keeps the scalar flux-divergence (the F7N close default).  The
        # deformation operator returns the UNCOUPLED tendency K*(2u_xx+u_zz+w_xz);
        # multiply by the field face mass to enter the dry-mass-coupled tendency
        # space, exactly as the scalar diffusion does.  On the flat hydrostatic slab
        # WRF's g*dz/dnw*rho coupling reduces to the same dry-mass face weight
        # (|dnw|=rho*g*dz/mu => g*dz/|dnw|*rho = mu), so this is WRF-faithful.
        th_t = th_t + conservative_constant_k_diffusion_tendency(haloed.theta, mass=mass_h, k_m2_s=nu, dx_m=dx, dy_m=dy, dz_m=dz)
        if bool(namelist.use_deformation_momentum_diffusion):
            unit_rho = jnp.ones_like(haloed.theta)
            du_def, dw_def = wrf_deformation_momentum_tendency(
                haloed.u, haloed.w, rho=unit_rho, k_m2_s=nu, dx_m=dx, dz_m=dz,
            )
            u_t = u_t + mass_u * du_def
            w_t = w_t + mass_f * dw_def
            # v: one-row slab has degenerate y-deformation; keep the scalar
            # flux-divergence (D22/D12 vanish for ny=1, so this is identical to the
            # deformation v-diffusion on the slab).
            v_t = v_t + conservative_constant_k_diffusion_tendency(haloed.v, mass=mass_v, k_m2_s=nu, dx_m=dx, dy_m=dy, dz_m=dz)
        else:
            u_t = u_t + conservative_constant_k_diffusion_tendency(haloed.u, mass=mass_u, k_m2_s=nu, dx_m=dx, dy_m=dy, dz_m=dz)
            v_t = v_t + conservative_constant_k_diffusion_tendency(haloed.v, mass=mass_v, k_m2_s=nu, dx_m=dx, dy_m=dy, dz_m=dz)
            w_t = w_t + conservative_constant_k_diffusion_tendency(haloed.w, mass=mass_f, k_m2_s=nu, dx_m=dx, dy_m=dy, dz_m=dz)

    # WRF rk_tendency adds the large-step horizontal pressure-gradient force to
    # the *coupled* large-step ru/rv_tend (module_em.F:1325 ->
    # horizontal_pressure_gradient, module_big_step_utilities_em.F:2459-2466).
    # This is the steady gradient that drives the mean circulation; it is a
    # DIFFERENT split term from the small-step advance_uv acoustic PGF
    # (module_small_step_em.F:828-868), which uses the work-array perturbation
    # pressure that restarts ~0 at each RK stage -- NOT a double-count.  The
    # operational cadence applies ru/rv_tend only inside advance_uv (one
    # forward-Euler per acoustic substep, u += dts*ru_tend), matching WRF; the
    # earlier add_scaled_tendencies forward-Euler of the dynamics fields has been
    # removed so there is no double-application.
    ru_pgf, rv_pgf = large_step_horizontal_pgf(
        haloed,
        metrics,
        dx_m=dx,
        dy_m=dy,
        non_hydrostatic=True,
        top_lid=bool(namelist.top_lid),
    )
    u_t = u_t + ru_pgf
    v_t = v_t + rv_pgf

    # WRF rk_tendency adds the Coriolis force to the SAME coupled ru/rv_tend
    # immediately AFTER the horizontal PGF (module_em.F:717 PGF then :761 coriolis;
    # body module_big_step_utilities_em.F:3640).  This is the rotational body force
    # that lets the interior flow reach geostrophic balance; its complete absence
    # was the proven root cause of the below-persistence, wrong-sign-u Canary winds
    # (proofs/wind/case3_v10_momentum_budget_findings.md).  ``f=0`` for idealized
    # cases makes every Coriolis term identically zero, so the warm-bubble / Straka
    # / oracle dycore gates stay bit-identical.  ``specified`` follows WRF's
    # nested/specified boundary edge-face exclusion for the real (boundary-driven)
    # case; for periodic idealized runs the choice is moot under f=0.
    ru_cor, rv_cor = large_step_coriolis(
        haloed,
        metrics,
        specified=bool(namelist.run_boundary),
    )
    u_t = u_t + ru_cor
    v_t = v_t + rv_cor

    tendencies = tendencies.replace(u=u_t, v=v_t, w=w_t, theta=th_t)

    # WRF rk_addtend_dry per-stage merge (module_em.F:1711-1786): field-specific
    # map/mass coupling of RK1-fixed physics tendencies.  Physics-off + periodic
    # dry gate => all *_tendf == 0, so this is numerically identity but keeps the
    # cadence WRF-faithful for the coupled physics path.
    return rk_addtend_dry(
        tendencies,
        DryPhysicsTendencies(),
        rk_step=int(rk_step),
        metrics=metrics,
        mut=_base_mu(haloed),
    )


def _rk_scan_step(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    *,
    debug: bool = False,
    lead_seconds=None,
) -> OperationalCarry:
    origin = apply_halo(carry.state, halo_spec(namelist.grid))
    rk1_reference = origin

    def advance_stage(stage_carry: OperationalCarry, stage: _RKStageDescriptor) -> OperationalCarry:
        haloed = apply_halo(stage_carry.state, halo_spec(namelist.grid))
        # WRF rk_tendency builds the per-stage large-step tendencies (advection,
        # diffusion, and the LARGE-STEP horizontal PGF; module_em.F:1325) and
        # rk_addtend_dry merges the RK1-fixed physics tendencies; both are inside
        # _augment_large_step_tendencies.  The large-step momentum tendency is
        # consumed ONLY inside the acoustic small-step advance_uv (one
        # forward-Euler per substep: u += dts*ru_tend, module_small_step_em.F:805),
        # exactly as WRF does -- there is no separate add_scaled_tendencies
        # forward-Euler of the dynamics prognostics (that was the Sprint A/B
        # double-application that prevented u/v from moving).  The stage-entry
        # physical state (carried forward across RK stages) is the small-step
        # prognostic ``u_2``; ``rk1_reference`` is the RK reference ``u_1``.
        tendencies = compute_advection_tendencies(haloed, namelist.tendencies, namelist.grid)
        tendencies = _augment_large_step_tendencies(
            haloed, tendencies, namelist, rk_step=int(stage.rk_step)
        )
        candidate = apply_halo(stage_carry.state, halo_spec(namelist.grid))
        prep = small_step_prep_wrf(
            candidate,
            int(stage.rk_step),
            float(stage.dt_rk),
            metrics=namelist.metrics,
            reference_state=rk1_reference,
            ww=stage_carry.ww,
        )
        pressure = calc_p_rho_wrf(prep, step=0, non_hydrostatic=True)
        stage_carry = _acoustic_scan(
            stage_carry.replace(state=candidate),
            namelist,
            stage=stage,
            prep=prep,
            pressure=pressure,
            tendencies=tendencies,
            lead_seconds=lead_seconds,
        )
        return stage_carry.replace(state=apply_halo(stage_carry.state, halo_spec(namelist.grid)))

    # Static RK sequencing avoids per-stage scalar dispatch inside the profiled
    # timestep loop. WRF solve_em.F:1472-1479 runs one RK1 acoustic small step
    # and half the configured sound steps for RK2.
    # Legacy test anchor for the prior dynamic form:
    # lambda value: advance_stage(value, 1.0 / 3.0, 1)
    if debug:
        jax.debug.print("GPUWRF_M6B_RK1_ACOUSTIC_LOOP_ENTER substeps=1")
    dt = float(namelist.dt_s)
    configured_sound_steps = int(namelist.acoustic_substeps)
    stages = (
        _RKStageDescriptor(1, dt / 3.0, dt / 3.0, 1),
        _RKStageDescriptor(2, 0.5 * dt, dt / float(configured_sound_steps), max(1, configured_sound_steps // 2)),
        _RKStageDescriptor(3, dt, dt / float(configured_sound_steps), configured_sound_steps),
    )
    carry = carry.replace(state=origin)
    carry = advance_stage(carry, stages[0])
    carry = advance_stage(carry, stages[1])
    return advance_stage(carry, stages[2])


def _coupled_core_extras(state: State) -> dict[str, jax.Array]:
    return {
        "qv": state.qv,
        "qc": state.qc,
        "qr": state.qr,
        "qi": state.qi,
        "qs": state.qs,
        "qg": state.qg,
        "qke": state.qke,
        "t_skin": state.t_skin,
        "xland": state.xland,
        "lakemask": state.lakemask,
        "lu_index": state.lu_index,
        "u_bdy": state.u_bdy,
        "v_bdy": state.v_bdy,
        "theta_bdy": state.theta_bdy,
        "qv_bdy": state.qv_bdy,
        "ph_bdy": state.ph_bdy,
        "mu_bdy": state.mu_bdy,
    }


def _state_from_coupled_core(snapshot: dict[str, jax.Array], template: State, theta_offset: jax.Array, dt_s: float) -> State:
    theta = jnp.asarray(snapshot["theta"]) + theta_offset
    p_pert = jnp.asarray(snapshot["p"])
    ph_pert = jnp.asarray(snapshot["ph"])
    mu_pert = jnp.asarray(snapshot["mu"])
    p_total = template.p_total - template.p_perturbation + p_pert
    ph_total = template.ph_total - template.ph_perturbation + ph_pert
    mu_total = template.mu_total - template.mu_perturbation + mu_pert
    return template.replace(
        u=jnp.asarray(snapshot["u"]),
        v=jnp.asarray(snapshot["v"]),
        w=jnp.asarray(snapshot["w"]),
        theta=theta,
        qv=template.qv + jnp.asarray(snapshot["qv_phys_tend"]) * float(dt_s),
        qc=template.qc + jnp.asarray(snapshot["qc_phys_tend"]) * float(dt_s),
        qr=template.qr + jnp.asarray(snapshot["qr_phys_tend"]) * float(dt_s),
        qi=template.qi + jnp.asarray(snapshot["qi_phys_tend"]) * float(dt_s),
        qs=template.qs + jnp.asarray(snapshot["qs_phys_tend"]) * float(dt_s),
        qg=template.qg + jnp.asarray(snapshot["qg_phys_tend"]) * float(dt_s),
        qke=template.qke + jnp.asarray(snapshot["qke_phys_tend"]) * float(dt_s),
        p=p_total,
        p_total=p_total,
        p_perturbation=p_pert,
        ph=ph_total,
        ph_total=ph_total,
        ph_perturbation=ph_pert,
        mu=mu_total,
        mu_total=mu_total,
        mu_perturbation=mu_pert,
    )


def _carry_from_coupled_core(snapshot: dict[str, jax.Array], template: State, theta_offset: jax.Array, dt_s: float, *, rthraten: jax.Array | None = None) -> OperationalCarry:
    next_state = _state_from_coupled_core(snapshot, template, theta_offset, float(dt_s))
    return OperationalCarry(
        state=next_state,
        t_2ave=jnp.asarray(snapshot["t_2ave"]) + theta_offset,
        ww=jnp.asarray(snapshot["ww"]),
        mudf=jnp.asarray(snapshot["mudf"]),
        muave=jnp.asarray(snapshot["muave"]),
        muts=jnp.asarray(snapshot["muts"]),
        ph_tend=jnp.asarray(snapshot["ph_tend"]),
        u_save=next_state.u,
        v_save=next_state.v,
        w_save=next_state.w,
        t_save=next_state.theta,
        ph_save=next_state.ph,
        mu_save=jnp.asarray(snapshot["mu"]),
        ww_save=jnp.asarray(snapshot["ww"]),
        # Preserve the held radiative theta tendency across the coupled core
        # (it is refreshed in the physics chain, not the dycore core).
        rthraten=jnp.zeros_like(next_state.theta) if rthraten is None else rthraten,
    )


def _coupled_core_step(carry: OperationalCarry, namelist: OperationalNamelist, step_index) -> OperationalCarry:
    acoustic = _acoustic_core_state(carry, namelist)
    theta_offset = _theta_base_offset(carry.state.theta)
    snapshot = coupled_timestep_core(
        acoustic,
        namelist.metrics,
        CoupledCoreConfig(
            dt=float(namelist.dt_s),
            dx=float(namelist.grid.projection.dx_m),
            dy=float(namelist.grid.projection.dy_m),
            acoustic_substeps=int(namelist.acoustic_substeps),
            rk_order=int(namelist.rk_order),
            epssm=float(namelist.epssm),
            top_lid=bool(namelist.top_lid),
            physics_enabled=True,
            boundary_enabled=True,
            boundary_config=namelist.boundary_config,
        ),
        extras=_coupled_core_extras(carry.state),
        step_index=step_index,
    )
    return _carry_from_coupled_core(snapshot, carry.state, theta_offset, float(namelist.dt_s), rthraten=carry.rthraten)


def _physics_boundary_step_with_limiter_diagnostics(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    step_index,
    *,
    run_radiation: bool,
    debug: bool = False,
) -> tuple[OperationalCarry, dict[str, jax.Array]]:
    physical_origin = carry.state
    # Forecast clock for this step (traced scalar). Hoisted above the dycore so the
    # in-acoustic-loop NORMAL-momentum boundary targets are interpolated at the
    # step-start lead (matching WRF, which fixes ru_tend/rv_tend at the step start);
    # also reused below by rrtmg + the end-of-step lateral boundary nudge.
    lead_seconds = step_index.astype(jnp.float64) * float(namelist.dt_s)
    carry = _rk_scan_step(carry, namelist, debug=debug, lead_seconds=lead_seconds)
    next_state = carry.state
    limiter_diagnostics = _empty_theta_limiter_diagnostics(next_state.theta)
    if not bool(namelist.disable_guards):
        next_state, limiter_diagnostics = _limit_guarded_dynamics_state_with_diagnostics(next_state, physical_origin)
        next_state = next_state.replace(
            qv=_valid_mixing_ratio(next_state.qv, physical_origin.qv),
            qc=_valid_mixing_ratio(next_state.qc, physical_origin.qc),
            qr=_valid_mixing_ratio(next_state.qr, physical_origin.qr),
            qi=_valid_mixing_ratio(next_state.qi, physical_origin.qi),
            qs=_valid_mixing_ratio(next_state.qs, physical_origin.qs),
            qg=_valid_mixing_ratio(next_state.qg, physical_origin.qg),
        )
    if bool(namelist.run_physics):
        # Gate-1 physics call order: thompson -> surface -> mynn -> rrtmg(cadence).
        # Thompson microphysics is gated ONLY by run_physics, NOT by disable_guards.
        # (Coupling fix 2026-05-30: previously Thompson was wired behind
        # `if not disable_guards`, which silently dropped the validated B1
        # microphysics whenever guards were turned off -- making the operational
        # safety net load-bearing for moisture physics, in violation of the
        # guards-must-not-be-load-bearing rule. surface/mynn/rrtmg always ran;
        # only Thompson was mistakenly tied to the guard flag.)
        next_state = thompson_adapter(next_state, float(namelist.dt_s))
        next_state = surface_adapter(next_state, float(namelist.dt_s))
        next_state = mynn_adapter(next_state, float(namelist.dt_s), namelist.grid)
        # B3 radiation cadence -- WRF-faithful HELD-RATE (Sprint coupler-fp64 FIX #2,
        # GPT P0-2). WRF recomputes the radiative theta tendency RTHRATEN (K/s) only
        # once per radt interval (module_radiation_driver.F run_param gate) and then
        # ADDS dt*RTHRATEN into theta at EVERY dynamics step over that interval
        # (phy_ra_ten, module_physics_addtendc.F). The previous code lumped the whole
        # interval (dt*cadence*rate) at one step, so the intervening dynamics/micro/PBL
        # saw a wrong temperature trajectory. Here the held rate lives in the carry
        # (resident on device, no host transfer): refreshed at the cadence step,
        # applied every step.
        def _refresh_rthraten(_unused) -> jnp.ndarray:
            return rrtmg_theta_tendency(
                next_state,
                namelist.grid,
                time_utc=namelist.time_utc,
                lead_seconds=lead_seconds,
            )

        if isinstance(run_radiation, bool):
            # STATIC gate (production segmented path): the rrtmg recompute is either
            # traced or absent -- the radiation branch is fully resolved at trace
            # time, so each radiation interval is its own compiled scan.
            held_rthraten = _refresh_rthraten(None) if run_radiation else carry.rthraten
        else:
            # TRACED gate (single-scan path): run_radiation is a traced bool
            # (step_index %% cadence == 0). jax.lax.cond keeps the WHOLE forecast in
            # ONE scan (one compile regardless of length) while still recomputing
            # RRTMG only at the cadence; the held rate is reused on the other steps.
            held_rthraten = jax.lax.cond(
                run_radiation, _refresh_rthraten, lambda _u: carry.rthraten, None
            )
        # Apply the HELD radiative theta tendency every dynamics step (theta += dt*rate),
        # matching WRF's phy_ra_ten. fp64-safe: held_rthraten and theta share dtype.
        next_state = next_state.replace(
            theta=next_state.theta + float(namelist.dt_s) * held_rthraten
        )
        carry = carry.replace(rthraten=held_rthraten)
    if bool(namelist.run_boundary):
        bounded = apply_lateral_boundaries(
            next_state, lead_seconds, float(namelist.dt_s), namelist.boundary_config, namelist.metrics
        )
        if bool(namelist.disable_guards):
            next_state = bounded
        else:
            next_state = bounded.replace(
                u=_finite_or_origin(bounded.u, physical_origin.u),
                v=_finite_or_origin(bounded.v, physical_origin.v),
                w=_finite_or_origin(bounded.w, physical_origin.w),
                theta=_finite_or_origin(bounded.theta, physical_origin.theta),
                qv=_valid_mixing_ratio(bounded.qv, physical_origin.qv),
                p=_finite_or_origin(bounded.p, physical_origin.p),
                ph=_finite_or_origin(bounded.ph, physical_origin.ph),
                p_total=_finite_or_origin(bounded.p_total, physical_origin.p_total),
                ph_total=_finite_or_origin(bounded.ph_total, physical_origin.ph_total),
                p_perturbation=_finite_or_origin(bounded.p_perturbation, physical_origin.p_perturbation),
                ph_perturbation=_finite_or_origin(bounded.ph_perturbation, physical_origin.ph_perturbation),
            )
            next_state = _limit_guarded_mass_state(next_state, physical_origin)
    next_state = _enforce_operational_precision(next_state, force_fp64=bool(namelist.force_fp64))
    return carry.replace(state=next_state), limiter_diagnostics


def _physics_boundary_step(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    step_index,
    *,
    run_radiation: bool,
    debug: bool = False,
) -> OperationalCarry:
    next_carry, _diagnostics = _physics_boundary_step_with_limiter_diagnostics(
        carry,
        namelist,
        step_index,
        run_radiation=run_radiation,
        debug=debug,
    )
    return next_carry


def _scan_forecast_segment(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    *,
    start_step: int,
    steps: int,
    run_radiation: bool,
    debug: bool = False,
) -> OperationalCarry:
    indices = jnp.arange(start_step, start_step + steps, dtype=jnp.int32)

    def body(scan_carry: OperationalCarry, step_index):
        return _physics_boundary_step(scan_carry, namelist, step_index, run_radiation=run_radiation, debug=debug), None

    next_carry, _ = jax.lax.scan(body, carry, indices)
    return next_carry


def _scan_forecast_segment_with_limiter_diagnostics(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    *,
    start_step: int,
    steps: int,
    run_radiation: bool,
    debug: bool = False,
) -> tuple[OperationalCarry, dict[str, jax.Array]]:
    indices = jnp.arange(start_step, start_step + steps, dtype=jnp.int32)

    def body(scan_carry: OperationalCarry, step_index):
        next_carry, diagnostics = _physics_boundary_step_with_limiter_diagnostics(
            scan_carry,
            namelist,
            step_index,
            run_radiation=run_radiation,
            debug=debug,
        )
        diagnostics = dict(diagnostics)
        diagnostics["step_index"] = step_index
        return next_carry, diagnostics

    next_carry, diagnostics = jax.lax.scan(body, carry, indices)
    return next_carry, diagnostics


def _concat_theta_limiter_diagnostics(chunks: list[dict[str, jax.Array]]) -> dict[str, jax.Array]:
    return {key: jnp.concatenate([chunk[key] for chunk in chunks], axis=0) for key in chunks[0]}


# --------------------------------------------------------------------------
# M9 operational diagnostics carry (coupler_interface.md §4, §6 item 1)
# --------------------------------------------------------------------------


class M9Diagnostics(NamedTuple):
    """The M9 operational divergence-map surface fields, all mass-point (ny,nx).

    Side-channel only -- recomputed from the post-step State at OUTPUT cadence,
    not prognostic leaves. SWDOWN/GLW W m^-2; HFX/LH W m^-2 (upward +); PBLH m;
    TSK/T2 K; U10/V10 m s^-1; PSFC Pa. ``swdown``/``glw`` follow the forecast
    clock (namelist.time_utc + lead_seconds) so the diurnal cycle is captured.
    """

    swdown: jax.Array
    glw: jax.Array
    hfx: jax.Array
    lh: jax.Array
    pblh: jax.Array
    tsk: jax.Array
    t2: jax.Array
    u10: jax.Array
    v10: jax.Array
    psfc: jax.Array


def _psfc_from_state(state: State) -> jax.Array:
    """Surface pressure (Pa) = column-bottom total pressure (mass point, ny,nx).

    coupler_interface.md §4 sources PSFC from mu_total+pb at the column bottom;
    ``state.p`` already carries the total pressure (pb + p'), so its bottom level
    is the diagnosed surface pressure for the M9 map.
    """
    return state.p[0, :, :]


def compute_m9_diagnostics(
    state: State,
    namelist: OperationalNamelist,
    lead_seconds,
) -> M9Diagnostics:
    """Recompute the M9 surface map from a post-step State (side-channel only)."""
    surf = surface_layer_diagnostics(state, namelist.grid)
    rad = rrtmg_radiation_diagnostics(
        state, namelist.grid, time_utc=namelist.time_utc, lead_seconds=lead_seconds
    )
    return M9Diagnostics(
        swdown=rad.swdown,
        glw=rad.glw,
        hfx=surf.hfx,
        lh=surf.lh,
        pblh=surf.pblh,
        tsk=state.t_skin,
        t2=surf.t2,
        u10=surf.u10,
        v10=surf.v10,
        psfc=_psfc_from_state(state),
    )


@partial(jax.jit, static_argnames=("n_steps", "cadence"))
def _advance_chunk(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    start_step,
    *,
    n_steps: int,
    cadence: int,
) -> OperationalCarry:
    """Advance one output interval as a SINGLE compiled scan (no diagnostics).

    Radiation is gated by the traced ``step_index %% cadence == 0`` predicate via
    ``_physics_boundary_step``'s cond path, so this is byte-identical to the
    production per-step cadence.  ``start_step`` is TRACED (only ``n_steps``/
    ``cadence`` static) so equal-length intervals reuse the SAME compiled executable
    -- one compile for the whole forecast, not one per interval.  Kept SEPARATE from
    the diagnostics call so the dynamics scratch is freed before the large RRTMG
    diagnostic transient is allocated (peak-memory bound, Task 2 OOM fix).
    """
    run_physics = bool(namelist.run_physics)
    start_step = jnp.asarray(start_step, dtype=jnp.int32)
    indices = start_step + jnp.arange(int(n_steps), dtype=jnp.int32)

    def body(scan_carry: OperationalCarry, step_index):
        if run_physics:
            run_radiation = jnp.equal(jnp.mod(step_index, int(cadence)), 0)
        else:
            run_radiation = False
        next_carry = _physics_boundary_step(
            scan_carry, namelist, step_index, run_radiation=run_radiation, debug=False
        )
        return next_carry, None

    carry, _ = jax.lax.scan(body, carry, indices)
    return carry


@jax.jit
def _m9_snapshot(carry: OperationalCarry, namelist: OperationalNamelist, lead_seconds) -> M9Diagnostics:
    """Compute the M9 surface map once from a post-chunk State (separate program).

    Isolated in its own ``jax.jit`` so XLA cannot co-schedule the ~15 GiB RRTMG
    g-point diagnostic transient with the dynamics-chunk scratch; the host loop
    blocks after the chunk so the chunk scratch is freed first.
    """
    return compute_m9_diagnostics(carry.state, namelist, lead_seconds)


def run_forecast_operational_with_m9_diagnostics(
    state: State,
    namelist: OperationalNamelist,
    hours: float,
    *,
    output_cadence_steps: int = 60,
) -> tuple[State, M9Diagnostics]:
    """Run the operational forecast and emit the M9 surface map at output cadence.

    Materializes the M9 surface diagnostics ONLY at the OUTPUT cadence (never every
    step) and bounds peak memory to (forecast working set + ONE RRTMG diagnostic
    transient), independent of forecast length.

    OOM FIX (Sprint perf-diag Task 2).  Two compounding problems killed the previous
    implementation at 1080 steps (+3h, >20 GB OOM):

    1. ``compute_m9_diagnostics`` was called inside the per-step scan body and
       ``jax.lax.scan`` stacked ``(diag, emit)`` for EVERY step -- 1080 copies of all
       10 surface maps plus every step's diagnostic intermediates kept live.
    2. ``compute_m9_diagnostics`` re-runs the FULL RRTMG SW+LW column solver, whose
       g-point intermediate is ~15 GiB on this d02 grid.  Even computing it a few
       times inside ONE jit lets XLA overlap those transients (measured: a single
       jit over the 3h forecast tried to allocate 27.8 GiB).

    Fix: a HOST-driven loop walks one output interval at a time, calling the jit'd
    ``_advance_chunk_and_snapshot`` (each chunk is ONE compiled scan, reused across
    intervals) and ``block_until_ready``-ing between chunks so each RRTMG transient is
    freed before the next chunk allocates its own.  The host loop runs only
    ``steps // out_cad`` iterations (e.g. 3 for +3h hourly) -- NOT per step -- so there
    is no per-timestep host/device transfer.  The dynamics are byte-identical to the
    production scan (same per-step body, same traced radiation schedule); only the
    emitted-snapshot set differs.  ``run_forecast_operational`` is untouched.
    """
    if int(namelist.rk_order) != 3:
        raise ValueError("operational mode currently supports RK3 only")
    if int(output_cadence_steps) <= 0:
        raise ValueError("output_cadence_steps must be positive")
    cadence = int(namelist.radiation_cadence_steps)
    if cadence <= 0:
        raise ValueError("radiation_cadence_steps must be positive")

    carry = initial_operational_carry(
        _enforce_operational_precision(state, force_fp64=bool(namelist.force_fp64))
    )
    steps = _steps_for_hours(hours, float(namelist.dt_s))
    out_cad = int(output_cadence_steps)

    # Output-interval boundaries: every multiple of out_cad up to steps, plus a final
    # partial interval if steps is not a multiple of out_cad (so the final state is
    # always emitted).
    boundaries: list[int] = list(range(out_cad, steps + 1, out_cad))
    if not boundaries or boundaries[-1] != steps:
        boundaries.append(steps)

    dt_s = float(namelist.dt_s)
    diag_chunks: list[M9Diagnostics] = []
    start = 1
    for end in boundaries:
        n = end - start + 1
        carry = _advance_chunk(
            carry, namelist, jnp.asarray(start, dtype=jnp.int32), n_steps=n, cadence=cadence
        )
        # Free the dynamics-chunk scratch BEFORE the RRTMG diagnostic transient is
        # allocated, then free the transient before the next chunk -- this is what
        # bounds peak memory to (working set + ONE transient) for any forecast length.
        jax.block_until_ready(carry.state.theta)
        diag = _m9_snapshot(carry, namelist, jnp.asarray(float(end) * dt_s, dtype=jnp.float64))
        jax.block_until_ready(diag.t2)
        diag_chunks.append(
            M9Diagnostics(*(getattr(diag, name)[None, ...] for name in M9Diagnostics._fields))
        )
        start = end + 1

    all_diags = M9Diagnostics(
        *(jnp.concatenate([getattr(chunk, name) for chunk in diag_chunks], axis=0)
          for name in M9Diagnostics._fields)
    )
    return carry.state, all_diags


@partial(jax.jit, static_argnames=("hours",), donate_argnums=(0,))
def run_forecast_operational(state: State, namelist: OperationalNamelist, hours: float) -> State:
    """Run an operational forecast as one compiled, device-resident scan.

    No diagnostics, host-read callbacks, host array pulls, or sanitizers are
    present in this path. ``hours`` is static so the timestep count is fixed at
    compile time and the whole forecast lowers as one JAX program.
    """

    if int(namelist.rk_order) != 3:
        raise ValueError("operational mode currently supports RK3 only")
    # Honour namelist.force_fp64 at the PUBLIC entry: the in-scan enforcement
    # (line ~1471) upcasts each step's output to fp64 when force_fp64, so the
    # INITIAL carry must also be fp64 or jax.lax.scan rejects the carry dtype
    # mismatch -- and the production path would otherwise start fp32 (GPT
    # re-confirm: proofs that pre-upcast manually did not exercise this entry).
    initial = initial_operational_carry(
        _enforce_operational_precision(state, force_fp64=bool(namelist.force_fp64))
    )
    steps = _steps_for_hours(hours, float(namelist.dt_s))
    cadence = int(namelist.radiation_cadence_steps)
    if cadence <= 0:
        raise ValueError("radiation_cadence_steps must be positive")

    carry = initial
    step = 1
    while step <= steps:
        next_radiation = ((step + cadence - 1) // cadence) * cadence
        if bool(namelist.run_physics) and next_radiation <= steps:
            non_radiation_steps = next_radiation - step
            if non_radiation_steps:
                carry = _scan_forecast_segment(
                    carry,
                    namelist,
                    start_step=step,
                    steps=non_radiation_steps,
                    run_radiation=False,
                    debug=False,
                )
            carry = _scan_forecast_segment(
                carry,
                namelist,
                start_step=next_radiation,
                steps=1,
                run_radiation=True,
                debug=False,
            )
            step = next_radiation + 1
        else:
            carry = _scan_forecast_segment(
                carry,
                namelist,
                start_step=step,
                steps=steps - step + 1,
                run_radiation=False,
                debug=False,
            )
            step = steps + 1
    return carry.state


def run_forecast_operational_segmented(
    state: State,
    namelist: OperationalNamelist,
    hours: float,
    *,
    segment_steps: int | None = None,
) -> State:
    """Run a long operational forecast as a HOST loop over ONE compiled segment.

    Long-run (24-72h) compile-blowup remedy that keeps compile O(segment) and peak
    GPU memory bounded, independent of forecast length.

    Why this exists.  ``run_forecast_operational`` is a Python while-loop that emits
    one ``jax.lax.scan`` per radiation interval, so the number of distinct XLA scan
    subcomputations -- and thus COMPILE time / peak memory -- grows with the forecast
    length (measured: +12h did not compile in 37 min).  This entry instead compiles a
    SINGLE fixed-length inner segment (``_advance_chunk`` with a static ``n_steps``)
    and drives it from a host ``for`` loop, carrying ``State`` across segments and
    ``block_until_ready``-ing between them so each segment's dynamics scratch is freed
    before the next segment allocates.  Compile happens ONCE for the full-length
    segment (every equal-length segment reuses the same executable via the traced
    ``start_step``); a single shorter compile covers a final partial tail segment.

    Equivalence.  Global step indices run ``1..steps`` exactly as in
    ``run_forecast_operational_single_scan`` and ``run_forecast_operational``; the
    in-segment radiation gate is the SAME traced ``step_index %% cadence == 0``
    predicate.  Because the segments are contiguous in the global step index, RRTMG
    fires on exactly the same global steps as the single scan, so the result is
    BITWISE identical to the single scan and round-off identical to the validated
    segmented while-loop (proof: proofs/perf/segscan_equiv.json -- seg-vs-single max
    abs diff == 0 on every field at 0.2h and 0.6h incl. the radiation step; seg-vs-
    production differs only at FP round-off from cond-vs-direct RRTMG application).

    ``segment_steps`` defaults to one radiation cadence interval so radiation fires
    exactly once at each full segment's last step; any positive value is accepted
    (the radiation schedule is unaffected by where the segment boundaries fall).
    """

    if int(namelist.rk_order) != 3:
        raise ValueError("operational mode currently supports RK3 only")
    cadence = int(namelist.radiation_cadence_steps)
    if cadence <= 0:
        raise ValueError("radiation_cadence_steps must be positive")
    seg = int(segment_steps) if segment_steps is not None else cadence
    if seg <= 0:
        raise ValueError("segment_steps must be positive")

    carry = initial_operational_carry(
        _enforce_operational_precision(state, force_fp64=bool(namelist.force_fp64))
    )
    steps = _steps_for_hours(hours, float(namelist.dt_s))

    # Host loop over contiguous fixed-length segments covering global steps 1..steps.
    # Every full segment has identical static ``n_steps`` so it reuses ONE compiled
    # executable (``start_step`` is traced); a final partial segment compiles once.
    start = 1
    while start <= steps:
        n = min(seg, steps - start + 1)
        carry = _advance_chunk(
            carry, namelist, jnp.asarray(start, dtype=jnp.int32), n_steps=int(n), cadence=cadence
        )
        # Block so this segment's device scratch is freed before the next segment's
        # buffers are allocated -- this is what bounds peak memory to one segment's
        # working set regardless of forecast length.
        jax.block_until_ready(carry.state.theta)
        start += n
    return carry.state


@partial(jax.jit, static_argnames=("hours",), donate_argnums=(0,))
def run_forecast_operational_single_scan(state: State, namelist: OperationalNamelist, hours: float) -> State:
    """Whole forecast as ONE jax.lax.scan -- compile-blowup remedy for 24-72h.

    The production ``run_forecast_operational`` Python while-loop emits one
    ``jax.lax.scan`` per radiation interval (a non-radiation scan plus an isolated
    1-step radiation scan), so the number of distinct XLA scan subcomputations -- and
    thus the COMPILE time -- scales with the forecast length: ~4 scans at 1h, 12 at
    3h, 96 at 24h, 288 at 72h.  Measured: the cold compile of the 3h (12-scan)
    program exceeds ~30 min (proofs/perf -- the +3h's ~32 min was almost entirely
    this cold compile; warmed steady-state is ~45 ms/step).  At 24-72h the segmented
    compile is a hard wall.

    This entry collapses the whole forecast into a SINGLE scan whose trip count is
    the static step total, and gates RRTMG with ``jax.lax.cond`` on the traced
    predicate ``(step_index %% cadence == 0)``.  Compile cost is then independent of
    forecast length (one scan body), while the per-step cadence and the RRTMG firing
    schedule are numerically IDENTICAL to the segmented path (cond fires RRTMG on
    exactly the same steps).  Warmed throughput is unchanged.  This is the
    recommended path for long-lead / ensemble runs; the segmented production path is
    left untouched and remains the validated default until this entry passes its own
    short-horizon equivalence gate (proofs/perf/single_scan_equiv.json).
    """

    if int(namelist.rk_order) != 3:
        raise ValueError("operational mode currently supports RK3 only")
    initial = initial_operational_carry(
        _enforce_operational_precision(state, force_fp64=bool(namelist.force_fp64))
    )
    steps = _steps_for_hours(hours, float(namelist.dt_s))
    cadence = int(namelist.radiation_cadence_steps)
    if cadence <= 0:
        raise ValueError("radiation_cadence_steps must be positive")
    run_physics = bool(namelist.run_physics)

    indices = jnp.arange(1, steps + 1, dtype=jnp.int32)

    def body(scan_carry: OperationalCarry, step_index):
        if run_physics:
            run_radiation = jnp.equal(jnp.mod(step_index, cadence), 0)
        else:
            run_radiation = False  # static: no radiation branch traced at all
        next_carry = _physics_boundary_step(
            scan_carry, namelist, step_index, run_radiation=run_radiation, debug=False
        )
        return next_carry, None

    carry, _ = jax.lax.scan(body, initial, indices)
    return carry.state


@partial(jax.jit, static_argnames=("hours",), donate_argnums=(0,))
def run_forecast_operational_with_limiter_diagnostics(
    state: State,
    namelist: OperationalNamelist,
    hours: float,
) -> tuple[State, dict[str, jax.Array]]:
    """Run an operational forecast and return INV-10 theta limiter diagnostics."""

    if int(namelist.rk_order) != 3:
        raise ValueError("operational mode currently supports RK3 only")
    # Honour namelist.force_fp64 at the PUBLIC entry: the in-scan enforcement
    # (line ~1471) upcasts each step's output to fp64 when force_fp64, so the
    # INITIAL carry must also be fp64 or jax.lax.scan rejects the carry dtype
    # mismatch -- and the production path would otherwise start fp32 (GPT
    # re-confirm: proofs that pre-upcast manually did not exercise this entry).
    initial = initial_operational_carry(
        _enforce_operational_precision(state, force_fp64=bool(namelist.force_fp64))
    )
    steps = _steps_for_hours(hours, float(namelist.dt_s))
    cadence = int(namelist.radiation_cadence_steps)
    if cadence <= 0:
        raise ValueError("radiation_cadence_steps must be positive")

    carry = initial
    step = 1
    diagnostic_chunks: list[dict[str, jax.Array]] = []
    while step <= steps:
        next_radiation = ((step + cadence - 1) // cadence) * cadence
        if bool(namelist.run_physics) and next_radiation <= steps:
            non_radiation_steps = next_radiation - step
            if non_radiation_steps:
                carry, diagnostics = _scan_forecast_segment_with_limiter_diagnostics(
                    carry,
                    namelist,
                    start_step=step,
                    steps=non_radiation_steps,
                    run_radiation=False,
                    debug=False,
                )
                diagnostic_chunks.append(diagnostics)
            carry, diagnostics = _scan_forecast_segment_with_limiter_diagnostics(
                carry,
                namelist,
                start_step=next_radiation,
                steps=1,
                run_radiation=True,
                debug=False,
            )
            diagnostic_chunks.append(diagnostics)
            step = next_radiation + 1
        else:
            carry, diagnostics = _scan_forecast_segment_with_limiter_diagnostics(
                carry,
                namelist,
                start_step=step,
                steps=steps - step + 1,
                run_radiation=False,
                debug=False,
            )
            diagnostic_chunks.append(diagnostics)
            step = steps + 1
    return carry.state, _concat_theta_limiter_diagnostics(diagnostic_chunks)


@partial(jax.jit, static_argnames=("hours", "debug"), donate_argnums=(0,))
def run_forecast_operational_debug(state: State, namelist: OperationalNamelist, hours: float, *, debug: bool = False) -> State:
    """Diagnostic operational forecast entry point with static debug markers."""

    if int(namelist.rk_order) != 3:
        raise ValueError("operational mode currently supports RK3 only")
    # Honour namelist.force_fp64 at the PUBLIC entry: the in-scan enforcement
    # (line ~1471) upcasts each step's output to fp64 when force_fp64, so the
    # INITIAL carry must also be fp64 or jax.lax.scan rejects the carry dtype
    # mismatch -- and the production path would otherwise start fp32 (GPT
    # re-confirm: proofs that pre-upcast manually did not exercise this entry).
    initial = initial_operational_carry(
        _enforce_operational_precision(state, force_fp64=bool(namelist.force_fp64))
    )
    steps = _steps_for_hours(hours, float(namelist.dt_s))
    cadence = int(namelist.radiation_cadence_steps)
    if cadence <= 0:
        raise ValueError("radiation_cadence_steps must be positive")

    carry = initial
    step = 1
    while step <= steps:
        next_radiation = ((step + cadence - 1) // cadence) * cadence
        if bool(namelist.run_physics) and next_radiation <= steps:
            non_radiation_steps = next_radiation - step
            if non_radiation_steps:
                carry = _scan_forecast_segment(
                    carry,
                    namelist,
                    start_step=step,
                    steps=non_radiation_steps,
                    run_radiation=False,
                    debug=debug,
                )
            carry = _scan_forecast_segment(
                carry,
                namelist,
                start_step=next_radiation,
                steps=1,
                run_radiation=True,
                debug=debug,
            )
            step = next_radiation + 1
        else:
            carry = _scan_forecast_segment(
                carry,
                namelist,
                start_step=step,
                steps=steps - step + 1,
                run_radiation=False,
                debug=debug,
            )
            step = steps + 1
    return carry.state


__all__ = [
    "OperationalNamelist",
    "M9Diagnostics",
    "compute_m9_diagnostics",
    "run_forecast_operational",
    "run_forecast_operational_segmented",
    "run_forecast_operational_single_scan",
    "run_forecast_operational_debug",
    "run_forecast_operational_with_limiter_diagnostics",
    "run_forecast_operational_with_m9_diagnostics",
]
