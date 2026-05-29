"""GPU-resident operational forecast loop for M6 perf-design.

This module is deliberately separate from the M6B validation savepoint ladder.
It runs timestep/RK/acoustic loops inside one JAX entry point and leaves debug
snapshots/sanitizers out of the compiled path.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
from jax import config
import jax.numpy as jnp

from gpuwrf.contracts.grid import DycoreMetrics, GridSpec
from gpuwrf.contracts.state import State, Tendencies
from gpuwrf.contracts.precision import DEFAULT_DTYPES, STATE_FIELD_ORDER
from gpuwrf.contracts.halo import apply_halo
from gpuwrf.coupling.boundary_apply import BoundaryConfig, DEFAULT_BOUNDARY_CONFIG, apply_lateral_boundaries
from gpuwrf.coupling.physics_couplers import mynn_adapter, rrtmg_adapter, surface_adapter, thompson_adapter
from gpuwrf.dynamics.advection import compute_advection_tendencies, halo_spec
from gpuwrf.dynamics.acoustic_wrf import (
    calc_coef_w_wrf_coefficients,
    diagnose_pressure_al_alt,
    horizontal_pressure_gradient,
    moisture_coupling_factors,
)
from gpuwrf.dynamics.core.acoustic import AcousticCoreConfig, AcousticCoreState, acoustic_substep_core
from gpuwrf.dynamics.core.advance_w import GRAVITY_M_S2, dry_cqw
from gpuwrf.dynamics.core.calc_p_rho import CalcPRhoStep0, calc_p_rho_wrf
from gpuwrf.dynamics.core.coupled import CoupledCoreConfig, coupled_timestep_core
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
            radiation_cadence_steps=radiation_cadence_steps,
            boundary_config=boundary_config,
            use_vertical_solver=use_vertical_solver,
            disable_guards=disable_guards,
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


def _enforce_operational_precision(state: State) -> State:
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
    """Apply the dycore theta limiter and dry-mass guard after one RK3 step."""

    mass = _theta_mass_weights(candidate.theta, candidate.mu_total)
    lower, upper = _theta_level_monotonic_bounds(origin.theta)
    theta, diagnostics = _positive_definite_theta_increment_limiter(
        candidate.theta,
        origin.theta,
        mass,
        lower_bound=lower,
        upper_bound=upper,
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
        coef_mut=carry.muts,
    )


def _acoustic_core_state_from_prep(
    carry: OperationalCarry,
    prep: SmallStepPrepState,
    pressure: CalcPRhoStep0,
    namelist: OperationalNamelist,
    tendencies: Tendencies,
) -> AcousticCoreState:
    """Build the acoustic work-state directly from WRF ``small_step_prep``."""

    state = prep.entry_state
    theta_pert = (state.theta - prep.theta_offset).astype(jnp.float64)
    ph_base = state.ph_total - state.ph_perturbation
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
        muave=carry.muave,
        muts=prep.muts,
        muu=prep.muu,
        muv=prep.muv,
        mudf=carry.mudf,
        theta=theta_pert,
        theta_1=prep.t_save,
        theta_ave=carry.t_2ave - prep.theta_offset,
        theta_tend=namelist.tendencies.theta,
        mu_tend=namelist.tendencies.mu,
        ph_tend=carry.ph_tend,
        ph=prep.ph_work,
        p=pressure.p,
        t_2ave=carry.t_2ave - prep.theta_offset,
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
    )


def _carry_from_finished_stage(carry: OperationalCarry, prep: SmallStepPrepState, acoustic: AcousticCoreState) -> OperationalCarry:
    next_state = small_step_finish_wrf(prep, acoustic)
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
        acoustic.coef_mut if acoustic.coef_mut is not None else acoustic.muts,
        namelist.metrics,
        dt=float(dt_sub),
        epssm=float(namelist.epssm),
        top_lid=bool(namelist.top_lid),
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
        ),
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
) -> OperationalCarry:
    acoustic = _acoustic_core_state_from_prep(carry, prep, pressure, namelist, tendencies)
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

        def body(scan_acoustic: AcousticCoreState, _):
            return acoustic_substep_core(
                scan_acoustic,
                a=a,
                alpha=alpha,
                gamma=gamma,
                cfg=AcousticCoreConfig(
                    dt=float(stage.dts_rk),
                    dx=float(namelist.grid.projection.dx_m),
                    dy=float(namelist.grid.projection.dy_m),
                    epssm=float(namelist.epssm),
                    top_lid=bool(namelist.top_lid),
                ),
                cqw=cqw_field,
            ), None

        acoustic, _ = jax.lax.scan(body, acoustic, xs=None, length=int(stage.number_of_small_timesteps))
        next_carry = _carry_from_finished_stage(carry, prep, acoustic)
        return next_carry.replace(state=apply_halo(next_carry.state, halo_spec(namelist.grid)))

    del tendencies
    return _with_save_family(carry, carry.state)


def _rk_scan_step(carry: OperationalCarry, namelist: OperationalNamelist, *, debug: bool = False) -> OperationalCarry:
    origin = apply_halo(carry.state, halo_spec(namelist.grid))
    rk1_reference = origin

    def advance_stage(stage_carry: OperationalCarry, stage: _RKStageDescriptor) -> OperationalCarry:
        haloed = apply_halo(stage_carry.state, halo_spec(namelist.grid))
        # WRF rk_tendency builds the large-step DYNAMIC tendencies (advection,
        # curvature, diffusion).  The horizontal pressure-gradient force is NOT
        # part of the large-step tendency -- it is applied inside the acoustic
        # small-step advance_uv (module_small_step_em.F:802-942).  Adding the
        # PGF here as well double-counts the gradient and seeds an acoustic
        # imbalance, so F7.A feeds advance_uv the dynamic (advection) tendency
        # only.  (Flux-form advection itself is Sprint B; for the periodic dry
        # gate it is inert.)
        tendencies = compute_advection_tendencies(haloed, namelist.tendencies, namelist.grid)
        candidate = add_scaled_tendencies(origin, tendencies, float(stage.dt_rk))
        candidate = apply_halo(candidate, halo_spec(namelist.grid))
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


def _carry_from_coupled_core(snapshot: dict[str, jax.Array], template: State, theta_offset: jax.Array, dt_s: float) -> OperationalCarry:
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
    return _carry_from_coupled_core(snapshot, carry.state, theta_offset, float(namelist.dt_s))


def _physics_boundary_step_with_limiter_diagnostics(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    step_index,
    *,
    run_radiation: bool,
    debug: bool = False,
) -> tuple[OperationalCarry, dict[str, jax.Array]]:
    physical_origin = carry.state
    carry = _rk_scan_step(carry, namelist, debug=debug)
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
        if not bool(namelist.disable_guards):
            next_state = thompson_adapter(next_state, float(namelist.dt_s))
        next_state = surface_adapter(next_state, float(namelist.dt_s))
        next_state = mynn_adapter(next_state, float(namelist.dt_s), namelist.grid)
        if run_radiation:
            next_state = rrtmg_adapter(next_state, float(namelist.dt_s), namelist.grid)
    if bool(namelist.run_boundary):
        lead_seconds = step_index.astype(jnp.float64) * float(namelist.dt_s)
        bounded = apply_lateral_boundaries(next_state, lead_seconds, float(namelist.dt_s), namelist.boundary_config)
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
    next_state = _enforce_operational_precision(next_state)
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


@partial(jax.jit, static_argnames=("hours",), donate_argnums=(0,))
def run_forecast_operational(state: State, namelist: OperationalNamelist, hours: float) -> State:
    """Run an operational forecast as one compiled, device-resident scan.

    No diagnostics, host-read callbacks, host array pulls, or sanitizers are
    present in this path. ``hours`` is static so the timestep count is fixed at
    compile time and the whole forecast lowers as one JAX program.
    """

    if int(namelist.rk_order) != 3:
        raise ValueError("operational mode currently supports RK3 only")
    initial = initial_operational_carry(_enforce_operational_precision(state))
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


@partial(jax.jit, static_argnames=("hours",), donate_argnums=(0,))
def run_forecast_operational_with_limiter_diagnostics(
    state: State,
    namelist: OperationalNamelist,
    hours: float,
) -> tuple[State, dict[str, jax.Array]]:
    """Run an operational forecast and return INV-10 theta limiter diagnostics."""

    if int(namelist.rk_order) != 3:
        raise ValueError("operational mode currently supports RK3 only")
    initial = initial_operational_carry(_enforce_operational_precision(state))
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
    initial = initial_operational_carry(_enforce_operational_precision(state))
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
    "run_forecast_operational",
    "run_forecast_operational_debug",
    "run_forecast_operational_with_limiter_diagnostics",
]
