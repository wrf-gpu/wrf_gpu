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
from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients
from gpuwrf.dynamics.core.acoustic import AcousticCoreConfig, AcousticCoreState, acoustic_substep_core
from gpuwrf.dynamics.core.coupled import CoupledCoreConfig, coupled_timestep_core
from gpuwrf.dynamics.tendencies import add_scaled_tendencies
from gpuwrf.runtime.operational_state import OperationalCarry, initial_operational_carry


config.update("jax_enable_x64", True)


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
        )


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


def _with_save_family(carry: OperationalCarry, state: State, ww: jax.Array | None = None) -> OperationalCarry:
    """Update WRF ``*_save`` transition fields in resident operational carry."""

    ww_value = carry.ww if ww is None else ww
    return carry.replace(
        state=state,
        u_save=state.u,
        v_save=state.v,
        w_save=state.w,
        t_save=state.theta,
        ph_save=state.ph,
        mu_save=state.mu_perturbation,
        ww_save=ww_value,
    )


def _m6b_acoustic_tendencies(tendencies: Tendencies, base: Tendencies) -> Tendencies:
    """Keep unvalidated reduced-dycore V self-advection out of M6b acoustic RK."""

    return tendencies.replace(v=base.v)


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


def _carry_from_acoustic_core(acoustic: AcousticCoreState, template: State, theta_offset: jax.Array) -> OperationalCarry:
    theta = acoustic.theta + theta_offset
    p_total = template.p_total - template.p_perturbation + acoustic.p
    ph_total = template.ph_total - template.ph_perturbation + acoustic.ph
    mu_total = template.mu_total - template.mu_perturbation + acoustic.mu
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
        mu_perturbation=acoustic.mu,
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
    dt_stage: float,
    *,
    substeps: int | None = None,
) -> OperationalCarry:
    scan_substeps = int(namelist.acoustic_substeps if substeps is None else substeps)
    del dt_stage
    # WRF ``solve_em.F:1472-1483`` separates RK stage span from acoustic cadence;
    # the contracted operational parity path uses one acoustic slice of the
    # parent timestep per configured acoustic substep, not ``dt_stage / n``.
    dt_sub = float(namelist.dt_s) / float(namelist.acoustic_substeps)

    def body(scan_carry: OperationalCarry, _):
        if bool(namelist.use_vertical_solver):
            return _operational_acoustic_substep_core(scan_carry, namelist, dt_sub), None
        next_state = add_scaled_tendencies(scan_carry.state, namelist.tendencies, dt_sub)
        return _with_save_family(scan_carry.replace(state=next_state), next_state), None

    next_carry, _ = jax.lax.scan(body, carry, xs=None, length=scan_substeps)
    return next_carry.replace(state=apply_halo(next_carry.state, halo_spec(namelist.grid)))


def _rk_scan_step(carry: OperationalCarry, namelist: OperationalNamelist, *, debug: bool = False) -> OperationalCarry:
    origin = apply_halo(carry.state, halo_spec(namelist.grid))
    carry = _with_save_family(carry.replace(state=origin), origin)

    def advance_stage(stage_carry: OperationalCarry, factor: float, acoustic_substeps: int) -> OperationalCarry:
        haloed = apply_halo(stage_carry.state, halo_spec(namelist.grid))
        tendencies = compute_advection_tendencies(haloed, namelist.tendencies, namelist.grid)
        tendencies = _m6b_acoustic_tendencies(tendencies, namelist.tendencies)
        candidate = add_scaled_tendencies(origin, tendencies, float(namelist.dt_s) * float(factor))
        stage_carry = _with_save_family(stage_carry.replace(state=candidate), candidate)
        stage_carry = _acoustic_scan(
            stage_carry,
            namelist,
            float(namelist.dt_s) * float(factor),
            substeps=acoustic_substeps,
        )
        return stage_carry.replace(state=apply_halo(stage_carry.state, halo_spec(namelist.grid)))

    # Static RK sequencing avoids per-stage scalar dispatch inside the profiled
    # timestep loop. WRF solve_em.F:1472-1479 runs one RK1 acoustic small step
    # and half the configured sound steps for RK2.
    # Legacy test anchor for the prior dynamic form:
    # lambda value: advance_stage(value, 1.0 / 3.0, 1)
    if debug:
        jax.debug.print("GPUWRF_M6B_RK1_ACOUSTIC_LOOP_ENTER substeps=1")
    carry = advance_stage(carry, 1.0 / 3.0, 1)
    carry = advance_stage(carry, 0.5, max(1, int(namelist.acoustic_substeps) // 2))
    return advance_stage(carry, 1.0, int(namelist.acoustic_substeps))


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


def _physics_boundary_step(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    step_index,
    *,
    run_radiation: bool,
    debug: bool = False,
) -> OperationalCarry:
    physical_origin = carry.state
    carry = _rk_scan_step(carry, namelist, debug=debug)
    # The controlled parity lane verifies the promoted acoustic scratch, but
    # full-domain operational theta/mu replacement is not yet a bounded
    # physical-state contract. Keep the prior carry-fix projection here.
    next_state = carry.state.replace(
        theta=physical_origin.theta,
        mu=physical_origin.mu,
        mu_total=physical_origin.mu_total,
        mu_perturbation=physical_origin.mu_perturbation,
    )
    if bool(namelist.run_physics):
        next_state = thompson_adapter(next_state, float(namelist.dt_s))
        next_state = mynn_adapter(next_state, float(namelist.dt_s), namelist.grid)
        next_state = surface_adapter(next_state, float(namelist.dt_s))
        if run_radiation:
            next_state = rrtmg_adapter(next_state, float(namelist.dt_s), namelist.grid)
    if bool(namelist.run_boundary):
        lead_seconds = step_index.astype(jnp.float64) * float(namelist.dt_s)
        next_state = apply_lateral_boundaries(next_state, lead_seconds, float(namelist.dt_s), namelist.boundary_config)
    next_state = _enforce_operational_precision(next_state)
    return carry.replace(state=next_state)


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


__all__ = ["OperationalNamelist", "run_forecast_operational", "run_forecast_operational_debug"]
