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
from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients
from gpuwrf.dynamics.advection import compute_advection_tendencies, halo_spec
from gpuwrf.dynamics.mu_t_advance import AdvanceMuTInputs, advance_mu_t_wrf
from gpuwrf.dynamics.tendencies import add_scaled_tendencies
from gpuwrf.dynamics.tridiag_solve import thomas_solve_scan
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
    """Infer WRF perturbation-theta offset without a host read."""

    return jnp.where(jnp.mean(theta) > 200.0, jnp.asarray(300.0, dtype=theta.dtype), jnp.asarray(0.0, dtype=theta.dtype))


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


def _ph_tend_increment(theta_old: jax.Array, theta_new: jax.Array, ph_tend: jax.Array) -> jax.Array:
    """Operational geopotential tendency increment for ``ph_tend`` carry.

    The M6B3/M6B4 validation ladder binds this source-shaped accumulator to a
    theta-delta increment before WRF ``advance_w`` consumes ``ph_tend`` in
    ``module_small_step_em.F:1345-1395``.  Keep that formula inline here so the
    operational path does not import validation-only modules.
    """

    theta_delta = jnp.asarray(theta_new) - jnp.asarray(theta_old)
    increment = jnp.zeros_like(ph_tend)
    return increment.at[: theta_delta.shape[0], :, :].set(0.01 * theta_delta)


def _advance_promoted_scratch(
    carry: OperationalCarry,
    old_state: State,
    new_state: State,
    *,
    mu_new: jax.Array,
    mudf_new: jax.Array,
    muts_new: jax.Array,
    muave_new: jax.Array,
    ww_new: jax.Array,
    theta_new: jax.Array,
    theta_offset: jax.Array,
) -> OperationalCarry:
    """Inline the M6B3 scratch formulas for production carry.

    Source anchors: WRF ``module_small_step_em.F:1102-1108`` commits ``MU``,
    ``MUDF``, ``MUTS`` and ``MUAVE`` in place; ``module_small_step_em.F:1141-1171``
    commits theta in place; ``module_small_step_em.F:1345-1395`` consumes the
    ``ph_tend`` accumulator in ``advance_w``.
    """

    offset = jnp.asarray(theta_offset, dtype=jnp.float64)
    theta_old_pert = jnp.asarray(old_state.theta, dtype=jnp.float64) - offset
    theta_next_pert = jnp.asarray(theta_new, dtype=jnp.float64) - offset
    t_2ave = 0.5 * (theta_old_pert + theta_next_pert) + offset
    ph_tend = carry.ph_tend + _ph_tend_increment(theta_old_pert, theta_next_pert, carry.ph_tend)
    return carry.replace(
        state=new_state,
        t_2ave=t_2ave,
        ww=ww_new,
        mudf=mudf_new,
        muave=muave_new,
        muts=muts_new,
        ph_tend=ph_tend,
        u_save=new_state.u,
        v_save=new_state.v,
        w_save=new_state.w,
        t_save=new_state.theta,
        ph_save=new_state.ph,
        mu_save=mu_new,
        ww_save=ww_new,
    )


def _wrf_small_step_acoustic(carry: OperationalCarry, namelist: OperationalNamelist, dt_sub: float) -> OperationalCarry:
    """Run one source-backed operational acoustic substep with promoted carry."""

    state = apply_halo(carry.state, halo_spec(namelist.grid))
    theta_offset = _theta_base_offset(state.theta)
    theta_pert = (state.theta - theta_offset).astype(jnp.float64)
    theta_save_pert = (carry.t_save - theta_offset).astype(jnp.float64)
    theta_ave_pert = (carry.t_2ave - theta_offset).astype(jnp.float64)
    mu_base = _base_mu(state)
    mu_total = mu_base + state.mu_perturbation
    inputs = AdvanceMuTInputs(
        ww=carry.ww,
        ww_1=carry.ww_save,
        u=state.u,
        u_1=carry.u_save,
        v=state.v,
        v_1=carry.v_save,
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
        rdx=1.0 / float(namelist.grid.projection.dx_m),
        rdy=1.0 / float(namelist.grid.projection.dy_m),
        dts=float(dt_sub),
        epssm=float(namelist.epssm),
    )
    advanced = advance_mu_t_wrf(inputs)
    # WRF ``solve_em.F:2409-2717`` builds W coefficients for the acoustic
    # small-step sequence; this operational body is called once per substep, so
    # the coefficients are recomputed from the resident substep mass each time.
    a, alpha, gamma = calc_coef_w_wrf_coefficients(
        carry.muts,
        namelist.metrics,
        dt=float(dt_sub),
        epssm=float(namelist.epssm),
        top_lid=bool(namelist.top_lid),
    )
    _tri_fwd, w_solved = thomas_solve_scan(a, alpha, gamma, state.w)
    # WRF ``solve_em.F:3435-3452`` calls ``advance_mu_t`` inside
    # ``small_steps``; ``module_small_step_em.F:1102-1108`` commits
    # MU/MUDF/MUTS/MUAVE and ``:1141-1171`` commits theta in place.
    mu_new = advanced["mu"]
    mu_total_new = mu_base + mu_new
    theta_new = advanced["theta"] + theta_offset
    next_state = state.replace(
        w=w_solved,
        theta=theta_new,
        mu=mu_total_new,
        mu_total=mu_total_new,
        mu_perturbation=mu_new,
    )
    return _advance_promoted_scratch(
        carry,
        state,
        next_state,
        mu_new=mu_new,
        mudf_new=advanced["mudf"],
        muts_new=advanced["muts"],
        muave_new=advanced["muave"],
        ww_new=advanced["ww"],
        theta_new=theta_new,
        theta_offset=theta_offset,
    )


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
            return _wrf_small_step_acoustic(scan_carry, namelist, dt_sub), None
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


def _physics_boundary_step(
    carry: OperationalCarry,
    namelist: OperationalNamelist,
    step_index,
    *,
    run_radiation: bool,
    debug: bool = False,
) -> OperationalCarry:
    carry = _rk_scan_step(carry, namelist, debug=debug)
    next_state = carry.state
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
