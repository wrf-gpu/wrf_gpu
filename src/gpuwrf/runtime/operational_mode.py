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


def _ph_tend_increment(old_state: State, new_state: State, dt_sub: float) -> jax.Array:
    """Operational geopotential tendency increment for ``ph_tend`` carry.

    WRF's small-step path accumulates geopotential tendency around the
    ``advance_w`` recurrence; the operational carry uses the resident PH delta
    over the acoustic substep so the accumulator remains source-shaped without
    importing validation helpers.
    """

    return (jnp.asarray(new_state.ph) - jnp.asarray(old_state.ph)) / jnp.asarray(dt_sub, dtype=new_state.ph.dtype)


def _advance_promoted_scratch(
    carry: OperationalCarry,
    old_state: State,
    new_state: State,
    *,
    mu_new: jax.Array,
    ww_new: jax.Array,
    dt_sub: float,
    epssm: float,
) -> OperationalCarry:
    """Inline the M6B3 scratch formulas for production carry.

    Source anchors: WRF ``module_small_step_em.F:1066-1175`` for ``ww``,
    ``muave`` and ``muts``; WRF small-step theta averaging for ``t_2ave``; and
    WRF small-step geopotential tendency accumulation for ``ph_tend``.
    """

    mu_old = old_state.mu_perturbation
    mu_base = _base_mu(old_state)
    t_2ave = 0.5 * (jnp.asarray(old_state.theta) + jnp.asarray(new_state.theta))
    muave = 0.5 * ((1.0 + float(epssm)) * mu_new + (1.0 - float(epssm)) * mu_old)
    muts = mu_base + mu_new
    ph_tend = carry.ph_tend + _ph_tend_increment(old_state, new_state, dt_sub)
    return carry.replace(
        state=new_state,
        t_2ave=t_2ave,
        ww=ww_new,
        muave=muave,
        muts=muts,
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
        mudf=jnp.zeros_like(state.mu_perturbation),
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
    a, alpha, gamma = calc_coef_w_wrf_coefficients(
        mu_total,
        namelist.metrics,
        dt=float(dt_sub),
        epssm=float(namelist.epssm),
        top_lid=bool(namelist.top_lid),
    )
    _tri_fwd, w_solved = thomas_solve_scan(a, alpha, gamma, state.w)
    # The promoted WRF scratch recurrence is resident, but operational
    # prognostic theta/mu remain on the existing ADR-007 state path until a
    # separate savepoint-aligned composition sprint approves replacing them.
    mu_new = state.mu_perturbation
    next_state = state.replace(
        w=w_solved,
    )
    return _advance_promoted_scratch(
        carry,
        state,
        next_state,
        mu_new=mu_new,
        ww_new=advanced["ww"],
        dt_sub=dt_sub,
        epssm=float(namelist.epssm),
    )


def _acoustic_scan(carry: OperationalCarry, namelist: OperationalNamelist, dt_stage: float) -> OperationalCarry:
    dt_sub = float(dt_stage) / float(namelist.acoustic_substeps)

    def body(scan_carry: OperationalCarry, _):
        if bool(namelist.use_vertical_solver):
            return _wrf_small_step_acoustic(scan_carry, namelist, dt_sub), None
        next_state = add_scaled_tendencies(scan_carry.state, namelist.tendencies, dt_sub)
        return _with_save_family(scan_carry.replace(state=next_state), next_state), None

    next_carry, _ = jax.lax.scan(body, carry, xs=None, length=int(namelist.acoustic_substeps))
    return next_carry.replace(state=apply_halo(next_carry.state, halo_spec(namelist.grid)))


def _rk_scan_step(carry: OperationalCarry, namelist: OperationalNamelist) -> OperationalCarry:
    origin = apply_halo(carry.state, halo_spec(namelist.grid))
    carry = _with_save_family(carry.replace(state=origin), origin)
    rk_indices = jnp.arange(int(namelist.rk_order), dtype=jnp.int32)

    def advance_stage(stage_carry: OperationalCarry, factor: float, use_acoustic: bool) -> OperationalCarry:
        haloed = apply_halo(stage_carry.state, halo_spec(namelist.grid))
        tendencies = compute_advection_tendencies(haloed, namelist.tendencies, namelist.grid)
        candidate = add_scaled_tendencies(origin, tendencies, float(namelist.dt_s) * float(factor))
        stage_carry = _with_save_family(stage_carry.replace(state=candidate), candidate)
        if use_acoustic:
            stage_carry = _acoustic_scan(stage_carry, namelist, float(namelist.dt_s) * float(factor))
        return stage_carry.replace(state=apply_halo(stage_carry.state, halo_spec(namelist.grid))), None

    def body(stage_carry: OperationalCarry, stage_index):
        return jax.lax.switch(
            stage_index,
            (
                lambda value: advance_stage(value, 1.0 / 3.0, False),
                lambda value: advance_stage(value, 0.5, True),
                lambda value: advance_stage(value, 1.0, True),
            ),
            stage_carry,
        )

    final_carry, _ = jax.lax.scan(body, carry, rk_indices, length=int(namelist.rk_order))
    return final_carry


def _physics_boundary_step(carry: OperationalCarry, namelist: OperationalNamelist, step_index) -> OperationalCarry:
    carry = _rk_scan_step(carry, namelist)
    next_state = carry.state
    if bool(namelist.run_physics):
        next_state = thompson_adapter(next_state, float(namelist.dt_s))
        next_state = mynn_adapter(next_state, float(namelist.dt_s), namelist.grid)
        next_state = surface_adapter(next_state, float(namelist.dt_s))
        run_radiation = (step_index % int(namelist.radiation_cadence_steps)) == 0
        next_state = jax.lax.cond(
            run_radiation,
            lambda value: rrtmg_adapter(value, float(namelist.dt_s), namelist.grid),
            lambda value: value,
            next_state,
        )
    if bool(namelist.run_boundary):
        lead_seconds = step_index.astype(jnp.float64) * float(namelist.dt_s)
        next_state = apply_lateral_boundaries(next_state, lead_seconds, float(namelist.dt_s), namelist.boundary_config)
    next_state = _enforce_operational_precision(next_state)
    return carry.replace(state=next_state)


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
    indices = jnp.arange(1, steps + 1, dtype=jnp.int32)

    def body(carry: OperationalCarry, step_index):
        return _physics_boundary_step(carry, namelist, step_index), None

    final_carry, _ = jax.lax.scan(body, initial, indices)
    return final_carry.state


__all__ = ["OperationalNamelist", "run_forecast_operational"]
