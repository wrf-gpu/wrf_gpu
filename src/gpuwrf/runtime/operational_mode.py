"""GPU-resident operational forecast loop for M6 perf-design.

This module is deliberately separate from the M6B validation savepoint ladder.
It carries only the production ``State`` leaves, runs timestep/RK/acoustic loops
inside one JAX entry point, and leaves debug snapshots/sanitizers out of the
compiled path.
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
from gpuwrf.dynamics.acoustic import acoustic_once
from gpuwrf.dynamics.acoustic_wrf import vertical_acoustic_update
from gpuwrf.dynamics.advection import compute_advection_tendencies, halo_spec
from gpuwrf.dynamics.tendencies import add_scaled_tendencies


config.update("jax_enable_x64", True)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class OperationalNamelist:
    """Static runtime controls plus resident metric/tendency leaves.

    ``grid`` and scalar controls are static cache keys. ``tendencies`` and
    ``metrics`` are device leaves so no host/device transfer is needed in the
    timestep loop. The production carry remains the existing ADR-007 ``State``
    and excludes M6B validation scratch fields such as ``t_2ave``, ``ww``,
    ``muave``, ``muts``, ``ph_tend`` and ``*_save``.
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


def _acoustic_scan(state: State, namelist: OperationalNamelist, dt_stage: float) -> State:
    dt_sub = float(dt_stage) / float(namelist.acoustic_substeps)

    def body(carry: State, _):
        haloed = apply_halo(carry, halo_spec(namelist.grid))
        next_state = acoustic_once(haloed, namelist.grid, dt_sub)
        if bool(namelist.use_vertical_solver):
            next_state = vertical_acoustic_update(
                next_state,
                None,
                namelist.metrics,
                dt=dt_sub,
                epssm=float(namelist.epssm),
                top_lid=bool(namelist.top_lid),
            )
        return next_state, None

    next_state, _ = jax.lax.scan(body, state, xs=None, length=int(namelist.acoustic_substeps))
    return apply_halo(next_state, halo_spec(namelist.grid))


def _rk_scan_step(state: State, namelist: OperationalNamelist) -> State:
    origin = apply_halo(state, halo_spec(namelist.grid))
    rk_indices = jnp.arange(int(namelist.rk_order), dtype=jnp.int32)

    def advance_stage(stage_state: State, factor: float, use_acoustic: bool) -> State:
        haloed = apply_halo(stage_state, halo_spec(namelist.grid))
        tendencies = compute_advection_tendencies(haloed, namelist.tendencies, namelist.grid)
        candidate = add_scaled_tendencies(origin, tendencies, float(namelist.dt_s) * float(factor))
        if use_acoustic:
            candidate = _acoustic_scan(candidate, namelist, float(namelist.dt_s) * float(factor))
        return apply_halo(candidate, halo_spec(namelist.grid)), None

    def body(stage_state: State, stage_index):
        return jax.lax.switch(
            stage_index,
            (
                lambda value: advance_stage(value, 1.0 / 3.0, False),
                lambda value: advance_stage(value, 0.5, True),
                lambda value: advance_stage(value, 1.0, True),
            ),
            stage_state,
        )

    final_state, _ = jax.lax.scan(body, origin, rk_indices, length=int(namelist.rk_order))
    return final_state


def _physics_boundary_step(state: State, namelist: OperationalNamelist, step_index) -> State:
    next_state = _rk_scan_step(state, namelist)
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
    return _enforce_operational_precision(next_state)


@partial(jax.jit, static_argnames=("hours",), donate_argnums=(0,))
def run_forecast_operational(state: State, namelist: OperationalNamelist, hours: float) -> State:
    """Run an operational forecast as one compiled, device-resident scan.

    No diagnostics, host-read callbacks, host array pulls, or sanitizers are
    present in this path. ``hours`` is static so the timestep count is fixed at
    compile time and the whole forecast lowers as one JAX program.
    """

    if int(namelist.rk_order) != 3:
        raise ValueError("operational mode currently supports RK3 only")
    initial = _enforce_operational_precision(state)
    steps = _steps_for_hours(hours, float(namelist.dt_s))
    indices = jnp.arange(1, steps + 1, dtype=jnp.int32)

    def body(carry: State, step_index):
        return _physics_boundary_step(carry, namelist, step_index), None

    final_state, _ = jax.lax.scan(body, initial, indices)
    return final_state


__all__ = ["OperationalNamelist", "run_forecast_operational"]
