"""v0.22 G2 moving-nest and adaptive-timestep small-grid validation gate."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import DomainHierarchy, DomainNest
from gpuwrf.nesting.adaptive_timestep import (
    AdaptiveTimeStepConfig,
    AdaptiveTimeStepState,
    adapt_timestep,
)
from gpuwrf.nesting.moving import (
    MovingNestBounds,
    apply_move_to_edge,
    planned_vortex_move,
    shift_array_for_nest_move,
)
from gpuwrf.runtime.domain_tree import DomainEdge, run_domain_tree_callbacks


@dataclass(frozen=True)
class _Carry:
    theta: jnp.ndarray
    dt_s: float
    max_horiz_cfl: float
    max_vert_cfl: float = 0.6

    def replace(self, **updates: Any) -> "_Carry":
        values = {
            "theta": self.theta,
            "dt_s": self.dt_s,
            "max_horiz_cfl": self.max_horiz_cfl,
            "max_vert_cfl": self.max_vert_cfl,
        }
        values.update(updates)
        return _Carry(**values)


def _all_finite(*arrays: Any) -> bool:
    return all(bool(np.isfinite(np.asarray(array)).all()) for array in arrays)


def run_gate(*, output: str | Path | None = None) -> dict[str, Any]:
    hierarchy = DomainHierarchy.from_edges(
        ("d01", "d02"),
        (DomainNest("d01", "d02", 3, 4, 4),),
    )
    child_shape = (1, 9, 12)
    child_theta = jnp.arange(np.prod(child_shape), dtype=jnp.float64).reshape(child_shape)
    fill_theta = jnp.full(child_shape, -999.0, dtype=jnp.float64)
    parent_theta = jnp.ones((1, 6, 6), dtype=jnp.float64)
    carries = {
        "d01": _Carry(parent_theta, dt_s=10.0, max_horiz_cfl=1.20),
        "d02": _Carry(child_theta, dt_s=10.0 / 3.0, max_horiz_cfl=0.25),
    }
    dt_config = AdaptiveTimeStepConfig(
        target_cfl=1.2,
        target_hcfl=0.84,
        min_time_step_s=2.0,
        max_time_step_s=12.0,
        max_step_increase_pct=20.0,
    )
    dt_history: list[tuple[str, int, float]] = []
    move_history: list[tuple[int, int, int, int]] = []

    def adaptive_dt(name: str, carry: _Carry, start_step: int):
        if name != "d01":
            return carry, carry.dt_s
        result = adapt_timestep(
            AdaptiveTimeStepState(
                dt_s=carry.dt_s,
                last_dt_s=carry.dt_s,
                max_vert_cfl=carry.max_vert_cfl,
                max_horiz_cfl=carry.max_horiz_cfl,
                current_seconds=(start_step - 1) * carry.dt_s,
                advance_count=start_step,
            ),
            dt_config,
        )
        dt_history.append((name, int(start_step), float(result.dt_s)))
        # Step 1 has high CFL and steps down; step 2 has low CFL and exercises
        # the WRF max_step_increase_pct cap back upward.
        next_cfl = 0.25 if start_step == 1 else carry.max_horiz_cfl
        return carry.replace(dt_s=result.dt_s, max_horiz_cfl=next_cfl), result.dt_s

    def move(edge: DomainEdge, parent: _Carry, child: _Carry, parent_step: int):
        del parent
        if parent_step != 1:
            return None
        planned = planned_vortex_move(
            vortex_i=12.0,
            vortex_j=5.0,
            child_nx=child.theta.shape[-1],
            child_ny=child.theta.shape[-2],
            parent_grid_ratio=edge.parent_grid_ratio,
        )
        moved = apply_move_to_edge(
            edge.spec,
            planned,
            bounds=MovingNestBounds(
                parent_nx=20,
                parent_ny=20,
                child_nx=child.theta.shape[-1],
                child_ny=child.theta.shape[-2],
                parent_grid_ratio=edge.parent_grid_ratio,
            ),
        )
        shifted = shift_array_for_nest_move(
            child.theta,
            planned,
            parent_grid_ratio=edge.parent_grid_ratio,
            fill=fill_theta,
        )
        move_history.append(
            (
                edge.spec.i_parent_start,
                edge.spec.j_parent_start,
                moved.i_parent_start,
                moved.j_parent_start,
            )
        )
        return DomainEdge(moved, edge.weights, edge.feedback_weights), child.replace(theta=shifted)

    def advance(name: str, carry: _Carry, start_step: int, n_steps: int) -> _Carry:
        del name, start_step
        return carry.replace(theta=carry.theta + float(n_steps) * float(carry.dt_s))

    def force(edge: DomainEdge, parent: _Carry, child: _Carry) -> _Carry:
        # Make the moved edge visible to the proof without manufacturing a pass:
        # only the metadata contributes this tiny offset, and field finiteness is
        # checked independently after the state shift.
        offset = 1.0e-3 * (edge.spec.i_parent_start + edge.spec.j_parent_start)
        return child.replace(theta=child.theta + offset + 0.0 * jnp.mean(parent.theta))

    result = run_domain_tree_callbacks(
        hierarchy,
        carries,
        root_steps=2,
        advance=advance,
        force=force,
        adaptive_dt=adaptive_dt,
        move=move,
        block_between=False,
    )
    final_child = result.carries["d02"]
    moved = move_history[0] if move_history else None
    shifted_expected = np.asarray(child_theta[:, :, 3:])
    shifted_actual = np.asarray(final_child.theta[:, :, :9])
    # Remove the deterministic advance/force offsets before comparing the shifted
    # overlap.  d02 advances 6 own steps at its carried fixed dt (10/3), and force
    # adds two tiny edge offsets: first moved start 5+4, then unchanged 5+4.
    offset = 6.0 * (10.0 / 3.0) + 2.0 * 1.0e-3 * (5 + 4)
    shifted_overlap_ok = bool(np.allclose(shifted_actual - offset, shifted_expected, rtol=0.0, atol=1.0e-9))
    exposed_fill_ok = bool(np.allclose(np.asarray(final_child.theta[:, :, 9:] - offset), -999.0))

    global_edge = DomainNest("d01", "d02", 3, 10, 4)
    global_moved = apply_move_to_edge(
        global_edge,
        planned_vortex_move(vortex_i=12.0, vortex_j=5.0, child_nx=9, child_ny=9, parent_grid_ratio=3),
        bounds=MovingNestBounds(
            parent_nx=12,
            parent_ny=12,
            child_nx=9,
            child_ny=9,
            parent_grid_ratio=3,
            global_x=True,
        ),
    )

    nested_child_dt = adapt_timestep(
        AdaptiveTimeStepState(
            dt_s=4.0,
            last_dt_s=4.0,
            max_vert_cfl=0.5,
            max_horiz_cfl=0.5,
            advance_count=2,
        ),
        AdaptiveTimeStepConfig(
            target_cfl=1.2,
            target_hcfl=0.84,
            min_time_step_s=1.0,
            max_time_step_s=4.0,
            max_step_increase_pct=20.0,
        ),
        nested_parent_dt_s=10.0,
    )

    event_kinds = [event[0] for event in result.events]
    dt_values = [item[2] for item in dt_history]
    pass_gate = (
        moved == (4, 4, 5, 4)
        and shifted_overlap_ok
        and exposed_fill_ok
        and len(dt_values) >= 2
        and dt_values[0] < 10.0
        and dt_values[1] > dt_values[0]
        and _all_finite(final_child.theta)
        and global_moved.i_parent_start == 1
        and nested_child_dt.num_small_steps == 3
        and abs(nested_child_dt.dt_s - (10.0 / 3.0)) <= 1.0e-12
    )
    payload: dict[str, Any] = {
        "schema": "gpuwrf.v022.moving_nest_adaptive_gate",
        "schema_version": 1,
        "verdict": "PASS" if pass_gate else "FAIL",
        "wrf_reference": {
            "root": "<DATA_ROOT>/src/wrf_pristine",
            "moving_driver": "WRF/share/mediation_nest_move.F::med_nest_move/time_for_move2",
            "state_shift": "WRF/dyn_em/shift_domain_em.F::shift_domain_em",
            "domain_dims": "WRF/frame/module_domain.F::adjust_domain_dims_for_move",
            "adaptive_dt": "WRF/dyn_em/adapt_timestep_em.F::adapt_timestep/calc_dt",
            "global_polar": "WRF/dyn_em/module_polarfft.F and periodic_x branches",
        },
        "moving_nest": {
            "initial_start": [4, 4],
            "moved_start": [moved[2], moved[3]] if moved else None,
            "dx_parent": 1 if moved else 0,
            "dy_parent": 0,
            "fine_cell_shift": 3,
            "event_present": "move" in event_kinds,
            "shifted_overlap_ok": shifted_overlap_ok,
            "exposed_fill_ok": exposed_fill_ok,
            "finite": _all_finite(final_child.theta),
        },
        "adaptive_dt": {
            "dt_history": dt_values,
            "first_step_decreased": dt_values[0] < 10.0 if dt_values else False,
            "second_step_increased": dt_values[1] > dt_values[0] if len(dt_values) > 1 else False,
            "dt_events": [event for event in result.events if event[0] == "dt"],
            "nested_parent_divisor": {
                "parent_dt_s": nested_child_dt.parent_dt_s,
                "child_dt_s": nested_child_dt.dt_s,
                "num_small_steps": nested_child_dt.num_small_steps,
            },
        },
        "global_nests": {
            "periodic_x_start_before": global_edge.i_parent_start,
            "periodic_x_start_after": global_moved.i_parent_start,
            "periodic_x_wrap_exercised": global_moved.i_parent_start == 1,
            "status": "scaffold: metadata wrap only; polar filtering/full global runtime not landed",
        },
        "landed_vs_scaffold": {
            "landed": [
                "WRF one-parent-cell moving-nest decision primitive",
                "DomainNest i_parent_start/j_parent_start reposition helper",
                "resident child fine-grid state shift with exposed-cell fill hook",
                "run_domain_tree_callbacks move hook before forcedown",
                "WRF calc_dt-style adaptive timestep planner",
                "run_domain_tree_callbacks adaptive_dt hook and dt audit events",
                "periodic/global-x metadata wrap helper",
            ],
            "scaffold_deferred": [
                "full operational dynamic-dt compiled dycore without recompilation churn",
                "runtime rebuilding of NestForceWeights/feedback weights after every move",
                "WRF terrain/base-state reblend and start_domain choreography after moves",
                "full vortex-center diagnostics from PSFC/height/xlat/xlong fields",
                "polar FFT/global-nest physics validation",
                "375-variable wrfout/auxhist stream expansion",
            ],
        },
        "transfer_audit": {
            "gpu_used": False,
            "host_device_transfer_inside_timestep_loop": False,
            "note": "CPU small-grid JAX gate; production GPU runs must use scripts/with_gpu_lock.sh.",
        },
        "events": [list(event) for event in result.events],
        "argv": sys.argv,
    }
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = ["run_gate"]
