from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")

import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import DomainNest
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
from gpuwrf.validation.moving_nest_gate import run_gate


def test_v022_moving_nest_adaptive_gate(tmp_path):
    payload = run_gate(output=tmp_path / "g2_moving.json")

    assert payload["verdict"] == "PASS"
    assert payload["moving_nest"]["moved_start"] == [5, 4]
    assert payload["moving_nest"]["event_present"] is True
    assert payload["moving_nest"]["shifted_overlap_ok"] is True
    assert payload["moving_nest"]["exposed_fill_ok"] is True
    assert payload["moving_nest"]["finite"] is True
    assert payload["adaptive_dt"]["first_step_decreased"] is True
    assert payload["adaptive_dt"]["second_step_increased"] is True
    assert payload["adaptive_dt"]["nested_parent_divisor"]["num_small_steps"] == 3
    assert payload["global_nests"]["periodic_x_wrap_exercised"] is True


def test_vortex_move_is_clipped_to_one_parent_cell():
    move = planned_vortex_move(
        vortex_i=100.0,
        vortex_j=-100.0,
        child_nx=21,
        child_ny=21,
        parent_grid_ratio=3,
    )

    assert move.dx_parent == 1
    assert move.dy_parent == -1


def test_shift_array_fills_exposed_cells_without_wrap():
    field = jnp.arange(2 * 4 * 6, dtype=jnp.float64).reshape(2, 4, 6)
    fill = jnp.full_like(field, -7.0)
    move = planned_vortex_move(
        vortex_i=8.0,
        vortex_j=2.5,
        child_nx=6,
        child_ny=4,
        parent_grid_ratio=3,
    )
    shifted = shift_array_for_nest_move(field, move, parent_grid_ratio=3, fill=fill)

    np.testing.assert_array_equal(np.asarray(shifted[..., :3]), np.asarray(field[..., 3:]))
    np.testing.assert_array_equal(np.asarray(shifted[..., 3:]), -7.0)


def test_apply_move_to_edge_clamps_or_wraps():
    edge = DomainNest("d01", "d02", 3, 10, 4)
    move = planned_vortex_move(vortex_i=12.0, vortex_j=5.0, child_nx=9, child_ny=9, parent_grid_ratio=3)

    clamped = apply_move_to_edge(
        edge,
        move,
        bounds=MovingNestBounds(parent_nx=12, parent_ny=12, child_nx=9, child_ny=9, parent_grid_ratio=3),
    )
    wrapped = apply_move_to_edge(
        edge,
        move,
        bounds=MovingNestBounds(
            parent_nx=12,
            parent_ny=12,
            child_nx=9,
            child_ny=9,
            parent_grid_ratio=3,
            global_x=True,
        ),
    )

    assert clamped.i_parent_start == 10
    assert wrapped.i_parent_start == 1


def test_adaptive_timestep_reduces_and_parent_divides_child():
    cfg = AdaptiveTimeStepConfig(
        target_cfl=1.2,
        target_hcfl=0.84,
        min_time_step_s=1.0,
        max_time_step_s=4.0,
        max_step_increase_pct=20.0,
    )
    reduced = adapt_timestep(
        AdaptiveTimeStepState(
            dt_s=10.0,
            last_dt_s=10.0,
            max_vert_cfl=0.8,
            max_horiz_cfl=1.2,
            advance_count=1,
        ),
        cfg,
    )
    child = adapt_timestep(
        AdaptiveTimeStepState(
            dt_s=4.0,
            last_dt_s=4.0,
            max_vert_cfl=0.5,
            max_horiz_cfl=0.5,
            advance_count=2,
        ),
        cfg,
        nested_parent_dt_s=10.0,
    )

    assert reduced.dt_s < 10.0
    assert child.num_small_steps == 3
    assert abs(child.dt_s - (10.0 / 3.0)) <= 1.0e-12
