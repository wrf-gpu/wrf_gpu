from __future__ import annotations

import re

import jax

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import State, Tendencies
from gpuwrf.profiling.budget import compiled_text
from gpuwrf.timestep.dummy_loop import run_dummy_loop


def test_1000_step_dummy_loop_preserves_shape_dtype():
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)
    tendencies = Tendencies.zeros(grid)

    next_state, next_tendencies = run_dummy_loop(state, tendencies, 3.0, 1000)
    next_state.theta.block_until_ready()

    for before, after in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(next_state), strict=True):
        assert after.shape == before.shape
        assert after.dtype == before.dtype
    for before, after in zip(
        jax.tree_util.tree_leaves(tendencies), jax.tree_util.tree_leaves(next_tendencies), strict=True
    ):
        assert after.shape == before.shape
        assert after.dtype == before.dtype


def test_dummy_loop_lowers_to_single_jitted_scan_hlo():
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)
    tendencies = Tendencies.zeros(grid)

    text = compiled_text(run_dummy_loop.lower(state, tendencies, 3.0, 1000).compile())

    assert "while" in text.lower()
    assert re.search(r"\b(fusion|custom-call|while)\(", text)
