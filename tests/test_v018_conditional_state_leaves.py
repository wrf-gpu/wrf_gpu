"""v0.18 #37 default-path guard for conditional additive State leaves."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import (
    AEROSOL_CONDITIONAL_LEAVES,
    CONDITIONAL_STATE_LEAVES,
    HAIL_CONDITIONAL_LEAVES,
    State,
    _state_field_shapes,
    conditional_state_leaves_for_mp,
)


PRE_HAIL_BASE_LEAF_COUNT = len(State.__slots__) - len(CONDITIONAL_STATE_LEAVES)


def _state_for_mp(grid: GridSpec, mp_physics: int) -> State:
    return State(**{
        name: jnp.zeros(shape, dtype=DEFAULT_DTYPES.dtype_for(name))
        for name, shape in _state_field_shapes(grid, mp_physics=mp_physics).items()
    })


def test_default_mp8_state_carries_pre_hail_base_leaf_set_only() -> None:
    grid = GridSpec.canary_3km_template()
    state = _state_for_mp(grid, 8)

    assert PRE_HAIL_BASE_LEAF_COUNT == 57  # v0.20 S1: -3 legacy p/ph/mu aliases removed
    assert len(_state_field_shapes(grid, mp_physics=8)) == PRE_HAIL_BASE_LEAF_COUNT
    assert len(state.active_field_names()) == PRE_HAIL_BASE_LEAF_COUNT
    assert len(jax.tree_util.tree_leaves(state)) == PRE_HAIL_BASE_LEAF_COUNT
    assert conditional_state_leaves_for_mp(8) == ()
    for leaf in CONDITIONAL_STATE_LEAVES:
        assert getattr(state, leaf) is None, leaf


def test_hail_and_aerosol_schemes_materialize_only_their_static_lanes() -> None:
    grid = GridSpec.canary_3km_template()

    for mp in (24, 26):
        state = _state_for_mp(grid, mp)
        assert conditional_state_leaves_for_mp(mp) == HAIL_CONDITIONAL_LEAVES
        assert len(jax.tree_util.tree_leaves(state)) == PRE_HAIL_BASE_LEAF_COUNT + len(HAIL_CONDITIONAL_LEAVES)
        for leaf in HAIL_CONDITIONAL_LEAVES:
            assert getattr(state, leaf) is not None, (mp, leaf)
        for leaf in AEROSOL_CONDITIONAL_LEAVES:
            assert getattr(state, leaf) is None, (mp, leaf)

    aero = _state_for_mp(grid, 28)
    assert conditional_state_leaves_for_mp(28) == AEROSOL_CONDITIONAL_LEAVES
    assert len(jax.tree_util.tree_leaves(aero)) == PRE_HAIL_BASE_LEAF_COUNT + len(AEROSOL_CONDITIONAL_LEAVES)
    for leaf in AEROSOL_CONDITIONAL_LEAVES:
        assert getattr(aero, leaf) is not None, leaf
    for leaf in HAIL_CONDITIONAL_LEAVES:
        assert getattr(aero, leaf) is None, leaf
