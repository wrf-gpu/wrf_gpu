"""v0.18 #37 default-path guard for conditional additive State leaves."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts import precision as precision_contract
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import (
    AEROSOL_CONDITIONAL_LEAVES,
    CONDITIONAL_STATE_LEAVES,
    HAIL_CONDITIONAL_LEAVES,
    State,
    _state_field_shapes,
    conditional_state_leaves_for_mp,
)


MYNN_HISTORY_LEAVES = getattr(precision_contract, "MYNN_DIAGNOSTIC_LEAVES", ())
assert MYNN_HISTORY_LEAVES in ((), ("el_pbl", "maxmf", "maxwidth", "ztop_plume"))
GWDO_HISTORY_LEAVES = getattr(precision_contract, "GWDO_DIAGNOSTIC_LEAVES", ())
assert GWDO_HISTORY_LEAVES in ((), ("dtaux3d", "dtauy3d", "dusfcg", "dvsfcg"))
PRE_HAIL_BASE_LEAF_COUNT = len(State.__slots__) - len(CONDITIONAL_STATE_LEAVES) - len(GWDO_HISTORY_LEAVES)


def _assert_diagnostic_lanes(state: State, grid: GridSpec) -> None:
    # MYNN is required independently of MP. GWDO is optional under gwd_opt=0
    # and must not be routed through the microphysics conditional selector.
    assert not set(GWDO_HISTORY_LEAVES) & set(CONDITIONAL_STATE_LEAVES)
    for name in GWDO_HISTORY_LEAVES:
        assert getattr(state, name) is None, name
    for name in MYNN_HISTORY_LEAVES:
        value = getattr(state, name)
        shape = (grid.nz, grid.ny, grid.nx) if name == "el_pbl" else (grid.ny, grid.nx)
        assert value.shape == shape, name
        assert value.dtype == jnp.float32, name
        assert value.nbytes == 4 * int(np.prod(shape)), name
        assert np.asarray(value).tobytes() == np.zeros(shape, dtype=np.float32).tobytes(), name


def _state_for_mp(grid: GridSpec, mp_physics: int) -> State:
    return State(**{
        name: jnp.zeros(shape, dtype=DEFAULT_DTYPES.dtype_for(name))
        for name, shape in _state_field_shapes(grid, mp_physics=mp_physics).items()
    })


def test_default_mp8_state_carries_pre_hail_base_leaf_set_only() -> None:
    grid = GridSpec.canary_3km_template()
    state = _state_for_mp(grid, 8)

    assert len(State.__slots__) == 76 + len(MYNN_HISTORY_LEAVES) + len(GWDO_HISTORY_LEAVES)
    assert PRE_HAIL_BASE_LEAF_COUNT == 62 + len(MYNN_HISTORY_LEAVES)
    assert len(_state_field_shapes(grid, mp_physics=8)) == PRE_HAIL_BASE_LEAF_COUNT
    assert len(state.active_field_names()) == PRE_HAIL_BASE_LEAF_COUNT
    assert len(jax.tree_util.tree_leaves(state)) == PRE_HAIL_BASE_LEAF_COUNT
    assert conditional_state_leaves_for_mp(8) == ()
    assert state.active_field_names() == tuple(
        name for name in State.__slots__
        if name not in CONDITIONAL_STATE_LEAVES and name not in GWDO_HISTORY_LEAVES
    )
    _assert_diagnostic_lanes(state, grid)
    for leaf in CONDITIONAL_STATE_LEAVES:
        assert getattr(state, leaf) is None, leaf


def test_hail_and_aerosol_schemes_materialize_only_their_static_lanes() -> None:
    grid = GridSpec.canary_3km_template()

    for mp in (24, 26):
        state = _state_for_mp(grid, mp)
        assert conditional_state_leaves_for_mp(mp) == HAIL_CONDITIONAL_LEAVES
        assert len(jax.tree_util.tree_leaves(state)) == PRE_HAIL_BASE_LEAF_COUNT + len(HAIL_CONDITIONAL_LEAVES)
        assert state.active_field_names() == tuple(
            name for name in State.__slots__
            if name not in GWDO_HISTORY_LEAVES
            and (name not in CONDITIONAL_STATE_LEAVES or name in HAIL_CONDITIONAL_LEAVES)
        )
        _assert_diagnostic_lanes(state, grid)
        for leaf in HAIL_CONDITIONAL_LEAVES:
            assert getattr(state, leaf) is not None, (mp, leaf)
        for leaf in AEROSOL_CONDITIONAL_LEAVES:
            assert getattr(state, leaf) is None, (mp, leaf)

    aero = _state_for_mp(grid, 28)
    assert conditional_state_leaves_for_mp(28) == AEROSOL_CONDITIONAL_LEAVES
    assert len(jax.tree_util.tree_leaves(aero)) == PRE_HAIL_BASE_LEAF_COUNT + len(AEROSOL_CONDITIONAL_LEAVES)
    assert aero.active_field_names() == tuple(
        name for name in State.__slots__
        if name not in GWDO_HISTORY_LEAVES
        and (name not in CONDITIONAL_STATE_LEAVES or name in AEROSOL_CONDITIONAL_LEAVES)
    )
    _assert_diagnostic_lanes(aero, grid)
    for leaf in AEROSOL_CONDITIONAL_LEAVES:
        assert getattr(aero, leaf) is not None, leaf
    for leaf in HAIL_CONDITIONAL_LEAVES:
        assert getattr(aero, leaf) is None, leaf
