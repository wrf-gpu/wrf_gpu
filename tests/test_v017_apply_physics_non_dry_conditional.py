"""Regression guard for the #37 conditional-leaf crash in physics non-dry updates.

`_apply_physics_non_dry_updates` iterates `_PHYSICS_NON_DRY_INCREMENT_FIELDS`,
which now contains the static-scheme-gated conditional leaves
nwfa/nifa/qh/Nh/qvolg/qvolh.  On a DEFAULT mp=8 State those leaves are None on
every State object, so an unguarded `getattr(a,n) + (getattr(b,n) - getattr(c,n))`
would do `None + (None - None)` -> TypeError on every physics step.  The function
must skip leaves that are not active (None) and must NOT materialize them.

CPU-safe: `State.zeros` requires a GPU device, so the mp=8 State is built
directly from `_state_field_shapes` (the same pattern tests/_full_state uses).
"""

import jax.numpy as jnp

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import (
    CONDITIONAL_STATE_LEAVES,
    State,
    _state_field_shapes,
)
from gpuwrf.runtime.operational_mode import (
    _apply_physics_non_dry_updates,
    _PHYSICS_NON_DRY_INCREMENT_FIELDS,
)


def _mp8_state() -> State:
    grid = GridSpec.canary_3km_template()
    shapes = _state_field_shapes(grid, mp_physics=8)
    fields = {k: jnp.zeros(v, dtype=DEFAULT_DTYPES.dtype_for(k)) for k, v in shapes.items()}
    return State(**fields)


def test_apply_physics_non_dry_updates_mp8_none_conditional_leaves_no_crash() -> None:
    state = _mp8_state()

    # Precondition: the conditional additive leaves really are None on mp=8.
    none_increment_leaves = tuple(
        n for n in _PHYSICS_NON_DRY_INCREMENT_FIELDS if getattr(state, n) is None
    )
    assert none_increment_leaves, "expected conditional increment leaves to be None on mp=8"
    assert set(none_increment_leaves) == (
        set(CONDITIONAL_STATE_LEAVES) & set(_PHYSICS_NON_DRY_INCREMENT_FIELDS)
    )

    # All three step States are mp=8 here -> every conditional leaf is None.
    out = _apply_physics_non_dry_updates(state, state, state)  # must NOT raise

    # The inactive conditional leaves must stay None (skipped, not materialized).
    for name in none_increment_leaves:
        assert getattr(out, name) is None, name
