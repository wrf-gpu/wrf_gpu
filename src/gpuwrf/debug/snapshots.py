"""Debug-only state snapshot callbacks for post-hoc triage."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import jax
import jax.numpy as jnp

from gpuwrf.contracts.state import State


_SNAPSHOTS: "OrderedDict[str, dict[str, Any]]" = OrderedDict()


def _record_snapshot(stage: str, ring_size: int, *leaves) -> None:
    """Stores host copies for debug runs without participating in production HLO."""

    fields = State.__slots__
    state_leaves = leaves[: len(fields)]
    _SNAPSHOTS[stage] = {name: state_leaves[index] for index, name in enumerate(fields)}
    while len(_SNAPSHOTS) > int(ring_size):
        _SNAPSHOTS.popitem(last=False)


def snapshot(state: State, stage: str, *, enabled: bool, ring_size: int = 8) -> State:
    """Returns immediately in production; records a ring of states in debug traces."""

    if not enabled:
        return state
    token = jnp.asarray(0, dtype=state.theta.dtype)
    jax.debug.callback(_record_snapshot, stage, int(ring_size), *jax.tree_util.tree_leaves(state), token)
    return state


def dump_snapshots(state: State | None = None) -> dict[str, dict[str, Any]]:
    """Returns the current debug snapshot ring for post-hoc inspection."""

    del state
    return dict(_SNAPSHOTS)
