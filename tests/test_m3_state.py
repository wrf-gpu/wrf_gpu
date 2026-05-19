from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, Tendencies


def _platform(array) -> str:
    return array.devices().copy().pop().platform


def test_state_zeros_allocates_gpu_shapes_and_dtype():
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)

    assert state.u.shape == (grid.nz, grid.ny, grid.nx + 1)
    assert state.v.shape == (grid.nz, grid.ny + 1, grid.nx)
    assert state.w.shape == (grid.nz + 1, grid.ny, grid.nx)
    assert state.theta.shape == (grid.nz, grid.ny, grid.nx)
    assert state.qv.shape == (grid.nz, grid.ny, grid.nx)
    assert state.p.shape == (grid.nz, grid.ny, grid.nx)
    assert state.ph.shape == (grid.nz + 1, grid.ny, grid.nx)
    assert state.mu.shape == (grid.ny, grid.nx)
    assert all(leaf.dtype == jnp.float64 for leaf in jax.tree_util.tree_leaves(state))
    assert all(_platform(leaf) == "gpu" for leaf in jax.tree_util.tree_leaves(state))


def test_state_and_tendency_bytes_match_manual_sum():
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)
    tendencies = Tendencies.zeros(grid)

    manual_state = sum(int(leaf.size) * int(leaf.dtype.itemsize) for leaf in jax.tree_util.tree_leaves(state))
    manual_tendencies = sum(int(leaf.size) * int(leaf.dtype.itemsize) for leaf in jax.tree_util.tree_leaves(tendencies))

    assert state.bytes() == manual_state
    assert tendencies.bytes() == manual_tendencies


def test_precision_registry_rejects_unknown_field():
    assert DEFAULT_DTYPES.dtype_for("theta") == jnp.float64
    try:
        DEFAULT_DTYPES.dtype_for("rain")
    except KeyError as exc:
        assert "rain" in str(exc)
    else:
        raise AssertionError("precision registry accepted an unknown field")
