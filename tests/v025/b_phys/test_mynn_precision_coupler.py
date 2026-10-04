"""Retained-interface/type regressions; WRF fidelity uses real-column oracles."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling import physics_couplers as C
from test_v014_dry_source_leaf_wiring import _grid, _state


@pytest.fixture(autouse=True)
def native_flag(monkeypatch):
    monkeypatch.setenv("GPUWRF_MYNN_FP32_COLUMNS", "1")
    monkeypatch.setenv("GPUWRF_MYNN_CLOUDMIX", "1")
    jax.clear_caches()
    yield
    jax.clear_caches()


@pytest.mark.parametrize("grid_backed", [False, True])
def test_native_column_and_surface_inputs_are_real32(grid_backed):
    grid = _grid()
    state = _state(grid)
    column = C._mynn_column_from_state(state, grid if grid_backed else None)
    surface = C._surface_fluxes_from_state(state)
    assert {v.dtype for v in jax.tree.leaves((column, surface))} == {np.dtype("float32")}


def test_traced_fresh_call_preserves_retained_carry_dtype():
    grid = _grid()
    state = _state(grid)
    output = jax.eval_shape(lambda s, flag: C.mynn_adapter_with_source_leaves(
        s, 10.0, grid, first_timestep=flag), state, jnp.asarray(True))
    before = jax.tree.leaves(state)
    after = jax.tree.leaves(output.state)
    assert len(before) == len(after)
    assert all((a.shape, a.dtype) == (b.shape, b.dtype) for a, b in zip(before, after))
    assert output.rthblten.dtype == state.theta.dtype
    assert output.rqvblten.dtype == state.qv.dtype
