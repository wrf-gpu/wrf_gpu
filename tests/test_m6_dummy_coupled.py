from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp

from gpuwrf.contracts.state import Tendencies
from gpuwrf.coupling.physics_couplers import mynn_adapter, rrtmg_adapter, surface_adapter, thompson_adapter
from gpuwrf.profiling.transfer_audit import block_until_ready


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "m6_run_dummy_coupled.py"
_SPEC = importlib.util.spec_from_file_location("m6_run_dummy_coupled", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
make_dummy_grid = _MODULE.make_dummy_grid
make_initial_state = _MODULE.make_initial_state
run_dummy_coupled = _MODULE.run_dummy_coupled


def _assert_shape_dtype_preserved(before, after):
    for left, right in zip(jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(after), strict=True):
        assert right.shape == left.shape
        assert right.dtype == left.dtype


def test_m6_physics_adapters_preserve_state_pytree_contract():
    grid = make_dummy_grid(8, 8, 8)
    state = make_initial_state(grid)

    for adapter in (thompson_adapter, mynn_adapter, surface_adapter, rrtmg_adapter):
        after = adapter(state, 1.0)
        block_until_ready(after)
        _assert_shape_dtype_preserved(state, after)
        assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(after))


def test_m6_dummy_coupled_small_scan_preserves_shape_dtype_and_finiteness():
    grid = make_dummy_grid(8, 8, 8)
    state = make_initial_state(grid)
    tendencies = Tendencies.zeros(grid)

    out = run_dummy_coupled(state, tendencies, grid, 1.0, 2, n_acoustic=1, debug=False)
    block_until_ready(out)

    _assert_shape_dtype_preserved(state, out)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(out))
