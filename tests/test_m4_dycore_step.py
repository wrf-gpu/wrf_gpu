from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.dynamics.step import run, step
from gpuwrf.profiling.transfer_audit import block_until_ready
from gpuwrf.validation.tier2 import density_current_state, make_ideal_grid


def test_step_preserves_pytree_shape_and_dtype():
    grid = make_ideal_grid(4, 6, 6)
    state, tendencies = density_current_state(grid)
    out = step(state, tendencies, grid, 0.25, n_acoustic=2, debug=False)
    block_until_ready(out)
    for before, after in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(out), strict=True):
        assert after.shape == before.shape
        assert after.dtype == before.dtype


def test_run_preserves_no_nan_inf_on_small_density_current():
    grid = make_ideal_grid(4, 6, 6)
    state, tendencies = density_current_state(grid)
    out = run(state, tendencies, grid, 0.25, 3, n_acoustic=2, debug=False)
    block_until_ready(out)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(out))


def test_public_debug_true_path_compiles_on_small_case():
    grid = make_ideal_grid(3, 5, 5)
    state, tendencies = density_current_state(grid)
    out = step(state, tendencies, grid, 0.1, n_acoustic=1, debug=True)
    block_until_ready(out)
    assert bool(jnp.all(jnp.isfinite(out.theta)))
