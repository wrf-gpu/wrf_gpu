from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import State
from gpuwrf.coupling.boundary_apply import SIDE_INDEX, apply_lateral_boundaries, interpolate_boundary_leaf


def _boundary(grid: GridSpec, z_len: int, value: float, dtype=jnp.float32):
    side = max(grid.nx + 1, grid.ny + 1)
    return jnp.ones((2, 4, z_len, side), dtype=dtype) * value


def test_interpolate_boundary_leaf_uses_hourly_linear_replay():
    leaf = jnp.zeros((2, 4, 1, 3), dtype=jnp.float32)
    leaf = leaf.at[1].set(10.0)

    half_hour = interpolate_boundary_leaf(leaf, 1800.0)

    assert np.allclose(np.asarray(half_hour), 5.0)


def test_apply_lateral_boundaries_sets_specified_zone_and_relaxes_inner_zone():
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid).replace(
        theta=jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float32),
        qv=jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float32),
        u=jnp.zeros((grid.nz, grid.ny, grid.nx + 1), dtype=jnp.float32),
        v=jnp.zeros((grid.nz, grid.ny + 1, grid.nx), dtype=jnp.float32),
        ph=jnp.zeros((grid.nz + 1, grid.ny, grid.nx), dtype=jnp.float64),
        mu=jnp.zeros((grid.ny, grid.nx), dtype=jnp.float64),
        theta_bdy=_boundary(grid, grid.nz, 10.0),
        qv_bdy=_boundary(grid, grid.nz, 0.002),
        u_bdy=_boundary(grid, grid.nz, 3.0),
        v_bdy=_boundary(grid, grid.nz, 4.0),
        ph_bdy=_boundary(grid, grid.nz + 1, 5.0, dtype=jnp.float64),
        mu_bdy=_boundary(grid, 1, 6.0, dtype=jnp.float64),
    )

    out = apply_lateral_boundaries(state, 0.0, 60.0)
    jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf, out)

    assert np.allclose(np.asarray(out.theta[:, 1:-1, 0]), 10.0)
    assert np.allclose(np.asarray(out.theta[:, 3:5, 1]), 1.2)
    assert np.allclose(np.asarray(out.qv[:, 1:-1, 0]), 0.002)
    assert np.all(np.asarray(out.qv) >= 0.0)
    assert np.allclose(np.asarray(out.mu[1:-1, 0]), 6.0)
    assert np.allclose(np.asarray(out.ph[:, 1:-1, 0]), 5.0)


def test_boundary_side_order_matches_west_east_south_north_contract():
    assert SIDE_INDEX == {"W": 0, "E": 1, "S": 2, "N": 3}
