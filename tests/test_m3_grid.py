from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.contracts.grid import GridSpec


def test_gridspec_pytree_round_trip_and_hashable():
    grid = GridSpec.canary_3km_template()

    leaves, treedef = jax.tree_util.tree_flatten(grid)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert rebuilt.projection.kind == "lambert"
    assert rebuilt.bc.source in {"AIFS", "GFS", "ERA5", "ideal"}
    assert rebuilt.staggering == "c-grid"
    assert rebuilt.terrain_height.shape == (grid.ny, grid.nx)
    assert rebuilt == grid
    assert hash(grid) == hash(rebuilt)


def test_canary_3km_template_metadata():
    grid = GridSpec.canary_3km_template()

    assert grid.projection.dx_m == 3000.0
    assert grid.projection.dy_m == 3000.0
    assert grid.halo_width == 2
    assert grid.bc.fields == ("u", "v", "T", "qv", "p_s")
    assert grid.vertical.eta_levels.dtype == jnp.float64


def test_jit_cache_hit_on_equivalent_grids():
    grid1 = GridSpec.canary_3km_template()
    grid2 = GridSpec.canary_3km_template()

    assert grid1 == grid2
    assert hash(grid1) == hash(grid2)

    @jax.jit(static_argnames=("grid",))
    def grid_shape_value(grid: GridSpec):
        return grid.nx + grid.ny + grid.nz

    assert int(grid_shape_value(grid1)) == 26
    assert int(grid_shape_value(grid2)) == 26


def test_invalid_halo_width_rejected_by_gridspec():
    grid = GridSpec.canary_3km_template()

    try:
        GridSpec(
            grid.projection,
            grid.terrain,
            grid.vertical,
            grid.bc,
            grid.eta_levels,
            grid.terrain_height,
            halo_width=5,
        )
    except ValueError as exc:
        assert "halo_width" in str(exc)
    else:
        raise AssertionError("GridSpec accepted invalid halo width")
