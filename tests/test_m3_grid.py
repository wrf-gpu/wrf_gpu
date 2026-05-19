from __future__ import annotations

import jax

from gpuwrf.contracts.grid import GridSpec


def test_gridspec_pytree_round_trip_and_hashable():
    grid = GridSpec.canary_3km_template()

    leaves, treedef = jax.tree_util.tree_flatten(grid)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert rebuilt.projection.kind == "lambert"
    assert rebuilt.bc.source in {"AIFS", "GFS", "ERA5", "ideal"}
    assert rebuilt.staggering == "c-grid"
    assert rebuilt.terrain_height.shape == (grid.ny, grid.nx)
    assert hash(grid) == hash(rebuilt)


def test_canary_3km_template_metadata():
    grid = GridSpec.canary_3km_template()

    assert grid.projection.dx_m == 3000.0
    assert grid.projection.dy_m == 3000.0
    assert grid.halo_width == 2
    assert grid.bc.fields == ("u", "v", "T", "qv", "p_s")


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
