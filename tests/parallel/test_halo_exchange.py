from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.halo import HaloSpec, apply_halo
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.runtime.sharding import (
    ShardingConfig,
    merge_state_x,
    partition_state_x,
    x_partition_bounds,
)


def _deterministic_state(grid: GridSpec) -> State:
    fields = {}
    for index, (name, shape) in enumerate(_state_field_shapes(grid).items()):
        dtype = DEFAULT_DTYPES.dtype_for(name)
        size = 1
        for dim in shape:
            size *= int(dim)
        values = jnp.arange(size, dtype=jnp.float64).reshape(shape) + float(index * 1000)
        if dtype == jnp.int32:
            fields[name] = values.astype(jnp.int32)
        else:
            fields[name] = values.astype(dtype)
    return State(**fields)


def test_x_partition_bounds_requires_divisible_domain():
    assert x_partition_bounds(8, 4) == ((0, 2), (2, 4), (4, 6), (6, 8))

    with pytest.raises(ValueError, match="divisible"):
        x_partition_bounds(10, 4)


def test_partition_merge_state_x_roundtrip_with_halos():
    grid = GridSpec.canary_3km_template()
    state = _deterministic_state(grid)
    sharded = partition_state_x(state, grid, num_partitions=4, halo_width=2, fill_halos=True)
    merged = merge_state_x(sharded, grid, halo_width=2)

    for name in ("theta", "qv", "u", "v", "w", "mu", "lu_index", "u_bdy"):
        assert jnp.array_equal(getattr(merged, name), getattr(state, name)), name


@pytest.mark.skipif(len(jax.local_devices()) < 4, reason="requires fake or real 4-device mesh")
def test_ppermute_apply_halo_matches_periodic_state_partition():
    grid = GridSpec.canary_3km_template()
    state = _deterministic_state(grid)
    width = 2
    num_partitions = 4
    unfilled = partition_state_x(
        state,
        grid,
        num_partitions=num_partitions,
        halo_width=width,
        fill_halos=False,
    )
    expected = partition_state_x(
        state,
        grid,
        num_partitions=num_partitions,
        halo_width=width,
        fill_halos=True,
    )
    cfg = ShardingConfig(enabled=True, num_partitions=num_partitions, halo_width=width)
    spec = HaloSpec(
        width=width,
        fields_to_exchange=("theta", "u", "v", "w", "mu"),
        edge_type="periodic",
        sharding=cfg,
    )

    def local_exchange(local_state):
        return apply_halo(local_state, spec)

    haloed = jax.pmap(local_exchange, axis_name=cfg.axis_name)(unfilled)

    for name in spec.fields_to_exchange:
        assert jnp.array_equal(getattr(haloed, name), getattr(expected, name)), name


def test_apply_halo_stays_identity_without_sharding():
    grid = GridSpec.canary_3km_template()
    state = _deterministic_state(grid)
    spec = HaloSpec(width=2, fields_to_exchange=("theta",), edge_type="periodic")

    assert apply_halo(state, spec) is state
