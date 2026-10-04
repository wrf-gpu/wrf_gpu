"""Precision is an explicit contract; tree transforms must preserve it."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.grid import DycoreMetrics, GridSpec


def test_explicit_fp32_grid_roundtrip_and_jit():
    grid = GridSpec.canary_3km_template()
    metrics = replace(grid.metrics, precision="fp32")
    grid32 = replace(grid, eta_levels=grid.eta_levels.astype(jnp.float32),
                     terrain_height=grid.terrain_height.astype(jnp.float32),
                     metrics=metrics)
    leaves, tree = jax.tree_util.tree_flatten(grid32)
    assert all(x.dtype == jnp.float32 for x in leaves)
    rebuilt = jax.tree_util.tree_unflatten(tree, leaves)
    assert rebuilt == grid32
    assert hash(rebuilt) == hash(grid32)
    result = jax.jit(lambda g: g.metrics.msftx * g.metrics.c1h[0])(grid32)
    assert result.dtype == jnp.float32
    assert grid32.vertical.eta_levels is grid32.eta_levels


def test_fp64_remains_default_and_preserves_values():
    grid = GridSpec.canary_3km_template()
    metrics = replace(grid.metrics)
    assert metrics.precision == "fp64"
    for name in metrics._array_names():
        assert getattr(metrics, name) is getattr(grid.metrics, name)
    rebuilt = jax.tree_util.tree_map(lambda x: x, grid)
    for old, new in zip(jax.tree_util.tree_leaves(grid),
                        jax.tree_util.tree_leaves(rebuilt), strict=True):
        np.testing.assert_array_equal(old, new, strict=True)


def test_mixed_grid_precision_rejected():
    grid = GridSpec.canary_3km_template()
    with pytest.raises(TypeError, match="metrics must match"):
        replace(grid, eta_levels=grid.eta_levels.astype(jnp.float32),
                terrain_height=grid.terrain_height.astype(jnp.float32))
    with pytest.raises(ValueError, match="unsupported grid precision"):
        replace(grid.metrics, precision="fp16")


def test_fp32_flat_metrics_with_x64_enabled():
    metrics = DycoreMetrics.flat(ny=4, nx=5, nz=4,
                                eta_levels=jnp.linspace(1., 0., 5),
                                top_pressure_pa=5000., precision="fp32")
    assert all(getattr(metrics, name).dtype == jnp.float32
               for name in metrics._array_names())
    assert metrics.dnw[0] < 0
