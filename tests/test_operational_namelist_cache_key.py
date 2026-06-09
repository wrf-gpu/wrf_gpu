import dataclasses

import jax
import jax.numpy as jnp
import pytest

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import (
    DEFAULT_ACOUSTIC_PRECISION_MODE,
    AcousticPrecisionMode,
    acoustic_precision_mode_label,
)
from gpuwrf.contracts.state import Tendencies
from gpuwrf.runtime.operational_mode import OperationalNamelist
from gpuwrf.runtime.operational_mode import _StaticHolder


def _cpu_tendencies(grid: GridSpec) -> Tendencies:
    nz, ny, nx = grid.nz, grid.ny, grid.nx

    def z(shape):
        return jnp.zeros(shape, dtype=jnp.float64)

    return Tendencies(
        z((nz, ny, nx + 1)),
        z((nz, ny + 1, nx)),
        z((nz + 1, ny, nx)),
        z((nz, ny, nx)),
        z((nz, ny, nx)),
        z((nz, ny, nx)),
        z((nz + 1, ny, nx)),
        z((ny, nx)),
    )


def test_static_holder_none_hash_is_stable_across_flatten_rebuilds():
    """Disabled static bundles must not fragment JIT cache keys."""

    first = _StaticHolder(None)
    second = _StaticHolder(None)

    assert first == second
    assert hash(first) == hash(second)


def test_static_holder_real_bundle_hashes_by_identity():
    bundle = object()
    same = _StaticHolder(bundle)
    again = _StaticHolder(bundle)
    different = _StaticHolder(object())

    assert same == again
    assert hash(same) == hash(again)
    assert same != different


def test_acoustic_precision_mode_default_label_roundtrips_as_static_aux():
    grid = GridSpec.canary_3km_template()
    nml = OperationalNamelist.from_grid(grid, tendencies=_cpu_tendencies(grid))

    assert nml.acoustic_precision_mode == DEFAULT_ACOUSTIC_PRECISION_MODE
    assert acoustic_precision_mode_label(None) == DEFAULT_ACOUSTIC_PRECISION_MODE

    leaves, treedef = jax.tree_util.tree_flatten(nml)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert rebuilt.acoustic_precision_mode == DEFAULT_ACOUSTIC_PRECISION_MODE


def test_mixed_perturb_fp32_mode_is_static_cache_key_only():
    grid = GridSpec.canary_3km_template()
    nml = OperationalNamelist.from_grid(grid, tendencies=_cpu_tendencies(grid))
    mixed = dataclasses.replace(
        nml,
        acoustic_precision_mode=AcousticPrecisionMode.MIXED_PERTURB_FP32,
    )

    default_leaves, default_treedef = jax.tree_util.tree_flatten(nml)
    mixed_leaves, mixed_treedef = jax.tree_util.tree_flatten(mixed)

    assert mixed.acoustic_precision_mode == "mixed_perturb_fp32"
    assert mixed_treedef != default_treedef
    assert len(mixed_leaves) == len(default_leaves)
    assert all(
        getattr(left, "shape", None) == getattr(right, "shape", None)
        and getattr(left, "dtype", None) == getattr(right, "dtype", None)
        for left, right in zip(default_leaves, mixed_leaves, strict=True)
    )


def test_acoustic_precision_mode_invalid_value_fails_closed():
    grid = GridSpec.canary_3km_template()
    nml = OperationalNamelist.from_grid(grid, tendencies=_cpu_tendencies(grid))

    with pytest.raises(ValueError, match="unsupported acoustic_precision_mode"):
        dataclasses.replace(nml, acoustic_precision_mode="global_fp32")
