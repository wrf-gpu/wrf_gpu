from __future__ import annotations

import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp

from gpuwrf.debug.asserts import assert_finite, assert_physical_bounds
from gpuwrf.debug.snapshots import dump_snapshots, snapshot
from gpuwrf.profiling.budget import compiled_text
from gpuwrf.validation.tier2 import density_current_state, make_ideal_grid


def test_disabled_assert_finite_returns_same_object():
    x = jnp.ones((4,), dtype=jnp.float64)
    assert assert_finite(x, "x", enabled=False) is x


def test_disabled_assert_bounds_returns_same_object():
    x = jnp.ones((4,), dtype=jnp.float64)
    assert assert_physical_bounds(x, 0.0, 2.0, "x", enabled=False) is x


def test_disabled_snapshot_returns_same_state():
    grid = make_ideal_grid(3, 4, 4)
    state, _ = density_current_state(grid)
    assert snapshot(state, "stage", enabled=False) is state
    assert isinstance(dump_snapshots(), dict)


def test_disabled_assert_does_not_emit_finiteness_hlo():
    def with_disabled(x):
        return assert_finite(x, "x", enabled=False)

    def stripped(x):
        return x

    x = jnp.ones((4,), dtype=jnp.float64)
    left = compiled_text(jax.jit(with_disabled).lower(x).compile()).lower()
    right = compiled_text(jax.jit(stripped).lower(x).compile()).lower()
    assert "is-finite" not in left
    assert "compare" not in left
    assert left.count("copy") == right.count("copy")


def test_hlo_diff_artifact_is_empty_when_present():
    path = Path("artifacts/m4/hlo_dump/dycore_step_debug_vs_stripped.diff")
    if not path.exists():
        return
    assert path.stat().st_size == 0
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
