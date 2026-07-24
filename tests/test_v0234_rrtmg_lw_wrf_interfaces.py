"""Backward compatibility and source gates for WRF RRTMG-LW P8W/T8W leaves."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.rrtmg_lw import (
    RRTMGLWColumnState,
    _flatten_lw_state,
    _lw_extended_pressure_profiles,
    _pad_lw_state,
    _pressure_interfaces,
    _slice_lw_state,
    _temperature_interfaces,
)


def _state(*, explicit_interfaces: bool) -> RRTMGLWColumnState:
    temperature = jnp.asarray(
        [[[291.0, 267.0, 238.0], [292.0, 268.0, 239.0]]], dtype=jnp.float64
    )
    pressure = jnp.asarray(
        [[[90000.0, 51000.0, 17000.0], [91000.0, 52000.0, 18000.0]]],
        dtype=jnp.float64,
    )
    zero = jnp.zeros_like(temperature)
    pressure_interfaces = _pressure_interfaces(pressure) if explicit_interfaces else None
    temperature_interfaces = (
        _temperature_interfaces(temperature) if explicit_interfaces else None
    )
    return RRTMGLWColumnState(
        temperature,
        pressure,
        jnp.full_like(temperature, 1.0e-3),
        zero,
        zero,
        zero,
        zero,
        zero,
        jnp.asarray([[292.0, 293.0]], dtype=jnp.float64),
        jnp.asarray([[0.97, 0.98]], dtype=jnp.float64),
        jnp.full_like(temperature, 500.0),
        jnp.full_like(temperature, 0.8),
        5000.0,
        pressure_interfaces=pressure_interfaces,
        temperature_interfaces=temperature_interfaces,
    )


def test_legacy_positional_constructor_and_none_fallback_are_unchanged() -> None:
    legacy = _state(explicit_interfaces=False)
    assert legacy.pressure_interfaces is None
    assert legacy.temperature_interfaces is None
    assert legacy.top_pressure_pa == 5000.0

    old_call, old_extended, old_buffer = _lw_extended_pressure_profiles(
        legacy.p, legacy.top_pressure_pa
    )
    explicit_none, none_extended, none_buffer = _lw_extended_pressure_profiles(
        legacy.p, legacy.top_pressure_pa, None
    )
    np.testing.assert_array_equal(np.asarray(old_call), np.asarray(explicit_none))
    np.testing.assert_array_equal(np.asarray(old_extended), np.asarray(none_extended))
    np.testing.assert_array_equal(np.asarray(old_buffer), np.asarray(none_buffer))

    leaves, treedef = jax.tree_util.tree_flatten(legacy)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    # Optional None nodes add no dynamic JAX leaves; the historical 12-array
    # pytree and positional top_pressure_pa call remain intact.
    assert len(leaves) == 12
    assert rebuilt == legacy


def test_explicit_wrf_interfaces_are_dynamic_pytree_leaves_and_survive_tiling() -> None:
    explicit = _state(explicit_interfaces=True)
    leaves, treedef = jax.tree_util.tree_flatten(explicit)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert len(leaves) == 14
    assert rebuilt == explicit

    model_interfaces, _extended, _buffer = _lw_extended_pressure_profiles(
        explicit.p,
        explicit.top_pressure_pa,
        explicit.pressure_interfaces,
    )
    expected = np.asarray(explicit.pressure_interfaces).copy()
    expected[..., -1] = 5000.0
    np.testing.assert_array_equal(np.asarray(model_interfaces), expected)

    flat = _flatten_lw_state(explicit, (1, 2), 2)
    padded = _pad_lw_state(flat, 2, 4)
    sliced = _slice_lw_state(padded, jnp.asarray(0, dtype=jnp.int32), 2, 4)
    np.testing.assert_array_equal(
        np.asarray(sliced.pressure_interfaces),
        np.reshape(np.asarray(explicit.pressure_interfaces), (2, 4)),
    )
    np.testing.assert_array_equal(
        np.asarray(sliced.temperature_interfaces),
        np.reshape(np.asarray(explicit.temperature_interfaces), (2, 4)),
    )


def test_interface_leaves_must_be_paired_and_have_nz_plus_one_shape() -> None:
    legacy = _state(explicit_interfaces=False)
    good = _pressure_interfaces(legacy.p)
    with pytest.raises(ValueError, match="must be supplied together"):
        legacy.replace(pressure_interfaces=good)
    with pytest.raises(ValueError, match="column-interface shape"):
        legacy.replace(
            pressure_interfaces=legacy.p,
            temperature_interfaces=legacy.T,
        )
