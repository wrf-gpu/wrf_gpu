"""Backward compatibility and source gates for WRF RRTMG-SW P8W/T8W leaves."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.rrtmg_sw import (
    RRTMGSWColumnState,
    _extend_with_wrf_top_layer,
    _flatten_sw_state,
    _pad_sw_state,
    _pressure_interfaces,
    _slice_sw_state,
    _sw_extended_profiles,
)


def _state(*, explicit_interfaces: bool) -> RRTMGSWColumnState:
    temperature = jnp.asarray(
        [[[291.0, 267.0, 238.0], [292.0, 268.0, 239.0]]], dtype=jnp.float64
    )
    pressure = jnp.asarray(
        [[[90000.0, 51000.0, 17000.0], [91000.0, 52000.0, 18000.0]]],
        dtype=jnp.float64,
    )
    zero = jnp.zeros_like(temperature)
    if explicit_interfaces:
        pressure_interfaces = _pressure_interfaces(pressure).at[..., -1].set(5000.0)
        temperature_interfaces = jnp.concatenate(
            (
                temperature[..., :1] + 2.0,
                0.5 * (temperature[..., :-1] + temperature[..., 1:]),
                temperature[..., -1:] - 3.0,
            ),
            axis=-1,
        )
    else:
        pressure_interfaces = None
        temperature_interfaces = None
    return RRTMGSWColumnState(
        temperature,
        pressure,
        jnp.full_like(temperature, 1.0e-3),
        zero,
        zero,
        zero,
        zero,
        zero,
        jnp.asarray([[0.12, 0.18]], dtype=jnp.float64),
        jnp.asarray([[0.35, 0.42]], dtype=jnp.float64),
        jnp.full_like(temperature, 500.0),
        jnp.full_like(temperature, 0.8),
        0.99,
        pressure_interfaces=pressure_interfaces,
        temperature_interfaces=temperature_interfaces,
    )


def test_legacy_positional_constructor_and_none_profile_are_unchanged() -> None:
    legacy = _state(explicit_interfaces=False)
    assert legacy.pressure_interfaces is None
    assert legacy.temperature_interfaces is None

    old_original = _pressure_interfaces(legacy.p)
    old_top_pressure = 0.5 * old_original[..., -1:]
    old_extended = jnp.concatenate(
        (old_original, jnp.full_like(old_top_pressure, 1.0e-3)), axis=-1
    )
    old_p_ext = jnp.concatenate((legacy.p, old_top_pressure), axis=-1)
    old_t_ext = _extend_with_wrf_top_layer(legacy.T)
    actual = _sw_extended_profiles(legacy)
    for candidate, reference in zip(
        actual, (old_original, old_extended, old_p_ext, old_t_ext), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(candidate), np.asarray(reference))

    leaves, treedef = jax.tree_util.tree_flatten(legacy)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    # Optional None nodes add no dynamic leaves, preserving the historical
    # 13-array pytree and positional solar_source_scale call.
    assert len(leaves) == 13
    assert rebuilt == legacy


def test_explicit_wrf_interfaces_drive_top_layer_and_survive_tiling() -> None:
    explicit = _state(explicit_interfaces=True)
    leaves, treedef = jax.tree_util.tree_flatten(explicit)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert len(leaves) == 15
    assert rebuilt == explicit

    model_interfaces, extended, p_ext, t_ext = _sw_extended_profiles(explicit)
    np.testing.assert_array_equal(
        np.asarray(model_interfaces), np.asarray(explicit.pressure_interfaces)
    )
    np.testing.assert_array_equal(
        np.asarray(extended[..., :-1]), np.asarray(explicit.pressure_interfaces)
    )
    np.testing.assert_array_equal(
        np.asarray(p_ext[..., -1]),
        0.5 * np.asarray(explicit.pressure_interfaces[..., -1]),
    )
    # WRF's extra SW layer uses the model-top T8W, not the top mass-level T3D.
    np.testing.assert_array_equal(
        np.asarray(t_ext[..., -1]),
        np.asarray(explicit.temperature_interfaces[..., -1]),
    )

    flat = _flatten_sw_state(explicit, (1, 2), 2)
    padded = _pad_sw_state(flat, 2, 4)
    sliced = _slice_sw_state(padded, jnp.asarray(0, dtype=jnp.int32), 2, 4)
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
