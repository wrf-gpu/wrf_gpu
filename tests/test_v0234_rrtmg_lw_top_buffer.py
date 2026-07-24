"""WRF-source regressions for the RRTMG-LW above-model-top buffer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.rrtmg_lw import (
    RRTMGLWColumnState,
    _LW_BUFFER_PPROF,
    _LW_BUFFER_TPROF,
    _lw_buffer_layer_count,
    _lw_buffer_temperatures,
    _lw_extended_pressure_profiles,
    _temperature_interfaces,
)


def _state(*, top_pressure_pa: float | None) -> RRTMGLWColumnState:
    layer = jnp.asarray([[290.0, 260.0, 225.0, 215.0]], dtype=jnp.float64)
    pressure = jnp.asarray([[90000.0, 50000.0, 20000.0, 6000.0]], dtype=jnp.float64)
    zero = jnp.zeros_like(layer)
    surface = jnp.asarray([292.0], dtype=jnp.float64)
    return RRTMGLWColumnState(
        T=layer,
        p=pressure,
        qv=jnp.full_like(layer, 1.0e-3),
        qc=zero,
        qi=zero,
        qs=zero,
        qg=zero,
        cloud_fraction=zero,
        surface_temperature=surface,
        surface_emissivity=jnp.asarray([0.98], dtype=jnp.float64),
        dz=jnp.full_like(layer, 500.0),
        rho=jnp.full_like(layer, 0.8),
        top_pressure_pa=top_pressure_pa,
    )


def _wrf_temperature_reference(
    t_layer: np.ndarray,
    t_level: np.ndarray,
    pressure_interfaces_pa: np.ndarray,
    original_layers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Literal replay of module_ra_rrtmg_lw.F:12347-12379."""

    pprof = np.asarray(_LW_BUFFER_PPROF, dtype=np.float64)
    tprof = np.asarray(_LW_BUFFER_TPROF, dtype=np.float64)
    plev_mb = np.asarray(pressure_interfaces_pa, dtype=np.float64) * 0.01
    plev_mb[..., -1] = 0.0
    varint = np.empty_like(plev_mb)
    for index, pressure_mb in np.ndenumerate(plev_mb):
        below = np.flatnonzero(pprof < pressure_mb)
        if below.size:
            k0 = max(int(below[0]) - 1, 0)
        else:
            k0 = pprof.size - 1
        k1 = min(k0 + 1, pprof.size - 1)
        weight = 0.0 if k0 == k1 else (pressure_mb - pprof[k0]) / (pprof[k1] - pprof[k0])
        varint[index] = weight * (tprof[k1] - tprof[k0]) + tprof[k0]

    expected_level = np.asarray(t_level, dtype=np.float64).copy()
    offset = expected_level[..., original_layers - 1] - varint[..., original_layers - 1]
    expected_level[..., original_layers:] = varint[..., original_layers:] + offset[..., None]
    expected_layer = np.asarray(t_layer, dtype=np.float64).copy()
    for level in range(original_layers, expected_level.shape[-1]):
        expected_layer[..., level - 1] = 0.5 * (
            expected_level[..., level] + expected_level[..., level - 1]
        )
    return expected_layer, expected_level


def test_wrf_nint_and_4mb_pressure_ladder_at_50mb_top() -> None:
    # Fortran NINT(12.5) is 13, not Python round(12.5)==12.
    assert _lw_buffer_layer_count(5000.0) == 13
    p = jnp.asarray([[90000.0, 6000.0]], dtype=jnp.float64)
    model_interfaces, all_interfaces, buffer_layers = _lw_extended_pressure_profiles(
        p, 5000.0
    )

    expected_interfaces = np.concatenate(
        (np.asarray([5000.0]), np.arange(4600.0, 0.0, -400.0), np.asarray([0.0]))
    )
    np.testing.assert_array_equal(np.asarray(model_interfaces)[0, -1:], expected_interfaces[:1])
    np.testing.assert_array_equal(np.asarray(all_interfaces)[0, p.shape[-1] :], expected_interfaces)
    np.testing.assert_array_equal(
        np.asarray(buffer_layers)[0],
        0.5 * (expected_interfaces[:-1] + expected_interfaces[1:]),
    )


def test_wrf_top_buffer_temperature_profile_matches_literal_source_replay() -> None:
    state = _state(top_pressure_pa=5000.0)
    _, pressure_interfaces, buffer_pressure = _lw_extended_pressure_profiles(
        state.p, state.top_pressure_pa
    )
    original_layers = state.p.shape[-1]
    buffer_count = buffer_pressure.shape[-1]
    t_layer = jnp.concatenate(
        (state.T, jnp.repeat(state.T[..., -1:], buffer_count, axis=-1)), axis=-1
    )
    t_level = _temperature_interfaces(state.T)
    t_level = jnp.concatenate(
        (t_level, jnp.repeat(t_level[..., -1:], buffer_count, axis=-1)), axis=-1
    )

    actual_layer, actual_level = _lw_buffer_temperatures(
        t_layer, t_level, pressure_interfaces, original_layers
    )
    expected_layer, expected_level = _wrf_temperature_reference(
        np.asarray(t_layer),
        np.asarray(t_level),
        np.asarray(pressure_interfaces),
        original_layers,
    )
    np.testing.assert_allclose(np.asarray(actual_layer), expected_layer, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(actual_level), expected_level, rtol=0.0, atol=1.0e-12)


def test_top_pressure_is_static_pytree_metadata_and_legacy_constructor_survives() -> None:
    explicit = _state(top_pressure_pa=5000.0)
    leaves, treedef = jax.tree_util.tree_flatten(explicit)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert len(leaves) == 12
    assert rebuilt.top_pressure_pa == 5000.0
    assert explicit.replace(T=explicit.T).top_pressure_pa == 5000.0

    legacy = _state(top_pressure_pa=None)
    assert legacy.top_pressure_pa is None
    assert legacy != explicit


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_invalid_static_model_top_fails_loudly(bad: float) -> None:
    with pytest.raises(ValueError, match="top_pressure_pa"):
        _lw_buffer_layer_count(bad)
