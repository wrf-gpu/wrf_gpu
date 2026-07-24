"""Focused WRF-source regression for the Noah-MP humidity handoff."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.noahmp_coupler import (
    _mynn_pbl_surface_density,
    assemble_noahmp_forcing,
)
from gpuwrf.physics.surface_constants import P608, R_D


jax.config.update("jax_enable_x64", True)


class _State:
    pass


class _Radiation:
    soldn = jnp.zeros((1, 3), dtype=jnp.float64)
    lwdn = jnp.full((1, 3), 300.0, dtype=jnp.float64)
    cosz = jnp.zeros((1, 3), dtype=jnp.float64)


class _Clock:
    julian = 60.0
    yearlen = 365.0


def test_wrf_noahmp_qvapor_mixing_ratio_becomes_specific_humidity():
    """Match module_sf_noahmpdrv.F ``Q_ML=QV3D/(1+QV3D)`` exactly."""

    qv = np.asarray([[0.0125, 0.004, -1.0e-6]], dtype=np.float64)
    state = _State()
    state.qv = jnp.asarray(qv[..., None])
    state.t_air = jnp.asarray([[299.25, 281.75, 290.0]], dtype=jnp.float64)
    state.p = jnp.asarray([[95000.0, 91234.0, 98000.0]], dtype=jnp.float64)
    state.psfc = jnp.asarray([[101325.0, 92000.0, 99000.0]], dtype=jnp.float64)
    state.u = jnp.zeros((1, 3, 1), dtype=jnp.float64)
    state.v = jnp.zeros((1, 3, 1), dtype=jnp.float64)
    state.dz = jnp.full((1, 3, 1), 60.0, dtype=jnp.float64)

    forcing = assemble_noahmp_forcing(state, None, _Radiation(), _Clock(), 6.0)
    expected_specific = qv / (1.0 + qv)

    np.testing.assert_array_equal(np.asarray(forcing.qair), expected_specific)
    assert not np.array_equal(np.asarray(forcing.qair), qv)
    # The negative sample proves the implementation did not add a non-WRF clamp.
    assert float(np.asarray(forcing.qair)[0, 2]) < 0.0


def test_mynn_density_keeps_qvapor_mixing_ratio_after_noahmp_conversion():
    """Keep MYNN line 3960 distinct from the Noah-MP specific-humidity seam."""

    qv = np.asarray([[0.0125, 0.004]], dtype=np.float64)

    class _Forcing:
        psfc = jnp.asarray([[101325.0, 91234.0]], dtype=jnp.float64)
        sfctmp = jnp.asarray([[299.25, 281.75]], dtype=jnp.float64)
        qair = jnp.asarray(qv / (1.0 + qv), dtype=jnp.float64)

    expected = np.asarray(_Forcing.psfc) / (
        R_D * (np.asarray(_Forcing.sfctmp) + P608 * qv)
    )
    actual = np.asarray(_mynn_pbl_surface_density(_Forcing(), qv))
    fallback = np.asarray(_mynn_pbl_surface_density(_Forcing()))

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(fallback, expected, rtol=0.0, atol=3.0e-16)
