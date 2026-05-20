from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.thompson_saturation import saturation_mixing_ratio_ice, saturation_mixing_ratio_liquid


def test_saturation_liquid_matches_wrf_polynomial_reference():
    p = jnp.asarray([100000.0], dtype=jnp.float64)
    T = jnp.asarray([273.16], dtype=jnp.float64)
    qvs = np.asarray(saturation_mixing_ratio_liquid(p, T))[0]
    assert abs(qvs - 0.003827458721406073) < 1.0e-12


def test_saturation_ice_is_below_liquid_for_cold_temperature():
    p = jnp.asarray([70000.0], dtype=jnp.float64)
    T = jnp.asarray([253.15], dtype=jnp.float64)
    qvs_liq = np.asarray(saturation_mixing_ratio_liquid(p, T))[0]
    qvs_ice = np.asarray(saturation_mixing_ratio_ice(p, T))[0]
    assert qvs_ice < qvs_liq
    assert qvs_ice > 0.0
