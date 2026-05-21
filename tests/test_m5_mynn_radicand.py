from __future__ import annotations

import numpy as np
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp

from gpuwrf.physics.mynn_pbl import _flux_richardson


def test_flux_richardson_negative_radicand_matches_wrf_nan_boundary():
    ri = 1.5
    ri1 = 0.7
    ri2 = 0.1
    ri3 = 3.0
    ri4 = 2.0
    rfc = 0.2

    assert ri3 * ri3 - 4.0 * ri4 > 0.0
    assert ri * ri - ri3 * ri + ri4 < 0.0

    with np.errstate(invalid="ignore"):
        wrf_plain_sqrt = np.minimum(ri1 * (ri + ri2 - np.sqrt(ri * ri - ri3 * ri + ri4)), rfc)
    jax_value = _flux_richardson(
        jnp.asarray(ri, dtype=jnp.float64),
        jnp.asarray(ri1, dtype=jnp.float64),
        jnp.asarray(ri2, dtype=jnp.float64),
        jnp.asarray(ri3, dtype=jnp.float64),
        jnp.asarray(ri4, dtype=jnp.float64),
        jnp.asarray(rfc, dtype=jnp.float64),
    )

    assert np.isnan(wrf_plain_sqrt)
    assert np.isnan(float(np.asarray(jax_value)))
