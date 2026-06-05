from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "")

import numpy as np
import jax.numpy as jnp

from gpuwrf.physics.mynn_constants import QKEMIN
from gpuwrf.physics.mynn_pbl import _wrf_qke_minmax


def test_wrf_qke_minmax_matches_finite_bounds_and_suppresses_nan():
    raw = jnp.asarray([np.nan, -1.0, QKEMIN * 0.1, 0.25, 200.0], dtype=jnp.float64)
    bounded = np.asarray(_wrf_qke_minmax(raw))

    assert np.all(np.isfinite(bounded))
    assert bounded[0] == QKEMIN
    assert bounded[1] == QKEMIN
    assert bounded[2] == QKEMIN
    assert bounded[3] == 0.25
    assert bounded[4] == 150.0


def test_wrf_qke_minmax_is_unchanged_for_in_range_finite_values():
    raw = jnp.asarray([QKEMIN, 1.0e-3, 1.0, 149.0], dtype=jnp.float64)
    bounded = np.asarray(_wrf_qke_minmax(raw))

    assert np.array_equal(bounded, np.asarray(raw))
