from __future__ import annotations

import inspect

import jax.numpy as jnp
import numpy as np

from gpuwrf.physics import rrtmg_lw, rrtmg_sw


def test_sw_delta_scaling_matches_joseph_wiscombe_formula():
    tau = jnp.asarray([2.0], dtype=jnp.float64)
    omega = jnp.asarray([0.8], dtype=jnp.float64)
    asymmetry = jnp.asarray([0.6], dtype=jnp.float64)

    tau_p, omega_p, g_p = rrtmg_sw._delta_scale(tau, omega, asymmetry)

    f = 0.6**2
    np.testing.assert_allclose(np.asarray(tau_p), [(1.0 - f * 0.8) * 2.0])
    np.testing.assert_allclose(np.asarray(omega_p), [((1.0 - f) * 0.8) / (1.0 - f * 0.8)])
    np.testing.assert_allclose(np.asarray(g_p), [(0.6 - f) / (1.0 - f)])


def test_sw_reftra_transparent_layer_is_identity():
    tau = jnp.ones((1, 1, 1, 1), dtype=jnp.float64)
    omega = jnp.zeros_like(tau)
    asymmetry = jnp.zeros_like(tau)
    active = jnp.zeros_like(tau)
    mu0 = jnp.asarray([0.7], dtype=jnp.float64)

    direct_ref, diffuse_ref, direct_trans, diffuse_trans = rrtmg_sw._reftra_eddington(tau, omega, asymmetry, mu0, active)

    np.testing.assert_allclose(np.asarray(direct_ref), 0.0)
    np.testing.assert_allclose(np.asarray(diffuse_ref), 0.0)
    np.testing.assert_allclose(np.asarray(direct_trans), 1.0)
    np.testing.assert_allclose(np.asarray(diffuse_trans), 1.0)


def test_sw_kernel_no_longer_uses_fabricated_log_tau_curve():
    source = inspect.getsource(rrtmg_sw)
    assert "log1p" not in source
    assert "vapor_path" not in source


def test_lw_diffusivity_angles_match_rrtmg_bounds():
    dry = rrtmg_lw._lw_diffusivity(jnp.asarray([0.0], dtype=jnp.float64))
    moist = rrtmg_lw._lw_diffusivity(jnp.asarray([10.0], dtype=jnp.float64))

    assert dry.shape == (1, 16)
    assert moist.shape == (1, 16)
    np.testing.assert_allclose(np.asarray(dry[0, [0, 3, 9, 15]]), 1.66)
    assert float(jnp.min(dry[:, 1:9])) >= 1.50
    assert float(jnp.max(dry[:, 1:9])) <= 1.80
    assert float(jnp.min(moist[:, 1:9])) >= 1.50
    assert float(jnp.max(moist[:, 1:9])) <= 1.80
