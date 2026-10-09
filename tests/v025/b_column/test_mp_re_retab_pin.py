"""MO18 critic pin (proposed for tests/v025/b_column): WRF RRTMG LW caller T-dependent ice fallback.

Pristine phys/module_ra_rrtmg_lw.F (has_reqi branch, ~:12199-12210):
    reice = MAX(5., re_ice*1.E6)
    if (reice .LE. 5. .AND. cldfra3d .gt. 0.) then
       idx_rei = int(t3d-179.); idx_rei = min(max(idx_rei,1),75); corr = t3d - int(t3d)
       reice = retab(idx_rei)*(1.-corr) + retab(idx_rei+1)*corr; reice = MAX(reice, 5.0)
The expected values come from the LITERAL data statement (~:11544), independent of the port's
run-time parser, so deleting the interpolation, the clamp or the cf gate fails this test (E39/E98).
"""
import numpy as np
import jax.numpy as jnp
import pytest

from gpuwrf.physics import rrtmg_mp_re as R

# retab(1..95) copied verbatim from the pristine data statement.
RETAB = np.array([
    5.92779, 6.26422, 6.61973, 6.99539, 7.39234,
    7.81177, 8.25496, 8.72323, 9.21800, 9.74075, 10.2930,
    10.8765, 11.4929, 12.1440, 12.8317, 13.5581, 14.2319,
    15.0351, 15.8799, 16.7674, 17.6986, 18.6744, 19.6955,
    20.7623, 21.8757, 23.0364, 24.2452, 25.5034, 26.8125,
    27.7895, 28.6450, 29.4167, 30.1088, 30.7306, 31.2943,
    31.8151, 32.3077, 32.7870, 33.2657, 33.7540, 34.2601,
    34.7892, 35.3442, 35.9255, 36.5316, 37.1602, 37.8078,
    38.4720, 39.1508, 39.8442, 40.5552, 41.2912, 42.0635,
    42.8876, 43.7863, 44.7853, 45.9170, 47.2165, 48.7221,
    50.4710, 52.4980, 54.8315, 57.4898, 60.4785, 63.7898,
    65.5604, 71.2885, 75.4113, 79.7368, 84.2351, 88.8833,
    93.6658, 98.5739, 103.603, 108.752, 114.025, 119.424,
    124.954, 130.630, 136.457, 142.446, 148.608, 154.956,
    161.503, 168.262, 175.248, 182.473, 189.952, 197.699,
    205.728, 214.055, 222.694, 231.661, 240.971, 250.639], np.float32)


def _wrf_reice(t):
    t = np.float32(t)
    idx = min(max(int(t - np.float32(179.)), 1), 75)  # Fortran 1-based
    corr = np.float32(t - np.float32(int(t)))
    r = RETAB[idx - 1] * (np.float32(1.) - corr) + RETAB[idx] * corr
    return np.float32(max(r, np.float32(5.0)))


# 230.4/250.7: interior interpolation; 170.0/179.9: low clamp idx 1; 260.5/273.9: high clamp idx 75
# (corr still from T, a WRF quirk); 215.0: corr == 0 exactly.
TEMPS = (230.4, 250.7, 170.0, 179.9, 260.5, 273.9, 215.0)


def test_retab_count_and_parser_match_literal_table():
    assert RETAB.size == 95
    np.testing.assert_array_equal(np.asarray(R._retab()), RETAB)


@pytest.mark.parametrize("re_ice_m", [0.0, 4.99e-6])
def test_ice_fallback_is_wrf_retab_interpolation(re_ice_m):
    n = len(TEMPS)
    T = jnp.asarray([TEMPS], jnp.float32)
    ones = jnp.ones((1, n), jnp.float32)
    r = R.prepare_radiation_radii(T, ones, 2. * ones, 1e-5 * ones, re_ice_m * ones, 2e-5 * ones)
    expected = np.array([[_wrf_reice(t) for t in TEMPS]], np.float32)
    got = np.asarray(r.ice_um)
    assert np.isfinite(got).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(got, expected, rtol=2e-6, atol=0)
    # no-interpolation mutant (retab(idx) only) differs by >= 0.07 um at the interior points
    assert abs(float(got[0, 0]) - float(RETAB[50])) > 0.2


def test_ice_fallback_needs_cloud_and_small_radius():
    T = jnp.asarray([[230.4, 230.4, 230.4]], jnp.float32)
    cf = jnp.asarray([[0., 1., 1.]], jnp.float32)
    re_ice = jnp.asarray([[0., 5.01e-6, 60e-6]], jnp.float32)
    one = jnp.ones_like(T)
    r = np.asarray(R.prepare_radiation_radii(T, cf, 2. * one, 1e-5 * one, re_ice, 2e-5 * one).ice_um)
    assert np.isfinite(r).all()
    np.testing.assert_allclose(r, [[5., 5.01, 60.]], rtol=2e-6, atol=0)
