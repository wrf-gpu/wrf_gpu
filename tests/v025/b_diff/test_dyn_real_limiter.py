"""GPUWRF_DYN_REAL_ALL theta limiter: REAL arithmetic, exact pass-through, conserved mass."""
import numpy as np
import jax.numpy as jnp
import pytest

from gpuwrf.runtime import operational_mode as op


def _fields(seed, n_bad):
    rng = np.random.default_rng(seed)
    shape = (8, 12, 14)
    origin = 290.0 + 20.0 * rng.random(shape)
    candidate = origin + rng.normal(0.0, 0.5, shape)
    mass = 1.0e4 * (0.5 + rng.random(shape))
    flat = candidate.reshape(-1)
    idx = rng.choice(flat.size, n_bad, replace=False)
    flat[idx[: n_bad // 2]] = -50.0        # below the floor
    flat[idx[n_bad // 2:]] = 5000.0        # above the cap
    return (jnp.asarray(candidate, jnp.float32), jnp.asarray(origin, jnp.float32), jnp.asarray(mass, jnp.float32))


def test_real_limiter_is_exact_when_nothing_is_limited():
    cand, orig, mass = _fields(0, 0)
    out, diag = op._positive_definite_theta_increment_limiter_real(
        cand, orig, mass, minimum_k=op._THETA_LIMITER_MIN_K, maximum_k=op._THETA_LIMITER_MAX_K,
        lower_bound=None, upper_bound=None)
    assert out.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(out), np.asarray(cand))
    assert int(diag["theta_limited_cell_count"]) == 0
    assert float(diag["theta_mass_residual"]) == 0.0


@pytest.mark.parametrize("seed", [1, 2])
def test_real_limiter_conserves_mass_and_matches_fp64_counts(seed):
    cand, orig, mass = _fields(seed, 6)
    kw = dict(minimum_k=op._THETA_LIMITER_MIN_K, maximum_k=op._THETA_LIMITER_MAX_K, lower_bound=None, upper_bound=None)
    out, diag = op._positive_definite_theta_increment_limiter_real(cand, orig, mass, **kw)
    ref, rdiag = op._positive_definite_theta_increment_limiter(cand, orig, mass, **{k: v for k, v in kw.items()})
    assert int(diag["theta_limited_cell_count"]) == int(rdiag["theta_limited_cell_count"]) == 6
    o = np.asarray(out, np.float64)
    assert np.all(o >= op._THETA_LIMITER_MIN_K) and np.all(o <= op._THETA_LIMITER_MAX_K)
    m = np.asarray(mass, np.float64)
    target = np.sum(np.asarray(cand, np.float64) * m)
    # REAL redistribution: conserved to REAL rounding of the redistributed cells.
    assert abs(np.sum(o * m) - target) / target < 1e-6
    np.testing.assert_allclose(o, np.asarray(ref, np.float64), rtol=2e-6, atol=1e-3)
