"""Boundary/layout regression; full WRF fixtures are a separate required gate."""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import thompson_column as tc
from gpuwrf.kernels.phys_thompson_sedimentation import fill_down, sediment_one_species


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("nz", [1, 44, 64])
def test_column_layout_and_flux(dtype, nz):
    rng = np.random.default_rng(817)
    shape = (2, 3, nz)
    vt = jnp.asarray(rng.uniform(0, 4, shape), dtype)
    active = jnp.asarray(rng.uniform(0, 1, shape) > .5)
    active = active.at[0, 0, :].set(False)
    q = jnp.asarray(rng.uniform(0, 1.e-4, shape), dtype)
    num = q * 1.e8
    dz = jnp.full(shape, 100, dtype)
    rho = jnp.asarray(rng.uniform(.5, 1.3, shape), dtype)
    steps = jnp.asarray([[1, 2, 4], [3, 1, 2]], dtype)
    os.environ["GPUWRF_THOMPSON_COLUMN_SED"] = "0"
    reference_fill = jax.jit(tc._fill_down)(vt, active)
    reference_sed = jax.jit(tc._sed_one_species)(q, num, vt, vt, dz, rho, 18., steps)
    interpret = jax.default_backend() == "cpu"
    actual_fill = jax.jit(lambda v, a: fill_down(v, a, interpret=interpret))(vt, active)
    actual_sed = jax.jit(lambda *args: sediment_one_species(
        *args, tc.RR_SURF_THRESHOLD, interpret=interpret))(
            q, num, vt, vt, dz, rho, 18., steps)
    np.testing.assert_array_equal(actual_fill, reference_fill)
    # Compiler/FMA rounding, not a physics tolerance.
    rtol = 2.e-6 if dtype == jnp.float32 else 2.e-13
    for actual, reference in zip(actual_sed, reference_sed):
        np.testing.assert_allclose(actual, reference, rtol=rtol, atol=0)


def test_column_path_does_not_truncate_adaptive_steps(monkeypatch):
    vt = jnp.asarray([[30., 0.]], jnp.float32)
    dz = jnp.ones_like(vt)
    monkeypatch.setenv("GPUWRF_THOMPSON_COLUMN_SED", "1")
    steps = tc._nstep_per_column(vt, vt, dz, 1.)
    # Above NSED_MAX (16) on both paths. WRF REAL INT(DT/(dzq/vt)+1.) = 30 (1/30 rounds down in
    # REAL; the v0.3 release defaults set GPUWRF_THOMPSON_SED_FP32=1); the f64 path gives 31.
    assert float(steps[0]) == (30. if tc._column_sed_fp32_enabled() else 31.)
