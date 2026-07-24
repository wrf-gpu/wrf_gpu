"""Source/algebra gates for the nested RK1 theta sixth-order discriminator."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.dynamics.explicit_diffusion import wrf_sixth_order_scalar_tendf


WRF = Path("<USER_HOME>/src/wrf_pristine/WRF")


def _numpy_wrf_scalar(
    field: np.ndarray,
    mu: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    msftx: np.ndarray,
    msfty: np.ndarray,
    *,
    dt: float,
    factor: float,
    monotonic: bool,
) -> np.ndarray:
    """Direct-loop transcription of WRF ``sixth_order_diffusion('m')``."""

    nz, ny, nx = field.shape
    mass = c1[:, None, None] * mu[None, :, :] + c2[:, None, None]
    out = np.zeros_like(field, dtype=np.float64)
    coef = factor * 0.015625 / (2.0 * dt)
    for k in range(nz):
        for j in range(3, ny - 3):
            for i in range(3, nx - 3):
                x0 = (
                    10.0 * (field[k, j, i] - field[k, j, i - 1])
                    - 5.0 * (field[k, j, i + 1] - field[k, j, i - 2])
                    + (field[k, j, i + 2] - field[k, j, i - 3])
                )
                x1 = (
                    10.0 * (field[k, j, i + 1] - field[k, j, i])
                    - 5.0 * (field[k, j, i + 2] - field[k, j, i - 1])
                    + (field[k, j, i + 3] - field[k, j, i - 2])
                )
                y0 = (
                    10.0 * (field[k, j, i] - field[k, j - 1, i])
                    - 5.0 * (field[k, j + 1, i] - field[k, j - 2, i])
                    + (field[k, j + 2, i] - field[k, j - 3, i])
                )
                y1 = (
                    10.0 * (field[k, j + 1, i] - field[k, j, i])
                    - 5.0 * (field[k, j + 2, i] - field[k, j - 1, i])
                    + (field[k, j + 3, i] - field[k, j - 2, i])
                )
                if monotonic:
                    if x0 * (field[k, j, i] - field[k, j, i - 1]) <= 0.0:
                        x0 = 0.0
                    if x1 * (field[k, j, i + 1] - field[k, j, i]) <= 0.0:
                        x1 = 0.0
                    if y0 * (field[k, j, i] - field[k, j - 1, i]) <= 0.0:
                        y0 = 0.0
                    if y1 * (field[k, j + 1, i] - field[k, j, i]) <= 0.0:
                        y1 = 0.0
                mx0 = 0.5 * (mass[k, j, i - 1] + mass[k, j, i])
                mx1 = 0.5 * (mass[k, j, i] + mass[k, j, i + 1])
                my0 = 0.5 * (mass[k, j - 1, i] + mass[k, j, i])
                my1 = 0.5 * (mass[k, j, i] + mass[k, j + 1, i])
                out[k, j, i] = coef * (
                    msftx[j, i] * (mx1 * x1 - mx0 * x0)
                    + msfty[j, i] * (my1 * y1 - my0 * y0)
                )
    return out


def test_pristine_wrf_theta_sixth_order_is_rk1_forward_mass_map_coupled():
    module_em = (WRF / "dyn_em/module_em.F").read_text()
    sixth = (WRF / "dyn_em/module_big_step_utilities_em.F").read_text()
    start = module_em.index("forward_step: IF( rk_step == 1 ) THEN")
    end = module_em.index("END IF forward_step", start)
    forward = module_em[start:end]
    assert "CALL sixth_order_diffusion( 'm', t,  t_tendf" in forward
    assert "if(config_flags%specified .or. config_flags%nested) specified = .true." in sixth
    assert "i_start = MAX(its,ids+3)" in sixth
    assert "i_end   = MIN(ide-4,ite)" in sixth
    assert "j_start = MAX(jts,jds+3)" in sixth
    assert "j_end   = MIN(jde-4,jte)" in sixth
    assert "mu_avg_p0 = 0.5 *" in sixth
    assert "tendency_x = diff_6th_coef * msftx(i,j)" in sixth
    assert "tendency_y = diff_6th_coef * msfty(i,j)" in sixth
    assert "t_tend(i,k,j) =  t_tend(i,k,j) +  t_tendf(i,k,j)/msfty(i,j)" in module_em

    runtime = Path("src/gpuwrf/runtime/operational_mode.py").read_text()
    build = runtime.index("rk1_forward_diff6_theta = None")
    stages = runtime.index("def advance_stage", build)
    consume = runtime.index("frozen_diff6_theta_tendency=rk1_forward_diff6_theta", stages)
    assert build < stages < consume


@pytest.mark.parametrize("monotonic", [False, True])
def test_scalar_tendf_matches_independent_numpy_source_oracle(monotonic: bool):
    rng = np.random.default_rng(20260714 + int(monotonic))
    nz, ny, nx = 4, 12, 14
    field = rng.normal(size=(nz, ny, nx))
    mu = rng.uniform(75_000.0, 98_000.0, size=(ny, nx))
    c1 = rng.uniform(0.2, 1.1, size=nz)
    c2 = rng.uniform(20.0, 300.0, size=nz)
    msftx = rng.uniform(0.92, 1.18, size=(ny, nx))
    msfty = rng.uniform(0.93, 1.16, size=(ny, nx))
    expected = _numpy_wrf_scalar(
        field,
        mu,
        c1,
        c2,
        msftx,
        msfty,
        dt=6.0,
        factor=0.12,
        monotonic=monotonic,
    )
    actual = np.asarray(
        wrf_sixth_order_scalar_tendf(
            jnp.asarray(field),
            jnp.asarray(mu),
            c1=jnp.asarray(c1),
            c2=jnp.asarray(c2),
            msftx=jnp.asarray(msftx),
            msfty=jnp.asarray(msfty),
            dt=6.0,
            diff_6th_factor=0.12,
            monotonic=monotonic,
            specified_or_nested=True,
        )
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-15, atol=2.0e-10)


def test_nested_operator_zeros_rings_zero_to_two_and_never_wraps_opposite_edge():
    nz, ny, nx = 2, 13, 15
    field = np.zeros((nz, ny, nx), dtype=np.float64)
    field[:, 3:-3, -1] = 7.0
    ones2 = np.ones((ny, nx), dtype=np.float64)
    out = np.asarray(
        wrf_sixth_order_scalar_tendf(
            jnp.asarray(field),
            jnp.asarray(ones2),
            c1=jnp.ones((nz,), dtype=jnp.float64),
            c2=jnp.zeros((nz,), dtype=jnp.float64),
            msftx=jnp.asarray(ones2),
            msfty=jnp.asarray(ones2),
            dt=6.0,
            diff_6th_factor=0.12,
            monotonic=False,
            specified_or_nested=True,
        )
    )
    yy, xx = np.indices((ny, nx))
    ring = np.minimum.reduce((yy, xx, ny - 1 - yy, nx - 1 - xx))
    np.testing.assert_array_equal(out[:, ring <= 2], 0.0)
    np.testing.assert_array_equal(out[:, 3:-3, 3], 0.0)
    assert np.any(out[:, 3:-3, -4] != 0.0)


@pytest.mark.parametrize("monotonic", [False, True])
def test_constant_and_affine_fields_have_zero_owned_sixth_difference(monotonic: bool):
    nz, ny, nx = 3, 13, 15
    kk, yy, xx = np.indices((nz, ny, nx), dtype=np.float64)
    ones = np.ones((ny, nx), dtype=np.float64)
    mu = 80_000.0 + 25.0 * yy[0] + 11.0 * xx[0]
    fields = (
        np.full((nz, ny, nx), 4.25, dtype=np.float64),
        1.5 * kk - 0.75 * yy + 0.125 * xx,
    )
    for field in fields:
        out = np.asarray(
            wrf_sixth_order_scalar_tendf(
                jnp.asarray(field),
                jnp.asarray(mu),
                c1=jnp.linspace(0.4, 0.9, nz),
                c2=jnp.linspace(100.0, 300.0, nz),
                msftx=jnp.asarray(ones),
                msfty=jnp.asarray(ones),
                dt=6.0,
                diff_6th_factor=0.12,
                monotonic=monotonic,
                specified_or_nested=True,
            )
        )
        np.testing.assert_allclose(out, 0.0, rtol=0.0, atol=1.0e-15)


def test_compact_interior_signal_closes_flux_conservation_with_unit_geometry():
    nz, ny, nx = 2, 19, 21
    field = np.zeros((nz, ny, nx), dtype=np.float64)
    field[:, 8:11, 9:12] = np.array([1.0, -0.5])[:, None, None]
    ones = np.ones((ny, nx), dtype=np.float64)
    out = np.asarray(
        wrf_sixth_order_scalar_tendf(
            jnp.asarray(field),
            jnp.asarray(ones),
            c1=jnp.ones((nz,), dtype=jnp.float64),
            c2=jnp.zeros((nz,), dtype=jnp.float64),
            msftx=jnp.asarray(ones),
            msfty=jnp.asarray(ones),
            dt=6.0,
            diff_6th_factor=0.12,
            monotonic=True,
            specified_or_nested=True,
        )
    )
    np.testing.assert_allclose(np.sum(out, axis=(1, 2)), 0.0, rtol=0.0, atol=2.0e-17)
