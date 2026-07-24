"""Focused CPU/source gates for the nested WRF diff_opt=1 scalar repair."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.coupling.boundary_apply import BoundaryConfig
from gpuwrf.dynamics.advection import apply_halo, halo_spec
from gpuwrf.dynamics.explicit_diffusion import (
    horizontal_diffusion_coord_scalar_tendency,
    wrf_nonperiodic_diffusion_metrics,
)
from gpuwrf.runtime.operational_mode import _diffopt1_dry_forward_tendencies
from tests.dynamics.test_diffopt1_smagorinsky_integration import (
    _build_grid,
    _build_state,
    _namelist,
)

jax.config.update("jax_enable_x64", True)


WRF_DIFFUSION = Path("<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_diffusion_em.F")
WRF_BIG_STEP = Path(
    "<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F"
)
WRF_DIFFUSION_SHA256 = (
    "a7d4570c97e51c635e86a0dbd628c6846457ac5b93d5a7af798b118c7d8d2d54"
)
WRF_BIG_STEP_SHA256 = (
    "bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _literal_nested_scalar(field, coefficient, mass, maps, dx, dy):
    """Independent NumPy transcription of WRF horizontal_diffusion_3dmp."""

    q = np.asarray(field, dtype=np.float64)
    k = np.asarray(coefficient, dtype=np.float64)
    mu = np.asarray(mass, dtype=np.float64)
    msftx, msfty, msfux, msfuy, msfvx, msfvy = maps
    out = np.zeros_like(q)
    ke = 0.5 * (k[:, 1:-1, 2:] + k[:, 1:-1, 1:-1])
    kw = 0.5 * (k[:, 1:-1, 1:-1] + k[:, 1:-1, :-2])
    me = 0.5 * (mu[:, 1:-1, 2:] + mu[:, 1:-1, 1:-1])
    mw = 0.5 * (mu[:, 1:-1, 1:-1] + mu[:, 1:-1, :-2])
    kn = 0.5 * (k[:, 2:, 1:-1] + k[:, 1:-1, 1:-1])
    ks = 0.5 * (k[:, 1:-1, 1:-1] + k[:, :-2, 1:-1])
    mn = 0.5 * (mu[:, 2:, 1:-1] + mu[:, 1:-1, 1:-1])
    ms = 0.5 * (mu[:, 1:-1, 1:-1] + mu[:, :-2, 1:-1])
    area = (msftx[1:-1, 1:-1] * msfty[1:-1, 1:-1])[None]
    xflux = area / dx * (
        (msfux[1:-1, 2:-1] / msfuy[1:-1, 2:-1])[None]
        * ke
        * me
        / dx
        * (q[:, 1:-1, 2:] - q[:, 1:-1, 1:-1])
        - (msfux[1:-1, 1:-2] / msfuy[1:-1, 1:-2])[None]
        * kw
        * mw
        / dx
        * (q[:, 1:-1, 1:-1] - q[:, 1:-1, :-2])
    )
    yflux = area / dy * (
        (msfvy[2:-1, 1:-1] / msfvx[2:-1, 1:-1])[None]
        * kn
        * mn
        / dy
        * (q[:, 2:, 1:-1] - q[:, 1:-1, 1:-1])
        - (msfvy[1:-2, 1:-1] / msfvx[1:-2, 1:-1])[None]
        * ks
        * ms
        / dy
        * (q[:, 1:-1, 1:-1] - q[:, :-2, 1:-1])
    )
    out[:, 1:-1, 1:-1] = xflux + yflux
    return out


def _nonuniform_maps(ny: int, nx: int):
    yy, xx = np.indices((ny, nx))
    msftx = 0.98 + 0.001 * yy + 0.0007 * xx
    msfty = 1.01 + 0.0004 * yy - 0.0003 * xx
    yu, xu = np.indices((ny, nx + 1))
    msfux = 0.99 + 0.0006 * yu + 0.0008 * xu
    msfuy = 1.02 - 0.0002 * yu + 0.0005 * xu
    yv, xv = np.indices((ny + 1, nx))
    msfvx = 1.01 + 0.0003 * yv - 0.0004 * xv
    msfvy = 0.97 + 0.0005 * yv + 0.0002 * xv
    return msftx, msfty, msfux, msfuy, msfvx, msfvy


def test_pristine_sources_bind_metrics_ownership_and_scalar_flux_order():
    assert _sha256(WRF_DIFFUSION) == WRF_DIFFUSION_SHA256
    assert _sha256(WRF_BIG_STEP) == WRF_BIG_STEP_SHA256
    diffusion = WRF_DIFFUSION.read_text()
    big_step = WRF_BIG_STEP.read_text()
    assert "SUBROUTINE compute_diff_metrics" in diffusion
    assert "SUBROUTINE cal_deform_and_div" in diffusion
    assert "SUBROUTINE smag2d_km" in diffusion
    smag = diffusion[diffusion.index("SUBROUTINE smag2d_km") :]
    assert "config_flags%nested) i_start = MAX(ids+1,its)" in smag
    scalar = big_step[big_step.index("SUBROUTINE horizontal_diffusion_3dmp") :]
    assert scalar.index("specified = .false.") < scalar.index("mkrdxm=")
    assert "config_flags%nested) specified = .true." in scalar
    assert "msftx(i,j)*msfty(i,j)*rdx" in scalar


def test_nonperiodic_diffusion_metrics_match_independent_wrf_equations():
    nz, ny, nx = 4, 6, 8
    zz, yy, xx = np.indices((nz + 1, ny, nx))
    z = 50.0 + 300.0 * zz + 3.0 * yy + 2.0 * xx + 0.2 * zz * xx
    ph = 9.81 * z
    zx, zy, rdzw = wrf_nonperiodic_diffusion_metrics(
        jnp.asarray(ph), dx_m=1000.0, dy_m=1200.0
    )
    zx_ref = np.zeros_like(z)
    zy_ref = np.zeros_like(z)
    zx_ref[:, :, 1:] = (z[:, :, 1:] - z[:, :, :-1]) / 1000.0
    zy_ref[:, 1:, :] = (z[:, 1:, :] - z[:, :-1, :]) / 1200.0
    rdzw_ref = 1.0 / (z[1:] - z[:-1])
    np.testing.assert_allclose(np.asarray(zx), zx_ref, rtol=0.0, atol=3e-16)
    np.testing.assert_allclose(np.asarray(zy), zy_ref, rtol=0.0, atol=3e-16)
    np.testing.assert_allclose(np.asarray(rdzw), rdzw_ref, rtol=2e-15, atol=0.0)


def test_nested_map_scalar_matches_literal_wrf_and_zeroes_outer_ring():
    rng = np.random.default_rng(234)
    nz, ny, nx = 4, 8, 10
    field = rng.standard_normal((nz, ny, nx))
    coefficient = 20.0 + 80.0 * rng.random((nz, ny, nx))
    coefficient[:, 0, :] = coefficient[:, -1, :] = 0.0
    coefficient[:, :, 0] = coefficient[:, :, -1] = 0.0
    mass = 8.0e4 + 2.0e3 * rng.random((nz, ny, nx))
    maps = _nonuniform_maps(ny, nx)
    actual = horizontal_diffusion_coord_scalar_tendency(
        jnp.asarray(field),
        jnp.asarray(coefficient),
        jnp.asarray(mass),
        dx_m=900.0,
        dy_m=1100.0,
        msftx=jnp.asarray(maps[0]),
        msfty=jnp.asarray(maps[1]),
        msfux=jnp.asarray(maps[2]),
        msfuy=jnp.asarray(maps[3]),
        msfvx=jnp.asarray(maps[4]),
        msfvy=jnp.asarray(maps[5]),
        nonperiodic_owned=True,
    )
    expected = _literal_nested_scalar(field, coefficient, mass, maps, 900.0, 1100.0)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=3e-15, atol=2e-15)
    assert np.count_nonzero(np.asarray(actual)[:, 0, :]) == 0
    assert np.count_nonzero(np.asarray(actual)[:, -1, :]) == 0
    assert np.count_nonzero(np.asarray(actual)[:, :, 0]) == 0
    assert np.count_nonzero(np.asarray(actual)[:, :, -1]) == 0


def test_map_periodic_operator_conserves_weighted_coupled_scalar():
    rng = np.random.default_rng(235)
    nz, ny, nx = 3, 7, 9
    field = rng.standard_normal((nz, ny, nx))
    coefficient = 30.0 + 40.0 * rng.random((nz, ny, nx))
    mass = 8.5e4 + 1000.0 * rng.random((nz, ny, nx))
    msftx, msfty, msfux, msfuy, msfvx, msfvy = _nonuniform_maps(ny, nx)
    # Close both staggered face metrics periodically.
    msfux[:, -1] = msfux[:, 0]
    msfuy[:, -1] = msfuy[:, 0]
    msfvx[-1, :] = msfvx[0, :]
    msfvy[-1, :] = msfvy[0, :]
    tendency = np.asarray(
        horizontal_diffusion_coord_scalar_tendency(
            jnp.asarray(field),
            jnp.asarray(coefficient),
            jnp.asarray(mass),
            dx_m=900.0,
            dy_m=1100.0,
            msftx=jnp.asarray(msftx),
            msfty=jnp.asarray(msfty),
            msfux=jnp.asarray(msfux),
            msfuy=jnp.asarray(msfuy),
            msfvx=jnp.asarray(msfvx),
            msfvy=jnp.asarray(msfvy),
        )
    )
    weighted = np.sum(tendency / (msftx * msfty)[None], axis=(1, 2))
    np.testing.assert_allclose(weighted, 0.0, rtol=0.0, atol=2e-13)


def test_nested_path_selects_source_momentum_bundle_and_exact_ownership():
    grid = _build_grid(ny=10, nx=12, nz=5, dx=1000.0)
    state = apply_halo(_build_state(grid), halo_spec(grid))
    ideal = _namelist(grid, diff_opt=1, km_opt=4, hypsometric_opt=1)
    nested_grid = dataclasses.replace(
        grid,
        bc=dataclasses.replace(grid.bc, source="live-parent"),
    )
    nested = _namelist(
        nested_grid,
        diff_opt=1,
        km_opt=4,
        hypsometric_opt=1,
        boundary_config=BoundaryConfig(force_geopotential=False),
    )
    old = _diffopt1_dry_forward_tendencies(state, ideal)
    candidate = _diffopt1_dry_forward_tendencies(state, nested)
    for old_momentum, new_momentum in zip(old[:3], candidate[:3], strict=True):
        old_array = np.asarray(old_momentum)
        new_array = np.asarray(new_momentum)
        assert np.isfinite(new_array).all()
        assert not np.array_equal(new_array, old_array)
    # U excludes outer x faces and y mass rows; V excludes outer y faces and
    # x mass columns; W excludes both horizontal rings and vertical faces.
    assert np.count_nonzero(np.asarray(candidate[0])[:, 0, :]) == 0
    assert np.count_nonzero(np.asarray(candidate[0])[:, -1, :]) == 0
    assert np.count_nonzero(np.asarray(candidate[0])[:, :, 0]) == 0
    assert np.count_nonzero(np.asarray(candidate[0])[:, :, -1]) == 0
    assert np.count_nonzero(np.asarray(candidate[1])[:, 0, :]) == 0
    assert np.count_nonzero(np.asarray(candidate[1])[:, -1, :]) == 0
    assert np.count_nonzero(np.asarray(candidate[1])[:, :, 0]) == 0
    assert np.count_nonzero(np.asarray(candidate[1])[:, :, -1]) == 0
    assert np.count_nonzero(np.asarray(candidate[2])[0]) == 0
    assert np.count_nonzero(np.asarray(candidate[2])[-1]) == 0
    assert np.max(np.abs(np.asarray(candidate[3] - old[3]))) > 0.0
    assert np.count_nonzero(np.asarray(candidate[3])[:, 0, :]) == 0
    assert np.count_nonzero(np.asarray(candidate[3])[:, -1, :]) == 0
    assert np.count_nonzero(np.asarray(candidate[3])[:, :, 0]) == 0
    assert np.count_nonzero(np.asarray(candidate[3])[:, :, -1]) == 0


def test_candidate_sources_have_no_observer_clamp_or_transfer_surface():
    sources = (
        inspect.getsource(wrf_nonperiodic_diffusion_metrics),
        inspect.getsource(horizontal_diffusion_coord_scalar_tendency),
        inspect.getsource(_diffopt1_dry_forward_tendencies),
    )
    for source in sources:
        for forbidden in (
            "device_get",
            "pure_callback",
            "debug.callback",
            "io_callback",
            "host_callback",
            "jnp.nan_to_num",
            "clip(",
        ):
            assert forbidden not in source
