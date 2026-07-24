"""Independent literal CPU oracles for the v0234 S1 source candidates."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.dynamics.core.rk_addtend_dry import large_step_horizontal_curvature
from gpuwrf.dynamics.explicit_diffusion import (
    wrf_nested_horizontal_diffusion_momentum_tendency,
)
from tests.dynamics.test_diffopt1_smagorinsky_integration import _build_grid, _build_state
from tests.test_v0234_nested_scalar_diffusion_source_repair import _nonuniform_maps

jax.config.update("jax_enable_x64", True)


def _literal_momentum_hdiff(u, v, w, kh, mut, c1h, c2h, c1f, c2f, maps, dx, dy):
    """Loop transcription of WRF horizontal_diffusion U/V/W branches."""

    msftx, msfty, msfux, msfuy, msfvx, msfvy = maps
    nz, ny, nx = kh.shape
    du, dv, dw = np.zeros_like(u), np.zeros_like(v), np.zeros_like(w)
    for k in range(nz):
        for j in range(1, ny - 1):
            for i in range(1, nx):
                mass = lambda jj, ii: c1h[k] * mut[jj, ii] + c2h[k]
                mkrdxm = (msftx[j, i - 1] / msfty[j, i - 1]) * mass(j, i - 1) * kh[k, j, i - 1] / dx
                mkrdxp = (msftx[j, i] / msfty[j, i]) * mass(j, i) * kh[k, j, i] / dx
                mrdx = msfux[j, i] * msfuy[j, i] / dx
                mkrdym = (
                    (msfuy[j, i] + msfuy[j - 1, i])
                    / (msfux[j, i] + msfux[j - 1, i])
                    * 0.25 * (mass(j, i) + mass(j - 1, i) + mass(j - 1, i - 1) + mass(j, i - 1))
                    * 0.25 * (kh[k, j, i] + kh[k, j - 1, i] + kh[k, j - 1, i - 1] + kh[k, j, i - 1])
                    / dy
                )
                mkrdyp = (
                    (msfuy[j, i] + msfuy[j + 1, i])
                    / (msfux[j, i] + msfux[j + 1, i])
                    * 0.25 * (mass(j, i) + mass(j + 1, i) + mass(j + 1, i - 1) + mass(j, i - 1))
                    * 0.25 * (kh[k, j, i] + kh[k, j + 1, i] + kh[k, j + 1, i - 1] + kh[k, j, i - 1])
                    / dy
                )
                mrdy = msfux[j, i] * msfuy[j, i] / dy
                du[k, j, i] = (
                    mrdx * (mkrdxp * (u[k, j, i + 1] - u[k, j, i]) - mkrdxm * (u[k, j, i] - u[k, j, i - 1]))
                    + mrdy * (mkrdyp * (u[k, j + 1, i] - u[k, j, i]) - mkrdym * (u[k, j, i] - u[k, j - 1, i]))
                )
        for j in range(1, ny):
            for i in range(1, nx - 1):
                mass = lambda jj, ii: c1h[k] * mut[jj, ii] + c2h[k]
                mkrdxm = (
                    (msfvx[j, i] + msfvx[j, i - 1]) / (msfvy[j, i] + msfvy[j, i - 1])
                    * 0.25 * (mass(j, i) + mass(j - 1, i) + mass(j - 1, i - 1) + mass(j, i - 1))
                    * 0.25 * (kh[k, j, i] + kh[k, j - 1, i] + kh[k, j - 1, i - 1] + kh[k, j, i - 1])
                    / dx
                )
                mkrdxp = (
                    (msfvx[j, i] + msfvx[j, i + 1]) / (msfvy[j, i] + msfvy[j, i + 1])
                    * 0.25 * (mass(j, i) + mass(j - 1, i) + mass(j - 1, i + 1) + mass(j, i + 1))
                    * 0.25 * (kh[k, j, i] + kh[k, j - 1, i] + kh[k, j - 1, i + 1] + kh[k, j, i + 1])
                    / dx
                )
                mrdx = msfvx[j, i] * msfvy[j, i] / dx
                mkrdym = (msfty[j - 1, i] / msftx[j - 1, i]) * kh[k, j - 1, i] / dy
                mkrdyp = (msfty[j, i] / msftx[j, i]) * kh[k, j, i] / dy
                mrdy = msfvx[j, i] * msfvy[j, i] / dy
                dv[k, j, i] = (
                    mrdx * (mkrdxp * (v[k, j, i + 1] - v[k, j, i]) - mkrdxm * (v[k, j, i] - v[k, j, i - 1]))
                    + mrdy * (mkrdyp * (v[k, j + 1, i] - v[k, j, i]) - mkrdym * (v[k, j, i] - v[k, j - 1, i]))
                )
    for k in range(1, nz):
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                mass = lambda jj, ii: c1f[k] * mut[jj, ii] + c2f[k]
                mkrdxm = (
                    msfux[j, i] / msfuy[j, i]
                    * 0.25 * (mass(j, i) + mass(j, i - 1) + mass(j, i) + mass(j, i - 1))
                    * 0.25 * (kh[k, j, i] + kh[k, j, i - 1] + kh[k - 1, j, i] + kh[k - 1, j, i - 1]) / dx
                )
                mkrdxp = (
                    msfux[j, i + 1] / msfuy[j, i + 1]
                    * 0.25 * (mass(j, i + 1) + mass(j, i) + mass(j, i + 1) + mass(j, i))
                    * 0.25 * (kh[k, j, i + 1] + kh[k, j, i] + kh[k - 1, j, i + 1] + kh[k - 1, j, i]) / dx
                )
                mrdx = msftx[j, i] * msfty[j, i] / dx
                mkrdym = (
                    msfvy[j, i] / msfvx[j, i]
                    * 0.25 * (mass(j, i) + mass(j - 1, i) + mass(j, i) + mass(j - 1, i))
                    * 0.25 * (kh[k, j, i] + kh[k, j - 1, i] + kh[k - 1, j, i] + kh[k - 1, j - 1, i]) / dy
                )
                mkrdyp = (
                    msfvy[j + 1, i] / msfvx[j + 1, i]
                    * 0.25 * (mass(j + 1, i) + mass(j, i) + mass(j + 1, i) + mass(j, i))
                    * 0.25 * (kh[k, j + 1, i] + kh[k, j, i] + kh[k - 1, j + 1, i] + kh[k - 1, j, i]) / dy
                )
                mrdy = msftx[j, i] * msfty[j, i] / dy
                dw[k, j, i] = (
                    mrdx * (mkrdxp * (w[k, j, i + 1] - w[k, j, i]) - mkrdxm * (w[k, j, i] - w[k, j, i - 1]))
                    + mrdy * (mkrdyp * (w[k, j + 1, i] - w[k, j, i]) - mkrdym * (w[k, j, i] - w[k, j - 1, i]))
                )
    return du, dv, dw


def test_nested_momentum_diffusion_matches_literal_wrf_loops():
    rng = np.random.default_rng(2341)
    nz, ny, nx = 4, 7, 8
    u = rng.normal(size=(nz, ny, nx + 1)); v = rng.normal(size=(nz, ny + 1, nx)); w = rng.normal(size=(nz + 1, ny, nx))
    kh = 10 + 30 * rng.random((nz, ny, nx)); mut = 7.5e4 + 4e3 * rng.random((ny, nx))
    c1h = np.linspace(0.2, 0.9, nz); c2h = np.linspace(500, 1500, nz)
    c1f = np.linspace(0.1, 1.0, nz + 1); c2f = np.linspace(300, 1800, nz + 1)
    maps = _nonuniform_maps(ny, nx); dx, dy = 900.0, 1100.0
    actual = wrf_nested_horizontal_diffusion_momentum_tendency(
        *map(jnp.asarray, (u, v, w, kh, mut)), c1h=jnp.asarray(c1h), c2h=jnp.asarray(c2h),
        c1f=jnp.asarray(c1f), c2f=jnp.asarray(c2f), msftx=jnp.asarray(maps[0]),
        msfty=jnp.asarray(maps[1]), msfux=jnp.asarray(maps[2]), msfuy=jnp.asarray(maps[3]),
        msfvx=jnp.asarray(maps[4]), msfvy=jnp.asarray(maps[5]), dx_m=dx, dy_m=dy,
    )
    expected = _literal_momentum_hdiff(u, v, w, kh, mut, c1h, c2h, c1f, c2f, maps, dx, dy)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_allclose(np.asarray(got), want, rtol=0.0, atol=1e-12)


def _literal_curvature(state, metrics, dx, dy):
    u, v, w, mut = map(np.asarray, (state.u, state.v, state.w, state.mu_total))
    c1h, c2h, c1f, c2f = map(np.asarray, (metrics.c1h, metrics.c2h, metrics.c1f, metrics.c2f))
    muu = np.pad(mut, ((0, 0), (1, 1)), mode="edge"); muu = 0.5 * (muu[:, :-1] + muu[:, 1:])
    muv = np.pad(mut, ((1, 1), (0, 0)), mode="edge"); muv = 0.5 * (muv[:-1] + muv[1:])
    ru = u * (c1h[:, None, None] * muu + c2h[:, None, None]) / np.asarray(metrics.msfuy)[None]
    rv = v * (c1h[:, None, None] * muv + c2h[:, None, None]) / np.asarray(metrics.msfvx)[None]
    rw = w * (c1f[:, None, None] * mut + c2f[:, None, None]) / np.asarray(metrics.msfty)[None]
    msfuy, msfvx, msfvy = map(np.asarray, (metrics.msfuy, metrics.msfvx, metrics.msfvy))
    vx = 0.5 * (u[:, :, :-1] + u[:, :, 1:]) * (msfvx[1:] - msfvx[:-1])[None] / dy - 0.5 * (v[:, :-1] + v[:, 1:]) * (msfuy[:, 1:] - msfuy[:, :-1])[None] / dx
    cu, cv = np.zeros_like(u), np.zeros_like(v); rr = 1 / 6370e3
    nz, ny, nx = mut.shape[0] if mut.ndim == 3 else u.shape[0], mut.shape[0], mut.shape[1]
    for k in range(nz):
        for j in range(ny):
            for i in range(1, nx):
                cu[k, j, i] = 0.5 * (vx[k, j, i] + vx[k, j, i - 1]) * 0.25 * (rv[k, j + 1, i - 1] + rv[k, j + 1, i] + rv[k, j, i - 1] + rv[k, j, i]) - u[k, j, i] * rr * 0.25 * (rw[k + 1, j, i - 1] + rw[k, j, i - 1] + rw[k + 1, j, i] + rw[k, j, i])
        for j in range(1, ny):
            for i in range(nx):
                cv[k, j, i] = -0.5 * (vx[k, j, i] + vx[k, j - 1, i]) * 0.25 * (ru[k, j, i] + ru[k, j, i + 1] + ru[k, j - 1, i] + ru[k, j - 1, i + 1]) - (msfvy[j, i] / msfvx[j, i]) * v[k, j, i] * rr * 0.25 * (rw[k + 1, j - 1, i] + rw[k, j - 1, i] + rw[k + 1, j, i] + rw[k, j, i])
    return cu, cv


def test_horizontal_curvature_matches_literal_wrf_loops():
    rng = np.random.default_rng(2342); nz, ny, nx = 4, 6, 7
    grid = _build_grid(ny=ny, nx=nx, nz=nz, dx=1000.0); state = _build_state(grid)
    maps = _nonuniform_maps(ny, nx)
    metrics = dataclasses.replace(grid.metrics, msftx=jnp.asarray(maps[0]), msfty=jnp.asarray(maps[1]), msfux=jnp.asarray(maps[2]), msfuy=jnp.asarray(maps[3]), msfvx=jnp.asarray(maps[4]), msfvy=jnp.asarray(maps[5]))
    state = state.replace(u=jnp.asarray(rng.normal(size=(nz, ny, nx + 1))), v=jnp.asarray(rng.normal(size=(nz, ny + 1, nx))), w=jnp.asarray(rng.normal(size=(nz + 1, ny, nx))), mu_total=jnp.asarray(7.5e4 + 2e3 * rng.random((ny, nx))))
    got = large_step_horizontal_curvature(state, metrics, dx_m=900.0, dy_m=1100.0, specified=True)
    want = _literal_curvature(state, metrics, 900.0, 1100.0)
    for actual, expected in zip(got, want, strict=True):
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=0.0, atol=1e-12)


def test_changed_operators_have_no_host_callback_primitive():
    nz, ny, nx = 3, 5, 6
    rng = np.random.default_rng(2343)
    maps = _nonuniform_maps(ny, nx)
    arrays = (
        jnp.asarray(rng.normal(size=(nz, ny, nx + 1))),
        jnp.asarray(rng.normal(size=(nz, ny + 1, nx))),
        jnp.asarray(rng.normal(size=(nz + 1, ny, nx))),
        jnp.asarray(10 + rng.random(size=(nz, ny, nx))),
        jnp.asarray(7.5e4 + rng.random(size=(ny, nx))),
    )
    constants = tuple(jnp.asarray(value) for value in (
        np.linspace(0.2, 0.9, nz), np.linspace(500, 1500, nz),
        np.linspace(0.1, 1.0, nz + 1), np.linspace(300, 1800, nz + 1),
    ))

    def diffusion(*values):
        u, v, w, kh, mut, c1h, c2h, c1f, c2f = values
        return wrf_nested_horizontal_diffusion_momentum_tendency(
            u, v, w, kh, mut, c1h=c1h, c2h=c2h, c1f=c1f, c2f=c2f,
            msftx=jnp.asarray(maps[0]), msfty=jnp.asarray(maps[1]),
            msfux=jnp.asarray(maps[2]), msfuy=jnp.asarray(maps[3]),
            msfvx=jnp.asarray(maps[4]), msfvy=jnp.asarray(maps[5]),
            dx_m=900.0, dy_m=1100.0,
        )

    grid = _build_grid(ny=ny, nx=nx, nz=nz, dx=1000.0)
    metrics = dataclasses.replace(
        grid.metrics, msftx=jnp.asarray(maps[0]), msfty=jnp.asarray(maps[1]),
        msfux=jnp.asarray(maps[2]), msfuy=jnp.asarray(maps[3]),
        msfvx=jnp.asarray(maps[4]), msfvy=jnp.asarray(maps[5]),
    )
    state = _build_state(grid).replace(u=arrays[0], v=arrays[1], w=arrays[2], mu_total=arrays[4])
    diffusion_jaxpr = str(jax.make_jaxpr(diffusion)(*arrays, *constants))
    curvature_jaxpr = str(jax.make_jaxpr(
        lambda value: large_step_horizontal_curvature(
            value, metrics, dx_m=900.0, dy_m=1100.0, specified=True
        )
    )(state))
    forbidden = ("pure_callback", "io_callback", "host_callback", "device_get")
    assert not any(token in diffusion_jaxpr for token in forbidden)
    assert not any(token in curvature_jaxpr for token in forbidden)
