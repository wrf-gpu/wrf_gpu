"""Analytic-oracle tests for diff_opt=1 + km_opt=4 (2-D Smagorinsky).

Operators under test (``gpuwrf.dynamics.explicit_diffusion``):
  * ``horizontal_deformation_2d``                 -- WRF cal_deform_and_div (flat slab)
  * ``smag2d_horizontal_km``                       -- WRF smag2d_km (km_opt=4)
  * ``horizontal_diffusion_coord_scalar_tendency`` -- WRF horizontal_diffusion(_3dmp)
  * ``horizontal_diffusion_coord_momentum_tendency``

Reference = unmodified WRF dyn_em/module_diffusion_em.F + module_big_step_utilities_em.F.
Each test builds smooth periodic analytic fields and compares the JAX operator to an
independent NumPy reference of the literal WRF formula (no copy of the JAX staggering).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gpuwrf.dynamics.explicit_diffusion import (
    C_S_DEFAULT,
    PRANDTL,
    horizontal_deformation_2d,
    horizontal_diffusion_coord_scalar_tendency,
    horizontal_diffusion_coord_momentum_tendency,
    smag2d_horizontal_km,
)


# ---------------------------------------------------------------------------
# 1. Deformation tensor (WRF cal_deform_and_div, flat slab)
# ---------------------------------------------------------------------------


def test_deformation_matches_analytic_field():
    """D11=2du/dx, D22=2dv/dy, D12=du/dy+dv/dx vs analytic derivatives of a smooth flow."""

    nz, ny, nx = 4, 24, 32
    dx, dy = 1000.0, 1000.0
    # x on u-faces (west face of cell i is at x=i*dx); cell centre at (i+0.5)*dx.
    kx = 2.0 * np.pi / (nx * dx)
    ky = 2.0 * np.pi / (ny * dy)
    xf = (np.arange(nx + 1)) * dx          # u face x-coords
    yc = (np.arange(ny) + 0.5) * dy        # mass cell y-coords
    xc = (np.arange(nx) + 0.5) * dx        # mass cell x-coords
    yf = (np.arange(ny + 1)) * dy          # v face y-coords

    # u(x_face, y_cell) = sin(kx*x)*cos(ky*y);  v(x_cell, y_face)=cos(kx*x)*sin(ky*y)
    u = (np.sin(kx * xf)[None, :] * np.cos(ky * yc)[:, None])[None, :, :]
    u = np.broadcast_to(u, (nz, ny, nx + 1)).copy()
    v = (np.cos(kx * xc)[None, :] * np.sin(ky * yf)[:, None])[None, :, :]
    v = np.broadcast_to(v, (nz, ny + 1, nx)).copy()

    d11, d22, d12 = horizontal_deformation_2d(
        jnp.asarray(u), jnp.asarray(v), dx_m=dx, dy_m=dy
    )
    d11 = np.asarray(d11)
    d22 = np.asarray(d22)
    d12 = np.asarray(d12)

    # D11 = 2 du/dx at mass cell (centred difference of u faces) -> analytic 2*kx*cos(kx*xc)*cos(ky*yc)
    d11_o = 2.0 * (np.sin(kx * xf[1:]) - np.sin(kx * xf[:-1]))[None, :] / dx * np.cos(ky * yc)[:, None]
    d11_o = np.broadcast_to(d11_o[None], (nz, ny, nx))
    assert np.allclose(d11, d11_o, rtol=1e-12, atol=1e-12)

    # D22 = 2 dv/dy at mass cell.
    d22_o = 2.0 * np.cos(kx * xc)[None, :] * (np.sin(ky * yf[1:]) - np.sin(ky * yf[:-1]))[:, None] / dy
    d22_o = np.broadcast_to(d22_o[None], (nz, ny, nx))
    assert np.allclose(d22, d22_o, rtol=1e-12, atol=1e-12)

    # D12 at SW corner (i,j): du/dy = (u(i,j)-u(i,j-1))/dy ; dv/dx = (v(i,j)-v(i-1,j))/dx
    u_m = u[:, :, :nx]
    dudy = (u_m - np.roll(u_m, 1, axis=1)) / dy
    v_m = v[:, :ny, :]
    dvdx = (v_m - np.roll(v_m, 1, axis=2)) / dx
    d12_o = dudy + dvdx
    assert np.allclose(d12, d12_o, rtol=1e-12, atol=1e-12)


def test_deformation_zero_on_solid_body():
    """Uniform translation has zero deformation."""

    nz, ny, nx = 3, 16, 20
    u = jnp.full((nz, ny, nx + 1), 5.0, dtype=jnp.float64)
    v = jnp.full((nz, ny + 1, nx), -3.0, dtype=jnp.float64)
    d11, d22, d12 = horizontal_deformation_2d(u, v, dx_m=500.0, dy_m=500.0)
    assert np.allclose(np.asarray(d11), 0.0, atol=1e-12)
    assert np.allclose(np.asarray(d22), 0.0, atol=1e-12)
    assert np.allclose(np.asarray(d12), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Smagorinsky Kh (WRF smag2d_km)
# ---------------------------------------------------------------------------


def _smag_oracle(d11, d22, d12, dx, dy, c_s, pr):
    """Literal NumPy transcription of WRF smag2d_km (msf=1, diff_opt=1 -> no slope)."""

    # tmp = 0.25*(D12(i,j)+D12(i,j+1)+D12(i+1,j)+D12(i+1,j+1)) ; corner (i,j)=SW of cell.
    nw = np.roll(d12, -1, axis=1)
    se = np.roll(d12, -1, axis=2)
    ne = np.roll(np.roll(d12, -1, axis=1), -1, axis=2)
    tmp = 0.25 * (d12 + nw + se + ne)
    def2 = 0.25 * (d11 - d22) ** 2 + tmp ** 2
    mlen_h = np.sqrt(dx * dy)
    xkmh = c_s * c_s * mlen_h * mlen_h * np.sqrt(def2)
    xkmh = np.minimum(xkmh, 10.0 * mlen_h)
    xkhh = xkmh / pr
    return xkmh, xkhh


def test_smag2d_km_matches_wrf_formula():
    """xkmh/xkhh from the JAX op equal the literal WRF smag2d_km formula."""

    rng = np.random.default_rng(0)
    nz, ny, nx = 4, 18, 22
    dx, dy = 750.0, 750.0
    # small smooth-ish random deformations (avoid hitting the 10*mlen_h cap so the
    # un-capped formula is exercised too).
    d11 = 1e-4 * rng.standard_normal((nz, ny, nx))
    d22 = 1e-4 * rng.standard_normal((nz, ny, nx))
    d12 = 1e-4 * rng.standard_normal((nz, ny, nx))

    xkmh, xkhh = smag2d_horizontal_km(
        jnp.asarray(d11), jnp.asarray(d22), jnp.asarray(d12),
        dx_m=dx, dy_m=dy, c_s=C_S_DEFAULT, prandtl=PRANDTL,
    )
    xkmh_o, xkhh_o = _smag_oracle(d11, d22, d12, dx, dy, C_S_DEFAULT, PRANDTL)

    assert np.allclose(np.asarray(xkmh), xkmh_o, rtol=1e-12, atol=1e-14)
    assert np.allclose(np.asarray(xkhh), xkhh_o, rtol=1e-12, atol=1e-14)
    # heat diffusivity is exactly 3x momentum (prandtl=1/3).
    assert np.allclose(np.asarray(xkhh), 3.0 * np.asarray(xkmh), rtol=1e-12)


def test_smag2d_km_cap_engages():
    """The 10*mlen_h ceiling (WRF :2019) caps large deformations."""

    nz, ny, nx = 2, 8, 8
    dx, dy = 1000.0, 1000.0
    big = np.full((nz, ny, nx), 1.0)  # def2 huge -> uncapped K >> 10*mlen_h
    xkmh, _ = smag2d_horizontal_km(
        jnp.asarray(big), jnp.zeros_like(jnp.asarray(big)), jnp.zeros_like(jnp.asarray(big)),
        dx_m=dx, dy_m=dy,
    )
    mlen_h = np.sqrt(dx * dy)
    assert np.allclose(np.asarray(xkmh), 10.0 * mlen_h, rtol=1e-12)


def test_smag2d_km_zero_deformation():
    """No deformation -> zero eddy viscosity (no spurious background K)."""

    z = jnp.zeros((3, 8, 8), dtype=jnp.float64)
    xkmh, xkhh = smag2d_horizontal_km(z, z, z, dx_m=1000.0, dy_m=1000.0)
    assert np.allclose(np.asarray(xkmh), 0.0, atol=1e-14)
    assert np.allclose(np.asarray(xkhh), 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# 3. Coordinate-surface scalar diffusion (WRF horizontal_diffusion / _3dmp)
# ---------------------------------------------------------------------------


def _hdiff_oracle(field, K, mass, dx, dy):
    """Literal WRF horizontal_diffusion scalar-branch flux divergence (msf=1)."""

    rdx, rdy = 1.0 / dx, 1.0 / dy
    ke = 0.5 * (np.roll(K, -1, axis=2) + K)
    kw = 0.5 * (K + np.roll(K, 1, axis=2))
    me = 0.5 * (np.roll(mass, -1, axis=2) + mass)
    mw = 0.5 * (mass + np.roll(mass, 1, axis=2))
    tend = rdx * (
        ke * me * rdx * (np.roll(field, -1, axis=2) - field)
        - kw * mw * rdx * (field - np.roll(field, 1, axis=2))
    )
    if field.shape[1] > 1:
        kn = 0.5 * (np.roll(K, -1, axis=1) + K)
        ks = 0.5 * (K + np.roll(K, 1, axis=1))
        mn = 0.5 * (np.roll(mass, -1, axis=1) + mass)
        ms = 0.5 * (mass + np.roll(mass, 1, axis=1))
        tend = tend + rdy * (
            kn * mn * rdy * (np.roll(field, -1, axis=1) - field)
            - ks * ms * rdy * (field - np.roll(field, 1, axis=1))
        )
    return tend


def test_coord_scalar_matches_wrf_flux_divergence():
    """Variable-K scalar diffusion equals the literal WRF flux-divergence formula."""

    rng = np.random.default_rng(1)
    nz, ny, nx = 5, 20, 24
    dx, dy = 800.0, 800.0
    xc = (np.arange(nx) + 0.5) * dx
    yc = (np.arange(ny) + 0.5) * dy
    kx = 2.0 * np.pi / (nx * dx)
    ky = 2.0 * np.pi / (ny * dy)
    field = (np.sin(kx * xc)[None, :] * np.cos(ky * yc)[:, None])[None]
    field = np.broadcast_to(field, (nz, ny, nx)).copy() + 300.0
    # spatially varying K (Smagorinsky-like) and column mass.
    K = 50.0 + 30.0 * (np.cos(kx * xc)[None, :] * np.sin(ky * yc)[:, None])[None]
    K = np.broadcast_to(K, (nz, ny, nx)).copy()
    mass = 9.0e4 + 1.0e3 * rng.standard_normal((nz, ny, nx))

    tend = horizontal_diffusion_coord_scalar_tendency(
        jnp.asarray(field), jnp.asarray(K), jnp.asarray(mass), dx_m=dx, dy_m=dy
    )
    tend_o = _hdiff_oracle(field, K, mass, dx, dy)
    assert np.allclose(np.asarray(tend), tend_o, rtol=1e-11, atol=1e-12)


def test_coord_scalar_conserves_mass_weighted_integral():
    """Periodic flux divergence has zero domain-sum (conserves the field integral)."""

    rng = np.random.default_rng(2)
    nz, ny, nx = 4, 16, 18
    dx, dy = 1000.0, 1000.0
    field = 300.0 + rng.standard_normal((nz, ny, nx))
    K = 60.0 + 20.0 * rng.random((nz, ny, nx))
    mass = 9.0e4 + 1.0e3 * rng.standard_normal((nz, ny, nx))
    tend = horizontal_diffusion_coord_scalar_tendency(
        jnp.asarray(field), jnp.asarray(K), jnp.asarray(mass), dx_m=dx, dy_m=dy
    )
    # the tendency is already mass-coupled (d(mass*field)/dt); periodic flux
    # divergence sums to ~0 per level -> conserves the integral of mass*field.
    per_level = np.asarray(jnp.sum(tend, axis=(1, 2)))
    assert np.allclose(per_level, 0.0, atol=1e-6), f"max|level sum|={np.max(np.abs(per_level))}"


def test_coord_scalar_down_gradient():
    """A single-mode bump is diffused down-gradient (decays the mode)."""

    nz, ny, nx = 4, 1, 32
    dx, dy = 500.0, 500.0
    xc = (np.arange(nx) + 0.5) * dx
    kx = 2.0 * np.pi / (nx * dx)
    field = np.broadcast_to(np.sin(kx * xc)[None, None, :], (nz, ny, nx)).copy()
    K = np.full((nz, ny, nx), 75.0)
    mass = np.full((nz, ny, nx), 9.0e4)
    tend = np.asarray(horizontal_diffusion_coord_scalar_tendency(
        jnp.asarray(field), jnp.asarray(K), jnp.asarray(mass), dx_m=dx, dy_m=dy
    ))
    corr = np.sum(tend[0, 0, :] * field[0, 0, :])
    assert corr < 0.0, f"expected down-gradient (negative correlation), got {corr}"


def test_coord_scalar_perturbation_base_offset():
    """base_3d subtraction: diffusing field == diffusing field+const with base offset."""

    rng = np.random.default_rng(3)
    nz, ny, nx = 3, 12, 14
    dx = dy = 1000.0
    pert = rng.standard_normal((nz, ny, nx))
    base = 300.0 + np.linspace(-1, 1, nz)[:, None, None] * np.ones((nz, ny, nx))
    field = base + pert
    K = np.full((nz, ny, nx), 50.0)
    mass = np.full((nz, ny, nx), 9.0e4)
    # diffusing (field - base) should equal diffusing pert directly.
    t_base = horizontal_diffusion_coord_scalar_tendency(
        jnp.asarray(field), jnp.asarray(K), jnp.asarray(mass), dx_m=dx, dy_m=dy,
        base_3d=jnp.asarray(base),
    )
    t_pert = horizontal_diffusion_coord_scalar_tendency(
        jnp.asarray(pert), jnp.asarray(K), jnp.asarray(mass), dx_m=dx, dy_m=dy,
    )
    assert np.allclose(np.asarray(t_base), np.asarray(t_pert), rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. Momentum diffusion staggers
# ---------------------------------------------------------------------------


def test_coord_momentum_uniform_flow_zero():
    """Uniform u/v/w -> zero horizontal diffusion (no spurious source)."""

    nz, ny, nx = 4, 12, 16
    u = jnp.full((nz, ny, nx + 1), 7.0, dtype=jnp.float64)
    v = jnp.full((nz, ny + 1, nx), -2.0, dtype=jnp.float64)
    w = jnp.full((nz + 1, ny, nx), 0.5, dtype=jnp.float64)
    K = jnp.full((nz, ny, nx), 60.0, dtype=jnp.float64)
    mass_u = jnp.full((nz, ny, nx + 1), 9.0e4, dtype=jnp.float64)
    mass_v = jnp.full((nz, ny + 1, nx), 9.0e4, dtype=jnp.float64)
    mass_f = jnp.full((nz + 1, ny, nx), 9.0e4, dtype=jnp.float64)
    du, dv, dw = horizontal_diffusion_coord_momentum_tendency(
        u, v, w, K, mass_u, mass_v, mass_f, dx_m=1000.0, dy_m=1000.0
    )
    assert np.allclose(np.asarray(du), 0.0, atol=1e-9)
    assert np.allclose(np.asarray(dv), 0.0, atol=1e-9)
    assert np.allclose(np.asarray(dw), 0.0, atol=1e-9)


def test_coord_momentum_shapes_preserved():
    """Output staggers match the input staggers (periodic wrap faces restored)."""

    nz, ny, nx = 3, 10, 12
    rng = np.random.default_rng(4)
    u = jnp.asarray(rng.standard_normal((nz, ny, nx + 1)))
    v = jnp.asarray(rng.standard_normal((nz, ny + 1, nx)))
    w = jnp.asarray(rng.standard_normal((nz + 1, ny, nx)))
    K = jnp.asarray(40.0 + 10.0 * rng.random((nz, ny, nx)))
    mass_u = jnp.asarray(9.0e4 + rng.standard_normal((nz, ny, nx + 1)))
    mass_v = jnp.asarray(9.0e4 + rng.standard_normal((nz, ny + 1, nx)))
    mass_f = jnp.asarray(9.0e4 + rng.standard_normal((nz + 1, ny, nx)))
    du, dv, dw = horizontal_diffusion_coord_momentum_tendency(
        u, v, w, K, mass_u, mass_v, mass_f, dx_m=900.0, dy_m=900.0
    )
    assert du.shape == u.shape
    assert dv.shape == v.shape
    assert dw.shape == w.shape


def test_operators_are_jittable():
    """All public operators trace cleanly under jit (no host transfer / python branch)."""

    nz, ny, nx = 3, 8, 10
    u = jnp.ones((nz, ny, nx + 1))
    v = jnp.ones((nz, ny + 1, nx))

    @jax.jit
    def go(u, v):
        d11, d22, d12 = horizontal_deformation_2d(u, v, dx_m=1000.0, dy_m=1000.0)
        xkmh, xkhh = smag2d_horizontal_km(d11, d22, d12, dx_m=1000.0, dy_m=1000.0)
        return jnp.sum(xkmh) + jnp.sum(xkhh)

    out = go(u, v)
    assert np.isfinite(float(out))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
