"""Analytic-oracle tests for the WRF deformation-tensor momentum diffusion.

Source operator: ``gpuwrf.dynamics.explicit_diffusion.
wrf_deformation_momentum_tendency`` -- the flat-slab reduction of WRF
``horizontal_diffusion_{u,w}_2`` + ``vertical_diffusion_{u,w}_2``
(``module_diffusion_em.F``).  For constant density ``rho`` and constant ``K``
the density-weighted stress divergence reduces to

    du/dt = rho*K*( 2 u_xx + u_zz + w_xz )
    dw/dt = rho*K*( w_xx + 2 w_zz + u_xz )

(the diagonal factor 2 from D11=2u_x / D33=2w_z, plus the off-diagonal cross
terms w_xz in u and u_xz in w that the scalar Laplacian omits).  These tests
build smooth periodic-x analytic fields and compare the operator to a
second-order finite-difference oracle of that closed form.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gpuwrf.dynamics.explicit_diffusion import wrf_deformation_momentum_tendency


def _fd_oracle(u_mass, w, dx, dz, K, rho):
    """Second-order FD of rho*K*(2 u_xx+u_zz+w_xz) and rho*K*(w_xx+2 w_zz+u_xz)."""

    nz = w.shape[0] - 1
    nx = w.shape[-1]
    # u_xx periodic on mass-x
    u_xx = (np.roll(u_mass, -1, axis=2) - 2.0 * u_mass + np.roll(u_mass, 1, axis=2)) / dx**2
    # u_zz interior mass levels (rigid -> 0 at 0,nz-1)
    u_zz = np.zeros_like(u_mass)
    u_zz[1:-1] = (u_mass[2:] - 2.0 * u_mass[1:-1] + u_mass[:-2]) / dz**2
    # w on w faces -> w_xz at mass levels: d/dz(dw/dx) ; dw/dx at w faces centered
    wx = (np.roll(w, -1, axis=2) - np.roll(w, 1, axis=2)) / (2.0 * dx)
    w_xz = (wx[1 : nz + 1] - wx[0:nz]) / dz  # mass levels
    # NOTE: rho is constant here, so rho on mass levels == rho on w faces and the
    # closed-form below is an independent textbook deformation operator (not a copy
    # of the JAX operator's staggering).
    du = rho * K * (2.0 * u_xx + u_zz + w_xz)

    # w_xx periodic on w faces
    w_xx = (np.roll(w, -1, axis=2) - 2.0 * w + np.roll(w, 1, axis=2)) / dx**2
    # w_zz interior w faces
    w_zz = np.zeros_like(w)
    w_zz[1:nz] = (w[2 : nz + 1] - 2.0 * w[1:nz] + w[0 : nz - 1]) / dz**2
    # u_xz at w faces: d/dx(du/dz); du/dz at w faces 1..nz-1
    uz = np.zeros_like(w)
    uz[1:nz] = (u_mass[1:nz] - u_mass[0 : nz - 1]) / dz
    u_xz = (np.roll(uz, -1, axis=2) - np.roll(uz, 1, axis=2)) / (2.0 * dx)
    rho_w = rho[0, 0, 0]  # constant
    dw = rho_w * K * (w_xx + 2.0 * w_zz + u_xz)
    return du, dw


def test_deformation_matches_fd_oracle_interior():
    """Operator equals the FD closed-form oracle in the vertical interior."""

    nz, nx = 24, 32
    dx, dz = 100.0, 100.0
    K = 75.0
    rho_val = 1.1
    x = (np.arange(nx) + 0.5) * dx
    z = (np.arange(nz) + 0.5) * dz
    zf = np.arange(nz + 1) * dz
    kx = 2.0 * np.pi / (nx * dx)
    # smooth periodic-x fields with vertical structure (shape (nz,1,nx) / (nz+1,1,nx))
    u_mass = (np.sin(kx * x)[None, :] * np.cos(np.pi * z / (nz * dz))[:, None])[:, None, :].copy()
    w = (np.cos(kx * x)[None, :] * np.sin(np.pi * zf / (nz * dz))[:, None])[:, None, :].copy()
    rho = np.full((nz, 1, nx), rho_val)

    u_face = np.concatenate([u_mass, u_mass[:, :, :1]], axis=2)  # (nz,1,nx+1) periodic
    du, dw = wrf_deformation_momentum_tendency(
        jnp.asarray(u_face), jnp.asarray(w), rho=jnp.asarray(rho),
        k_m2_s=K, dx_m=dx, dz_m=dz,
    )
    du = np.asarray(du)[:, :, :nx]
    dw = np.asarray(dw)
    du_o, dw_o = _fd_oracle(u_mass, w, dx, dz, K, rho)

    # du matches the closed-form oracle to round-off (identical compact stencils).
    r = slice(3, nz - 3)
    assert np.allclose(du[r], du_o[r], rtol=1e-9, atol=1e-9), (
        f"du max|d|={np.max(np.abs(du[r]-du_o[r]))}"
    )
    # dw: the WRF flux-divergence assembles titau13 at w-faces then differences in x
    # (two-stage centered d/dx), which is a *different but equally 2nd-order* stencil
    # for the w_xx cross piece than the compact 3-point oracle.  They agree to ~1%
    # at this resolution and converge at 2nd order (verified separately).  Require
    # close agreement with a tolerance that admits the valid stencil difference.
    rf = slice(3, nz - 2)
    rel = np.max(np.abs(dw[rf] - dw_o[rf])) / np.max(np.abs(dw_o[rf]))
    assert rel < 0.02, f"dw relative diff {rel} exceeds 2% (stencil mismatch?)"


def test_deformation_dw_cross_term_converges_second_order():
    """The dw cross-term stencil converges to the closed form at 2nd order."""

    def err(nx):
        nz = 24
        dx = 3200.0 / nx
        dz = 100.0
        K = 75.0
        x = (np.arange(nx) + 0.5) * dx
        z = (np.arange(nz) + 0.5) * dz
        zf = np.arange(nz + 1) * dz
        kx = 2.0 * np.pi / (nx * dx)
        u_mass = (np.sin(kx * x)[None, :] * np.cos(np.pi * z / (nz * dz))[:, None])[:, None, :].copy()
        w = (np.cos(kx * x)[None, :] * np.sin(np.pi * zf / (nz * dz))[:, None])[:, None, :].copy()
        rho = np.ones((nz, 1, nx))
        u_face = np.concatenate([u_mass, u_mass[:, :, :1]], axis=2)
        _, dw = wrf_deformation_momentum_tendency(
            jnp.asarray(u_face), jnp.asarray(w), rho=jnp.asarray(rho), k_m2_s=K, dx_m=dx, dz_m=dz
        )
        _, dw_o = _fd_oracle(u_mass, w, dx, dz, K, rho)
        rf = slice(3, nz - 2)
        return np.max(np.abs(np.asarray(dw)[rf] - dw_o[rf]))

    e1 = err(32)
    e2 = err(64)
    # 2nd-order: halving dx reduces the stencil-difference error by ~4x.
    assert e2 < e1 / 3.0, f"convergence rate too slow: e1={e1}, e2={e2}, ratio={e1/e2}"


def test_deformation_is_down_gradient():
    """A single-mode bump diffuses down-gradient (sign check)."""

    nz, nx = 20, 32
    dx, dz = 100.0, 100.0
    K = 75.0
    x = (np.arange(nx) + 0.5) * dx
    kx = 2.0 * np.pi / (nx * dx)
    u_mass = np.broadcast_to(np.sin(kx * x)[None, None, :], (nz, 1, nx)).copy()
    w = np.zeros((nz + 1, 1, nx))
    rho = np.ones((nz, 1, nx))
    u_face = np.concatenate([u_mass, u_mass[:, :, :1]], axis=2)
    du, _ = wrf_deformation_momentum_tendency(
        jnp.asarray(u_face), jnp.asarray(w), rho=jnp.asarray(rho),
        k_m2_s=K, dx_m=dx, dz_m=dz,
    )
    du = np.asarray(du)[:, :, :nx]
    # down-gradient: du/dt should be opposite-sign to u (decays the mode).
    mid = nz // 2
    corr = np.sum(du[mid, 0, :] * u_mass[mid, 0, :])
    assert corr < 0.0, f"expected down-gradient (negative correlation), got {corr}"


def test_deformation_zero_on_uniform_flow():
    """Uniform u, zero w -> zero diffusion tendency (no spurious source)."""

    nz, nx = 16, 24
    dx, dz = 100.0, 100.0
    u_face = jnp.ones((nz, 1, nx + 1), dtype=jnp.float64) * 7.3
    w = jnp.zeros((nz + 1, 1, nx), dtype=jnp.float64)
    rho = jnp.ones((nz, 1, nx), dtype=jnp.float64)
    du, dw = wrf_deformation_momentum_tendency(u_face, w, rho=rho, k_m2_s=75.0, dx_m=dx, dz_m=dz)
    assert np.allclose(np.asarray(du), 0.0, atol=1e-12)
    assert np.allclose(np.asarray(dw), 0.0, atol=1e-12)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
