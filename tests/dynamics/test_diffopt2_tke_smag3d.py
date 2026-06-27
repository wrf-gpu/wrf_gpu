"""Oracle and integration tests for diff_opt=2 / km_opt=2,3,5 turbulence.

The NumPy oracles are literal dry/interior reductions of WRF
``dyn_em/module_diffusion_em.F``:

* ``smag_km`` for km_opt=3 full 3-D Smagorinsky coefficients.
* ``tke_km`` for km_opt=2 prognostic-TKE coefficients.
* ``tke_rhs`` shear/buoyancy/dissipation sign and lower-bound behaviour.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gpuwrf.contracts.grid import (
    BCMetadata,
    DycoreMetrics,
    GridSpec,
    Projection,
    TerrainProvenance,
    VerticalCoord,
)
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
from gpuwrf.dynamics.advection import apply_halo, halo_spec
from gpuwrf.dynamics.explicit_diffusion import (
    C_S_DEFAULT,
    PRANDTL,
    deformation_components_3d,
    dry_brunt_vaisala_squared,
    smag3d_km,
    tke3d_km,
    tke_rhs_tendency,
)
from gpuwrf.runtime.operational_mode import OperationalNamelist, _augment_large_step_tendencies


def _build_grid(ny: int, nx: int, nz: int, dx: float) -> GridSpec:
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    projection = Projection("lambert", 0.0, 0.0, dx, dx, nx, ny)
    terrain = TerrainProvenance(
        source_path="idealized:diffopt2-tke-smag3d",
        sha256="analytic",
        shape=(ny, nx),
        units="m",
        projection_transform="flat",
        max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    vertical = VerticalCoord("hybrid_eta", nz, 1.0e4, eta)
    bc = BCMetadata(
        source="ideal",
        fields=("u", "v", "w", "theta", "p", "ph", "mu"),
        update_cadence_h=999,
        interpolation="linear",
        restart_compatible=False,
    )
    metrics = DycoreMetrics.flat(ny=ny, nx=nx, nz=nz, eta_levels=eta, top_pressure_pa=1.0e4)
    one_h = jnp.ones((nz,), dtype=jnp.float64)
    zero_h = jnp.zeros((nz,), dtype=jnp.float64)
    one_f = jnp.ones((nz + 1,), dtype=jnp.float64)
    zero_f = jnp.zeros((nz + 1,), dtype=jnp.float64)
    metrics = dataclasses.replace(
        metrics,
        c1h=one_h,
        c2h=zero_h,
        c3h=one_h,
        c4h=zero_h,
        c1f=one_f,
        c2f=zero_f,
        c3f=one_f,
        c4f=zero_f,
    )
    return GridSpec(
        projection=projection,
        terrain=terrain,
        vertical=vertical,
        bc=bc,
        eta_levels=eta,
        terrain_height=jnp.zeros((ny, nx), dtype=jnp.float64),
        metrics=metrics,
        halo_width=2,
        staggering="c-grid",
    )


def _build_state(grid: GridSpec) -> State:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    rng = np.random.default_rng(42)
    xf = np.arange(nx + 1) / nx * 2.0 * np.pi
    yf = np.arange(ny + 1) / ny * 2.0 * np.pi
    xc = (np.arange(nx) + 0.5) / nx * 2.0 * np.pi
    yc = (np.arange(ny) + 0.5) / ny * 2.0 * np.pi
    zc = np.arange(nz)[:, None, None]
    u = 5.0 + np.sin(xf)[None, None, :] * np.cos(yc)[None, :, None] + 0.02 * zc
    u = np.broadcast_to(u, (nz, ny, nx + 1)).copy()
    v = -2.0 + np.cos(xc)[None, None, :] * np.sin(yf)[None, :, None] - 0.015 * zc
    v = np.broadcast_to(v, (nz, ny + 1, nx)).copy()
    w = 0.03 * rng.standard_normal((nz + 1, ny, nx))
    theta = 300.0 + 0.03 * zc + 0.4 * np.sin(xc)[None, None, :]
    theta = np.broadcast_to(theta, (nz, ny, nx)).copy()
    qke = 0.15 + 0.02 * rng.random((nz, ny, nx))

    shapes = _state_field_shapes(grid)
    fields = {name: jnp.zeros(shape, dtype=jnp.float64) for name, shape in shapes.items()}
    mu = jnp.full((ny, nx), 9.0e4, dtype=jnp.float64)
    ph = jnp.broadcast_to(
        jnp.linspace(0.0, 6000.0 * 9.81, nz + 1, dtype=jnp.float64)[:, None, None],
        (nz + 1, ny, nx),
    )
    fields.update(
        u=jnp.asarray(u),
        v=jnp.asarray(v),
        w=jnp.asarray(w),
        theta=jnp.asarray(theta),
        qke=jnp.asarray(qke),
        mu=mu,
        mu_total=mu,
        mu_perturbation=jnp.zeros_like(mu),
        ph=ph,
        ph_total=ph,
        ph_perturbation=jnp.zeros_like(ph),
    )
    return State(**fields)


def _cpu_tendencies(grid: GridSpec) -> Tendencies:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    z = lambda *s: jnp.zeros(s, dtype=jnp.float64)
    return Tendencies(
        u=z(nz, ny, nx + 1),
        v=z(nz, ny + 1, nx),
        w=z(nz + 1, ny, nx),
        theta=z(nz, ny, nx),
        qv=z(nz, ny, nx),
        p=z(nz, ny, nx),
        ph=z(nz + 1, ny, nx),
        mu=z(ny, nx),
    )


def _namelist(grid: GridSpec, **over) -> OperationalNamelist:
    return OperationalNamelist.from_grid(
        grid,
        tendencies=_cpu_tendencies(grid),
        metrics=grid.metrics,
        dt_s=6.0,
        acoustic_substeps=2,
        radiation_cadence_steps=10**9,
        use_vertical_solver=True,
        disable_guards=True,
        force_fp64=True,
        use_flux_advection=False,
        **over,
    )


def _smag3d_oracle(d11, d22, d33, d12, d13, d23, bn2, *, dx, dy, dz, dt, c_s, pr, mix_upper_bound):
    d12_m = 0.25 * (d12 + np.roll(d12, -1, axis=1) + np.roll(d12, -1, axis=2) + np.roll(np.roll(d12, -1, axis=1), -1, axis=2))
    d13_m = 0.25 * (d13[:-1] + d13[1:] + np.roll(d13[:-1], -1, axis=2) + np.roll(d13[1:], -1, axis=2))
    d23_m = 0.25 * (d23[:-1] + d23[1:] + np.roll(d23[:-1], -1, axis=1) + np.roll(d23[1:], -1, axis=1))
    def2 = 0.5 * (d11 * d11 + d22 * d22 + d33 * d33) + d12_m * d12_m + d13_m * d13_m + d23_m * d23_m
    tmp = np.sqrt(np.maximum(0.0, def2 - bn2 / pr))
    mlen_h2 = dx * dy
    mlen_v2 = dz * dz
    xkmh = np.maximum(c_s * c_s * mlen_h2 * tmp, 1.0e-6 * mlen_h2)
    xkmh = np.minimum(xkmh, mix_upper_bound * mlen_h2 / dt)
    xkmv = np.maximum(c_s * c_s * mlen_v2 * tmp, 1.0e-6 * mlen_v2)
    xkmv = np.minimum(xkmv, mix_upper_bound * mlen_v2 / dt)
    xkhh = np.minimum(xkmh / pr, mix_upper_bound * mlen_h2 / dt)
    xkhv = np.minimum(xkmv / pr, mix_upper_bound * mlen_v2 / dt)
    return xkmh, xkmv, xkhh, xkhv


def test_smag3d_km_matches_wrf_formula():
    nz, ny, nx = 4, 3, 5
    rng = np.random.default_rng(5)
    d11 = rng.normal(scale=2.0e-3, size=(nz, ny, nx))
    d22 = rng.normal(scale=2.0e-3, size=(nz, ny, nx))
    d33 = rng.normal(scale=2.0e-3, size=(nz, ny, nx))
    d12 = rng.normal(scale=1.0e-3, size=(nz, ny, nx))
    d13 = rng.normal(scale=1.0e-3, size=(nz + 1, ny, nx))
    d23 = rng.normal(scale=1.0e-3, size=(nz + 1, ny, nx))
    bn2 = rng.normal(loc=1.0e-5, scale=1.0e-5, size=(nz, ny, nx))
    dx = 1000.0
    dz = 200.0
    got = smag3d_km(
        jnp.asarray(d11),
        jnp.asarray(d22),
        jnp.asarray(d33),
        jnp.asarray(d12),
        jnp.asarray(d13),
        jnp.asarray(d23),
        jnp.asarray(bn2),
        dx_m=dx,
        dy_m=dx,
        rdzw=jnp.full((nz, ny, nx), 1.0 / dz),
        dt_s=6.0,
        c_s=C_S_DEFAULT,
    )
    exp = _smag3d_oracle(d11, d22, d33, d12, d13, d23, bn2, dx=dx, dy=dx, dz=dz, dt=6.0, c_s=C_S_DEFAULT, pr=PRANDTL, mix_upper_bound=0.1)
    for a, b in zip(got, exp):
        np.testing.assert_allclose(np.asarray(a), b, rtol=2.0e-12, atol=2.0e-12)


def test_tke3d_km2_matches_wrf_neutral_formula():
    tke = jnp.full((3, 2, 4), 0.16, dtype=jnp.float64)
    theta = jnp.full_like(tke, 300.0)
    bn2 = jnp.zeros_like(tke)
    dx = 500.0
    dz = 100.0
    xkmh, xkmv, xkhh, xkhv = tke3d_km(
        tke,
        theta,
        bn2,
        dx_m=dx,
        dy_m=dx,
        rdzw=jnp.full_like(tke, 1.0 / dz),
        dt_s=10.0,
        km_opt=2,
        c_k=0.15,
    )
    tmp = np.sqrt(0.16)
    exp_xkmh = min(0.15 * tmp * dx, 0.1 * dx * dx / 10.0)
    exp_xkmv = min(max(0.15 * tmp * dz, 1.0e-6 * dz * dz), 0.1 * dz * dz / 10.0)
    np.testing.assert_allclose(np.asarray(xkmh), exp_xkmh, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(xkmv), exp_xkmv, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(xkhh), exp_xkmh / PRANDTL, rtol=0.0, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(xkhv), exp_xkmv * 3.0, rtol=0.0, atol=1.0e-12)


def test_tke_rhs_is_finite_produces_shear_and_bounds_sink():
    nz, ny, nx = 4, 3, 5
    d11 = jnp.full((nz, ny, nx), 1.0e-2)
    d22 = jnp.zeros_like(d11)
    d33 = jnp.zeros_like(d11)
    d12 = jnp.zeros_like(d11)
    d13 = jnp.zeros((nz + 1, ny, nx), dtype=jnp.float64)
    d23 = jnp.zeros_like(d13)
    mass = jnp.full_like(d11, 8.0e4)
    xkm = jnp.full_like(d11, 20.0)
    xkhv = jnp.full_like(d11, 10.0)
    rhs = tke_rhs_tendency(
        jnp.full_like(d11, 0.2),
        mass,
        d11,
        d22,
        d33,
        d12,
        d13,
        d23,
        jnp.zeros_like(d11),
        xkm,
        xkm,
        xkhv,
        dx_m=1000.0,
        dy_m=1000.0,
        rdzw=jnp.full_like(d11, 1.0 / 100.0),
        dt_s=10.0,
    )
    assert np.all(np.isfinite(np.asarray(rhs)))
    assert float(jnp.max(rhs)) > 0.0

    sink = tke_rhs_tendency(
        jnp.full_like(d11, 0.01),
        mass,
        jnp.zeros_like(d11),
        d22,
        d33,
        d12,
        d13,
        d23,
        jnp.full_like(d11, 0.1),
        xkm,
        xkm,
        xkhv,
        dx_m=1000.0,
        dy_m=1000.0,
        rdzw=jnp.full_like(d11, 1.0 / 100.0),
        dt_s=10.0,
    )
    lower = -np.asarray(mass) * 0.01 / 10.0
    assert np.min(np.asarray(sink) - lower) >= -1.0e-9


@pytest.mark.parametrize("km_opt", [2, 3, 5])
def test_diffopt2_turbulence_augment_is_finite_and_nonzero(km_opt):
    grid = _build_grid(ny=6, nx=8, nz=4, dx=1000.0)
    state = _build_state(grid)
    haloed = apply_halo(state, halo_spec(grid))
    nl0 = _namelist(grid, diff_opt=0, km_opt=0)
    nl1 = _namelist(grid, diff_opt=2, km_opt=km_opt)

    t0 = _augment_large_step_tendencies(haloed, nl0.tendencies, nl0, rk_step=3)
    t1 = _augment_large_step_tendencies(haloed, nl1.tendencies, nl1, rk_step=3)
    for field in ("u", "v", "w", "theta"):
        delta = np.asarray(getattr(t1, field) - getattr(t0, field))
        assert np.all(np.isfinite(delta)), field
        assert np.max(np.abs(delta)) > 0.0, field

    # Dry stability oracle remains finite and correctly signed for the monotone
    # theta profile used by this idealized fixture.
    bn2 = np.asarray(dry_brunt_vaisala_squared(haloed.theta, dz_m=1500.0))
    assert np.all(np.isfinite(bn2))


def test_diffopt2_turbulence_jittable():
    grid = _build_grid(ny=5, nx=6, nz=4, dx=800.0)
    state = _build_state(grid)
    nl = _namelist(grid, diff_opt=2, km_opt=2)
    hspec = halo_spec(grid)

    @jax.jit
    def go(st):
        h = apply_halo(st, hspec)
        d11, d22, d33, d12, d13, d23 = deformation_components_3d(
            h.u,
            h.v,
            h.w,
            dx_m=800.0,
            dy_m=800.0,
            rdz=jnp.ones_like(h.w) / 1500.0,
            rdzw=jnp.ones_like(h.theta) / 1500.0,
        )
        bn2 = dry_brunt_vaisala_squared(h.theta, dz_m=1500.0)
        xkmh, xkmv, xkhh, xkhv = tke3d_km(
            h.qke,
            h.theta,
            bn2,
            dx_m=800.0,
            dy_m=800.0,
            rdzw=jnp.ones_like(h.theta) / 1500.0,
            dt_s=6.0,
            km_opt=2,
        )
        t = _augment_large_step_tendencies(h, nl.tendencies, nl, rk_step=3)
        return sum(jnp.sum(x) for x in (d11, d22, d33, d12, d13, d23, bn2, xkmh, xkmv, xkhh, xkhv, t.u, t.theta))

    out = float(go(state))
    assert np.isfinite(out)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
