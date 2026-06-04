"""CPU integration test: diff_opt=1/km_opt=4 wired into the large-step tendencies.

Verifies (CPU fp64, no GPU needed):
  1. The diff_opt=0 baseline tendency is BIT-IDENTICAL with and without the new
     code present -- i.e. the new branch is inert when not selected (no regression
     to the existing diff_opt=2 / const-K idealized path).
  2. diff_opt=1/km_opt=4 adds a finite, down-gradient horizontal-diffusion
     contribution to a sheared velocity field, and the whole augment step is
     jit-traceable (no host transfer / python-level data branch in the loop).
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
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    _augment_large_step_tendencies,
)


def _build_grid(ny: int, nx: int, nz: int, dx: float) -> GridSpec:
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    top_p = 1.0e4
    projection = Projection("lambert", 0.0, 0.0, dx, dx, nx, ny)
    terrain = TerrainProvenance(
        source_path="idealized:smag-integration",
        sha256="analytic",
        shape=(ny, nx),
        units="m",
        projection_transform="flat",
        max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    vertical = VerticalCoord("hybrid_eta", nz, 1.0e4, eta)
    bc = BCMetadata(
        source="ideal", fields=("u", "v", "w", "theta", "p", "ph", "mu"),
        update_cadence_h=999, interpolation="linear", restart_compatible=False,
    )
    m = DycoreMetrics.flat(ny=ny, nx=nx, nz=nz, eta_levels=eta, top_pressure_pa=top_p)
    # pure-sigma c1/c2 (idealized hybrid_opt=0) so the top-face mass is nonzero.
    one_h = jnp.ones((nz,), dtype=jnp.float64)
    zero_h = jnp.zeros((nz,), dtype=jnp.float64)
    one_f = jnp.ones((nz + 1,), dtype=jnp.float64)
    zero_f = jnp.zeros((nz + 1,), dtype=jnp.float64)
    m = dataclasses.replace(
        m, c1h=one_h, c2h=zero_h, c3h=one_h, c4h=zero_h,
        c1f=one_f, c2f=zero_f, c3f=one_f, c4f=zero_f,
    )
    return GridSpec(
        projection=projection, terrain=terrain, vertical=vertical, bc=bc,
        eta_levels=eta, terrain_height=jnp.zeros((ny, nx), dtype=jnp.float64),
        metrics=m, halo_width=2, staggering="c-grid",
    )


def _build_state(grid: GridSpec) -> State:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    rng = np.random.default_rng(7)
    xf = (np.arange(nx + 1)) / nx * 2 * np.pi
    yc = (np.arange(ny) + 0.5) / ny * 2 * np.pi
    xc = (np.arange(nx) + 0.5) / nx * 2 * np.pi
    yf = (np.arange(ny)) / ny * 2 * np.pi  # ny rows for v faces interior
    # sheared, divergent-free-ish smooth velocity (gives nonzero D11/D22/D12).
    u = (np.sin(xf)[None, :] * np.cos(yc)[:, None])[None]
    u = np.broadcast_to(u, (nz, ny, nx + 1)).copy() + 5.0
    v = np.zeros((nz, ny + 1, nx))
    v[:, :ny, :] = (np.cos(xc)[None, :] * np.sin(yf)[:, None])[None]
    w = 0.1 * rng.standard_normal((nz + 1, ny, nx))
    theta = 300.0 + 0.5 * (np.sin(xc)[None, :] * np.ones((ny, 1)))[None] * np.ones((nz, 1, 1))

    # Build the State directly on the CPU device (State.zeros mandates a GPU).
    shapes = _state_field_shapes(grid)
    fields = {name: jnp.zeros(shape, dtype=jnp.float64) for name, shape in shapes.items()}
    mu = jnp.full((ny, nx), 9.0e4, dtype=jnp.float64)
    ph = jnp.broadcast_to(
        jnp.linspace(0.0, 1.0e5, nz + 1, dtype=jnp.float64)[:, None, None], (nz + 1, ny, nx)
    )
    fields.update(
        u=jnp.asarray(u), v=jnp.asarray(v), w=jnp.asarray(w),
        theta=jnp.asarray(theta),
        mu=mu, mu_total=mu, mu_perturbation=jnp.zeros_like(mu),
        ph=ph, ph_total=ph, ph_perturbation=jnp.zeros_like(ph),
    )
    return State(**fields)


def _cpu_tendencies(grid: GridSpec) -> Tendencies:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    z = lambda *s: jnp.zeros(s, dtype=jnp.float64)
    return Tendencies(
        u=z(nz, ny, nx + 1), v=z(nz, ny + 1, nx), w=z(nz + 1, ny, nx),
        theta=z(nz, ny, nx), qv=z(nz, ny, nx), p=z(nz, ny, nx),
        ph=z(nz + 1, ny, nx), mu=z(ny, nx),
    )


def _namelist(grid: GridSpec, **over) -> OperationalNamelist:
    return OperationalNamelist.from_grid(
        grid, tendencies=_cpu_tendencies(grid), metrics=grid.metrics,
        dt_s=10.0, acoustic_substeps=4,
        radiation_cadence_steps=10**9, use_vertical_solver=True,
        disable_guards=True, force_fp64=True, use_flux_advection=True, **over,
    )


def test_baseline_unchanged_when_smag_not_selected():
    """diff_opt=0 large-step tendency is identical to diff_opt=1/km_opt=4-disabled run."""

    grid = _build_grid(ny=12, nx=16, nz=6, dx=4000.0)
    state = _build_state(grid)
    haloed = apply_halo(state, halo_spec(grid))

    nl0 = _namelist(grid)  # diff_opt=0/km_opt=0 default
    t0 = _augment_large_step_tendencies(haloed, nl0.tendencies, nl0, rk_step=3)

    # Same namelist but explicitly diff_opt=0 (km_opt arbitrary) -> identical.
    nl_off = _namelist(grid, diff_opt=0, km_opt=4)
    t_off = _augment_large_step_tendencies(haloed, nl_off.tendencies, nl_off, rk_step=3)

    for f in ("u", "v", "w", "theta"):
        a = np.asarray(getattr(t0, f))
        b = np.asarray(getattr(t_off, f))
        assert np.array_equal(a, b), f"{f} changed when smag NOT selected"


def test_smag_adds_finite_down_gradient_diffusion():
    """diff_opt=1/km_opt=4 adds a finite, down-gradient diffusion to the baseline."""

    grid = _build_grid(ny=12, nx=16, nz=6, dx=4000.0)
    state = _build_state(grid)
    haloed = apply_halo(state, halo_spec(grid))

    nl0 = _namelist(grid, diff_opt=0, km_opt=0)
    nl1 = _namelist(grid, diff_opt=1, km_opt=4)

    t0 = _augment_large_step_tendencies(haloed, nl0.tendencies, nl0, rk_step=3)
    t1 = _augment_large_step_tendencies(haloed, nl1.tendencies, nl1, rk_step=3)

    du = np.asarray(t1.u) - np.asarray(t0.u)
    dth = np.asarray(t1.theta) - np.asarray(t0.theta)
    # diffusion contribution is nonzero and finite.
    assert np.all(np.isfinite(du)) and np.all(np.isfinite(dth))
    assert np.max(np.abs(du)) > 0.0, "smag added no u diffusion"
    assert np.max(np.abs(dth)) > 0.0, "smag added no theta diffusion"

    # down-gradient: the added theta diffusion (mass-coupled) opposes the theta
    # perturbation -> negative correlation with (theta - theta_mean) per column.
    th = np.asarray(haloed.theta)
    th_pert = th - th.mean(axis=2, keepdims=True)
    corr = np.sum(dth * th_pert)
    assert corr < 0.0, f"theta diffusion not down-gradient (corr={corr})"


def test_smag_augment_is_jittable():
    """The full augment step with diff_opt=1/km_opt=4 traces under jit."""

    grid = _build_grid(ny=8, nx=10, nz=5, dx=4000.0)
    state = _build_state(grid)
    nl1 = _namelist(grid, diff_opt=1, km_opt=4)
    hspec = halo_spec(grid)

    @jax.jit
    def go(st):
        h = apply_halo(st, hspec)
        t = _augment_large_step_tendencies(h, nl1.tendencies, nl1, rk_step=3)
        return jnp.sum(t.u) + jnp.sum(t.v) + jnp.sum(t.w) + jnp.sum(t.theta)

    out = float(go(state))
    assert np.isfinite(out)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
