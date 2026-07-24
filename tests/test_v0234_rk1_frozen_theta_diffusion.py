"""Focused CPU/source gates for WRF's nested RK1 dry diffusion bundle."""

from __future__ import annotations

import hashlib
import inspect
import dataclasses
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.state import BaseState
from gpuwrf.dynamics.advection import apply_halo, halo_spec
from gpuwrf.dynamics.acoustic_wrf import CVPM, P0_PA, R_D
from gpuwrf.runtime.operational_mode import (
    _augment_large_step_tendencies,
    _diffopt1_dry_forward_tendencies,
    _rk_scan_step,
)
from gpuwrf.integration.nested_pipeline import _domain_int, _make_namelist
from tests.dynamics.test_diffopt1_smagorinsky_integration import (
    _build_grid,
    _build_state,
    _namelist,
)

jax.config.update("jax_enable_x64", True)


WRF_MODULE_EM = Path("<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_em.F")
WRF_MODULE_EM_SHA256 = "11105cbf8255f30ca6a44cd7429a92cedce1fb91db6ce90fd7217002a72fb7fa"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distinct_stage_state(origin):
    nz, ny, nx = origin.theta.shape
    yy = jnp.linspace(-1.0, 1.0, ny, dtype=origin.theta.dtype)[None, :, None]
    xx = jnp.linspace(-1.0, 1.0, nx, dtype=origin.theta.dtype)[None, None, :]
    theta_wave = jnp.broadcast_to(0.2 * yy * xx, (nz, ny, nx))
    u_wave = jnp.sin(
        jnp.linspace(0.0, 2.0 * jnp.pi, origin.u.shape[-1], dtype=origin.u.dtype)
    )[None, None, :]
    return origin.replace(
        theta=origin.theta + theta_wave,
        u=origin.u + jnp.broadcast_to(0.35 * u_wave, origin.u.shape),
    )


def test_pristine_wrf_freezes_dry_tendf_at_rk1_and_reuses_it():
    assert _sha256(WRF_MODULE_EM) == WRF_MODULE_EM_SHA256
    source = WRF_MODULE_EM.read_text()
    forward = source.index("forward_step: IF( rk_step == 1 ) THEN")
    theta_diff = source.index("CALL horizontal_diffusion_3dmp", forward)
    forward_end = source.index("END IF forward_step", theta_diff)
    addtend = source.index("SUBROUTINE rk_addtend_dry", forward_end)
    reuse = source.index("t_tend(i,k,j) =  t_tend(i,k,j) +  t_tendf", addtend)
    assert forward < theta_diff < forward_end < addtend < reuse
    for name in ("ru_tendf", "rv_tendf", "rw_tendf", "t_tendf"):
        assert name in source[forward:forward_end] or name == "t_tendf"


def test_frozen_bundle_reuses_rk1_values_for_all_dry_diffusion_tendencies():
    grid = _build_grid(ny=12, nx=16, nz=6, dx=4000.0)
    origin = apply_halo(_build_state(grid), halo_spec(grid))
    stage = _distinct_stage_state(origin)
    namelist = _namelist(grid, diff_opt=1, km_opt=4)

    rk1_forward = _diffopt1_dry_forward_tendencies(origin, namelist)
    stage_recomputed = _diffopt1_dry_forward_tendencies(stage, namelist)
    for frozen, recomputed in zip(rk1_forward, stage_recomputed, strict=True):
        assert np.max(np.abs(np.asarray(frozen - recomputed))) > 0.0

    parent = _augment_large_step_tendencies(
        stage,
        namelist.tendencies,
        namelist,
        rk_step=2,
    )
    candidate = _augment_large_step_tendencies(
        stage,
        namelist.tendencies,
        namelist,
        rk_step=2,
        frozen_diffopt1_tendencies=rk1_forward,
    )

    for name, frozen, recomputed in zip(
        ("u", "v", "w", "theta"), rk1_forward, stage_recomputed, strict=True
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(candidate, name) - getattr(parent, name)),
            np.asarray(frozen - recomputed),
            rtol=0.0,
            atol=2e-12,
        )


def test_candidate_off_path_ignores_frozen_operand_bit_exact():
    grid = _build_grid(ny=8, nx=10, nz=5, dx=4000.0)
    state = apply_halo(_build_state(grid), halo_spec(grid))
    namelist = _namelist(grid, diff_opt=0, km_opt=4)
    sentinel = tuple(
        jnp.full_like(getattr(state, name), jnp.nan)
        for name in ("u", "v", "w", "theta")
    )
    baseline = _augment_large_step_tendencies(
        state,
        namelist.tendencies,
        namelist,
        rk_step=3,
    )
    with_unused_operand = _augment_large_step_tendencies(
        state,
        namelist.tendencies,
        namelist,
        rk_step=3,
        frozen_diffopt1_tendencies=sentinel,
    )
    for name in ("u", "v", "w", "theta", "mu"):
        np.testing.assert_array_equal(
            np.asarray(getattr(with_unused_operand, name)),
            np.asarray(getattr(baseline, name)),
        )


def test_operational_hoist_precedes_stage_and_has_no_observer_surface():
    source = inspect.getsource(_rk_scan_step)
    assert source.count("_diffopt1_dry_forward_tendencies(") == 1
    assert source.index("rk1_forward_diffopt1 =") < source.index("def advance_stage")
    assert "frozen_diffopt1_tendencies=rk1_forward_diffopt1" in source
    for forbidden in (
        "device_get",
        "pure_callback",
        "debug.callback",
        "io_callback",
        "host_callback",
    ):
        assert forbidden not in source


def test_t_init_inversion_is_literal_and_has_no_safety_clamp():
    source = inspect.getsource(_diffopt1_dry_forward_tendencies)
    assert "alb = -dphb / denominator" in source
    assert "alb = dphb / (phm * jnp.log(pfd / pfu))" in source
    assert "pressure_ratio = (pb / P0_PA) ** CVPM" in source
    for forbidden in ("jnp.where", "jnp.maximum", "clip("):
        assert forbidden not in source


def test_forward_dry_bundle_is_jittable_and_finite():
    grid = _build_grid(ny=8, nx=10, nz=5, dx=4000.0)
    namelist = _namelist(grid, diff_opt=1, km_opt=4)
    hspec = halo_spec(grid)

    @jax.jit
    def evaluate(state):
        return _diffopt1_dry_forward_tendencies(apply_halo(state, hspec), namelist)

    results = tuple(np.asarray(value) for value in evaluate(_build_state(grid)))
    assert [value.shape for value in results] == [
        (grid.nz, grid.ny, grid.nx + 1),
        (grid.nz, grid.ny + 1, grid.nx),
        (grid.nz + 1, grid.ny, grid.nx),
        (grid.nz, grid.ny, grid.nx),
    ]
    assert all(np.all(np.isfinite(value)) for value in results)
    assert all(np.max(np.abs(value)) > 0.0 for value in results)


def test_theta_diffusion_uses_wrf_t_init_base_from_retained_base_state():
    grid = _build_grid(ny=8, nx=10, nz=5, dx=4000.0)
    state = apply_halo(_build_state(grid), halo_spec(grid))
    namelist = _namelist(grid, diff_opt=1, km_opt=4)
    base = BaseState(
        pb=jnp.zeros_like(state.p_total),
        phb=jnp.zeros_like(state.ph_total),
        mub=jnp.zeros_like(state.mu_total),
        t0=jnp.full_like(state.theta, 300.0),
        theta_base=state.theta,
    )
    without_t_init = _diffopt1_dry_forward_tendencies(state, namelist)
    with_t_init = _diffopt1_dry_forward_tendencies(
        state,
        namelist,
        base_state=base,
    )
    for fallback, corrected in zip(without_t_init[:3], with_t_init[:3], strict=True):
        np.testing.assert_array_equal(np.asarray(corrected), np.asarray(fallback))
    assert np.max(np.abs(np.asarray(without_t_init[3]))) > 0.0
    np.testing.assert_array_equal(np.asarray(with_t_init[3]), 0.0)


def test_106_leaf_fallback_recovers_same_t_init_from_pb_phb_mub():
    grid = _build_grid(ny=8, nx=10, nz=5, dx=4000.0)
    state = apply_halo(_build_state(grid), halo_spec(grid))
    metrics = grid.metrics
    nz, ny, nx = state.theta.shape
    mub = jnp.full((ny, nx), 9.0e4, dtype=jnp.float64)
    pb = jnp.broadcast_to(
        jnp.linspace(9.0e4, 2.0e4, nz, dtype=jnp.float64)[:, None, None],
        (nz, ny, nx),
    )
    theta_base = jnp.broadcast_to(
        jnp.linspace(298.0, 430.0, nz, dtype=jnp.float64)[:, None, None],
        (nz, ny, nx),
    )
    alb = (R_D / P0_PA) * theta_base * (pb / P0_PA) ** CVPM
    mass_h = metrics.c1h[:, None, None] * mub[None, :, :] + metrics.c2h[:, None, None]
    increments = -metrics.dnw[:, None, None] * mass_h * alb
    phb = jnp.concatenate(
        (
            jnp.zeros((1, ny, nx), dtype=jnp.float64),
            jnp.cumsum(increments, axis=0),
        ),
        axis=0,
    )
    state = state.replace(
        p_total=pb,
        p_perturbation=jnp.zeros_like(pb),
        ph_total=phb,
        ph_perturbation=jnp.zeros_like(phb),
        mu_total=mub,
        mu_perturbation=jnp.zeros_like(mub),
        theta=theta_base + 0.1 * jnp.sin(jnp.linspace(0.0, 2.0 * jnp.pi, nx))[None, None, :],
    )
    base = BaseState(
        pb=pb,
        phb=phb,
        mub=mub,
        t0=jnp.full_like(theta_base, 300.0),
        theta_base=theta_base,
    )
    namelist = _namelist(grid, diff_opt=1, km_opt=4)
    explicit = _diffopt1_dry_forward_tendencies(state, namelist, base_state=base)
    recovered = _diffopt1_dry_forward_tendencies(state, namelist)
    for left, right in zip(explicit, recovered, strict=True):
        np.testing.assert_allclose(
            np.asarray(right), np.asarray(left), rtol=2e-13, atol=2e-11
        )


def test_106_leaf_opt2_fallback_recovers_log_hydrostatic_t_init():
    grid = _build_grid(ny=8, nx=10, nz=5, dx=4000.0)
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    c3f = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    c3h = 0.5 * (c3f[:-1] + c3f[1:])
    metrics = dataclasses.replace(
        grid.metrics,
        c3f=c3f,
        c4f=jnp.zeros_like(c3f),
        c3h=c3h,
        c4h=jnp.zeros_like(c3h),
    )
    grid = dataclasses.replace(grid, metrics=metrics)
    state = apply_halo(_build_state(grid), halo_spec(grid))
    mub = jnp.full((ny, nx), 9.0e4, dtype=jnp.float64)
    p_top = jnp.reshape(metrics.p_top, ())
    pfu = metrics.c3f[1:, None, None] * mub[None, :, :] + p_top
    pfd = metrics.c3f[:-1, None, None] * mub[None, :, :] + p_top
    phm = metrics.c3h[:, None, None] * mub[None, :, :] + p_top
    pb = jnp.broadcast_to(phm, (nz, ny, nx))
    theta_base = jnp.broadcast_to(
        jnp.linspace(298.0, 430.0, nz, dtype=jnp.float64)[:, None, None],
        (nz, ny, nx),
    )
    alb = (R_D / P0_PA) * theta_base * (pb / P0_PA) ** CVPM
    increments = phm * jnp.log(pfd / pfu) * alb
    phb = jnp.concatenate(
        (
            jnp.zeros((1, ny, nx), dtype=jnp.float64),
            jnp.cumsum(increments, axis=0),
        ),
        axis=0,
    )
    theta_wave = 0.1 * jnp.sin(
        jnp.linspace(0.0, 2.0 * jnp.pi, nx, dtype=jnp.float64)
    )[None, None, :]
    state = state.replace(
        p_total=pb,
        p_perturbation=jnp.zeros_like(pb),
        ph_total=phb,
        ph_perturbation=jnp.zeros_like(phb),
        mu_total=mub,
        mu_perturbation=jnp.zeros_like(mub),
        theta=theta_base + theta_wave,
    )
    base = BaseState(
        pb=pb,
        phb=phb,
        mub=mub,
        t0=jnp.full_like(theta_base, 300.0),
        theta_base=theta_base,
    )
    namelist = _namelist(
        grid,
        diff_opt=1,
        km_opt=4,
        hypsometric_opt=2,
    )
    explicit = _diffopt1_dry_forward_tendencies(state, namelist, base_state=base)
    recovered = _diffopt1_dry_forward_tendencies(state, namelist)
    for left, right in zip(explicit, recovered, strict=True):
        np.testing.assert_allclose(
            np.asarray(right), np.asarray(left), rtol=2e-13, atol=2e-11
        )


def test_nested_namelist_binds_per_domain_diffusion_options():
    class Run:
        namelist = {"dynamics": {"diff_opt": [1, 1, 1], "km_opt": [4, 4, 4]}}

    assert [_domain_int(Run(), "dynamics", "diff_opt", f"d0{i}") for i in (1, 2, 3)] == [1, 1, 1]
    assert [_domain_int(Run(), "dynamics", "km_opt", f"d0{i}") for i in (1, 2, 3)] == [4, 4, 4]
    grid = _build_grid(ny=8, nx=10, nz=5, dx=4000.0)
    namelist = _make_namelist(
        grid=grid,
        tendencies=_namelist(grid).tendencies,
        metrics=grid.metrics,
        dt_s=6.0,
        parent_dt_s=18.0,
        run_start=datetime(2025, 3, 1, tzinfo=timezone.utc),
        radiation_static=None,
        cu_physics=0,
        diff_opt=1,
        km_opt=4,
    )
    assert (namelist.diff_opt, namelist.km_opt) == (1, 4)
