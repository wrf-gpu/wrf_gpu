"""Operational-path validation of the PD / monotonic scalar-advection wiring.

Sprint: wire the proven positive-definite (``scalar_adv_opt=1``) / monotonic
(``=2``) flux limiters into the OPERATIONAL RK3 scalar-advection path
(``runtime.operational_mode._augment_large_step_tendencies``).

The standalone limiter (``flux_advection.advect_scalar_flux_limited``) is already
proven in ``test_pd_monotonic_advection.py`` (14 tests, WRF-Fortran parity).  This
suite proves the OPERATIONAL WIRING: that ``_augment_large_step_tendencies``
selects the limiter purely from ``namelist.scalar_adv_opt`` on the final RK3
stage, that the resulting coupled theta tendency keeps a positive scalar
non-negative (opt=1) / free of new extrema (opt=2) and conserves coupled mass,
and -- the ABSOLUTE GUARDRAIL -- that ``scalar_adv_opt=0`` (the default) emits a
coupled theta tendency BYTE-IDENTICAL to the plain ``advect_scalar_flux`` path.

The grid / metrics come from the real Skamarock warm-bubble idealized setup
(``build_warm_bubble_setup``), so the test exercises the actual operational
metrics (c1h/c2h, rdnw, fnm/fnp, map factors), not a hand-rolled fixture.  The
state's theta is replaced by ``THETA_BASE + blob`` for a NON-NEGATIVE transported
scalar and a uniform horizontal wind advects it on the periodic slab.

CPU-jax dev path: JAX_PLATFORMS=cpu PYTHONPATH=src taskset -c 0-3 pytest ...
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax import config

config.update("jax_enable_x64", True)

from gpuwrf.contracts.halo import apply_halo
from gpuwrf.dynamics.advection import halo_spec
from gpuwrf.dynamics.flux_advection import advect_scalar_flux, couple_velocities_periodic
from gpuwrf.ic_generators.idealized import build_warm_bubble_setup
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    _augment_large_step_tendencies,
    _theta_base_offset,
)


THETA_BASE = 300.0


def _base_setup():
    setup = build_warm_bubble_setup(require_gpu=False)
    return setup


def _state_with_scalar(template_state, blob, *, u_const, v_const):
    """Replace theta with THETA_BASE+blob, set a uniform horizontal wind, mu=1."""

    blob = jnp.asarray(blob, dtype=jnp.float64)
    nz, ny, nx = blob.shape
    theta = THETA_BASE + blob
    u = jnp.full((nz, ny, nx + 1), float(u_const), dtype=jnp.float64)
    v = jnp.full((nz, ny + 1, nx), float(v_const), dtype=jnp.float64)
    w = jnp.zeros((nz + 1, ny, nx), dtype=jnp.float64)
    mu = jnp.ones((ny, nx), dtype=jnp.float64)
    return template_state.replace(theta=theta, u=u, v=v, w=w, mu_total=mu, mu=mu)


def _namelist(base_namelist, *, scalar_adv_opt):
    return dataclasses.replace(
        base_namelist,
        scalar_adv_opt=int(scalar_adv_opt),
        use_flux_advection=True,
        run_physics=False,
        run_boundary=False,
        const_nu_m2_s=0.0,
        diff_6th_opt=0,
        km_opt=0,
        dt_s=1.0,
    )


def _coupled_theta_tendency(state, namelist, *, rk_step):
    """Run the OPERATIONAL augment path and return the coupled theta tendency."""

    haloed = apply_halo(state, halo_spec(namelist.grid))
    origin = apply_halo(state, halo_spec(namelist.grid))  # start-of-step _1 reference
    # Reuse the setup's resident ZERO tendencies (built CPU-compatibly); the
    # augment also reads namelist.tendencies internally as the base.
    base_tend = namelist.tendencies
    out = _augment_large_step_tendencies(
        haloed, base_tend, namelist, rk_step=rk_step, step_origin=origin
    )
    return out.theta


def _mass_h(state, namelist):
    m = namelist.metrics
    return m.c1h[:, None, None] * state.mu_total[None, :, :] + m.c2h[:, None, None]


def _blob_grid(setup):
    nz = setup.grid.nz
    ny = setup.grid.ny
    nx = setup.grid.nx
    return nz, ny, nx


# ----------------------------------------------------------------------------
# ABSOLUTE GUARDRAIL: scalar_adv_opt=0 (default) is byte-identical to the plain
# advect_scalar_flux path through the operational augment.
# ----------------------------------------------------------------------------


def test_operational_default_path_byte_identical_to_plain():
    setup = _base_setup()
    nz, ny, nx = _blob_grid(setup)
    xs = np.arange(nx)
    blob = (0.5 * (1.0 + np.cos(np.clip((xs - nx / 2) / 4.0, -np.pi, np.pi))))[None, None, :] * np.ones((nz, ny, nx))
    state = _state_with_scalar(setup.state, blob, u_const=20.0, v_const=0.0)
    nl0 = _namelist(setup.namelist, scalar_adv_opt=0)

    tend_op = _coupled_theta_tendency(state, nl0, rk_step=3)

    # the plain advect_scalar_flux the wiring must reduce to (opt=0)
    haloed = apply_halo(state, halo_spec(setup.grid))
    dx = float(setup.grid.projection.dx_m)
    dy = float(setup.grid.projection.dy_m)
    m = nl0.metrics
    vel = couple_velocities_periodic(
        haloed.u, haloed.v, haloed.mu_total,
        c1h=m.c1h, c2h=m.c2h, dnw=m.dnw,
        rdx=1.0 / dx, rdy=1.0 / dy,
        msfuy=m.msfuy, msfvx=m.msfvx, msftx=m.msftx, msfux=m.msfux, msfvy=m.msfvy,
    )
    offset = _theta_base_offset(haloed.theta)
    plain = advect_scalar_flux(
        haloed.theta - offset, vel,
        mut=haloed.mu_total, c1=m.c1h,
        rdx=1.0 / dx, rdy=1.0 / dy, rdzw=m.rdnw, fzm=m.fnm, fzp=m.fnp,
    )
    np.testing.assert_array_equal(np.asarray(tend_op), np.asarray(plain))


def test_operational_limiter_inactive_on_non_final_rk_stages():
    """opt=1/2 on RK stages 1 and 2 must still use the plain path (WRF: rk3-only)."""

    setup = _base_setup()
    nz, ny, nx = _blob_grid(setup)
    xs = np.arange(nx)
    blob = (np.maximum(0.0, 1.0 - np.abs(xs - nx / 2) / 4.0))[None, None, :] * np.ones((nz, ny, nx))
    state = _state_with_scalar(setup.state, blob, u_const=25.0, v_const=0.0)

    nl0 = _namelist(setup.namelist, scalar_adv_opt=0)
    for opt in (1, 2):
        nlx = _namelist(setup.namelist, scalar_adv_opt=opt)
        for rk in (1, 2):
            plain = _coupled_theta_tendency(state, nl0, rk_step=rk)
            limited_stage = _coupled_theta_tendency(state, nlx, rk_step=rk)
            np.testing.assert_array_equal(
                np.asarray(limited_stage), np.asarray(plain),
                err_msg=f"opt={opt} rk_step={rk} must equal plain (limiter is rk3-only)",
            )


# ----------------------------------------------------------------------------
# (a) POSITIVITY / (b) MONOTONICITY via a multi-step forward integration through
# the operational augment.
# ----------------------------------------------------------------------------


def _advect_n_steps(setup, blob, *, opt, n_steps, u_const, v_const):
    nl = _namelist(setup.namelist, scalar_adv_opt=opt)
    field = np.asarray(blob, dtype=np.float64)
    for _ in range(n_steps):
        st = _state_with_scalar(setup.state, field, u_const=u_const, v_const=v_const)
        # field_old (WRF scalar_old / _1) is fixed at the START of each model step;
        # for a single-RK-step forward integration origin == current state.
        tend = _coupled_theta_tendency(st, nl, rk_step=3)
        mass = _mass_h(st, nl)
        coupled_new = mass * jnp.asarray(field) + float(nl.dt_s) * tend
        field = np.asarray(coupled_new / mass)
    return field


def test_operational_pd_keeps_scalar_nonnegative():
    setup = _base_setup()
    nz, ny, nx = _blob_grid(setup)
    xs = np.arange(nx)
    # sharp positive top-hat: high-order h5 undershoots below zero.
    blob = (np.where(np.abs(xs - nx / 2) <= 4.0, 1.0, 0.0))[None, None, :] * np.ones((nz, ny, nx))

    pd = _advect_n_steps(setup, blob, opt=1, n_steps=30, u_const=40.0, v_const=0.0)
    plain = _advect_n_steps(setup, blob, opt=0, n_steps=30, u_const=40.0, v_const=0.0)

    assert pd.min() >= -1.0e-12, f"PD operational path undershot: min={pd.min()}"
    assert plain.min() < -1.0e-6, f"sanity: plain h5 must undershoot, got min={plain.min()}"


def test_operational_mono_introduces_no_new_extrema():
    setup = _base_setup()
    nz, ny, nx = _blob_grid(setup)
    xs = np.arange(nx)
    blob = (np.where(np.abs(xs - nx / 2) <= 4.0, 1.0, 0.0))[None, None, :] * np.ones((nz, ny, nx))
    lo, hi = float(blob.min()), float(blob.max())

    mono = _advect_n_steps(setup, blob, opt=2, n_steps=30, u_const=40.0, v_const=0.0)
    plain = _advect_n_steps(setup, blob, opt=0, n_steps=30, u_const=40.0, v_const=0.0)

    assert mono.min() >= lo - 1.0e-12, f"mono undershot below initial min: {mono.min()} < {lo}"
    assert mono.max() <= hi + 1.0e-12, f"mono overshot above initial max: {mono.max()} > {hi}"
    assert (plain.min() < lo - 1.0e-6) or (plain.max() > hi + 1.0e-6), (
        f"sanity: plain h5 must over/undershoot; min={plain.min()} max={plain.max()}"
    )


# ----------------------------------------------------------------------------
# (c) CONSERVATION: a SINGLE augment-path tendency conserves coupled scalar mass
# (the periodic flux divergence telescopes to zero) for opt 0/1/2.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("opt", [0, 1, 2])
def test_operational_tendency_conserves_coupled_mass(opt):
    setup = _base_setup()
    nz, ny, nx = _blob_grid(setup)
    xs = np.arange(nx)
    blob = (np.where(np.abs(xs - nx / 2) <= 4.0, 1.0, 0.0))[None, None, :] * np.ones((nz, ny, nx))
    state = _state_with_scalar(setup.state, blob, u_const=40.0, v_const=0.0)
    nl = _namelist(setup.namelist, scalar_adv_opt=opt)

    tend = _coupled_theta_tendency(state, nl, rk_step=3)
    total_tend = float(jnp.sum(tend))
    scale = float(jnp.sum(jnp.abs(tend))) + 1.0e-30
    assert abs(total_tend) / scale < 1.0e-12, f"opt={opt} mass-tend not conserved: {total_tend}"
