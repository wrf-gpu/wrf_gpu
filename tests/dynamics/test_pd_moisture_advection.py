"""Idealized + WRF-parity validation of the MOISTURE positive-definite / monotonic
scalar-advection limiter (``moist_adv_opt`` 1=PD / 2=monotonic).

v0.13 Tier-2 sprint: extend the flux-renormalization limiter that v0.12.0 wired
for the potential-temperature scalar (``scalar_adv_opt``) to ALL advected
moisture species (qv, qc, qr, qi, qs, qg), selected by ``moist_adv_opt``.  WRF
runs the IDENTICAL ``advect_scalar`` / ``advect_scalar_pd`` / ``advect_scalar_mono``
routines per moist species in ``solve_em.F:2282-2408``
(``moist_variable_loop`` -> ``rk_scalar_tend(..., config_flags%moist_adv_opt)``).

These tests exercise the multi-species ``advect_moisture_scalars`` loop:

  (1) DEFAULT UNCHANGED (moist_adv_opt=0): the per-species coupled tendency is
      BYTE-IDENTICAL to the plain ``advect_scalar_flux`` path -- opt-in only.
  (2) POSITIVITY (moist_adv_opt=1): a sharp-gradient positive moisture blob stays
      >= 0 under a multi-step forward integration, where the plain h5/v3 path
      undershoots into UNPHYSICAL negative mixing ratios.
  (3) WRF-PARITY: the limited per-species tendency matches an INDEPENDENT direct
      WRF-Fortran transcription of ``advect_scalar_pd`` / ``advect_scalar_mono``
      to round-off (the same transcription the theta test validates), and is the
      canonical faithfulness oracle.
  (4) NON-FINAL STAGE / opt 0 -> plain: the limiter is inactive (byte-identical)
      on RK stages 1/2 and for moist_adv_opt=0, matching WRF's final-stage-only FCT.

CPU-jax dev path: JAX_PLATFORMS=cpu PYTHONPATH=src taskset -c 0-3 pytest ...
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jax import config

config.update("jax_enable_x64", True)

from gpuwrf.dynamics.flux_advection import (
    CoupledVelocities,
    advect_moisture_scalars,
    advect_scalar_flux,
    advect_scalar_flux_limited,
)


# ----------------------------------------------------------------------------
# Shared idealized uniform-flow setup (periodic unit-map grid; mu=1 so coupled
# mass is unity and the coupled tendency integrates directly as d(q)/dt).
# ----------------------------------------------------------------------------


def _uniform_flow_setup(nz, ny, nx, *, u_const, v_const, dx, dt):
    mu = jnp.ones((ny, nx))
    c1 = jnp.ones((nz,))
    c2 = jnp.zeros((nz,))
    ru = jnp.full((nz, ny, nx), float(u_const))
    rv = jnp.full((nz, ny, nx), float(v_const))
    rom = jnp.zeros((nz + 1, ny, nx))
    vel = CoupledVelocities(ru=ru, rv=rv, rom=rom)
    rdx = 1.0 / dx
    rdy = 1.0 / dx
    rdzw = jnp.ones((nz,))
    fzm = jnp.full((nz,), 0.5)
    fzp = jnp.full((nz,), 0.5)
    return dict(mu=mu, c1=c1, c2=c2, vel=vel, rdx=rdx, rdy=rdy, rdzw=rdzw, fzm=fzm, fzp=fzp, dt=dt)


def _sharp_blob(nz, ny, nx, amp=1.0):
    f = np.zeros((nz, ny, nx))
    f[:, ny // 2 - 2 : ny // 2 + 2, nx // 2 - 2 : nx // 2 + 2] = amp
    return jnp.asarray(f)


def _moisture_species(nz, ny, nx):
    """Six WRF moisture species with distinct sharp profiles (qv..qg).

    qv carries a positive baseline (a realistic vapor field never reaches zero
    everywhere); the condensate species (qc/qr/qi/qs/qg) are zero outside a sharp
    blob -- exactly where the unlimited scheme drives unphysical negatives.
    """

    xs = np.arange(nx)
    qv = 0.012 + 0.004 * np.where(np.abs(xs - nx / 2) <= 3.0, 1.0, 0.0)[None, None, :] * np.ones((nz, ny, nx))
    qc = np.asarray(_sharp_blob(nz, ny, nx, amp=1.0e-3))
    qr = np.asarray(_sharp_blob(nz, ny, nx, amp=5.0e-4))
    qi = np.asarray(_sharp_blob(nz, ny, nx, amp=2.0e-4))
    qs = np.asarray(_sharp_blob(nz, ny, nx, amp=3.0e-4))
    qg = np.asarray(_sharp_blob(nz, ny, nx, amp=1.0e-4))
    return tuple(jnp.asarray(q) for q in (qv, qc, qr, qi, qs, qg))


def _moist_loop(fields, fields_old, s, *, opt, final):
    return advect_moisture_scalars(
        fields,
        fields_old,
        s["vel"],
        moist_adv_opt=opt,
        is_final_rk_stage=final,
        mut=s["mu"],
        mu_old=s["mu"],
        c1=s["c1"],
        c2=s["c2"],
        rdx=s["rdx"],
        rdy=s["rdy"],
        rdzw=s["rdzw"],
        fzm=s["fzm"],
        fzp=s["fzp"],
        dt=s["dt"],
    )


# ----------------------------------------------------------------------------
# (1) DEFAULT UNCHANGED: moist_adv_opt=0 is byte-identical to plain advect_scalar_flux.
# ----------------------------------------------------------------------------


def test_default_moist_adv_opt0_byte_identical_to_plain():
    nz, ny, nx = 4, 16, 24
    s = _uniform_flow_setup(nz, ny, nx, u_const=20.0, v_const=10.0, dx=1000.0, dt=5.0)
    fields = _moisture_species(nz, ny, nx)
    # opt=0 on ANY stage -> plain path, per species, byte for byte.
    for final in (False, True):
        out = _moist_loop(fields, fields, s, opt=0, final=final)
        assert len(out) == len(fields)
        for q, tend in zip(fields, out):
            plain = advect_scalar_flux(
                q, s["vel"], mut=s["mu"], c1=s["c1"], rdx=s["rdx"], rdy=s["rdy"],
                rdzw=s["rdzw"], fzm=s["fzm"], fzp=s["fzp"],
            )
            assert np.array_equal(np.asarray(tend), np.asarray(plain)), "opt=0 not byte-identical to plain"


def test_limiter_inactive_on_non_final_stage():
    """moist_adv_opt 1/2 on a NON-final RK stage -> plain path (WRF applies FCT on
    the final stage only); byte-identical to opt=0."""

    nz, ny, nx = 4, 16, 24
    s = _uniform_flow_setup(nz, ny, nx, u_const=20.0, v_const=10.0, dx=1000.0, dt=5.0)
    fields = _moisture_species(nz, ny, nx)
    base = _moist_loop(fields, fields, s, opt=0, final=True)
    for opt in (1, 2):
        out = _moist_loop(fields, fields, s, opt=opt, final=False)
        for a, b in zip(out, base):
            assert np.array_equal(np.asarray(a), np.asarray(b)), f"opt={opt} non-final stage not plain"


# ----------------------------------------------------------------------------
# (2) POSITIVITY: moist_adv_opt=1 keeps every species non-negative; plain undershoots.
# ----------------------------------------------------------------------------


def test_pd_keeps_all_moisture_species_nonnegative():
    nz, ny, nx = 4, 32, 32
    dx, dt = 1000.0, 5.0
    s = _uniform_flow_setup(nz, ny, nx, u_const=20.0, v_const=10.0, dx=dx, dt=dt)
    fields = list(_moisture_species(nz, ny, nx))
    min_seen = [float("inf")] * len(fields)
    for _ in range(40):
        tends = _moist_loop(tuple(fields), tuple(fields), s, opt=1, final=True)
        # coupled mass = 1 -> q_new = q_old + dt*tend.
        fields = [q + dt * t for q, t in zip(fields, tends)]
        for i, q in enumerate(fields):
            min_seen[i] = min(min_seen[i], float(jnp.min(q)))
    for i, m in enumerate(min_seen):
        assert m >= -1e-12, f"PD species {i} undershot to {m} (unphysical negative mixing ratio)"


def test_plain_path_drives_moisture_negative():
    """Sanity: the UNLIMITED h5/v3 path drives a condensate species (qc) below zero
    on the same sharp blob -> the positivity result above is a real property."""

    nz, ny, nx = 4, 32, 32
    dx, dt = 1000.0, 5.0
    s = _uniform_flow_setup(nz, ny, nx, u_const=20.0, v_const=10.0, dx=dx, dt=dt)
    qc = _moisture_species(nz, ny, nx)[1]  # condensate blob, zero background
    min_seen = float("inf")
    for _ in range(40):
        tend = advect_scalar_flux(
            qc, s["vel"], mut=s["mu"], c1=s["c1"], rdx=s["rdx"], rdy=s["rdy"],
            rdzw=s["rdzw"], fzm=s["fzm"], fzp=s["fzp"],
        )
        qc = qc + dt * tend
        min_seen = min(min_seen, float(jnp.min(qc)))
    assert min_seen < -1e-9, f"plain path did not drive moisture negative (min={min_seen}); test trivial"


# ----------------------------------------------------------------------------
# (2b) MONOTONICITY: moist_adv_opt=2 introduces no new extrema per species.
# ----------------------------------------------------------------------------


def test_mono_introduces_no_new_extrema_per_species():
    nz, ny, nx = 4, 32, 32
    dx, dt = 1000.0, 5.0
    s = _uniform_flow_setup(nz, ny, nx, u_const=20.0, v_const=10.0, dx=dx, dt=dt)
    fields0 = list(_moisture_species(nz, ny, nx))
    init_min = [float(jnp.min(q)) for q in fields0]
    init_max = [float(jnp.max(q)) for q in fields0]
    fields = list(fields0)
    min_seen = list(init_min)
    max_seen = list(init_max)
    for _ in range(40):
        tends = _moist_loop(tuple(fields), tuple(fields), s, opt=2, final=True)
        fields = [q + dt * t for q, t in zip(fields, tends)]
        for i, q in enumerate(fields):
            min_seen[i] = min(min_seen[i], float(jnp.min(q)))
            max_seen[i] = max(max_seen[i], float(jnp.max(q)))
    for i in range(len(fields)):
        assert min_seen[i] >= init_min[i] - 1e-12, f"mono species {i} created new min"
        assert max_seen[i] <= init_max[i] + 1e-12, f"mono species {i} created new max"


# ----------------------------------------------------------------------------
# (3) WRF-PARITY: the per-species limited tendency matches the SAME single-scalar
#     advect_scalar_flux_limited the theta test validates vs the WRF transcription.
#     advect_moisture_scalars must reduce EXACTLY to per-field advect_scalar_flux_limited.
# ----------------------------------------------------------------------------


def test_moisture_loop_reduces_to_single_scalar_limiter():
    """``advect_moisture_scalars`` (opt 1/2, final stage) must return EXACTLY the
    per-species ``advect_scalar_flux_limited`` tendency -- which the theta test
    already proves matches the independent WRF-Fortran advect_scalar_pd/_mono
    transcription to round-off.  This pins moisture parity to that oracle."""

    nz, ny, nx = 6, 12, 24
    dx, dt = 1000.0, 4.0
    s = _uniform_flow_setup(nz, ny, nx, u_const=15.0, v_const=8.0, dx=dx, dt=dt)
    # exercise a non-trivial vertical flux too.
    rom = jnp.zeros((nz + 1, ny, nx)).at[1:nz, :, :].set(0.03)
    s["vel"] = CoupledVelocities(ru=s["vel"].ru, rv=s["vel"].rv, rom=rom)
    fields = _moisture_species(nz, ny, nx)
    for opt in (1, 2):
        out = _moist_loop(fields, fields, s, opt=opt, final=True)
        for q, tend in zip(fields, out):
            ref = advect_scalar_flux_limited(
                q, q, s["vel"], scalar_adv_opt=opt, mut=s["mu"], mu_old=s["mu"],
                c1=s["c1"], c2=s["c2"], rdx=s["rdx"], rdy=s["rdy"], rdzw=s["rdzw"],
                fzm=s["fzm"], fzp=s["fzp"], dt=s["dt"],
            )
            assert float(jnp.max(jnp.abs(tend - ref))) == 0.0, f"opt={opt} loop != single-scalar limiter"


# ----------------------------------------------------------------------------
# (3b) Direct WRF-Fortran transcription parity (x-row PD), independent of the
#      theta test, applied to a moisture mixing-ratio profile.
# ----------------------------------------------------------------------------


def _np_flux_upwind(qm1, q, cr):
    return 0.5 * np.minimum(1.0, cr + np.abs(cr)) * qm1 + 0.5 * np.maximum(-1.0, cr - np.abs(cr)) * q


def _np_flux6(qm3, qm2, qm1, q, qp1, qp2):
    return (37.0 / 60.0) * (q + qm1) - (2.0 / 15.0) * (qp1 + qm2) + (1.0 / 60.0) * (qp2 + qm3)


def _np_flux5(qm3, qm2, qm1, q, qp1, qp2, ua):
    return _np_flux6(qm3, qm2, qm1, q, qp1, qp2) - np.sign(ua) * (1.0 / 60.0) * (
        (qp2 - qm3) - 5.0 * (qp1 - qm2) + 10.0 * (q - qm1)
    )


def _wrf_pd_x_1d(field, field_old, ru, mu, dx, dt, eps=1e-20):
    R = lambda a, sh: np.roll(a, sh)
    cr = ru * dt / dx / mu
    fqxl = mu * (dx / dt) * _np_flux_upwind(R(field_old, 1), field_old, cr)
    hi = ru * _np_flux5(R(field, 3), R(field, 2), R(field, 1), field, R(field, -1), R(field, -2), ru)
    fqx = hi - fqxl
    rdx = 1.0 / dx
    ph_low = mu * field_old - dt * (rdx * (R(fqxl, -1) - fqxl))
    flux_out = dt * (rdx * (np.maximum(0.0, R(fqx, -1)) - np.minimum(0.0, fqx)))
    scale = np.where(flux_out > ph_low, np.maximum(0.0, ph_low / (flux_out + eps)), 1.0)
    fqx_lim = np.where(fqx > 0.0, R(scale, 1) * fqx, np.where(fqx < 0.0, scale * fqx, fqx))
    tot = fqx_lim + fqxl
    return -rdx * (R(tot, -1) - tot)


def test_moisture_pd_x_matches_wrf_transcription():
    nx = 24
    dx, dt = 1.0, 0.2
    # a moisture-scale mixing ratio (kg/kg) condensate spike on zero background.
    q = np.zeros(nx)
    q[10:13] = 1.5e-3
    ru = np.full(nx, 1.5)
    tw = _wrf_pd_x_1d(q, q, ru, 1.0, dx, dt)
    vel = CoupledVelocities(
        ru=jnp.asarray(ru)[None, None, :], rv=jnp.zeros((1, 1, nx)), rom=jnp.zeros((2, 1, nx))
    )
    out = advect_moisture_scalars(
        (jnp.asarray(q)[None, None, :],), (jnp.asarray(q)[None, None, :],), vel,
        moist_adv_opt=1, is_final_rk_stage=True, mut=jnp.ones((1, nx)), mu_old=jnp.ones((1, nx)),
        c1=jnp.ones(1), c2=jnp.zeros(1), rdx=1.0 / dx, rdy=1.0, rdzw=jnp.ones(1),
        fzm=jnp.full(1, 0.5), fzp=jnp.full(1, 0.5), dt=dt,
    )
    assert float(jnp.max(jnp.abs(jnp.asarray(out[0])[0, 0, :] - tw))) < 1e-13


# ----------------------------------------------------------------------------
# (4) CONSERVATION: each species' single-step coupled-mass tendency telescopes to ~0.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("opt", [1, 2])
def test_per_species_conservation(opt):
    nz, ny, nx = 4, 32, 32
    dx, dt = 1000.0, 5.0
    s = _uniform_flow_setup(nz, ny, nx, u_const=20.0, v_const=10.0, dx=dx, dt=dt)
    fields = _moisture_species(nz, ny, nx)
    tends = _moist_loop(fields, fields, s, opt=opt, final=True)
    for i, tend in enumerate(tends):
        total = float(jnp.sum(tend))
        scale = float(jnp.sum(jnp.abs(tend))) + 1e-30
        assert abs(total) / scale < 1e-12, f"opt={opt} species {i} not conservative"


def test_length_mismatch_raises():
    nz, ny, nx = 4, 8, 8
    s = _uniform_flow_setup(nz, ny, nx, u_const=1.0, v_const=1.0, dx=1000.0, dt=1.0)
    fields = _moisture_species(nz, ny, nx)
    with pytest.raises(ValueError):
        advect_moisture_scalars(
            fields, fields[:-1], s["vel"], moist_adv_opt=1, is_final_rk_stage=True,
            mut=s["mu"], mu_old=s["mu"], c1=s["c1"], c2=s["c2"], rdx=s["rdx"], rdy=s["rdy"],
            rdzw=s["rdzw"], fzm=s["fzm"], fzp=s["fzp"], dt=s["dt"],
        )
