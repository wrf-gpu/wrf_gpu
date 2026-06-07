"""Operational-path proof for the PD / monotonic scalar-advection RK3 wiring.

Sprint: wire ``advect_scalar_flux_limited`` (positive-definite / monotonic flux
limiters) into the OPERATIONAL RK3 scalar-advection path
(``runtime.operational_mode._augment_large_step_tendencies``), selected by
``namelist.scalar_adv_opt`` (WRF canonical: 0=plain, 1=PD, 2=monotonic), applied
on the final RK3 stage only (module_em.F:1265 ``rk_step == rk_order``).

This script regenerates the proof JSON written to ``proofs/pd_monotonic/``:

  operational_pd_rk3_proof.json
    * (a) POSITIVITY      -- opt=1 keeps a positive scalar >= 0 through the
                             operational augment; plain h5/v3 undershoots.
    * (b) MONOTONICITY    -- opt=2 introduces NO new extrema; plain over/undershoots.
    * (c) CONSERVATION    -- the coupled-mass tendency telescopes to ~0 (opt 0/1/2).
    * (d) DEFAULT GUARD   -- opt=0 coupled theta tendency is BYTE-IDENTICAL to the
                             plain advect_scalar_flux path; the limiter is inactive
                             on RK stages 1/2 (byte-identical to plain).
    * (e) FULL RK3 SCAN   -- a complete _rk_scan_step runs finite for opt 0/1/2.

The DEFAULT-path (opt=0) no-regression idealized-dycore gates -- Straka density
current + Skamarock warm bubble -> RAN_TO_COMPLETION -- are recorded separately
in ``proofs/pd_monotonic/operational_idealized_no_regression.json`` (they take
several minutes of CPU sim and are produced by ``run_density_current_case`` /
``run_warm_bubble_case`` with ``require_gpu=False``).

CPU-jax dev path: JAX_PLATFORMS=cpu PYTHONPATH=src taskset -c 0-3 \
    python proofs/pd_monotonic/operational_pd_rk3_proof.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp

from gpuwrf.contracts.halo import apply_halo
from gpuwrf.dynamics.advection import halo_spec
from gpuwrf.dynamics.flux_advection import advect_scalar_flux, couple_velocities_periodic
from gpuwrf.ic_generators.idealized import build_warm_bubble_setup
from gpuwrf.io.scheme_catalog import classify_control
from gpuwrf.runtime.operational_mode import (
    _augment_large_step_tendencies,
    _rk_scan_step,
    _theta_base_offset,
    initial_operational_carry,
)

THETA_BASE = 300.0
PROOF_DIR = Path(__file__).resolve().parent


def _state_with_scalar(template, blob, *, u_const, v_const):
    blob = jnp.asarray(blob, dtype=jnp.float64)
    nz, ny, nx = blob.shape
    theta = THETA_BASE + blob
    u = jnp.full((nz, ny, nx + 1), float(u_const), dtype=jnp.float64)
    v = jnp.full((nz, ny + 1, nx), float(v_const), dtype=jnp.float64)
    w = jnp.zeros((nz + 1, ny, nx), dtype=jnp.float64)
    mu = jnp.ones((ny, nx), dtype=jnp.float64)
    return template.replace(theta=theta, u=u, v=v, w=w, mu_total=mu, mu=mu)


def _namelist(base, *, scalar_adv_opt):
    return dataclasses.replace(
        base, scalar_adv_opt=int(scalar_adv_opt), use_flux_advection=True,
        run_physics=False, run_boundary=False, const_nu_m2_s=0.0,
        diff_6th_opt=0, km_opt=0, dt_s=1.0,
    )


def _coupled_tend(state, nl, *, rk_step):
    haloed = apply_halo(state, halo_spec(nl.grid))
    origin = apply_halo(state, halo_spec(nl.grid))
    out = _augment_large_step_tendencies(
        haloed, nl.tendencies, nl, rk_step=rk_step, step_origin=origin
    )
    return out.theta


def _mass_h(state, nl):
    m = nl.metrics
    return m.c1h[:, None, None] * state.mu_total[None, :, :] + m.c2h[:, None, None]


def _advect_n(setup, blob, *, opt, n, u_const):
    nl = _namelist(setup.namelist, scalar_adv_opt=opt)
    field = np.asarray(blob, dtype=np.float64)
    for _ in range(n):
        st = _state_with_scalar(setup.state, field, u_const=u_const, v_const=0.0)
        tend = _coupled_tend(st, nl, rk_step=3)
        mass = _mass_h(st, nl)
        field = np.asarray((mass * jnp.asarray(field) + float(nl.dt_s) * tend) / mass)
    return field


def main() -> int:
    setup = build_warm_bubble_setup(require_gpu=False)
    nz, ny, nx = setup.grid.nz, setup.grid.ny, setup.grid.nx
    xs = np.arange(nx)
    tophat = (np.where(np.abs(xs - nx / 2) <= 4.0, 1.0, 0.0))[None, None, :] * np.ones((nz, ny, nx))
    lo, hi = float(tophat.min()), float(tophat.max())

    results: dict = {}

    # (a) positivity / (b) monotonicity (multi-step forward via the augment).
    pd = _advect_n(setup, tophat, opt=1, n=30, u_const=40.0)
    mono = _advect_n(setup, tophat, opt=2, n=30, u_const=40.0)
    plain = _advect_n(setup, tophat, opt=0, n=30, u_const=40.0)
    results["positivity_opt1"] = {
        "pd_min": float(pd.min()),
        "plain_min": float(plain.min()),
        "passed": bool(pd.min() >= -1e-12 and plain.min() < -1e-6),
    }
    results["monotonicity_opt2"] = {
        "mono_min": float(mono.min()), "mono_max": float(mono.max()),
        "init_min": lo, "init_max": hi,
        "plain_min": float(plain.min()), "plain_max": float(plain.max()),
        "passed": bool(
            mono.min() >= lo - 1e-12 and mono.max() <= hi + 1e-12
            and (plain.min() < lo - 1e-6 or plain.max() > hi + 1e-6)
        ),
    }

    # (c) conservation of the single-step coupled-mass tendency, opt 0/1/2.
    cons = {}
    for opt in (0, 1, 2):
        st = _state_with_scalar(setup.state, tophat, u_const=40.0, v_const=0.0)
        tend = _coupled_tend(st, _namelist(setup.namelist, scalar_adv_opt=opt), rk_step=3)
        total = float(jnp.sum(tend))
        scale = float(jnp.sum(jnp.abs(tend))) + 1e-30
        cons[f"opt{opt}"] = {"rel_total_tend": abs(total) / scale, "passed": abs(total) / scale < 1e-12}
    results["conservation"] = cons

    # (d) default-path guardrail: opt=0 augment == plain advect_scalar_flux (bytes);
    # limiter inactive on RK stages 1/2.
    st = _state_with_scalar(setup.state, tophat, u_const=40.0, v_const=0.0)
    nl0 = _namelist(setup.namelist, scalar_adv_opt=0)
    tend_op = np.asarray(_coupled_tend(st, nl0, rk_step=3))
    haloed = apply_halo(st, halo_spec(setup.grid))
    dx = float(setup.grid.projection.dx_m)
    dy = float(setup.grid.projection.dy_m)
    m = nl0.metrics
    vel = couple_velocities_periodic(
        haloed.u, haloed.v, haloed.mu_total, c1h=m.c1h, c2h=m.c2h, dnw=m.dnw,
        rdx=1.0 / dx, rdy=1.0 / dy, msfuy=m.msfuy, msfvx=m.msfvx,
        msftx=m.msftx, msfux=m.msfux, msfvy=m.msfvy,
    )
    plain_tend = np.asarray(advect_scalar_flux(
        haloed.theta - _theta_base_offset(haloed.theta), vel,
        mut=haloed.mu_total, c1=m.c1h, rdx=1.0 / dx, rdy=1.0 / dy,
        rdzw=m.rdnw, fzm=m.fnm, fzp=m.fnp,
    ))
    default_bytes = bool(np.array_equal(tend_op, plain_tend))
    rk12_bytes = True
    for opt in (1, 2):
        nlx = _namelist(setup.namelist, scalar_adv_opt=opt)
        for rk in (1, 2):
            a = np.asarray(_coupled_tend(st, nl0, rk_step=rk))
            b = np.asarray(_coupled_tend(st, nlx, rk_step=rk))
            rk12_bytes = rk12_bytes and bool(np.array_equal(a, b))
    results["default_guardrail"] = {
        "opt0_byte_identical_to_plain": default_bytes,
        "limiter_inactive_on_rk1_rk2": rk12_bytes,
        "passed": bool(default_bytes and rk12_bytes),
    }

    # (e) full RK3 scan runs finite for opt 0/1/2.
    carry = initial_operational_carry(setup.state)
    scan = {}
    for opt in (0, 1, 2):
        nl = dataclasses.replace(
            setup.namelist, scalar_adv_opt=opt, use_flux_advection=True,
            run_physics=False, run_boundary=False,
        )
        out = _rk_scan_step(carry, nl)
        th = np.asarray(out.state.theta)
        scan[f"opt{opt}"] = {"finite": bool(np.all(np.isfinite(th))), "min": float(th.min()), "max": float(th.max())}
    results["full_rk3_scan_finite"] = scan

    # catalog classification snapshot.
    results["catalog"] = {
        f"{key}={v}": classify_control(key, v).status.value
        for key in ("moist_adv_opt", "scalar_adv_opt")
        for v in (0, 1, 2, 3, 4)
    }

    all_passed = bool(
        results["positivity_opt1"]["passed"]
        and results["monotonicity_opt2"]["passed"]
        and all(c["passed"] for c in cons.values())
        and results["default_guardrail"]["passed"]
        and all(s["finite"] for s in scan.values())
    )
    results["verdict"] = "PASS" if all_passed else "FAIL"

    PROOF_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROOF_DIR / "operational_pd_rk3_proof.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"\nWROTE {out_path}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
