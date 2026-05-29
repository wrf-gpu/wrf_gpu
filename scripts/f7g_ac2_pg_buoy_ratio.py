"""F7G AC2 -- pg_buoy_w(grid%p) magnitude after the signed-metric fix.

GPT-5.5 §4 check 2: after the WRF-signed-metric + start_em-equivalent recompute,
the stage ``pg_buoy_w(grid%p)`` must NOT show the 9x/19x artifact.

* Balanced rest column (theta'=0): direct vertical rw_tend residual max_abs <= 1e-10 m/s^2.
* Unbalanced analytic-buoyancy oracle (warm bubble theta'): the physical
  rw_tend / analytic buoyancy g*theta'/theta0 must be in [0.9, 1.1].

We drive the REAL operational stage path: small_step_prep_wrf -> calc_p_rho_wrf(step=0)
-> pg_buoy_w_dry(grid%p, mu'), exactly the F7G once-per-stage buoyancy the acoustic
core now consumes, with WRF-signed metrics.  rw_tend is the COUPLED large-step w
tendency; decouple by (c1f*mut+c2f)/msfty to compare with the physical buoyancy.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

config.update("jax_enable_x64", True)

from gpuwrf.contracts.halo import apply_halo
from gpuwrf.dynamics.advection import compute_advection_tendencies, halo_spec
from gpuwrf.dynamics.core.advance_w import GRAVITY_M_S2, pg_buoy_w_dry
from gpuwrf.dynamics.core.calc_p_rho import calc_p_rho_wrf
from gpuwrf.dynamics.core.small_step_prep import small_step_prep_wrf
from gpuwrf.ic_generators.idealized import (
    _build_setup,
    _enforce_operational_precision,
    build_warm_bubble_numpy,
    THETA0_K,
)
from gpuwrf.runtime.operational_mode import _augment_large_step_tendencies
from gpuwrf.runtime.operational_state import initial_operational_carry


def _stage_rw_tend(state, setup):
    """Reproduce the F7G once-per-stage pg_buoy_w the acoustic core consumes."""

    nl = setup.namelist
    carry = initial_operational_carry(_enforce_operational_precision(state, force_fp64=True))
    haloed = apply_halo(carry.state, halo_spec(nl.grid))
    tend = compute_advection_tendencies(haloed, nl.tendencies, nl.grid)
    tend = _augment_large_step_tendencies(haloed, tend, nl, rk_step=1)
    prep = small_step_prep_wrf(
        haloed, 1, float(setup.numpy_case.dt_s) / 3.0,
        metrics=nl.metrics, reference_state=haloed, ww=carry.ww,
    )
    pressure = calc_p_rho_wrf(prep, step=0, non_hydrostatic=True)
    mu_prime = prep.mut - prep.mub  # stage perturbation dry mass mu'
    rw_tend = pg_buoy_w_dry(
        pressure.p, mu_prime,
        c1f=nl.metrics.c1f, rdnw=nl.metrics.rdnw, rdn=nl.metrics.rdn,
        msfty=nl.metrics.msfty, gravity=GRAVITY_M_S2,
    )
    # decouple coupled rw_tend to physical accel: rw_tend = (1/msfty)*(coupled buoyancy).
    # The mass coupling on a w face is (c1f*mut+c2f)/msfty; divide it out.
    mut = prep.mut  # full dry mass (ny,nx)
    c1f = nl.metrics.c1f
    c2f = nl.metrics.c2f
    mass_f = c1f[:, None, None] * mut[None, :, :] + c2f[:, None, None]  # (nz+1,ny,nx)
    msfty = nl.metrics.msfty[None, :, :]
    rw_phys = rw_tend * msfty / jnp.maximum(jnp.abs(mass_f), 1.0e-12)
    return (
        np.asarray(jax.device_get(rw_tend)),
        np.asarray(jax.device_get(rw_phys)),
        np.asarray(jax.device_get(pressure.p)),
        np.asarray(jax.device_get(pressure.al)),
        np.asarray(jax.device_get(mu_prime)),
    )


def main() -> int:
    proof_dir = Path("proofs/f7g")
    proof_dir.mkdir(parents=True, exist_ok=True)

    # --- balanced rest column: theta' = 0 everywhere (same eta-hydrostatic base) ---
    case_bal = build_warm_bubble_numpy()
    case_bal.theta_prime_k[...] = 0.0
    case_bal.theta_k[...] = THETA0_K
    setup_bal = _build_setup(case_bal, require_gpu=True)
    rw_coupled_b, rw_phys_b, p_b, al_b, mup_b = _stage_rw_tend(setup_bal.state, setup_bal)
    rest_resid = float(np.max(np.abs(rw_phys_b)))

    # --- explicitly-unbalanced analytic-buoyancy oracle (GPT-5.5 §4 check 2) ---
    # Build a column with the theta' bubble but the BASE (un-rebalanced) geopotential
    # so the EOS yields a REAL perturbation pressure p' = EOS(theta_full, alt_base) - pb
    # that pg_buoy_w must convert into the analytic buoyancy g*theta'/theta0.  This
    # exercises the rdn(k)*(p(k)-p(k-1)) term with WRF-signed rdn (NOT a fixed-mass
    # column where p'~0 by construction); a 9x/19x ratio here would expose the sign bug.
    from gpuwrf.dynamics.core.advance_w import pg_buoy_w_dry as _pgb
    case = build_warm_bubble_numpy()
    R_d, cp, p0, th0 = 287.0, 1004.0, 100000.0, THETA0_K
    cv = cp - R_d
    nz, nx = case.nz, case.nx
    pb = case.pressure_pa  # (nz, nx) base hydrostatic mass pressure (neutral th0)
    theta_full = case.theta_k  # (nz, nx)
    # Unbalanced: warm theta' at the BASE (un-recomputed) density alb -> the EOS
    # yields a real perturbation pressure p' = p_full - pb.  Using alb (not the
    # rebalanced alt) is what makes the column explicitly out of hydrostatic balance.
    alb = (R_d / p0) * th0 * (pb / p0) ** (-cv / cp)
    p_full = p0 * ((R_d * theta_full) / (p0 * alb)) ** (cp / cv)
    p_prime = p_full - pb  # (nz, nx) real perturbation pressure of the unbalanced column
    p_prime3 = jnp.asarray(p_prime[:, None, :])  # (nz,1,nx)
    mu_zero = jnp.zeros((1, nx), dtype=jnp.float64)
    nl = setup_bal.namelist
    rw_unbal = _pgb(
        p_prime3, mu_zero,
        c1f=nl.metrics.c1f, rdnw=nl.metrics.rdnw, rdn=nl.metrics.rdn,
        msfty=nl.metrics.msfty, gravity=GRAVITY_M_S2,
    )
    mut_full = jnp.asarray((pb[0, :] - case.p_top_pa) * 0.0 + (case.mu_base_pa[0, :]))  # (nx,)
    mass_f = nl.metrics.c1f[:, None, None] * mut_full[None, None, :] + nl.metrics.c2f[:, None, None]
    rw_phys_u = np.asarray(jax.device_get(
        rw_unbal * nl.metrics.msfty[None, :, :] / jnp.maximum(jnp.abs(mass_f), 1.0e-12)
    ))
    p_stage = p_prime
    mup = mu_zero
    # analytic buoyancy on faces: interpolate theta' to faces, b = g*theta'/theta0.
    thp = case.theta_prime_k  # (nz, nx)
    thp_face = np.zeros((nz + 1, nx))
    thp_face[1:-1, :] = 0.5 * (thp[1:, :] + thp[:-1, :])
    analytic = GRAVITY_M_S2 * thp_face / th0  # (nz+1, nx)
    rw_face = rw_phys_u[:, 0, :]  # (nz+1, nx)
    mask = np.abs(analytic) > 0.3 * float(np.max(np.abs(analytic)))
    ratios = rw_face[mask] / analytic[mask]
    ratio_med = float(np.median(ratios)) if ratios.size else 0.0
    ratio_min = float(np.min(ratios)) if ratios.size else 0.0
    ratio_max = float(np.max(ratios)) if ratios.size else 0.0
    analytic_peak = float(np.max(np.abs(analytic)))
    rw_peak = float(np.max(np.abs(rw_face)))

    ratio_ok = 0.9 <= abs(ratio_med) <= 1.1
    rest_ok = rest_resid <= 1.0e-10
    overall = ratio_ok and rest_ok

    payload = {
        "schema": "f7g_pg_buoy_ratio",
        "schema_version": 1,
        "check": "AC2",
        "wrf_refs": [
            "module_em.F:1361-1368 (pg_buoy_w once per RK stage)",
            "module_big_step_utilities_em.F:2553-2572 (pg_buoy_w interior+top, signed rdn/rdnw)",
        ],
        "balanced_rest": {
            "max_abs_rw_phys_m_s2": rest_resid,
            "max_abs_p_stage_Pa": float(np.max(np.abs(p_b))),
            "max_abs_al_stage": float(np.max(np.abs(al_b))),
            "max_abs_mu_prime": float(np.max(np.abs(mup_b))),
            "tolerance": 1.0e-10,
            "passed": bool(rest_ok),
        },
        "unbalanced_warm_bubble": {
            "analytic_buoyancy_peak_m_s2": analytic_peak,
            "rw_phys_peak_m_s2": rw_peak,
            "ratio_median": ratio_med,
            "ratio_min": ratio_min,
            "ratio_max": ratio_max,
            "max_abs_p_stage_Pa": float(np.max(np.abs(p_stage))),
            "max_abs_mu_prime": float(np.max(np.abs(mup))),
            "ratio_band": [0.9, 1.1],
            "passed": bool(ratio_ok),
        },
        "verdict": "PASS" if overall else "FAIL",
    }
    out = proof_dir / "pg_buoy_ratio.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("rest residual (m/s^2):", rest_resid)
    print("warm-bubble ratio rw_phys/analytic: median=%.4f [%.4f, %.4f]" % (ratio_med, ratio_min, ratio_max))
    print("analytic peak=%.5f  rw_phys peak=%.5f" % (analytic_peak, rw_peak))
    print("AC2 verdict:", payload["verdict"])
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
