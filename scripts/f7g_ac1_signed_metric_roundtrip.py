"""F7G AC1 -- algebraic round-trip of the WRF-signed calc_p_rho_phi inverse.

GPT-5.5 council diagnosis (gpt-council-findings.md Q1/Q3, §4 check 1): WRF builds
the idealized perturbation geopotential with SIGNED metrics

    ph'(k+1) = ph'(k) - dnw(k) * ( ((c1h*mub+c2h)+c1h*mu')*al(k) + c1h*mu'*alb(k) )
               (module_initialize_ideal.F:1124-1129 / :1308-1313, dnw<0)

and diagnoses al back with the SIGNED calc_p_rho_phi inverse

    al(k) = -1/((c1h*mub+c2h)+c1h*mu') * ( alb(k)*c1h*mu' + rdnw(k)*(ph'(k+1)-ph'(k)) )
            (start_em.F:819-828 / module_big_step_utilities_em.F:1023-1030, rdnw<0).

These are exact discrete inverses BY CONSTRUCTION only when dnw/rdnw carry the WRF
sign.  The JAX idealized path used POSITIVE |dnw|/|rdnw| (grid.py / idealized.py),
so a balanced ph' diagnosed al with the wrong sign -> the 19x pg_buoy_w artifact.

This check is pure numpy (no integration): build al_init = alt_full - alb, integrate
ph' with the recurrence, diagnose al_calc with calc_p_rho_phi, require
max_abs(al_calc - al_init) <= 1e-12 for the WRF-signed convention, and SHOW the
positive-convention failure for contrast.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

R_DRY_AIR = 287.0
CP_DRY_AIR = 1004.0
CV_DRY_AIR = CP_DRY_AIR - R_DRY_AIR
P0_PA = 100000.0
THETA0_K = 300.0
GRAVITY_M_S2 = 9.81
CVPM = -CV_DRY_AIR / CP_DRY_AIR  # WRF cvpm = -cv/cp exponent on (p/p0)


def _alpha_dry(theta_k: np.ndarray, pressure_pa: np.ndarray) -> np.ndarray:
    theta_k = np.asarray(theta_k, dtype=np.float64)
    pressure_pa = np.asarray(pressure_pa, dtype=np.float64)
    return (R_DRY_AIR / P0_PA) * theta_k * (pressure_pa / P0_PA) ** CVPM


def _uniform_z_hydrostatic_base(z_face_m, theta0_k, p_surface_pa=P0_PA):
    z_face = np.asarray(z_face_m, dtype=np.float64)
    nz = int(z_face.size) - 1
    p_face = np.zeros(nz + 1, dtype=np.float64)
    p_face[0] = float(p_surface_pa)
    for k in range(nz):
        dz = z_face[k + 1] - z_face[k]
        alpha_lo = _alpha_dry(np.array([theta0_k]), np.array([p_face[k]]))[0]
        p_mid = p_face[k] - 0.5 * GRAVITY_M_S2 * dz / alpha_lo
        alpha_mid = _alpha_dry(np.array([theta0_k]), np.array([p_mid]))[0]
        p_face[k + 1] = p_face[k] - GRAVITY_M_S2 * dz / alpha_mid
    p_top = float(p_face[-1])
    mu = float(p_face[0] - p_top)
    eta_levels = (p_face - p_top) / mu
    p_mass = 0.5 * (p_face[:-1] + p_face[1:])
    return eta_levels, p_mass, mu


def _build_column(case: str):
    if case == "warm_bubble":
        dz = 250.0
        nz = 40
        z_face = np.arange(nz + 1, dtype=np.float64) * dz
        z_m = (np.arange(nz, dtype=np.float64) + 0.5) * dz
        center_z, rad_z, amp = 2000.0, 2000.0, 2.0
        # column through the bubble center (xrad = 0)
        radius = np.abs(z_m - center_z) / rad_z
        theta_prime = amp * np.where(radius <= 1.0, 0.5 * (1.0 + np.cos(np.pi * radius)), 0.0)
    elif case == "density_current":
        dz = 100.0
        nz = 60
        z_face = np.arange(nz + 1, dtype=np.float64) * dz
        z_m = (np.arange(nz, dtype=np.float64) + 0.5) * dz
        center_z, rad_z, amp = 3000.0, 2000.0, -15.0
        radius = np.abs(z_m - center_z) / rad_z
        theta_prime = amp * np.where(radius <= 1.0, 0.5 * (1.0 + np.cos(np.pi * radius)), 0.0)
    else:
        raise ValueError(case)
    eta, p_mass, mu = _uniform_z_hydrostatic_base(z_face, THETA0_K)
    theta_full = THETA0_K + theta_prime
    alt_full = _alpha_dry(theta_full, p_mass)
    alb = _alpha_dry(np.full(nz, THETA0_K), p_mass)
    al_init = alt_full - alb
    return dict(eta=eta, p_mass=p_mass, mu=mu, nz=nz, al_init=al_init, alb=alb)


def _roundtrip(column, *, signed: bool):
    """Integrate ph' from al_init then diagnose al_calc; return max_abs error.

    Pure-sigma idealized metrics: c1h=1, c2h=0, mu'=0.  So the recurrence and the
    inverse reduce to the dnw/rdnw sign test that the council pinpointed.
    """
    eta = column["eta"]
    mu = column["mu"]
    nz = column["nz"]
    al_init = column["al_init"]
    alb = column["alb"]
    # mu' = 0 (fixed-mass thermal); c1h=1, c2h=0 (pure sigma)
    mass_h = mu  # (c1h*mub+c2h) + c1h*mu' = mu
    mu_pert = 0.0

    if signed:
        dnw = eta[1:] - eta[:-1]  # WRF signed (negative)
    else:
        dnw = np.abs(eta[1:] - eta[:-1])  # legacy positive |dnw|
    rdnw = 1.0 / dnw

    # WRF recurrence ph'(k+1) = ph'(k) - dnw(k)*( mass_h*al + c1h*mu'*alb )
    ph = np.zeros(nz + 1, dtype=np.float64)
    for k in range(nz):
        ph[k + 1] = ph[k] - dnw[k] * (mass_h * al_init[k] + mu_pert * alb[k])

    # WRF calc_p_rho_phi inverse: al = -1/mass_h * ( alb*c1h*mu' + rdnw*(ph(k+1)-ph(k)) )
    al_calc = -(alb * mu_pert + rdnw * (ph[1:] - ph[:-1])) / mass_h
    err = np.abs(al_calc - al_init)
    return float(np.max(err)), al_calc, ph


def main() -> int:
    proof_dir = Path("proofs/f7g")
    proof_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    overall_pass = True
    for case in ("warm_bubble", "density_current"):
        column = _build_column(case)
        err_signed, al_signed, ph_signed = _roundtrip(column, signed=True)
        err_positive, _, _ = _roundtrip(column, signed=False)
        passed = err_signed <= 1.0e-12
        overall_pass = overall_pass and passed
        results[case] = {
            "nz": column["nz"],
            "max_abs_err_signed_rdnw": err_signed,
            "max_abs_err_positive_rdnw": err_positive,
            "tolerance": 1.0e-12,
            "passed_signed": bool(passed),
            "al_init_min": float(np.min(column["al_init"])),
            "al_init_max": float(np.max(column["al_init"])),
            "ph_prime_top_signed_m2_s2": float(ph_signed[-1]),
        }
    payload = {
        "schema": "f7g_signed_metric_roundtrip",
        "schema_version": 1,
        "check": "AC1",
        "description": (
            "Algebraic round-trip of the WRF-signed calc_p_rho_phi inverse against the "
            "idealized perturbation-geopotential recurrence. Signed dnw/rdnw must give "
            "max_abs(al_calc-al_init) <= 1e-12; positive |dnw| breaks the inverse."
        ),
        "wrf_refs": [
            "module_initialize_ideal.F:711-713 (dnw=znw(k+1)-znw(k), rdnw=1/dnw, signed)",
            "module_initialize_ideal.F:1124-1129 / :1308-1313 (ph' recurrence)",
            "start_em.F:819-828 (calc_p_rho_phi al inverse, hypsometric_opt=1)",
            "module_big_step_utilities_em.F:1023-1030 (timestep calc_p_rho_phi al)",
        ],
        "cases": results,
        "verdict": "PASS" if overall_pass else "FAIL",
    }
    out = proof_dir / "signed_metric_roundtrip.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({c: results[c]["max_abs_err_signed_rdnw"] for c in results}, indent=2))
    print(json.dumps({c + "_positive": results[c]["max_abs_err_positive_rdnw"] for c in results}, indent=2))
    print("AC1 verdict:", payload["verdict"])
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
