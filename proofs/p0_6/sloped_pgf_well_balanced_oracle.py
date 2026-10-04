"""P0-6 sloped-coordinate horizontal PGF well-balanced oracle.

This is an independent NumPy transcription of pristine WRF
``dyn_em/module_big_step_utilities_em.F:2183-2404`` run against the current
``gpuwrf.dynamics.core.rk_addtend_dry.large_step_horizontal_pgf`` implementation.

The fixture is a deterministic, savepoint-shaped analytic hydrostatic-rest
column over steep terrain.  It is not a JAX-vs-JAX self-compare and it does not
claim an extracted WRF-binary savepoint; the JSON proof records that limitation.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp

from gpuwrf.contracts.grid import DycoreMetrics, GridSpec
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.dynamics.acoustic_wrf import CVPM, P0_PA, R_D
from gpuwrf.dynamics.core.rk_addtend_dry import large_step_horizontal_pgf


PARITY_TOLERANCE = 2.0e-9
WELL_BALANCED_ACCEL_TOLERANCE_M_S2 = 1.0e-10
DX_M = 3000.0
DY_M = 3000.0
GRAVITY = 9.81


def _x_pair_3d_np(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.pad(field, ((0, 0), (0, 0), (1, 1)), mode="edge")
    return padded[:, :, :-1], padded[:, :, 1:]


def _y_pair_3d_np(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.pad(field, ((0, 0), (1, 1), (0, 0)), mode="edge")
    return padded[:, :-1, :], padded[:, 1:, :]


def _x_pair_2d_np(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.pad(field, ((0, 0), (1, 1)), mode="edge")
    return padded[:, :-1], padded[:, 1:]


def _y_pair_2d_np(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.pad(field, ((1, 1), (0, 0)), mode="edge")
    return padded[:-1, :], padded[1:, :]


def _dpn_faces_np(
    left: np.ndarray,
    right: np.ndarray,
    *,
    cf1: float,
    cf2: float,
    cf3: float,
    fnm: np.ndarray,
    fnp: np.ndarray,
    top_lid: bool,
) -> np.ndarray:
    pair_sum = left + right
    nz = pair_sum.shape[0]
    dpn = np.zeros((nz + 1,) + pair_sum.shape[1:], dtype=np.float64)
    dpn[0] = 0.5 * (cf1 * pair_sum[0] + cf2 * pair_sum[1] + cf3 * pair_sum[2])
    dpn[1:nz] = 0.5 * (
        fnm[1:, None, None] * pair_sum[1:, :, :] + fnp[1:, None, None] * pair_sum[:-1, :, :]
    )
    if top_lid:
        dpn[nz] = 0.5 * (cf1 * pair_sum[-1] + cf2 * pair_sum[-2] + cf3 * pair_sum[-3])
    return dpn


def _vertical_dpn_bracket_1d(
    p_profile: np.ndarray,
    mu_perturbation: float,
    metrics: dict[str, np.ndarray | float],
    *,
    top_lid: bool,
) -> np.ndarray:
    """Return the WRF fourth-term bracket for horizontally uniform p/mu."""

    nz = p_profile.shape[0]
    pair_sum = 2.0 * p_profile
    dpn = np.zeros((nz + 1,), dtype=np.float64)
    dpn[0] = 0.5 * (
        float(metrics["cf1"]) * pair_sum[0]
        + float(metrics["cf2"]) * pair_sum[1]
        + float(metrics["cf3"]) * pair_sum[2]
    )
    fnm = np.asarray(metrics["fnm"], dtype=np.float64)
    fnp = np.asarray(metrics["fnp"], dtype=np.float64)
    dpn[1:nz] = 0.5 * (fnm[1:] * pair_sum[1:] + fnp[1:] * pair_sum[:-1])
    if top_lid:
        dpn[nz] = 0.5 * (
            float(metrics["cf1"]) * pair_sum[-1]
            + float(metrics["cf2"]) * pair_sum[-2]
            + float(metrics["cf3"]) * pair_sum[-3]
        )
    c1h = np.asarray(metrics["c1h"], dtype=np.float64)
    rdnw = np.asarray(metrics["rdnw"], dtype=np.float64)
    return rdnw * (dpn[1:] - dpn[:-1]) - c1h * float(mu_perturbation)


def _wrf_large_step_pgf_np(
    case: dict[str, np.ndarray | float],
    *,
    top_lid: bool = False,
) -> dict[str, np.ndarray]:
    ph = np.asarray(case["ph_perturbation"], dtype=np.float64)
    p = np.asarray(case["p_perturbation"], dtype=np.float64)
    pb = np.asarray(case["pb"], dtype=np.float64)
    al = np.asarray(case["al"], dtype=np.float64)
    alt = np.asarray(case["alt"], dtype=np.float64)
    php = np.asarray(case["php"], dtype=np.float64)
    mu_total = np.asarray(case["mu_total"], dtype=np.float64)
    mu_pert = np.asarray(case["mu_perturbation"], dtype=np.float64)
    c1h = np.asarray(case["c1h"], dtype=np.float64)
    c2h = np.asarray(case["c2h"], dtype=np.float64)
    rdnw = np.asarray(case["rdnw"], dtype=np.float64)
    msfux = np.asarray(case["msfux"], dtype=np.float64)
    msfuy = np.asarray(case["msfuy"], dtype=np.float64)
    msfvx = np.asarray(case["msfvx"], dtype=np.float64)
    msfvy = np.asarray(case["msfvy"], dtype=np.float64)
    fnm = np.asarray(case["fnm"], dtype=np.float64)
    fnp = np.asarray(case["fnp"], dtype=np.float64)
    rdx = 1.0 / float(case["dx_m"])
    rdy = 1.0 / float(case["dy_m"])

    ph_l, ph_r = _x_pair_3d_np(ph)
    p_l, p_r = _x_pair_3d_np(p)
    pb_l, pb_r = _x_pair_3d_np(pb)
    al_l, al_r = _x_pair_3d_np(al)
    alt_l, alt_r = _x_pair_3d_np(alt)
    mu_l, mu_r = _x_pair_2d_np(mu_total)
    mu_pert_l, mu_pert_r = _x_pair_2d_np(mu_pert)
    mass_u = c1h[:, None, None] * (0.5 * (mu_l + mu_r))[None, :, :] + c2h[:, None, None]
    ph_term_x = (ph_r[1:] - ph_l[1:]) + (ph_r[:-1] - ph_l[:-1])
    p_term_x = (alt_l + alt_r) * (p_r - p_l)
    pb_term_x = (al_l + al_r) * (pb_r - pb_l)
    msf_u = (msfux / msfuy)[None, :, :]
    dpx = msf_u * 0.5 * rdx * mass_u * (ph_term_x + p_term_x + pb_term_x)

    php_l, php_r = _x_pair_3d_np(php)
    dpn_x = _dpn_faces_np(
        p_l,
        p_r,
        cf1=float(case["cf1"]),
        cf2=float(case["cf2"]),
        cf3=float(case["cf3"]),
        fnm=fnm,
        fnp=fnp,
        top_lid=top_lid,
    )
    bracket_x = rdnw[:, None, None] * (dpn_x[1:] - dpn_x[:-1]) - 0.5 * (
        c1h[:, None, None] * (mu_pert_l + mu_pert_r)[None, :, :]
    )
    dpx = dpx + msf_u * rdx * (php_r - php_l) * bracket_x

    ph_s, ph_n = _y_pair_3d_np(ph)
    p_s, p_n = _y_pair_3d_np(p)
    pb_s, pb_n = _y_pair_3d_np(pb)
    al_s, al_n = _y_pair_3d_np(al)
    alt_s, alt_n = _y_pair_3d_np(alt)
    mu_s, mu_n = _y_pair_2d_np(mu_total)
    mu_pert_s, mu_pert_n = _y_pair_2d_np(mu_pert)
    mass_v = c1h[:, None, None] * (0.5 * (mu_s + mu_n))[None, :, :] + c2h[:, None, None]
    ph_term_y = (ph_n[1:] - ph_s[1:]) + (ph_n[:-1] - ph_s[:-1])
    p_term_y = (alt_s + alt_n) * (p_n - p_s)
    pb_term_y = (al_s + al_n) * (pb_n - pb_s)
    msf_v = (msfvy / msfvx)[None, :, :]
    dpy = msf_v * 0.5 * rdy * mass_v * (ph_term_y + p_term_y + pb_term_y)

    php_s, php_n = _y_pair_3d_np(php)
    dpn_y = _dpn_faces_np(
        p_s,
        p_n,
        cf1=float(case["cf1"]),
        cf2=float(case["cf2"]),
        cf3=float(case["cf3"]),
        fnm=fnm,
        fnp=fnp,
        top_lid=top_lid,
    )
    bracket_y = rdnw[:, None, None] * (dpn_y[1:] - dpn_y[:-1]) - 0.5 * (
        c1h[:, None, None] * (mu_pert_s + mu_pert_n)[None, :, :]
    )
    dpy = dpy + msf_v * rdy * (php_n - php_s) * bracket_y

    return {
        "ru_tend": -dpx,
        "rv_tend": -dpy,
        "dpx": dpx,
        "dpy": dpy,
        "mass_u": mass_u,
        "mass_v": mass_v,
    }


def _build_metrics(grid: GridSpec) -> DycoreMetrics:
    base = grid.metrics
    assert base is not None
    ny, nx = grid.ny, grid.nx
    jj, ii = np.indices((ny, nx), dtype=np.float64)
    msftx = 0.93 + 0.006 * ii + 0.004 * jj
    msfty = 0.95 + 0.003 * ii + 0.005 * jj
    msfux_core = 0.92 + 0.005 * ii + 0.003 * jj
    msfuy_core = 0.96 + 0.004 * ii + 0.006 * jj
    msfvx_core = 0.97 + 0.003 * ii + 0.004 * jj
    msfvy_core = 0.91 + 0.006 * ii + 0.002 * jj
    return replace(
        base,
        msftx=jnp.asarray(msftx),
        msfty=jnp.asarray(msfty),
        msfux=jnp.asarray(np.pad(msfux_core, ((0, 0), (0, 1)), mode="edge")),
        msfuy=jnp.asarray(np.pad(msfuy_core, ((0, 0), (0, 1)), mode="edge")),
        msfvx=jnp.asarray(np.pad(msfvx_core, ((0, 1), (0, 0)), mode="edge")),
        msfvy=jnp.asarray(np.pad(msfvy_core, ((0, 1), (0, 0)), mode="edge")),
        provenance="p0_6_sloped_pgf_analytic_steep_terrain",
    )


def _island_terrain(ny: int, nx: int) -> np.ndarray:
    y = np.linspace(-1.0, 1.0, ny, dtype=np.float64)
    x = np.linspace(-1.0, 1.0, nx, dtype=np.float64)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    cone = 3550.0 * np.exp(-((xx / 0.42) ** 2 + (yy / 0.32) ** 2))
    ridge = 950.0 * np.exp(-((xx + 0.25) / 0.18) ** 2) * np.exp(-(yy / 0.85) ** 2)
    return cone + ridge


def _empty_state_arrays(grid: GridSpec) -> dict[str, jnp.ndarray]:
    return {field: jnp.zeros(shape, dtype=jnp.float64) for field, shape in _state_field_shapes(grid).items()}


def _theta_for_alt(alt: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    return alt / ((R_D / P0_PA) * ((pressure / P0_PA) ** CVPM))


def _build_case() -> tuple[State, DycoreMetrics, dict[str, np.ndarray | float]]:
    grid = GridSpec.canary_3km_template()
    metrics = _build_metrics(grid)
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    c1h = np.asarray(metrics.c1h)
    c2h = np.asarray(metrics.c2h)
    c1f = np.asarray(metrics.c1f)
    rdnw = np.asarray(metrics.rdnw)
    dnw = np.asarray(metrics.dnw)
    fnm = np.asarray(metrics.fnm)
    fnp = np.asarray(metrics.fnp)
    ht = _island_terrain(ny, nx)

    # Hydrostatic-rest pressure is horizontally uniform.  The vertical profile is
    # intentionally non-linear so WRF's fourth-term bracket is nonzero.
    eta_mass = c1h
    p_profile = 150.0 * eta_mass + 42.0 * eta_mass**2 + 8.0 * np.sin(np.pi * eta_mass)
    mu_pert_scalar = 18.0
    mub_scalar = 85000.0
    bracket = _vertical_dpn_bracket_1d(
        p_profile,
        mu_pert_scalar,
        {
            "cf1": float(metrics.cf1),
            "cf2": float(metrics.cf2),
            "cf3": float(metrics.cf3),
            "fnm": fnm,
            "fnp": fnp,
            "rdnw": rdnw,
            "c1h": c1h,
        },
        top_lid=False,
    )
    # WRF's first-three PGF terms use the full dry mass in ``muu/muv`` while
    # the nonhydrostatic fourth-term bracket subtracts perturbation mu.
    mass_h = c1h * (mub_scalar + mu_pert_scalar) + c2h
    phsum_per_m = -2.0 * GRAVITY * bracket / (mass_h + bracket)
    ph_face_per_m = np.zeros((nz + 1,), dtype=np.float64)
    for k in range(nz):
        ph_face_per_m[k + 1] = phsum_per_m[k] - ph_face_per_m[k]

    ph_pert = ph_face_per_m[:, None, None] * ht[None, :, :]
    z_ref = np.linspace(0.0, 16000.0, nz + 1, dtype=np.float64)
    phb = GRAVITY * (z_ref[:, None, None] + ht[None, :, :])
    ph_total = phb + ph_pert
    php = 0.5 * (ph_total[:-1] + ph_total[1:])

    p_pert = p_profile[:, None, None] * np.ones((nz, ny, nx), dtype=np.float64)
    pb_profile = 93000.0 - 5200.0 * (1.0 - eta_mass)
    pb = pb_profile[:, None, None] * np.ones((nz, ny, nx), dtype=np.float64)
    p_total = pb + p_pert
    alt_profile = 0.82 + 0.08 * (1.0 - eta_mass)
    alt = alt_profile[:, None, None] * np.ones((nz, ny, nx), dtype=np.float64)
    theta = _theta_for_alt(alt, p_total)
    mu_pert = np.full((ny, nx), mu_pert_scalar, dtype=np.float64)
    mu_total = np.full((ny, nx), mub_scalar + mu_pert_scalar, dtype=np.float64)
    mass_h_3d = c1h[:, None, None] * mub_scalar + c2h[:, None, None]
    al = -(
        alt * c1h[:, None, None] * mu_pert[None, :, :]
        + rdnw[:, None, None] * (ph_pert[1:] - ph_pert[:-1])
    ) / mass_h_3d

    arrays = _empty_state_arrays(grid)
    arrays.update(
        {
            "theta": jnp.asarray(theta),
            "p": jnp.asarray(p_pert),
            "p_total": jnp.asarray(p_total),
            "p_perturbation": jnp.asarray(p_pert),
            "ph": jnp.asarray(ph_pert),
            "ph_total": jnp.asarray(ph_total),
            "ph_perturbation": jnp.asarray(ph_pert),
            "mu": jnp.asarray(mu_pert),
            "mu_total": jnp.asarray(mu_total),
            "mu_perturbation": jnp.asarray(mu_pert),
        }
    )
    state = State(**arrays)

    case: dict[str, np.ndarray | float] = {
        "ph_perturbation": ph_pert,
        "p_perturbation": p_pert,
        "pb": pb,
        "al": al,
        "alt": alt,
        "php": php,
        "mu_total": mu_total,
        "mu_perturbation": mu_pert,
        "c1h": c1h,
        "c2h": c2h,
        "c1f": c1f,
        "dnw": dnw,
        "rdnw": rdnw,
        "cf1": float(metrics.cf1),
        "cf2": float(metrics.cf2),
        "cf3": float(metrics.cf3),
        "fnm": fnm,
        "fnp": fnp,
        "msfux": np.asarray(metrics.msfux),
        "msfuy": np.asarray(metrics.msfuy),
        "msfvx": np.asarray(metrics.msfvx),
        "msfvy": np.asarray(metrics.msfvy),
        "dx_m": DX_M,
        "dy_m": DY_M,
        "terrain_height": ht,
        "p_profile": p_profile,
        "fourth_term_bracket": bracket,
        "phsum_per_m": phsum_per_m,
    }
    return state, metrics, case


def _max_abs_diff(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64))))


def _summary_stats(array: np.ndarray) -> dict[str, float]:
    arr = np.asarray(array, dtype=np.float64)
    return {
        "max_abs": float(np.max(np.abs(arr))),
        "mean_abs": float(np.mean(np.abs(arr))),
        "rms": float(np.sqrt(np.mean(arr * arr))),
    }


def build_report(*, write_json: bool = False) -> dict[str, Any]:
    state, metrics, case = _build_case()
    wrf = _wrf_large_step_pgf_np(case, top_lid=False)
    ru_actual, rv_actual = large_step_horizontal_pgf(
        state,
        metrics,
        dx_m=DX_M,
        dy_m=DY_M,
        non_hydrostatic=True,
        top_lid=False,
    )
    ru_np = np.asarray(ru_actual, dtype=np.float64)
    rv_np = np.asarray(rv_actual, dtype=np.float64)

    ru_accel = ru_np / wrf["mass_u"]
    rv_accel = rv_np / wrf["mass_v"]
    wrf_ru_accel = wrf["ru_tend"] / wrf["mass_u"]
    wrf_rv_accel = wrf["rv_tend"] / wrf["mass_v"]
    parity = {
        "ru_tend_max_abs_diff": _max_abs_diff(ru_np, wrf["ru_tend"]),
        "rv_tend_max_abs_diff": _max_abs_diff(rv_np, wrf["rv_tend"]),
        "ru_accel_max_abs_diff_m_s2": _max_abs_diff(ru_accel, wrf_ru_accel),
        "rv_accel_max_abs_diff_m_s2": _max_abs_diff(rv_accel, wrf_rv_accel),
    }
    max_spurious_accel = max(
        _summary_stats(ru_accel)["max_abs"],
        _summary_stats(rv_accel)["max_abs"],
    )
    reference_max_spurious_accel = max(
        _summary_stats(wrf_ru_accel)["max_abs"],
        _summary_stats(wrf_rv_accel)["max_abs"],
    )
    parity_pass = all(value <= PARITY_TOLERANCE for value in parity.values())
    well_balanced_pass = max_spurious_accel <= WELL_BALANCED_ACCEL_TOLERANCE_M_S2
    current_status = "PASS" if parity_pass and well_balanced_pass else "FAIL"

    report = {
        "schema": "p0_6_sloped_pgf_well_balanced_oracle",
        "schema_version": 1,
        "status": current_status,
        "diagnostic_completed": True,
        "source": {
            "wrf_pristine_root": "$WRF_PRISTINE_ROOT",
            "horizontal_pressure_gradient": "dyn_em/module_big_step_utilities_em.F:2183-2404",
            "current_gpu_operator": "src/gpuwrf/dynamics/core/rk_addtend_dry.py:large_step_horizontal_pgf",
        },
        "fixture": {
            "kind": "savepoint-shaped analytic WRF transcription",
            "wrf_binary_savepoint_extracted": False,
            "wrf_binary_savepoint_limitation": (
                "No committed P0-6 horizontal_pressure_gradient WRF-binary savepoint fixture was present in this worktree; "
                "this proof uses the pristine-WRF formula as an independent NumPy oracle."
            ),
            "hydrostatic_rest_construction": (
                "Horizontally uniform perturbation pressure and dry-mass perturbation produce a nonzero WRF fourth-term "
                "vertical bracket.  Perturbation geopotential is solved so the first-three PGF terms and the terrain-slope "
                "fourth term cancel discretely over a steep island terrain."
            ),
            "nx": int(np.asarray(case["terrain_height"]).shape[1]),
            "ny": int(np.asarray(case["terrain_height"]).shape[0]),
            "nz": int(np.asarray(case["p_profile"]).shape[0]),
            "dx_m": DX_M,
            "dy_m": DY_M,
            "terrain_max_m": float(np.max(np.asarray(case["terrain_height"]))),
            "terrain_max_slope_proxy_m_per_cell": float(
                max(
                    np.max(np.abs(np.diff(np.asarray(case["terrain_height"]), axis=1))),
                    np.max(np.abs(np.diff(np.asarray(case["terrain_height"]), axis=0))),
                )
            ),
            "fourth_term_bracket_max_abs": float(np.max(np.abs(np.asarray(case["fourth_term_bracket"])))),
        },
        "tolerances": {
            "parity_abs": PARITY_TOLERANCE,
            "well_balanced_acceleration_m_s2": WELL_BALANCED_ACCEL_TOLERANCE_M_S2,
        },
        "wrf_reference": {
            "ru_accel": _summary_stats(wrf_ru_accel),
            "rv_accel": _summary_stats(wrf_rv_accel),
            "max_abs_spurious_acceleration_m_s2": reference_max_spurious_accel,
        },
        "current_operator": {
            "status": current_status,
            "parity_pass": bool(parity_pass),
            "well_balanced_pass": bool(well_balanced_pass),
            "parity": parity,
            "ru_accel": _summary_stats(ru_accel),
            "rv_accel": _summary_stats(rv_accel),
            "max_abs_spurious_acceleration_m_s2": max_spurious_accel,
            "requires_operator_rewrite": bool(not current_status == "PASS"),
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "JAX_PLATFORM_NAME": os.environ.get("JAX_PLATFORM_NAME"),
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        },
    }
    if write_json:
        path = ROOT / "proofs" / "p0_6" / "sloped_pgf_well_balanced_oracle.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    report = build_report(write_json=True)
    print(
        json.dumps(
            {
                "status": report["status"],
                "max_abs_spurious_acceleration_m_s2": report["current_operator"][
                    "max_abs_spurious_acceleration_m_s2"
                ],
                "parity": report["current_operator"]["parity"],
                "requires_operator_rewrite": report["current_operator"]["requires_operator_rewrite"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
