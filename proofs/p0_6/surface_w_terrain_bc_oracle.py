"""P0-6 terrain-following surface-w lower-boundary oracle.

This proof isolates WRF ``advance_w`` lower-boundary chain rule
(``dyn_em/module_small_step_em.F:1372-1394``) from the current production feed
choice in ``src/gpuwrf/dynamics/core/acoustic.py:620-645``.

It tests both:

* WRF-coupled acoustic work-array winds, which are the pristine WRF feed.
* Decoupled stage winds, which are the current production deviation.

The result is a diagnostic oracle: the script exits successfully when it writes
the proof object, even if the current production feed fails WRF parity.
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
from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients
from gpuwrf.dynamics.core.advance_w import GRAVITY_M_S2, advance_w_wrf, dry_cqw


SURFACE_PARITY_TOLERANCE = 1.0e-9
FINITE_GROWTH_TOLERANCE = 1.0e-9
DX_M = 3000.0
DY_M = 3000.0
DTS = 1.0
EPSSM = 0.5


def _build_metrics(grid: GridSpec) -> DycoreMetrics:
    base = grid.metrics
    assert base is not None
    ny, nx = grid.ny, grid.nx
    jj, ii = np.indices((ny, nx), dtype=np.float64)
    msftx = 0.94 + 0.004 * ii + 0.003 * jj
    msfty = 0.96 + 0.003 * ii + 0.004 * jj
    msfux_core = 0.93 + 0.004 * ii + 0.002 * jj
    msfuy_core = 0.97 + 0.002 * ii + 0.005 * jj
    msfvx_core = 0.98 + 0.003 * ii + 0.003 * jj
    msfvy_core = 0.92 + 0.005 * ii + 0.002 * jj
    # The idealized flat metrics have c2*=0, which makes the open-top face mass
    # zero at c1f(kde)=0.  Real WRF hybrid coordinates carry p_top through c2f,
    # so include that here to keep the implicit solve finite.
    p_top = float(np.asarray(base.p_top))
    c2h = p_top * (1.0 - np.asarray(base.c1h))
    c2f = p_top * (1.0 - np.asarray(base.c1f))
    return replace(
        base,
        msftx=jnp.asarray(msftx),
        msfty=jnp.asarray(msfty),
        msfux=jnp.asarray(np.pad(msfux_core, ((0, 0), (0, 1)), mode="edge")),
        msfuy=jnp.asarray(np.pad(msfuy_core, ((0, 0), (0, 1)), mode="edge")),
        msfvx=jnp.asarray(np.pad(msfvx_core, ((0, 1), (0, 0)), mode="edge")),
        msfvy=jnp.asarray(np.pad(msfvy_core, ((0, 1), (0, 0)), mode="edge")),
        c2h=jnp.asarray(c2h),
        c2f=jnp.asarray(c2f),
        provenance="p0_6_surface_w_terrain_bc_oracle",
    )


def _terrain(ny: int, nx: int) -> np.ndarray:
    y = np.linspace(-1.0, 1.0, ny, dtype=np.float64)
    x = np.linspace(-1.0, 1.0, nx, dtype=np.float64)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    cone = 3650.0 * np.exp(-((xx / 0.38) ** 2 + (yy / 0.34) ** 2))
    ridge = 1200.0 * np.exp(-((xx - 0.28) / 0.16) ** 2) * np.exp(-((yy + 0.12) / 0.80) ** 2)
    return cone + ridge


def _surface_w_np(
    *,
    u: np.ndarray,
    v: np.ndarray,
    ht: np.ndarray,
    msftx: np.ndarray,
    msfty: np.ndarray,
    cf1: float,
    cf2: float,
    cf3: float,
    rdx: float,
    rdy: float,
) -> np.ndarray:
    """Pristine-WRF chain-rule surface ``w(i,1,j)`` transcription."""

    ht_dy_n = np.pad(ht, ((0, 1), (0, 0)), mode="edge")[1:, :] - ht
    ht_dy_s = ht - np.pad(ht, ((1, 0), (0, 0)), mode="edge")[:-1, :]
    ht_dx_e = np.pad(ht, ((0, 0), (0, 1)), mode="edge")[:, 1:] - ht
    ht_dx_w = ht - np.pad(ht, ((0, 0), (1, 0)), mode="edge")[:, :-1]
    v_n = cf1 * v[0, 1:, :] + cf2 * v[1, 1:, :] + cf3 * v[2, 1:, :]
    v_s = cf1 * v[0, :-1, :] + cf2 * v[1, :-1, :] + cf3 * v[2, :-1, :]
    u_e = cf1 * u[0, :, 1:] + cf2 * u[1, :, 1:] + cf3 * u[2, :, 1:]
    u_w = cf1 * u[0, :, :-1] + cf2 * u[1, :, :-1] + cf3 * u[2, :, :-1]
    return (
        msfty * 0.5 * float(rdy) * (ht_dy_n * v_n + ht_dy_s * v_s)
        + msftx * 0.5 * float(rdx) * (ht_dx_e * u_e + ht_dx_w * u_w)
    )


def _x_face_pair_2d(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.pad(field, ((0, 0), (1, 1)), mode="edge")
    return padded[:, :-1], padded[:, 1:]


def _y_face_pair_2d(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.pad(field, ((1, 1), (0, 0)), mode="edge")
    return padded[:-1, :], padded[1:, :]


def _build_case() -> dict[str, Any]:
    grid = GridSpec.canary_3km_template()
    metrics = _build_metrics(grid)
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    ht = _terrain(ny, nx)
    jj_u, ii_u = np.indices((ny, nx + 1), dtype=np.float64)
    jj_v, ii_v = np.indices((ny + 1, nx), dtype=np.float64)
    kk_u = np.arange(nz, dtype=np.float64)[:, None, None]
    kk_v = np.arange(nz, dtype=np.float64)[:, None, None]

    # Physical stage winds: O(1-10 m/s), vertically varying near the surface.
    u_physical = (
        8.0
        + 0.18 * kk_u
        + 1.4 * np.sin(0.55 * ii_u[None, :, :])
        - 0.45 * np.cos(0.35 * jj_u[None, :, :])
    )
    v_physical = (
        -4.0
        + 0.12 * kk_v
        + 1.1 * np.cos(0.42 * jj_v[None, :, :])
        + 0.35 * np.sin(0.50 * ii_v[None, :, :])
    )

    mut = np.full((ny, nx), 85000.0, dtype=np.float64)
    mu_total = mut.copy()
    muu_l, muu_r = _x_face_pair_2d(mu_total)
    muv_s, muv_n = _y_face_pair_2d(mu_total)
    c1h = np.asarray(metrics.c1h)
    c2h = np.asarray(metrics.c2h)
    mass_u = c1h[:, None, None] * (0.5 * (muu_l + muu_r))[None, :, :] + c2h[:, None, None]
    mass_v = c1h[:, None, None] * (0.5 * (muv_s + muv_n))[None, :, :] + c2h[:, None, None]
    u_coupled = mass_u * u_physical / np.asarray(metrics.msfuy)[None, :, :]
    v_coupled = mass_v * v_physical / np.asarray(metrics.msfvx)[None, :, :]

    phb_z = np.linspace(0.0, 16000.0, nz + 1, dtype=np.float64)
    phb = GRAVITY_M_S2 * (phb_z[:, None, None] + ht[None, :, :])
    cqw = dry_cqw(nz, ny, nx, dtype=jnp.float64)
    c2a = jnp.ones((nz, ny, nx), dtype=jnp.float64)
    a, alpha, gamma = calc_coef_w_wrf_coefficients(
        jnp.asarray(mut),
        metrics,
        dt=DTS,
        epssm=EPSSM,
        top_lid=False,
        cqw=cqw,
        c2a=c2a,
    )

    return {
        "grid": grid,
        "metrics": metrics,
        "ht": ht,
        "mut": mut,
        "mu_total": mu_total,
        "u_physical": u_physical,
        "v_physical": v_physical,
        "u_coupled": u_coupled,
        "v_coupled": v_coupled,
        "phb": phb,
        "cqw": cqw,
        "c2a": c2a,
        "a": a,
        "alpha": alpha,
        "gamma": gamma,
    }


def _advance_once(
    case: dict[str, Any],
    *,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray | None = None,
    ph: np.ndarray | None = None,
    t_2ave: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metrics: DycoreMetrics = case["metrics"]
    nz = int(metrics.c1h.shape[0])
    ny, nx = case["mut"].shape
    w0 = np.zeros((nz + 1, ny, nx), dtype=np.float64) if w is None else w
    ph0 = np.zeros((nz + 1, ny, nx), dtype=np.float64) if ph is None else ph
    t2ave0 = np.zeros((nz, ny, nx), dtype=np.float64) if t_2ave is None else t_2ave
    zeros_mass = jnp.zeros((nz, ny, nx), dtype=jnp.float64)
    zeros_face = jnp.zeros((nz + 1, ny, nx), dtype=jnp.float64)
    zeros_2d = jnp.zeros((ny, nx), dtype=jnp.float64)

    w_next, ph_next, t2ave_next = advance_w_wrf(
        w=jnp.asarray(w0),
        rw_tend=zeros_face,
        ww=zeros_face,
        u=jnp.asarray(u),
        v=jnp.asarray(v),
        mu_work=zeros_2d,
        mut=jnp.asarray(case["mut"]),
        muave=zeros_2d,
        muts=jnp.asarray(case["mut"]),
        t_2ave=jnp.asarray(t2ave0),
        t_2=zeros_mass,
        t_1=zeros_mass,
        ph=jnp.asarray(ph0),
        ph_1=zeros_face,
        phb=jnp.asarray(case["phb"]),
        ph_tend=zeros_face,
        ht=jnp.asarray(case["ht"]),
        c2a=case["c2a"],
        cqw=case["cqw"],
        alt=jnp.ones((nz, ny, nx), dtype=jnp.float64),
        a=case["a"],
        alpha=case["alpha"],
        gamma=case["gamma"],
        c1h=metrics.c1h,
        c2h=metrics.c2h,
        c1f=metrics.c1f,
        c2f=metrics.c2f,
        rdnw=metrics.rdnw,
        rdn=metrics.rdn,
        fnm=metrics.fnm,
        fnp=metrics.fnp,
        cf1=metrics.cf1,
        cf2=metrics.cf2,
        cf3=metrics.cf3,
        msftx=metrics.msftx,
        msfty=metrics.msfty,
        rdx=1.0 / DX_M,
        rdy=1.0 / DY_M,
        dts=DTS,
        epssm=EPSSM,
        top_lid=False,
        damp_opt=0,
        w_damping=0,
    )
    return np.asarray(w_next), np.asarray(ph_next), np.asarray(t2ave_next)


def _run_repeated(case: dict[str, Any], *, u: np.ndarray, v: np.ndarray, steps: int = 8) -> dict[str, Any]:
    w = None
    ph = None
    t2ave = None
    surface_max = []
    interior_max = []
    finite = []
    for _ in range(steps):
        w, ph, t2ave = _advance_once(case, u=u, v=v, w=w, ph=ph, t_2ave=t2ave)
        surface_max.append(float(np.max(np.abs(w[0]))))
        interior_max.append(float(np.max(np.abs(w[1:]))))
        finite.append(bool(np.isfinite(w).all() and np.isfinite(ph).all() and np.isfinite(t2ave).all()))
    surface_growth_ratio = surface_max[-1] / max(surface_max[0], 1.0e-300)
    interior_growth_ratio = interior_max[-1] / max(interior_max[0], 1.0e-300)
    k0_only_growth = surface_growth_ratio > (1.0 + FINITE_GROWTH_TOLERANCE) and interior_growth_ratio <= (
        1.0 + FINITE_GROWTH_TOLERANCE
    )
    return {
        "steps": steps,
        "all_finite": bool(all(finite)),
        "surface_max_abs_by_step": surface_max,
        "interior_max_abs_by_step": interior_max,
        "surface_growth_ratio": float(surface_growth_ratio),
        "interior_growth_ratio": float(interior_growth_ratio),
        "k0_only_growth_detected": bool(k0_only_growth),
        "no_k0_only_growth_pass": bool(all(finite) and not k0_only_growth),
    }


def _max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))


def _stats(arr: np.ndarray) -> dict[str, float]:
    a = np.asarray(arr, dtype=np.float64)
    return {
        "max_abs": float(np.max(np.abs(a))),
        "mean_abs": float(np.mean(np.abs(a))),
        "rms": float(np.sqrt(np.mean(a * a))),
    }


def build_report(*, write_json: bool = False) -> dict[str, Any]:
    case = _build_case()
    metrics: DycoreMetrics = case["metrics"]
    expected_coupled = _surface_w_np(
        u=case["u_coupled"],
        v=case["v_coupled"],
        ht=case["ht"],
        msftx=np.asarray(metrics.msftx),
        msfty=np.asarray(metrics.msfty),
        cf1=float(metrics.cf1),
        cf2=float(metrics.cf2),
        cf3=float(metrics.cf3),
        rdx=1.0 / DX_M,
        rdy=1.0 / DY_M,
    )
    expected_decoupled = _surface_w_np(
        u=case["u_physical"],
        v=case["v_physical"],
        ht=case["ht"],
        msftx=np.asarray(metrics.msftx),
        msfty=np.asarray(metrics.msfty),
        cf1=float(metrics.cf1),
        cf2=float(metrics.cf2),
        cf3=float(metrics.cf3),
        rdx=1.0 / DX_M,
        rdy=1.0 / DY_M,
    )

    w_coupled, ph_coupled, t2_coupled = _advance_once(case, u=case["u_coupled"], v=case["v_coupled"])
    w_decoupled, ph_decoupled, t2_decoupled = _advance_once(case, u=case["u_physical"], v=case["v_physical"])
    coupled_surface = w_coupled[0]
    decoupled_surface = w_decoupled[0]

    coupled_parity_diff = _max_abs_diff(coupled_surface, expected_coupled)
    decoupled_formula_diff = _max_abs_diff(decoupled_surface, expected_decoupled)
    decoupled_vs_wrf_diff = _max_abs_diff(decoupled_surface, expected_coupled)
    coupled_formula_pass = coupled_parity_diff <= SURFACE_PARITY_TOLERANCE
    decoupled_formula_pass = decoupled_formula_diff <= SURFACE_PARITY_TOLERANCE
    current_production_feed_pass = decoupled_vs_wrf_diff <= SURFACE_PARITY_TOLERANCE

    coupled_repeated = _run_repeated(case, u=case["u_coupled"], v=case["v_coupled"], steps=8)
    decoupled_repeated = _run_repeated(case, u=case["u_physical"], v=case["v_physical"], steps=8)
    coupled_solve_finite = bool(np.isfinite(w_coupled).all() and np.isfinite(ph_coupled).all() and np.isfinite(t2_coupled).all())
    decoupled_solve_finite = bool(
        np.isfinite(w_decoupled).all() and np.isfinite(ph_decoupled).all() and np.isfinite(t2_decoupled).all()
    )
    ratio_decoupled_over_coupled = _stats(decoupled_surface)["max_abs"] / max(_stats(coupled_surface)["max_abs"], 1.0e-300)

    status = "PASS" if coupled_formula_pass and current_production_feed_pass else "FAIL"
    report = {
        "schema": "p0_6_surface_w_terrain_bc_oracle",
        "schema_version": 1,
        "status": status,
        "diagnostic_completed": True,
        "summary": {
            "advance_w_formula_with_wrf_coupled_feed": (
                "PASS" if coupled_formula_pass and coupled_solve_finite and coupled_repeated["no_k0_only_growth_pass"] else "FAIL"
            ),
            "current_production_decoupled_feed": "PASS" if current_production_feed_pass else "FAIL",
            "surface_w_max_abs_diff_vs_wrf_coupled": decoupled_vs_wrf_diff,
            "decoupled_over_coupled_surface_w": float(ratio_decoupled_over_coupled),
            "documented_decoupled_u1_v1_feed_is_problem": bool(not current_production_feed_pass),
        },
        "source": {
            "wrf_pristine_root": "$WRF_PRISTINE_ROOT",
            "advance_w_surface_bc": "dyn_em/module_small_step_em.F:1372-1394",
            "current_advance_w_formula": "src/gpuwrf/dynamics/core/advance_w.py:274-303",
            "current_production_feed": "src/gpuwrf/dynamics/core/acoustic.py:620-645",
        },
        "fixture": {
            "kind": "analytic steep synthetic island ridge + WRF advance_w surface path",
            "wrf_binary_savepoint_extracted": False,
            "wrf_binary_savepoint_limitation": (
                "No committed P0-6 advance_w WRF-binary savepoint fixture was present in this worktree; "
                "this proof uses the pristine-WRF chain-rule formula and current advance_w_wrf CPU execution."
            ),
            "nx": int(case["ht"].shape[1]),
            "ny": int(case["ht"].shape[0]),
            "nz": int(case["u_physical"].shape[0]),
            "dx_m": DX_M,
            "dy_m": DY_M,
            "terrain_max_m": float(np.max(case["ht"])),
            "terrain_max_slope_proxy_m_per_cell": float(
                max(np.max(np.abs(np.diff(case["ht"], axis=0))), np.max(np.abs(np.diff(case["ht"], axis=1))))
            ),
            "median_low_level_mass_scale": float(
                np.median(
                    np.asarray(metrics.c1h[:3])[:, None, None] * case["mut"][None, :, :]
                    + np.asarray(metrics.c2h[:3])[:, None, None]
                )
            ),
        },
        "tolerances": {
            "surface_w_abs": SURFACE_PARITY_TOLERANCE,
            "k0_growth_ratio_extra": FINITE_GROWTH_TOLERANCE,
        },
        "advance_w_formula_with_wrf_coupled_feed": {
            "status": "PASS" if coupled_formula_pass and coupled_solve_finite and coupled_repeated["no_k0_only_growth_pass"] else "FAIL",
            "surface_w_parity_max_abs_diff": coupled_parity_diff,
            "surface_w": _stats(coupled_surface),
            "implicit_solve_finite": coupled_solve_finite,
            "repeated_step_harness": coupled_repeated,
            "requires_advance_w_formula_rewrite": bool(
                not (coupled_formula_pass and coupled_solve_finite and coupled_repeated["no_k0_only_growth_pass"])
            ),
        },
        "current_production_decoupled_feed": {
            "status_vs_wrf_coupled_oracle": "PASS" if current_production_feed_pass else "FAIL",
            "surface_w_max_abs_diff_vs_wrf_coupled": decoupled_vs_wrf_diff,
            "surface_w_parity_vs_decoupled_formula_max_abs_diff": decoupled_formula_diff,
            "decoupled_formula_pass": bool(decoupled_formula_pass),
            "surface_w": _stats(decoupled_surface),
            "implicit_solve_finite": decoupled_solve_finite,
            "repeated_step_harness": decoupled_repeated,
            "max_abs_decoupled_over_wrf_coupled_surface_w": float(ratio_decoupled_over_coupled),
            "documented_decoupled_u1_v1_feed_is_problem": bool(not current_production_feed_pass),
            "requires_feed_or_coupling_fix": bool(not current_production_feed_pass),
        },
        "interpretation": (
            "The advance_w.py chain-rule formula matches the WRF coupled-work oracle when supplied WRF-coupled "
            "u/v work arrays.  The current acoustic.py decoupled u_1/v_1 feed fails WRF surface-w parity and "
            "under-energizes the terrain lower boundary by the dry-mass scale."
        ),
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
        path = ROOT / "proofs" / "p0_6" / "surface_w_terrain_bc_oracle.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    report = build_report(write_json=True)
    print(
        json.dumps(
            {
                "status": report["status"],
                "advance_w_formula_with_wrf_coupled_feed": report["advance_w_formula_with_wrf_coupled_feed"][
                    "status"
                ],
                "current_production_decoupled_feed": report["current_production_decoupled_feed"][
                    "status_vs_wrf_coupled_oracle"
                ],
                "surface_w_max_abs_diff_vs_wrf_coupled": report["current_production_decoupled_feed"][
                    "surface_w_max_abs_diff_vs_wrf_coupled"
                ],
                "decoupled_over_coupled_surface_w": report["current_production_decoupled_feed"][
                    "max_abs_decoupled_over_wrf_coupled_surface_w"
                ],
                "documented_decoupled_u1_v1_feed_is_problem": report["current_production_decoupled_feed"][
                    "documented_decoupled_u1_v1_feed_is_problem"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
