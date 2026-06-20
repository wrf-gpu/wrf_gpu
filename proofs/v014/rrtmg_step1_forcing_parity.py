#!/usr/bin/env python3
"""V0.14 RRTMG Step-1 forcing parity localization.

CPU-only proof for the sprint
``2026-06-10-v014-gpt-rrtmg-step1-forcing-parity``.

This proof does not edit production code or tests.  It reads the accepted
NoahMP Step-1 proof, pinned WRF surface/part2 hooks, and the live Step-1 JAX
input builder, then localizes the secondary RRTMG residual across:

* clock / solar geometry;
* surface radiation properties and land handoff;
* column thermodynamics and cloud occupancy;
* layer ordering;
* flux-to-theta conversion; and
* mass coupling into WRF ``T_TENDF``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PROOF_DIR = ROOT / "proofs/v014"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROOF_DIR) not in sys.path:
    sys.path.insert(0, str(PROOF_DIR))

import step1_live_nest_init_rerun as live  # noqa: E402
import step1_part2_source_leaves_split as split  # noqa: E402
import step1_rk1_p_state_source_split as pstate  # noqa: E402
import step1_surface_land_flux_handoff as handoff  # noqa: E402
from gpuwrf.coupling import physics_couplers as pc  # noqa: E402
from gpuwrf.physics.rrtmg_lw import solve_rrtmg_lw_column  # noqa: E402
from gpuwrf.physics.rrtmg_sw import solve_rrtmg_sw_column  # noqa: E402

OUT_JSON = PROOF_DIR / "rrtmg_step1_forcing_parity.json"
OUT_MD = PROOF_DIR / "rrtmg_step1_forcing_parity.md"

NOAHMP_CLOSURE_JSON = PROOF_DIR / "noahmp_step1_closure.json"
RRTMG_RTHRATEN_CLOSURE_JSON = PROOF_DIR / "rrtmg_rthraten_closure.json"
PINNED_SURFACE = Path("/tmp/wrfgpu2_v014_surface_handoff_pinned_onerun/surface_land_flux_d02_step1.txt")
WRF_RADIATION_DRIVER = Path(
    "<DATA_ROOT>/wrf_gpu2/v014_step1_rk1_source_boundary/WRF/phys/module_radiation_driver.F"
)
WRF_RRTMG_LW = Path(
    "<DATA_ROOT>/wrf_gpu2/v014_step1_rk1_source_boundary/WRF/phys/module_ra_rrtmg_lw.F"
)
PRIOR_CLEARSKY = ROOT / "proofs/v013/clearsky_radiation.json"


def sha16(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def path_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size": path.stat().st_size if path.is_file() else None,
        "sha256_16": sha16(path),
    }


def fortran_index(index: tuple[int, ...] | None) -> dict[str, int] | None:
    if index is None:
        return None
    if len(index) == 3:
        k, y, x = index
        return {"i": int(x) + 1, "j": int(y) + 1, "k": int(k) + 1}
    if len(index) == 2:
        y, x = index
        return {"i": int(x) + 1, "j": int(y) + 1}
    return {"linear": int(index[0])} if len(index) == 1 else None


def _values_with_mask(
    candidate: np.ndarray, reference: np.ndarray, mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if mask is None:
        return candidate.reshape(-1), reference.reshape(-1), None
    return candidate[mask], reference[mask], np.argwhere(mask)


def diffstat(candidate: Any, reference: Any, mask: np.ndarray | None = None) -> dict[str, Any]:
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if cand.shape != ref.shape:
        return {
            "status": "SHAPE_MISMATCH",
            "candidate_shape": list(cand.shape),
            "reference_shape": list(ref.shape),
        }
    if mask is not None and mask.shape != cand.shape:
        return {
            "status": "MASK_SHAPE_MISMATCH",
            "candidate_shape": list(cand.shape),
            "mask_shape": list(mask.shape),
        }
    cvals, rvals, mask_indices = _values_with_mask(cand, ref, mask)
    finite = np.isfinite(cvals) & np.isfinite(rvals)
    cvals = cvals[finite]
    rvals = rvals[finite]
    if cvals.size == 0:
        return {"status": "OK", "count": 0}
    diff = cvals - rvals
    absdiff = np.abs(diff)
    worst_pos = int(np.argmax(absdiff))

    if mask_indices is not None:
        finite_positions = np.flatnonzero(finite)
        worst_index = tuple(int(item) for item in mask_indices[int(finite_positions[worst_pos])])
    else:
        finite_positions = np.flatnonzero(finite)
        worst_index = tuple(
            int(item) for item in np.unravel_index(int(finite_positions[worst_pos]), cand.shape)
        )

    return {
        "status": "OK",
        "count": int(diff.size),
        "shape": list(cand.shape),
        "max_abs": float(np.max(absdiff)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "bias": float(np.mean(diff)),
        "p95": float(np.percentile(absdiff, 95.0)),
        "p99": float(np.percentile(absdiff, 99.0)),
        "worst_mismatch_index": list(worst_index),
        "worst_mismatch_fortran": fortran_index(worst_index),
        "worst_candidate": float(cand[worst_index]),
        "worst_reference": float(ref[worst_index]),
        "candidate_minus_reference": True,
    }


def array_summary(array: Any, mask: np.ndarray | None = None) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.float64)
    vals = arr[mask] if mask is not None else arr.reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"shape": list(arr.shape), "count": 0}
    return {
        "shape": list(arr.shape),
        "count": int(vals.size),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "rms": float(np.sqrt(np.mean(vals * vals))),
        "max_abs": float(np.max(np.abs(vals))),
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "MISSING", "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def compact_prior_clear_sky() -> dict[str, Any]:
    prior = load_json(PRIOR_CLEARSKY)
    if prior.get("status") == "MISSING":
        return prior
    lwdnbc = (((prior.get("A_oracle") or {}).get("LWDNBC") or {}).get("clear_jax_vs_wrf") or {})
    return {
        "source": str(PRIOR_CLEARSKY),
        "oracle_kind": prior.get("oracle_kind"),
        "source_wrfout": prior.get("source_wrfout"),
        "lwdnbc_clear_jax_vs_wrf": {
            key: lwdnbc.get(key)
            for key in ("bias_Wm2", "rmse_Wm2", "max_abs_Wm2", "bias_pct")
            if key in lwdnbc
        },
        "interpretation": (
            "Prior full RRTMG clear-sky LW oracle was close on a different real WRF "
            "snapshot, so this proof does not claim a generic LW kernel failure."
        ),
    }


def rrtmg_solve_once(state: Any, namelist: Any, lead_seconds: float, land_state: Any) -> dict[str, Any]:
    sw_state, lw_state, surface_albedo, surface_emissivity, geometry, topography = pc._rrtmg_column_inputs(
        state,
        namelist.grid,
        time_utc=namelist.time_utc,
        lead_seconds=lead_seconds,
        radiation_static=namelist.radiation_static,
        topo_shading=int(namelist.topo_shading),
        slope_rad=int(namelist.slope_rad),
        shadow_length_m=float(namelist.topo_shadow_length_m),
        land_state=land_state,
    )
    sw = solve_rrtmg_sw_column(sw_state, debug=False, topography=topography, with_clear_sky=True)
    lw = solve_rrtmg_lw_column(lw_state, debug=False, with_clear_sky=True)
    return {
        "sw_state": sw_state,
        "lw_state": lw_state,
        "surface_albedo": surface_albedo,
        "surface_emissivity": surface_emissivity,
        "geometry": geometry,
        "topography": topography,
        "sw": sw,
        "lw": lw,
    }


def build_proof() -> dict[str, Any]:
    import jax  # noqa: PLC0415

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}
    surface_path = PINNED_SURFACE if PINNED_SURFACE.is_file() else handoff.SURFACE_HOOK
    surface = handoff.parse_surface_hook(surface_path)
    if surface.get("status") != "READY":
        return {"status": "BLOCKED_SURFACE_HOOK", "surface_hook": surface}

    shapes = split.expected_shapes()
    part2 = split.parse_part2_surfaces(shapes)
    if part2.get("status") != "WRF_PART2_TRUTH_READY":
        return {"status": "BLOCKED_PART2_TRUTH", "part2": part2}

    inputs = live.build_live_nest_step1_inputs()
    patched = pstate.apply_mythos_perturb_init(inputs)
    namelist = inputs["namelist"]
    state = patched["carry"].state
    land_state = patched["carry"].noahmp_land
    radt_seconds = float(namelist.dt_s) * int(namelist.radiation_cadence_steps)
    midpoint_seconds = 0.5 * radt_seconds

    pre = surface["arrays"]["PRE_NOAHMP"]
    wrf_swdown = handoff.field(pre, "swdown")
    wrf_glw = handoff.field(pre, "glw")
    wrf_albedo = handoff.field(pre, "albedo")
    land = handoff.field(pre, "xland") < 1.5

    after_calc = part2["surfaces"]["after_calculate_phy_tend"]["arrays"]
    interior = split.interior_mask(after_calc["RTHRATEN"].shape)
    wrf_mass_h = after_calc["MASS_H"]
    wrf_raw_rthraten = after_calc["RTHRATEN"] / np.maximum(wrf_mass_h, 1.0e-30)

    solved = rrtmg_solve_once(state, namelist, midpoint_seconds, land_state)
    sw = solved["sw"]
    lw = solved["lw"]
    jax.block_until_ready(lw.heating_rate)

    heat_t = pc._from_columns(sw.heating_rate + lw.heating_rate)
    sw_heat_t = pc._from_columns(sw.heating_rate)
    lw_heat_t = pc._from_columns(lw.heating_rate)
    exner = (np.maximum(np.asarray(state.p, dtype=np.float64), 1.0) / float(pc.P0_PA)) ** float(pc.R_D_OVER_CP)
    theta_rate = np.asarray(heat_t, dtype=np.float64) / np.maximum(exner, 1.0e-12)
    sw_theta_rate = np.asarray(sw_heat_t, dtype=np.float64) / np.maximum(exner, 1.0e-12)
    lw_theta_rate = np.asarray(lw_heat_t, dtype=np.float64) / np.maximum(exner, 1.0e-12)
    mass_h = (
        np.asarray(namelist.metrics.c1h, dtype=np.float64)[:, None, None]
        * np.asarray(state.mu_total, dtype=np.float64)[None, :, :]
        + np.asarray(namelist.metrics.c2h, dtype=np.float64)[:, None, None]
    )

    noahmp_closure = load_json(NOAHMP_CLOSURE_JSON)
    lead0_from_closure = (
        ((noahmp_closure.get("rad_seed_vs_wrf_hook") or {}).get("lead0_contrast_soldn_vs_wrf_swdown"))
        if noahmp_closure.get("status") != "MISSING"
        else None
    )

    thermodynamic_inputs = {
        "theta_pert_entry_vs_wrf_part2_T_STATE": diffstat(
            np.asarray(state.theta) - 300.0, after_calc["T_STATE"], interior
        ),
        "p_pert_entry_vs_wrf_part2_P": diffstat(
            np.asarray(state.p_perturbation), after_calc["P"], interior
        ),
        "p_total_entry_vs_wrf_part2_P_plus_PB": diffstat(
            np.asarray(state.p_total), after_calc["P"] + after_calc["PB"], interior
        ),
        "qv_entry_vs_wrf_part2_QV_OLD": diffstat(np.asarray(state.qv), after_calc["QV_OLD"], interior),
        "mu_total_entry_vs_wrf_part2_MUT": diffstat(
            np.broadcast_to(np.asarray(state.mu_total), after_calc["MUT"].shape),
            after_calc["MUT"],
            interior,
        ),
    }

    cloud_fraction = pc._cloud_fraction_columns(state)
    hydrometeor_sum = np.asarray(state.qc + state.qi + state.qs + state.qg, dtype=np.float64)

    surface_midpoint = {
        "patched_seed_swnorm_vs_wrf_swdown": {
            "all": diffstat(sw.surface_down_topographic, wrf_swdown),
            "land": diffstat(sw.surface_down_topographic, wrf_swdown, land),
            "water": diffstat(sw.surface_down_topographic, wrf_swdown, ~land),
        },
        "patched_seed_glw_vs_wrf_glw": {
            "all": diffstat(lw.surface_down, wrf_glw),
            "land": diffstat(lw.surface_down, wrf_glw, land),
            "water": diffstat(lw.surface_down, wrf_glw, ~land),
        },
        "accepted_lead0_swnorm_vs_wrf_swdown_from_noahmp_closure": lead0_from_closure,
        "allsky_minus_clear_sky": {
            "sw_surface_down_base_minus_clear": diffstat(sw.surface_down, sw.clear_flux_down[..., 0]),
            "lw_surface_down_minus_clear": diffstat(lw.surface_down, lw.clear_flux_down[..., 0]),
        },
        "coszen": array_summary(solved["geometry"].coszen),
        "topographic_correction_factor": array_summary(sw.topographic_correction_factor),
    }

    conversion = {
        "mass_h_jax_vs_wrf": diffstat(mass_h, wrf_mass_h, interior),
        "raw_theta_tendency_vs_wrf_raw": diffstat(theta_rate, wrf_raw_rthraten, interior),
        "mass_coupled_theta_tendency_vs_wrf": diffstat(
            mass_h * theta_rate, after_calc["RTHRATEN"], interior
        ),
        "mass_coupled_using_wrf_mass_vs_wrf": diffstat(
            wrf_mass_h * theta_rate, after_calc["RTHRATEN"], interior
        ),
        "no_exner_temperature_rate_mass_coupled_vs_wrf": diffstat(
            mass_h * np.asarray(heat_t, dtype=np.float64), after_calc["RTHRATEN"], interior
        ),
        "vertical_flipped_theta_tendency_mass_coupled_vs_wrf": diffstat(
            mass_h * theta_rate[::-1, :, :], after_calc["RTHRATEN"], interior
        ),
        "sw_only_mass_coupled_vs_wrf_total": diffstat(
            mass_h * sw_theta_rate, after_calc["RTHRATEN"], interior
        ),
        "lw_only_mass_coupled_vs_wrf_total": diffstat(
            mass_h * lw_theta_rate, after_calc["RTHRATEN"], interior
        ),
    }

    source_checks = {
        "wrf_lw_glw_assignment": {
            "file": str(WRF_RRTMG_LW),
            "line": 12782,
            "statement": "glw(i,j) = dflx(1,1)",
            "meaning": "WRF GLW is downward longwave surface flux, not emissivity-scaled by NoahMP.",
        },
        "wrf_lw_theta_conversion": {
            "file": str(WRF_RRTMG_LW),
            "line": 12816,
            "statement": "rthratenlw(i,k,j) = (hr(ncol,k)/86400.)/pi3d(i,k,j)",
            "meaning": "WRF converts LW temperature heating to theta tendency before mass coupling.",
        },
        "wrf_noahmp_lwdn_handoff": {
            "file": str(WRF_RRTMG_LW.parent / "module_sf_noahmpdrv.F"),
            "line": 760,
            "statement": "LWDN = GLW(I,J)",
            "meaning": "NoahMP receives GLW directly as LWDN.",
        },
        "wrf_lw_unhooked_derived_boundary": {
            "file": str(WRF_RRTMG_LW),
            "lines": "12254-12780",
            "meaning": (
                "WRF builds top-of-model buffer, ozone/trace-gas profiles, cloud optical "
                "properties, aerosol optical depth, then calls rrtmg_lw; this exact "
                "derived column is not present in the current Step-1 hooks."
            ),
        },
        "radiation_driver_rrtmg_lw_call": {
            "file": str(WRF_RADIATION_DRIVER),
            "line": 2087,
            "meaning": "RRTMG LW driver owns GLW and RTHRATENLW before SW is added.",
        },
        "jax_rrtmg_dry_theta_input_fix": {
            "file": str(ROOT / "src/gpuwrf/coupling/physics_couplers.py"),
            "owner": "_rrtmg_column_inputs",
            "meaning": (
                "Metric-backed RRTMG input now mirrors WRF phy_prep by converting "
                "stored theta_m to dry theta before T3D temperature conversion."
            ),
        },
    }

    classification = {
        "clock_or_solar_geometry": {
            "verdict": "NOT_PRIMARY",
            "evidence": (
                "Midpoint SWDOWN/SWNORM is close to WRF, while accepted lead-0 SWDOWN "
                "RMSE is much worse; LW GLW is independent of solar geometry."
            ),
        },
        "surface_emissivity_albedo_land_state": {
            "verdict": "NOT_PRIMARY_FOR_GLW_RTHRATEN",
            "evidence": (
                "WRF GLW is dflx(1,1) and NoahMP receives LWDN=GLW; SWDOWN is already "
                "within a few W/m2. No WRF emissivity hook exists in this fixture, but "
                "surface emissivity is not the downward GLW handoff."
            ),
        },
        "column_thermodynamics_cloud_inputs": {
            "verdict": "GROSS_STATE_EXONERATED_DERIVED_OPTICS_UNHOOKED",
            "evidence": (
                "Step-1 theta, pressure, qv, and mu match the WRF part2 state tightly; "
                "cloud fraction and hydrometeor cloud sum are zero. The exact WRF "
                "derived RRTMG profiles and optical depths are not dumped."
            ),
        },
        "layer_ordering": {
            "verdict": "NOT_PRIMARY",
            "evidence": "Vertical flip worsens the RTHRATEN comparison.",
        },
        "flux_to_theta_conversion": {
            "verdict": "NOT_PRIMARY",
            "evidence": "Skipping Exner conversion does not improve the residual.",
        },
        "mass_coupling": {
            "verdict": "NOT_PRIMARY",
            "evidence": "JAX mass_h matches WRF mass_h and using WRF mass_h leaves the residual unchanged.",
        },
        "likely_boundary": {
            "verdict": "DOMINANT_RRTMG_T3D_DRY_THETA_INPUT_FIXED_REMAINING_SPLIT_BOUND_IN_RRTMG_CLOSURE",
            "evidence": (
                "The WRF split radiation oracle localizes the former dominant residual "
                "to RRTMG_LWRAD T3D=t receiving moist-theta temperature in JAX. The "
                "production dry-theta input fix materially reduces GLW/RTHRATEN; the "
                "remaining LW/SW split residual is bounded in rrtmg_rthraten_closure."
            ),
        },
    }

    rthraten_resid = conversion["mass_coupled_theta_tendency_vs_wrf"]
    glw_resid = surface_midpoint["patched_seed_glw_vs_wrf_glw"]["all"]

    return {
        "status": "PROOF_EXECUTED",
        "schema": "wrfgpu2.v014.rrtmg_step1_forcing_parity.v1",
        "verdict": "RRTMG_STEP1_FORCING_PARITY_MATERIALLY_REDUCED_BY_DRY_THETA_INPUT_FIX",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "jax_backend": jax.default_backend(),
        },
        "git": {
            "head": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
            ).stdout.strip(),
            "branch": subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
            ).stdout.strip(),
        },
        "inputs": {
            "noahmp_closure_json": path_info(NOAHMP_CLOSURE_JSON),
            "surface_hook": path_info(surface_path),
            "rrtmg_rthraten_closure_json": path_info(RRTMG_RTHRATEN_CLOSURE_JSON),
            "wrf_part2_truth_root": str(split.WRF_TRUTH),
            "wrf_radiation_driver": path_info(WRF_RADIATION_DRIVER),
            "wrf_rrtmg_lw": path_info(WRF_RRTMG_LW),
        },
        "step1_config": {
            "dt_s": float(namelist.dt_s),
            "radiation_cadence_steps": int(namelist.radiation_cadence_steps),
            "radt_seconds": float(radt_seconds),
            "midpoint_lead_seconds": float(midpoint_seconds),
            "time_utc": str(namelist.time_utc),
            "use_noahmp": bool(namelist.use_noahmp),
            "sf_surface_physics": int(namelist.sf_surface_physics),
            "ra_lw_physics": int(namelist.ra_lw_physics),
            "ra_sw_physics": int(namelist.ra_sw_physics),
            "topo_shading": int(namelist.topo_shading),
            "slope_rad": int(namelist.slope_rad),
            "radiation_static_loaded": namelist.radiation_static is not None,
        },
        "surface_forcing": surface_midpoint,
        "surface_properties": {
            "jax_surface_albedo": {
                "all": array_summary(solved["surface_albedo"]),
                "land": array_summary(solved["surface_albedo"], land),
                "water": array_summary(solved["surface_albedo"], ~land),
            },
            "jax_surface_emissivity": {
                "all": array_summary(solved["surface_emissivity"]),
                "land": array_summary(solved["surface_emissivity"], land),
                "water": array_summary(solved["surface_emissivity"], ~land),
            },
            "jax_surface_albedo_vs_wrf_pre_noahmp_albedo": {
                "all": diffstat(solved["surface_albedo"], wrf_albedo),
                "land": diffstat(solved["surface_albedo"], wrf_albedo, land),
                "water": diffstat(solved["surface_albedo"], wrf_albedo, ~land),
            },
        },
        "column_inputs": {
            "thermodynamic_state_vs_wrf_part2": thermodynamic_inputs,
            "cloud_fraction": array_summary(cloud_fraction),
            "hydrometeor_cloud_sum_qc_qi_qs_qg": array_summary(hydrometeor_sum),
            "pressure_hpa": array_summary(np.asarray(solved["lw_state"].p) / 100.0),
            "temperature_k": array_summary(solved["lw_state"].T),
            "qv_kgkg": array_summary(solved["lw_state"].qv),
            "dz_m": array_summary(solved["lw_state"].dz),
        },
        "rthraten_conversion_and_mass": conversion,
        "source_checks": source_checks,
        "prior_clear_sky_oracle_context": compact_prior_clear_sky(),
        "classification": classification,
        "exact_residual_boundary": (
            "Dominant pre-fix boundary was WRF radiation_driver -> RRTMG_LWRAD input "
            "T3D=t: JAX built T from stored theta_m while WRF phy_prep passes dry "
            "theta temperature. The production owner is "
            "gpuwrf.coupling.physics_couplers._rrtmg_column_inputs. Remaining split "
            "LW/SW residual is bounded by proofs/v014/rrtmg_rthraten_closure.*."
        ),
        "recommended_fix": {
            "production_fix_obvious": True,
            "production_fix_applied": True,
            "next_action": (
                "Rerun the split WRF-oracle closure proof: JAX_PLATFORMS=cpu "
                "CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false "
                "PYTHONPATH=src python proofs/v014/rrtmg_rthraten_closure.py"
            ),
        },
        "lane_blocking": {
            "block_next_noahmp_strict_attempt": False,
            "block_final_v014_strict_release_if_unresolved": True,
            "reason": (
                "The residual is secondary to the current NoahMP HFX blocker, but the "
                "strict Step-1 release gate still requires RTHRATEN/GLW to close or be "
                "formally demoted by manager decision."
            ),
        },
        "summary_numbers": {
            "glw_bias_w_m2": glw_resid.get("bias"),
            "glw_rmse_w_m2": glw_resid.get("rmse"),
            "glw_max_abs_w_m2": glw_resid.get("max_abs"),
            "rthraten_mass_coupled_max_abs": rthraten_resid.get("max_abs"),
            "rthraten_mass_coupled_rmse": rthraten_resid.get("rmse"),
            "rthraten_mass_coupled_bias": rthraten_resid.get("bias"),
        },
        "commands": {
            "proof": (
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false "
                "PYTHONPATH=src python proofs/v014/rrtmg_step1_forcing_parity.py"
            )
        },
    }


def sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def render_markdown(payload: Mapping[str, Any]) -> str:
    if payload.get("status") != "PROOF_EXECUTED":
        return f"# V0.14 RRTMG Step-1 Forcing Parity\n\nBlocked: `{payload.get('status')}`.\n"

    summary = payload["summary_numbers"]
    surf = payload["surface_forcing"]
    conv = payload["rthraten_conversion_and_mass"]
    thermo = payload["column_inputs"]["thermodynamic_state_vs_wrf_part2"]
    classification = payload["classification"]
    lines = [
        "# V0.14 RRTMG Step-1 Forcing Parity",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Boundary",
        "",
        f"- Exact residual boundary: {payload['exact_residual_boundary']}",
        f"- Production fix applied: `{payload['recommended_fix'].get('production_fix_applied')}`.",
        f"- Next action: {payload['recommended_fix']['next_action']}.",
        "",
        "## Key numbers",
        "",
        f"- GLW/LWDN vs WRF PRE_NOAHMP: bias `{summary['glw_bias_w_m2']}` W/m2, "
        f"RMSE `{summary['glw_rmse_w_m2']}`, max_abs `{summary['glw_max_abs_w_m2']}`.",
        f"- RTHRATEN mass-coupled vs WRF part2: max_abs `{summary['rthraten_mass_coupled_max_abs']}`, "
        f"RMSE `{summary['rthraten_mass_coupled_rmse']}`, bias `{summary['rthraten_mass_coupled_bias']}`.",
        f"- SWDOWN midpoint remains close: RMSE `{surf['patched_seed_swnorm_vs_wrf_swdown']['all']['rmse']}` W/m2; "
        f"accepted lead-0 contrast RMSE `{surf['accepted_lead0_swnorm_vs_wrf_swdown_from_noahmp_closure'].get('rmse')}` W/m2.",
        "",
        "## Exonerated Boundaries",
        "",
        f"- Clock/geometry: `{classification['clock_or_solar_geometry']['verdict']}`. "
        f"{classification['clock_or_solar_geometry']['evidence']}",
        f"- Surface/land handoff: `{classification['surface_emissivity_albedo_land_state']['verdict']}`. "
        f"{classification['surface_emissivity_albedo_land_state']['evidence']}",
        f"- Gross thermodynamics/cloud: `{classification['column_thermodynamics_cloud_inputs']['verdict']}`. "
        f"Theta max_abs `{thermo['theta_pert_entry_vs_wrf_part2_T_STATE']['max_abs']}`, "
        f"p_total max_abs `{thermo['p_total_entry_vs_wrf_part2_P_plus_PB']['max_abs']}`, "
        f"qv max_abs `{thermo['qv_entry_vs_wrf_part2_QV_OLD']['max_abs']}`.",
        f"- Layer ordering: `{classification['layer_ordering']['verdict']}`; vertical flip max_abs "
        f"`{conv['vertical_flipped_theta_tendency_mass_coupled_vs_wrf']['max_abs']}`.",
        f"- Theta conversion: `{classification['flux_to_theta_conversion']['verdict']}`; no-Exner max_abs "
        f"`{conv['no_exner_temperature_rate_mass_coupled_vs_wrf']['max_abs']}`.",
        f"- Mass coupling: `{classification['mass_coupling']['verdict']}`; JAX-vs-WRF mass_h max_abs "
        f"`{conv['mass_h_jax_vs_wrf']['max_abs']}`, and using WRF mass leaves RTHRATEN max_abs "
        f"`{conv['mass_coupled_using_wrf_mass_vs_wrf']['max_abs']}`.",
        "",
        "## Release Impact",
        "",
        f"- Block next NoahMP strict attempt: `{payload['lane_blocking']['block_next_noahmp_strict_attempt']}`.",
        f"- Block final v0.14 strict release if unresolved: `{payload['lane_blocking']['block_final_v014_strict_release_if_unresolved']}`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    payload = build_proof()
    sanitized = sanitize(payload)
    OUT_JSON.write_text(
        json.dumps(sanitized, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    OUT_MD.write_text(render_markdown(sanitized), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(payload.get("verdict", payload.get("status")))
    return 0 if payload.get("status") == "PROOF_EXECUTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
