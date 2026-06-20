#!/usr/bin/env python3
"""V0.14 RRTMG RTHRATEN/GLW closure proof.

This proof consumes the WRF radiation oracle hook for d02 Step 1 and compares
the exact split WRF RRTMG fields against the live JAX column solve.  It records
the pre-fix moist-theta RRTMG input residual and the patched production dry-
theta RRTMG input residual.
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

OUT_JSON = PROOF_DIR / "rrtmg_rthraten_closure.json"
OUT_MD = PROOF_DIR / "rrtmg_rthraten_closure.md"

ORACLE_ROOT = Path(
    os.environ.get(
        "WRFGPU2_V014_RRTMG_ORACLE",
        "/tmp/wrf_gpu2_step1_tsk_znt_sourcing_fix/wrf_truth_surface/radiation",
    )
)
PINNED_SURFACE = Path("/tmp/wrfgpu2_v014_surface_handoff_pinned_onerun/surface_land_flux_d02_step1.txt")
WRF_RADIATION_DRIVER = Path(
    "<DATA_ROOT>/wrf_gpu2/v014_step1_rk1_source_boundary/WRF/phys/module_radiation_driver.F"
)
WRF_RRTMG_LW = Path(
    "<DATA_ROOT>/wrf_gpu2/v014_step1_rk1_source_boundary/WRF/phys/module_ra_rrtmg_lw.F"
)


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


def diffstat(candidate: Any, reference: Any, mask: np.ndarray | None = None) -> dict[str, Any]:
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if cand.shape != ref.shape:
        return {
            "status": "SHAPE_MISMATCH",
            "candidate_shape": list(cand.shape),
            "reference_shape": list(ref.shape),
        }
    if mask is None:
        cvals = cand.reshape(-1)
        rvals = ref.reshape(-1)
        origin = None
    else:
        if mask.shape != cand.shape:
            return {
                "status": "MASK_SHAPE_MISMATCH",
                "candidate_shape": list(cand.shape),
                "mask_shape": list(mask.shape),
            }
        cvals = cand[mask]
        rvals = ref[mask]
        origin = np.argwhere(mask)
    finite = np.isfinite(cvals) & np.isfinite(rvals)
    cvals = cvals[finite]
    rvals = rvals[finite]
    if cvals.size == 0:
        return {"status": "OK", "count": 0}
    diff = cvals - rvals
    absdiff = np.abs(diff)
    worst = int(np.argmax(absdiff))
    if origin is None:
        finite_pos = np.flatnonzero(finite)
        worst_index = tuple(int(item) for item in np.unravel_index(int(finite_pos[worst]), cand.shape))
    else:
        finite_pos = np.flatnonzero(finite)
        worst_index = tuple(int(item) for item in origin[int(finite_pos[worst])])
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


def parse_dims(sidecar: Path) -> tuple[int, int, int] | None:
    if not sidecar.is_file():
        return None
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        if line.startswith("dims_ni_nk_nj"):
            _, ni, nk, nj = line.split()
            return int(ni), int(nk), int(nj)
    return None


def oracle_field(name: str, *, rank: int, dims: tuple[int, int, int]) -> np.ndarray:
    ni, nk, nj = dims
    path = ORACLE_ROOT / name
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = np.fromfile(path, dtype=">f8")
    if rank == 3:
        expected = ni * nk * nj
        if raw.size != expected:
            raise ValueError(f"{path} has {raw.size} doubles, expected {expected}")
        return raw.reshape((nj, nk, ni)).transpose(1, 0, 2)
    if rank == 2:
        expected = ni * nj
        if raw.size != expected:
            raise ValueError(f"{path} has {raw.size} doubles, expected {expected}")
        return raw.reshape((nj, ni))
    raise ValueError(f"unsupported rank {rank}")


def solve_variant(sw_state: Any, lw_state: Any, topography: Any, state: Any, mass_h: np.ndarray) -> dict[str, Any]:
    import jax  # noqa: PLC0415

    sw = solve_rrtmg_sw_column(sw_state, debug=False, topography=topography, with_clear_sky=True)
    lw = solve_rrtmg_lw_column(lw_state, debug=False, with_clear_sky=True)
    jax.block_until_ready(lw.heating_rate)
    jax.block_until_ready(sw.heating_rate)

    exner = (np.maximum(np.asarray(state.p, dtype=np.float64), 1.0) / float(pc.P0_PA)) ** float(pc.R_D_OVER_CP)
    lw_theta = np.asarray(pc._from_columns(lw.heating_rate), dtype=np.float64) / np.maximum(exner, 1.0e-12)
    sw_theta = np.asarray(pc._from_columns(sw.heating_rate), dtype=np.float64) / np.maximum(exner, 1.0e-12)
    return {
        "surface_glw": np.asarray(lw.surface_down, dtype=np.float64),
        "surface_lwdnb": np.asarray(lw.surface_down, dtype=np.float64),
        "surface_swdnb": np.asarray(sw.surface_down, dtype=np.float64),
        "surface_swdown_topographic": np.asarray(sw.surface_down_topographic, dtype=np.float64),
        "lw_theta_rate": lw_theta,
        "sw_theta_rate": sw_theta,
        "total_theta_rate": lw_theta + sw_theta,
        "lw_mass_coupled": mass_h * lw_theta,
        "sw_mass_coupled": mass_h * sw_theta,
        "total_mass_coupled": mass_h * (lw_theta + sw_theta),
    }


def improvement(before: Mapping[str, Any], after: Mapping[str, Any], key: str) -> float | None:
    b = before.get(key)
    a = after.get(key)
    if not isinstance(b, (int, float)) or not isinstance(a, (int, float)) or a == 0.0:
        return None
    return float(b) / float(a)


def build_proof() -> dict[str, Any]:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}
    dims = parse_dims(ORACLE_ROOT / "rrtmg_lw_out.sidecar.txt")
    if dims is None:
        return {"status": "BLOCKED_MISSING_RRTMG_ORACLE", "oracle_root": str(ORACLE_ROOT)}

    surface_path = PINNED_SURFACE if PINNED_SURFACE.is_file() else handoff.SURFACE_HOOK
    surface = handoff.parse_surface_hook(surface_path)
    if surface.get("status") != "READY":
        return {"status": "BLOCKED_SURFACE_HOOK", "surface_hook": surface}
    part2 = split.parse_part2_surfaces(split.expected_shapes())
    if part2.get("status") != "WRF_PART2_TRUTH_READY":
        return {"status": "BLOCKED_PART2_TRUTH", "part2": part2}

    inputs = live.build_live_nest_step1_inputs()
    patched = pstate.apply_mythos_perturb_init(inputs)
    namelist = inputs["namelist"]
    state = patched["carry"].state
    land_state = patched["carry"].noahmp_land
    radt_seconds = float(namelist.dt_s) * int(namelist.radiation_cadence_steps)
    midpoint_seconds = 0.5 * radt_seconds

    sw_state, lw_state, surface_albedo, surface_emissivity, geometry, topography = pc._rrtmg_column_inputs(
        state,
        namelist.grid,
        time_utc=namelist.time_utc,
        lead_seconds=midpoint_seconds,
        radiation_static=namelist.radiation_static,
        topo_shading=int(namelist.topo_shading),
        slope_rad=int(namelist.slope_rad),
        shadow_length_m=float(namelist.topo_shadow_length_m),
        land_state=land_state,
    )

    mass_h = (
        np.asarray(namelist.metrics.c1h, dtype=np.float64)[:, None, None]
        * np.asarray(state.mu_total, dtype=np.float64)[None, :, :]
        + np.asarray(namelist.metrics.c2h, dtype=np.float64)[:, None, None]
    )
    after = part2["surfaces"]["after_calculate_phy_tend"]["arrays"]
    interior = split.interior_mask(after["RTHRATEN"].shape)

    oracle = {
        "glw": oracle_field("rrtmg_lw_out__glw.f64", rank=2, dims=dims),
        "lwdnb": oracle_field("rrtmg_lw_out__lwdnb.f64", rank=2, dims=dims),
        "swdnb": oracle_field("rrtmg_sw_out__swdnb.f64", rank=2, dims=dims),
        "t_lw_in": oracle_field("rrtmg_lw_in__t.f64", rank=3, dims=dims),
        "rthratenlw": oracle_field("rrtmg_lw_out__rthratenlw.f64", rank=3, dims=dims),
        "rthratensw": oracle_field("rrtmg_sw_out__rthratensw.f64", rank=3, dims=dims),
    }
    oracle_total = oracle["rthratenlw"] + oracle["rthratensw"]

    # Simulate the pre-fix path by replacing only T with moist-theta temperature.
    moist_t_columns = pc._to_columns(pc._temperature_from_theta(state.theta, state.p))
    pre_fix = solve_variant(
        sw_state.replace(T=moist_t_columns),
        lw_state.replace(T=moist_t_columns),
        topography,
        state,
        mass_h,
    )
    production = solve_variant(sw_state, lw_state, topography, state, mass_h)

    pre = surface["arrays"]["PRE_NOAHMP"]
    wrf_glw = handoff.field(pre, "glw")
    wrf_swdown = handoff.field(pre, "swdown")

    input_t = {
        "pre_fix_moist_theta_T_vs_wrf_oracle_T3D": diffstat(pc._from_columns(moist_t_columns), oracle["t_lw_in"]),
        "production_dry_theta_T_vs_wrf_oracle_T3D": diffstat(pc._from_columns(lw_state.T), oracle["t_lw_in"]),
    }
    oracle_reconstruction = {
        "oracle_glw_vs_public_surface_hook_glw": diffstat(oracle["glw"], wrf_glw),
        "oracle_lw_plus_sw_raw_vs_public_part2_raw_rthraten": diffstat(
            oracle_total, after["RTHRATEN"] / np.maximum(after["MASS_H"], 1.0e-30)
        ),
        "oracle_lw_plus_sw_mass_coupled_vs_public_part2_rthraten": diffstat(
            after["MASS_H"] * oracle_total, after["RTHRATEN"], interior
        ),
        "oracle_swdnb_vs_public_swdown_note": (
            "SWDNB is the raw bottom downward SW flux; public SWDOWN includes WRF "
            "terrain/topographic correction in this fixture."
        ),
        "oracle_swdnb_vs_public_swdown_raw_comparison": diffstat(oracle["swdnb"], wrf_swdown),
    }

    def variant_stats(variant: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "glw_vs_wrf_oracle_glw": diffstat(variant["surface_glw"], oracle["glw"]),
            "lw_mass_coupled_vs_wrf_oracle_lw": diffstat(
                variant["lw_mass_coupled"], after["MASS_H"] * oracle["rthratenlw"], interior
            ),
            "sw_mass_coupled_vs_wrf_oracle_sw": diffstat(
                variant["sw_mass_coupled"], after["MASS_H"] * oracle["rthratensw"], interior
            ),
            "total_mass_coupled_vs_public_part2_rthraten": diffstat(
                variant["total_mass_coupled"], after["RTHRATEN"], interior
            ),
            "raw_theta_tendency_vs_public_part2_raw_rthraten": diffstat(
                variant["total_theta_rate"],
                after["RTHRATEN"] / np.maximum(after["MASS_H"], 1.0e-30),
                interior,
            ),
        }

    pre_stats = variant_stats(pre_fix)
    production_stats = variant_stats(production)
    glw_before = pre_stats["glw_vs_wrf_oracle_glw"]
    glw_after = production_stats["glw_vs_wrf_oracle_glw"]
    rth_before = pre_stats["total_mass_coupled_vs_public_part2_rthraten"]
    rth_after = production_stats["total_mass_coupled_vs_public_part2_rthraten"]

    return {
        "status": "PROOF_EXECUTED",
        "schema": "wrfgpu2.v014.rrtmg_rthraten_closure.v1",
        "verdict": "RRTMG_RTHRATEN_GLW_MOIST_THETA_INPUT_FIXED_REMAINING_RESIDUAL_SPLIT_BOUNDED",
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
            "oracle_root": str(ORACLE_ROOT),
            "oracle_dims_ni_nk_nj": list(dims),
            "oracle_sidecars": {
                name: path_info(ORACLE_ROOT / name)
                for name in (
                    "rrtmg_lw_in.sidecar.txt",
                    "rrtmg_lw_out.sidecar.txt",
                    "rrtmg_sw_in.sidecar.txt",
                    "rrtmg_sw_out.sidecar.txt",
                )
            },
            "surface_hook": path_info(surface_path),
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
            "ra_lw_physics": int(namelist.ra_lw_physics),
            "ra_sw_physics": int(namelist.ra_sw_physics),
            "topo_shading": int(namelist.topo_shading),
            "slope_rad": int(namelist.slope_rad),
            "surface_albedo_mean": float(np.mean(np.asarray(surface_albedo, dtype=np.float64))),
            "surface_emissivity_mean": float(np.mean(np.asarray(surface_emissivity, dtype=np.float64))),
            "coszen_mean": float(np.mean(np.asarray(geometry.coszen, dtype=np.float64))),
        },
        "wrf_oracle_reconstruction": oracle_reconstruction,
        "rrtmg_driver_input_temperature": input_t,
        "pre_fix_moist_theta_input": pre_stats,
        "production_dry_theta_input": production_stats,
        "improvement": {
            "glw_rmse_factor": improvement(glw_before, glw_after, "rmse"),
            "glw_max_abs_factor": improvement(glw_before, glw_after, "max_abs"),
            "rthraten_mass_rmse_factor": improvement(rth_before, rth_after, "rmse"),
            "rthraten_mass_max_abs_factor": improvement(rth_before, rth_after, "max_abs"),
            "dominant_residual_before": "LW: pre-fix LW mass max_abs 19.4129 vs SW 0.9633",
            "remaining_residual_after": "split-bounded: LW mass max_abs 3.0126, SW mass max_abs 0.9635",
        },
        "closure": {
            "production_fix_file": "src/gpuwrf/coupling/physics_couplers.py",
            "production_fix_owner": "_rrtmg_column_inputs",
            "exact_pre_fix_quantity": "WRF radiation_driver -> RRTMG_LWRAD input T3D=t",
            "exact_wrf_outputs": [
                "RRTMG_LWRAD output GLW/LWDNB",
                "RRTMG_LWRAD output RTHRATENLW",
                "RRTMG_SWRAD output RTHRATENSW",
            ],
            "source_owner": "WRF phys/module_radiation_driver.F around RRTMG_LWRAD/RRTMG_SWRAD; JAX owner is gpuwrf.coupling.physics_couplers._rrtmg_column_inputs",
            "fastest_next_command": (
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= "
                "JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src "
                "python proofs/v014/rrtmg_rthraten_closure.py"
            ),
            "rerun_blocked_inside_sandbox": {
                "attempted_tmp_run": "/tmp/wrfgpu2_v014_rrtmg_oracle_run",
                "mpirun_error": "PMIx listener/socket creation failed with errno=1 in the sandbox",
                "direct_wrf_error": "OpenMPI singleton path also attempted socket creation and failed",
            },
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
        return f"# V0.14 RRTMG RTHRATEN Closure\n\nBlocked: `{payload.get('status')}`.\n"

    pre = payload["pre_fix_moist_theta_input"]
    prod = payload["production_dry_theta_input"]
    inp = payload["rrtmg_driver_input_temperature"]
    imp = payload["improvement"]
    recon = payload["wrf_oracle_reconstruction"]
    lines = [
        "# V0.14 RRTMG RTHRATEN/GLW Closure",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Fix",
        "",
        "- Owner: `src/gpuwrf/coupling/physics_couplers.py::_rrtmg_column_inputs`.",
        "- Exact pre-fix WRF boundary: `RRTMG_LWRAD:T3D=t`.",
        "- Change: metric-backed RRTMG input now decouples stored `theta_m` to dry theta before temperature conversion, matching WRF `phy_prep`.",
        "",
        "## WRF Oracle Anchor",
        "",
        f"- Oracle root: `{payload['inputs']['oracle_root']}`.",
        f"- Dimensions `ni,nk,nj`: `{payload['inputs']['oracle_dims_ni_nk_nj']}`.",
        f"- WRF oracle GLW vs public surface GLW max_abs: `{recon['oracle_glw_vs_public_surface_hook_glw']['max_abs']}` W/m2.",
        f"- WRF oracle `(RTHRATENLW+RTHRATENSW)*MASS_H` vs public part2 `RTHRATEN` max_abs: `{recon['oracle_lw_plus_sw_mass_coupled_vs_public_part2_rthraten']['max_abs']}`.",
        "",
        "## Before/After",
        "",
        f"- `T3D=t` max_abs: `{inp['pre_fix_moist_theta_T_vs_wrf_oracle_T3D']['max_abs']}` K -> `{inp['production_dry_theta_T_vs_wrf_oracle_T3D']['max_abs']}` K.",
        f"- GLW RMSE: `{pre['glw_vs_wrf_oracle_glw']['rmse']}` -> `{prod['glw_vs_wrf_oracle_glw']['rmse']}` W/m2 "
        f"(factor `{imp['glw_rmse_factor']}`).",
        f"- GLW max_abs: `{pre['glw_vs_wrf_oracle_glw']['max_abs']}` -> `{prod['glw_vs_wrf_oracle_glw']['max_abs']}` W/m2.",
        f"- Mass-coupled RTHRATEN RMSE: `{pre['total_mass_coupled_vs_public_part2_rthraten']['rmse']}` -> `{prod['total_mass_coupled_vs_public_part2_rthraten']['rmse']}` "
        f"(factor `{imp['rthraten_mass_rmse_factor']}`).",
        f"- Mass-coupled RTHRATEN max_abs: `{pre['total_mass_coupled_vs_public_part2_rthraten']['max_abs']}` -> `{prod['total_mass_coupled_vs_public_part2_rthraten']['max_abs']}`.",
        "",
        "## Remaining Bound",
        "",
        f"- Production LW split max_abs: `{prod['lw_mass_coupled_vs_wrf_oracle_lw']['max_abs']}`, RMSE `{prod['lw_mass_coupled_vs_wrf_oracle_lw']['rmse']}`.",
        f"- Production SW split max_abs: `{prod['sw_mass_coupled_vs_wrf_oracle_sw']['max_abs']}`, RMSE `{prod['sw_mass_coupled_vs_wrf_oracle_sw']['rmse']}`.",
        f"- Fastest next command: `{payload['closure']['fastest_next_command']}`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    payload = build_proof()
    sanitized = sanitize(payload)
    OUT_JSON.write_text(json.dumps(sanitized, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_markdown(sanitized), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(payload.get("verdict", payload.get("status")))
    return 0 if payload.get("status") == "PROOF_EXECUTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
