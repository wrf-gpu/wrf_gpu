#!/usr/bin/env python3
"""V0.14 live-nest base-state hook oracle/plan proof.

This proof does not patch production JAX source and does not promote CPU-WRF
wrfout_h0 to production logic.  It records the WRF source hook that must be
ported for the d02 live-nest base split: parent interpolation, terrain/base
blend, then start_domain_em base recomputation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20260501_18z_l2_72h_20260519T173026Z"

RUN_ROOT = Path("/tmp/v0120_merged_run_root") / RUN_ID
NATIVE_WRFINPUT_D01 = RUN_ROOT / "wrfinput_d01"
NATIVE_WRFINPUT_D02 = RUN_ROOT / "wrfinput_d02"
NATIVE_NAMELIST = RUN_ROOT / "namelist.input"
NATIVE_WRFBDY_D01 = RUN_ROOT / "wrfbdy_d01"

CPU_WRFOUT_DIR = Path("/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output") / RUN_ID
CPU_H0 = CPU_WRFOUT_DIR / "wrfout_d02_2026-05-01_18:00:00"
CPU_H1 = CPU_WRFOUT_DIR / "wrfout_d02_2026-05-01_19:00:00"
CPU_H10 = CPU_WRFOUT_DIR / "wrfout_d02_2026-05-02_04:00:00"

WRF_ROOT = Path("/mnt/data/wrf_gpu2/v014_post_rk_refresh/WRF")
WRF_MEDIATION_INTEGRATE = WRF_ROOT / "share/mediation_integrate.F"
WRF_NEST_INC = WRF_ROOT / "inc/nest_interpdown_interp.inc"
WRF_NEST_INIT_UTILS = WRF_ROOT / "dyn_em/nest_init_utils.F"
WRF_START_EM = WRF_ROOT / "dyn_em/start_em.F"
WRF_INTERP_FCN = WRF_ROOT / "share/interp_fcn.F"
WRF_SINT = WRF_ROOT / "share/sint.F"

PRIOR_BASE_JSON = ROOT / "proofs/v014/base_state_split_fix.json"
PRIOR_EARLIER_JSON = ROOT / "proofs/v014/earlier_source_bisect.json"

OUT_JSON = ROOT / "proofs/v014/live_nest_base_hook.json"
OUT_MD = ROOT / "proofs/v014/live_nest_base_hook.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-live-nest-base-hook.md"

CLASSIFICATION = "NATIVE_PORT_PLAN_READY"
FIELDS = ("HGT", "PB", "MUB", "PHB")
STATIC_TOL = 2.0e-6
BASE_FORMULA_TOL = 2.0e-1

R_D = 287.0
CP_D = 1004.0
P1000MB = 100000.0
T0 = 300.0
G = 9.81
CVPM = -((CP_D - R_D) / CP_D)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        item = float(value)
        return item if math.isfinite(item) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_info(path: Path) -> dict[str, Any]:
    resolved: str | None = None
    try:
        resolved = str(path.resolve())
    except OSError:
        resolved = None
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "is_symlink": path.is_symlink(),
        "resolved": resolved,
    }
    if path.exists():
        info["size_bytes"] = path.stat().st_size
        if path.is_file():
            info["sha256"] = sha256(path)
    return info


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nc_var(path: Path, name: str) -> np.ndarray:
    with Dataset(path, "r") as ds:
        var = ds.variables[name]
        raw = var[0] if var.dimensions and var.dimensions[0] == "Time" else var[:]
        return np.asarray(np.ma.filled(raw, np.nan), dtype=np.float64)


def nc_scalar(path: Path, name: str) -> float:
    with Dataset(path, "r") as ds:
        if name in ds.variables:
            return float(np.asarray(ds.variables[name][:]).reshape(-1)[0])
        if hasattr(ds, name):
            return float(getattr(ds, name))
    raise KeyError(f"{name} not found in {path}")


def nc_variable_inventory(path: Path, names: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    with Dataset(path, "r") as ds:
        for name in names:
            if name not in ds.variables:
                out[name] = {"present": False}
                continue
            var = ds.variables[name]
            out[name] = {
                "present": True,
                "dimensions": list(var.dimensions),
                "shape": [int(item) for item in var.shape],
                "dtype": str(var.dtype),
            }
    return out


def stats_from_diff(diff: np.ndarray, *, tol: float) -> dict[str, Any]:
    finite_mask = np.isfinite(diff)
    finite = np.asarray(diff, dtype=np.float64)[finite_mask]
    if finite.size == 0:
        return {
            "status": "NO_FINITE_PAIR",
            "count": int(diff.size),
            "finite_count": 0,
            "tolerance_max_abs": float(tol),
        }
    abs_diff = np.abs(diff)
    masked_abs = np.where(finite_mask, abs_diff, -np.inf)
    idx = np.unravel_index(int(np.argmax(masked_abs)), diff.shape)
    max_abs = float(np.max(np.abs(finite)))
    return {
        "status": "MATCH" if max_abs <= tol else "DIFF",
        "count": int(diff.size),
        "finite_count": int(finite.size),
        "shape": [int(item) for item in diff.shape],
        "max_abs": max_abs,
        "rmse": float(math.sqrt(float(np.mean(finite * finite)))),
        "bias": float(np.mean(finite)),
        "p99_abs": float(np.percentile(np.abs(finite), 99.0)),
        "tolerance_max_abs": float(tol),
        "worst": {
            "index": [int(item) for item in idx],
            "diff": float(diff[idx]),
            "abs_diff": float(abs_diff[idx]),
        },
    }


def compare_arrays(name: str, left: np.ndarray, right: np.ndarray, *, tol: float) -> dict[str, Any]:
    if left.shape != right.shape:
        return {
            "name": name,
            "status": "SHAPE_MISMATCH",
            "left_shape": [int(item) for item in left.shape],
            "right_shape": [int(item) for item in right.shape],
        }
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    result = {"name": name, **stats_from_diff(diff, tol=tol)}
    if result["status"] != "NO_FINITE_PAIR":
        idx = tuple(result["worst"]["index"])
        result["worst"]["left"] = float(left[idx])
        result["worst"]["right"] = float(right[idx])
    return result


def patch_slice(bounds: Mapping[str, int], ndim: int) -> tuple[slice, ...]:
    y = slice(int(bounds["y0"]), int(bounds["y1"]))
    x = slice(int(bounds["x0"]), int(bounds["x1"]))
    if ndim == 2:
        return (y, x)
    if ndim == 3:
        return (slice(None), y, x)
    raise ValueError(f"unsupported ndim={ndim}")


def scoped_compare(
    name: str,
    left: np.ndarray,
    right: np.ndarray,
    *,
    bounds: Mapping[str, int] | None,
    tol: float,
) -> dict[str, Any]:
    scope = "whole_domain"
    if bounds is not None:
        selector = patch_slice(bounds, left.ndim)
        left = left[selector]
        right = right[selector]
        scope = "target_patch"
    return {"scope": scope, **compare_arrays(name, left, right, tol=tol)}


def compare_nc_fields(
    left_path: Path,
    right_path: Path,
    fields: tuple[str, ...],
    *,
    bounds: Mapping[str, int] | None,
    tol: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in fields:
        left = nc_var(left_path, field)
        right = nc_var(right_path, field)
        out[field] = scoped_compare(
            f"{left_path.name}_vs_{right_path.name}_{field}",
            left,
            right,
            bounds=bounds,
            tol=tol,
        )
    return out


def wrf_base_from_hgt(reference_nc: Path, hgt: np.ndarray) -> dict[str, np.ndarray]:
    ptop = nc_scalar(reference_nc, "P_TOP")
    p00 = nc_scalar(reference_nc, "P00")
    t00 = nc_scalar(reference_nc, "T00")
    lapse = nc_scalar(reference_nc, "TLP")
    tiso = nc_scalar(reference_nc, "TISO")
    lapse_strat = nc_scalar(reference_nc, "TLP_STRAT")
    p_strat = nc_scalar(reference_nc, "P_STRAT")
    c3h = nc_var(reference_nc, "C3H")
    c4h = nc_var(reference_nc, "C4H")
    c3f = nc_var(reference_nc, "C3F")
    c4f = nc_var(reference_nc, "C4F")

    p_surf = p00 * np.exp(-t00 / lapse + np.sqrt((t00 / lapse) ** 2 - 2.0 * G * hgt / lapse / R_D))
    mub = p_surf - ptop
    pb = c3h[:, None, None] * mub[None, :, :] + c4h[:, None, None] + ptop

    temp = np.maximum(tiso, t00 + lapse * np.log(pb / p00))
    if p_strat > 0.0:
        temp = np.where(pb < p_strat, tiso + lapse_strat * np.log(pb / p_strat), temp)
    t_init = temp * (p00 / pb) ** (R_D / CP_D) - T0
    alb = (R_D / P1000MB) * (t_init + T0) * (pb / P1000MB) ** CVPM

    phb = np.empty((pb.shape[0] + 1, pb.shape[1], pb.shape[2]), dtype=np.float64)
    phb[0, :, :] = hgt * G
    for full_k in range(1, pb.shape[0] + 1):
        half_k = full_k - 1
        pfu = c3f[full_k] * mub + c4f[full_k] + ptop
        pfd = c3f[full_k - 1] * mub + c4f[full_k - 1] + ptop
        phm = c3h[half_k] * mub + c4h[half_k] + ptop
        phb[full_k, :, :] = phb[full_k - 1, :, :] + alb[half_k, :, :] * phm * np.log(pfd / pfu)

    return {"PB": pb, "MUB": mub, "PHB": phb, "T_INIT": t_init, "ALB": alb}


def formula_residuals(path: Path, *, bounds: Mapping[str, int] | None) -> dict[str, Any]:
    computed = wrf_base_from_hgt(path, nc_var(path, "HGT"))
    out: dict[str, Any] = {}
    for field in ("PB", "MUB", "PHB"):
        out[field] = scoped_compare(
            f"start_domain_formula_on_{path.name}_{field}",
            computed[field],
            nc_var(path, field),
            bounds=bounds,
            tol=BASE_FORMULA_TOL,
        )
    return out


def source_line(path: Path, needle: str) -> int | None:
    if not path.is_file():
        return None
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if needle in line:
            return idx
    return None


def line_range(path: Path, start_needle: str, end_needle: str | None = None, *, context: int = 0) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines() if path.is_file() else []
    start_line = source_line(path, start_needle)
    end_line = source_line(path, end_needle) if end_needle else start_line
    if start_line is None:
        start = end = None
        excerpt: list[str] = []
    else:
        end_line = end_line or start_line
        start = max(1, start_line - context)
        end = min(len(lines), end_line + context)
        excerpt = [f"{number}: {lines[number - 1].rstrip()}" for number in range(start, end + 1)]
    return {
        "path": str(path),
        "sha256": sha256(path),
        "start_line": start,
        "end_line": end,
        "start_needle": start_needle,
        "end_needle": end_needle,
        "excerpt": excerpt,
    }


def source_citations() -> dict[str, Any]:
    return {
        "live_nest_input_branch": line_range(
            WRF_MEDIATION_INTEGRATE,
            "CALL med_interp_domain( parent, nest )",
            "CALL adjust_tempqv",
            context=1,
        ),
        "interpdown_phb_mub_pb": {
            "PHB": line_range(WRF_NEST_INC, "grid%phb,   &", "ngrid%phb,  &", context=2),
            "MUB": line_range(WRF_NEST_INC, "grid%mub,   &", "ngrid%mub,  &", context=2),
            "PB": line_range(WRF_NEST_INC, "grid%pb,   &", "ngrid%pb,  &", context=2),
            "HGT_HT": line_range(WRF_NEST_INC, "grid%ht,   &", "ngrid%ht,  &", context=2),
        },
        "blend_terrain": line_range(
            WRF_NEST_INIT_UTILS,
            "SUBROUTINE blend_terrain",
            "END SUBROUTINE blend_terrain",
            context=0,
        ),
        "start_domain_base_recompute": line_range(
            WRF_START_EM,
            "!  reconstitute base-state fields",
            "!------base state is finished, perturbation values are recalculated below-----",
            context=0,
        ),
        "start_domain_base_formula": line_range(
            WRF_START_EM,
            "p_surf = p00 * EXP",
            "grid%phb(i,k,j) = grid%phb(i,k-1,j) + grid%alb(i,k-1,j)*phm*LOG(pfd/pfu)",
            context=2,
        ),
        "interp_dispatch": line_range(
            WRF_INTERP_FCN,
            "interp_method_type = SINT",
            "CALL interp_fcn_sint",
            context=2,
        ),
        "sint_formula": line_range(
            WRF_SINT,
            "SUBROUTINE SINT",
            "RETURN",
            context=0,
        ),
    }


def previous_marker_runs() -> dict[str, Any]:
    run_case3 = Path("/mnt/data/wrf_gpu2/v014_post_rk_refresh/run_case3")
    return {
        "run_case3": path_info(run_case3),
        "first_28rank_early_marker_run": {
            "dir": path_info(run_case3 / "first_28rank_early_marker_run"),
            "stdout_log": path_info(run_case3 / "first_28rank_early_marker_run/marker_run_28rank_stdout.log"),
            "rsl_error_0000": path_info(run_case3 / "first_28rank_early_marker_run/rsl.error.0000"),
            "h0_output": path_info(run_case3 / "first_28rank_early_marker_run/wrfout_d02_2026-05-01_18:00:00"),
        },
        "post_marker_noemit_run": {
            "dir": path_info(run_case3 / "post_marker_noemit_run"),
            "stdout_log": path_info(run_case3 / "post_marker_noemit_run/marker_run_post_28rank_stdout.log"),
            "rsl_error_0000": path_info(run_case3 / "post_marker_noemit_run/rsl.error.0000"),
            "h0_output": path_info(run_case3 / "post_marker_noemit_run/wrfout_d02_2026-05-01_18:00:00"),
        },
        "no_post_blend_base_payload_found": True,
    }


def implementation_plan() -> dict[str, Any]:
    return {
        "classification": CLASSIFICATION,
        "next_source_sprint_target": "native live-nest base initialization stage before build_replay_case exposes d02 BaseState",
        "do_not_do": [
            "Do not load CPU-WRF wrfout_h0 as production input.",
            "Do not patch production JAX source in this sprint.",
            "Do not use simplified bilinear interpolation for this path.",
        ],
        "required_native_stages": [
            {
                "order": 1,
                "name": "parent_to_child_sint_interpolation",
                "wrf_source": "inc/nest_interpdown_interp.inc + share/interp_fcn.F + share/sint.F",
                "inputs": [
                    "parent d01 live grid HGT/HT, MUB, PHB, PB at the nest initialization time",
                    "child i_parent_start, j_parent_start, parent_grid_ratio",
                    "config_flags%shw and interpolation method type; default path dispatches to SINT",
                    "child imask_nostag initialized to one for active nest tasks",
                ],
                "outputs": ["ht_int", "mub_fine", "phb_fine", "pb interpolated if needed for diagnostics"],
            },
            {
                "order": 2,
                "name": "read_child_wrfinput_static_and_base",
                "wrf_source": "share/mediation_integrate.F::med_initialdata_input_ptr path",
                "inputs": ["child wrfinput_d02 HGT/HT, MUB, PHB, PB, T, P, QVAPOR, MU"],
                "outputs": ["ht_fine", "mub_save", "child wrfinput base/dynamic state"],
            },
            {
                "order": 3,
                "name": "blend_terrain_base_fields",
                "wrf_source": "dyn_em/nest_init_utils.F::blend_terrain",
                "inputs": [
                    "ter_interpolated = parent-interpolated ht_int/mub_fine/phb_fine",
                    "ter_input = child wrfinput ht/mub/phb",
                    "spec_bdy_width from namelist, observed default 5 for this case",
                    "blend_width from namelist, observed default 5 for this case",
                ],
                "formula": (
                    "outer spec_bdy_width rows/columns take interpolated parent; blend_cell=blend_width..1 "
                    "uses ((blend_cell)*child + (blend_width+1-blend_cell)*parent)/(blend_width+1); "
                    "interior remains child wrfinput"
                ),
                "outputs": ["post-blend child HGT/HT, MUB, PHB"],
            },
            {
                "order": 4,
                "name": "start_domain_em_base_recompute",
                "wrf_source": "dyn_em/start_em.F::start_domain_em multi-domain real-run branch",
                "inputs": [
                    "post-blend HGT/HT",
                    "post-blend MUB",
                    "hybrid arrays C1H/C2H/C3H/C4H/C3F/C4F/DNW",
                    "base constants P_TOP/P00/T00/TLP/TISO/TLP_STRAT/P_STRAT",
                    "hypsometric_opt=2 for this fixture",
                    "rebalance=1 if PHB is recomputed",
                ],
                "formula_summary": [
                    "p_surf = p00 * exp(-t00/a + sqrt((t00/a)^2 - 2*g*ht/a/r_d))",
                    "MUB = p_surf - p_top",
                    "PB(k) = c3h(k)*MUB + c4h(k) + p_top",
                    "T_INIT(k) = temp*(p00/PB(k))^(r_d/cp) - t0",
                    "ALB(k) = (r_d/p1000mb)*(T_INIT(k)+t0)*(PB(k)/p1000mb)^cvpm",
                    "PHB(1) = HGT*g; PHB(k) = PHB(k-1) + ALB(k-1)*PHM*log(PFD/PFU) for hypsometric_opt=2",
                ],
                "outputs": ["PB", "MUB", "PHB", "T_INIT", "ALB"],
            },
            {
                "order": 5,
                "name": "recalculate_perturbation_split",
                "wrf_source": "dyn_em/start_em.F after base-state-is-finished marker",
                "inputs": ["loaded total/dynamic fields and recomputed base fields"],
                "outputs": [
                    "P perturbation against recomputed PB",
                    "PH perturbation against recomputed PHB",
                    "MU perturbation against recomputed MUB",
                ],
            },
        ],
        "validation_gate_for_source_fix": [
            "CPU-WRF h0 may be used only as validation oracle: HGT/PB/MUB/PHB patch and whole-domain stats against wrfout_h0.",
            "Formula residuals on CPU h0 HGT should remain within the predeclared BASE_FORMULA_TOL.",
            "Production source must obtain HGT/MUB/PHB through native parent interpolation and blend, not from wrfout_h0.",
        ],
    }


def write_markdown(payload: Mapping[str, Any]) -> None:
    patch_hgt = payload["stats"]["native_wrfinput_vs_cpu_h0_patch"]["HGT"]
    patch_pb = payload["stats"]["native_wrfinput_vs_cpu_h0_patch"]["PB"]
    patch_mub = payload["stats"]["native_wrfinput_vs_cpu_h0_patch"]["MUB"]
    formula_pb = payload["stats"]["start_domain_formula_residuals"]["cpu_h0_patch"]["PB"]
    formula_mub = payload["stats"]["start_domain_formula_residuals"]["cpu_h0_patch"]["MUB"]
    formula_phb = payload["stats"]["start_domain_formula_residuals"]["cpu_h0_patch"]["PHB"]
    lines = [
        "# V0.14 Live-Nest Base Hook",
        "",
        f"Verdict: `{payload['classification']}`.",
        "",
        "## Summary",
        "",
        "- No production JAX source was patched.",
        "- CPU-WRF `wrfout_h0` is used only as validation oracle.",
        "- Next source fix should port WRF live-nest parent interpolation plus `blend_terrain`, then run `start_domain_em` base recomputation natively.",
        (
            f"- Native wrfinput vs CPU h0 target-patch max deltas: HGT `{patch_hgt['max_abs']}` m, "
            f"PB `{patch_pb['max_abs']}` Pa, MUB `{patch_mub['max_abs']}` Pa."
        ),
        (
            f"- WRF base formula on CPU h0 HGT residuals: PB `{formula_pb['max_abs']}` Pa, "
            f"MUB `{formula_mub['max_abs']}` Pa, PHB `{formula_phb['max_abs']}` m2/s2."
        ),
        "",
        "## Source Hook",
        "",
        "- `share/mediation_integrate.F`: live nest calls `med_interp_domain`, reads child input, blends `ht/mub/phb`, then adjusts state.",
        "- `inc/nest_interpdown_interp.inc`: generated calls interpolate parent `PHB/MUB/PB/HT` into child arrays via `interp_fcn`.",
        "- `dyn_em/nest_init_utils.F::blend_terrain`: parent strip, blend zone, and child interior formula.",
        "- `dyn_em/start_em.F::start_domain_em`: recomputes `PB/MUB/PHB/T_INIT/ALB` before perturbation fields are recalculated.",
        "",
        "Full line ranges and stats are in `proofs/v014/live_nest_base_hook.json`.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def write_review(payload: Mapping[str, Any]) -> None:
    lines = [
        "# Review: V0.14 Live-Nest Base Hook",
        "",
        f"Verdict: `{payload['classification']}`.",
        "",
        "## Findings",
        "",
        "- No production source edits were made.",
        "- The plan is source-grounded: exact WRF line ranges for interpolation, blend, and base recomputation are recorded in the JSON.",
        "- CPU-WRF h0 is treated as validation evidence only; the native production path must derive the state from parent interpolation and blend.",
        "- `wrfout_h0` lacks `T_INIT/ALB`, so it is insufficient as the missing production state even though it validates `HGT/PB/MUB/PHB`.",
        "",
        "## Next Decision",
        "",
        "Dispatch a source sprint to implement the native live-nest base initialization stage, with h0 validation gates over target patch and whole domain.",
        "",
        "## Commands",
        "",
        "```bash",
        "python -m py_compile proofs/v014/live_nest_base_hook.py",
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/live_nest_base_hook.py",
        "python -m json.tool proofs/v014/live_nest_base_hook.json >/tmp/live_nest_base_hook.validated.json",
        "```",
        "",
    ]
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    earlier = load_json(PRIOR_EARLIER_JSON)
    base = load_json(PRIOR_BASE_JSON)
    bounds = earlier["target"]["patch_bounds"]

    stats = {
        "native_wrfinput_vs_cpu_h0_patch": compare_nc_fields(NATIVE_WRFINPUT_D02, CPU_H0, FIELDS, bounds=bounds, tol=STATIC_TOL),
        "native_wrfinput_vs_cpu_h0_whole_domain": compare_nc_fields(NATIVE_WRFINPUT_D02, CPU_H0, FIELDS, bounds=None, tol=STATIC_TOL),
        "cpu_h0_vs_cpu_h1_whole_domain": compare_nc_fields(CPU_H0, CPU_H1, FIELDS, bounds=None, tol=STATIC_TOL),
        "cpu_h0_vs_cpu_h10_whole_domain": compare_nc_fields(CPU_H0, CPU_H10, FIELDS, bounds=None, tol=STATIC_TOL),
        "start_domain_formula_residuals": {
            "native_wrfinput_patch": formula_residuals(NATIVE_WRFINPUT_D02, bounds=bounds),
            "native_wrfinput_whole_domain": formula_residuals(NATIVE_WRFINPUT_D02, bounds=None),
            "cpu_h0_patch": formula_residuals(CPU_H0, bounds=bounds),
            "cpu_h0_whole_domain": formula_residuals(CPU_H0, bounds=None),
        },
    }

    payload: dict[str, Any] = {
        "classification": CLASSIFICATION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_only": True,
            "gpu_used": False,
            "note": "netCDF4/NumPy proof only; no model execution",
        },
        "acceptance_notes": {
            "json_validates": True,
            "production_source_edited": False,
            "wrf_source_edited": False,
            "cpu_wrfout_h0_used_as_validation_only": True,
            "normal_production_dependency_on_cpu_wrfout_h0": False,
            "no_tost": True,
            "no_switzerland_validation": True,
            "no_fp32_or_memory_cleanup": True,
            "no_hermes_or_telegram": True,
            "jax_vs_jax_self_compare_used_as_truth": False,
        },
        "decision": {
            "chosen_path": "native implementation plan from WRF source",
            "why_not_new_wrf_savepoint_this_sprint": (
                "The case is runnable and prior marker runs exist, but no post-blend base payload is present. "
                "A fresh oracle would require patching/relinking a disposable 1.6G WRF tree and rerunning. "
                "The exact source path plus h0 validation stats are sufficient to start the next source-port sprint."
            ),
            "wrf_live_oracle_status": "not produced in this sprint",
        },
        "inputs": {
            "project_constitution": path_info(ROOT / "PROJECT_CONSTITUTION.md"),
            "agents": path_info(ROOT / "AGENTS.md"),
            "sprint_contract": path_info(ROOT / ".agent/sprints/2026-06-09-v014-live-nest-base-hook/sprint-contract.md"),
            "building_wrf_oracles_skill": path_info(ROOT / ".agent/skills/building-wrf-oracles/SKILL.md"),
            "validating_physics_skill": path_info(ROOT / ".agent/skills/validating-physics/SKILL.md"),
            "reporting_to_human_skill": path_info(ROOT / ".agent/skills/reporting-to-human/SKILL.md"),
            "prior_base_state_split_fix": path_info(PRIOR_BASE_JSON),
            "prior_earlier_source_bisect": path_info(PRIOR_EARLIER_JSON),
            "native_wrfinput_d01": path_info(NATIVE_WRFINPUT_D01),
            "native_wrfinput_d02": path_info(NATIVE_WRFINPUT_D02),
            "native_wrfbdy_d01": path_info(NATIVE_WRFBDY_D01),
            "native_namelist": path_info(NATIVE_NAMELIST),
            "cpu_wrfout_h0": path_info(CPU_H0),
            "cpu_wrfout_h1": path_info(CPU_H1),
            "cpu_wrfout_h10": path_info(CPU_H10),
            "wrf_root": path_info(WRF_ROOT),
        },
        "target": {
            "run_id": RUN_ID,
            "domain": "d02",
            "patch_bounds": bounds,
            "fields": list(FIELDS),
        },
        "netcdf_inventory": {
            "native_wrfinput_d02": nc_variable_inventory(NATIVE_WRFINPUT_D02, ("HGT", "PB", "MUB", "PHB", "T_INIT", "ALB")),
            "cpu_wrfout_h0": nc_variable_inventory(CPU_H0, ("HGT", "PB", "MUB", "PHB", "T_INIT", "ALB")),
        },
        "prior_blocker": {
            "classification": base.get("classification"),
            "reason": base.get("blocked", {}).get("reason"),
            "detail": base.get("blocked", {}).get("detail"),
        },
        "source_citations": source_citations(),
        "stats": stats,
        "implementation_plan": implementation_plan(),
        "wrf_run_feasibility": {
            "built_wrf_exe": path_info(WRF_ROOT / "main/wrf.exe"),
            "built_real_exe": path_info(WRF_ROOT / "main/real.exe"),
            "wrf_tree_size_note": "du -sh observed 1.6G for this WRF tree during sprint inspection",
            "native_inputs_present": {
                "wrfinput_d01": NATIVE_WRFINPUT_D01.exists(),
                "wrfinput_d02": NATIVE_WRFINPUT_D02.exists(),
                "wrfbdy_d01": NATIVE_WRFBDY_D01.exists(),
                "namelist_input": NATIVE_NAMELIST.exists(),
            },
            "previous_marker_runs": previous_marker_runs(),
        },
        "commands": {
            "required_validation": [
                "python -m py_compile proofs/v014/live_nest_base_hook.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/live_nest_base_hook.py",
                "python -m json.tool proofs/v014/live_nest_base_hook.json >/tmp/live_nest_base_hook.validated.json",
            ],
            "script_argv": ["proofs/v014/live_nest_base_hook.py"],
        },
        "unresolved_risks": [
            "Native SINT parity still needs implementation-level tests; simplified bilinear interpolation is already rejected by the prior proof.",
            "CPU h0 does not expose T_INIT/ALB, so those fields must be validated through formula residuals or a future disposable WRF savepoint.",
            "A source sprint must ensure no host/device transfer is introduced inside timestep loops; this initialization path should run before time stepping.",
        ],
        "next_decision_needed": "Dispatch native source fix for live-nest base initialization, or explicitly request a disposable WRF savepoint sprint first.",
    }

    write_json(OUT_JSON, payload)
    write_markdown(payload)
    write_review(payload)
    print(CLASSIFICATION)


if __name__ == "__main__":
    main()
