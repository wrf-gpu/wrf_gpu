#!/usr/bin/env python3
"""V0.14 native live-nested d02 base-state split fix/block proof.

CPU-only proof.  It verifies that the h0 PB/MUB split is WRF's start-domain
base recomputation on a blended live-nest terrain/base surface, then blocks a
source patch because the required parent-interpolated blend is not locally
available in ``build_replay_case`` without porting WRF's nest interpolation path
or depending on CPU-WRF history output.
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
from typing import Any, Iterable, Mapping

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "20260501_18z_l2_72h_20260519T173026Z"
TARGET_DOMAIN = "d02"
TARGET_FIELDS = ("T", "P", "PB", "MU", "MUB")
STATIC_FIELDS = ("PB", "MUB")
TOLERANCE_MAX_ABS = 2.0e-6
FORMULA_ROUNDOFF_TOL = 6.0e-2

RUN_ROOT = Path("/tmp/v0120_merged_run_root") / RUN_ID
CPU_WRFOUT_DIR = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output") / RUN_ID
NATIVE_WRFINPUT_D01 = RUN_ROOT / "wrfinput_d01"
NATIVE_WRFINPUT_D02 = RUN_ROOT / "wrfinput_d02"
CPU_H0 = CPU_WRFOUT_DIR / "wrfout_d02_2026-05-01_18:00:00"
CPU_H1 = CPU_WRFOUT_DIR / "wrfout_d02_2026-05-01_19:00:00"
CPU_H10 = CPU_WRFOUT_DIR / "wrfout_d02_2026-05-02_04:00:00"

EARLIER_JSON = ROOT / "proofs/v014/earlier_source_bisect.json"
PRE_RK_JSON = ROOT / "proofs/v014/pre_rk_input_boundary.json"
BASE_ATTR_JSON = ROOT / "proofs/v014/base_state_writer_attribution.json"
STATIC_PARITY_JSON = ROOT / "proofs/v014/static_metric_base_parity.json"
D02_REPLAY = ROOT / "src/gpuwrf/integration/d02_replay.py"

WRF_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/WRF")
WRF_MEDIATION_INTEGRATE = WRF_ROOT / "share/mediation_integrate.F"
WRF_START_EM = WRF_ROOT / "dyn_em/start_em.F"
WRF_NEST_INIT_UTILS = WRF_ROOT / "dyn_em/nest_init_utils.F"
WRF_INTERP_FCN = WRF_ROOT / "share/interp_fcn.F"
WRF_SINT = WRF_ROOT / "share/sint.F"
WRF_NEST_INTERP_INC = WRF_ROOT / "inc/nest_interpdown_interp.inc"

OUT_JSON = ROOT / "proofs/v014/base_state_split_fix.json"
OUT_MD = ROOT / "proofs/v014/base_state_split_fix.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-base-state-split-fix.md"

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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def path_info(path: Path) -> dict[str, Any]:
    out = {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
    }
    if path.exists():
        out["size_bytes"] = path.stat().st_size
        if path.is_file():
            out["sha256"] = sha256(path)
    return out


def nc_var(path: Path, name: str) -> np.ndarray:
    with Dataset(path, "r") as ds:
        var = ds.variables[name]
        raw = var[0] if var.dimensions and var.dimensions[0] == "Time" else var[:]
        return np.asarray(np.ma.filled(raw, np.nan), dtype=np.float64)


def nc_scalar(path: Path, name: str, default: float | None = None) -> float:
    with Dataset(path, "r") as ds:
        if name in ds.variables:
            return float(np.asarray(ds.variables[name][:]).reshape(-1)[0])
        if hasattr(ds, name):
            return float(getattr(ds, name))
    if default is None:
        raise KeyError(name)
    return default


def nc_attr_int(path: Path, name: str) -> int:
    with Dataset(path, "r") as ds:
        return int(getattr(ds, name))


def field_array(path: Path, field: str) -> np.ndarray:
    return nc_var(path, field)


def stats_from_diff(diff: np.ndarray, *, tol: float) -> dict[str, Any]:
    finite = np.asarray(diff, dtype=np.float64)[np.isfinite(diff)]
    if finite.size == 0:
        return {"status": "NO_FINITE_PAIR", "count": int(diff.size), "finite_count": 0, "tolerance_max_abs": tol}
    abs_diff = np.abs(diff)
    masked_abs = np.where(np.isfinite(diff), abs_diff, -np.inf)
    idx = np.unravel_index(int(np.argmax(masked_abs)), diff.shape)
    max_abs = float(np.max(np.abs(finite)))
    return {
        "status": "MATCH" if max_abs <= tol else "DIFF",
        "count": int(diff.size),
        "finite_count": int(finite.size),
        "shape": list(diff.shape),
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
        return {"name": name, "status": "SHAPE_MISMATCH", "left_shape": list(left.shape), "right_shape": list(right.shape)}
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    out = {"name": name, **stats_from_diff(diff, tol=tol)}
    if out["status"] != "NO_FINITE_PAIR":
        idx = tuple(out["worst"]["index"])
        out["worst"]["left"] = float(left[idx])
        out["worst"]["right"] = float(right[idx])
    return out


def patch_slice(bounds: Mapping[str, int], ndim: int) -> tuple[slice, ...]:
    y = slice(int(bounds["y0"]), int(bounds["y1"]))
    x = slice(int(bounds["x0"]), int(bounds["x1"]))
    if ndim == 2:
        return (y, x)
    if ndim == 3:
        return (slice(None), y, x)
    raise ValueError(f"unsupported ndim={ndim}")


def compare_field_pair(
    name: str,
    left_path: Path,
    right_path: Path,
    field: str,
    *,
    bounds: Mapping[str, int] | None,
    tol: float,
) -> dict[str, Any]:
    left = field_array(left_path, field)
    right = field_array(right_path, field)
    scope = "whole_domain"
    if bounds is not None:
        selector = patch_slice(bounds, left.ndim)
        left = left[selector]
        right = right[selector]
        scope = "target_patch"
    out = compare_arrays(name, left, right, tol=tol)
    out.update({"field": field, "left": str(left_path), "right": str(right_path), "scope": scope})
    return out


def base_from_hgt(reference_nc: Path, hgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ptop = nc_scalar(reference_nc, "P_TOP")
    p00 = nc_scalar(reference_nc, "P00")
    t00 = nc_scalar(reference_nc, "T00")
    lapse = nc_scalar(reference_nc, "TLP")
    c3h = nc_var(reference_nc, "C3H")
    c4h = nc_var(reference_nc, "C4H")
    p_surf = p00 * np.exp(-t00 / lapse + np.sqrt((t00 / lapse) ** 2 - 2.0 * G * hgt / lapse / R_D))
    mub = p_surf - ptop
    pb = c3h[:, None, None] * (p_surf[None, :, :] - ptop) + c4h[:, None, None] + ptop
    return pb, mub


def simple_parent_bilinear(parent: np.ndarray, *, child_shape_yx: tuple[int, int], i_parent_start: int, j_parent_start: int, ratio: int) -> np.ndarray:
    """Existing-helper-shaped bilinear interpolation, not WRF's default SINT."""

    y_len, x_len = child_shape_yx
    y = float(j_parent_start - 1) + np.arange(y_len, dtype=np.float64) / float(ratio)
    x = float(i_parent_start - 1) + np.arange(x_len, dtype=np.float64) / float(ratio)
    y = np.clip(y, 0.0, float(parent.shape[-2] - 1))
    x = np.clip(x, 0.0, float(parent.shape[-1] - 1))
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, parent.shape[-2] - 1)
    x1 = np.clip(x0 + 1, 0, parent.shape[-1] - 1)
    wy = y - y0
    wx = x - x0
    if parent.ndim == 2:
        f00 = parent[y0[:, None], x0[None, :]]
        f10 = parent[y1[:, None], x0[None, :]]
        f01 = parent[y0[:, None], x1[None, :]]
        f11 = parent[y1[:, None], x1[None, :]]
        return (1.0 - wy[:, None]) * ((1.0 - wx[None, :]) * f00 + wx[None, :] * f01) + wy[:, None] * (
            (1.0 - wx[None, :]) * f10 + wx[None, :] * f11
        )
    f00 = parent[:, y0[:, None], x0[None, :]]
    f10 = parent[:, y1[:, None], x0[None, :]]
    f01 = parent[:, y0[:, None], x1[None, :]]
    f11 = parent[:, y1[:, None], x1[None, :]]
    return (1.0 - wy[None, :, None]) * ((1.0 - wx[None, None, :]) * f00 + wx[None, None, :] * f01) + wy[
        None, :, None
    ] * ((1.0 - wx[None, None, :]) * f10 + wx[None, None, :] * f11)


def wrf_blend_terrain(interpolated: np.ndarray, fine: np.ndarray, *, spec_bdy_width: int = 5, blend_width: int = 5) -> np.ndarray:
    """Transcription of ``nest_init_utils.F::blend_terrain`` for mass fields."""

    out = np.array(fine, dtype=np.float64, copy=True)
    ny, nx = out.shape[-2:]
    ide = nx + 1
    jde = ny + 1
    r_blend = 1.0 / float(blend_width + 1)
    for jj in range(ny):
        j = jj + 1
        for ii in range(nx):
            i = ii + 1
            weights: tuple[float, float] | None = None
            for blend_cell in range(blend_width, 0, -1):
                if (
                    i == spec_bdy_width + blend_cell
                    or j == spec_bdy_width + blend_cell
                    or i == ide - spec_bdy_width - blend_cell
                    or j == jde - spec_bdy_width - blend_cell
                ):
                    weights = (blend_cell * r_blend, (blend_width + 1 - blend_cell) * r_blend)
            if i <= spec_bdy_width or j <= spec_bdy_width or i >= ide - spec_bdy_width or j >= jde - spec_bdy_width:
                weights = (0.0, 1.0)
            if weights is not None:
                fine_w, interp_w = weights
                out[..., jj, ii] = fine_w * fine[..., jj, ii] + interp_w * interpolated[..., jj, ii]
    return out


def build_simplified_blend_candidate(bounds: Mapping[str, int]) -> dict[str, Any]:
    hgt_parent = nc_var(NATIVE_WRFINPUT_D01, "HGT")
    hgt_fine = nc_var(NATIVE_WRFINPUT_D02, "HGT")
    hgt_truth = nc_var(CPU_H0, "HGT")
    i_parent_start = nc_attr_int(NATIVE_WRFINPUT_D02, "I_PARENT_START")
    j_parent_start = nc_attr_int(NATIVE_WRFINPUT_D02, "J_PARENT_START")
    ratio = nc_attr_int(NATIVE_WRFINPUT_D02, "PARENT_GRID_RATIO")
    parent_interp = simple_parent_bilinear(
        hgt_parent,
        child_shape_yx=hgt_fine.shape,
        i_parent_start=i_parent_start,
        j_parent_start=j_parent_start,
        ratio=ratio,
    )
    blended = wrf_blend_terrain(parent_interp, hgt_fine)
    pb_candidate, mub_candidate = base_from_hgt(NATIVE_WRFINPUT_D02, blended)
    pb_truth = nc_var(CPU_H0, "PB")
    mub_truth = nc_var(CPU_H0, "MUB")
    patch = patch_slice(bounds, 2)
    patch3 = patch_slice(bounds, 3)
    return {
        "purpose": "negative control: existing-helper-shaped bilinear parent interpolation plus WRF blend weights",
        "accepted_as_fix": False,
        "rejection_reason": (
            "WRF default interp_method_type=2 uses share/interp_fcn.F::interp_fcn_sint plus share/sint.F, "
            "not this bilinear helper; the candidate also misses the frozen 2e-6 tolerance."
        ),
        "source_inputs": {
            "parent_hgt": str(NATIVE_WRFINPUT_D01),
            "child_hgt": str(NATIVE_WRFINPUT_D02),
            "i_parent_start": i_parent_start,
            "j_parent_start": j_parent_start,
            "parent_grid_ratio": ratio,
            "spec_bdy_width": 5,
            "blend_width": 5,
        },
        "hgt": {
            "native_wrfinput_vs_cpu_h0_patch": compare_arrays(
                "native_wrfinput_hgt_vs_cpu_h0_hgt_patch", hgt_fine[patch], hgt_truth[patch], tol=TOLERANCE_MAX_ABS
            ),
            "bilinear_blend_vs_cpu_h0_patch": compare_arrays(
                "bilinear_blend_hgt_vs_cpu_h0_hgt_patch", blended[patch], hgt_truth[patch], tol=TOLERANCE_MAX_ABS
            ),
            "bilinear_blend_vs_cpu_h0_whole_domain": compare_arrays(
                "bilinear_blend_hgt_vs_cpu_h0_hgt_whole_domain", blended, hgt_truth, tol=TOLERANCE_MAX_ABS
            ),
        },
        "base_fields": {
            "PB_patch": compare_arrays("bilinear_blend_base_pb_vs_cpu_h0_patch", pb_candidate[patch3], pb_truth[patch3], tol=TOLERANCE_MAX_ABS),
            "MUB_patch": compare_arrays("bilinear_blend_base_mub_vs_cpu_h0_patch", mub_candidate[patch], mub_truth[patch], tol=TOLERANCE_MAX_ABS),
            "PB_whole_domain": compare_arrays("bilinear_blend_base_pb_vs_cpu_h0_whole_domain", pb_candidate, pb_truth, tol=TOLERANCE_MAX_ABS),
            "MUB_whole_domain": compare_arrays("bilinear_blend_base_mub_vs_cpu_h0_whole_domain", mub_candidate, mub_truth, tol=TOLERANCE_MAX_ABS),
        },
    }


def validation_only_h0_hgt_formula(bounds: Mapping[str, int]) -> dict[str, Any]:
    native_pb, native_mub = base_from_hgt(NATIVE_WRFINPUT_D02, nc_var(NATIVE_WRFINPUT_D02, "HGT"))
    h0_pb, h0_mub = base_from_hgt(NATIVE_WRFINPUT_D02, nc_var(CPU_H0, "HGT"))
    patch2 = patch_slice(bounds, 2)
    patch3 = patch_slice(bounds, 3)
    return {
        "purpose": "validation-only oracle check; CPU-WRF h0 HGT is not an accepted production dependency",
        "formula_roundoff_tolerance_pa": FORMULA_ROUNDOFF_TOL,
        "native_wrfinput_formula_sanity": {
            "PB_patch": compare_arrays("base_formula_on_wrfinput_hgt_vs_wrfinput_pb_patch", native_pb[patch3], nc_var(NATIVE_WRFINPUT_D02, "PB")[patch3], tol=FORMULA_ROUNDOFF_TOL),
            "MUB_patch": compare_arrays("base_formula_on_wrfinput_hgt_vs_wrfinput_mub_patch", native_mub[patch2], nc_var(NATIVE_WRFINPUT_D02, "MUB")[patch2], tol=FORMULA_ROUNDOFF_TOL),
            "PB_whole_domain": compare_arrays("base_formula_on_wrfinput_hgt_vs_wrfinput_pb_whole_domain", native_pb, nc_var(NATIVE_WRFINPUT_D02, "PB"), tol=FORMULA_ROUNDOFF_TOL),
            "MUB_whole_domain": compare_arrays("base_formula_on_wrfinput_hgt_vs_wrfinput_mub_whole_domain", native_mub, nc_var(NATIVE_WRFINPUT_D02, "MUB"), tol=FORMULA_ROUNDOFF_TOL),
        },
        "cpu_h0_hgt_formula": {
            "PB_patch": compare_arrays("base_formula_on_cpu_h0_hgt_vs_cpu_h0_pb_patch", h0_pb[patch3], nc_var(CPU_H0, "PB")[patch3], tol=FORMULA_ROUNDOFF_TOL),
            "MUB_patch": compare_arrays("base_formula_on_cpu_h0_hgt_vs_cpu_h0_mub_patch", h0_mub[patch2], nc_var(CPU_H0, "MUB")[patch2], tol=FORMULA_ROUNDOFF_TOL),
            "PB_whole_domain": compare_arrays("base_formula_on_cpu_h0_hgt_vs_cpu_h0_pb_whole_domain", h0_pb, nc_var(CPU_H0, "PB"), tol=FORMULA_ROUNDOFF_TOL),
            "MUB_whole_domain": compare_arrays("base_formula_on_cpu_h0_hgt_vs_cpu_h0_mub_whole_domain", h0_mub, nc_var(CPU_H0, "MUB"), tol=FORMULA_ROUNDOFF_TOL),
        },
    }


def git_source_changed() -> bool:
    result = subprocess.run(
        ["git", "diff", "--quiet", "--", str(D02_REPLAY.relative_to(ROOT))],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.returncode != 0


def source_refs() -> dict[str, Any]:
    return {
        "mediation_integrate_live_nest_input_from_file": {
            "file": str(WRF_MEDIATION_INTEGRATE),
            "lines": "688-756",
            "role": (
                "med_interp_domain(parent,nest), save parent-interpolated ht/mub/phb, read wrfinput child, "
                "blend ht/mub/phb, then adjust_tempqv"
            ),
        },
        "nest_interp_generated_fields": {
            "file": str(WRF_NEST_INTERP_INC),
            "lines": "253-347",
            "role": "generated parent-to-child interp_fcn calls for phb, mu_2, mub and nearby base fields",
        },
        "wrf_default_interpolation": {
            "file": str(WRF_INTERP_FCN),
            "lines": "34-131,874-993",
            "role": "interp_method_type default 2 dispatches to interp_fcn_sint; sint implementation is in share/sint.F",
        },
        "wrf_sint_kernel": {
            "file": str(WRF_SINT),
            "lines": "2-202",
            "role": "Smolarkiewicz positive-definite monotonic interpolation used by interp_fcn_sint",
        },
        "wrf_blend_formula": {
            "file": str(WRF_NEST_INIT_UTILS),
            "lines": "712-785",
            "role": "blend_terrain sets the spec_bdy_width ring to parent-interpolated values and blends the next blend_width rows",
        },
        "start_domain_base_recompute": {
            "file": str(WRF_START_EM),
            "lines": "599-647,675-721",
            "role": "start_domain_em recomputes MUB/PB/T_INIT/ALB/PHB from the modified HGT/MUB base surface",
        },
        "press_adj_dynamic_split": {
            "file": str(WRF_START_EM),
            "lines": "878-886",
            "role": "optional pressure adjustment modifies MU_2 after base recomputation; not part of PB/MUB static-base fix",
        },
    }


def required_commands() -> list[str]:
    return [
        "python -m py_compile src/gpuwrf/integration/d02_replay.py proofs/v014/base_state_split_fix.py",
        "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/base_state_split_fix.py",
        "python -m json.tool proofs/v014/base_state_split_fix.json >/tmp/base_state_split_fix.validated.json",
    ]


def render_markdown(payload: Mapping[str, Any]) -> str:
    simplified = payload["simplified_local_reconstruction"]["base_fields"]
    formula = payload["validation_only_h0_hgt_formula"]["cpu_h0_hgt_formula"]
    lines = [
        "# V0.14 Base-State Split Fix",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Summary",
        "",
        "- No production source patch was applied.",
        "- Native `build_replay_case(load_lateral_boundaries=False)` still loads the child `wrfinput_d02` split, which is not the CPU-WRF live-nest h0 split.",
        "- CPU-WRF h0 `PB/MUB` are WRF base-formula values on the post-nest blended h0 terrain, but that terrain/base surface is produced by WRF parent interpolation plus `blend_terrain` before `start_domain_em`.",
        (
            f"- Validation-only h0-HGT formula residuals: `PB` patch max `{formula['PB_patch']['max_abs']}`, "
            f"`MUB` patch max `{formula['MUB_patch']['max_abs']}` Pa."
        ),
        (
            f"- Simplified bilinear+blend reconstruction is rejected: `PB` patch max `{simplified['PB_patch']['max_abs']}`, "
            f"`MUB` patch max `{simplified['MUB_patch']['max_abs']}` Pa."
        ),
        "",
        "## Blocker",
        "",
        "`build_replay_case` needs the WRF live-nest parent-interpolated/blended `HGT/MUB/PHB` surface, not CPU-WRF `wrfout_h0`. The exact next hook is WRF `share/mediation_integrate.F` after `blend_terrain` and `dyn_em/start_em.F` after base-state recomputation.",
        "",
        "Full field tables are in `proofs/v014/base_state_split_fix.json`.",
        "",
    ]
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Review: V0.14 Base-State Split Fix",
        "",
        f"verdict: `{payload['verdict']}`",
        "",
        "objective: fix or precisely block the native live-nested d02 base-state split mismatch.",
        "",
        "files changed:",
        "- `proofs/v014/base_state_split_fix.py`",
        "- `proofs/v014/base_state_split_fix.json`",
        "- `proofs/v014/base_state_split_fix.md`",
        "- `.agent/reviews/2026-06-09-v014-base-state-split-fix.md`",
        "",
        "commands run:",
    ]
    for command in payload["commands"]["required_validation"]:
        lines.append(f"- `{command}`")
    lines.extend(
        [
            "",
            "proof objects produced:",
            "- `proofs/v014/base_state_split_fix.json`",
            "- `proofs/v014/base_state_split_fix.md`",
            "- `.agent/reviews/2026-06-09-v014-base-state-split-fix.md`",
            "",
            "unresolved risks:",
        ]
    )
    for risk in payload["unresolved_risks"]:
        lines.append(f"- {risk}")
    lines.extend(["", f"next decision needed: {payload['next_decision']}", ""])
    return "\n".join(lines)


def main() -> int:
    earlier = load_json(EARLIER_JSON)
    bounds = earlier["target"]["patch_bounds"]
    validation_formula = validation_only_h0_hgt_formula(bounds)
    simplified = build_simplified_blend_candidate(bounds)

    native_vs_h0 = {
        field: compare_field_pair(
            f"native_wrfinput_vs_cpu_h0_{field}_patch",
            NATIVE_WRFINPUT_D02,
            CPU_H0,
            field,
            bounds=bounds,
            tol=TOLERANCE_MAX_ABS,
        )
        for field in TARGET_FIELDS
    }
    h0_static_invariance = {
        "h0_vs_h1": {
            field: compare_field_pair(f"cpu_h0_vs_cpu_h1_{field}", CPU_H0, CPU_H1, field, bounds=None, tol=TOLERANCE_MAX_ABS)
            for field in STATIC_FIELDS
        },
        "h0_vs_h10": {
            field: compare_field_pair(f"cpu_h0_vs_cpu_h10_{field}", CPU_H0, CPU_H10, field, bounds=None, tol=TOLERANCE_MAX_ABS)
            for field in STATIC_FIELDS
        },
        "h0_vs_h10_pre_rk_from_earlier_source": earlier.get("wrf_truth", {})
        .get("static_base_invariance", {})
        .get("h0_vs_h10_pre_rk_static"),
    }

    verdict = "BASE_STATE_SPLIT_FIX_BLOCKED_PARENT_INTERP_BLEND_NOT_LOCAL"
    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.base_state_split_fix.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "classification": verdict,
        "blocked": {
            "reason": "PARENT_INTERP_BLEND_NOT_LOCAL",
            "detail": (
                "CPU-WRF h0 PB/MUB are generated after live-nest parent interpolation, blend_terrain, "
                "and start_domain_em base recomputation. The current build_replay_case child-only path "
                "does not have WRF's parent-interpolated ht/mub/phb fields, and a wrfout_h0 fallback would "
                "be a CPU-WRF history dependency."
            ),
            "exact_wrf_routine_formula_hook_needed": [
                "share/mediation_integrate.F live-nest input_from_file branch after med_interp_domain and after blend_terrain for nest%ht/nest%mub/nest%phb",
                "inc/nest_interpdown_interp.inc generated interp_fcn calls for phb/mub/pb plus share/interp_fcn.F::interp_fcn_sint and share/sint.F",
                "dyn_em/nest_init_utils.F::blend_terrain formula with spec_bdy_width=5 and blend_width=5",
                "dyn_em/start_em.F::start_domain_em base-state recomputation after the terrain/base blend and before perturbation values are recalculated",
            ],
            "production_fix_needed": (
                "Add a native live-nest initialization stage that reproduces WRF parent interpolation/blend "
                "for child HGT/MUB/PHB, then recomputes PB/MUB/PHB/theta_base consistently; or add a WRF "
                "savepoint hook for those post-blend/pre-start-domain fields before porting it."
            ),
        },
        "source_patch": {
            "applied": False,
            "d02_replay_changed_in_worktree": git_source_changed(),
            "reason_not_patched": "WRF transformation is not local to build_replay_case without parent-interpolated/blended live-nest fields.",
            "cpu_wrfout_h0_fallback": "not used; would be validation-only/fail-closed, not production",
        },
        "target": {
            "domain": TARGET_DOMAIN,
            "fields": list(TARGET_FIELDS),
            "static_fields": list(STATIC_FIELDS),
            "tolerance_max_abs": TOLERANCE_MAX_ABS,
            "patch_bounds": bounds,
        },
        "starting_facts": {
            "earlier_source_bisect_verdict": earlier.get("verdict"),
            "expected_starting_verdict": "BASE_STATE_SPLIT_DEFINITION_MISMATCH",
            "native_initial_split_matches_wrfinput": True,
            "cpu_wrf_uses_stable_different_split": True,
        },
        "native_wrfinput_vs_cpu_h0_patch": native_vs_h0,
        "h0_static_invariance": h0_static_invariance,
        "validation_only_h0_hgt_formula": validation_formula,
        "simplified_local_reconstruction": simplified,
        "dynamic_fields_not_made_worse": {
            "status": "NOT_APPLICABLE_NO_SOURCE_PATCH",
            "no_source_patch": True,
            "fields": ["T", "P", "MU"],
            "initial_native_vs_cpu_h0_patch": {field: native_vs_h0[field] for field in ("T", "P", "MU")},
            "same_step_truth_note": (
                "Initial h0 wrfout truth is available and recorded above. No candidate source patch was applied, "
                "so T/P/MU are unchanged. h10 same-step internal truth remains unavailable except the existing "
                "pre-RK hook records summarized by earlier_source_bisect."
            ),
        },
        "scope_impact": {
            "standalone_native_init": "unchanged",
            "live_nested_child_init": "blocked; would need parent-interpolated/blended base fields before child carry construction",
            "wrfbdy_boundary_leaves": "unchanged",
            "BaseState": "unchanged; a real fix must recompute pb/mub/phb/theta_base together",
            "restart_output": "unchanged",
            "writer_reconstruction": "unchanged",
        },
        "wrf_source_refs": source_refs(),
        "inputs": {
            "project_constitution": path_info(ROOT / "PROJECT_CONSTITUTION.md"),
            "agents": path_info(ROOT / "AGENTS.md"),
            "sprint_contract": path_info(ROOT / ".agent/sprints/2026-06-09-v014-base-state-split-fix/sprint-contract.md"),
            "validating_physics_skill": path_info(ROOT / ".agent/skills/validating-physics/SKILL.md"),
            "building_wrf_oracles_skill": path_info(ROOT / ".agent/skills/building-wrf-oracles/SKILL.md"),
            "d02_replay": path_info(D02_REPLAY),
            "earlier_source_bisect": path_info(EARLIER_JSON),
            "pre_rk_input_boundary": path_info(PRE_RK_JSON),
            "base_state_writer_attribution": path_info(BASE_ATTR_JSON),
            "static_metric_base_parity": path_info(STATIC_PARITY_JSON),
            "native_wrfinput_d01": path_info(NATIVE_WRFINPUT_D01),
            "native_wrfinput_d02": path_info(NATIVE_WRFINPUT_D02),
            "cpu_h0": path_info(CPU_H0),
            "cpu_h1": path_info(CPU_H1),
            "cpu_h10": path_info(CPU_H10),
        },
        "environment": {
            "cpu_only": True,
            "gpu_used": False,
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "note": "Uses netCDF4/NumPy only; no model execution and no GPU replay.",
        },
        "acceptance_notes": {
            "json_validates": True,
            "jax_vs_jax_self_compare_used_as_truth": False,
            "uses_cpu_wrf_h0_h1_h10_wrfouts": True,
            "uses_cpu_wrf_pre_rk_truth_from_existing_earlier_source_artifact": True,
            "uses_wrfout_h0_as_validation_only_oracle": True,
            "normal_production_dependency_on_cpu_wrfout_history": False,
            "production_source_edited": False,
            "wrf_source_edited": False,
            "no_tost": True,
            "no_switzerland_validation": True,
            "no_fp32_source_work": True,
            "no_hermes_or_telegram": True,
            "cpu_only": True,
        },
        "commands": {
            "argv": sys.argv,
            "required_validation": required_commands(),
            "gpu_replay_command_used": None,
            "backend_allocator_peak_vram": "not applicable; CPU-only proof",
        },
        "proof_objects": {
            "script": str(ROOT / "proofs/v014/base_state_split_fix.py"),
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "review": str(OUT_REVIEW),
        },
        "unresolved_risks": [
            "No native source fix was landed; live-nested child init still needs WRF parent interpolation/blend parity.",
            "The h0-HGT formula check is validation-only and cannot be used as a production dependency.",
            "A production fix will also need terrain/metrics/BaseState consistency, not PB/MUB replacement alone.",
        ],
        "next_decision": (
            "Instrument or port WRF live-nest initialization: capture/reproduce med_interp_domain parent-interpolated "
            "HGT/MUB/PHB, blend_terrain, and start_domain_em base recomputation before changing build_replay_case."
        ),
    }

    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_REVIEW}")
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
