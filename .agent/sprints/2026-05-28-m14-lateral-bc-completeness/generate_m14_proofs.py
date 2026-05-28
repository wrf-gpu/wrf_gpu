#!/usr/bin/env python
"""Generate M14 boundary-completeness proof objects."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.coupling.boundary_apply import DEFAULT_BOUNDARY_CONFIG, SIDES, apply_lateral_boundaries, interpolate_boundary_leaf  # noqa: E402
from gpuwrf.integration.d02_replay import build_replay_case  # noqa: E402
from gpuwrf.io.boundary_replay import _common_extent, _gpu_side_strip, decode_wrfbdy  # noqa: E402


RUN_DIR = Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z")
WRF_HOUR1 = RUN_DIR / "wrfout_d02_2026-05-21_19:00:00"
GPU_HOUR1 = Path("/tmp/m14_lateral_bc_20260521/wrfout_d02_2026-05-21_19:00:00")
M11_HOUR1 = Path("/tmp/m11_theta_pd_limiter_20260521/wrfout_d02_2026-05-21_19:00:00")
PROOF_DIR = ROOT / "proofs/m14"
PIPELINE_PROOF = PROOF_DIR / "pipeline_run_20260521.json"


VARIABLES: dict[str, tuple[str, str]] = {
    "U": ("u", "u_bdy"),
    "V": ("v", "v_bdy"),
    "W": ("w", "w_bdy"),
    "T": ("theta", "theta_bdy"),
    "QVAPOR": ("qv", "qv_bdy"),
    "P": ("p_perturbation", "p_bdy"),
    "PB": ("base_pressure", "pb_bdy"),
    "PH": ("ph_perturbation", "ph_bdy"),
    "PHB": ("base_geopotential", "phb_bdy"),
    "MU": ("mu_perturbation", "mu_bdy"),
    "MUB": ("base_mu", "mub_bdy"),
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "UNKNOWN"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _state_field(state: Any, name: str):
    if name == "base_pressure":
        return state.p_total - state.p_perturbation
    if name == "base_geopotential":
        return state.ph_total - state.ph_perturbation
    if name == "base_mu":
        return state.mu_total - state.mu_perturbation
    return getattr(state, name)


def _truth_side(boundary, side: str, *, lead_seconds: float):
    forcing = np.asarray(interpolate_boundary_leaf(boundary, lead_seconds))
    truth = forcing[SIDES.index(side)]
    if truth.ndim == 3 and truth.shape[1] == 1:
        truth = truth[:, 0, :]
    return truth


def _stats(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    diff = left - right
    rmse = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0
    denom = float(np.mean(np.abs(right))) + 1.0e-12 if diff.size else 1.0
    return {
        "rmse": rmse,
        "rel_rmse": float(rmse / denom),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
    }


def boundary_strip_parity() -> dict[str, Any]:
    case = build_replay_case(RUN_DIR, domain="d02")
    state = case.state.replace(p=case.state.p_total, ph=case.state.ph_total, mu=case.state.mu_total)
    applied = apply_lateral_boundaries(state, 0.0, 10.0, DEFAULT_BOUNDARY_CONFIG)
    width = int(case.metadata["boundary"]["bdy_width"])

    payload: dict[str, Any] = {
        "_metadata": {
            "schema": "m14_boundary_strip_parity_v1",
            "commit": _git_commit(),
            "run_dir": str(RUN_DIR),
            "domain": "d02",
            "lead_seconds": 0.0,
            "dt_s": 10.0,
            "bdy_width": width,
            "threshold_rel_rmse": 1.0e-6,
            "oracle": "GPU-applied WRF specified-zone boundary row compared to d02 hourly wrfout side-history replay leaves.",
            "full_relax_strip_note": "The five-cell relaxation strip is width-checked separately; rows outside spec_zone are nudged by WRF-style relaxation and are not expected to equal forcing targets exactly after one apply.",
            "wrfbdy_d02_status": "MISSING",
            "wrfbdy_d02_note": "No wrfbdy_d02 exists under /mnt/data/canairy_meteo for the pinned 20260521 run; wrfbdy_d01 only provides U/V/W/T/QVAPOR/PH/MU and cannot cover P/PB/PHB/MUB.",
            "theta_transform": "T boundary leaves are stored as absolute theta = WRF T + 300 K.",
        }
    }
    pass_count = 0
    for var, (field_name, boundary_name) in VARIABLES.items():
        field = np.asarray(_state_field(applied, field_name), dtype=np.float64)
        boundary = getattr(state, boundary_name)
        side_records: dict[str, Any] = {}
        full_side_records: dict[str, Any] = {}
        merged_left: list[np.ndarray] = []
        merged_right: list[np.ndarray] = []
        full_left: list[np.ndarray] = []
        full_right: list[np.ndarray] = []
        for side in SIDES:
            full_l = _gpu_side_strip(field, side, width)
            right = _truth_side(boundary, side, lead_seconds=0.0)
            left_index, right_index = _common_extent(full_l, right)
            full_l = full_l[left_index]
            full_r = right[right_index]
            left_common = full_l[:1]
            right_common = full_r[:1]
            trim = 10
            if left_common.shape[-1] > 2 * trim:
                left_common = left_common[..., trim:-trim]
                right_common = right_common[..., trim:-trim]
            side_records[side] = {
                **_stats(left_common, right_common),
                "compared_shape": list(left_common.shape),
            }
            full_side_records[side] = {
                **_stats(full_l, full_r),
                "compared_shape": list(full_l.shape),
            }
            merged_left.append(left_common.reshape(-1))
            merged_right.append(right_common.reshape(-1))
            full_left.append(full_l.reshape(-1))
            full_right.append(full_r.reshape(-1))
        total = _stats(np.concatenate(merged_left), np.concatenate(merged_right))
        full_total = _stats(np.concatenate(full_left), np.concatenate(full_right))
        verdict = "PASS" if total["rel_rmse"] <= 1.0e-6 else "FAIL"
        if verdict == "PASS":
            pass_count += 1
        payload[var] = {
            "rel_rmse": total["rel_rmse"],
            "rmse": total["rmse"],
            "max_abs": total["max_abs"],
            "verdict": verdict,
            "sides": side_records,
            "full_relax_strip_diagnostic": {
                "rel_rmse": full_total["rel_rmse"],
                "rmse": full_total["rmse"],
                "max_abs": full_total["max_abs"],
                "sides": full_side_records,
            },
        }
    payload["_metadata"]["pass_count"] = pass_count
    payload["_metadata"]["variable_count"] = len(VARIABLES)
    payload["_metadata"]["headline"] = f"{pass_count}/{len(VARIABLES)} variables PASS"
    return payload


def _parse_bdy_control(namelist: Path) -> dict[str, int]:
    text = namelist.read_text(encoding="utf-8")
    result = {}
    for key in ("spec_bdy_width", "spec_zone", "relax_zone"):
        match = re.search(rf"^\s*{key}\s*=\s*([0-9]+)", text, flags=re.MULTILINE)
        if match:
            result[key] = int(match.group(1))
    return result


def width_parity(case_metadata: dict[str, Any]) -> dict[str, Any]:
    state = build_replay_case(RUN_DIR, domain="d02").state
    namelist_values = _parse_bdy_control(RUN_DIR / "namelist.input")
    wrfbdy_d01 = RUN_DIR / "wrfbdy_d01"
    wrfbdy_d02 = RUN_DIR / "wrfbdy_d02"
    decoded_d01 = decode_wrfbdy(wrfbdy_d01, variables=("U",), time_index=0)
    widths = {
        boundary_name: int(getattr(state, boundary_name).shape[2])
        for _var, (_field_name, boundary_name) in VARIABLES.items()
    }
    expected = int(namelist_values.get("spec_bdy_width", DEFAULT_BOUNDARY_CONFIG.spec_bdy_width))
    return {
        "schema": "m14_relax_zone_width_parity_v1",
        "commit": _git_commit(),
        "run_dir": str(RUN_DIR),
        "namelist_bdy_control": namelist_values,
        "wrfbdy_d01": {"path": str(wrfbdy_d01), "bdy_width": int(decoded_d01["bdy_width"])},
        "wrfbdy_d02": {"path": str(wrfbdy_d02), "exists": wrfbdy_d02.exists(), "status": "MISSING"},
        "state_boundary_widths": widths,
        "metadata_boundary": case_metadata.get("boundary", {}),
        "verdict": "PASS" if all(value == expected for value in widths.values()) and int(decoded_d01["bdy_width"]) == expected else "FAIL",
    }


def _split_status() -> dict[str, Any]:
    pipeline = json.loads(PIPELINE_PROOF.read_text(encoding="utf-8")) if PIPELINE_PROOF.is_file() else {}
    return {
        "schema": "m14_interior_vs_boundary_split_v1",
        "commit": _git_commit(),
        "case": "20260521",
        "domain": "d02",
        "boundary_width_cells": 10,
        "gpu_hour1": str(GPU_HOUR1),
        "wrf_hour1": str(WRF_HOUR1),
        "m11_baseline_hour1": str(M11_HOUR1),
        "status": "BLOCKED",
        "verdict": "M14_BLOCKED",
        "reason": "No M14 hour-1 wrfout was produced; the 1h pipeline emitted PIPELINE_BLOCKED after nonfinite model state at hour 1.",
        "pipeline_verdict": pipeline.get("verdict"),
        "pipeline_reason": pipeline.get("reason"),
        "baseline_available": {"m11": M11_HOUR1.is_file(), "wrf": WRF_HOUR1.is_file()},
        "acceptance_checks": {
            "boundary_strip_rmse_drops_vs_m11": "NOT_EVALUATED",
            "interior_rmse_holds_or_improves": "NOT_EVALUATED",
        },
    }


def main() -> int:
    PROOF_DIR.mkdir(parents=True, exist_ok=True)
    strip = boundary_strip_parity()
    _write_json(PROOF_DIR / "boundary_strip_parity.json", strip)
    case_metadata = build_replay_case(RUN_DIR, domain="d02").metadata
    _write_json(PROOF_DIR / "relax_zone_width_parity.json", width_parity(case_metadata))
    _write_json(PROOF_DIR / "interior_vs_boundary_split.json", _split_status())
    print(strip["_metadata"]["headline"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
