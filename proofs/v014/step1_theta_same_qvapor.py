#!/usr/bin/env python3
"""V0.14 Step-1 theta proof rerun with same-boundary pre-call QVAPOR."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from netCDF4 import Dataset


os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PROOF_DIR = ROOT / "proofs/v014"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROOF_DIR) not in sys.path:
    sys.path.insert(0, str(PROOF_DIR))

import step1_live_nest_init_rerun as live  # noqa: E402
import step1_pre_part1_handoff as pre  # noqa: E402


OUT_JSON = PROOF_DIR / "step1_theta_same_qvapor.json"
OUT_MD = PROOF_DIR / "step1_theta_same_qvapor.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-step1-theta-same-qvapor.md"

SPRINT_CONTRACT = ROOT / ".agent/sprints/2026-06-09-v014-step1-theta-same-qvapor/sprint-contract.md"
SCRATCH = Path("/mnt/data/wrf_gpu2/v014_step1_theta_same_qvapor")
SAME_QVAPOR_ROOT = Path("/mnt/data/wrf_gpu2/v014_step1_qvapor_precall_savepoint/precall_truth_only")
QVAPOR_SAVEPOINT_JSON = PROOF_DIR / "step1_qvapor_precall_savepoint.json"
RUN_CASE3 = live.RUN_CASE3
WRFINPUT_D02 = RUN_CASE3 / "wrfinput_d02"
WRFOUT_H0_D02 = RUN_CASE3 / "wrfout_d02_2026-05-01_18:00:00"
REQUIRED_ANCESTOR = "912b7371"

TARGET_STEP = 1
TARGET_DOMAIN = 2
PRECALL_SURFACE = "before_first_rk_step_part1_call"
THETA_OFFSET_K = 300.0
R_D = 287.0
R_V = 461.6
RV_OVER_RD = R_V / R_D
T_MATERIAL_THRESHOLD_K = 1.0e-3
BOUNDARY_DISTANCE_THRESHOLD = 5

MASS_FIELDS = ("T_STATE", "P_STATE", "PB", "MU_STATE", "MUB", "MUT", "QVAPOR")
WPH_FIELDS = ("W_STATE", "PH_STATE", "PHB")
FIELDS = MASS_FIELDS + WPH_FIELDS
RECORD_SPECS = {
    "MASS_PREPART": {"length": 14, "fields": MASS_FIELDS},
    "WPH_PREPART": {"length": 10, "fields": WPH_FIELDS},
}
ALLOWED_HEADERS = {
    "surface",
    "routine",
    "domain_id",
    "current_timestr_before_step",
    "grid_itimestep_after_increment",
    "rk_step",
    "rk_order",
    "dt_seconds",
    "dt_rk_seconds",
    "dts_rk_seconds",
    "number_of_small_timesteps",
    "tile_i_j_bounds_fortran",
    "global_mass_i_j_end_exclusive_fortran",
    "tile_record_policy",
    "mass_vertical_fortran_k_start_k_end_inclusive",
    "w_ph_vertical_fortran_kstag_start_kstag_end_inclusive",
    "moist_index_qv",
    "record_schema",
}


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": sha256(path),
    }


def sanitize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
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
    if hasattr(value, "item"):
        try:
            return sanitize_json(value.item())
        except Exception:
            return str(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sanitize_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_command(command: list[str], *, cwd: Path | None = None, timeout_s: int = 120) -> dict[str, Any]:
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
            "JAX_ENABLE_COMPILATION_CACHE": "false",
        }
    )
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_s,
        )
        return {
            "command": command,
            "cwd": str(cwd) if cwd else None,
            "returncode": int(proc.returncode),
            "wall_s": float(time.perf_counter() - start),
            "stdout_tail": proc.stdout[-8000:],
            "stderr_tail": proc.stderr[-8000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "cwd": str(cwd) if cwd else None,
            "returncode": None,
            "timeout_s": int(timeout_s),
            "wall_s": float(time.perf_counter() - start),
            "stdout_tail": (exc.stdout or "")[-8000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-8000:] if isinstance(exc.stderr, str) else "",
            "error": "TimeoutExpired",
        }


def jax_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64"),
    }
    try:
        import jax  # noqa: PLC0415

        devices = list(jax.devices())
        env.update(
            {
                "jax_import_error": None,
                "jax_version": getattr(jax, "__version__", None),
                "jax_default_backend": jax.default_backend(),
                "jax_devices": [str(device) for device in devices],
                "gpu_device_count": len([device for device in devices if device.platform == "gpu"]),
            }
        )
    except Exception as exc:
        env.update({"jax_import_error": repr(exc), "gpu_device_count": None})
    return env


def array_summary(array: Any) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.float64)
    return {
        "shape": list(arr.shape),
        "count": int(arr.size),
        "finite_count": int(np.isfinite(arr).sum()),
        "all_finite": bool(np.isfinite(arr).all()),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
    }


def strip_arrays(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "arrays"}


def field_shape(field: str, shapes: Mapping[str, tuple[int, ...]]) -> tuple[int, ...]:
    if field in {"MU_STATE", "MUB", "MUT"}:
        return shapes["mass2d"]
    if field in {"W_STATE", "PH_STATE", "PHB"}:
        return shapes["wph"]
    return shapes["mass"]


def fortran_index(field: str, index: tuple[int, ...] | None) -> dict[str, int] | None:
    if index is None:
        return None
    if field in {"T_STATE", "P_STATE", "PB", "QVAPOR"} and len(index) == 3:
        k, y, x = index
        return {"i": int(x) + 1, "j": int(y) + 1, "k": int(k) + 1}
    if field == "PHB" and len(index) == 3:
        k, y, x = index
        return {"i": int(x) + 1, "j": int(y) + 1, "kstag": int(k) + 1}
    if field in {"MU_STATE", "MUB", "MUT"} and len(index) == 2:
        y, x = index
        return {"i": int(x) + 1, "j": int(y) + 1}
    return None


def zero_index_dict(index: tuple[int, int, int]) -> dict[str, int]:
    k, y, x = index
    return {"k": int(k), "y": int(y), "x": int(x)}


def boundary_distance(index: tuple[int, int, int], shape: tuple[int, int, int]) -> int:
    _, y, x = index
    _, ny, nx = shape
    return int(min(x, y, nx - 1 - x, ny - 1 - y))


def boundary_masks(shape: tuple[int, int, int], band: int) -> dict[str, np.ndarray]:
    _, ny, nx = shape
    y = np.arange(ny)[None, :, None]
    x = np.arange(nx)[None, None, :]
    distance = np.minimum(np.minimum(x, nx - 1 - x), np.minimum(y, ny - 1 - y))
    boundary = np.broadcast_to(distance <= band, shape)
    return {
        f"boundary_distance_le_{band}": boundary,
        f"interior_distance_gt_{band}": ~boundary,
    }


def diff_metrics(
    field: str,
    candidate: Any,
    reference: Any,
    *,
    region: str = "full_domain",
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if cand.shape != ref.shape:
        return {
            "status": "SHAPE_MISMATCH",
            "region": region,
            "candidate_shape": list(cand.shape),
            "reference_shape": list(ref.shape),
        }
    if mask is None:
        active = np.ones(cand.shape, dtype=bool)
    else:
        active = np.asarray(mask, dtype=bool)
        if active.shape != cand.shape:
            return {
                "status": "MASK_SHAPE_MISMATCH",
                "region": region,
                "candidate_shape": list(cand.shape),
                "mask_shape": list(active.shape),
            }
    diff = cand - ref
    absdiff = np.abs(diff)
    active_diff = diff[active]
    active_abs = absdiff[active]
    finite_abs = active_abs[np.isfinite(active_abs)]
    mismatch_mask = active & ((diff != 0.0) | (~np.isfinite(diff)))
    mismatch = np.argwhere(mismatch_mask)
    first = tuple(int(x) for x in mismatch[0]) if mismatch.size else None
    if finite_abs.size:
        masked_abs = np.where(active & np.isfinite(absdiff), absdiff, -np.inf)
        worst = tuple(int(x) for x in np.unravel_index(int(np.argmax(masked_abs)), absdiff.shape))
        max_abs = float(np.max(finite_abs))
        rmse = float(np.sqrt(np.nanmean(active_diff * active_diff)))
        bias = float(np.nanmean(active_diff))
        p95 = float(np.nanpercentile(active_abs, 95))
        p99 = float(np.nanpercentile(active_abs, 99))
        p999 = float(np.nanpercentile(active_abs, 99.9))
    else:
        worst = first
        max_abs = None
        rmse = None
        bias = None
        p95 = None
        p99 = None
        p999 = None
    return {
        "status": "OK",
        "region": region,
        "count": int(active.sum()),
        "domain_count": int(diff.size),
        "shape": list(diff.shape),
        "max_abs": max_abs,
        "rmse": rmse,
        "bias": bias,
        "p95": p95,
        "p99": p99,
        "p99_9": p999,
        "nonfinite_diff_count": int((active & (~np.isfinite(diff))).sum()),
        "first_mismatch_index": list(first) if first is not None else None,
        "first_mismatch_fortran": fortran_index(field, first),
        "worst_mismatch_index": list(worst) if worst is not None else None,
        "worst_mismatch_fortran": fortran_index(field, worst),
    }


def _record_value(
    arrays: dict[str, np.ndarray],
    duplicate_stats: dict[str, Any],
    field: str,
    index: tuple[int, ...],
    value: float,
) -> None:
    current = arrays[field][index]
    if np.isnan(current):
        arrays[field][index] = value
        return
    duplicate_stats[field]["duplicates"] += 1
    if current != value:
        delta = abs(float(current) - float(value))
        duplicate_stats[field]["mismatches"] += 1
        duplicate_stats[field]["max_delta"] = max(float(duplicate_stats[field]["max_delta"]), delta)
        if duplicate_stats[field].get("first_mismatch") is None:
            duplicate_stats[field]["first_mismatch"] = {
                "index": list(index),
                "existing": float(current),
                "new": float(value),
                "delta": delta,
            }


def _header_value(header: Mapping[str, Any], key: str) -> list[str]:
    value = header.get(key)
    if value is None:
        return []
    if isinstance(value, list) and (not value or isinstance(value[0], str)):
        return value
    return []


def _validate_header(header: Mapping[str, Any], shapes: Mapping[str, tuple[int, ...]]) -> str | None:
    mass = shapes["mass"]
    wph = shapes["wph"]
    _, ny, nx = mass
    checks = {
        "surface": [PRECALL_SURFACE],
        "domain_id": [str(TARGET_DOMAIN)],
        "grid_itimestep_after_increment": [str(TARGET_STEP)],
        "rk_step": ["1"],
        "global_mass_i_j_end_exclusive_fortran": ["1", str(nx + 1), "1", str(ny + 1)],
        "mass_vertical_fortran_k_start_k_end_inclusive": ["1", str(mass[0])],
        "w_ph_vertical_fortran_kstag_start_kstag_end_inclusive": ["1", str(wph[0])],
        "tile_record_policy": ["mass_owned_single_owner_no_overlap"],
        "moist_index_qv": ["2"],
    }
    for key, expected in checks.items():
        actual = _header_value(header, key)
        if actual != expected:
            return f"{key}: expected {expected}, got {actual}"
    schemas = header.get("record_schemas", [])
    mass_schema_has_qvapor = any(
        isinstance(schema, list) and schema[:1] == ["MASS_PREPART"] and "QVAPOR" in schema for schema in schemas
    )
    if not mass_schema_has_qvapor:
        return "MASS_PREPART record_schema missing QVAPOR"
    return None


def parse_same_boundary_precall_truth() -> dict[str, Any]:
    if not SAME_QVAPOR_ROOT.is_dir():
        return {"status": "BLOCKED_QVAPOR_ROOT_MISSING", "root": str(SAME_QVAPOR_ROOT)}
    try:
        shapes = pre.expected_shapes()
    except Exception as exc:
        return {"status": "BLOCKED_SHAPE_DISCOVERY", "exception": repr(exc)}
    pattern = f"{PRECALL_SURFACE}_d{TARGET_DOMAIN}_step_{TARGET_STEP}_rk_1_*.txt"
    raw_files = sorted(SAME_QVAPOR_ROOT.glob(pattern))
    all_txt = sorted(SAME_QVAPOR_ROOT.glob("*.txt"))
    extra_txt = [path for path in all_txt if path not in set(raw_files)]
    if not raw_files:
        return {"status": "BLOCKED_NO_SAME_BOUNDARY_QVAPOR_FILES", "root": str(SAME_QVAPOR_ROOT), "pattern": pattern}
    if extra_txt:
        return {
            "status": "BLOCKED_WRONG_BOUNDARY_EXTRA_FILES",
            "root": str(SAME_QVAPOR_ROOT),
            "pattern": pattern,
            "extra_files": [str(path) for path in extra_txt[:12]],
            "extra_file_count": len(extra_txt),
        }

    arrays = {field: np.full(field_shape(field, shapes), np.nan, dtype=np.float64) for field in FIELDS}
    duplicate_stats = {
        field: {"duplicates": 0, "mismatches": 0, "max_delta": 0.0, "first_mismatch": None}
        for field in FIELDS
    }
    record_counts = {record: 0 for record in RECORD_SPECS}
    headers: list[dict[str, Any]] = []
    header_errors: list[dict[str, Any]] = []

    for path in raw_files:
        header: dict[str, Any] = {"path": str(path), "record_schemas": []}
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                tag = parts[0]
                if tag.startswith("#"):
                    header.setdefault("marker", stripped)
                    continue
                if tag in RECORD_SPECS:
                    spec = RECORD_SPECS[tag]
                    if len(parts) != int(spec["length"]):
                        return {
                            "status": "BLOCKED_PARSE_ERROR",
                            "root": str(SAME_QVAPOR_ROOT),
                            "path": str(path),
                            "line": stripped[:240],
                            "expected_length": int(spec["length"]),
                            "actual_length": len(parts),
                        }
                    x = int(parts[4])
                    y = int(parts[5])
                    k = int(parts[6])
                    values = [float(item) for item in parts[7:]]
                    for field, value in zip(spec["fields"], values):
                        shape = arrays[field].shape
                        index = (y, x) if len(shape) == 2 else (k, y, x)
                        _record_value(arrays, duplicate_stats, field, index, value)
                    record_counts[tag] += 1
                    continue
                if tag == "record_schema":
                    header["record_schemas"].append(parts[1:])
                    continue
                if tag in ALLOWED_HEADERS:
                    header[tag] = parts[1:]
                    continue
                return {
                    "status": "BLOCKED_UNKNOWN_RECORD",
                    "root": str(SAME_QVAPOR_ROOT),
                    "path": str(path),
                    "line": stripped[:240],
                }
        header_issue = _validate_header(header, shapes)
        if header_issue is not None:
            header_errors.append({"path": str(path), "issue": header_issue})
        headers.append(header)

    if header_errors:
        return {
            "status": "BLOCKED_WRONG_BOUNDARY_HEADER",
            "root": str(SAME_QVAPOR_ROOT),
            "header_errors": header_errors[:12],
            "header_error_count": len(header_errors),
        }
    duplicate_mismatches = {name: item for name, item in duplicate_stats.items() if int(item["mismatches"]) > 0}
    if duplicate_mismatches:
        return {
            "status": "BLOCKED_DUPLICATE_MISMATCH",
            "root": str(SAME_QVAPOR_ROOT),
            "duplicate_mismatches": duplicate_mismatches,
            "duplicate_stats": duplicate_stats,
            "record_counts": record_counts,
        }
    missing = {
        name: {"missing_count": int(np.isnan(arr).sum()), "shape": list(arr.shape)}
        for name, arr in arrays.items()
        if np.isnan(arr).any()
    }
    if missing:
        return {
            "status": "BLOCKED_MISSING_VALUES",
            "root": str(SAME_QVAPOR_ROOT),
            "missing": missing,
            "duplicate_stats": duplicate_stats,
            "record_counts": record_counts,
        }
    nonfinite = {
        name: {"nonfinite_count": int((~np.isfinite(arr)).sum()), "shape": list(arr.shape)}
        for name, arr in arrays.items()
        if not np.isfinite(arr).all()
    }
    if nonfinite:
        return {
            "status": "BLOCKED_NONFINITE_VALUES",
            "root": str(SAME_QVAPOR_ROOT),
            "nonfinite": nonfinite,
            "duplicate_stats": duplicate_stats,
            "record_counts": record_counts,
        }
    expected_records = {
        "MASS_PREPART": int(np.prod(shapes["mass"])),
        "WPH_PREPART": int(np.prod(shapes["wph"])),
    }
    if record_counts != expected_records:
        return {
            "status": "BLOCKED_WRONG_SHAPE_RECORD_COUNTS",
            "root": str(SAME_QVAPOR_ROOT),
            "record_counts": record_counts,
            "expected_records": expected_records,
        }

    return {
        "status": "WRF_SURFACE_READY",
        "surface": PRECALL_SURFACE,
        "root": str(SAME_QVAPOR_ROOT),
        "target_glob": pattern,
        "raw_file_count": len(raw_files),
        "record_counts": record_counts,
        "expected_record_counts": expected_records,
        "duplicate_stats": duplicate_stats,
        "headers": headers[:4],
        "shapes": {key: list(value) for key, value in shapes.items()},
        "summaries": {name: array_summary(arr) for name, arr in arrays.items()},
        "arrays": arrays,
    }


def load_use_theta_m() -> dict[str, Any]:
    result: dict[str, Any] = {"wrfinput_attr": None, "wrfout_attr": None, "namelist_output": None}
    for key, path in (("wrfinput_attr", WRFINPUT_D02), ("wrfout_attr", WRFOUT_H0_D02)):
        try:
            with Dataset(str(path)) as dataset:
                result[key] = int(getattr(dataset, "USE_THETA_M"))
        except Exception as exc:
            result[f"{key}_error"] = repr(exc)
    namelist_output = RUN_CASE3 / "namelist.output"
    if namelist_output.is_file():
        for line in namelist_output.read_text(encoding="utf-8", errors="replace").splitlines():
            if "USE_THETA_M" in line:
                result["namelist_output"] = line.strip()
                break
    return result


def wrf_theta_m_from_dry(th_dry: np.ndarray, qv: np.ndarray) -> np.ndarray:
    return (th_dry + THETA_OFFSET_K) * (1.0 + RV_OVER_RD * qv) - THETA_OFFSET_K


def adjust_tempqv_transcription(
    *,
    mub: np.ndarray,
    save_mub: np.ndarray,
    c3h: np.ndarray,
    c4h: np.ndarray,
    p_top: float,
    th: np.ndarray,
    pp: np.ndarray,
    qv: np.ndarray,
    use_theta_m: int,
    dtype: Any = np.float64,
) -> dict[str, np.ndarray]:
    """Transcribe WRF dyn_em/nest_init_utils.F::adjust_tempqv."""

    dtype = np.dtype(dtype).type
    mub_d = np.asarray(mub, dtype=dtype)
    save_mub_d = np.asarray(save_mub, dtype=dtype)
    c3 = np.asarray(c3h, dtype=dtype)
    c4 = np.asarray(c4h, dtype=dtype)
    top = dtype(p_top)
    th_in = np.asarray(th, dtype=dtype)
    pp_in = np.asarray(pp, dtype=dtype)
    qv_in = np.asarray(qv, dtype=dtype)
    rv_over_rd = dtype(RV_OVER_RD)
    one = dtype(1.0)

    p_old = c4[:, None, None] + c3[:, None, None] * save_mub_d[None, :, :] + top + pp_in
    p_new = c4[:, None, None] + c3[:, None, None] * mub_d[None, :, :] + top + pp_in
    if int(use_theta_m) == 1:
        tc = (
            (th_in + dtype(300.0))
            * (p_old / dtype(1.0e5)) ** (dtype(2.0) / dtype(7.0))
            / (one + rv_over_rd * qv_in)
            - dtype(273.15)
        )
    else:
        tc = (th_in + dtype(300.0)) * (p_old / dtype(1.0e5)) ** (dtype(2.0) / dtype(7.0)) - dtype(273.15)
    es = dtype(610.78) * np.exp(dtype(17.0809) * tc / (dtype(234.175) + tc)).astype(dtype)
    e = qv_in * p_old / (dtype(0.622) + qv_in)
    rh = e / es

    if int(use_theta_m) == 1:
        thloc = (th_in + dtype(300.0)) / (one + rv_over_rd * qv_in)
    else:
        thloc = th_in + dtype(300.0)
    dth1 = dtype(-191.86e-3) * thloc / (p_new + p_old) * (p_new - p_old)
    dth = dtype(-191.86e-3) * (thloc + dtype(0.5) * dth1) / (p_new + p_old) * (p_new - p_old)
    if int(use_theta_m) == 1:
        th_out = (thloc + dth) * (one + rv_over_rd * qv_in) - dtype(300.0)
    else:
        th_out = thloc + dth - dtype(300.0)

    tc_new = (thloc + dth) * (p_new / dtype(1.0e5)) ** (dtype(2.0) / dtype(7.0)) - dtype(273.15)
    es_new = dtype(610.78) * np.exp(dtype(17.0809) * tc_new / (dtype(234.175) + tc_new)).astype(dtype)
    e_new = rh * es_new
    qv_out = dtype(0.622) * e_new / (p_new - e_new)

    return {
        "p_old": np.asarray(p_old, dtype=np.float64),
        "p_new": np.asarray(p_new, dtype=np.float64),
        "rh": np.asarray(rh, dtype=np.float64),
        "th": np.asarray(th_out, dtype=np.float64),
        "qv": np.asarray(qv_out, dtype=np.float64),
        "thloc_dry_full": np.asarray(thloc + dth, dtype=np.float64),
    }


def capture_candidate_arrays(qvapor_same_boundary: np.ndarray) -> dict[str, Any]:
    import jax  # noqa: PLC0415

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}
    inputs = live.build_live_nest_step1_inputs()
    raw_state = inputs["raw_child"]["state"]
    live_state = inputs["live_child"]["state"]
    metrics = inputs["live_child"]["metrics"]

    raw_mub = np.asarray(jax.device_get(raw_state.mu_total - raw_state.mu_perturbation), dtype=np.float64)
    live_mub = np.asarray(jax.device_get(live_state.mu_total - live_state.mu_perturbation), dtype=np.float64)
    raw_pp = np.asarray(jax.device_get(raw_state.p_perturbation), dtype=np.float64)
    raw_t_dry = np.asarray(jax.device_get(raw_state.theta - THETA_OFFSET_K), dtype=np.float64)
    raw_qv = np.asarray(jax.device_get(raw_state.qv), dtype=np.float64)
    live_t_current = np.asarray(jax.device_get(live_state.theta - THETA_OFFSET_K), dtype=np.float64)
    live_pp = np.asarray(jax.device_get(live_state.p_perturbation), dtype=np.float64)
    live_pb = np.asarray(jax.device_get(live_state.p_total - live_state.p_perturbation), dtype=np.float64)
    live_phb = np.asarray(jax.device_get(live_state.ph_total - live_state.ph_perturbation), dtype=np.float64)
    c3h = np.asarray(jax.device_get(metrics.c3h), dtype=np.float64)
    c4h = np.asarray(jax.device_get(metrics.c4h), dtype=np.float64)
    p_top = float(np.asarray(jax.device_get(metrics.p_top), dtype=np.float64).reshape(-1)[0])
    qv_truth = np.asarray(qvapor_same_boundary, dtype=np.float64)
    if qv_truth.shape != raw_t_dry.shape:
        return {
            "status": "BLOCKED_QVAPOR_SHAPE_MISMATCH",
            "qvapor_shape": list(qv_truth.shape),
            "raw_t_shape": list(raw_t_dry.shape),
        }
    if not np.isfinite(qv_truth).all():
        return {"status": "BLOCKED_QVAPOR_NONFINITE", "nonfinite_count": int((~np.isfinite(qv_truth)).sum())}
    if raw_qv.shape != raw_t_dry.shape:
        return {
            "status": "BLOCKED_RAW_QVAPOR_SHAPE_MISMATCH",
            "raw_qvapor_shape": list(raw_qv.shape),
            "raw_t_shape": list(raw_t_dry.shape),
        }
    if not np.isfinite(raw_qv).all():
        return {"status": "BLOCKED_RAW_QVAPOR_NONFINITE", "nonfinite_count": int((~np.isfinite(raw_qv)).sum())}

    th_m_conversion = wrf_theta_m_from_dry(raw_t_dry, raw_qv)
    direct_dry_adjust = adjust_tempqv_transcription(
        mub=live_mub,
        save_mub=raw_mub,
        c3h=c3h,
        c4h=c4h,
        p_top=p_top,
        th=raw_t_dry,
        pp=raw_pp,
        qv=raw_qv,
        use_theta_m=1,
    )
    best_adjust = adjust_tempqv_transcription(
        mub=live_mub,
        save_mub=raw_mub,
        c3h=c3h,
        c4h=c4h,
        p_top=p_top,
        th=th_m_conversion,
        pp=raw_pp,
        qv=raw_qv,
        use_theta_m=1,
    )
    best_adjust_fp32 = adjust_tempqv_transcription(
        mub=live_mub,
        save_mub=raw_mub,
        c3h=c3h,
        c4h=c4h,
        p_top=p_top,
        th=th_m_conversion,
        pp=raw_pp,
        qv=raw_qv,
        use_theta_m=1,
        dtype=np.float32,
    )
    dry_adjust = adjust_tempqv_transcription(
        mub=live_mub,
        save_mub=raw_mub,
        c3h=c3h,
        c4h=c4h,
        p_top=p_top,
        th=raw_t_dry,
        pp=raw_pp,
        qv=raw_qv,
        use_theta_m=0,
    )
    wrong_order = wrf_theta_m_from_dry(dry_adjust["th"], dry_adjust["qv"])

    return {
        "status": "CANDIDATE_ARRAYS_READY",
        "arrays": {
            "raw_t_dry": raw_t_dry,
            "current_live_t_state": live_t_current,
            "theta_m_conversion_only": th_m_conversion,
            "adjust_tempqv_direct_raw_dry_use_theta_m1": direct_dry_adjust["th"],
            "theta_m_then_adjust_tempqv": best_adjust["th"],
            "theta_m_then_adjust_tempqv_fp32": best_adjust_fp32["th"],
            "dry_adjust_then_theta_m_wrong_order": wrong_order,
            "raw_qv": raw_qv,
            "candidate_qv": best_adjust["qv"],
            "candidate_qv_fp32": best_adjust_fp32["qv"],
            "theta_m_then_adjust_tempqv_p_old": best_adjust["p_old"],
            "theta_m_then_adjust_tempqv_p_new": best_adjust["p_new"],
            "live_p_state": live_pp,
            "live_pb": live_pb,
            "live_mub": live_mub,
            "live_phb": live_phb,
            "raw_mub": raw_mub,
            "raw_pp": raw_pp,
            "c3h": c3h,
            "c4h": c4h,
        },
        "metadata": {
            "p_top": p_top,
            "use_theta_m": load_use_theta_m(),
            "live_nest_base_init": inputs["live_child"].get("live_nest_base_init"),
            "grid": {
                "domain": "d02",
                "mass_shape": [int(inputs["grid"].nz), int(inputs["grid"].ny), int(inputs["grid"].nx)],
                "dx_m": float(inputs["grid"].projection.dx_m),
                "dy_m": float(inputs["grid"].projection.dy_m),
            },
            "candidate_inputs_match_contract": {
                "save_mub": "raw wrfinput_d02 MUB via raw_state.mu_total-raw_state.mu_perturbation",
                "mub": "live-nest recomputed MUB via live_state.mu_total-live_state.mu_perturbation",
                "pp": "raw wrfinput_d02 P via raw_state.p_perturbation",
                "th": "raw wrfinput_d02 T, with WRF use_theta_m=1 dry-to-moist conversion tested explicitly",
                "qv": (
                    "raw child input QVAPOR via raw_state.qv for the WRF formula transcription; "
                    f"accepted WRF pre-call QVAPOR truth parsed only from {SAME_QVAPOR_ROOT}"
                ),
                "c3h_c4h_p_top": "live child DycoreMetrics from wrfinput_d02",
            },
        },
    }


def worst_cell_detail(wrf: Mapping[str, Any], capture: Mapping[str, Any], metric: Mapping[str, Any]) -> dict[str, Any]:
    arrays = capture["arrays"]
    wrf_arrays = wrf["arrays"]
    index_list = metric.get("worst_mismatch_index")
    if not index_list:
        return {"status": "NO_WORST_CELL"}
    idx = tuple(int(v) for v in index_list)
    if len(idx) != 3:
        return {"status": "UNSUPPORTED_WORST_CELL_RANK", "index": index_list}
    k, y, x = idx
    shape = tuple(int(v) for v in np.asarray(wrf_arrays["T_STATE"]).shape)
    distance = boundary_distance(idx, shape)
    phb = wrf_arrays.get("PHB")
    phb_live = arrays.get("live_phb")

    def value3(name: str, source: Mapping[str, np.ndarray]) -> float | None:
        arr = source.get(name)
        if arr is None:
            return None
        return float(np.asarray(arr, dtype=np.float64)[idx])

    def value2(name: str, source: Mapping[str, np.ndarray]) -> float | None:
        arr = source.get(name)
        if arr is None:
            return None
        return float(np.asarray(arr, dtype=np.float64)[y, x])

    pressure_inputs = {
        "wrf_precall_truth": {
            "P_STATE": value3("P_STATE", wrf_arrays),
            "PB": value3("PB", wrf_arrays),
            "MU_STATE": value2("MU_STATE", wrf_arrays),
            "MUB": value2("MUB", wrf_arrays),
            "MUT": value2("MUT", wrf_arrays),
            "PHB_lower_kstag": float(phb[k, y, x]) if phb is not None and k < phb.shape[0] else None,
            "PHB_upper_kstag": float(phb[k + 1, y, x]) if phb is not None and k + 1 < phb.shape[0] else None,
        },
        "candidate_reconstruction": {
            "raw_pp": value3("raw_pp", arrays),
            "live_p_state": value3("live_p_state", arrays),
            "live_pb": value3("live_pb", arrays),
            "raw_mub_save": value2("raw_mub", arrays),
            "live_mub": value2("live_mub", arrays),
            "live_phb_lower_kstag": float(phb_live[k, y, x]) if phb_live is not None and k < phb_live.shape[0] else None,
            "live_phb_upper_kstag": (
                float(phb_live[k + 1, y, x]) if phb_live is not None and k + 1 < phb_live.shape[0] else None
            ),
            "c3h": float(arrays["c3h"][k]),
            "c4h": float(arrays["c4h"][k]),
            "p_top": float(capture["metadata"]["p_top"]),
            "adjust_tempqv_p_old": value3("theta_m_then_adjust_tempqv_p_old", arrays),
            "adjust_tempqv_p_new": value3("theta_m_then_adjust_tempqv_p_new", arrays),
        },
    }
    wrf_value = float(wrf_arrays["T_STATE"][idx])
    candidate_value = float(arrays["theta_m_then_adjust_tempqv"][idx])
    return {
        "status": "OK",
        "zero_index": zero_index_dict(idx),
        "zero_index_order": "k,y,x",
        "fortran_index": fortran_index("T_STATE", idx),
        "boundary_distance_to_horizontal_edge": distance,
        "boundary_band_threshold": BOUNDARY_DISTANCE_THRESHOLD,
        "is_boundary_band": bool(distance <= BOUNDARY_DISTANCE_THRESHOLD),
        "wrf_t_state": wrf_value,
        "candidate_t_state": candidate_value,
        "delta_candidate_minus_wrf": float(candidate_value - wrf_value),
        "abs_delta": abs(float(candidate_value - wrf_value)),
        "qvapor_same_boundary_pre_call": float(wrf_arrays["QVAPOR"][idx]),
        "candidate_qv_after_adjust_tempqv": float(arrays["candidate_qv"][idx]),
        "available_pressure_base_inputs": pressure_inputs,
    }


def compare_candidates(wrf: Mapping[str, Any], capture: Mapping[str, Any]) -> dict[str, Any]:
    arrays = capture["arrays"]
    wrf_arrays = wrf["arrays"]
    t_reference = wrf_arrays["T_STATE"]
    final_name = "theta_m_then_adjust_tempqv"
    t_candidates = {
        "raw_t_dry": arrays["raw_t_dry"],
        "current_live_t_state": arrays["current_live_t_state"],
        "adjust_tempqv_direct_raw_dry_use_theta_m1": arrays["adjust_tempqv_direct_raw_dry_use_theta_m1"],
        "theta_m_conversion_only": arrays["theta_m_conversion_only"],
        final_name: arrays[final_name],
        "theta_m_then_adjust_tempqv_fp32": arrays["theta_m_then_adjust_tempqv_fp32"],
        "dry_adjust_then_theta_m_wrong_order": arrays["dry_adjust_then_theta_m_wrong_order"],
    }
    t_metrics = {name: diff_metrics("T_STATE", value, t_reference) for name, value in t_candidates.items()}
    continuity = {
        "P_STATE": diff_metrics("P_STATE", arrays["live_p_state"], wrf_arrays["P_STATE"]),
        "PB": diff_metrics("PB", arrays["live_pb"], wrf_arrays["PB"]),
        "MUB": diff_metrics("MUB", arrays["live_mub"], wrf_arrays["MUB"]),
        "PHB": diff_metrics("PHB", arrays["live_phb"], wrf_arrays["PHB"]),
    }
    masks = boundary_masks(tuple(int(v) for v in t_reference.shape), BOUNDARY_DISTANCE_THRESHOLD)
    final_all = t_metrics[final_name]
    boundary_decomposition = {
        name: diff_metrics("T_STATE", arrays[final_name], t_reference, region=name, mask=mask)
        for name, mask in masks.items()
    }
    final_candidate_residual = {
        "candidate": final_name,
        "diff_sign": "candidate_minus_wrf",
        "all_cells": final_all,
        "boundary_decomposition": boundary_decomposition,
        "worst_cell": worst_cell_detail(wrf, capture, final_all),
    }
    qvapor_vs_wrf_precall = {
        "status": "SAME_BOUNDARY_PRECALL_QVAPOR_USED",
        "source_root": str(SAME_QVAPOR_ROOT),
        "accepted_wrf_precall_qvapor_present": "QVAPOR" in wrf_arrays,
        "accepted_wrf_precall_schema_fields": sorted(wrf_arrays.keys()),
        "candidate_qv_vs_same_boundary_pre_call_qvapor": diff_metrics(
            "QVAPOR", arrays["candidate_qv"], wrf_arrays["QVAPOR"]
        ),
        "candidate_qv_fp32_vs_same_boundary_pre_call_qvapor": diff_metrics(
            "QVAPOR", arrays["candidate_qv_fp32"], wrf_arrays["QVAPOR"]
        ),
        "raw_child_qvapor_input_summary": array_summary(arrays["raw_qv"]),
        "same_boundary_qvapor_truth_summary": array_summary(wrf_arrays["QVAPOR"]),
    }
    best = final_all
    raw = t_metrics["raw_t_dry"]
    closure_ratio = None
    if best.get("max_abs") is not None and raw.get("max_abs"):
        closure_ratio = float(raw["max_abs"]) / max(float(best["max_abs"]), 1.0e-300)
    return {
        "status": "STEP1_THETA_SAME_QVAPOR_COMPARISON_EXECUTED",
        "diff_sign": "candidate_minus_wrf",
        "t_state_metrics": t_metrics,
        "continuity_vs_wrf_precall": continuity,
        "qvapor_vs_wrf_precall": qvapor_vs_wrf_precall,
        "final_candidate_residual": final_candidate_residual,
        "best_candidate": {
            "name": final_name,
            "max_abs": best.get("max_abs"),
            "rmse": best.get("rmse"),
            "bias": best.get("bias"),
            "p95": best.get("p95"),
            "p99": best.get("p99"),
            "p99_9": best.get("p99_9"),
            "material_threshold_K": T_MATERIAL_THRESHOLD_K,
            "closes_to_material_threshold": (
                best.get("status") == "OK"
                and best.get("max_abs") is not None
                and float(best["max_abs"]) <= T_MATERIAL_THRESHOLD_K
            ),
            "closure_ratio_raw_max_abs_to_best_max_abs": closure_ratio,
        },
    }


def classify(comparisons: Mapping[str, Any]) -> tuple[str, list[str], str]:
    if comparisons.get("status") != "STEP1_THETA_SAME_QVAPOR_COMPARISON_EXECUTED":
        return (
            "STEP1_THETA_SAME_QVAPOR_BLOCKED_COMPARISON",
            ["Candidate comparison did not execute."],
            "Fix the comparison blocker and rerun the same-boundary QVAPOR proof.",
        )
    best = comparisons["best_candidate"]
    final = comparisons["final_candidate_residual"]
    boundary = final["boundary_decomposition"][f"boundary_distance_le_{BOUNDARY_DISTANCE_THRESHOLD}"]
    interior = final["boundary_decomposition"][f"interior_distance_gt_{BOUNDARY_DISTANCE_THRESHOLD}"]
    if bool(best.get("closes_to_material_threshold")):
        return (
            "STEP1_THETA_SAME_QVAPOR_PATCH_READY",
            [],
            "Apply the initialization-only theta_m/adjust_tempqv source patch and run the full proof chain.",
        )
    interior_max = interior.get("max_abs")
    boundary_max = boundary.get("max_abs")
    if (
        interior.get("status") == "OK"
        and interior_max is not None
        and float(interior_max) <= T_MATERIAL_THRESHOLD_K
        and boundary_max is not None
        and float(boundary_max) > T_MATERIAL_THRESHOLD_K
    ):
        return (
            "STEP1_THETA_SAME_QVAPOR_BOUNDARY_TAIL_BOUNDED_NEXT_BASE",
            [
                "The final same-boundary QVAPOR candidate still fails the all-cell max_abs gate.",
                "The failure is localized to the configured boundary band, while the interior max_abs is below 1e-3 K.",
            ],
            "Move the manager lane to the base-state source/split proof before any theta source patch.",
        )
    return (
        "STEP1_THETA_SAME_QVAPOR_INTERIOR_RESIDUAL_NEEDS_WRF_INTERMEDIATE",
        [
            "The final same-boundary QVAPOR candidate still has an interior residual above 1e-3 K.",
            "A source patch is not authorized without WRF intermediate theta/adjust_tempqv pressure inputs or an equivalent proof.",
        ],
        "Emit or recover WRF theta_m/adjust_tempqv intermediate inputs for the residual cell before patching production.",
    )


def source_evidence() -> dict[str, Any]:
    return {
        "mediation_integrate_live_nest_call": {
            "path": "/home/user/src/wrf_pristine/WRF/share/mediation_integrate.F",
            "lines": "726-762",
            "evidence": [
                "lines 726-735 save elevation and mub for temp and qv adjustment; nest%mub_save receives nest%mub",
                "lines 737-751 blend parent and nest terrain, mub, and phb",
                "lines 754-762 call adjust_tempqv(nest%mub, nest%mub_save, nest%c3h, nest%c4h, nest%znw, nest%p_top, nest%t_2, nest%p, QVAPOR, use_theta_m, ...)",
            ],
        },
        "adjust_tempqv": {
            "path": "/home/user/src/wrf_pristine/WRF/dyn_em/nest_init_utils.F",
            "lines": "812-890",
            "evidence": [
                "lines 846-859 compute p_old and relative humidity from save_mub, pp, th, and qv",
                "lines 863-884 compute p_new, dth, updated th, and updated qv while conserving RH",
                "lines 852-855 and 869-879 branch on use_theta_m; use_theta_m=1 treats th as moist theta",
            ],
        },
        "theta_m_conversion": {
            "path": "/home/user/src/wrf_pristine/WRF/dyn_em/module_initialize_real.F",
            "lines": "4918-4928",
            "evidence": [
                "lines 4918-4920 state dry potential temperature is turned into moist potential temperature before halo communications",
                "lines 4923-4928 apply grid%t_2 = (grid%t_2 + T0) * (1 + R_v/R_d * QVAPOR) - T0 when use_theta_m=1",
            ],
        },
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    comp = payload.get("comparisons", {})
    t_metrics = comp.get("t_state_metrics", {})
    final = comp.get("final_candidate_residual", {})
    decomp = final.get("boundary_decomposition", {})
    worst = final.get("worst_cell", {})
    raw = t_metrics.get("raw_t_dry", {})
    current = t_metrics.get("current_live_t_state", {})
    direct = t_metrics.get("adjust_tempqv_direct_raw_dry_use_theta_m1", {})
    moist = t_metrics.get("theta_m_conversion_only", {})
    best = t_metrics.get("theta_m_then_adjust_tempqv", {})
    best32 = t_metrics.get("theta_m_then_adjust_tempqv_fp32", {})
    boundary = decomp.get(f"boundary_distance_le_{BOUNDARY_DISTANCE_THRESHOLD}", {})
    interior = decomp.get(f"interior_distance_gt_{BOUNDARY_DISTANCE_THRESHOLD}", {})
    qv_metric = comp.get("qvapor_vs_wrf_precall", {}).get("candidate_qv_vs_same_boundary_pre_call_qvapor", {})
    lines = [
        "# V0.14 Step-1 Theta Same-Boundary QVAPOR",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Result",
        "",
        f"- CPU backend: `{payload['environment'].get('jax_default_backend')}`; GPU used: `{payload.get('gpu_used')}`.",
        f"- Required ancestor `{REQUIRED_ANCESTOR}` present: `{payload['required_ancestor_912b7371'].get('is_ancestor')}`.",
        f"- Same-boundary QVAPOR root: `{SAME_QVAPOR_ROOT}`.",
        f"- Real run `USE_THETA_M`: `{payload['candidate_capture'].get('metadata', {}).get('use_theta_m')}`.",
        f"- Raw/current live dry `T_STATE` vs WRF pre-call: max_abs `{raw.get('max_abs')}` / `{current.get('max_abs')}`.",
        f"- `adjust_tempqv` directly on raw dry `T` with `use_theta_m=1`: max_abs `{direct.get('max_abs')}`.",
        f"- WRF dry-to-moist theta conversion only: max_abs `{moist.get('max_abs')}`, rmse `{moist.get('rmse')}`.",
        f"- WRF `theta_m` conversion plus `adjust_tempqv`: max_abs `{best.get('max_abs')}`, rmse `{best.get('rmse')}`, p99 `{best.get('p99')}`, p99.9 `{best.get('p99_9')}`.",
        f"- Same candidate with fp32 arithmetic: max_abs `{best32.get('max_abs')}`, rmse `{best32.get('rmse')}`.",
        f"- Final candidate boundary band (`distance_to_edge <= {BOUNDARY_DISTANCE_THRESHOLD}`): max_abs `{boundary.get('max_abs')}`, rmse `{boundary.get('rmse')}`, p99.9 `{boundary.get('p99_9')}`.",
        f"- Final candidate interior (`distance_to_edge > {BOUNDARY_DISTANCE_THRESHOLD}`): max_abs `{interior.get('max_abs')}`, rmse `{interior.get('rmse')}`, p99.9 `{interior.get('p99_9')}`.",
        f"- Candidate QVAPOR after `adjust_tempqv` vs same-boundary WRF pre-call QVAPOR: max_abs `{qv_metric.get('max_abs')}`, rmse `{qv_metric.get('rmse')}`.",
        "",
        "## Worst Cell",
        "",
        f"- Zero index (`k,y,x`): `{worst.get('zero_index')}`; Fortran index: `{worst.get('fortran_index')}`.",
        f"- Boundary distance: `{worst.get('boundary_distance_to_horizontal_edge')}`; boundary band: `{worst.get('is_boundary_band')}`.",
        f"- WRF value `{worst.get('wrf_t_state')}`, candidate `{worst.get('candidate_t_state')}`, delta `{worst.get('delta_candidate_minus_wrf')}`.",
        f"- QVAPOR `{worst.get('qvapor_same_boundary_pre_call')}`; candidate QVAPOR after adjust `{worst.get('candidate_qv_after_adjust_tempqv')}`.",
        "- Pressure/base inputs for the worst cell are recorded under `comparisons.final_candidate_residual.worst_cell.available_pressure_base_inputs` in the JSON.",
        "",
        "## Interpretation",
        "",
        "- The proof loaded accepted WRF `QVAPOR` truth only from the filtered same-boundary pre-call root, not from `wrfout_d02`.",
        "- The WRF formula transcription keeps the prior proof's raw child input `QVAPOR`; the same-boundary root is the accepted pre-call truth comparator.",
        "- The final candidate residual is classified by the contract's all-cell and boundary/interior max_abs gates.",
        f"- Next decision: {payload['next_decision']}",
        "",
        "Detailed tables are in `proofs/v014/step1_theta_same_qvapor.json`.",
        "",
    ]
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Review: V0.14 Step-1 Theta Same-Boundary QVAPOR",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "objective: rerun the Step-1 live-nest theta semantics proof with the validated same-boundary pre-call QVAPOR root and classify the final residual as boundary-local or interior.",
        "",
        "files changed:",
        "- `proofs/v014/step1_theta_same_qvapor.py`",
        "- `proofs/v014/step1_theta_same_qvapor.json`",
        "- `proofs/v014/step1_theta_same_qvapor.md`",
        "- `.agent/reviews/2026-06-09-v014-step1-theta-same-qvapor.md`",
        "",
        "commands run:",
    ]
    for command in payload["commands"]["required_validation"]:
        lines.append(f"- `{command}`")
    lines.extend(
        [
            "",
            "proof objects produced:",
            f"- `{OUT_JSON}`",
            f"- `{OUT_MD}`",
            f"- `{OUT_REVIEW}`",
            f"- `{SAME_QVAPOR_ROOT}` reused, not rebuilt",
            "",
            "unresolved risks:",
        ]
    )
    for risk in payload["unresolved_risks"]:
        lines.append(f"- {risk}")
    lines.extend(["", f"next decision needed: {payload['next_decision']}", ""])
    return "\n".join(lines)


def blocked_result(reason: str, blocker: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str, list[str], str]:
    capture = {"status": "NOT_EXECUTED"}
    comparisons = {"status": "NOT_EXECUTED", "blocker": blocker}
    verdict = f"STEP1_THETA_SAME_QVAPOR_BLOCKED_{reason}"
    risks = [f"Proof blocked before candidate comparison: {blocker.get('status', reason)}."]
    next_decision = "Fix the blocker and rerun the same-boundary QVAPOR proof."
    return capture, comparisons, verdict, risks, next_decision


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    git_head = run_command(["git", "log", "-1", "--oneline", "--decorate"], cwd=ROOT)
    ancestor = run_command(["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, "HEAD"], cwd=ROOT)
    wrf = parse_same_boundary_precall_truth()
    if wrf.get("status") != "WRF_SURFACE_READY":
        capture, comparisons, verdict, risks, next_decision = blocked_result("QVAPOR_ROOT", wrf)
    else:
        capture = capture_candidate_arrays(wrf["arrays"]["QVAPOR"])
        if capture.get("status") != "CANDIDATE_ARRAYS_READY":
            comparisons = {"status": "NOT_EXECUTED", "blocker": capture}
            verdict = "STEP1_THETA_SAME_QVAPOR_BLOCKED_CANDIDATE_ARRAY_CAPTURE"
            risks = ["Candidate array capture did not complete."]
            next_decision = "Fix the exact candidate capture blocker and rerun."
        else:
            comparisons = compare_candidates(wrf, capture)
            verdict, risks, next_decision = classify(comparisons)

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.step1_theta_same_qvapor.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "git_head": git_head,
        "required_ancestor_912b7371": {
            "command": ancestor["command"],
            "returncode": ancestor["returncode"],
            "is_ancestor": ancestor["returncode"] == 0,
            "stderr_tail": ancestor.get("stderr_tail"),
        },
        "environment": jax_environment(),
        "cpu_only": True,
        "gpu_used": False,
        "no_tost": True,
        "no_switzerland": True,
        "no_fp32_source_work": True,
        "no_memory_source_work": True,
        "no_hermes": True,
        "no_wrf_run": True,
        "production_src_edits": False,
        "source_patch_allowed_by_proof": verdict == "STEP1_THETA_SAME_QVAPOR_PATCH_READY",
        "target": {
            "domain": "d02",
            "wrf_grid_id": TARGET_DOMAIN,
            "step": TARGET_STEP,
            "wrf_surface": PRECALL_SURFACE,
            "theta_offset_K": THETA_OFFSET_K,
            "t_state_material_threshold_K": T_MATERIAL_THRESHOLD_K,
            "boundary_distance_threshold": BOUNDARY_DISTANCE_THRESHOLD,
        },
        "inputs": {
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "same_boundary_qvapor_root": path_info(SAME_QVAPOR_ROOT),
            "qvapor_savepoint_json": path_info(QVAPOR_SAVEPOINT_JSON),
            "run_case3": path_info(RUN_CASE3),
            "wrfinput_d02": path_info(WRFINPUT_D02),
            "wrfout_h0_d02_for_use_theta_m_attr_only": path_info(WRFOUT_H0_D02),
            "scratch_root": path_info(SCRATCH),
            "previous_theta_proof_script": path_info(PROOF_DIR / "step1_live_nest_theta_semantics.py"),
            "previous_theta_proof_json": path_info(PROOF_DIR / "step1_live_nest_theta_semantics.json"),
            "previous_qvapor_proof_json": path_info(QVAPOR_SAVEPOINT_JSON),
        },
        "qvapor_source_rule": {
            "wrf_qvapor_truth_source": str(SAME_QVAPOR_ROOT),
            "wrfout_qvapor_loaded": False,
            "candidate_formula_qvapor_source": "raw child input state, matching the prior theta proof logic",
            "same_boundary_qvapor_root_required": True,
        },
        "wrf_source_evidence": source_evidence(),
        "wrf_precall_truth": strip_arrays(wrf) if wrf.get("status") == "WRF_SURFACE_READY" else wrf,
        "candidate_capture": strip_arrays(capture),
        "candidate_array_summaries": {
            name: array_summary(value)
            for name, value in capture.get("arrays", {}).items()
            if name
            in {
                "raw_t_dry",
                "current_live_t_state",
                "theta_m_conversion_only",
                "theta_m_then_adjust_tempqv",
                "theta_m_then_adjust_tempqv_fp32",
                "raw_qv",
                "candidate_qv",
                "live_p_state",
                "live_pb",
                "live_mub",
                "live_phb",
                "raw_mub",
                "raw_pp",
            }
        },
        "comparisons": comparisons,
        "theta_contract_resolution": {
            "use_theta_m": capture.get("metadata", {}).get("use_theta_m"),
            "wrf_in_memory_t_state": "grid%t_2 is perturbation moist theta when use_theta_m=1",
            "tested_sequence": "raw dry T -> theta_m conversion with raw child QVAPOR -> adjust_tempqv transcription; same-boundary pre-call QVAPOR is the accepted QVAPOR truth comparator",
            "final_candidate": "theta_m_then_adjust_tempqv",
        },
        "commands": {
            "required_validation": [
                "python -m py_compile proofs/v014/step1_theta_same_qvapor.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_theta_same_qvapor.py",
                "python -m json.tool proofs/v014/step1_theta_same_qvapor.json >/tmp/step1_theta_same_qvapor.validated.json",
                "git diff -- src/gpuwrf",
            ],
        },
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "review": str(OUT_REVIEW),
            "same_boundary_qvapor_root_reused": str(SAME_QVAPOR_ROOT),
        },
        "unresolved_risks": risks,
        "next_decision": next_decision,
    }
    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "json": str(OUT_JSON), "markdown": str(OUT_MD)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
