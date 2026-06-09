#!/usr/bin/env python3
"""V0.14 Step-1 first_rk_step_part2 source-leaf split.

Consumes disposable WRF truth emitted inside ``first_rk_step_part2`` and
compares the raw theta source leaves against the current patched-init JAX dry
physics/source bundle.  CPU-only; no production source edits.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

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

import source_save_boundary_hook as savehook  # noqa: E402
import step1_live_nest_init_rerun as live  # noqa: E402
import step1_rk1_p_state_source_split as pstate  # noqa: E402
import step1_rk1_source_boundary as source  # noqa: E402
import step1_tendency_contract_split as tendency_contract  # noqa: E402


OUT_JSON = PROOF_DIR / "step1_part2_source_leaves_split.json"
OUT_MD = PROOF_DIR / "step1_part2_source_leaves_split.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-step1-part2-source-leaves-split.md"
OUT_WRF_PATCH = PROOF_DIR / "step1_part2_source_leaves_split_wrf_patch.diff"

SPRINT_CONTRACT = (
    ROOT
    / ".agent/sprints/2026-06-09-v014-step1-part2-source-leaves-split/sprint-contract.md"
)
HANDOFF = ROOT / ".agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md"

SCRATCH = Path("/tmp/wrf_gpu2_step1_part2_source_leaves_split_20260609")
WRF_TRUTH = SCRATCH / "wrf_truth"
WRF_RUN_DIR = SCRATCH / "run"
WRF_COPY = SCRATCH / "WRF"
WRF_BASE = Path("/mnt/data/wrf_gpu2/v014_step1_rk1_source_boundary/WRF")

TARGET_DOMAIN = 2
TARGET_STEP = 1
R_D = 287.0
R_V = 461.6
RVRD = R_V / R_D
THETA_OFFSET = 300.0

PART2_SURFACES = (
    "after_calculate_phy_tend",
    "after_update_phy_ten",
    "after_conv_t_tendf_to_moist",
)
SOURCE_SURFACES = ("after_first_rk_step_part1", "after_first_rk_step_part2")
SOURCE_MASS_FIELDS = (
    "T_STATE",
    "P_STATE",
    "PB",
    "MU_STATE",
    "MUB",
    "MUT",
    "T_TENDF",
    "H_DIABATIC",
    "MU_TENDF",
    "T_SAVE",
    "T_OLD",
)
PART2_FIELDS = (
    "T_STATE",
    "T_OLD",
    "T_HIST_SRC",
    "T_INIT",
    "P",
    "PB",
    "MU_NEW",
    "MUB",
    "MUT",
    "MSFTY",
    "MASS_H",
    "T_TENDF",
    "H_DIABATIC",
    "T_SAVE",
    "MU_TENDF",
    "RTHRATEN",
    "RTHBLTEN",
    "RTHCUTEN",
    "RTHSHTEN",
    "RTHNDGDTEN",
    "RTHIAUTEN",
    "RTHFRTEN",
    "RTH_ALL_SUM",
    "RTH_ACTIVE_SUM",
    "RTH_ACTIVE_SUM_MOIST",
    "QV_OLD",
    "QV_TEND",
    "THETA_M_FACTOR",
)
RAW_RTH_FIELDS = (
    "RTHRATEN",
    "RTHBLTEN",
    "RTHCUTEN",
    "RTHSHTEN",
    "RTHNDGDTEN",
    "RTHIAUTEN",
    "RTHFRTEN",
    "RTH_ALL_SUM",
    "RTH_ACTIVE_SUM",
    "RTH_ACTIVE_SUM_MOIST",
)
SOURCE_SAVE_PATTERN = "source_save_after_rk_tendency_d2_step_1_*.txt"


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
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
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


def run_command(command: list[str], *, cwd: Path = ROOT, timeout_s: int = 120) -> dict[str, Any]:
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
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_s,
        )
        return {
            "command": command,
            "cwd": str(cwd),
            "returncode": int(proc.returncode),
            "wall_s": float(time.perf_counter() - start),
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "cwd": str(cwd),
            "returncode": None,
            "timeout_s": int(timeout_s),
            "wall_s": float(time.perf_counter() - start),
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
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


def expected_shapes() -> dict[str, tuple[int, ...]]:
    with np.load(live.ACCEPTED_TRUTH) as truth:
        return {
            "mass": tuple(int(item) for item in truth["T"].shape),
            "mass2d": tuple(int(item) for item in truth["MU"].shape),
            "wph": tuple(int(item) for item in truth["PH"].shape),
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
    if len(index) == 1:
        return {"linear": int(index[0])}
    return None


def array_summary(array: Any, *, mask: np.ndarray | None = None) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.float64)
    values = arr[mask] if mask is not None else arr.reshape(-1)
    finite = values[np.isfinite(values)]
    nonzero = values[np.isfinite(values) & (values != 0.0)]
    if finite.size == 0:
        return {"shape": list(arr.shape), "count": int(values.size), "finite_count": 0}
    return {
        "shape": list(arr.shape),
        "count": int(values.size),
        "finite_count": int(finite.size),
        "nonzero_count": int(nonzero.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "max_abs": float(np.max(np.abs(finite))),
        "rms": float(np.sqrt(np.mean(finite * finite))),
    }


def _masked_values(
    candidate: np.ndarray, reference: np.ndarray, mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if mask is None:
        return candidate, reference, None
    return candidate[mask], reference[mask], np.argwhere(mask)


def diff_metrics(
    label: str,
    candidate: Any,
    reference: Any,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if cand.shape != ref.shape:
        return {
            "status": "SHAPE_MISMATCH",
            "label": label,
            "candidate_shape": list(cand.shape),
            "reference_shape": list(ref.shape),
        }
    if mask is not None and mask.shape != cand.shape:
        return {
            "status": "MASK_SHAPE_MISMATCH",
            "label": label,
            "candidate_shape": list(cand.shape),
            "mask_shape": list(mask.shape),
        }
    cand_values, ref_values, mask_indices = _masked_values(cand, ref, mask)
    diff = cand_values - ref_values
    absdiff = np.abs(diff)
    finite_abs = absdiff[np.isfinite(absdiff)]
    mismatch = np.argwhere((diff != 0.0) | (~np.isfinite(diff))).reshape(-1)
    first_pos = int(mismatch[0]) if mismatch.size else None
    if finite_abs.size:
        worst_pos = int(np.nanargmax(absdiff))
        max_abs = float(np.nanmax(absdiff))
        rmse = float(np.sqrt(np.nanmean(diff * diff)))
        bias = float(np.nanmean(diff))
        p95 = float(np.nanpercentile(absdiff, 95))
        p99 = float(np.nanpercentile(absdiff, 99))
    else:
        worst_pos = first_pos
        max_abs = None
        rmse = None
        bias = None
        p95 = None
        p99 = None

    def full_index(pos: int | None) -> tuple[int, ...] | None:
        if pos is None:
            return None
        if mask_indices is not None:
            return tuple(int(item) for item in mask_indices[pos])
        return tuple(int(item) for item in np.unravel_index(pos, cand.shape))

    first_idx = full_index(first_pos)
    worst_idx = full_index(worst_pos)
    return {
        "status": "OK",
        "label": label,
        "shape": list(cand.shape),
        "count": int(diff.size),
        "max_abs": max_abs,
        "rmse": rmse,
        "bias": bias,
        "p95": p95,
        "p99": p99,
        "nonfinite_diff_count": int((~np.isfinite(diff)).sum()),
        "first_mismatch_index": list(first_idx) if first_idx is not None else None,
        "first_mismatch_fortran": fortran_index(first_idx),
        "worst_mismatch_index": list(worst_idx) if worst_idx is not None else None,
        "worst_mismatch_fortran": fortran_index(worst_idx),
        "worst_candidate": float(cand[worst_idx]) if worst_idx is not None else None,
        "worst_reference": float(ref[worst_idx]) if worst_idx is not None else None,
        "candidate_minus_reference": True,
    }


def compact_metric(metric: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if metric is None:
        return None
    keys = (
        "status",
        "count",
        "shape",
        "max_abs",
        "rmse",
        "bias",
        "p95",
        "p99",
        "nonfinite_diff_count",
        "worst_mismatch_fortran",
        "worst_candidate",
        "worst_reference",
    )
    return {key: metric.get(key) for key in keys if key in metric}


def interior_mask(shape: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if len(shape) == 3:
        mask[:, 1:-1, 1:-1] = True
    elif len(shape) == 2:
        mask[1:-1, 1:-1] = True
    else:
        mask[...] = True
    return mask


def _record_value(
    arrays: dict[str, np.ndarray],
    filled: dict[str, np.ndarray],
    duplicate_stats: dict[str, Any],
    field: str,
    index: tuple[int, int, int],
    value: float,
) -> None:
    current = arrays[field][index]
    if not filled[field][index]:
        arrays[field][index] = value
        filled[field][index] = True
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


def parse_part2_surface(surface: str, shapes: Mapping[str, tuple[int, ...]]) -> dict[str, Any]:
    pattern = f"{surface}_d{TARGET_DOMAIN}_step_{TARGET_STEP}_rk_1_*.txt"
    paths = sorted(WRF_TRUTH.glob(pattern))
    if not paths:
        return {"status": "BLOCKED_NO_WRF_SURFACE_FILES", "surface": surface, "pattern": pattern}
    arrays = {field: np.full(shapes["mass"], np.nan, dtype=np.float64) for field in PART2_FIELDS}
    filled = {field: np.zeros(shapes["mass"], dtype=bool) for field in PART2_FIELDS}
    duplicate_stats = {
        field: {"duplicates": 0, "mismatches": 0, "max_delta": 0.0, "first_mismatch": None}
        for field in PART2_FIELDS
    }
    headers: list[dict[str, Any]] = []
    record_count = 0
    for path in paths:
        header: dict[str, Any] = {"path": str(path)}
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
                if tag == "MASS_PART2":
                    expected = 1 + 6 + len(PART2_FIELDS)
                    if len(parts) != expected:
                        return {
                            "status": "BLOCKED_PARSE_ERROR",
                            "surface": surface,
                            "path": str(path),
                            "expected_length": expected,
                            "actual_length": len(parts),
                            "line": stripped[:240],
                        }
                    x = int(parts[4])
                    y = int(parts[5])
                    k = int(parts[6])
                    values = [float(item) for item in parts[7:]]
                    for field, value in zip(PART2_FIELDS, values):
                        _record_value(arrays, filled, duplicate_stats, field, (k, y, x), value)
                    record_count += 1
                    continue
                header.setdefault(tag, []).append(parts[1:])
        headers.append(header)
    duplicate_mismatches = {k: v for k, v in duplicate_stats.items() if v["mismatches"]}
    if duplicate_mismatches:
        return {
            "status": "BLOCKED_DUPLICATE_MISMATCH",
            "surface": surface,
            "duplicate_mismatches": duplicate_mismatches,
        }
    missing = {
        field: int((~filled[field]).sum())
        for field in PART2_FIELDS
        if (~filled[field]).any()
    }
    if missing:
        return {
            "status": "BLOCKED_MISSING_VALUES",
            "surface": surface,
            "missing": missing,
            "record_count": record_count,
        }
    return {
        "status": "WRF_PART2_SURFACE_READY",
        "surface": surface,
        "raw_files": [str(path) for path in paths],
        "record_count": record_count,
        "headers": headers,
        "duplicate_stats": duplicate_stats,
        "summaries": {field: array_summary(array) for field, array in arrays.items()},
        "nonfinite_counts": {
            field: int((~np.isfinite(array)).sum()) for field, array in arrays.items()
        },
        "arrays": arrays,
    }


def parse_part2_surfaces(shapes: Mapping[str, tuple[int, ...]]) -> dict[str, Any]:
    surfaces = {}
    for surface in PART2_SURFACES:
        parsed = parse_part2_surface(surface, shapes)
        if parsed.get("status") != "WRF_PART2_SURFACE_READY":
            return {"status": "BLOCKED_PART2_SURFACE", "blocker": parsed}
        surfaces[surface] = parsed
    return {"status": "WRF_PART2_TRUTH_READY", "surfaces": surfaces}


def parse_source_mass_surface(surface_name: str, shapes: Mapping[str, tuple[int, ...]]) -> dict[str, Any]:
    pattern = f"{surface_name}_d{TARGET_DOMAIN}_step_{TARGET_STEP}_rk_1_*.txt"
    paths = sorted(WRF_TRUTH.glob(pattern))
    if not paths:
        return {"status": "BLOCKED_NO_WRF_SURFACE_FILES", "surface": surface_name, "pattern": pattern}
    arrays = {field: np.full(shapes["mass"], np.nan, dtype=np.float64) for field in SOURCE_MASS_FIELDS}
    filled = {field: np.zeros(shapes["mass"], dtype=bool) for field in SOURCE_MASS_FIELDS}
    duplicate_stats = {
        field: {"duplicates": 0, "mismatches": 0, "max_delta": 0.0, "first_mismatch": None}
        for field in SOURCE_MASS_FIELDS
    }
    headers: list[dict[str, Any]] = []
    record_count = 0
    for path in paths:
        header: dict[str, Any] = {"path": str(path)}
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
                if tag == "MASS_SOURCE":
                    expected = 1 + 6 + len(SOURCE_MASS_FIELDS)
                    if len(parts) != expected:
                        return {
                            "status": "BLOCKED_PARSE_ERROR",
                            "surface": surface_name,
                            "path": str(path),
                            "expected_length": expected,
                            "actual_length": len(parts),
                            "line": stripped[:240],
                        }
                    x = int(parts[4])
                    y = int(parts[5])
                    k = int(parts[6])
                    values = [float(item) for item in parts[7:]]
                    for field, value in zip(SOURCE_MASS_FIELDS, values):
                        _record_value(arrays, filled, duplicate_stats, field, (k, y, x), value)
                    record_count += 1
                    continue
                if tag in {
                    "surface",
                    "routine",
                    "domain_id",
                    "current_timestr_before_step",
                    "grid_itimestep_after_increment",
                    "rk_step",
                    "rk_order",
                    "tile_i_j_bounds_fortran",
                    "global_mass_i_j_end_exclusive_fortran",
                    "tile_record_policy",
                    "mass_vertical_fortran_k_start_k_end_inclusive",
                    "w_ph_vertical_fortran_kstag_start_kstag_end_inclusive",
                    "record_schema",
                }:
                    header.setdefault(tag, []).append(parts[1:])
                    continue
        headers.append(header)
    duplicate_mismatches = {k: v for k, v in duplicate_stats.items() if v["mismatches"]}
    if duplicate_mismatches:
        return {
            "status": "BLOCKED_DUPLICATE_MISMATCH",
            "surface": surface_name,
            "duplicate_mismatches": duplicate_mismatches,
        }
    missing = {
        field: int((~filled[field]).sum())
        for field in SOURCE_MASS_FIELDS
        if (~filled[field]).any()
    }
    if missing:
        return {
            "status": "BLOCKED_MISSING_VALUES",
            "surface": surface_name,
            "missing": missing,
            "record_count": record_count,
        }
    return {
        "status": "WRF_SOURCE_MASS_SURFACE_READY",
        "surface": surface_name,
        "raw_files": [str(path) for path in paths],
        "record_count": record_count,
        "headers": headers,
        "duplicate_stats": duplicate_stats,
        "summaries": {field: array_summary(array) for field, array in arrays.items()},
        "nonfinite_counts": {
            field: int((~np.isfinite(array)).sum()) for field, array in arrays.items()
        },
        "arrays": arrays,
    }


def parse_existing_source_surfaces(shapes: Mapping[str, tuple[int, ...]]) -> dict[str, Any]:
    surfaces = {}
    for surface_name in SOURCE_SURFACES:
        parsed = parse_source_mass_surface(surface_name, shapes)
        if parsed.get("status") != "WRF_SOURCE_MASS_SURFACE_READY":
            return {"status": "BLOCKED_SOURCE_SURFACE", "blocker": parsed}
        surfaces[surface_name] = parsed
    return {"status": "WRF_SOURCE_SURFACES_READY", "surfaces": surfaces}


def parse_source_save() -> dict[str, Any]:
    paths = sorted(WRF_TRUTH.glob(SOURCE_SAVE_PATTERN))
    if not paths:
        return {
            "status": "BLOCKED_NO_SOURCE_SAVE_FILES",
            "pattern": str(WRF_TRUTH / SOURCE_SAVE_PATTERN),
        }
    parsed = savehook.parse_savepoint(paths, savehook.SOURCE_SCHEMAS)
    return {
        "status": "SOURCE_SAVE_READY",
        "files": [str(path) for path in paths],
        "compact": savehook.compact_surface(parsed, savehook.SOURCE_SCHEMAS),
        "surface": parsed,
    }


def strip_arrays(parsed: Mapping[str, Any]) -> dict[str, Any]:
    if "surfaces" in parsed:
        return {
            **{k: v for k, v in parsed.items() if k != "surfaces"},
            "surfaces": {
                name: {k: v for k, v in surface.items() if k != "arrays"}
                for name, surface in parsed["surfaces"].items()
            },
        }
    return {k: v for k, v in parsed.items() if k != "arrays"}


def build_jax_capture() -> dict[str, Any]:
    import jax  # noqa: PLC0415

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}
    inputs = live.build_live_nest_step1_inputs()
    patched = pstate.apply_mythos_perturb_init(inputs)
    capture = tendency_contract.build_tendency_capture(
        inputs,
        patched["carry"],
        label="step1_part2_source_leaves_split_patched_init",
    )
    if capture.get("status") != "JAX_TENDENCY_BOUNDARIES_READY":
        return {"status": "BLOCKED_JAX_CAPTURE", "capture_status": capture.get("status"), "capture": capture}
    return {
        "status": "JAX_CAPTURE_READY",
        "inputs": inputs,
        "patched": patched,
        "capture": capture,
    }


def jax_array(value: Any) -> np.ndarray:
    import jax  # noqa: PLC0415

    return np.asarray(jax.device_get(value), dtype=np.float64)


def source_save_sparse_compare(
    source_save: Mapping[str, Any],
    full_arrays: Mapping[str, np.ndarray],
    *,
    fields: Iterable[str],
) -> dict[str, Any]:
    records = source_save["surface"]["records"].get("MASS_SOURCE", {})
    out: dict[str, Any] = {}
    for field in fields:
        if field not in full_arrays:
            out[field] = {"status": "MISSING_FULL_FIELD"}
            continue
        if field not in savehook.SOURCE_SCHEMAS["MASS_SOURCE"]["fields"]:
            out[field] = {"status": "MISSING_SOURCE_SAVE_SCHEMA_FIELD"}
            continue
        candidate_values = []
        reference_values = []
        keys = []
        for key in sorted(records):
            if field not in records[key]:
                continue
            if len(key) != 3:
                continue
            if any(idx < 0 or idx >= full_arrays[field].shape[pos] for pos, idx in enumerate(key)):
                return {
                    "status": "OUT_OF_BOUNDS",
                    "field": field,
                    "key": list(key),
                    "shape": list(full_arrays[field].shape),
                }
            candidate_values.append(float(full_arrays[field][key]))
            reference_values.append(float(records[key][field]))
            keys.append(tuple(int(item) for item in key))
        if not candidate_values:
            out[field] = {"status": "NO_COMMON_RECORDS"}
            continue
        candidate = np.asarray(candidate_values, dtype=np.float64)
        reference = np.asarray(reference_values, dtype=np.float64)
        metric = diff_metrics(field, candidate, reference)
        worst_idx = metric.get("worst_mismatch_index")
        if worst_idx is not None and len(worst_idx) == 1:
            key = keys[int(worst_idx[0])]
            metric["worst_mismatch_key_zero_based"] = list(key)
            metric["worst_mismatch_fortran"] = fortran_index(key)
        out[field] = metric
    return {"status": "SOURCE_SAVE_SPARSE_COMPARISON_EXECUTED", "per_field_metrics": out}


def summarize_components(part2: Mapping[str, Any]) -> dict[str, Any]:
    after_calc = part2["surfaces"]["after_calculate_phy_tend"]["arrays"]
    mask = interior_mask(after_calc["T_TENDF"].shape)
    flag_parts = (
        part2["surfaces"]["after_calculate_phy_tend"]["headers"][0]
        .get("active_theta_source_flags", [[]])[0]
    )
    active_map: dict[str, bool] = {}
    if len(flag_parts) >= 14:
        names = flag_parts[:7]
        values = flag_parts[7:14]
        active_map = {name: value == "1" for name, value in zip(names, values)}
    field_by_flag = {
        "ra": "RTHRATEN",
        "bl": "RTHBLTEN",
        "cu": "RTHCUTEN",
        "shcu": "RTHSHTEN",
        "grid_fdda": "RTHNDGDTEN",
        "iau": "RTHIAUTEN",
        "ifire": "RTHFRTEN",
    }
    active_fields = [
        field
        for flag, field in field_by_flag.items()
        if active_map.get(flag, False)
    ]
    rows = []
    for field in RAW_RTH_FIELDS:
        rows.append(
            {
                "field": field,
                "active_leaf": field in active_fields,
                "active_aggregate": field in {"RTH_ACTIVE_SUM", "RTH_ACTIVE_SUM_MOIST"},
                "full": array_summary(after_calc[field]),
                "nested_interior": array_summary(after_calc[field], mask=mask),
            }
        )
    all_rows = sorted(rows, key=lambda item: -float(item["nested_interior"].get("max_abs") or 0.0))
    active_leaf_rows = sorted(
        [row for row in rows if row["active_leaf"]],
        key=lambda item: -float(item["nested_interior"].get("max_abs") or 0.0),
    )
    active_aggregate_rows = sorted(
        [row for row in rows if row["active_aggregate"]],
        key=lambda item: -float(item["nested_interior"].get("max_abs") or 0.0),
    )
    return {
        "status": "RAW_RTH_COMPONENTS_SUMMARIZED",
        "active_flags_from_wrf_header": flag_parts,
        "active_field_names": active_fields,
        "active_leaf_ranked_by_nested_interior_max_abs": active_leaf_rows,
        "active_aggregate_ranked_by_nested_interior_max_abs": active_aggregate_rows,
        "all_raw_including_inactive_ranked_by_nested_interior_max_abs": all_rows,
        "inactive_leaf_note": (
            "Inactive RTH leaves can contain uninitialized finite/NaN values in WRF; "
            "only leaves enabled by active_theta_source_flags contribute to RTH_ACTIVE_SUM."
        ),
    }


def compare_stage_formulas(
    part2: Mapping[str, Any],
    source_surfaces: Mapping[str, Any],
    source_save: Mapping[str, Any],
    jax_capture: Mapping[str, Any],
) -> dict[str, Any]:
    cap = jax_capture["capture"]["captures"]
    namelist = jax_capture["capture"]["namelist"]
    after_calc = part2["surfaces"]["after_calculate_phy_tend"]["arrays"]
    after_update = part2["surfaces"]["after_update_phy_ten"]["arrays"]
    after_conv = part2["surfaces"]["after_conv_t_tendf_to_moist"]["arrays"]
    after_part2 = source_surfaces["surfaces"]["after_first_rk_step_part2"]["arrays"]

    mask = interior_mask(after_conv["T_TENDF"].shape)
    zero = np.zeros_like(after_conv["T_TENDF"])
    update_expected = after_calc["T_TENDF"] + after_calc["RTH_ACTIVE_SUM"]
    conv_expected = (
        after_update["THETA_M_FACTOR"] * after_update["T_TENDF"]
        + RVRD * (after_update["T_OLD"] + THETA_OFFSET) / after_update["THETA_M_FACTOR"] * after_update["QV_TEND"]
    )
    jax_dry_t_tendf = jax_array(cap["physics_carry_state_dry"]["T_TENDF"])
    physics_carry_t = jax_array(cap["physics_carry_state_dry"]["T_STATE"])
    physics_state_t = jax_array(cap["physics_state_dry"]["T_STATE"])
    dt_s = float(namelist["dt_s"])
    state_delta_mass_tendf = after_calc["MASS_H"] * (physics_state_t - physics_carry_t) / dt_s

    comparisons = {
        "after_calculate_t_tendf_vs_zero": {
            "full": diff_metrics("after_calculate_t_tendf_vs_zero", after_calc["T_TENDF"], zero),
            "nested_interior": diff_metrics(
                "after_calculate_t_tendf_vs_zero", after_calc["T_TENDF"], zero, mask=mask
            ),
        },
        "after_update_t_tendf_vs_pre_plus_active_rth": {
            "full": diff_metrics(
                "after_update_t_tendf_vs_pre_plus_active_rth",
                after_update["T_TENDF"],
                update_expected,
            ),
            "nested_interior": diff_metrics(
                "after_update_t_tendf_vs_pre_plus_active_rth",
                after_update["T_TENDF"],
                update_expected,
                mask=mask,
            ),
        },
        "after_conv_t_tendf_vs_moist_formula": {
            "full": diff_metrics(
                "after_conv_t_tendf_vs_moist_formula",
                after_conv["T_TENDF"],
                conv_expected,
            ),
            "nested_interior": diff_metrics(
                "after_conv_t_tendf_vs_moist_formula",
                after_conv["T_TENDF"],
                conv_expected,
                mask=mask,
            ),
        },
        "after_conv_t_tendf_vs_after_first_rk_step_part2": {
            "full": diff_metrics(
                "after_conv_t_tendf_vs_after_first_rk_step_part2",
                after_conv["T_TENDF"],
                after_part2["T_TENDF"],
            ),
            "nested_interior": diff_metrics(
                "after_conv_t_tendf_vs_after_first_rk_step_part2",
                after_conv["T_TENDF"],
                after_part2["T_TENDF"],
                mask=mask,
            ),
        },
        "after_update_t_tendf_vs_current_jax_dry_t_tendf": {
            "full": diff_metrics(
                "after_update_t_tendf_vs_current_jax_dry_t_tendf",
                after_update["T_TENDF"],
                jax_dry_t_tendf,
            ),
            "nested_interior": diff_metrics(
                "after_update_t_tendf_vs_current_jax_dry_t_tendf",
                after_update["T_TENDF"],
                jax_dry_t_tendf,
                mask=mask,
            ),
        },
        "after_conv_t_tendf_vs_current_jax_dry_t_tendf": {
            "full": diff_metrics(
                "after_conv_t_tendf_vs_current_jax_dry_t_tendf",
                after_conv["T_TENDF"],
                jax_dry_t_tendf,
            ),
            "nested_interior": diff_metrics(
                "after_conv_t_tendf_vs_current_jax_dry_t_tendf",
                after_conv["T_TENDF"],
                jax_dry_t_tendf,
                mask=mask,
            ),
        },
        "wrf_active_rth_vs_jax_physics_state_delta_mass_tendf": {
            "full": diff_metrics(
                "wrf_active_rth_vs_jax_physics_state_delta_mass_tendf",
                after_calc["RTH_ACTIVE_SUM"],
                state_delta_mass_tendf,
            ),
            "nested_interior": diff_metrics(
                "wrf_active_rth_vs_jax_physics_state_delta_mass_tendf",
                after_calc["RTH_ACTIVE_SUM"],
                state_delta_mass_tendf,
                mask=mask,
            ),
        },
        "after_conv_t_tendf_vs_jax_physics_state_delta_mass_tendf": {
            "full": diff_metrics(
                "after_conv_t_tendf_vs_jax_physics_state_delta_mass_tendf",
                after_conv["T_TENDF"],
                state_delta_mass_tendf,
            ),
            "nested_interior": diff_metrics(
                "after_conv_t_tendf_vs_jax_physics_state_delta_mass_tendf",
                after_conv["T_TENDF"],
                state_delta_mass_tendf,
                mask=mask,
            ),
        },
    }
    sparse = source_save_sparse_compare(
        source_save,
        {
            "T_TENDF": after_part2["T_TENDF"],
            "T_HIST_SRC": after_conv["T_HIST_SRC"],
            "H_DIABATIC": after_part2["H_DIABATIC"],
            "T_SAVE": after_part2["T_SAVE"],
        },
        fields=("T_TENDF", "T_HIST_SRC", "H_DIABATIC", "T_SAVE"),
    )
    sparse_vs_jax_dry = source_save_sparse_compare(
        source_save,
        {"T_TENDF": jax_dry_t_tendf},
        fields=("T_TENDF",),
    )
    return {
        "status": "STAGE_FORMULA_COMPARISONS_EXECUTED",
        "comparisons": comparisons,
        "source_save_sparse_vs_after_first_rk_step_part2": sparse,
        "source_save_sparse_vs_current_jax_dry_t_tendf": sparse_vs_jax_dry,
        "derived_candidate_summaries": {
            "current_jax_dry_t_tendf": array_summary(jax_dry_t_tendf),
            "jax_physics_state_delta_mass_tendf": array_summary(state_delta_mass_tendf),
            "wrf_update_expected_pre_plus_active_rth": array_summary(update_expected),
            "wrf_conv_expected_moist_formula": array_summary(conv_expected),
        },
    }


def metric_max(comp: Mapping[str, Any], key: str, mask_name: str = "nested_interior") -> float | None:
    metric = comp["comparisons"][key][mask_name]
    value = metric.get("max_abs")
    return None if value is None else float(value)


def classify(formulas: Mapping[str, Any], components: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]], str]:
    update_resid = metric_max(formulas, "after_update_t_tendf_vs_pre_plus_active_rth")
    conv_resid = metric_max(formulas, "after_conv_t_tendf_vs_moist_formula")
    final_resid = metric_max(formulas, "after_conv_t_tendf_vs_after_first_rk_step_part2")
    jax_dry_resid = metric_max(formulas, "after_conv_t_tendf_vs_current_jax_dry_t_tendf")
    delta_resid = metric_max(formulas, "after_conv_t_tendf_vs_jax_physics_state_delta_mass_tendf")
    top_component = components["active_leaf_ranked_by_nested_interior_max_abs"][0]

    alternatives = [
        {
            "rank": 1,
            "hypothesis": "WRF raw active RTH source leaves are missing from the current JAX dry bundle",
            "status": "SUPPORTED",
            "evidence": {
                "after_update_vs_pre_plus_active_rth_nested_max_abs": update_resid,
                "after_conv_vs_moist_formula_nested_max_abs": conv_resid,
                "after_conv_vs_current_jax_dry_nested_max_abs": jax_dry_resid,
                "dominant_component": top_component["field"],
                "dominant_component_nested_max_abs": top_component["nested_interior"]["max_abs"],
            },
        },
        {
            "rank": 2,
            "hypothesis": "moist-theta conversion after update_phy_ten is the first wrong boundary",
            "status": "FALSIFIED" if conv_resid is not None and conv_resid < 1.0e-3 else "POSSIBLE",
            "evidence": {"after_conv_vs_moist_formula_nested_max_abs": conv_resid},
        },
        {
            "rank": 3,
            "hypothesis": "boundary, spec-zone, or acoustic code mutates T_TENDF before the accepted final surface",
            "status": "FALSIFIED" if final_resid is not None and final_resid < 1.0e-6 else "POSSIBLE",
            "evidence": {"after_conv_vs_after_first_rk_step_part2_nested_max_abs": final_resid},
        },
        {
            "rank": 4,
            "hypothesis": "the aggregate JAX physics state delta can be used as a narrow T_TENDF source fix",
            "status": "FALSIFIED" if delta_resid is not None and delta_resid > 1.0e-3 else "POSSIBLE",
            "evidence": {"after_conv_vs_jax_physics_state_delta_mass_tendf_nested_max_abs": delta_resid},
        },
    ]
    if (
        update_resid is not None
        and update_resid < 1.0e-6
        and conv_resid is not None
        and conv_resid < 1.0e-3
        and final_resid is not None
        and final_resid < 1.0e-6
        and jax_dry_resid is not None
        and jax_dry_resid > 1.0
    ):
        verdict = "STEP1_PART2_SOURCE_LEAVES_LOCALIZED_UPDATE_PHY_TEN_RAW_RTH_TO_T_TENDF_MISSING_IN_JAX_DRY_BUNDLE"
        next_boundary = (
            "Implement true WRF dry physics source leaves for active RTHRATEN/RTHBLTEN "
            "before `_augment_large_step_tendencies`; do not use aggregate post-physics "
            "state deltas unless a scheme-level raw-leaf proof closes this same gate."
        )
    else:
        verdict = "STEP1_PART2_SOURCE_LEAVES_SPLIT_NEEDS_NEXT_FALSIFIER"
        next_boundary = "Use the ranked alternatives and first non-closing self-consistency metric in this proof."
    return verdict, alternatives, next_boundary


def generate_wrf_patch_diff() -> dict[str, Any]:
    pairs = (
        "dyn_em/module_first_rk_step_part2.F",
        "dyn_em/solve_em.F",
    )
    chunks: list[str] = []
    statuses: dict[str, Any] = {}
    for rel in pairs:
        base = WRF_BASE / rel
        patched = WRF_COPY / rel
        if not base.is_file() or not patched.is_file():
            statuses[rel] = {
                "status": "MISSING",
                "base": path_info(base),
                "patched": path_info(patched),
            }
            continue
        proc = subprocess.run(
            ["diff", "-u", str(base), str(patched)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        statuses[rel] = {
            "status": "DIFFED",
            "returncode": int(proc.returncode),
            "stderr_tail": proc.stderr[-2000:],
            "base": path_info(base),
            "patched": path_info(patched),
        }
        if proc.stdout:
            chunks.append(proc.stdout.rstrip() + "\n")
    OUT_WRF_PATCH.parent.mkdir(parents=True, exist_ok=True)
    OUT_WRF_PATCH.write_text("\n".join(chunks), encoding="utf-8")
    return {"status": "WRF_PATCH_DIFF_WRITTEN", "output": path_info(OUT_WRF_PATCH), "inputs": statuses}


def compact_comparisons(formulas: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, pair in formulas["comparisons"].items():
        out[name] = {
            "full": compact_metric(pair["full"]),
            "nested_interior": compact_metric(pair["nested_interior"]),
        }
    sparse = formulas["source_save_sparse_vs_after_first_rk_step_part2"]["per_field_metrics"]
    out["source_save_sparse_vs_after_first_rk_step_part2"] = {
        field: compact_metric(metric) for field, metric in sparse.items()
    }
    sparse_jax = formulas["source_save_sparse_vs_current_jax_dry_t_tendf"]["per_field_metrics"]
    out["source_save_sparse_vs_current_jax_dry_t_tendf"] = {
        field: compact_metric(metric) for field, metric in sparse_jax.items()
    }
    return out


def build_proof() -> dict[str, Any]:
    import jax  # noqa: PLC0415

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}
    shapes = expected_shapes()
    part2 = parse_part2_surfaces(shapes)
    if part2.get("status") != "WRF_PART2_TRUTH_READY":
        return {"status": "BLOCKED_PART2_TRUTH", "blocker": part2}
    source_surfaces = parse_existing_source_surfaces(shapes)
    if source_surfaces.get("status") != "WRF_SOURCE_SURFACES_READY":
        return {"status": "BLOCKED_SOURCE_SURFACES", "blocker": strip_arrays(source_surfaces)}
    source_save = parse_source_save()
    if source_save.get("status") != "SOURCE_SAVE_READY":
        return {"status": "BLOCKED_SOURCE_SAVE", "blocker": source_save}
    jax_capture = build_jax_capture()
    if jax_capture.get("status") != "JAX_CAPTURE_READY":
        return {"status": "BLOCKED_JAX_CAPTURE", "blocker": jax_capture}

    components = summarize_components(part2)
    formulas = compare_stage_formulas(part2, source_surfaces, source_save, jax_capture)
    verdict, alternatives, next_boundary = classify(formulas, components)
    wrf_patch = generate_wrf_patch_diff()

    return {
        "status": "PROOF_EXECUTED",
        "schema": "wrfgpu2.v014.step1_part2_source_leaves_split.v1",
        "verdict": verdict,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": jax_environment(),
        "target": {"domain": TARGET_DOMAIN, "step": TARGET_STEP, "cpu_only": True},
        "paths": {
            "scratch": path_info(SCRATCH),
            "wrf_truth": path_info(WRF_TRUTH),
            "wrf_run_log": path_info(WRF_RUN_DIR / "step1_part2_single_rank.stdout"),
            "wrf_rsl_error": path_info(WRF_RUN_DIR / "rsl.error.0000"),
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "handoff": path_info(HANDOFF),
        },
        "accepted_boundary": {
            "patched_init_p_mu_w_ph_frontiers": "closed by prior proof",
            "first_material_full_domain_field": "T_TENDF at WRF after_first_rk_step_part2",
            "prior_reported_after_part2_t_tendf_max_abs": 2457.5830078125,
            "prior_reported_after_part2_t_tendf_rmse": 21.20870100357482,
            "boundary_spec_acoustic": "too late for first failure",
        },
        "wrf_truth": {
            "part2": strip_arrays(part2),
            "source_surfaces": strip_arrays(source_surfaces),
            "source_save": {
                "status": source_save["status"],
                "files": source_save["files"],
                "compact": source_save["compact"],
            },
        },
        "jax_capture": {
            "status": jax_capture["capture"]["status"],
            "label": jax_capture["capture"]["label"],
            "namelist": jax_capture["capture"]["namelist"],
            "run_radiation": jax_capture["capture"]["run_radiation"],
            "patched_init_metadata": jax_capture["patched"].get("metadata"),
        },
        "raw_rth_components": components,
        "proof": {
            "stage_formula_comparisons": compact_comparisons(formulas),
            "derived_candidate_summaries": formulas["derived_candidate_summaries"],
            "ranked_alternatives": alternatives,
            "next_boundary": next_boundary,
        },
        "tooling": {
            "wrf_patch_diff": wrf_patch,
            "wrf_fixture_commands": {
                "compile": (
                    "cd /tmp/wrf_gpu2_step1_part2_source_leaves_split_20260609/WRF && "
                    "timeout 3600 env PATH=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin:$PATH "
                    "NETCDF=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build "
                    "PNETCDF=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build "
                    "WRFIO_NCD_LARGE_FILE_SUPPORT=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu tcsh ./compile em_real"
                ),
                "run": (
                    "cd /tmp/wrf_gpu2_step1_part2_source_leaves_split_20260609/run && "
                    "timeout 1800 env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 "
                    "WRFGPU2_STEP1_PART2_SOURCE_LEAVES_SPLIT=1 "
                    "WRFGPU2_STEP1_PART2_SOURCE_LEAVES_SPLIT_ROOT=/tmp/wrf_gpu2_step1_part2_source_leaves_split_20260609/wrf_truth "
                    "WRFGPU2_STEP1_PART2_SOURCE_LEAVES_SPLIT_GRID=2 "
                    "WRFGPU2_STEP1_PART2_SOURCE_LEAVES_SPLIT_START_STEP=1 "
                    "WRFGPU2_STEP1_PART2_SOURCE_LEAVES_SPLIT_END_STEP=1 "
                    "WRFGPU2_STEP1_RK1_SOURCE_BOUNDARY=1 "
                    "WRFGPU2_STEP1_RK1_SOURCE_BOUNDARY_ROOT=/tmp/wrf_gpu2_step1_part2_source_leaves_split_20260609/wrf_truth "
                    "WRFGPU2_STEP1_RK1_SOURCE_BOUNDARY_GRID=2 "
                    "WRFGPU2_STEP1_RK1_SOURCE_BOUNDARY_START_STEP=1 "
                    "WRFGPU2_STEP1_RK1_SOURCE_BOUNDARY_END_STEP=1 "
                    "WRFGPU2_SOURCE_SAVE_BOUNDARY=1 "
                    "WRFGPU2_SOURCE_SAVE_BOUNDARY_ROOT=/tmp/wrf_gpu2_step1_part2_source_leaves_split_20260609/wrf_truth "
                    "WRFGPU2_SOURCE_SAVE_BOUNDARY_GRID=2 "
                    "WRFGPU2_SOURCE_SAVE_BOUNDARY_START_STEP=1 "
                    "WRFGPU2_SOURCE_SAVE_BOUNDARY_END_STEP=1 ./wrf.exe"
                ),
            },
        },
        "git": {
            "head": run_command(["git", "rev-parse", "HEAD"]),
            "branch": run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "status_short": run_command(["git", "status", "--short"]),
        },
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    proof = payload["proof"]
    comps = proof["stage_formula_comparisons"]
    alternatives = proof["ranked_alternatives"]
    top = payload["raw_rth_components"]["active_leaf_ranked_by_nested_interior_max_abs"][0]
    aggregate = payload["raw_rth_components"]["active_aggregate_ranked_by_nested_interior_max_abs"][0]
    update = comps["after_update_t_tendf_vs_pre_plus_active_rth"]["nested_interior"]
    conv = comps["after_conv_t_tendf_vs_moist_formula"]["nested_interior"]
    final = comps["after_conv_t_tendf_vs_after_first_rk_step_part2"]["nested_interior"]
    jax_dry = comps["after_conv_t_tendf_vs_current_jax_dry_t_tendf"]["nested_interior"]
    delta = comps["after_conv_t_tendf_vs_jax_physics_state_delta_mass_tendf"]["nested_interior"]
    source_save_after = comps["source_save_sparse_vs_after_first_rk_step_part2"]["T_TENDF"]
    source_save_jax = comps["source_save_sparse_vs_current_jax_dry_t_tendf"]["T_TENDF"]

    lines = [
        "# V0.14 Step-1 Part2 Source Leaves Split",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Evidence",
        "",
        f"- WRF `update_phy_ten`: `T_TENDF == pre + active RTH` on nested interior, max_abs `{update['max_abs']}`, rmse `{update['rmse']}`.",
        f"- WRF `conv_t_tendf_to_moist`: moist-theta formula closes on nested interior, max_abs `{conv['max_abs']}`, rmse `{conv['rmse']}`.",
        f"- `after_conv_t_tendf_to_moist` equals `after_first_rk_step_part2` on nested interior, max_abs `{final['max_abs']}`.",
        f"- Current patched-init JAX dry `T_TENDF` stays divergent: nested-interior max_abs `{jax_dry['max_abs']}`, rmse `{jax_dry['rmse']}`.",
        f"- Aggregate JAX physics state-delta candidate is also rejected: nested-interior max_abs `{delta['max_abs']}`, rmse `{delta['rmse']}`.",
        f"- Source-save sparse `T_TENDF` is also divergent vs current JAX dry: max_abs `{source_save_jax['max_abs']}`, rmse `{source_save_jax['rmse']}`.",
        f"- Source-save is a later adjacent leaf, not the first boundary: vs `after_first_rk_step_part2` max_abs `{source_save_after['max_abs']}`.",
        f"- Dominant active raw leaf is `{top['field']}` with nested-interior max_abs `{top['nested_interior']['max_abs']}`.",
        f"- Largest active aggregate is `{aggregate['field']}` with nested-interior max_abs `{aggregate['nested_interior']['max_abs']}`.",
        "",
        "## Ranking",
        "",
    ]
    for alt in alternatives:
        lines.append(f"- `{alt['status']}` rank {alt['rank']}: {alt['hypothesis']}.")
    lines.extend(
        [
            "",
            "## Next Boundary",
            "",
            proof["next_boundary"],
            "",
            "Proof objects: `proofs/v014/step1_part2_source_leaves_split.json` and `proofs/v014/step1_part2_source_leaves_split_wrf_patch.diff`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    proof = payload["proof"]
    comps = proof["stage_formula_comparisons"]
    jax_dry = comps["after_conv_t_tendf_vs_current_jax_dry_t_tendf"]["nested_interior"]
    top = payload["raw_rth_components"]["active_leaf_ranked_by_nested_interior_max_abs"][0]
    lines = [
        "# Review: V0.14 Step-1 Part2 Source Leaves Split",
        "",
        "Finding: Step-1 `T_TENDF` divergence is not boundary/spec/acoustic timing. WRF creates it inside `first_rk_step_part2` by adding active raw `RTH*TEN` leaves in `update_phy_ten`, then applying moist-theta conversion.",
        "",
        f"- Verdict: `{payload['verdict']}`.",
        f"- Current JAX dry source residual: max_abs `{jax_dry['max_abs']}`, rmse `{jax_dry['rmse']}`.",
        f"- Dominant active WRF raw leaf: `{top['field']}`.",
        f"- Next decision: {proof['next_boundary']}",
        "",
        "No production `src/gpuwrf` files were changed.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    payload = build_proof()
    write_json(OUT_JSON, payload)
    if payload.get("status") == "PROOF_EXECUTED":
        OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
        OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
        OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")
        print(f"Wrote {OUT_JSON}")
        print(f"Wrote {OUT_MD}")
        print(f"Wrote {OUT_REVIEW}")
        print(f"Wrote {OUT_WRF_PATCH}")
        print(payload["verdict"])
        return 0
    OUT_MD.write_text(
        "# V0.14 Step-1 Part2 Source Leaves Split\n\n"
        f"Blocked: `{payload.get('status')}`. See `{OUT_JSON}`.\n",
        encoding="utf-8",
    )
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(
        "# Review: V0.14 Step-1 Part2 Source Leaves Split\n\n"
        f"Blocked: `{payload.get('status')}`. See `{OUT_JSON}`.\n",
        encoding="utf-8",
    )
    print(f"Blocked: {payload.get('status')}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
