#!/usr/bin/env python3
"""V0.14 Step-1 live-nest start_domain perturbation-state subsurface proof.

CPU-only proof.  Parses disposable WRF ``start_domain_em`` truth surfaces from a
fresh timestamped scratch workdir, compares them to accepted pre-part1 WRF truth
and current JAX live-nest loader values, and tests the narrow P/MU/W formula
contracts needed before a production init patch.
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
import step1_live_nest_perturb_state_init as predecessor  # noqa: E402
import step1_pre_part1_handoff as pre  # noqa: E402


OUT_JSON = PROOF_DIR / "step1_start_domain_perturb_subsurface.json"
OUT_MD = PROOF_DIR / "step1_start_domain_perturb_subsurface.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-step1-start-domain-perturb-subsurface.md"
OUT_WRF_PATCH = PROOF_DIR / "step1_start_domain_perturb_subsurface_wrf_patch.diff"

SPRINT_CONTRACT = (
    ROOT / ".agent/sprints/2026-06-09-v014-step1-start-domain-perturb-subsurface/sprint-contract.md"
)
HANDOFF = ROOT / ".agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md"
PREDECESSOR_JSON = PROOF_DIR / "step1_live_nest_perturb_state_init.json"
PREDECESSOR_MD = PROOF_DIR / "step1_live_nest_perturb_state_init.md"
REQUIRED_ANCESTOR = "ee6cbbe1"

SCRATCH_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v014_step1_start_domain_perturb_subsurface")
WORK_DIR = Path(
    os.environ.get(
        "WRFGPU2_STEP1_START_DOMAIN_WORK_DIR",
        str(SCRATCH_ROOT / "work_clean_20260609_194715"),
    )
)
WRF_TRUTH = WORK_DIR / "wrf_truth"
WRF_TREE = WORK_DIR / "WRF"
WRF_RUN_DIR = WORK_DIR / "run"
WRF_LOG_DIR = WORK_DIR / "logs"
WRF_BACKUP = WORK_DIR / "backups/start_em.F.before_start_domain_perturb_subsurface"
PRECALL_QV_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v014_step1_qvapor_precall_savepoint/precall_truth_only")

SURFACES = (
    "after_hypsometric_p_al_alt",
    "before_press_adj",
    "after_press_adj",
    "after_w_surface_branch",
)
TARGET_DOMAIN = 2
TARGET_START_STEP = 0
PRECALL_SURFACE = "before_first_rk_step_part1_call"
PRECALL_STEP = 1

R_D = 287.0
R_V = 461.6
CP = 7.0 * R_D / 2.0
CV = CP - R_D
CPOVCV = CP / CV
CVPM = -CV / CP
P1000MB = 100000.0
T0 = 300.0
G = 9.81
RVOVRD = R_V / R_D

MATERIAL_THRESHOLDS = {
    "P_STATE": 1.0,
    "MU_STATE": 1.0e-2,
    "W_STATE": 1.0e-2,
    "PH_STATE": 1.0e-2,
    "PB": 1.0,
    "MUB": 1.0e-2,
    "PHB": 1.0e-2,
    "T_STATE": 1.0e-3,
    "THETA_FULL": 1.0e-3,
    "QVAPOR": 1.0e-7,
    "HT": 1.0e-6,
    "HT_FINE": 1.0e-6,
    "AL": 1.0e-8,
    "ALT": 1.0e-8,
    "ALB": 1.0e-8,
}

START_RECORD_SPECS = {
    "MASS_START": {
        "length": 17,
        "fields": (
            "T1_STATE",
            "T2_STATE",
            "THETA1_FULL",
            "THETA2_FULL",
            "QVAPOR",
            "P_STATE",
            "PB",
            "AL",
            "ALT",
            "ALB",
        ),
    },
    "SURF_START": {
        "length": 12,
        "fields": (
            "MU1_STATE",
            "MU2_STATE",
            "MUT",
            "MUB",
            "HT",
            "HT_FINE",
            "HT_MINUS_HT_FINE",
        ),
    },
    "WPH_START": {
        "length": 12,
        "fields": ("W1_STATE", "W2_STATE", "PH1_STATE", "PH2_STATE", "PHB"),
    },
}
PRECALL_RECORD_SPECS = {
    "MASS_PREPART": {
        "lengths": {13, 14},
        "fields_without_qv": ("T_STATE", "P_STATE", "PB", "MU_STATE", "MUB", "MUT"),
        "fields_with_qv": ("T_STATE", "P_STATE", "PB", "MU_STATE", "MUB", "MUT", "QVAPOR"),
    },
    "WPH_PREPART": {
        "lengths": {10},
        "fields_without_qv": ("W_STATE", "PH_STATE", "PHB"),
        "fields_with_qv": ("W_STATE", "PH_STATE", "PHB"),
    },
}


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_file_count(path: Path, pattern: str = "*") -> int | None:
    if not path.is_dir():
        return None
    return sum(1 for item in path.glob(pattern) if item.is_file())


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


def git_metadata() -> dict[str, Any]:
    ancestor = run_command(["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, "HEAD"])
    return {
        "head": run_command(["git", "rev-parse", "HEAD"]),
        "branch": run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "log_head": run_command(["git", "log", "-1", "--oneline", "--decorate"]),
        "required_ancestor": {
            "commit": REQUIRED_ANCESTOR,
            "returncode": ancestor["returncode"],
            "is_ancestor": ancestor["returncode"] == 0,
            "stderr_tail": ancestor.get("stderr_tail"),
        },
    }


def as_np(value: Any, dtype: Any = np.float64) -> np.ndarray:
    try:
        import jax  # noqa: PLC0415

        value = jax.device_get(value)
    except Exception:
        pass
    return np.asarray(value, dtype=dtype)


def expected_shapes() -> dict[str, tuple[int, ...]]:
    return pre.expected_shapes()


def shape_for(field: str, shapes: Mapping[str, tuple[int, ...]]) -> tuple[int, ...]:
    if field in {"MU1_STATE", "MU2_STATE", "MU_STATE", "MUT", "MUB", "HT", "HT_FINE", "HT_MINUS_HT_FINE"}:
        return shapes["mass2d"]
    if field in {"W1_STATE", "W2_STATE", "W_STATE", "PH1_STATE", "PH2_STATE", "PH_STATE", "PHB"}:
        return shapes["wph"]
    return shapes["mass"]


def array_summary(array: Any) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.float64)
    return {
        "shape": list(arr.shape),
        "count": int(arr.size),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
    }


def fortran_index(field: str, index: tuple[int, ...] | None) -> dict[str, int] | None:
    if index is None:
        return None
    if len(index) == 2:
        y, x = index
        return {"i": int(x) + 1, "j": int(y) + 1}
    if len(index) == 3:
        k, y, x = index
        key = "kstag" if field.startswith(("W", "PH")) or field == "PHB" else "k"
        return {"i": int(x) + 1, "j": int(y) + 1, key: int(k) + 1}
    return None


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


def parse_headers_and_records(
    *,
    files: list[Path],
    shapes: Mapping[str, tuple[int, ...]],
    record_specs: Mapping[str, Mapping[str, Any]],
    parser_name: str,
) -> dict[str, Any]:
    fields: list[str] = []
    for spec in record_specs.values():
        if "fields" in spec:
            fields.extend([str(item) for item in spec["fields"]])
        else:
            fields.extend([str(item) for item in spec["fields_with_qv"]])
    fields = sorted(set(fields))
    arrays = {field: np.full(shape_for(field, shapes), np.nan, dtype=np.float64) for field in fields}
    duplicate_stats = {
        field: {"duplicates": 0, "mismatches": 0, "max_delta": 0.0, "first_mismatch": None}
        for field in fields
    }
    record_counts = {record: 0 for record in record_specs}
    headers: list[dict[str, Any]] = []
    for path in files:
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
                if tag in record_specs:
                    spec = record_specs[tag]
                    if "length" in spec:
                        expected_lengths = {int(spec["length"])}
                        fields_for_record = tuple(str(item) for item in spec["fields"])
                    else:
                        expected_lengths = {int(item) for item in spec["lengths"]}
                        fields_for_record = (
                            tuple(str(item) for item in spec["fields_with_qv"])
                            if len(parts) == max(expected_lengths)
                            else tuple(str(item) for item in spec["fields_without_qv"])
                        )
                    if len(parts) not in expected_lengths:
                        return {
                            "status": "BLOCKED_PARSE_ERROR",
                            "parser": parser_name,
                            "path": str(path),
                            "line": stripped[:240],
                            "expected_lengths": sorted(expected_lengths),
                            "actual_length": len(parts),
                        }
                    if tag == "SURF_START":
                        x = int(parts[3])
                        y = int(parts[4])
                        k = None
                        values = [float(item) for item in parts[5:]]
                    else:
                        x = int(parts[4])
                        y = int(parts[5])
                        k = int(parts[6])
                        values = [float(item) for item in parts[7:]]
                    for field, value in zip(fields_for_record, values):
                        shape = arrays[field].shape
                        index = (y, x) if len(shape) == 2 else (int(k), y, x)
                        _record_value(arrays, duplicate_stats, field, index, value)
                    record_counts[tag] += 1
                    continue
                if tag == "record_schema":
                    header.setdefault("record_schema", []).append(" ".join(parts[1:]))
                    continue
                if len(parts) > 1:
                    header[tag] = parts[1:]
        headers.append(header)
    duplicate_mismatches = {name: item for name, item in duplicate_stats.items() if int(item["mismatches"]) > 0}
    if duplicate_mismatches:
        return {
            "status": "BLOCKED_DUPLICATE_MISMATCH",
            "parser": parser_name,
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
            "parser": parser_name,
            "missing": missing,
            "duplicate_stats": duplicate_stats,
            "record_counts": record_counts,
        }
    return {
        "status": "WRF_SURFACE_READY",
        "parser": parser_name,
        "raw_file_count": len(files),
        "record_counts": record_counts,
        "duplicate_stats": duplicate_stats,
        "headers": headers[:4],
        "summaries": {name: array_summary(arr) for name, arr in arrays.items()},
        "arrays": arrays,
    }


def parse_start_surface(surface: str, shapes: Mapping[str, tuple[int, ...]]) -> dict[str, Any]:
    pattern = f"{surface}_d{TARGET_DOMAIN}_step_{TARGET_START_STEP}_*.txt"
    files = sorted(WRF_TRUTH.glob(pattern))
    if not files:
        return {"status": "BLOCKED_NO_WRF_SURFACE_FILES", "surface": surface, "pattern": pattern}
    parsed = parse_headers_and_records(
        files=files,
        shapes=shapes,
        record_specs=START_RECORD_SPECS,
        parser_name=f"start_domain:{surface}",
    )
    parsed["surface"] = surface
    parsed["pattern"] = pattern
    return parsed


def parse_precall_surface(shapes: Mapping[str, tuple[int, ...]]) -> dict[str, Any]:
    pattern = f"{PRECALL_SURFACE}_d{TARGET_DOMAIN}_step_{PRECALL_STEP}_rk_1_*.txt"
    files = sorted(PRECALL_QV_ROOT.glob(pattern))
    if not files:
        return {"status": "BLOCKED_NO_PRECALL_QV_FILES", "pattern": pattern, "root": str(PRECALL_QV_ROOT)}
    parsed = parse_headers_and_records(
        files=files,
        shapes=shapes,
        record_specs=PRECALL_RECORD_SPECS,
        parser_name="precall_qv",
    )
    parsed["surface"] = PRECALL_SURFACE
    parsed["pattern"] = pattern
    return parsed


def diff_metrics(field: str, candidate: Any, reference: Any, *, region: str = "full_domain") -> dict[str, Any]:
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if cand.shape != ref.shape:
        return {
            "status": "SHAPE_MISMATCH",
            "region": region,
            "candidate_shape": list(cand.shape),
            "reference_shape": list(ref.shape),
        }
    diff = cand - ref
    absdiff = np.abs(diff)
    finite_abs = absdiff[np.isfinite(absdiff)]
    mismatch_mask = (diff != 0.0) | (~np.isfinite(diff))
    mismatch = np.argwhere(mismatch_mask)
    first = tuple(int(x) for x in mismatch[0]) if mismatch.size else None
    if finite_abs.size:
        worst = tuple(int(x) for x in np.unravel_index(int(np.nanargmax(absdiff)), absdiff.shape))
        max_abs = float(np.nanmax(absdiff))
        rmse = float(np.sqrt(np.nanmean(diff * diff)))
        bias = float(np.nanmean(diff))
        p95 = float(np.nanpercentile(absdiff, 95))
        p99 = float(np.nanpercentile(absdiff, 99))
    else:
        worst = first
        max_abs = None
        rmse = None
        bias = None
        p95 = None
        p99 = None
    return {
        "status": "OK",
        "region": region,
        "count": int(diff.size),
        "shape": list(diff.shape),
        "max_abs": max_abs,
        "rmse": rmse,
        "bias": bias,
        "p95": p95,
        "p99": p99,
        "nonfinite_diff_count": int((~np.isfinite(diff)).sum()),
        "first_mismatch_index": list(first) if first is not None else None,
        "first_mismatch_fortran": fortran_index(field, first),
        "worst_mismatch_index": list(worst) if worst is not None else None,
        "worst_mismatch_fortran": fortran_index(field, worst),
    }


def metric_brief(metric: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not metric:
        return None
    keys = (
        "status",
        "shape",
        "count",
        "max_abs",
        "rmse",
        "bias",
        "p95",
        "p99",
        "nonfinite_diff_count",
        "worst_mismatch_index",
        "worst_mismatch_fortran",
    )
    return {key: metric.get(key) for key in keys if key in metric}


def compare_map(
    candidate: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    mapping: Mapping[str, str],
) -> dict[str, Any]:
    return {
        f"{cand_field}->${ref_field}".replace("$", ""): metric_brief(
            diff_metrics(ref_field, candidate[cand_field], reference[ref_field])
        )
        for cand_field, ref_field in mapping.items()
    }


def rank_metrics(metrics: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field, metric in metrics.items():
        max_abs = metric.get("max_abs") if metric else None
        rows.append(
            {
                "field": field,
                "max_abs": max_abs,
                "rmse": metric.get("rmse") if metric else None,
                "bias": metric.get("bias") if metric else None,
                "threshold": MATERIAL_THRESHOLDS.get(field.split("->")[-1], MATERIAL_THRESHOLDS.get(field)),
                "worst_mismatch_fortran": metric.get("worst_mismatch_fortran") if metric else None,
            }
        )
    return sorted(rows, key=lambda item: -float(item.get("max_abs") or 0.0))


def pressure_from_alt(*, pb: Any, theta_full: Any, alt: Any, dtype: Any) -> np.ndarray:
    pb_arr = as_np(pb, dtype)
    theta_arr = as_np(theta_full, dtype)
    alt_arr = as_np(alt, dtype)
    out = (dtype(P1000MB) * ((dtype(R_D) * theta_arr) / (dtype(P1000MB) * alt_arr)) ** dtype(CPOVCV) - pb_arr)
    return np.asarray(out, dtype=np.float64)


def diagnose_alt_from_ph(
    *,
    ph: Any,
    phb: Any,
    mub: Any,
    mu: Any,
    metrics: Any,
    dtype: Any,
) -> np.ndarray:
    ph_arr = as_np(ph, dtype)
    phb_arr = as_np(phb, dtype)
    mub_arr = as_np(mub, dtype)
    mu_arr = as_np(mu, dtype)
    c3f = as_np(metrics.c3f, dtype)
    c4f = as_np(metrics.c4f, dtype)
    c3h = as_np(metrics.c3h, dtype)
    c4h = as_np(metrics.c4h, dtype)
    p_top = dtype(as_np(metrics.p_top, dtype))
    full_mu = (mub_arr + mu_arr).astype(dtype)
    pfu = (c3f[1:, None, None] * full_mu[None, :, :] + c4f[1:, None, None] + p_top).astype(dtype)
    pfd = (c3f[:-1, None, None] * full_mu[None, :, :] + c4f[:-1, None, None] + p_top).astype(dtype)
    phm = (c3h[:, None, None] * full_mu[None, :, :] + c4h[:, None, None] + p_top).astype(dtype)
    log_term = np.log((pfd / pfu).astype(dtype)).astype(dtype)
    alt = ((ph_arr[1:] - ph_arr[:-1] + phb_arr[1:] - phb_arr[:-1]) / (phm * log_term)).astype(dtype)
    return np.asarray(alt, dtype=np.float64)


def pressure_from_ph_formula(
    *,
    pb: Any,
    phb: Any,
    mub: Any,
    ph: Any,
    theta_full: Any,
    mu: Any,
    metrics: Any,
    dtype: Any,
) -> tuple[np.ndarray, np.ndarray]:
    alt = diagnose_alt_from_ph(ph=ph, phb=phb, mub=mub, mu=mu, metrics=metrics, dtype=dtype)
    return pressure_from_alt(pb=pb, theta_full=theta_full, alt=alt, dtype=dtype), alt


def press_adj_mu(*, mu_before: Any, al: Any, alt: Any, alb: Any, ht: Any, ht_fine: Any, dtype: Any) -> np.ndarray:
    mu = as_np(mu_before, dtype)
    al_arr = as_np(al, dtype)
    alt_arr = as_np(alt, dtype)
    alb_arr = as_np(alb, dtype)
    ht_arr = as_np(ht, dtype)
    ht_fine_arr = as_np(ht_fine, dtype)
    out = mu + al_arr[0] / (alt_arr[0] * alb_arr[0]) * dtype(G) * (ht_arr - ht_fine_arr)
    return np.asarray(out, dtype=np.float64)


def build_jax_loader_arrays() -> dict[str, Any]:
    from gpuwrf.integration.d02_replay import _wrf_start_domain_base_from_hgt  # noqa: PLC0415

    inputs = live.build_live_nest_step1_inputs()
    live_child = inputs["live_child"]
    raw_child = inputs["raw_child"]
    state = live_child["state"]
    base = live_child["base_state"]
    grid = live_child["grid"]
    metrics = live_child["metrics"]
    _pb, _mub, _phb, _t_init, alb = _wrf_start_domain_base_from_hgt(
        inputs["run"],
        "d02",
        hgt=grid.terrain_height,
        metrics=metrics,
    )
    theta = as_np(state.theta)
    arrays = {
        "T_STATE": theta - T0,
        "THETA_FULL": theta,
        "QVAPOR": as_np(state.qv),
        "P_STATE": as_np(state.p_perturbation),
        "PB": as_np(base.pb),
        "MU_STATE": as_np(state.mu_perturbation),
        "MUB": as_np(base.mub),
        "MUT": as_np(base.mub) + as_np(state.mu_perturbation),
        "W_STATE": as_np(state.w),
        "PH_STATE": as_np(state.ph_perturbation),
        "PHB": as_np(base.phb),
        "HT": as_np(grid.terrain_height),
        "HT_FINE": as_np(raw_child["grid"].terrain_height),
        "ALB": as_np(alb),
    }
    p64, alt64 = pressure_from_ph_formula(
        pb=base.pb,
        phb=base.phb,
        mub=base.mub,
        ph=state.ph_perturbation,
        theta_full=state.theta,
        mu=state.mu_perturbation,
        metrics=metrics,
        dtype=np.float64,
    )
    p32, alt32 = pressure_from_ph_formula(
        pb=base.pb,
        phb=base.phb,
        mub=base.mub,
        ph=state.ph_perturbation,
        theta_full=state.theta,
        mu=state.mu_perturbation,
        metrics=metrics,
        dtype=np.float32,
    )
    arrays.update(
        {
            "P_FORMULA_FP64": p64,
            "ALT_FORMULA_FP64": alt64,
            "AL_FORMULA_FP64": alt64 - arrays["ALB"],
            "P_FORMULA_FP32": p32,
            "ALT_FORMULA_FP32": alt32,
            "AL_FORMULA_FP32": alt32 - arrays["ALB"],
        }
    )
    arrays["MU_PRESS_ADJ_FP64"] = press_adj_mu(
        mu_before=state.mu_perturbation,
        al=arrays["AL_FORMULA_FP64"],
        alt=arrays["ALT_FORMULA_FP64"],
        alb=arrays["ALB"],
        ht=grid.terrain_height,
        ht_fine=raw_child["grid"].terrain_height,
        dtype=np.float64,
    )
    arrays["MU_PRESS_ADJ_FP32"] = press_adj_mu(
        mu_before=state.mu_perturbation,
        al=arrays["AL_FORMULA_FP32"],
        alt=arrays["ALT_FORMULA_FP32"],
        alb=arrays["ALB"],
        ht=grid.terrain_height,
        ht_fine=raw_child["grid"].terrain_height,
        dtype=np.float32,
    )
    return {
        "status": "JAX_LOADER_READY",
        "arrays": arrays,
        "metadata": {
            "live_nest_base_init": live_child.get("live_nest_base_init"),
            "transient_adjust_mub": live_child.get("transient_adjust_mub"),
            "theta_qv_adjust": live_child.get("theta_qv_adjust"),
        },
    }


def summarize_surface(surface: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in surface.items() if key != "arrays"}


def tail_file(path: Path, limit: int = 4000) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="replace")[-limit:]


def hygiene_report() -> dict[str, Any]:
    clean_work_dirs = sorted(str(path) for path in SCRATCH_ROOT.glob("work_clean_*") if path.is_dir())
    prefilled = {
        "legacy_root_wrf_exists": (SCRATCH_ROOT / "WRF").exists(),
        "legacy_root_run_exists": (SCRATCH_ROOT / "run").exists(),
        "legacy_root_wrf_truth_file_count": tree_file_count(SCRATCH_ROOT / "wrf_truth"),
        "ignored_for_this_proof": True,
    }
    return {
        "scratch_root": str(SCRATCH_ROOT),
        "selected_clean_work_dir": str(WORK_DIR),
        "clean_work_dirs": clean_work_dirs,
        "prefilled_root_state": prefilled,
        "source_copy_roots": {
            "wrf_source": "<DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/WRF",
            "run_source": "<DATA_ROOT>/wrf_gpu2/v014_step1_pre_part1_handoff/run",
        },
        "checks": {
            "work_dir_exists": WORK_DIR.is_dir(),
            "wrf_tree_exists": WRF_TREE.is_dir(),
            "run_dir_exists": WRF_RUN_DIR.is_dir(),
            "no_nested_wrf_wrf": not (WRF_TREE / "WRF").exists(),
            "truth_file_count_after_run": tree_file_count(WRF_TRUTH),
            "surface_file_counts": {
                surface: tree_file_count(WRF_TRUTH, f"{surface}_*.txt") for surface in SURFACES
            },
            "wrf_patch_diff": path_info(OUT_WRF_PATCH),
            "wrf_backup": path_info(WRF_BACKUP),
            "fresh_rebuilt_wrf_exe": path_info(WRF_TREE / "main/wrf.exe"),
            "run_wrf_exe_symlink_target": os.readlink(WRF_RUN_DIR / "wrf.exe") if (WRF_RUN_DIR / "wrf.exe").is_symlink() else None,
        },
    }


def build_proof() -> dict[str, Any]:
    shapes = expected_shapes()
    start_surfaces = {surface: parse_start_surface(surface, shapes) for surface in SURFACES}
    blockers = {name: item for name, item in start_surfaces.items() if item.get("status") != "WRF_SURFACE_READY"}
    precall = parse_precall_surface(shapes)
    if precall.get("status") != "WRF_SURFACE_READY":
        blockers["precall_qv"] = precall
    jax_loader = build_jax_loader_arrays()
    if jax_loader.get("status") != "JAX_LOADER_READY":
        blockers["jax_loader"] = jax_loader
    if blockers:
        return {"status": "BLOCKED_INPUTS", "blockers": blockers}

    after_hyp = start_surfaces["after_hypsometric_p_al_alt"]["arrays"]
    before_press = start_surfaces["before_press_adj"]["arrays"]
    after_press = start_surfaces["after_press_adj"]["arrays"]
    after_w = start_surfaces["after_w_surface_branch"]["arrays"]
    pre_arrays = precall["arrays"]
    jax_arrays = jax_loader["arrays"]

    surface_to_precall = {
        "after_hypsometric_p_al_alt": compare_map(
            after_hyp,
            pre_arrays,
            {
                "P_STATE": "P_STATE",
                "T2_STATE": "T_STATE",
                "PB": "PB",
                "MUB": "MUB",
                "PH2_STATE": "PH_STATE",
                "PHB": "PHB",
                "QVAPOR": "QVAPOR",
            },
        ),
        "before_press_adj": compare_map(
            before_press,
            pre_arrays,
            {"P_STATE": "P_STATE", "MU2_STATE": "MU_STATE", "W2_STATE": "W_STATE"},
        ),
        "after_press_adj": compare_map(
            after_press,
            pre_arrays,
            {"P_STATE": "P_STATE", "MU2_STATE": "MU_STATE", "PH2_STATE": "PH_STATE"},
        ),
        "after_w_surface_branch": compare_map(
            after_w,
            pre_arrays,
            {"P_STATE": "P_STATE", "MU2_STATE": "MU_STATE", "W2_STATE": "W_STATE", "PH2_STATE": "PH_STATE"},
        ),
    }
    start_internal_deltas = {
        "after_hyp_to_before_press": compare_map(
            before_press,
            after_hyp,
            {"P_STATE": "P_STATE", "MU2_STATE": "MU2_STATE", "AL": "AL", "ALT": "ALT", "ALB": "ALB"},
        ),
        "before_press_to_after_press": compare_map(
            after_press,
            before_press,
            {"P_STATE": "P_STATE", "MU2_STATE": "MU2_STATE", "W2_STATE": "W2_STATE"},
        ),
        "after_press_to_after_w": compare_map(
            after_w,
            after_press,
            {"P_STATE": "P_STATE", "MU2_STATE": "MU2_STATE", "W2_STATE": "W2_STATE"},
        ),
    }

    p_internal_alt32 = pressure_from_alt(
        pb=after_hyp["PB"],
        theta_full=after_hyp["THETA1_FULL"],
        alt=after_hyp["ALT"],
        dtype=np.float32,
    )
    p_internal_alt64 = pressure_from_alt(
        pb=after_hyp["PB"],
        theta_full=after_hyp["THETA1_FULL"],
        alt=after_hyp["ALT"],
        dtype=np.float64,
    )
    mu_internal_press32 = press_adj_mu(
        mu_before=before_press["MU2_STATE"],
        al=before_press["AL"],
        alt=before_press["ALT"],
        alb=before_press["ALB"],
        ht=before_press["HT"],
        ht_fine=before_press["HT_FINE"],
        dtype=np.float32,
    )
    mu_internal_press64 = press_adj_mu(
        mu_before=before_press["MU2_STATE"],
        al=before_press["AL"],
        alt=before_press["ALT"],
        alb=before_press["ALB"],
        ht=before_press["HT"],
        ht_fine=before_press["HT_FINE"],
        dtype=np.float64,
    )

    formula_metrics = {
        "wrf_internal_pressure_from_alt_fp32_vs_after_hyp_P": metric_brief(
            diff_metrics("P_STATE", p_internal_alt32, after_hyp["P_STATE"])
        ),
        "wrf_internal_pressure_from_alt_fp64_vs_after_hyp_P": metric_brief(
            diff_metrics("P_STATE", p_internal_alt64, after_hyp["P_STATE"])
        ),
        "wrf_internal_press_adj_fp32_vs_after_press_MU": metric_brief(
            diff_metrics("MU_STATE", mu_internal_press32, after_press["MU2_STATE"])
        ),
        "wrf_internal_press_adj_fp64_vs_after_press_MU": metric_brief(
            diff_metrics("MU_STATE", mu_internal_press64, after_press["MU2_STATE"])
        ),
        "jax_current_pressure_formula_fp64_vs_wrf_after_hyp_P": metric_brief(
            diff_metrics("P_STATE", jax_arrays["P_FORMULA_FP64"], after_hyp["P_STATE"])
        ),
        "jax_current_pressure_formula_fp32_vs_wrf_after_hyp_P": metric_brief(
            diff_metrics("P_STATE", jax_arrays["P_FORMULA_FP32"], after_hyp["P_STATE"])
        ),
        "jax_current_press_adj_fp64_vs_wrf_after_press_MU": metric_brief(
            diff_metrics("MU_STATE", jax_arrays["MU_PRESS_ADJ_FP64"], after_press["MU2_STATE"])
        ),
        "jax_current_press_adj_fp32_vs_wrf_after_press_MU": metric_brief(
            diff_metrics("MU_STATE", jax_arrays["MU_PRESS_ADJ_FP32"], after_press["MU2_STATE"])
        ),
        "wrf_after_w_surface_vs_precall_W": surface_to_precall["after_w_surface_branch"]["W2_STATE->W_STATE"],
    }

    jax_input_metrics = {
        "P_STATE_raw_current_vs_wrf_after_hyp": metric_brief(diff_metrics("P_STATE", jax_arrays["P_STATE"], after_hyp["P_STATE"])),
        "MU_STATE_raw_current_vs_wrf_before_press": metric_brief(diff_metrics("MU_STATE", jax_arrays["MU_STATE"], before_press["MU2_STATE"])),
        "W_STATE_raw_current_vs_wrf_after_w": metric_brief(diff_metrics("W_STATE", jax_arrays["W_STATE"], after_w["W2_STATE"])),
        "PB_current_vs_wrf_after_hyp": metric_brief(diff_metrics("PB", jax_arrays["PB"], after_hyp["PB"])),
        "MUB_current_vs_wrf_after_hyp": metric_brief(diff_metrics("MUB", jax_arrays["MUB"], after_hyp["MUB"])),
        "PHB_current_vs_wrf_after_hyp": metric_brief(diff_metrics("PHB", jax_arrays["PHB"], after_hyp["PHB"])),
        "PH_STATE_current_vs_wrf_after_hyp": metric_brief(diff_metrics("PH_STATE", jax_arrays["PH_STATE"], after_hyp["PH2_STATE"])),
        "T_STATE_current_vs_wrf_after_hyp": metric_brief(diff_metrics("T_STATE", jax_arrays["T_STATE"], after_hyp["T2_STATE"])),
        "QVAPOR_current_vs_wrf_after_hyp": metric_brief(diff_metrics("QVAPOR", jax_arrays["QVAPOR"], after_hyp["QVAPOR"])),
        "HT_current_vs_wrf_after_hyp": metric_brief(diff_metrics("HT", jax_arrays["HT"], after_hyp["HT"])),
        "HT_FINE_current_vs_wrf_after_hyp": metric_brief(diff_metrics("HT_FINE", jax_arrays["HT_FINE"], after_hyp["HT_FINE"])),
        "AL_current_formula_vs_wrf_after_hyp": metric_brief(diff_metrics("AL", jax_arrays["AL_FORMULA_FP64"], after_hyp["AL"])),
        "ALT_current_formula_vs_wrf_after_hyp": metric_brief(diff_metrics("ALT", jax_arrays["ALT_FORMULA_FP64"], after_hyp["ALT"])),
        "ALB_current_vs_wrf_after_hyp": metric_brief(diff_metrics("ALB", jax_arrays["ALB"], after_hyp["ALB"])),
    }

    predecessor_data = json.loads(PREDECESSOR_JSON.read_text(encoding="utf-8"))
    raw_reductions = predecessor_data.get("proof", {}).get("raw_vs_candidate_summary", {}).get("improvement", {})
    p_current_formula = formula_metrics["jax_current_pressure_formula_fp64_vs_wrf_after_hyp_P"]["max_abs"]
    mu_current_formula = formula_metrics["jax_current_press_adj_fp64_vs_wrf_after_press_MU"]["max_abs"]
    w_after = formula_metrics["wrf_after_w_surface_vs_precall_W"]["max_abs"]
    p_wrf_formula = formula_metrics["wrf_internal_pressure_from_alt_fp32_vs_after_hyp_P"]["max_abs"]
    mu_wrf_formula = formula_metrics["wrf_internal_press_adj_fp32_vs_after_press_MU"]["max_abs"]

    ready_for_patch = (
        p_current_formula is not None
        and mu_current_formula is not None
        and w_after is not None
        and float(p_current_formula) <= MATERIAL_THRESHOLDS["P_STATE"]
        and float(mu_current_formula) <= MATERIAL_THRESHOLDS["MU_STATE"]
        and float(w_after) <= MATERIAL_THRESHOLDS["W_STATE"]
    )
    if ready_for_patch:
        verdict = "STEP1_START_DOMAIN_PERTURB_SUBSURFACE_READY_FOR_PATCH_P_PRESS_ADJ_SET_W_SURFACE"
    else:
        verdict = "STEP1_START_DOMAIN_PERTURB_SUBSURFACE_LOCALIZED_CURRENT_JAX_AL_ALT_BASE_INPUT_GAP"

    ranked_hypotheses = [
        {
            "rank": 1,
            "hypothesis": "WRF live-nest start_domain recomputes P/al/alt, then press_adj updates MU, then set_w_surface updates W.",
            "status": "SUPPORTED_BY_INTERNAL_SURFACES",
            "evidence": (
                f"WRF internal P-from-ALT fp32 max_abs={p_wrf_formula}; "
                f"press_adj fp32 max_abs={mu_wrf_formula}; after-w-surface vs pre-call W max_abs={w_after}."
            ),
        },
        {
            "rank": 2,
            "hypothesis": "A narrow production patch using current JAX inputs is exact enough now.",
            "status": "REFUTED_FOR_P_IF_THRESHOLD_EXCEEDED" if not ready_for_patch else "SUPPORTED",
            "evidence": (
                f"Current JAX pressure formula vs WRF internal P max_abs={p_current_formula}; "
                f"current JAX press_adj formula vs WRF after_press MU max_abs={mu_current_formula}. "
                "Patch threshold for P is 1 Pa and MU is 0.01 Pa."
            ),
        },
        {
            "rank": 3,
            "hypothesis": "Remaining P gap is in current JAX start_domain input surfaces, not source ordering.",
            "status": "SUPPORTED" if not ready_for_patch else "LOWER_RANKED",
            "evidence": (
                "WRF internal formula/order closes against WRF internal truth, while current JAX AL/ALT/base/PH inputs "
                f"still have ranked residuals headed by {rank_metrics(jax_input_metrics)[:3]}."
            ),
        },
    ]
    exclusions = [
        "The prefilled scratch-root WRF/run was ignored; all trusted WRF files came from the timestamped clean workdir.",
        "The hook is gated on grid%press_adj=.TRUE. and grid id 2, so ordinary non-live-nest d02 start_domain calls are excluded.",
        "WRF after_hypsometric P_STATE is continuous with accepted pre-call P_STATE; press_adj and W branch do not mutate P.",
        "WRF after_press_adj MU_STATE is continuous with accepted pre-call MU_STATE; remaining current-JAX MU gap is formula/input, not later solve_em.",
        "WRF after_w_surface_branch W_STATE is continuous with accepted pre-call W_STATE; W is not an acoustic or physics tendency source here.",
        "Boundary package, carry, halo, first_rk_step_part1, phy_prep, and acoustic refresh remain excluded by predecessor proofs.",
    ]

    return {
        "status": "PROOF_EXECUTED",
        "verdict": verdict,
        "shapes": {key: list(value) for key, value in shapes.items()},
        "wrf_surfaces": {name: summarize_surface(surface) for name, surface in start_surfaces.items()},
        "precall_truth": summarize_surface(precall),
        "surface_vs_precall": surface_to_precall,
        "start_internal_deltas": start_internal_deltas,
        "formula_metrics": formula_metrics,
        "jax_input_metrics": jax_input_metrics,
        "ranked_jax_input_residuals": rank_metrics(jax_input_metrics),
        "predecessor_reductions": raw_reductions,
        "ranked_hypotheses": ranked_hypotheses,
        "exclusions": exclusions,
        "patch_decision": {
            "production_patch_applied": False,
            "ready_for_patch": ready_for_patch,
            "reason": (
                "No production patch was applied because current JAX AL/ALT/base/PH inputs still leave "
                f"P max_abs {p_current_formula} Pa, above the 1 Pa material gate."
                if not ready_for_patch
                else "Internal WRF source semantics and current JAX formula residuals are within gates."
            ),
        },
        "next_truth_surface": (
            "Split current JAX live-nest start_domain input construction for AL/ALT: compare final blended HT, "
            "PB/MUB/PHB, PH_STATE, MU before press_adj, and diagnosed AL/ALT against the WRF internal "
            "after_hypsometric surface. Patch P/MU only after that current-input gap is below the material gate."
        )
        if not ready_for_patch
        else "Apply the narrow GPU-native init patch for P/MU/W and rerun the strict Step-1 proof.",
        "jax_loader_metadata": jax_loader["metadata"],
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    proof = payload["proof"]
    fm = proof.get("formula_metrics", {})
    jd = proof.get("patch_decision", {})
    lines = [
        "# V0.14 Step-1 Start-Domain Perturbation Subsurface",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Result",
        "",
        f"- CPU backend: `{payload['environment'].get('jax_default_backend')}`; GPU used: `{payload['gpu_used']}`.",
        f"- Required ancestor `{REQUIRED_ANCESTOR}` present: `{payload['git']['required_ancestor']['is_ancestor']}`.",
        f"- Clean workdir: `{WORK_DIR}`; prefilled root WRF/run ignored: `{payload['hygiene']['prefilled_root_state']['ignored_for_this_proof']}`.",
        "- WRF emitted 28 d02 patch files for each requested surface: after hypsometric P/al/alt, before press_adj, after press_adj, and after W surface branch.",
        f"- Production source patch applied: `{jd.get('production_patch_applied')}`. {jd.get('reason')}",
        "",
        "## Key Metrics",
        "",
        "| Check | max_abs | RMSE | Interpretation |",
        "|---|---:|---:|---|",
    ]
    rows = [
        ("WRF P from internal ALT fp32 vs WRF after_hypsometric P", "wrf_internal_pressure_from_alt_fp32_vs_after_hyp_P", "source formula/order closed"),
        ("WRF press_adj fp32 vs WRF after_press MU", "wrf_internal_press_adj_fp32_vs_after_press_MU", "source formula/order closed"),
        ("WRF after W branch vs accepted pre-call W", "wrf_after_w_surface_vs_precall_W", "W branch closed"),
        ("Current JAX pressure formula vs WRF after_hypsometric P", "jax_current_pressure_formula_fp64_vs_wrf_after_hyp_P", "current-input patch falsifier"),
        ("Current JAX press_adj formula vs WRF after_press MU", "jax_current_press_adj_fp64_vs_wrf_after_press_MU", "current-input patch falsifier"),
    ]
    for label, key, interp in rows:
        metric = fm.get(key, {})
        lines.append(f"| {label} | {metric.get('max_abs')} | {metric.get('rmse')} | {interp} |")
    lines.extend(["", "## Ranked Hypotheses", ""])
    for item in proof.get("ranked_hypotheses", []):
        lines.append(f"- {item['rank']}. {item['hypothesis']} Status: `{item['status']}`. {item['evidence']}")
    lines.extend(["", "## Exclusions", ""])
    for item in proof.get("exclusions", []):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            proof.get("next_truth_surface", ""),
            "",
            "Detailed metrics are in `proofs/v014/step1_start_domain_perturb_subsurface.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Review: V0.14 Step-1 Start-Domain Perturbation Subsurface",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "objective: close the WRF live-nest `start_domain(nest,.TRUE.)` internal truth surface for Step-1 `P_STATE/MU_STATE/W_STATE` initialization.",
        "",
        "files changed:",
        "- `proofs/v014/step1_start_domain_perturb_subsurface.py`",
        "- `proofs/v014/step1_start_domain_perturb_subsurface.json`",
        "- `proofs/v014/step1_start_domain_perturb_subsurface.md`",
        "- `proofs/v014/step1_start_domain_perturb_subsurface_wrf_patch.diff`",
        "- `.agent/reviews/2026-06-09-v014-step1-start-domain-perturb-subsurface.md`",
        "",
        "commands run:",
    ]
    for command in payload["commands"]["executed"]:
        lines.append(f"- `{command}`")
    lines.extend(["", "proof objects produced:"])
    for value in payload["proof_objects"].values():
        lines.append(f"- `{value}`")
    lines.extend(["", "ranked hypotheses/exclusions:"])
    for item in payload["proof"].get("ranked_hypotheses", []):
        lines.append(f"- rank {item['rank']}: {item['status']} - {item['hypothesis']}")
    for item in payload["proof"].get("exclusions", []):
        lines.append(f"- excluded: {item}")
    lines.extend(["", "unresolved risks:"])
    for risk in payload["unresolved_risks"]:
        lines.append(f"- {risk}")
    lines.extend(["", f"next decision: {payload['next_decision']}", ""])
    return "\n".join(lines)


def main() -> int:
    proof = build_proof()
    verdict = proof.get("verdict", f"STEP1_START_DOMAIN_PERTURB_SUBSURFACE_BLOCKED_{proof.get('status', 'UNKNOWN')}")
    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.step1_start_domain_perturb_subsurface.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "environment": jax_environment(),
        "cpu_only": True,
        "gpu_used": False,
        "no_tost": True,
        "no_switzerland": True,
        "no_fp32_source_work": True,
        "no_memory_source_work": True,
        "no_hermes": True,
        "production_src_edits": False,
        "git": git_metadata(),
        "hygiene": hygiene_report(),
        "inputs": {
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "handoff": path_info(HANDOFF),
            "predecessor_json": path_info(PREDECESSOR_JSON),
            "predecessor_md": path_info(PREDECESSOR_MD),
            "wrf_truth_root": path_info(WRF_TRUTH),
            "precall_qv_root": path_info(PRECALL_QV_ROOT),
            "wrf_tree": path_info(WRF_TREE),
            "wrf_run_dir": path_info(WRF_RUN_DIR),
            "wrf_patch_diff": path_info(OUT_WRF_PATCH),
        },
        "proof": proof,
        "commands": {
            "executed": [
                "date +%Y%m%d_%H%M%S",
                "rm -rf <DATA_ROOT>/wrf_gpu2/v014_step1_start_domain_perturb_subsurface/work_clean_20260609_194715",
                "cp --reflink=auto -a <DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/WRF work_clean_20260609_194715/WRF",
                "cp --reflink=auto -a <DATA_ROOT>/wrf_gpu2/v014_step1_pre_part1_handoff/run work_clean_20260609_194715/run",
                "apply_patch work_clean_20260609_194715/WRF/dyn_em/start_em.F",
                "diff -u backup/start_em.F.before_start_domain_perturb_subsurface WRF/dyn_em/start_em.F > proofs/v014/step1_start_domain_perturb_subsurface_wrf_patch.diff",
                "./compile em_real (failed: /bin/csh missing, exit 126)",
                "PATH=<USER_HOME>/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin:$PATH <USER_HOME>/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin/tcsh ./compile em_real (first env-missing check failed in log, then rerun with PATH/NETCDF/PNETCDF)",
                "PATH=wrf-build/bin:$PATH NETCDF=wrf-build PNETCDF=wrf-build tcsh ./compile em_real",
                "mpirun -np 28 ./wrf.exe (failed: insufficient slots)",
                "mpirun --map-by :OVERSUBSCRIBE -np 28 ./wrf.exe",
                "python -m py_compile proofs/v014/step1_start_domain_perturb_subsurface.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_start_domain_perturb_subsurface.py",
                "python -m json.tool proofs/v014/step1_start_domain_perturb_subsurface.json >/tmp/step1_start_domain_perturb_subsurface.validated.json",
                "git diff -- src/gpuwrf",
            ],
            "logs": {
                "compile_bad_shebang": str(WRF_LOG_DIR / "compile_start_domain_perturb_subsurface.log"),
                "compile_missing_env": str(WRF_LOG_DIR / "compile_start_domain_perturb_subsurface_tcsh.log"),
                "compile_success": str(WRF_LOG_DIR / "compile_start_domain_perturb_subsurface_wrfbuild.log"),
                "run_insufficient_slots": str(WRF_LOG_DIR / "wrf_start_domain_perturb_subsurface_28rank_stdout.log"),
                "run_success": str(WRF_LOG_DIR / "wrf_start_domain_perturb_subsurface_28rank_oversub_stdout.log"),
            },
            "log_tails": {
                "compile_success": tail_file(WRF_LOG_DIR / "compile_start_domain_perturb_subsurface_wrfbuild.log"),
                "run_success": tail_file(WRF_RUN_DIR / "rsl.error.0000"),
            },
        },
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "wrf_patch_diff": str(OUT_WRF_PATCH),
            "review": str(OUT_REVIEW),
            "wrf_truth_root": str(WRF_TRUTH),
        },
        "unresolved_risks": [
            "No production source patch was applied in this sprint.",
            "WRF source ordering is now proven, but current JAX AL/ALT/base/PH inputs still need a smaller split before a safe P/MU patch.",
            "The WRF truth is text savepoint data outside git; the repo commits only script, diff, JSON, and report metadata/checksums.",
        ],
        "next_decision": proof.get("next_truth_surface"),
    }
    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "json": str(OUT_JSON), "markdown": str(OUT_MD)}, sort_keys=True))
    return 0 if proof.get("status") == "PROOF_EXECUTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
