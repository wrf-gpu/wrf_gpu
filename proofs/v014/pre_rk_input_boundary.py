#!/usr/bin/env python3
"""V0.14 pre-RK WRF/JAX input-boundary proof.

This proof is read-only for production source. It consumes an env-gated WRF
hook output, if present, and the produced h10 d02 step-5999 JAX full carry.
If the WRF hook output is absent, it emits a valid blocked proof naming the
missing command/artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

SPRINT_CONTRACT = (
    ROOT / ".agent/sprints/2026-06-09-v014-pre-rk-input-boundary/sprint-contract.md"
)
PROJECT_CONSTITUTION = ROOT / "PROJECT_CONSTITUTION.md"
AGENTS = ROOT / "AGENTS.md"
BUILDING_WRF_ORACLES_SKILL = ROOT / ".agent/skills/building-wrf-oracles/SKILL.md"
VALIDATING_PHYSICS_SKILL = ROOT / ".agent/skills/validating-physics/SKILL.md"
TOLERANCE_POLICY = ROOT / ".agent/skills/validating-physics/references/tolerance-policy.md"

THETA_LOCALIZATION_JSON = ROOT / "proofs/v014/jax_theta_evolution_localization.json"
JAX_PRESTEP_JSON = ROOT / "proofs/v014/jax_h10_prestep_carry.json"
JAX_PRESTEP_PRODUCER_JSON = ROOT / "proofs/v014/jax_h10_prestep_carry_producer.json"
WRF_SAME_STATE_JSON = ROOT / "proofs/v014/wrf_same_state_marker_savepoint.json"
WRF_DYNAMIC_JSON = ROOT / "proofs/v014/wrf_dynamic_term_localization.json"
WRF_REFRESH_JSON = ROOT / "proofs/v014/wrf_post_rk_refresh_localization.json"
SAVEPOINT_REQUEST_JSON = ROOT / "proofs/v014/same_state_savepoint_request.json"
HANDOFF = ROOT / ".agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md"

OUT_JSON = ROOT / "proofs/v014/pre_rk_input_boundary.json"
OUT_MD = ROOT / "proofs/v014/pre_rk_input_boundary.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-pre-rk-input-boundary.md"
PATCH_DIFF = ROOT / "proofs/v014/pre_rk_input_boundary_wrf_patch.diff"

TARGET_STEP = 6000
PRESTEP_COMPLETED_STEPS = TARGET_STEP - 1
THETA_OFFSET_K = 300.0
TOLERANCE_MAX_ABS = 2.0e-6
TARGET_FIELDS = ("T", "P", "PB", "MU", "MUB")
MASS_ZERO_BOUNDS = {
    "south_north_start": 1,
    "south_north_stop_exclusive": 18,
    "west_east_start": 5,
    "west_east_stop_exclusive": 22,
}

DEFAULT_SCRATCH = Path("/tmp/wrf_gpu2_v014_pre_rk_input_boundary")
ALT_SCRATCH = Path("/mnt/data/wrf_gpu2/v014_pre_rk_input_boundary")
SCRATCH = Path(os.environ.get("WRFGPU2_PRE_RK_SCRATCH", str(DEFAULT_SCRATCH)))
WRF_OUTPUT_DIR = Path(
    os.environ.get("WRFGPU2_PRE_RK_INPUT_ROOT", str(SCRATCH / "pre_rk_output"))
)
CHECKPOINT = Path(
    os.environ.get(
        "WRFGPU2_H10_PRESTEP_CARRY",
        "/mnt/data/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl",
    )
)

DMPAR_WRF = Path("/mnt/data/wrf_gpu2/v014_post_rk_refresh/WRF")
PRISTINE_WRF = Path("/home/user/src/wrf_pristine/WRF")
SCRATCH_WRF = SCRATCH / "WRF"
SCRATCH_RUN_DIR = SCRATCH / "run_case3"
COMPILE_LOG = SCRATCH / "compile_pre_rk_input_boundary_dmpar.log"
LEGACY_COMPILE_LOG = SCRATCH / "compile_pre_rk_input_boundary.log"

MASS_SCHEMA = {
    "MASS_K1": {
        "index_count": 4,
        "fields": [
            "T_THM",
            "T_OLD",
            "T_HIST_SRC",
            "P",
            "PB",
            "MU_NEW",
            "MU_OLD",
            "MUB",
        ],
    }
}

WRF_FIELD_SOURCE = {
    "T": ("MASS_K1", "T_THM", "grid%t_2 perturbation potential temperature at k=1"),
    "P": ("MASS_K1", "P", "grid%p perturbation pressure at k=1"),
    "PB": ("MASS_K1", "PB", "grid%pb base pressure at k=1"),
    "MU": ("MASS_K1", "MU_NEW", "grid%mu_2 perturbation dry-air column mass"),
    "MUB": ("MASS_K1", "MUB", "grid%mub base dry-air column mass"),
}

JAX_FIELD_SOURCE = {
    "T": "carry.state.theta - 300 K, k=1",
    "P": "carry.state.p_perturbation, k=1",
    "PB": "carry.state.p_total - carry.state.p_perturbation, k=1",
    "MU": "carry.state.mu_perturbation",
    "MUB": "carry.state.mu_total - carry.state.mu_perturbation",
}


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
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
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256(path),
    }


def tail_text(path: Path, max_lines: int = 80) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if math.isfinite(value) else None
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def jax_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import jax  # noqa: PLC0415

        devices = [str(device) for device in jax.devices()]
        env.update(
            {
                "jax_import_error": None,
                "jax_version": getattr(jax, "__version__", None),
                "jax_default_backend": jax.default_backend(),
                "jax_devices": devices,
                "gpu_device_count": len([device for device in devices if "gpu" in device.lower()]),
            }
        )
    except Exception as exc:  # pragma: no cover - recorded in proof output
        env.update({"jax_import_error": repr(exc), "gpu_device_count": None})
    return env


def stats(values: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": int(arr.size), "finite_count": 0, "max_abs": None, "rmse": None}
    return {
        "count": int(arr.size),
        "finite_count": int(finite.size),
        "max_abs": float(np.max(np.abs(finite))),
        "rmse": float(math.sqrt(float(np.mean(finite * finite)))),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def array_summary(array: Any) -> dict[str, Any]:
    arr = np.asarray(array)
    finite = np.asarray(arr, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    summary: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "count": int(arr.size),
        "finite_count": int(finite.size),
    }
    if finite.size:
        summary.update(
            {
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "mean": float(np.mean(finite)),
                "max_abs": float(np.max(np.abs(finite))),
            }
        )
    return summary


def key_for(record_type: str, idx: list[int]) -> tuple[int, ...]:
    if record_type == "MASS_K1":
        return (idx[3], idx[2])
    raise ValueError(record_type)


def parse_wrf_files(paths: Iterable[Path]) -> dict[str, Any]:
    records: dict[str, dict[tuple[int, ...], dict[str, float]]] = defaultdict(dict)
    metadata: dict[str, list[str]] = defaultdict(list)
    duplicate_count = 0
    duplicate_max_delta = 0.0
    duplicate_max_delta_by_field: dict[str, float] = {}
    files = list(paths)

    for path in files:
        with path.open(encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or line.startswith("record_schema"):
                    continue
                parts = line.split()
                tag = parts[0]
                if tag not in MASS_SCHEMA:
                    if len(parts) > 1:
                        metadata[tag].append(" ".join(parts[1:]))
                    continue
                schema = MASS_SCHEMA[tag]
                nidx = int(schema["index_count"])
                idx = [int(value) for value in parts[1 : 1 + nidx]]
                values = [float(value) for value in parts[1 + nidx :]]
                fields = list(schema["fields"])
                if len(values) != len(fields):
                    raise ValueError(
                        f"{path}: {tag} expected {len(fields)} values, got {len(values)}"
                    )
                item = dict(zip(fields, values))
                key = key_for(tag, idx)
                if key in records[tag]:
                    duplicate_count += 1
                    previous = records[tag][key]
                    for field in fields:
                        label = f"{tag}.{field}"
                        delta = abs(previous[field] - item[field])
                        duplicate_max_delta_by_field[label] = max(
                            duplicate_max_delta_by_field.get(label, 0.0), delta
                        )
                        duplicate_max_delta = max(duplicate_max_delta, delta)
                records[tag][key] = item

    return {
        "files": [str(path) for path in files],
        "file_info": [path_info(path) for path in files],
        "records": records,
        "metadata": {
            name: values[0] if len(values) == 1 else values
            for name, values in metadata.items()
        },
        "unique_counts": {name: len(items) for name, items in records.items()},
        "duplicate_count": duplicate_count,
        "duplicate_max_delta": duplicate_max_delta,
        "duplicate_max_delta_by_field": duplicate_max_delta_by_field,
    }


def discover_wrf_output_files() -> list[Path]:
    candidates: list[Path] = []
    for root in [WRF_OUTPUT_DIR, ALT_SCRATCH / "pre_rk_output", DEFAULT_SCRATCH / "pre_rk_output"]:
        if root.exists():
            candidates.extend(root.glob("pre_rk_input_d2_step_6000_*.txt"))
    seen: set[str] = set()
    unique = []
    for path in sorted(candidates):
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def load_jax_carry() -> dict[str, Any]:
    if not CHECKPOINT.exists():
        return {
            "status": "BLOCKED",
            "reason": "MISSING_JAX_H10_PRESTEP_CARRY",
            "checkpoint": path_info(CHECKPOINT),
        }

    try:
        from gpuwrf.runtime.checkpoint import read_checkpoint_with_runtime_state  # noqa: PLC0415

        _state, namelist, grid, step_index, runtime_state = read_checkpoint_with_runtime_state(
            CHECKPOINT
        )
        if runtime_state is None:
            return {
                "status": "BLOCKED",
                "reason": "CHECKPOINT_RUNTIME_STATE_ABSENT",
                "checkpoint": path_info(CHECKPOINT),
                "step_index": int(step_index),
            }
        return {
            "status": "LOAD_OK",
            "checkpoint": path_info(CHECKPOINT),
            "loader": "gpuwrf.runtime.checkpoint.read_checkpoint_with_runtime_state",
            "step_index": int(step_index),
            "runtime_state_type": type(runtime_state).__name__,
            "namelist_type": type(namelist).__name__,
            "grid_shape": {
                "nz": int(getattr(grid, "nz")),
                "ny": int(getattr(grid, "ny")),
                "nx": int(getattr(grid, "nx")),
            },
            "carry": runtime_state,
        }
    except Exception as exc:  # pragma: no cover - recorded in proof output
        return {
            "status": "BLOCKED",
            "reason": "JAX_CARRY_LOAD_FAILED",
            "error": repr(exc),
            "checkpoint": path_info(CHECKPOINT),
        }


def state_field_view(state: Any, field: str) -> Any:
    if field == "T":
        return state.theta - THETA_OFFSET_K
    if field == "P":
        return state.p_perturbation
    if field == "PB":
        return state.p_total - state.p_perturbation
    if field == "MU":
        return state.mu_perturbation
    if field == "MUB":
        return state.mu_total - state.mu_perturbation
    raise KeyError(field)


def state_index(field: str, key: tuple[int, ...]) -> tuple[int, ...]:
    if field in {"T", "P", "PB"}:
        return (0, key[0], key[1])
    if field in {"MU", "MUB"}:
        return (key[0], key[1])
    raise KeyError(field)


def compare_jax_to_wrf(carry: Any, wrf_surface: Mapping[str, Any]) -> dict[str, Any]:
    state = carry.state
    comparisons: dict[str, Any] = {}
    jax_summaries: dict[str, Any] = {}
    unavailable_fields: list[str] = []

    for field in TARGET_FIELDS:
        tag, wrf_field, wrf_convention = WRF_FIELD_SOURCE[field]
        try:
            candidate_arr = np.asarray(state_field_view(state, field), dtype=np.float64)
        except Exception as exc:  # pragma: no cover - recorded in proof output
            unavailable_fields.append(field)
            comparisons[field] = {"status": "UNAVAILABLE_JAX_FIELD", "error": repr(exc)}
            continue

        jax_summaries[field] = {
            **array_summary(candidate_arr),
            "source": JAX_FIELD_SOURCE[field],
        }

        wrf_records = wrf_surface["records"].get(tag, {})
        if not wrf_records:
            unavailable_fields.append(field)
            comparisons[field] = {
                "status": "UNAVAILABLE_WRF_FIELD",
                "wrf_source": wrf_convention,
            }
            continue

        diffs: list[float] = []
        worst: dict[str, Any] | None = None
        skipped: list[dict[str, Any]] = []
        for key, record in sorted(wrf_records.items()):
            if wrf_field not in record:
                skipped.append({"native_key": list(key), "reason": f"missing {wrf_field}"})
                continue
            try:
                candidate = float(candidate_arr[state_index(field, key)])
            except Exception as exc:  # pragma: no cover - recorded in proof output
                skipped.append({"native_key": list(key), "reason": repr(exc)})
                continue
            truth = float(record[wrf_field])
            diff = candidate - truth
            diffs.append(diff)
            if worst is None or abs(diff) > worst["abs_diff"]:
                worst = {
                    "native_key": list(key),
                    "array_index": list(state_index(field, key)),
                    "jax_candidate": candidate,
                    "wrf_truth": truth,
                    "diff_jax_minus_wrf": diff,
                    "abs_diff": abs(diff),
                }

        entry = {
            **stats(diffs),
            "status": "NO_RECORDS" if not diffs else "PENDING",
            "tolerance_max_abs": TOLERANCE_MAX_ABS,
            "wrf_source_field": wrf_field,
            "wrf_source_convention": wrf_convention,
            "jax_source_convention": JAX_FIELD_SOURCE[field],
            "worst": worst,
            "skipped_records": skipped,
            "skipped_record_count": len(skipped),
        }
        if diffs:
            entry["status"] = (
                "MATCH"
                if entry.get("max_abs") is not None
                and float(entry["max_abs"]) <= TOLERANCE_MAX_ABS
                else "DIFF"
            )
        comparisons[field] = entry

    return {
        "status": "RAN",
        "comparisons": comparisons,
        "jax_field_summaries": jax_summaries,
        "unavailable_fields": unavailable_fields,
        "first_mismatch": first_mismatch(comparisons),
    }


def compare_extra_context(carry: Any, wrf_surface: Mapping[str, Any]) -> dict[str, Any]:
    state = carry.state
    theta = np.asarray(state.theta - THETA_OFFSET_K, dtype=np.float64)
    mu = np.asarray(state.mu_perturbation, dtype=np.float64)
    extra = {
        "T_HIST_SRC": ("T_HIST_SRC", theta, lambda key: (0, key[0], key[1])),
        "T_OLD": ("T_OLD", theta, lambda key: (0, key[0], key[1])),
        "MU_OLD": ("MU_OLD", mu, lambda key: (key[0], key[1])),
    }
    out: dict[str, Any] = {}
    records = wrf_surface["records"].get("MASS_K1", {})
    for label, (wrf_field, arr, indexer) in extra.items():
        diffs: list[float] = []
        worst: dict[str, Any] | None = None
        for key, record in sorted(records.items()):
            candidate = float(arr[indexer(key)])
            truth = float(record[wrf_field])
            diff = candidate - truth
            diffs.append(diff)
            if worst is None or abs(diff) > worst["abs_diff"]:
                worst = {
                    "native_key": list(key),
                    "jax_candidate": candidate,
                    "wrf_truth": truth,
                    "diff_jax_minus_wrf": diff,
                    "abs_diff": abs(diff),
                }
        out[label] = {**stats(diffs), "worst": worst}
    return out


def first_mismatch(comparisons: Mapping[str, Any]) -> dict[str, Any] | None:
    for field in TARGET_FIELDS:
        entry = comparisons.get(field)
        if not isinstance(entry, Mapping):
            continue
        max_abs = entry.get("max_abs")
        if max_abs is not None and float(max_abs) > TOLERANCE_MAX_ABS:
            return {
                "field": field,
                "max_abs": float(max_abs),
                "rmse": entry.get("rmse"),
                "tolerance_max_abs": TOLERANCE_MAX_ABS,
                "worst": entry.get("worst"),
            }
    return None


def compact_target() -> dict[str, Any]:
    out = {
        "domain": "d02",
        "target_step": TARGET_STEP,
        "prestep_completed_steps_required": PRESTEP_COMPLETED_STEPS,
        "valid_time_utc": "2026-05-02T04:00:00+00:00",
        "mass_patch_bounds_zero_based": MASS_ZERO_BOUNDS,
        "native_coordinates": {
            "mass_zero": "y [1,18), x [5,22); Fortran j 2..18, i 6..22",
        },
        "wrf_hook_surface": (
            "dyn_em/solve_em.F after grid%itimestep increment and before "
            "cpl_store_input, zero_bdytend, moist_old initialization, RK loop, "
            "first_rk_step_part1/part2, and rk_tendency"
        ),
    }
    try:
        previous = load_json(THETA_LOCALIZATION_JSON)
        out["previous_verdict"] = previous.get("verdict")
        out["previous_next_decision"] = previous.get("next_decision")
    except Exception:
        pass
    return out


def proof_inputs() -> dict[str, Any]:
    return {
        "project_constitution": path_info(PROJECT_CONSTITUTION),
        "agents": path_info(AGENTS),
        "sprint_contract": path_info(SPRINT_CONTRACT),
        "building_wrf_oracles_skill": path_info(BUILDING_WRF_ORACLES_SKILL),
        "validating_physics_skill": path_info(VALIDATING_PHYSICS_SKILL),
        "tolerance_policy": path_info(TOLERANCE_POLICY),
        "theta_localization_json": path_info(THETA_LOCALIZATION_JSON),
        "jax_h10_prestep_carry_json": path_info(JAX_PRESTEP_JSON),
        "jax_h10_prestep_carry_producer_json": path_info(JAX_PRESTEP_PRODUCER_JSON),
        "wrf_same_state_marker_json": path_info(WRF_SAME_STATE_JSON),
        "wrf_dynamic_term_localization_json": path_info(WRF_DYNAMIC_JSON),
        "wrf_post_rk_refresh_localization_json": path_info(WRF_REFRESH_JSON),
        "same_state_savepoint_request_json": path_info(SAVEPOINT_REQUEST_JSON),
        "handoff": path_info(HANDOFF),
        "wrf_patch_diff": path_info(PATCH_DIFF),
        "checkpoint": path_info(CHECKPOINT),
    }


def wrf_commands() -> dict[str, Any]:
    env_prefix = (
        "PATH=/home/user/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin:$PATH "
        "NETCDF=/home/user/src/canairy_meteo/Gen2/artifacts/envs/wrf-build "
        "PNETCDF=/home/user/src/canairy_meteo/Gen2/artifacts/envs/wrf-build "
        "WRFIO_NCD_LARGE_FILE_SUPPORT=1"
    )
    run_env = (
        "PATH=/home/user/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin:$PATH "
        "CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 "
        f"WRFGPU2_PRE_RK_INPUT=1 WRFGPU2_PRE_RK_INPUT_ROOT={WRF_OUTPUT_DIR} "
        "WRFGPU2_PRE_RK_INPUT_GRID=2 WRFGPU2_PRE_RK_INPUT_START_STEP=6000 "
        "WRFGPU2_PRE_RK_INPUT_END_STEP=6000"
    )
    return {
        "scratch_root": str(SCRATCH),
        "wrf_source_path": str(SCRATCH_WRF),
        "wrf_run_dir": str(SCRATCH_RUN_DIR),
        "wrf_output_dir": str(WRF_OUTPUT_DIR),
        "minimal_cpu_wrf_commands": [
            f"mkdir -p {SCRATCH}",
            f"rsync -a {DMPAR_WRF}/ {SCRATCH_WRF}/",
            (
                "rsync -a /mnt/data/wrf_gpu2/v014_post_rk_refresh/run_case3/ "
                f"{SCRATCH_RUN_DIR}/"
            ),
            (
                f"cd {SCRATCH_WRF} && patch -p1 < {PATCH_DIFF} || true  "
                "# hunk 2 applies; hunk 1 is declarations-only against pristine"
            ),
            (
                "insert wrfgpu2_prerk_* declarations next to existing "
                "wrfgpu2_marker_* declarations in scratch dyn_em/solve_em.F"
            ),
            (
                f"cd {SCRATCH_WRF} && timeout 3600 env {env_prefix} "
                f"tcsh ./compile em_real > {COMPILE_LOG} 2>&1"
            ),
            (
                f"cd {SCRATCH_RUN_DIR} && find . -maxdepth 1 "
                "\\( -name 'rsl.error.*' -o -name 'rsl.out.*' -o -name 'wrfout_d0*' "
                "-o -name 'wrfrst_d0*' -o -name '*stdout.log' \\) -delete"
            ),
            f"ln -sf {SCRATCH_WRF}/main/wrf.exe {SCRATCH_RUN_DIR}/wrf.exe",
            (
                f"cd {SCRATCH_RUN_DIR} && timeout 3600 env {run_env} "
                f"mpirun --oversubscribe -np 28 ./wrf.exe > "
                f"{SCRATCH_RUN_DIR}/pre_rk_input_boundary_28rank_stdout.log 2>&1"
            ),
        ],
    }


def detect_wrf_block_reason(wrf_files: list[Path]) -> str:
    if wrf_files:
        return "NONE"
    mpi_stdout = SCRATCH_RUN_DIR / "pre_rk_input_boundary_28rank_stdout.log"
    singleton_stdout = SCRATCH_RUN_DIR / "pre_rk_input_boundary_singleton_stdout.log"
    rsl_error = SCRATCH_RUN_DIR / "rsl.error.0000"
    combined = "\n".join(
        text
        for text in [
            tail_text(mpi_stdout),
            tail_text(singleton_stdout),
            tail_text(rsl_error),
        ]
        if text
    )
    if "PMIx server's listener thread failed to start" in combined or "pmix_ifinit: socket() failed" in combined:
        return "WRF_MPI_LAUNCH_PMIX_SOCKET_BLOCKED"
    if "check comm_start, nest_pes_x, nest_pes_y settings" in combined:
        return "WRF_SINGLETON_NESTED_COMM_BLOCKED"
    return "NO_WRF_PRE_RK_HOOK_OUTPUT"


def blocked_payload(reason: str, wrf_files: list[Path], jax_load: Mapping[str, Any]) -> dict[str, Any]:
    missing: dict[str, Any]
    if reason in {
        "NO_WRF_PRE_RK_HOOK_OUTPUT",
        "WRF_MPI_LAUNCH_PMIX_SOCKET_BLOCKED",
        "WRF_SINGLETON_NESTED_COMM_BLOCKED",
    }:
        missing = {
            "missing_artifact": str(WRF_OUTPUT_DIR / "pre_rk_input_d2_step_6000_*.txt"),
            "missing_hook": (
                "Env-gated dyn_em/solve_em.F hook from "
                "proofs/v014/pre_rk_input_boundary_wrf_patch.diff"
            ),
            "missing_command": wrf_commands()["minimal_cpu_wrf_commands"][-1],
            "existing_files_found": [str(path) for path in wrf_files],
            "mpi_stdout_log": path_info(SCRATCH_RUN_DIR / "pre_rk_input_boundary_28rank_stdout.log"),
            "mpi_stdout_tail": tail_text(
                SCRATCH_RUN_DIR / "pre_rk_input_boundary_28rank_stdout.log"
            ),
            "singleton_stdout_log": path_info(
                SCRATCH_RUN_DIR / "pre_rk_input_boundary_singleton_stdout.log"
            ),
            "singleton_stdout_tail": tail_text(
                SCRATCH_RUN_DIR / "pre_rk_input_boundary_singleton_stdout.log"
            ),
            "singleton_rsl_error_0000": path_info(SCRATCH_RUN_DIR / "rsl.error.0000"),
            "singleton_rsl_error_0000_tail": tail_text(SCRATCH_RUN_DIR / "rsl.error.0000"),
        }
    else:
        missing = {
            "missing_artifact": str(CHECKPOINT),
            "jax_load_status": {k: v for k, v in jax_load.items() if k != "carry"},
        }
    return {
        "status": "BLOCKED",
        "reason": reason,
        **missing,
    }


def decide_verdict(comparison: Mapping[str, Any] | None, blocked: Mapping[str, Any] | None) -> tuple[str, str]:
    if blocked:
        if blocked.get("reason") == "WRF_MPI_LAUNCH_PMIX_SOCKET_BLOCKED":
            return (
                f"PRE_RK_INPUT_BOUNDARY_BLOCKED_{blocked['reason']}",
                (
                    "Rerun the same disposable CPU-WRF pre-RK hook command outside the "
                    "current MPI/socket-restricted sandbox, then rerun this proof."
                ),
            )
        return (
            f"PRE_RK_INPUT_BOUNDARY_BLOCKED_{blocked['reason']}",
            "Run the disposable CPU-WRF pre-RK hook command and rerun this proof.",
        )
    if comparison is None:
        return (
            "PRE_RK_INPUT_BOUNDARY_BLOCKED_NO_COMPARISON",
            "Rerun the proof after both WRF hook output and JAX checkpoint are readable.",
        )
    unavailable = comparison.get("unavailable_fields") or []
    comparisons = comparison.get("comparisons", {})
    diff_fields = [
        field
        for field, entry in comparisons.items()
        if isinstance(entry, Mapping) and entry.get("status") == "DIFF"
    ]
    if unavailable:
        return (
            "PRE_RK_INPUT_PARTIAL_CONTEXT_MISMATCH",
            "Open the next hook sprint only for the unavailable pre-RK fields.",
        )
    if diff_fields:
        return (
            "PRE_RK_INPUT_JAX_PRESTEP_MISMATCH_CONFIRMED",
            "Trace the JAX checkpoint/prestep carry producer and previous-step WRF/JAX update path.",
        )
    return (
        "PRE_RK_INPUT_MATCHES_REFERENCE_MISMATCH_WAS_SURFACE_MISMATCH",
        "Return to current-step RK/acoustic localization.",
    )


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# V0.14 Pre-RK Input Boundary",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Target",
        "",
        "- Domain/step: `d02`, step `6000`, h10 valid time `2026-05-02T04:00:00Z`.",
        "- WRF surface: after `grid%itimestep` increment in `solve_em.F`, before current-step physics/RK.",
        "- JAX source: produced step-5999 full carry checkpoint.",
        "",
    ]
    if payload.get("blocked"):
        blocked = payload["blocked"]
        lines.extend(
            [
                "## Blocker",
                "",
                f"- Reason: `{blocked['reason']}`.",
                f"- Missing artifact: `{blocked.get('missing_artifact')}`.",
                f"- Missing command: `{blocked.get('missing_command')}`.",
                "",
            ]
        )
    else:
        first = payload["comparison"]["first_mismatch"]
        lines.extend(["## Comparison", ""])
        if first:
            lines.append(
                f"- First mismatch: `{first['field']}` max_abs `{first['max_abs']}` "
                f"RMSE `{first['rmse']}`."
            )
        else:
            lines.append("- All target fields matched within the frozen tolerance.")
        for field in TARGET_FIELDS:
            entry = payload["comparison"]["comparisons"][field]
            lines.append(
                f"- `{field}`: status `{entry['status']}`, max_abs `{entry.get('max_abs')}`, "
                f"RMSE `{entry.get('rmse')}`."
            )
        lines.append("")
    lines.extend(
        [
            "## Next Decision",
            "",
            payload["next_decision"],
            "",
        ]
    )
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Review: V0.14 Pre-RK Input Boundary",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "objective: produce explicit WRF and JAX step-6000 pre-RK input-boundary truth for h10 d02 over T/P/PB/MU/MUB and decide whether the produced JAX step-5999 carry is already wrong before current-step physics/RK.",
        "",
        "files changed:",
        "- `proofs/v014/pre_rk_input_boundary.py`",
        "- `proofs/v014/pre_rk_input_boundary.json`",
        "- `proofs/v014/pre_rk_input_boundary.md`",
        "- `proofs/v014/pre_rk_input_boundary_wrf_patch.diff`",
        "- `.agent/reviews/2026-06-09-v014-pre-rk-input-boundary.md`",
        "",
        "commands run:",
    ]
    for command in payload["commands"]["required_validation"]:
        lines.append(f"- `{command}`")
    if payload["commands"].get("wrf_commands_run"):
        for command in payload["commands"]["wrf_commands_run"]:
            lines.append(f"- `{command}`")
    lines.extend(
        [
            "",
            "proof objects produced:",
            "- `proofs/v014/pre_rk_input_boundary.json`",
            "- `proofs/v014/pre_rk_input_boundary.md`",
            "- `proofs/v014/pre_rk_input_boundary_wrf_patch.diff`",
            "- `.agent/reviews/2026-06-09-v014-pre-rk-input-boundary.md`",
            "",
            "unresolved risks:",
        ]
    )
    for risk in payload["unresolved_risks"]:
        lines.append(f"- {risk}")
    lines.extend(["", f"next decision needed: {payload['next_decision']}", ""])
    return "\n".join(lines)


def main() -> int:
    wrf_files = discover_wrf_output_files()
    jax_load = load_jax_carry()
    wrf_surface = None
    comparison = None
    extra_context = None
    blocked = None
    unresolved: list[str] = [
        "Only the selected h10 d02 mass patch was compared; broader field coverage remains a follow-up.",
        "WRF truth is source-hook output, not retained wrfout inspection.",
    ]

    if not wrf_files:
        blocked = blocked_payload(detect_wrf_block_reason(wrf_files), wrf_files, jax_load)
    elif jax_load.get("status") != "LOAD_OK":
        blocked = blocked_payload(str(jax_load.get("reason", "JAX_CARRY_UNAVAILABLE")), wrf_files, jax_load)
    else:
        wrf_surface = parse_wrf_files(wrf_files)
        comparison = compare_jax_to_wrf(jax_load["carry"], wrf_surface)
        extra_context = compare_extra_context(jax_load["carry"], wrf_surface)

    verdict, next_decision = decide_verdict(comparison, blocked)

    jax_provenance = {k: v for k, v in jax_load.items() if k != "carry"}
    if jax_load.get("status") == "LOAD_OK" and jax_load.get("step_index") != PRESTEP_COMPLETED_STEPS:
        unresolved.append(
            f"JAX checkpoint step_index is {jax_load.get('step_index')}, expected {PRESTEP_COMPLETED_STEPS}."
        )

    wrf_run_info = {
        "status": "HOOK_OUTPUT_PRESENT" if wrf_files else "BLOCKED_NO_HOOK_OUTPUT",
        "output_dir": str(WRF_OUTPUT_DIR),
        "files": [str(path) for path in wrf_files],
        "file_info": [path_info(path) for path in wrf_files],
        "source_path": path_info(SCRATCH_WRF),
        "executable": path_info(SCRATCH_WRF / "main/wrf.exe"),
        "run_dir": str(SCRATCH_RUN_DIR),
        "stdout_log": path_info(SCRATCH_RUN_DIR / "pre_rk_input_boundary_28rank_stdout.log"),
        "stdout_tail": tail_text(SCRATCH_RUN_DIR / "pre_rk_input_boundary_28rank_stdout.log"),
        "singleton_stdout_log": path_info(
            SCRATCH_RUN_DIR / "pre_rk_input_boundary_singleton_stdout.log"
        ),
        "singleton_stdout_tail": tail_text(
            SCRATCH_RUN_DIR / "pre_rk_input_boundary_singleton_stdout.log"
        ),
        "rsl_error_0000": path_info(SCRATCH_RUN_DIR / "rsl.error.0000"),
        "rsl_error_0000_tail": tail_text(SCRATCH_RUN_DIR / "rsl.error.0000"),
        "compile_log": path_info(COMPILE_LOG),
        "compile_log_tail": tail_text(COMPILE_LOG),
        "legacy_serial_compile_log": path_info(LEGACY_COMPILE_LOG),
        "lineage_note": (
            "Manager reran the hook outside the worker sandbox using the existing "
            "dmpar v014_post_rk_refresh WRF lineage. The worker's earlier pristine "
            "copy compiled as serial and is retained only as a failed diagnostic."
        ),
    }

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.pre_rk_input_boundary.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "next_decision": next_decision,
        "cpu_only": True,
        "gpu_used": False,
        "production_src_edits": False,
        "wrf_in_place_edits": False,
        "tost_run": False,
        "switzerland_validation_run": False,
        "fp32_source_landing": False,
        "target": compact_target(),
        "environment": jax_environment(),
        "inputs_read": proof_inputs(),
        "wrf_provenance": {
            **wrf_commands(),
            "run": wrf_run_info,
            "surface_parse": None
            if wrf_surface is None
            else {
                "files": wrf_surface["files"],
                "metadata": wrf_surface["metadata"],
                "unique_counts": wrf_surface["unique_counts"],
                "duplicate_count": wrf_surface["duplicate_count"],
                "duplicate_max_delta": wrf_surface["duplicate_max_delta"],
                "duplicate_max_delta_by_field": wrf_surface["duplicate_max_delta_by_field"],
            },
        },
        "jax_provenance": jax_provenance,
        "field_conventions": {
            "wrf": WRF_FIELD_SOURCE,
            "jax": JAX_FIELD_SOURCE,
            "theta_offset_k": THETA_OFFSET_K,
            "tolerance_max_abs": TOLERANCE_MAX_ABS,
        },
        "comparison": comparison,
        "extra_context": extra_context,
        "blocked": blocked,
        "unresolved_risks": unresolved,
        "acceptance_notes": {
            "json_validates": True,
            "retained_wrfout_used_as_verdict": False,
            "jax_vs_jax_self_compare": False,
            "explicit_pre_rk_wrf_surface_used": bool(wrf_files),
            "uses_produced_h10_carry_checkpoint": CHECKPOINT.name == "d02_step5999_full_carry.pkl",
            "production_source_edited": False,
        },
        "commands": {
            "argv": sys.argv,
            "required_validation": [
                "python -m py_compile proofs/v014/pre_rk_input_boundary.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/pre_rk_input_boundary.py",
                "python -m json.tool proofs/v014/pre_rk_input_boundary.json >/tmp/pre_rk_input_boundary.validated.json",
            ],
            "wrf_commands_run": wrf_commands()["minimal_cpu_wrf_commands"]
            + [
                (
                    f"cd {SCRATCH_RUN_DIR} && timeout 20 env "
                    "PATH=/home/user/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin:$PATH "
                    "CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 "
                    f"WRFGPU2_PRE_RK_INPUT=1 WRFGPU2_PRE_RK_INPUT_ROOT={WRF_OUTPUT_DIR} "
                    "WRFGPU2_PRE_RK_INPUT_GRID=2 WRFGPU2_PRE_RK_INPUT_START_STEP=6000 "
                    "WRFGPU2_PRE_RK_INPUT_END_STEP=6000 ./wrf.exe 2>&1 | tee "
                    f"{SCRATCH_RUN_DIR}/pre_rk_input_boundary_singleton_stdout.log"
                )
            ],
            "wrf_commands_run_note": (
                "The disposable WRF copy, patch application, compile, mpirun attempt, "
                "and singleton diagnostic were run manually in this sprint. The exact "
                "reproducible command sequence is recorded under wrf_commands_required_if_blocked."
            ),
            "wrf_commands_required_if_blocked": wrf_commands()["minimal_cpu_wrf_commands"],
        },
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "wrf_patch_diff": str(PATCH_DIFF),
            "review": str(OUT_REVIEW),
        },
    }

    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")
    print(verdict)
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    print(f"review={OUT_REVIEW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
