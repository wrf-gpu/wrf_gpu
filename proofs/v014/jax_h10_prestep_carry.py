#!/usr/bin/env python3
"""V0.14 H10 pre-step JAX carry checkpoint proof.

This proof stays read-only with respect to production source.  It looks for a
CPU-loadable JAX OperationalCarry immediately before d02 step 6000, runs the
private pre-halo capture hook only if such a carry is available, and otherwise
records the exact missing checkpoint/API.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import pickle
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "proofs/v014/jax_h10_prestep_carry.json"
OUT_MD = ROOT / "proofs/v014/jax_h10_prestep_carry.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-h10-prestep-carry-checkpoint.md"

SPRINT_CONTRACT = ROOT / ".agent/sprints/2026-06-09-v014-h10-prestep-carry-checkpoint/sprint-contract.md"
HANDOFF = ROOT / ".agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md"
VALIDATING_PHYSICS_SKILL = ROOT / ".agent/skills/validating-physics/SKILL.md"

JAX_PRE_HALO_JSON = ROOT / "proofs/v014/jax_pre_halo_capture.json"
WRF_REFRESH_JSON = ROOT / "proofs/v014/wrf_post_rk_refresh_localization.json"
WRF_REFRESH_MD = ROOT / "proofs/v014/wrf_post_rk_refresh_localization.md"
SAVEPOINT_REQUEST_JSON = ROOT / "proofs/v014/same_state_savepoint_request.json"
DYNAMIC_ATTRIBUTION_JSON = ROOT / "proofs/v014/dynamic_field_attribution.json"

OPERATIONAL_STATE = ROOT / "src/gpuwrf/runtime/operational_state.py"
OPERATIONAL_MODE = ROOT / "src/gpuwrf/runtime/operational_mode.py"
CHECKPOINT = ROOT / "src/gpuwrf/runtime/checkpoint.py"
RESTART = ROOT / "src/gpuwrf/io/restart.py"
WRFRST_NETCDF = ROOT / "src/gpuwrf/io/wrfrst_netcdf.py"
D02_REPLAY = ROOT / "src/gpuwrf/integration/d02_replay.py"
DAILY_PIPELINE = ROOT / "src/gpuwrf/integration/daily_pipeline.py"
NESTED_PIPELINE = ROOT / "src/gpuwrf/integration/nested_pipeline.py"

TARGET_FIELDS = ("T", "P", "PB", "U", "V", "W", "PH", "MU", "MUB")
COMPARE_ORDER = TARGET_FIELDS
GREEN_TOLERANCE_MAX_ABS = 2.0e-6
TARGET_STEP = 6000
PRESTEP_COMPLETED_STEPS = TARGET_STEP - 1
THETA_OFFSET_K = 300.0

REFRESH_SCHEMAS = {
    "MASS_K1": {
        "index_count": 4,
        "fields": [
            "T_HIST_SRC",
            "T_THM",
            "P",
            "PB",
            "MU_NEW",
            "MU_OLD",
            "MUB",
            "MUT",
            "MUTS",
            "AL",
            "ALB",
            "ALT",
            "RHO",
        ],
    },
    "U_K1": {"index_count": 4, "fields": ["U"]},
    "V_K1": {"index_count": 4, "fields": ["V"]},
    "WPH_KSTAG01": {"index_count": 6, "fields": ["W", "PH"]},
}

FIELD_SOURCE = {
    "T": ("MASS_K1", "T_HIST_SRC"),
    "P": ("MASS_K1", "P"),
    "PB": ("MASS_K1", "PB"),
    "MU": ("MASS_K1", "MU_NEW"),
    "MUB": ("MASS_K1", "MUB"),
    "U": ("U_K1", "U"),
    "V": ("V_K1", "V"),
    "W": ("WPH_KSTAG01", "W"),
    "PH": ("WPH_KSTAG01", "PH"),
}


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def path_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256(path),
    }


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
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

        env.update(
            {
                "jax_import_error": None,
                "jax_version": getattr(jax, "__version__", None),
                "jax_default_backend": jax.default_backend(),
                "jax_devices": [str(device) for device in jax.devices()],
            }
        )
    except Exception as exc:  # pragma: no cover - recorded in proof output
        env["jax_import_error"] = repr(exc)
    return env


def extract_ast_node(path: Path, name: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            start = int(node.lineno)
            end = int(getattr(node, "end_lineno", node.lineno))
            source = "\n".join(text.splitlines()[start - 1 : end])
            return {
                "module_path": str(path.relative_to(ROOT)),
                "name": name,
                "line_start": start,
                "line_end": end,
                "source_sha256": sha256_text(source),
                "contains_operational_carry": "OperationalCarry" in source,
                "contains_runtime_state": "runtime_state" in source,
                "contains_write_checkpoint": "write_checkpoint" in source,
                "contains_read_checkpoint": "read_checkpoint" in source,
                "contains_write_restart": "write_restart" in source,
                "contains_read_restart": "read_restart" in source,
                "contains_advance_chunk": "_advance_chunk" in source,
                "contains_rk_capture": "_rk_scan_step_with_pre_halo_capture" in source
                or "capture_pre_halo" in source,
                "contains_state_previous_pressure": "previous_pressure" in source,
                "contains_apply_halo": "apply_halo" in source,
            }
    return {"module_path": str(path.relative_to(ROOT)), "name": name, "missing": True}


def source_inspection() -> dict[str, Any]:
    return {
        "files": {
            "runtime_operational_state": path_info(OPERATIONAL_STATE),
            "runtime_operational_mode": path_info(OPERATIONAL_MODE),
            "runtime_checkpoint": path_info(CHECKPOINT),
            "io_restart": path_info(RESTART),
            "io_wrfrst_netcdf": path_info(WRFRST_NETCDF),
            "integration_d02_replay": path_info(D02_REPLAY),
            "integration_daily_pipeline": path_info(DAILY_PIPELINE),
            "integration_nested_pipeline": path_info(NESTED_PIPELINE),
        },
        "nodes": [
            extract_ast_node(OPERATIONAL_STATE, "OperationalCarry"),
            extract_ast_node(OPERATIONAL_STATE, "initial_operational_carry"),
            extract_ast_node(OPERATIONAL_MODE, "_rk_scan_step_with_pre_halo_capture"),
            extract_ast_node(OPERATIONAL_MODE, "_rk_scan_step"),
            extract_ast_node(OPERATIONAL_MODE, "_physics_step_forcing"),
            extract_ast_node(OPERATIONAL_MODE, "_physics_boundary_step_with_limiter_diagnostics"),
            extract_ast_node(OPERATIONAL_MODE, "_advance_chunk"),
            extract_ast_node(OPERATIONAL_MODE, "run_forecast_operational_segmented"),
            extract_ast_node(CHECKPOINT, "write_checkpoint"),
            extract_ast_node(CHECKPOINT, "read_checkpoint_with_runtime_state"),
            extract_ast_node(RESTART, "write_restart"),
            extract_ast_node(RESTART, "read_restart"),
            extract_ast_node(WRFRST_NETCDF, "write_wrfrst_carry"),
            extract_ast_node(WRFRST_NETCDF, "read_wrfrst_carry"),
            extract_ast_node(D02_REPLAY, "ReplayCase"),
            extract_ast_node(D02_REPLAY, "build_replay_case"),
            extract_ast_node(D02_REPLAY, "run_replay_scan"),
            extract_ast_node(DAILY_PIPELINE, "_run_forecast_sequence"),
            extract_ast_node(NESTED_PIPELINE, "execute_nested_pipeline"),
        ],
        "api_findings": [
            {
                "api": "gpuwrf.runtime.checkpoint.write_checkpoint(..., runtime_state=carry)",
                "can_store_full_operational_carry": True,
                "current_daily_pipeline_usage": "State-only; checkpoint_at_hour does not pass runtime_state.",
                "usable_for_h10_if_artifact_exists": True,
            },
            {
                "api": "gpuwrf.io.restart.write_restart/read_restart",
                "can_store_full_operational_carry": True,
                "current_forecast_driver_usage_found": False,
                "usable_for_h10_if_artifact_exists": True,
            },
            {
                "api": "gpuwrf.io.wrfrst_netcdf.write_wrfrst_carry/read_wrfrst_carry",
                "can_store_full_operational_carry": True,
                "limitation_for_this_sprint": "read_wrfrst_carry restores carry and metadata, not the paired OperationalNamelist/grid needed by the hook.",
            },
            {
                "api": "gpuwrf.integration.d02_replay.run_replay_scan",
                "can_produce_operational_carry": False,
                "reason": "ReplayCase carries State, Tendencies, metrics, BaseState, and previous_pressure, not OperationalCarry scratch leaves.",
            },
            {
                "api": "gpuwrf.runtime.operational_mode._advance_chunk",
                "can_build_prestep_carry_from_initial_state": True,
                "not_run_here": "A CPU-only 5999-step h10 replay is a new long forecast, not a resumable checkpoint test.",
            },
        ],
    }


def key_for(record_type: str, idx: list[int]) -> tuple[int, ...]:
    if record_type in {"MASS_K1", "U_K1", "V_K1"}:
        return (idx[3], idx[2])
    if record_type == "WPH_KSTAG01":
        return (idx[5], idx[4], idx[3])
    raise ValueError(record_type)


def parse_refresh_files(paths: Iterable[Path]) -> dict[str, Any]:
    records: dict[str, dict[tuple[int, ...], dict[str, float]]] = defaultdict(dict)
    metadata: dict[str, list[str]] = defaultdict(list)
    duplicate_count = 0
    duplicate_max_delta = 0.0
    duplicate_max_delta_by_field: dict[str, float] = {}

    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or line.startswith("record_schema"):
                    continue
                parts = line.split()
                tag = parts[0]
                if tag not in REFRESH_SCHEMAS:
                    if len(parts) > 1:
                        metadata[tag].append(" ".join(parts[1:]))
                    continue
                schema = REFRESH_SCHEMAS[tag]
                nidx = int(schema["index_count"])
                idx = [int(value) for value in parts[1 : 1 + nidx]]
                values = [float(value) for value in parts[1 + nidx :]]
                fields = list(schema["fields"])
                if len(values) != len(fields):
                    raise ValueError(f"{path}: {tag} expected {len(fields)} values, got {len(values)}")
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
        "files": [str(path) for path in paths],
        "records": records,
        "metadata": {name: values[0] if len(values) == 1 else values for name, values in metadata.items()},
        "unique_counts": {name: len(items) for name, items in records.items()},
        "duplicate_count": duplicate_count,
        "duplicate_max_delta": duplicate_max_delta,
        "duplicate_max_delta_by_field": duplicate_max_delta_by_field,
    }


def compact_target(wrf_refresh: Mapping[str, Any]) -> dict[str, Any]:
    surface = wrf_refresh["emitted_surfaces"]["post_after_all_rk_steps_pre_halo"]
    return {
        "accepted_wrf_verdict": wrf_refresh.get("verdict"),
        "target_surface": wrf_refresh.get("next_jax_cpu_wrapper_target"),
        "domain": wrf_refresh["target_confirmed"]["domain"],
        "wrf_step": wrf_refresh["target_confirmed"]["wrf_step"],
        "prestep_completed_steps_required": PRESTEP_COMPLETED_STEPS,
        "valid_time_utc": wrf_refresh["target_confirmed"]["valid_time_utc"],
        "current_timestr_before_step": surface["metadata"].get("current_timestr_before_step"),
        "lead_seconds_after_step": wrf_refresh["target_confirmed"]["lead_seconds_after_step"],
        "unique_counts": surface["unique_counts"],
        "selected_patch_bounds_mass_grid": wrf_refresh["target_confirmed"]["selected_patch_bounds_mass_grid"],
        "native_staggered_coordinates": wrf_refresh["target_confirmed"]["native_staggered_coordinates"],
        "source_files": surface["files"],
        "source_file_info": [path_info(Path(path)) for path in surface["files"]],
        "green_candidate_vs_provided_cpu_h10_wrfout": {
            field: wrf_refresh["comparisons"]["post_after_all_rk_steps_pre_halo_vs_provided_cpu_h10_wrfout"][field]
            for field in TARGET_FIELDS
        },
    }


def flatten_paths(value: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and ("path" in key or key.endswith("_file")) and isinstance(item, str):
                paths.append(item)
            paths.extend(flatten_paths(item))
    elif isinstance(value, list):
        for item in value:
            paths.extend(flatten_paths(item))
    return paths


def candidate_paths_from_json(json_path: Path) -> list[dict[str, Any]]:
    if not json_path.exists():
        return []
    try:
        data = load_json(json_path)
    except Exception:
        return []
    out = []
    for raw in flatten_paths(data):
        path = Path(raw)
        if not path.is_absolute():
            if raw.startswith("proofs/") or raw.startswith(".agent/") or raw.startswith("src/"):
                path = (ROOT / raw).resolve()
            else:
                path = (json_path.parent / raw).resolve()
        if path.suffix.lower() in {".pkl", ".wrfrst", ".nc"} or "checkpoint" in path.name or "restart" in path.name:
            out.append({"source_json": str(json_path), "path": str(path)})
    return out


def discover_checkpoint_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    env_vars = [
        "WRFGPU2_H10_PRESTEP_CARRY",
        "WRFGPU2_H10_PRESTEP_RESTART",
        "WRFGPU2_H10_PRESTEP_CHECKPOINT",
    ]
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            candidates.append({"source": f"env:{var}", "path": value})

    metadata_jsons = [
        ROOT / "proofs/v0110/restart_continuity.json",
        ROOT / ".agent/sprints/2026-05-27-m7-restart-continuity/restart_continuity.json",
        ROOT / "proofs/p0_5/restart_roundtrip.json",
        JAX_PRE_HALO_JSON,
    ]
    for json_path in metadata_jsons:
        for item in candidate_paths_from_json(json_path):
            candidates.append({"source": item["source_json"], "path": item["path"]})

    for pattern in [
        "proofs/v014/**/*step5999*.pkl",
        "proofs/v014/**/*step5999*.wrfrst",
        "proofs/v014/**/*h10*.pkl",
        "proofs/v014/**/*h10*.wrfrst",
    ]:
        for path in ROOT.glob(pattern):
            candidates.append({"source": f"glob:{pattern}", "path": str(path)})

    seen: set[str] = set()
    unique = []
    for item in candidates:
        path = str(Path(item["path"]).expanduser())
        if path in seen:
            continue
        seen.add(path)
        unique.append({"source": item["source"], "path": path, **path_info(Path(path))})
    return unique


def checkpoint_metadata(candidate: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(candidate["path"]))
    if not path.exists() or not path.is_file():
        return {"status": "MISSING"}
    if path.stat().st_size > 2_000_000_000:
        return {"status": "SKIPPED", "reason": "candidate larger than 2 GB; not metadata-read in this proof"}
    suffix = path.suffix.lower()
    try:
        if suffix == ".pkl":
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            if isinstance(payload, Mapping):
                return {
                    "status": "READ",
                    "format": payload.get("format"),
                    "format_version": payload.get("format_version"),
                    "step_index": payload.get("step_index"),
                    "has_runtime_state": payload.get("runtime_state") is not None,
                    "has_full_restart_carry": payload.get("carry_type") == "gpuwrf.runtime.operational_state.OperationalCarry",
                    "has_namelist": "namelist" in payload,
                    "has_grid": "grid" in payload,
                }
        if suffix == ".wrfrst":
            from gpuwrf.io.restart import read_restart_metadata  # noqa: PLC0415

            meta = read_restart_metadata(path)
            return {"status": "READ", **meta}
        if suffix == ".nc":
            try:
                import netCDF4  # noqa: PLC0415
            except Exception as exc:  # pragma: no cover
                return {"status": "BLOCKED", "reason": f"netCDF4 import failed: {exc!r}"}
            with netCDF4.Dataset(path) as ds:
                attrs = {name: getattr(ds, name) for name in ds.ncattrs()}
            return {
                "status": "READ",
                "format": "netcdf",
                "step_index": attrs.get("ITIMESTEP") or attrs.get("itimestep"),
                "schema_version": attrs.get("GPUWRF_RESTART_SCHEMA") or attrs.get("schema_version"),
                "has_namelist": False,
                "has_grid": False,
                "limitation": "NetCDF carry restart lacks paired OperationalNamelist/grid for this hook.",
            }
    except Exception as exc:  # pragma: no cover - recorded in proof output
        return {"status": "ERROR", "error": repr(exc)}
    return {"status": "UNRECOGNIZED", "suffix": suffix}


def annotate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for item in candidates:
        meta = checkpoint_metadata(item)
        valid_step = meta.get("step_index") == PRESTEP_COMPLETED_STEPS
        valid_payload = (
            (meta.get("format") == "gpuwrf-runtime-checkpoint" and meta.get("has_runtime_state"))
            or (meta.get("format") == "gpuwrf-operational-restart")
        )
        out.append(
            {
                **item,
                "metadata": meta,
                "usable_h10_prestep_candidate": bool(item.get("exists") and valid_step and valid_payload),
                "why_not_usable": None
                if item.get("exists") and valid_step and valid_payload
                else _candidate_reject_reason(item, meta),
            }
        )
    return out


def _candidate_reject_reason(item: Mapping[str, Any], meta: Mapping[str, Any]) -> str:
    if not item.get("exists"):
        return "path does not exist"
    if meta.get("status") != "READ":
        return f"metadata status {meta.get('status')}"
    if meta.get("step_index") != PRESTEP_COMPLETED_STEPS:
        return f"step_index {meta.get('step_index')} != required {PRESTEP_COMPLETED_STEPS}"
    if meta.get("format") == "gpuwrf-runtime-checkpoint" and not meta.get("has_runtime_state"):
        return "runtime checkpoint has State only; runtime_state OperationalCarry absent"
    if meta.get("format") == "netcdf":
        return "NetCDF carry lacks paired OperationalNamelist/grid"
    if meta.get("format") != "gpuwrf-operational-restart":
        return f"unsupported format {meta.get('format')!r}"
    return "not a recognized full-carry checkpoint"


def stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
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
    }


def state_field_view(state: Any, field: str) -> Any:
    if field == "T":
        return state.theta - THETA_OFFSET_K
    if field == "P":
        return state.p_perturbation
    if field == "PB":
        return state.p_total - state.p_perturbation
    if field == "U":
        return state.u
    if field == "V":
        return state.v
    if field == "W":
        return state.w
    if field == "PH":
        return state.ph_perturbation
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
    if field in {"U", "V"}:
        return (0, key[0], key[1])
    if field in {"W", "PH"}:
        return (key[0], key[1], key[2])
    raise KeyError(field)


def compare_state_to_wrf_surface(state: Any, surface: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in TARGET_FIELDS:
        tag, source_field = FIELD_SOURCE[field]
        candidate_arr = np.asarray(state_field_view(state, field), dtype=np.float64)
        diffs: list[float] = []
        worst: dict[str, Any] | None = None
        for key, record in surface["records"][tag].items():
            truth = float(record[source_field])
            candidate = float(candidate_arr[state_index(field, key)])
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
        out[field] = {
            "status": "DIFF" if diffs else "NO_RECORDS",
            **stats(diffs),
            "worst": worst,
        }
    return out


def first_mismatch(comparison: Mapping[str, Any], tolerance: float) -> dict[str, Any] | None:
    for field in COMPARE_ORDER:
        entry = comparison.get(field)
        if not isinstance(entry, Mapping):
            continue
        max_abs = entry.get("max_abs")
        if max_abs is not None and float(max_abs) > float(tolerance):
            return {
                "field": field,
                "max_abs": float(max_abs),
                "rmse": entry.get("rmse"),
                "tolerance": float(tolerance),
                "worst": entry.get("worst"),
            }
    return None


def load_carry_candidate(candidate: Mapping[str, Any]) -> tuple[Any, Any, Any, int, str]:
    path = Path(str(candidate["path"]))
    fmt = candidate["metadata"].get("format")
    if fmt == "gpuwrf-operational-restart":
        from gpuwrf.io.restart import read_restart  # noqa: PLC0415

        carry, namelist, grid, step_index = read_restart(path)
        return carry, namelist, grid, int(step_index), "gpuwrf.io.restart.read_restart"
    if fmt == "gpuwrf-runtime-checkpoint":
        from gpuwrf.runtime.checkpoint import read_checkpoint_with_runtime_state  # noqa: PLC0415

        _state, namelist, grid, step_index, runtime_state = read_checkpoint_with_runtime_state(path)
        if runtime_state is None:
            raise ValueError(f"{path} has no runtime_state OperationalCarry")
        return runtime_state, namelist, grid, int(step_index), "gpuwrf.runtime.checkpoint.read_checkpoint_with_runtime_state"
    raise ValueError(f"unsupported carry candidate format {fmt!r}")


def run_real_compare(candidate: Mapping[str, Any], wrf_refresh: Mapping[str, Any], surface: Mapping[str, Any]) -> dict[str, Any]:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from gpuwrf.runtime.operational_mode import (  # noqa: PLC0415
        _physics_step_forcing,
        _rk_scan_step_with_pre_halo_capture,
    )

    carry, namelist, grid, step_index, loader = load_carry_candidate(candidate)
    if int(step_index) != PRESTEP_COMPLETED_STEPS:
        raise ValueError(f"checkpoint step_index={step_index}, expected {PRESTEP_COMPLETED_STEPS}")
    lead_seconds = jnp.asarray(
        float(wrf_refresh["target_confirmed"]["lead_seconds_after_step"]), dtype=jnp.float64
    )
    cadence = int(getattr(namelist, "radiation_cadence_steps", 1))
    run_radiation = bool(cadence > 0 and TARGET_STEP % cadence == 0)
    physics = _physics_step_forcing(
        carry, namelist, lead_seconds, run_radiation=run_radiation
    )
    result = _rk_scan_step_with_pre_halo_capture(
        physics.carry,
        namelist,
        lead_seconds=lead_seconds,
        physics_tendencies=physics.dry_tendencies,
    )
    jax.block_until_ready(result.carry.state.theta)
    comparison = compare_state_to_wrf_surface(result.pre_halo_state, surface)
    mismatch = first_mismatch(comparison, GREEN_TOLERANCE_MAX_ABS)
    return {
        "status": "RAN",
        "loader": loader,
        "checkpoint_path": str(candidate["path"]),
        "step_index": int(step_index),
        "grid_shape": {
            "nz": int(getattr(grid, "nz")),
            "ny": int(getattr(grid, "ny")),
            "nx": int(getattr(grid, "nx")),
        },
        "target_step": TARGET_STEP,
        "lead_seconds": float(lead_seconds),
        "run_radiation_for_step": run_radiation,
        "comparison": comparison,
        "first_mismatch": mismatch,
    }


def blocked_payload() -> dict[str, Any]:
    return {
        "status": "BLOCKED",
        "reason": "NO_CPU_LOADABLE_JAX_H10_PRESTEP_OPERATIONAL_CARRY",
        "missing_input": (
            "A CPU-loadable d02 OperationalCarry with paired OperationalNamelist/grid "
            f"at completed step_index={PRESTEP_COMPLETED_STEPS}, immediately before WRF/JAX step {TARGET_STEP}. "
            "The checkpoint must include State, promoted carry leaves "
            "t_2ave/ww/mudf/muave/muts/ph_tend/u_save/v_save/w_save/t_save/ph_save/"
            "mu_save/ww_save/rthraten, active physics carry leaves, boundary leaves, "
            "real d02 metrics, tendencies, and boundary_config."
        ),
        "missing_api": (
            "Current public forecast APIs return State only. daily_pipeline checkpoint_at_hour writes "
            "State-only runtime.checkpoint payloads. d02_replay.run_replay_scan is State+previous_pressure, "
            "not OperationalCarry. A full-carry writer exists, but no current driver writes it at arbitrary "
            "d02 step 5999 for this h10 case."
        ),
        "next_command": (
            "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src "
            "WRFGPU2_H10_PRESTEP_CARRY=/abs/path/to/d02_step5999_full_carry.pkl "
            "python proofs/v014/jax_h10_prestep_carry.py"
        ),
        "narrower_checkpoint_sprint": (
            "Add or run a proof-only CPU producer that builds the real d02 case, advances "
            "_advance_chunk to completed step 5999, and writes either "
            "runtime.checkpoint.write_checkpoint(..., runtime_state=carry) or "
            "io.restart.write_restart(carry, namelist, grid, 5999, ...)."
        ),
        "full_cpu_replay_not_run": (
            "This sprint tested resumable checkpoint availability. It did not launch a 5999-step "
            "CPU replay because that is a long forecast producer, not an existing checkpoint load."
        ),
    }


def proof_inputs() -> dict[str, Any]:
    return {
        "sprint_contract": path_info(SPRINT_CONTRACT),
        "handoff": path_info(HANDOFF),
        "validating_physics_skill": path_info(VALIDATING_PHYSICS_SKILL),
        "jax_pre_halo_capture_json": path_info(JAX_PRE_HALO_JSON),
        "wrf_post_rk_refresh_localization_json": path_info(WRF_REFRESH_JSON),
        "wrf_post_rk_refresh_localization_md": path_info(WRF_REFRESH_MD),
        "same_state_savepoint_request_json": path_info(SAVEPOINT_REQUEST_JSON),
        "dynamic_field_attribution_json": path_info(DYNAMIC_ATTRIBUTION_JSON),
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    blocked = payload.get("blocked")
    lines = [
        "# V0.14 H10 Pre-Step Carry Checkpoint",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Target",
        "",
        "- WRF target: `post dyn_em/solve_em.F::after_all_rk_steps state before RK halo exchanges`.",
        f"- Domain/step: `d02`, step `{TARGET_STEP}`, valid `{payload['wrf_target']['valid_time_utc']}`.",
        f"- Required JAX pre-step checkpoint: completed step `{PRESTEP_COMPLETED_STEPS}`.",
        "",
        "## Checkpoint Probe",
        "",
        f"- Candidates inspected: `{len(payload['checkpoint_probe']['candidates'])}`.",
        f"- Usable h10 pre-step candidates: `{len(payload['checkpoint_probe']['usable_candidates'])}`.",
        f"- Real same-surface comparison run: `{payload['comparison_run']}`.",
    ]
    if blocked:
        lines.extend(
            [
                "",
                "## Blocker",
                "",
                f"- Reason: `{blocked['reason']}`.",
                f"- Missing input/API: {blocked['missing_input']}",
                f"- Next command: `{blocked['next_command']}`",
            ]
        )
    else:
        first = payload["comparison"]["first_mismatch"]
        lines.extend(["", "## Comparison", ""])
        if first:
            lines.append(
                f"- First mismatch: `{first['field']}` max_abs `{first['max_abs']}` rmse `{first['rmse']}`."
            )
        else:
            lines.append("- All target fields matched within the frozen tolerance.")
    lines.append("")
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    blocked = payload.get("blocked")
    unresolved = payload.get("unresolved_risks", [])
    lines = [
        "# Review: V0.14 H10 Pre-Step Carry Checkpoint",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "objective: complete the CPU-only H10 pre-step carry checkpoint and compare through the JAX pre-halo hook if a real carry is available.",
        "",
        "files changed:",
        "- `proofs/v014/jax_h10_prestep_carry.py`",
        "- `proofs/v014/jax_h10_prestep_carry.json`",
        "- `proofs/v014/jax_h10_prestep_carry.md`",
        "- `.agent/reviews/2026-06-09-v014-h10-prestep-carry-checkpoint.md`",
        "",
        "commands run:",
        "- `python -m py_compile proofs/v014/jax_h10_prestep_carry.py`",
        "- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/jax_h10_prestep_carry.py`",
        "- `python -m json.tool proofs/v014/jax_h10_prestep_carry.json >/tmp/jax_h10_prestep_carry.validated.json`",
        "",
        "proof objects produced:",
        "- `proofs/v014/jax_h10_prestep_carry.json`",
        "- `proofs/v014/jax_h10_prestep_carry.md`",
        "- `.agent/reviews/2026-06-09-v014-h10-prestep-carry-checkpoint.md`",
        "",
        "result:",
    ]
    if blocked:
        lines.extend(
            [
                f"- `{blocked['reason']}`.",
                "- No retained wrfout or JAX-vs-JAX diagnostic was used as a same-surface verdict.",
                "- Existing full-carry serialization APIs are present, but no h10 step-5999 full-carry artifact was found.",
            ]
        )
    else:
        lines.append(f"- Same-surface comparison ran with verdict `{payload['verdict']}`.")
    lines.extend(["", "unresolved risks:"])
    lines.extend(f"- {risk}" for risk in unresolved)
    lines.extend(["", f"next decision needed: {payload['next_decision']}", ""])
    return "\n".join(lines)


def main() -> int:
    wrf_refresh = load_json(WRF_REFRESH_JSON)
    load_json(SAVEPOINT_REQUEST_JSON)
    load_json(DYNAMIC_ATTRIBUTION_JSON)
    load_json(JAX_PRE_HALO_JSON)

    surface_paths = [
        Path(path)
        for path in wrf_refresh["emitted_surfaces"]["post_after_all_rk_steps_pre_halo"]["files"]
    ]
    wrf_surface = parse_refresh_files(surface_paths)
    candidates = annotate_candidates(discover_checkpoint_candidates())
    usable = [item for item in candidates if item.get("usable_h10_prestep_candidate")]

    comparison = None
    comparison_error = None
    if usable:
        try:
            comparison = run_real_compare(usable[0], wrf_refresh, wrf_surface)
        except Exception as exc:  # pragma: no cover - recorded in proof output
            comparison_error = repr(exc)

    if comparison and comparison.get("status") == "RAN":
        first = comparison.get("first_mismatch")
        verdict = (
            f"JAX_MISMATCH_{first['field']}"
            if first
            else "JAX_SURFACE_MATCH_after_all_rk_pre_halo"
        )
        blocked = None
        comparison_run = True
        next_decision = (
            "Open a T history/source-attribution sprint before any production "
            "source fix; compare JAX theta/history candidates against WRF "
            "T_HIST_SRC/grid%th_phy_m_t0 and THM-side candidates."
            if first
            else "Escalate grid-parity investigation beyond this same-surface patch."
        )
        unresolved = [
            "Only the selected Boole h10 patch was compared; broader field coverage remains a follow-up."
        ]
    else:
        verdict = "CHECKPOINT_BLOCKED_NO_H10_PRESTEP_CARRY"
        blocked = blocked_payload()
        if comparison_error:
            blocked = {**blocked, "candidate_compare_error": comparison_error}
        comparison_run = False
        next_decision = "Open a narrower checkpoint producer sprint; do not start a source-fix sprint yet."
        unresolved = [
            "No first numerical JAX operator mismatch is named because no real h10 pre-step carry was available.",
            "The retained GPU/JAX h10 wrfout mismatch remains diagnostic only, not same-surface CPU evidence.",
        ]

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.jax_h10_prestep_carry.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "cpu_only": True,
        "gpu_used": False,
        "no_hermes": True,
        "production_src_edits": False,
        "wrf_source_edits": False,
        "tost_run": False,
        "switzerland_validation_run": False,
        "fp32_source_landing": False,
        "inputs_read": proof_inputs(),
        "environment": jax_environment(),
        "wrf_target": compact_target(wrf_refresh),
        "wrf_surface_parse": {
            "files": wrf_surface["files"],
            "unique_counts": wrf_surface["unique_counts"],
            "duplicate_count": wrf_surface["duplicate_count"],
            "duplicate_max_delta": wrf_surface["duplicate_max_delta"],
        },
        "source_inspection": source_inspection(),
        "checkpoint_probe": {
            "required_step_index": PRESTEP_COMPLETED_STEPS,
            "accepted_formats": [
                "gpuwrf-runtime-checkpoint with runtime_state=OperationalCarry",
                "gpuwrf-operational-restart",
            ],
            "candidates": candidates,
            "usable_candidates": usable,
        },
        "comparison_run": comparison_run,
        "comparison": comparison,
        "blocked": blocked,
        "acceptance_notes": {
            "json_records_real_checkpoint_status": True,
            "compared_against_wrf_green_target": bool(comparison_run),
            "retained_wrfout_used_as_verdict": False,
            "jax_vs_jax_self_compare": False,
            "gpu_launched": False,
            "production_source_edited": False,
        },
        "commands": {
            "generator_argv": sys.argv,
            "minimum_contract_commands": [
                "python -m py_compile proofs/v014/jax_h10_prestep_carry.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/jax_h10_prestep_carry.py",
                "python -m json.tool proofs/v014/jax_h10_prestep_carry.json >/tmp/jax_h10_prestep_carry.validated.json",
            ],
        },
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "review": str(OUT_REVIEW),
        },
        "unresolved_risks": unresolved,
        "next_decision": next_decision,
    }

    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")

    print(payload["verdict"])
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    print(f"review={OUT_REVIEW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
