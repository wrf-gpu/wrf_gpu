#!/usr/bin/env python3
"""Inventory the V0.14 full pre-RK WRF savepoint hook output.

This proof is a parser/reporter for the disposable WRF hook.  It does not claim
same-input JAX parity; it only decides whether the emitted WRF boundary is
sufficient for the strict one-step wrapper.
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


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")

ROOT = Path(__file__).resolve().parents[2]
PROOF_DIR = ROOT / "proofs/v014"
OUT_JSON = PROOF_DIR / "full_pre_rk_savepoint_hook.json"
OUT_MD = PROOF_DIR / "full_pre_rk_savepoint_hook.md"
PATCH_DIFF = PROOF_DIR / "full_pre_rk_savepoint_hook_wrf_patch.diff"

SCRATCH = Path("<DATA_ROOT>/wrf_gpu2/v014_full_pre_rk_savepoint_hook")
WRF_COPY = SCRATCH / "WRF"
RUN_DIR = SCRATCH / "run_case3"
OUTPUT_DIR = SCRATCH / "full_pre_rk_output"
COMPILE_LOG = SCRATCH / "compile_full_pre_rk.log"
RUN_LOG = RUN_DIR / "full_pre_rk_28rank_stdout.log"

TARGET_STEP = 6000
TARGET_DOMAIN = 2
TARGET_FIELDS = ("T", "P", "PB", "PH", "PHB", "MU", "MUB", "U", "V", "W")

SCHEMAS: dict[str, dict[str, Any]] = {
    "MASS_FULL": {
        "index_count": 6,
        "fields": [
            "T_THM",
            "T_OLD",
            "T_HIST_SRC",
            "P",
            "PB",
            "MU_NEW",
            "MU_OLD",
            "MUB",
            "T_INIT",
        ],
    },
    "U_FULL": {"index_count": 6, "fields": ["U_NEW", "U_OLD"]},
    "V_FULL": {"index_count": 6, "fields": ["V_NEW", "V_OLD"]},
    "WPH_FULL": {
        "index_count": 6,
        "fields": ["W_NEW", "W_OLD", "PH_NEW", "PH_OLD", "PHB"],
    },
    "MOIST_FULL": {"index_count": 7, "fields": ["MOIST"]},
    "SCALAR_FULL": {"index_count": 7, "fields": ["SCALAR"]},
}

DRY_SOURCE_FIELDS = [
    "ru_tendf",
    "rv_tendf",
    "rw_tendf",
    "ph_tendf",
    "t_tendf",
    "mu_tendf",
    "h_diabatic",
    "u_save",
    "v_save",
    "w_save",
    "ph_save",
    "t_save",
]


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


def tail_text(path: Path, lines: int = 80) -> list[str]:
    if not path.exists() or not path.is_file():
        return []
    data = path.read_text(errors="replace").splitlines()
    return data[-lines:]


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
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
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
                "gpu_device_count": len([item for item in devices if "gpu" in item.lower()]),
            }
        )
    except Exception as exc:  # pragma: no cover - proof metadata only
        env.update({"jax_import_error": repr(exc), "gpu_device_count": None})
    return env


def key_for(tag: str, idx: list[int]) -> tuple[int, ...]:
    if tag in {"MASS_FULL", "U_FULL", "V_FULL", "WPH_FULL"}:
        return (idx[5], idx[4], idx[3])
    if tag in {"MOIST_FULL", "SCALAR_FULL"}:
        return (idx[6], idx[5], idx[4], idx[3])
    raise ValueError(tag)


def parse_numbers(raw: Any) -> list[int]:
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    out: list[int] = []
    for part in str(raw).split():
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def parse_savepoint(paths: Iterable[Path]) -> dict[str, Any]:
    metadata: dict[str, list[str]] = defaultdict(list)
    schema_lines: list[str] = []
    records: dict[str, dict[tuple[int, ...], dict[str, float]]] = defaultdict(dict)
    native_indices: dict[str, dict[tuple[int, ...], list[int]]] = defaultdict(dict)
    duplicate_count = 0
    duplicate_max_delta = 0.0
    duplicate_max_delta_by_field: dict[str, float] = {}

    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    metadata["comment"].append(line[1:].strip())
                    continue
                if line.startswith("record_schema"):
                    schema_lines.append(line)
                    continue
                parts = line.split()
                tag = parts[0]
                if tag not in SCHEMAS:
                    metadata[tag].append(" ".join(parts[1:]) if len(parts) > 1 else "")
                    continue
                schema = SCHEMAS[tag]
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
                native_indices[tag][key] = idx

    compact_metadata = {
        key: values[0] if len(values) == 1 else values
        for key, values in sorted(metadata.items())
    }
    return {
        "files": [path_info(path) for path in paths],
        "metadata": compact_metadata,
        "schema_lines": schema_lines,
        "records": records,
        "native_indices": native_indices,
        "unique_counts": {tag: len(records.get(tag, {})) for tag in SCHEMAS},
        "duplicate_count": duplicate_count,
        "duplicate_max_delta": duplicate_max_delta,
        "duplicate_max_delta_by_field": duplicate_max_delta_by_field,
    }


def compact_surface(surface: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "files": surface["files"],
        "metadata": surface["metadata"],
        "schema_lines": surface["schema_lines"],
        "schemas": SCHEMAS,
        "unique_counts": surface["unique_counts"],
        "duplicate_count": surface["duplicate_count"],
        "duplicate_max_delta": surface["duplicate_max_delta"],
        "duplicate_max_delta_by_field": surface["duplicate_max_delta_by_field"],
        "key_ranges": {},
        "field_ranges": {},
    }
    for tag, records in surface["records"].items():
        if not records:
            continue
        keys = np.asarray(list(records), dtype=np.int64)
        out["key_ranges"][tag] = {
            "key_convention": "MASS/U/V/WPH=(zero_k, zero_y, zero_x); MOIST/SCALAR=(index, zero_k, zero_y, zero_x)",
            "min": keys.min(axis=0).tolist(),
            "max": keys.max(axis=0).tolist(),
        }
        fields = list(next(iter(records.values())).keys())
        ranges: dict[str, Any] = {}
        for field in fields:
            values = np.asarray([record[field] for record in records.values()], dtype=np.float64)
            finite = values[np.isfinite(values)]
            ranges[field] = {
                "count": int(values.size),
                "finite_count": int(finite.size),
                "min": float(np.min(finite)) if finite.size else None,
                "max": float(np.max(finite)) if finite.size else None,
            }
        out["field_ranges"][tag] = ranges
    return out


def patch_width_assessment(metadata: Mapping[str, Any]) -> dict[str, Any]:
    bounds = parse_numbers(metadata.get("mass_patch_zero_y0_y1_x0_x1_fortran_j0_j1_i0_i1"))
    if len(bounds) < 4:
        return {"status": "UNKNOWN", "reason": "mass patch bounds metadata absent"}
    y0, y1, x0, x1 = bounds[:4]
    halo = 8
    valid_y0, valid_y1 = y0 + halo, y1 - halo
    valid_x0, valid_x1 = x0 + halo, x1 - halo
    valid_count = max(0, valid_y1 - valid_y0) * max(0, valid_x1 - valid_x0)
    return {
        "status": "NOT_PRIMARY_BLOCKER" if valid_count > 0 else "PATCH_WIDTH_BLOCKED",
        "mass_patch_zero_based_bounds": {
            "south_north_start": y0,
            "south_north_stop_exclusive": y1,
            "west_east_start": x0,
            "west_east_stop_exclusive": x1,
        },
        "conservative_halo_radius_cells": halo,
        "candidate_valid_mass_bounds_after_halo": {
            "south_north_start": valid_y0,
            "south_north_stop_exclusive": valid_y1,
            "west_east_start": valid_x0,
            "west_east_stop_exclusive": valid_x1,
        },
        "candidate_valid_mass_cell_count": int(valid_count),
    }


def assess_sufficiency(surface: Mapping[str, Any]) -> dict[str, Any]:
    counts = surface["unique_counts"]
    metadata = surface["metadata"]
    present_tags = [tag for tag, count in counts.items() if count > 0]
    full_state_present = all(counts.get(tag, 0) > 0 for tag in ["MASS_FULL", "U_FULL", "V_FULL", "WPH_FULL"])
    active_moisture_present = counts.get("MOIST_FULL", 0) > 0
    source_status = metadata.get("rk_fixed_source_boundary_status")
    source_missing = bool(source_status)
    missing: list[str] = []
    if not full_state_present:
        missing.append("full native-staggered dry state records MASS_FULL/U_FULL/V_FULL/WPH_FULL")
    if not active_moisture_present:
        missing.append("active moisture leaves, including QVAPOR")
    if source_missing:
        missing.extend(DRY_SOURCE_FIELDS)
        missing.extend(["moist_old", "scalar_old"])

    if not any(item["exists"] for item in surface["files"]):
        verdict = "FULL_PRE_RK_HOOK_BLOCKED_NO_OUTPUT"
    elif not full_state_present:
        verdict = "FULL_PRE_RK_HOOK_BLOCKED_INCOMPLETE_NATIVE_STATE"
    elif source_missing:
        verdict = "FULL_PRE_RK_HOOK_BLOCKED_RK_FIXED_SOURCE_UNAVAILABLE_AT_STEP_ENTRY"
    else:
        verdict = "FULL_PRE_RK_HOOK_READY_FOR_JAX_LOADER"

    return {
        "verdict": verdict,
        "strict_same_input_ready": verdict == "FULL_PRE_RK_HOOK_READY_FOR_JAX_LOADER",
        "full_native_state_present": full_state_present,
        "active_moisture_present": active_moisture_present,
        "present_tags": present_tags,
        "missing_for_strict_same_input": missing,
        "rk_fixed_source_boundary_status": source_status,
        "reason": (
            "The exact post-itimestep step-entry boundary is before WRF computes "
            "current-step first_rk_step_part1/part2 physics tendencies and before "
            "rk_tendency zeroes/populates the save-family fields."
            if source_missing
            else "Hook output contains the required source leaves."
        ),
    }


def commands() -> dict[str, Any]:
    build_env = (
        "PATH=<USER_HOME>/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin:$PATH "
        "NETCDF=<USER_HOME>/src/canairy_meteo/Gen2/artifacts/envs/wrf-build "
        "PNETCDF=<USER_HOME>/src/canairy_meteo/Gen2/artifacts/envs/wrf-build "
        "WRFIO_NCD_LARGE_FILE_SUPPORT=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu"
    )
    run_env = (
        "PATH=<USER_HOME>/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin:$PATH "
        "CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 "
        f"WRFGPU2_FULL_PRE_RK=1 WRFGPU2_FULL_PRE_RK_ROOT={OUTPUT_DIR} "
        "WRFGPU2_FULL_PRE_RK_GRID=2 WRFGPU2_FULL_PRE_RK_START_STEP=6000 "
        "WRFGPU2_FULL_PRE_RK_END_STEP=6000"
    )
    return {
        "setup": [
            f"mkdir -p {SCRATCH}",
            f"rsync -a <DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/WRF/ {WRF_COPY}/",
            f"rsync -a <DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/run_case3/ {RUN_DIR}/",
            f"cd {WRF_COPY} && patch -p1 < {PATCH_DIFF}",
        ],
        "build": f"cd {WRF_COPY} && timeout 3600 env {build_env} tcsh ./compile em_real >{COMPILE_LOG} 2>&1",
        "run": (
            f"ln -sf {WRF_COPY}/main/wrf.exe {RUN_DIR}/wrf.exe && cd {RUN_DIR} && "
            f"timeout 3600 env {run_env} mpirun --oversubscribe -np 28 ./wrf.exe >{RUN_LOG} 2>&1"
        ),
        "minimum_validation": [
            "python -m py_compile proofs/v014/full_pre_rk_savepoint_hook.py",
            "python -m json.tool proofs/v014/full_pre_rk_savepoint_hook.json >/tmp/full_pre_rk_savepoint_hook.validated.json",
        ],
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    suff = payload["hook_sufficiency"]
    surface = payload["emitted_surface"]
    patch = payload["patch_width_assessment"]
    lines = [
        "# V0.14 Full Pre-RK Savepoint Hook",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Output",
        "",
        f"- Files: `{len(payload['wrf_provenance']['output_files'])}`.",
        f"- Unique records: `{surface['unique_counts']}`.",
        f"- Duplicate tile overlap max delta: `{surface['duplicate_max_delta']}`.",
        f"- Patch valid mass cells after 8-cell halo: `{patch.get('candidate_valid_mass_cell_count')}`.",
        "",
        "## Sufficiency",
        "",
        f"- Full native dry state present: `{suff['full_native_state_present']}`.",
        f"- Active moisture present: `{suff['active_moisture_present']}`.",
        f"- Strict same-input ready: `{suff['strict_same_input_ready']}`.",
        f"- Missing for strict same-input: `{suff['missing_for_strict_same_input']}`.",
        "",
        "The hook lands at the required step-entry boundary. At that exact point WRF has not yet produced current-step `*_tendf` or `*_save` inputs, so the downstream parity proof must fail closed unless a later accepted source boundary is provided.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    paths = sorted(OUTPUT_DIR.glob("full_pre_rk_step_entry_d2_step_6000_*.txt"))
    surface = parse_savepoint(paths)
    compact = compact_surface(surface)
    sufficiency = assess_sufficiency(surface)
    patch = patch_width_assessment(surface["metadata"])
    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.full_pre_rk_savepoint_hook.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": sufficiency["verdict"],
        "cpu_only": True,
        "gpu_used": False,
        "no_hermes": True,
        "production_src_edits": False,
        "tost_run": False,
        "switzerland_validation_run": False,
        "target": {
            "domain": f"d{TARGET_DOMAIN:02d}",
            "step": TARGET_STEP,
            "fields_requested": TARGET_FIELDS,
            "boundary": "after grid%itimestep increment before current-step physics/RK",
        },
        "environment": jax_environment(),
        "wrf_provenance": {
            "scratch_root": str(SCRATCH),
            "wrf_source_path": path_info(WRF_COPY),
            "run_dir": str(RUN_DIR),
            "output_dir": str(OUTPUT_DIR),
            "output_files": [str(path) for path in paths],
            "patch_diff": path_info(PATCH_DIFF),
            "solve_em": path_info(WRF_COPY / "dyn_em/solve_em.F"),
            "wrf_exe": path_info(WRF_COPY / "main/wrf.exe"),
            "run_wrf_exe": path_info(RUN_DIR / "wrf.exe"),
            "compile_log": path_info(COMPILE_LOG),
            "compile_log_tail": tail_text(COMPILE_LOG),
            "run_log": path_info(RUN_LOG),
            "run_log_tail": tail_text(RUN_LOG),
            "rsl_error_0000": path_info(RUN_DIR / "rsl.error.0000"),
            "rsl_error_0000_tail": tail_text(RUN_DIR / "rsl.error.0000"),
        },
        "commands": commands(),
        "emitted_surface": compact,
        "patch_width_assessment": patch,
        "hook_sufficiency": sufficiency,
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "wrf_patch_diff": str(PATCH_DIFF),
        },
    }
    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(payload["verdict"])
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
