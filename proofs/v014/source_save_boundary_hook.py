#!/usr/bin/env python3
"""Inventory the V0.14 WRF source/save boundary hook output.

The disposable WRF hook is intentionally outside production ``src/gpuwrf``.  It
emits the first d02 step-6000 boundary after WRF has produced current-step
``*_tendf`` and save-family leaves, but before ``relax_bdy_dry``,
``rk_addtend_dry``, or the first acoustic state update can mutate the native
dry state.
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
OUT_JSON = PROOF_DIR / "source_save_boundary_hook.json"
OUT_MD = PROOF_DIR / "source_save_boundary_hook.md"
PATCH_DIFF = PROOF_DIR / "source_save_boundary_hook_wrf_patch.diff"

SCRATCH = Path("/mnt/data/wrf_gpu2/v014_source_save_boundary")
WRF_COPY = SCRATCH / "WRF"
RUN_DIR = SCRATCH / "run_case3"
OUTPUT_DIR = SCRATCH / "source_save_output"
COMPILE_LOG = SCRATCH / "compile_source_save_boundary.log"
RUN_LOG = RUN_DIR / "source_save_boundary_28rank_stdout.log"
FULL_PRE_OUTPUT_DIR = Path("/mnt/data/wrf_gpu2/v014_full_pre_rk_savepoint_hook/full_pre_rk_output")

TARGET_STEP = 6000
TARGET_DOMAIN = 2

SOURCE_SCHEMAS: dict[str, dict[str, Any]] = {
    "MASS_SOURCE": {
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
            "T_TENDF",
            "H_DIABATIC",
            "T_SAVE",
        ],
    },
    "MASS2D_SOURCE": {
        "index_count": 4,
        "fields": ["MU_NEW", "MU_OLD", "MUB", "MUT", "MU_TENDF", "MU_SAVE"],
    },
    "U_SOURCE": {"index_count": 6, "fields": ["U_NEW", "U_OLD", "RU_TENDF", "U_SAVE", "RU_TEND"]},
    "V_SOURCE": {"index_count": 6, "fields": ["V_NEW", "V_OLD", "RV_TENDF", "V_SAVE", "RV_TEND"]},
    "WPH_SOURCE": {
        "index_count": 6,
        "fields": [
            "W_NEW",
            "W_OLD",
            "PH_NEW",
            "PH_OLD",
            "PHB",
            "RW_TENDF",
            "PH_TENDF",
            "W_SAVE",
            "PH_SAVE",
            "RW_TEND",
            "PH_TEND",
        ],
    },
    "MOIST_OLD_QV": {"index_count": 7, "fields": ["MOIST", "MOIST_OLD"]},
}

FULL_SCHEMAS: dict[str, dict[str, Any]] = {
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
    "WPH_FULL": {"index_count": 6, "fields": ["W_NEW", "W_OLD", "PH_NEW", "PH_OLD", "PHB"]},
}

DRY_REQUIREMENTS = {
    "ru_tendf": ("U_SOURCE", "RU_TENDF"),
    "rv_tendf": ("V_SOURCE", "RV_TENDF"),
    "rw_tendf": ("WPH_SOURCE", "RW_TENDF"),
    "ph_tendf": ("WPH_SOURCE", "PH_TENDF"),
    "t_tendf": ("MASS_SOURCE", "T_TENDF"),
    "mu_tendf": ("MASS2D_SOURCE", "MU_TENDF"),
    "h_diabatic": ("MASS_SOURCE", "H_DIABATIC"),
    "u_save": ("U_SOURCE", "U_SAVE"),
    "v_save": ("V_SOURCE", "V_SAVE"),
    "w_save": ("WPH_SOURCE", "W_SAVE"),
    "ph_save": ("WPH_SOURCE", "PH_SAVE"),
    "t_save": ("MASS_SOURCE", "T_SAVE"),
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


def tail_text(path: Path, lines: int = 80) -> list[str]:
    if not path.exists() or not path.is_file():
        return []
    return path.read_text(errors="replace").splitlines()[-lines:]


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
    except Exception as exc:  # pragma: no cover
        env.update({"jax_import_error": repr(exc), "gpu_device_count": None})
    return env


def key_for(tag: str, idx: list[int]) -> tuple[int, ...]:
    if tag in {"MASS_SOURCE", "U_SOURCE", "V_SOURCE", "WPH_SOURCE", "MASS_FULL", "U_FULL", "V_FULL", "WPH_FULL"}:
        return (idx[5], idx[4], idx[3])
    if tag == "MASS2D_SOURCE":
        return (idx[3], idx[2])
    if tag == "MOIST_OLD_QV":
        return (idx[6], idx[5], idx[4], idx[3])
    raise ValueError(tag)


def parse_savepoint(paths: Iterable[Path], schemas: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    metadata: dict[str, list[str]] = defaultdict(list)
    schema_lines: list[str] = []
    records: dict[str, dict[tuple[int, ...], dict[str, float]]] = defaultdict(dict)
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
                if tag not in schemas:
                    metadata[tag].append(" ".join(parts[1:]) if len(parts) > 1 else "")
                    continue
                schema = schemas[tag]
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
        "files": [path_info(path) for path in paths],
        "metadata": {key: values[0] if len(values) == 1 else values for key, values in sorted(metadata.items())},
        "schema_lines": schema_lines,
        "records": records,
        "unique_counts": {tag: len(records.get(tag, {})) for tag in schemas},
        "duplicate_count": duplicate_count,
        "duplicate_max_delta": duplicate_max_delta,
        "duplicate_max_delta_by_field": duplicate_max_delta_by_field,
    }


def compact_surface(surface: Mapping[str, Any], schemas: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "files": surface["files"],
        "metadata": surface["metadata"],
        "schema_lines": surface["schema_lines"],
        "schemas": schemas,
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
        keys = np.asarray(list(records.keys()), dtype=np.int64)
        out["key_ranges"][tag] = {"min": keys.min(axis=0).tolist(), "max": keys.max(axis=0).tolist()}
        ranges: dict[str, Any] = {}
        for field in next(iter(records.values())):
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


def parse_numbers(surface: Mapping[str, Any], key: str) -> list[int]:
    raw = surface["metadata"].get(key)
    if raw is None:
        return []
    if isinstance(raw, list):
        raw = raw[0]
    out: list[int] = []
    for item in str(raw).split():
        try:
            out.append(int(item))
        except ValueError:
            continue
    return out


def patch_assessment(surface: Mapping[str, Any]) -> dict[str, Any]:
    bounds = parse_numbers(surface, "mass_patch_zero_y0_y1_x0_x1_fortran_j0_j1_i0_i1")
    if len(bounds) < 4:
        return {"status": "UNKNOWN", "reason": "mass patch bounds metadata absent"}
    y0, y1, x0, x1 = bounds[:4]
    halo = 8
    valid_y0, valid_y1 = y0 + halo, y1 - halo
    valid_x0, valid_x1 = x0 + halo, x1 - halo
    valid_count = max(0, valid_y1 - valid_y0) * max(0, valid_x1 - valid_x0)
    return {
        "status": "PATCH_HAS_ONE_HALO_VALID_MASS_CELL" if valid_count == 1 else "PATCH_ASSESSED",
        "mass_patch_zero_based_bounds": [y0, y1, x0, x1],
        "conservative_halo_radius_cells": halo,
        "candidate_valid_mass_bounds_after_halo": [valid_y0, valid_y1, valid_x0, valid_x1],
        "candidate_valid_mass_cell_count": int(valid_count),
    }


def source_requirements(surface: Mapping[str, Any]) -> dict[str, Any]:
    present: dict[str, Any] = {}
    missing: list[str] = []
    for public_name, (tag, field) in DRY_REQUIREMENTS.items():
        records = surface["records"].get(tag, {})
        values = [record[field] for record in records.values() if field in record]
        finite_count = int(np.isfinite(np.asarray(values, dtype=np.float64)).sum()) if values else 0
        present[public_name] = {"tag": tag, "field": field, "count": len(values), "finite_count": finite_count}
        if not values or finite_count != len(values):
            missing.append(public_name)
    return {
        "dry_physics_tendencies_present": not missing,
        "present": present,
        "missing": missing,
        "moist_old_qv_present": bool(surface["records"].get("MOIST_OLD_QV")),
        "scalar_old_present": False,
        "scalar_old_status": "not initialized at this WRF boundary and not a DryPhysicsTendencies input",
    }


def state_preservation(source: Mapping[str, Any], full: Mapping[str, Any]) -> dict[str, Any]:
    mappings = [
        ("MASS_SOURCE", "MASS_FULL", ["T_THM", "T_OLD", "P", "PB", "MU_NEW", "MU_OLD", "MUB", "T_INIT"]),
        ("U_SOURCE", "U_FULL", ["U_NEW", "U_OLD"]),
        ("V_SOURCE", "V_FULL", ["V_NEW", "V_OLD"]),
        ("WPH_SOURCE", "WPH_FULL", ["W_NEW", "W_OLD", "PH_NEW", "PH_OLD", "PHB"]),
    ]
    comparisons: dict[str, Any] = {}
    worst = {"label": None, "max_abs": 0.0}
    total_compared = 0
    missing_common: list[str] = []
    for source_tag, full_tag, fields in mappings:
        src_records = source["records"].get(source_tag, {})
        full_records = full["records"].get(full_tag, {})
        common = sorted(set(src_records) & set(full_records))
        if not common:
            missing_common.append(f"{source_tag}_vs_{full_tag}")
        for field in fields:
            diffs = [abs(src_records[key][field] - full_records[key][field]) for key in common]
            total_compared += len(diffs)
            label = f"{source_tag}.{field}_vs_{full_tag}.{field}"
            max_abs = float(max(diffs)) if diffs else None
            comparisons[label] = {"count": len(diffs), "max_abs": max_abs}
            if max_abs is not None and max_abs > float(worst["max_abs"]):
                worst = {"label": label, "max_abs": max_abs}
    return {
        "compared_to_full_pre_rk_step_entry": True,
        "comparison_files": [str(path) for path in sorted(FULL_PRE_OUTPUT_DIR.glob("full_pre_rk_step_entry_d2_step_6000_*.txt"))],
        "comparisons": comparisons,
        "worst": worst,
        "total_compared_values": int(total_compared),
        "missing_common_record_sets": missing_common,
        "dry_native_state_preserved_exactly_on_overlap": total_compared > 0
        and not missing_common
        and float(worst["max_abs"]) == 0.0,
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    suff = payload["hook_sufficiency"]
    patch = payload.get("patch_width_assessment", {})
    state = payload.get("state_preservation") or {}
    lines = [
        "# V0.14 Source/Save Boundary Hook",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Output",
        "",
        f"- Files: `{len(payload['emitted_surface']['files'])}`.",
        f"- Unique records: `{payload['emitted_surface']['unique_counts']}`.",
        f"- Duplicate tile overlap max delta: `{payload['emitted_surface']['duplicate_max_delta']}`.",
        f"- Patch valid mass cells after 8-cell halo: `{patch.get('candidate_valid_mass_cell_count')}`.",
        "",
        "## Boundary",
        "",
        f"- Position: `{payload['boundary_ordering']['position']}`.",
        f"- Before first dry/acoustic mutation: `{payload['boundary_ordering']['before_first_dry_state_mutation']}`.",
        f"- Raw source/save leaves present: `{suff['dry_physics_tendencies_present']}`.",
        f"- Missing dry source leaves: `{suff['missing']}`.",
        f"- Step-entry dry state preserved on overlap: `{state.get('dry_native_state_preserved_exactly_on_overlap')}`.",
        "",
        "The hook closes the WRF source/save instrumentation gap. It does not by itself provide a full-domain JAX wrapper.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    source_files = sorted(OUTPUT_DIR.glob("source_save_after_rk_tendency_d2_step_6000_*.txt"))
    full_files = sorted(FULL_PRE_OUTPUT_DIR.glob("full_pre_rk_step_entry_d2_step_6000_*.txt"))
    source_surface = parse_savepoint(source_files, SOURCE_SCHEMAS)
    full_surface = parse_savepoint(full_files, FULL_SCHEMAS)
    requirements = source_requirements(source_surface)
    patch = patch_assessment(source_surface)
    if not source_files:
        verdict = "SOURCE_SAVE_HOOK_BLOCKED_NO_OUTPUT"
    elif requirements["dry_physics_tendencies_present"]:
        verdict = "SOURCE_SAVE_BOUNDARY_HOOK_READY"
    else:
        verdict = "SOURCE_SAVE_HOOK_BLOCKED_MISSING_" + "_".join(requirements["missing"])

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.source_save_boundary_hook.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "target": {"domain": TARGET_DOMAIN, "wrf_step": TARGET_STEP},
        "cpu_only": True,
        "gpu_used": False,
        "no_hermes": True,
        "production_src_edits": False,
        "environment": jax_environment(),
        "emitted_surface": compact_surface(source_surface, SOURCE_SCHEMAS),
        "full_pre_rk_reference": compact_surface(full_surface, FULL_SCHEMAS),
        "hook_sufficiency": requirements,
        "state_preservation": state_preservation(source_surface, full_surface) if source_files and full_files else None,
        "patch_width_assessment": patch,
        "boundary_ordering": {
            "position": "after first_rk_step_part1, first_rk_step_part2, and rk_tendency; before relax_bdy_dry, rk_addtend_dry, spec_bdy_dry, small_step_prep, and advance_uv",
            "first_source_generation_completed": True,
            "before_first_dry_state_mutation": True,
            "raw_tendf_before_rk_addtend_in_place_save_add": True,
        },
        "wrf_provenance": {
            "scratch_root": str(SCRATCH),
            "wrf_copy": path_info(WRF_COPY),
            "run_dir": path_info(RUN_DIR),
            "wrf_exe": path_info(WRF_COPY / "main/wrf.exe"),
            "run_wrf_exe": path_info(RUN_DIR / "wrf.exe"),
            "solve_em": path_info(WRF_COPY / "dyn_em/solve_em.F"),
            "compile_log": path_info(COMPILE_LOG),
            "compile_log_tail": tail_text(COMPILE_LOG),
            "run_log": path_info(RUN_LOG),
            "run_log_tail": tail_text(RUN_LOG),
            "rsl_error_0000": path_info(RUN_DIR / "rsl.error.0000"),
            "rsl_error_0000_tail": tail_text(RUN_DIR / "rsl.error.0000"),
        },
        "commands": {
            "copy": "mkdir -p /mnt/data/wrf_gpu2/v014_source_save_boundary && rsync -a /mnt/data/wrf_gpu2/v014_full_pre_rk_savepoint_hook/WRF /mnt/data/wrf_gpu2/v014_source_save_boundary/ && rsync -a /mnt/data/wrf_gpu2/v014_full_pre_rk_savepoint_hook/run_case3 /mnt/data/wrf_gpu2/v014_source_save_boundary/",
            "build": "cd /mnt/data/wrf_gpu2/v014_source_save_boundary/WRF && timeout 3600 env PATH=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin:$PATH NETCDF=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build PNETCDF=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build WRFIO_NCD_LARGE_FILE_SUPPORT=1 CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu tcsh ./compile em_real > /mnt/data/wrf_gpu2/v014_source_save_boundary/compile_source_save_boundary.log 2>&1",
            "run": "cd /mnt/data/wrf_gpu2/v014_source_save_boundary/run_case3 && timeout 3600 env PATH=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin:$PATH CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 WRFGPU2_SOURCE_SAVE_BOUNDARY=1 WRFGPU2_SOURCE_SAVE_BOUNDARY_ROOT=/mnt/data/wrf_gpu2/v014_source_save_boundary/source_save_output WRFGPU2_SOURCE_SAVE_BOUNDARY_GRID=2 WRFGPU2_SOURCE_SAVE_BOUNDARY_START_STEP=6000 WRFGPU2_SOURCE_SAVE_BOUNDARY_END_STEP=6000 mpirun --oversubscribe -np 28 ./wrf.exe > /mnt/data/wrf_gpu2/v014_source_save_boundary/run_case3/source_save_boundary_28rank_stdout.log 2>&1",
            "validation": [
                "python -m py_compile proofs/v014/source_save_boundary_hook.py",
                "python -m json.tool proofs/v014/source_save_boundary_hook.json >/tmp/source_save_boundary_hook.validated.json",
            ],
        },
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "patch_diff": str(PATCH_DIFF),
        },
    }
    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(verdict)
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
