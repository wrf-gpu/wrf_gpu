#!/usr/bin/env python3
"""Build the V0.14 dynamic term localization proof from emitted WRF text."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np


REPO = Path(__file__).resolve().parents[2]
SCRATCH = Path("/mnt/data/wrf_gpu2/v014_dynamic_terms")
TERM_DIR = SCRATCH / "term_output"
MARKER_DIR = SCRATCH / "marker_output"
RUN_DIR = SCRATCH / "run_case3"
WRF_COPY = SCRATCH / "WRF"
PRISTINE = Path("/home/enric/src/wrf_pristine/WRF")

CPU_H10 = Path(
    "/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/"
    "20260501_18z_l2_72h_20260519T173026Z/wrfout_d02_2026-05-02_04:00:00"
)
GPU_H10 = Path(
    "/tmp/v0120_powered_tost_runs/"
    "l2_d02_20260501_18z_l2_72h_20260519T173026Z/wrfout_d02_2026-05-02_04:00:00"
)
SCRATCH_H10 = RUN_DIR / "wrfout_d02_2026-05-02_04:00:00"

OUT_JSON = REPO / "proofs/v014/wrf_dynamic_term_localization.json"
OUT_MD = REPO / "proofs/v014/wrf_dynamic_term_localization.md"
OUT_REVIEW = REPO / ".agent/reviews/2026-06-09-v014-dynamic-term-localization.md"
PATCH_DIFF = REPO / "proofs/v014/wrf_dynamic_term_localization_patch.diff"

TERM_SCHEMAS = {
    "MASS_K1": {
        "index_count": 4,
        "fields": [
            "T_HIST_SRC",
            "T_THM",
            "T_OLD",
            "P",
            "PB",
            "MU_NEW",
            "MU_OLD",
            "MUT",
            "MUTS",
            "T_TEND",
            "T_TENDF",
            "MU_TEND",
            "MU_TENDF",
        ],
    },
    "U_K1": {
        "index_count": 4,
        "fields": ["U_NEW", "U_OLD", "U_SAVE", "RU_TEND", "RU_TENDF", "MUU", "MUUS"],
    },
    "V_K1": {
        "index_count": 4,
        "fields": ["V_NEW", "V_OLD", "V_SAVE", "RV_TEND", "RV_TENDF", "MUV", "MUVS"],
    },
    "WPH_KSTAG01": {
        "index_count": 6,
        "fields": [
            "W_NEW",
            "W_OLD",
            "W_SAVE",
            "PH_NEW",
            "PH_OLD",
            "PH_SAVE",
            "RW_TEND",
            "RW_TENDF",
            "PH_TEND",
            "PH_TENDF",
        ],
    },
}

MARKER_SCHEMAS = {
    "MASS_K1": {"index_count": 4, "fields": ["T", "P", "PB"]},
    "U_K1": {"index_count": 4, "fields": ["U"]},
    "V_K1": {"index_count": 4, "fields": ["V"]},
    "WPH_KSTAG01": {"index_count": 6, "fields": ["W", "PH"]},
}


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_text(args: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # pragma: no cover - proof metadata only
        return f"UNAVAILABLE: {exc}"


def key_for(record_type: str, idx: list[int]) -> tuple[int, ...]:
    if record_type in {"MASS_K1", "U_K1", "V_K1"}:
        return (idx[3], idx[2])
    if record_type == "WPH_KSTAG01":
        return (idx[5], idx[4], idx[3])
    raise ValueError(record_type)


def parse_files(paths: list[Path], schemas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    records: dict[str, dict[tuple[int, ...], dict[str, float]]] = defaultdict(dict)
    metadata: dict[str, list[str]] = defaultdict(list)
    duplicate_count = 0
    duplicate_max_delta = 0.0
    duplicate_max_delta_by_field: dict[str, float] = {}
    files = []
    for path in paths:
        files.append(str(path))
        with path.open() as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or line.startswith("record_schema"):
                    continue
                parts = line.split()
                tag = parts[0]
                if tag not in schemas:
                    if len(parts) > 1:
                        metadata[tag].append(" ".join(parts[1:]))
                    continue
                schema = schemas[tag]
                nidx = schema["index_count"]
                idx = [int(x) for x in parts[1 : 1 + nidx]]
                vals = [float(x) for x in parts[1 + nidx :]]
                fields = schema["fields"]
                if len(vals) != len(fields):
                    raise ValueError(f"{path}: {tag} expected {len(fields)} values, got {len(vals)}")
                item = dict(zip(fields, vals))
                key = key_for(tag, idx)
                if key in records[tag]:
                    duplicate_count += 1
                    prev = records[tag][key]
                    for name in fields:
                        label = f"{tag}.{name}"
                        duplicate_max_delta_by_field[label] = max(
                            duplicate_max_delta_by_field.get(label, 0.0),
                            abs(prev[name] - item[name]),
                        )
                    duplicate_max_delta = max(
                        duplicate_max_delta,
                        max(abs(prev[name] - item[name]) for name in fields),
                    )
                records[tag][key] = item
    return {
        "files": files,
        "metadata": {k: v[0] if len(v) == 1 else v for k, v in metadata.items()},
        "records": records,
        "unique_counts": {k: len(v) for k, v in records.items()},
        "duplicate_count": duplicate_count,
        "duplicate_max_delta": duplicate_max_delta,
        "duplicate_max_delta_by_field": duplicate_max_delta_by_field,
    }


def stats(values: list[float]) -> dict[str, float | int | None]:
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


def diff_stats(a: list[float], b: list[float]) -> dict[str, float | int | None]:
    if len(a) != len(b):
        raise ValueError((len(a), len(b)))
    return stats([x - y for x, y in zip(a, b)])


def pre_post_delta(pre: dict[str, Any], post: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tag, pre_records in pre["records"].items():
        tag_out = {}
        post_records = post["records"].get(tag, {})
        fields = TERM_SCHEMAS[tag]["fields"]
        common = sorted(set(pre_records) & set(post_records))
        for field in fields:
            tag_out[field] = diff_stats(
                [post_records[k][field] for k in common],
                [pre_records[k][field] for k in common],
            )
        out[tag] = {"common_count": len(common), "field_delta_stats": tag_out}
    return out


def compare_term_post_to_marker(post: dict[str, Any], marker: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "T_HIST_SRC_vs_marker_T": ("MASS_K1", "T_HIST_SRC", "T"),
        "T_THM_vs_marker_T": ("MASS_K1", "T_THM", "T"),
        "P_vs_marker_P": ("MASS_K1", "P", "P"),
        "PB_vs_marker_PB": ("MASS_K1", "PB", "PB"),
        "U_NEW_vs_marker_U": ("U_K1", "U_NEW", "U"),
        "V_NEW_vs_marker_V": ("V_K1", "V_NEW", "V"),
        "W_NEW_vs_marker_W": ("WPH_KSTAG01", "W_NEW", "W"),
        "PH_NEW_vs_marker_PH": ("WPH_KSTAG01", "PH_NEW", "PH"),
    }
    out = {}
    for label, (tag, post_field, marker_field) in mapping.items():
        common = sorted(set(post["records"][tag]) & set(marker["records"][tag]))
        out[label] = diff_stats(
            [post["records"][tag][k][post_field] for k in common],
            [marker["records"][tag][k][marker_field] for k in common],
        )
    return out


def marker_values_against_wrfout(marker: dict[str, Any], wrfout: Path) -> dict[str, Any]:
    var_map = {
        "T": ("MASS_K1", "T", lambda key: (0, key[0], key[1])),
        "P": ("MASS_K1", "P", lambda key: (0, key[0], key[1])),
        "PB": ("MASS_K1", "PB", lambda key: (0, key[0], key[1])),
        "U": ("U_K1", "U", lambda key: (0, key[0], key[1])),
        "V": ("V_K1", "V", lambda key: (0, key[0], key[1])),
        "W": ("WPH_KSTAG01", "W", lambda key: (key[0], key[1], key[2])),
        "PH": ("WPH_KSTAG01", "PH", lambda key: (key[0], key[1], key[2])),
    }
    out = {}
    with netCDF4.Dataset(wrfout) as ds:
        for var, (tag, field, idx_fn) in var_map.items():
            if var not in ds.variables:
                out[var] = {"missing": True}
                continue
            arr = ds.variables[var][0]
            emitted = []
            ref = []
            for key, item in marker["records"][tag].items():
                emitted.append(item[field])
                ref.append(float(arr[idx_fn(key)]))
            out[var] = diff_stats(emitted, ref)
    return out


def term_post_against_wrfout(post: dict[str, Any], wrfout: Path) -> dict[str, Any]:
    var_map = {
        "T_HIST_SRC": ("T", "MASS_K1", "T_HIST_SRC", lambda key: (0, key[0], key[1])),
        "T_THM": ("T", "MASS_K1", "T_THM", lambda key: (0, key[0], key[1])),
        "P": ("P", "MASS_K1", "P", lambda key: (0, key[0], key[1])),
        "PB": ("PB", "MASS_K1", "PB", lambda key: (0, key[0], key[1])),
        "U_NEW": ("U", "U_K1", "U_NEW", lambda key: (0, key[0], key[1])),
        "V_NEW": ("V", "V_K1", "V_NEW", lambda key: (0, key[0], key[1])),
        "W_NEW": ("W", "WPH_KSTAG01", "W_NEW", lambda key: (key[0], key[1], key[2])),
        "PH_NEW": ("PH", "WPH_KSTAG01", "PH_NEW", lambda key: (key[0], key[1], key[2])),
    }
    out = {}
    with netCDF4.Dataset(wrfout) as ds:
        for label, (var, tag, field, idx_fn) in var_map.items():
            arr = ds.variables[var][0]
            emitted = []
            ref = []
            for key, item in post["records"][tag].items():
                emitted.append(item[field])
                ref.append(float(arr[idx_fn(key)]))
            out[label] = diff_stats(emitted, ref)
    return out


def field_magnitude(records: dict[str, Any], fields_by_tag: dict[str, list[str]]) -> dict[str, Any]:
    out = {}
    for tag, fields in fields_by_tag.items():
        tag_out = {}
        for field in fields:
            tag_out[field] = stats([item[field] for item in records["records"][tag].values()])
        out[tag] = tag_out
    return out


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def first_marker_metadata(marker: dict[str, Any]) -> dict[str, Any]:
    md = marker["metadata"]
    def one(name: str) -> str:
        value = md[name]
        if isinstance(value, list):
            value = value[0]
        return str(value)
    return {
        "domain_id": int(one("domain_id").split()[0]),
        "current_timestr_before_step": one("current_timestr_before_step"),
        "grid_itimestep_after_increment": int(one("grid_itimestep_after_increment").split()[0]),
        "rk_step": int(one("rk_step").split()[0]),
        "rk_order": int(one("rk_order").split()[0]),
        "lead_seconds_after_step": float(one("lead_seconds_after_step").split()[0]),
    }


def main() -> None:
    dynamic = load_json(REPO / "proofs/v014/dynamic_field_attribution.json")
    request = load_json(REPO / "proofs/v014/same_state_savepoint_request.json")
    marker_savepoint = load_json(REPO / "proofs/v014/wrf_same_state_marker_savepoint.json")
    selected = dynamic["localization_manifest"]["selected_cells"][0]
    first_patch = selected["stagger_context"]["patch_bounds_mass_grid"]

    pre = parse_files(
        sorted(TERM_DIR.glob("terms_final_stage_pre_small_step_finish_d2_step_6000_*.txt")),
        TERM_SCHEMAS,
    )
    post = parse_files(
        sorted(TERM_DIR.glob("terms_final_stage_post_small_step_finish_d2_step_6000_*.txt")),
        TERM_SCHEMAS,
    )
    marker = parse_files(sorted(MARKER_DIR.glob("marker_post_d2_step_6000_*.txt")), MARKER_SCHEMAS)

    alignment_scratch = marker_values_against_wrfout(marker, SCRATCH_H10)
    alignment_cpu = marker_values_against_wrfout(marker, CPU_H10)
    divergence_gpu = marker_values_against_wrfout(marker, GPU_H10)
    post_vs_marker = compare_term_post_to_marker(post, marker)
    post_vs_scratch = term_post_against_wrfout(post, SCRATCH_H10)
    delta = pre_post_delta(pre, post)

    terms = field_magnitude(
        post,
        {
            "MASS_K1": ["T_TEND", "T_TENDF", "MU_TEND", "MU_TENDF"],
            "U_K1": ["RU_TEND", "RU_TENDF"],
            "V_K1": ["RV_TEND", "RV_TENDF"],
            "WPH_KSTAG01": ["RW_TEND", "RW_TENDF", "PH_TEND", "PH_TENDF"],
        },
    )

    marker_md = first_marker_metadata(marker)
    target_confirmed = {
        "domain": "d02",
        "valid_time_utc": selected["valid_time_utc"],
        "valid_time_wrf_history": "2026-05-02_04:00:00",
        "wrf_step": marker_md["grid_itimestep_after_increment"],
        "lead_seconds_after_step": marker_md["lead_seconds_after_step"],
        "selected_cell_zero_yx": selected["mass_index"],
        "selected_patch_bounds_mass_grid": first_patch,
        "matches_green_marker_patch": first_patch
        == {
            "halo_radius_cells": 8,
            "south_north_start": 1,
            "south_north_stop_exclusive": 18,
            "west_east_start": 5,
            "west_east_stop_exclusive": 22,
        },
    }

    proof = {
        "schema": "wrf_dynamic_term_localization_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "TERM_LAYER_EMITTED_final_stage_small_step_finish",
        "claim_boundary": {
            "model_fix": False,
            "root_cause_claim": False,
            "wrf_source_truth_layer_emitted": True,
            "jax_same_state_wrapper_run": False,
            "gpu_used_by_this_sprint": False,
        },
        "target_confirmed": target_confirmed,
        "provenance": {
            "source_wrf_pristine": str(PRISTINE),
            "source_wrf_head": run_text(["git", "rev-parse", "HEAD"], PRISTINE),
            "source_wrf_describe": run_text(["git", "describe", "--tags", "--dirty", "--always"], PRISTINE),
            "source_wrf_dirty_status_first_lines": run_text(["git", "status", "--short"], PRISTINE).splitlines()[:80],
            "green_marker_patch_sha256": sha256(REPO / "proofs/v014/wrf_same_state_marker_patch.diff"),
            "dynamic_patch_sha256": sha256(PATCH_DIFF),
            "wrf_exe_sha256": sha256(WRF_COPY / "main/wrf.exe"),
            "run_wrf_exe_sha256": sha256(RUN_DIR / "wrf.exe"),
            "solve_em_sha256": sha256(WRF_COPY / "dyn_em/solve_em.F"),
            "run_dir": str(RUN_DIR),
            "wrf_copy": str(WRF_COPY),
        },
        "inputs_read": {
            "wrf_same_state_marker_savepoint_json": str(REPO / "proofs/v014/wrf_same_state_marker_savepoint.json"),
            "same_state_savepoint_request_json": str(REPO / "proofs/v014/same_state_savepoint_request.json"),
            "dynamic_field_attribution_json": str(REPO / "proofs/v014/dynamic_field_attribution.json"),
            "marker_verdict": marker_savepoint["attempt_verdicts"]["current_thphy_marker"]["verdict"],
            "request_selected_cell_count": len(request["selection"]["selected_cells"]),
        },
        "emitted_surface": {
            "name": "final_stage_small_step_finish",
            "wrf_routine": "dyn_em/solve_em.F::solve_em",
            "pre_surface": "final_stage_pre_small_step_finish",
            "post_surface": "final_stage_post_small_step_finish",
            "stage": "rk_step=3/rk_order=3",
            "native_layers": "K1 for MASS/U/V and KSTAG 0..1 for W/PH",
            "files": {
                "pre": pre["files"],
                "post": post["files"],
                "post_marker": marker["files"],
            },
            "unique_counts": {
                "pre": pre["unique_counts"],
                "post": post["unique_counts"],
                "post_marker": marker["unique_counts"],
            },
            "duplicate_overlap": {
                "pre_count": pre["duplicate_count"],
                "pre_max_delta": pre["duplicate_max_delta"],
                "pre_max_delta_by_field": pre["duplicate_max_delta_by_field"],
                "post_count": post["duplicate_count"],
                "post_max_delta": post["duplicate_max_delta"],
                "post_max_delta_by_field": post["duplicate_max_delta_by_field"],
                "marker_count": marker["duplicate_count"],
                "marker_max_delta": marker["duplicate_max_delta"],
                "marker_max_delta_by_field": marker["duplicate_max_delta_by_field"],
            },
            "metadata": marker_md,
        },
        "small_step_finish_delta_post_minus_pre": delta,
        "post_surface_vs_post_marker": post_vs_marker,
        "post_surface_vs_scratch_h10_wrfout": post_vs_scratch,
        "post_marker_vs_scratch_h10_wrfout": alignment_scratch,
        "post_marker_vs_provided_cpu_h10_wrfout": alignment_cpu,
        "post_marker_vs_retained_gpu_h10_wrfout": divergence_gpu,
        "emitted_term_magnitude_post_surface": terms,
        "first_failing_surface": {
            "surface": "final_stage_post_small_step_finish_to_post_rk_history",
            "finding": (
                "The emitted post-small_step_finish tile-local layer is not the "
                "history-aligned h10 surface for P/V/W or THM-side T; the "
                "post-RK marker remains the green CPU history surface."
            ),
            "evidence": {
                "post_surface_vs_post_marker": post_vs_marker,
                "post_marker_vs_scratch_h10_wrfout": alignment_scratch,
            },
            "next_exact_layer": (
                "Instrument the pressure/rho/post-RK refresh path between "
                "small_step_finish and the accepted marker after after_all_rk_steps, "
                "or compare JAX only against the already-green post-RK marker state."
            ),
        },
        "history_t_lesson": {
            "history_source": "grid%th_phy_m_t0 / emitted T_HIST_SRC",
            "thm_side": "grid%t_2 / emitted T_THM",
            "T_HIST_SRC_vs_marker_T": post_vs_marker["T_HIST_SRC_vs_marker_T"],
            "T_THM_vs_marker_T": post_vs_marker["T_THM_vs_marker_T"],
        },
        "next_jax_wrappers": [
            "src/gpuwrf/dynamics/core/small_step_finish.py::small_step_finish_wrf",
            "src/gpuwrf/runtime/operational_mode.py::_carry_from_finished_stage",
            "src/gpuwrf/runtime/operational_mode.py::_rk_scan_step",
            "source tendency container feeding src/gpuwrf/dynamics/core/rk_addtend_dry.py::rk_addtend_dry",
        ],
        "next_sprint_recommendation": (
            "narrower WRF term emitter around pressure_rho_refresh/post-RK cadence "
            "before a broad JAX term wrapper; use the green post-RK marker as the "
            "history-state anchor."
        ),
        "commands": [
            "mkdir -p /mnt/data/wrf_gpu2/v014_dynamic_terms && rsync -a /mnt/data/wrf_gpu2/v014_same_state_wrf/WRF/ /mnt/data/wrf_gpu2/v014_dynamic_terms/WRF/ && rsync -a /mnt/data/wrf_gpu2/v014_same_state_wrf/run_case3/ /mnt/data/wrf_gpu2/v014_dynamic_terms/run_case3/ && cp /mnt/data/wrf_gpu2/v014_dynamic_terms/WRF/dyn_em/solve_em.F /mnt/data/wrf_gpu2/v014_dynamic_terms/WRF/dyn_em/solve_em.F.before_v014_dynamic_terms",
            "diff -u --label a/dyn_em/solve_em.F --label b/dyn_em/solve_em.F /mnt/data/wrf_gpu2/v014_dynamic_terms/WRF/dyn_em/solve_em.F.before_v014_dynamic_terms /mnt/data/wrf_gpu2/v014_dynamic_terms/WRF/dyn_em/solve_em.F > /mnt/data/wrf_gpu2/v014_dynamic_terms/wrf_dynamic_term_localization_patch.diff",
            "find /mnt/data/wrf_gpu2/v014_dynamic_terms/run_case3 -maxdepth 1 \\( -name 'rsl.error.*' -o -name 'rsl.out.*' -o -name 'wrfout_d0*' -o -name 'wrfrst_d0*' -o -name 'wrf_stdout.log' -o -name 'marker_run_post_thphy_28rank_stdout.log' \\) -delete",
            "timeout 3600 env PATH=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin:$PATH NETCDF=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build PNETCDF=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build WRFIO_NCD_LARGE_FILE_SUPPORT=1 tcsh ./compile em_real > /mnt/data/wrf_gpu2/v014_dynamic_terms/compile_dynamic_terms.log 2>&1",
            "ln -sf /mnt/data/wrf_gpu2/v014_dynamic_terms/WRF/main/wrf.exe /mnt/data/wrf_gpu2/v014_dynamic_terms/run_case3/wrf.exe",
            "timeout 3600 env PATH=/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin:$PATH CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 WRFGPU2_DYNAMIC_TERMS=1 WRFGPU2_DYNAMIC_TERMS_ROOT=/mnt/data/wrf_gpu2/v014_dynamic_terms/term_output WRFGPU2_DYNAMIC_TERMS_GRID=2 WRFGPU2_DYNAMIC_TERMS_START_STEP=6000 WRFGPU2_DYNAMIC_TERMS_END_STEP=6000 WRFGPU2_SAMESTATE=1 WRFGPU2_SAMESTATE_ROOT=/mnt/data/wrf_gpu2/v014_dynamic_terms/marker_output WRFGPU2_SAMESTATE_GRID=2 WRFGPU2_SAMESTATE_START_STEP=6000 WRFGPU2_SAMESTATE_END_STEP=6000 mpirun --oversubscribe -np 28 ./wrf.exe > /mnt/data/wrf_gpu2/v014_dynamic_terms/run_case3/dynamic_terms_28rank_stdout.log 2>&1",
            "cp /mnt/data/wrf_gpu2/v014_dynamic_terms/wrf_dynamic_term_localization_patch.diff proofs/v014/wrf_dynamic_term_localization_patch.diff",
            "python -m py_compile proofs/v014/wrf_dynamic_term_localization.py",
            "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/wrf_dynamic_term_localization.py",
            "python -m json.tool proofs/v014/wrf_dynamic_term_localization.json >/tmp/wrf_dynamic_term_localization.validated.json",
        ],
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "patch_diff": str(PATCH_DIFF),
            "review": str(OUT_REVIEW),
            "compile_log": str(SCRATCH / "compile_dynamic_terms.log"),
            "run_stdout": str(RUN_DIR / "dynamic_terms_28rank_stdout.log"),
            "rsl_logs": str(RUN_DIR / "rsl.error.*"),
        },
        "unresolved_risks": [
            "No JAX same-state wrapper was run; this artifact emits the WRF source layer but does not compare JAX terms.",
            "Only the first selected h10 patch and K1/KSTAG01 surface layer were emitted; full-column term localization remains follow-up work.",
            "The tile-local post-small_step_finish surface is not yet the green history surface for P/V/W.",
            "The post-h10 retained GPU wrfout is from the prior retained run, not a fresh same-state JAX execution.",
        ],
    }

    OUT_JSON.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    write_markdown(proof)
    write_review(proof)


def fmt_stat(s: dict[str, Any]) -> str:
    return f"count={s.get('count')} max_abs={s.get('max_abs')} rmse={s.get('rmse')}"


def table(stats_by_var: dict[str, Any], names: list[str]) -> str:
    lines = ["| Field | Count | Max abs | RMSE |", "| --- | ---: | ---: | ---: |"]
    for name in names:
        s = stats_by_var[name]
        lines.append(f"| {name} | {s.get('count')} | {s.get('max_abs')} | {s.get('rmse')} |")
    return "\n".join(lines)


def write_markdown(proof: dict[str, Any]) -> None:
    gpu = proof["post_marker_vs_retained_gpu_h10_wrfout"]
    scratch = proof["post_marker_vs_scratch_h10_wrfout"]
    cpu = proof["post_marker_vs_provided_cpu_h10_wrfout"]
    surface = proof["post_surface_vs_post_marker"]
    tlesson = proof["history_t_lesson"]
    md = f"""# V0.14 WRF Dynamic Term Localization

Verdict: `{proof['verdict']}`.

This sprint emitted the first compact source-derived dynamic layer from CPU WRF:
`final_stage_pre_small_step_finish` and `final_stage_post_small_step_finish`
inside `dyn_em/solve_em.F::solve_em`, at `d02` step 6000, valid h10
`2026-05-02_04:00:00`. No model source under repo `src/` was edited and no GPU
was used by this sprint.

## Target

- selected mass cell: zero-based `(y={proof['target_confirmed']['selected_cell_zero_yx']['south_north']}, x={proof['target_confirmed']['selected_cell_zero_yx']['west_east']})`
- selected patch: `{proof['target_confirmed']['selected_patch_bounds_mass_grid']}`
- marker time before step: `{proof['emitted_surface']['metadata']['current_timestr_before_step']}`
- WRF step: `{proof['emitted_surface']['metadata']['grid_itimestep_after_increment']}`
- RK stage: `{proof['emitted_surface']['stage']}`

## Emitted Layer

- pre files: `{len(proof['emitted_surface']['files']['pre'])}`
- post files: `{len(proof['emitted_surface']['files']['post'])}`
- unique post counts: `{proof['emitted_surface']['unique_counts']['post']}`
- duplicate post overlap: `{proof['emitted_surface']['duplicate_overlap']['post_count']}`, max delta `{proof['emitted_surface']['duplicate_overlap']['post_max_delta']}`
- emitted named terms: `RU/RV/RW/T/PH/MU_TEND` and `RU/RV/RW/T/PH/MU_TENDF`

## Boundary Result

The tile-local `post_small_step_finish` layer is a useful source-derived layer,
but it is not yet the history-aligned h10 surface for `P/V/W` or THM-side `T`.
The later post-RK marker remains the green history anchor.

Post `small_step_finish` surface vs post-RK marker:

{table(surface, ['T_HIST_SRC_vs_marker_T', 'T_THM_vs_marker_T', 'P_vs_marker_P', 'PB_vs_marker_PB', 'U_NEW_vs_marker_U', 'V_NEW_vs_marker_V', 'W_NEW_vs_marker_W', 'PH_NEW_vs_marker_PH'])}

## Marker Alignment

Post marker vs scratch h10 wrfout:

{table(scratch, ['T', 'P', 'PB', 'U', 'V', 'W', 'PH'])}

Post marker vs provided CPU h10 wrfout:

{table(cpu, ['T', 'P', 'PB', 'U', 'V', 'W', 'PH'])}

Retained GPU/JAX h10 wrfout minus the same WRF post marker still has the target
patch divergence:

{table(gpu, ['T', 'P', 'PB', 'U', 'V', 'W', 'PH'])}

## T Source

- `T_HIST_SRC` (`grid%th_phy_m_t0`) vs post marker T: {fmt_stat(tlesson['T_HIST_SRC_vs_marker_T'])}
- `T_THM` (`grid%t_2`) vs post marker T: {fmt_stat(tlesson['T_THM_vs_marker_T'])}

This preserves the green marker lesson: history `T` must come from
`grid%th_phy_m_t0`, not `grid%t_1/grid%t_2`.

## Next

Next exact layer: instrument the pressure/rho/post-RK refresh path between
`small_step_finish` and the accepted marker after `after_all_rk_steps`, or
compare JAX only against the already-green post-RK marker state. Do not claim a
root cause from this WRF-only layer.
"""
    OUT_MD.write_text(md)


def write_review(proof: dict[str, Any]) -> None:
    review = f"""# V0.14 Dynamic Term Localization Review

- objective: produce the first compact source-derived WRF dynamic term layer from the green h10 same-state marker.
- verdict: `{proof['verdict']}`.
- files changed: `proofs/v014/wrf_dynamic_term_localization.py`, `.json`, `.md`, `_patch.diff`, and this review.
- WRF copy/run paths: `{proof['provenance']['wrf_copy']}`, `{proof['provenance']['run_dir']}`.
- patch hash: `{proof['provenance']['dynamic_patch_sha256']}`.
- executable hash: `{proof['provenance']['wrf_exe_sha256']}`.
- proof objects: `{proof['proof_objects']['json']}`, `{proof['proof_objects']['markdown']}`, `{proof['proof_objects']['patch_diff']}`.
- emitted fields/terms: native `T/P/PB/U/V/W/PH`, `MU`, `MUT/MUTS`, mass-coupled `MUU/MUUS/MUV/MUVS`, and `RU/RV/RW/T/PH/MU_TEND` plus `*_TENDF`.
- unresolved risks: no JAX same-state wrapper run; only first selected patch and surface K1/KSTAG01 emitted; tile-local post-`small_step_finish` is not yet the green history surface for `P/V/W`.
- next decision needed: narrower WRF emitter around pressure/rho/post-RK cadence, then JAX CPU wrappers for the green same-state surface.
"""
    OUT_REVIEW.write_text(review)


if __name__ == "__main__":
    main()
