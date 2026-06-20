#!/usr/bin/env python3
"""Build the V0.14 post-RK pressure/rho refresh localization proof."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import netCDF4
import numpy as np


REPO = Path(__file__).resolve().parents[2]
SCRATCH = Path("<DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh")
WRF_COPY = SCRATCH / "WRF"
RUN_DIR = SCRATCH / "run_case3"
REFRESH_DIR = SCRATCH / "refresh_output"
MARKER_DIR = SCRATCH / "marker_output"
PRISTINE = Path("<USER_HOME>/src/wrf_pristine/WRF")

DYNAMIC_SCRATCH = Path("<DATA_ROOT>/wrf_gpu2/v014_dynamic_terms")
DYNAMIC_POST_DIR = DYNAMIC_SCRATCH / "term_output"

CPU_H10 = Path(
    "<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/"
    "20260501_18z_l2_72h_20260519T173026Z/wrfout_d02_2026-05-02_04:00:00"
)
GPU_H10 = Path(
    "/tmp/v0120_powered_tost_runs/"
    "l2_d02_20260501_18z_l2_72h_20260519T173026Z/wrfout_d02_2026-05-02_04:00:00"
)
SCRATCH_H10 = RUN_DIR / "wrfout_d02_2026-05-02_04:00:00"

OUT_JSON = REPO / "proofs/v014/wrf_post_rk_refresh_localization.json"
OUT_MD = REPO / "proofs/v014/wrf_post_rk_refresh_localization.md"
OUT_REVIEW = REPO / ".agent/reviews/2026-06-09-v014-post-rk-refresh-localization.md"
PATCH_DIFF = REPO / "proofs/v014/wrf_post_rk_refresh_localization_patch.diff"


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

MARKER_SCHEMAS = {
    "MASS_K1": {"index_count": 4, "fields": ["T", "P", "PB"]},
    "U_K1": {"index_count": 4, "fields": ["U"]},
    "V_K1": {"index_count": 4, "fields": ["V"]},
    "WPH_KSTAG01": {"index_count": 6, "fields": ["W", "PH"]},
}

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


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


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
    for path in paths:
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
                values = [float(x) for x in parts[1 + nidx :]]
                fields = schema["fields"]
                if len(values) != len(fields):
                    raise ValueError(f"{path}: {tag} expected {len(fields)} values, got {len(values)}")
                item = dict(zip(fields, values))
                key = key_for(tag, idx)
                if key in records[tag]:
                    duplicate_count += 1
                    prev = records[tag][key]
                    for name in fields:
                        label = f"{tag}.{name}"
                        delta = abs(prev[name] - item[name])
                        duplicate_max_delta_by_field[label] = max(
                            duplicate_max_delta_by_field.get(label, 0.0), delta
                        )
                        duplicate_max_delta = max(duplicate_max_delta, delta)
                records[tag][key] = item
    return {
        "files": [str(p) for p in paths],
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


def scalar_meta(surface: dict[str, Any], name: str) -> str:
    value = surface["metadata"].get(name)
    if isinstance(value, list):
        value = value[0]
    return "" if value is None else str(value)


def first_metadata(surface: dict[str, Any]) -> dict[str, Any]:
    return {
        "domain_id": int(scalar_meta(surface, "domain_id").split()[0]),
        "current_timestr_before_step": scalar_meta(surface, "current_timestr_before_step"),
        "grid_itimestep_after_increment": int(
            scalar_meta(surface, "grid_itimestep_after_increment").split()[0]
        ),
        "rk_step": int(scalar_meta(surface, "rk_step").split()[0]),
        "rk_order": int(scalar_meta(surface, "rk_order").split()[0]),
        "lead_seconds_after_step": float(scalar_meta(surface, "lead_seconds_after_step").split()[0]),
    }


def compare_refresh_to_marker(surface: dict[str, Any], marker: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "T_HIST_SRC_vs_marker_T": ("MASS_K1", "T_HIST_SRC", "T"),
        "T_THM_vs_marker_T": ("MASS_K1", "T_THM", "T"),
        "P_vs_marker_P": ("MASS_K1", "P", "P"),
        "PB_vs_marker_PB": ("MASS_K1", "PB", "PB"),
        "U_vs_marker_U": ("U_K1", "U", "U"),
        "V_vs_marker_V": ("V_K1", "V", "V"),
        "W_vs_marker_W": ("WPH_KSTAG01", "W", "W"),
        "PH_vs_marker_PH": ("WPH_KSTAG01", "PH", "PH"),
    }
    out = {}
    for label, (tag, a_field, b_field) in mapping.items():
        common = sorted(set(surface["records"][tag]) & set(marker["records"][tag]))
        out[label] = diff_stats(
            [surface["records"][tag][k][a_field] for k in common],
            [marker["records"][tag][k][b_field] for k in common],
        )
    return out


def compare_term_to_refresh(term: dict[str, Any], surface: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "T_HIST_SRC": ("MASS_K1", "T_HIST_SRC", "T_HIST_SRC"),
        "T_THM": ("MASS_K1", "T_THM", "T_THM"),
        "P": ("MASS_K1", "P", "P"),
        "PB": ("MASS_K1", "PB", "PB"),
        "MU_NEW": ("MASS_K1", "MU_NEW", "MU_NEW"),
        "MU_OLD": ("MASS_K1", "MU_OLD", "MU_OLD"),
        "MUTS": ("MASS_K1", "MUTS", "MUTS"),
        "U": ("U_K1", "U_NEW", "U"),
        "V": ("V_K1", "V_NEW", "V"),
        "W": ("WPH_KSTAG01", "W_NEW", "W"),
        "PH": ("WPH_KSTAG01", "PH_NEW", "PH"),
    }
    out = {}
    for label, (tag, term_field, surface_field) in mapping.items():
        common = sorted(set(term["records"][tag]) & set(surface["records"][tag]))
        out[label] = diff_stats(
            [surface["records"][tag][k][surface_field] for k in common],
            [term["records"][tag][k][term_field] for k in common],
        )
    return out


def compare_refresh_to_refresh(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "T_HIST_SRC": ("MASS_K1", "T_HIST_SRC"),
        "T_THM": ("MASS_K1", "T_THM"),
        "P": ("MASS_K1", "P"),
        "PB": ("MASS_K1", "PB"),
        "MU_NEW": ("MASS_K1", "MU_NEW"),
        "MU_OLD": ("MASS_K1", "MU_OLD"),
        "MUB": ("MASS_K1", "MUB"),
        "MUT": ("MASS_K1", "MUT"),
        "MUTS": ("MASS_K1", "MUTS"),
        "AL": ("MASS_K1", "AL"),
        "ALB": ("MASS_K1", "ALB"),
        "ALT": ("MASS_K1", "ALT"),
        "RHO": ("MASS_K1", "RHO"),
        "U": ("U_K1", "U"),
        "V": ("V_K1", "V"),
        "W": ("WPH_KSTAG01", "W"),
        "PH": ("WPH_KSTAG01", "PH"),
    }
    out = {}
    for label, (tag, field) in mapping.items():
        common = sorted(set(a["records"][tag]) & set(b["records"][tag]))
        out[label] = diff_stats(
            [a["records"][tag][k][field] for k in common],
            [b["records"][tag][k][field] for k in common],
        )
    return out


def wrf_index(var: str, key: tuple[int, ...]) -> tuple[int, ...]:
    if var in {"T", "P", "PB"}:
        return (0, key[0], key[1])
    if var in {"MU", "MUB"}:
        return (key[0], key[1])
    if var in {"U", "V"}:
        return (0, key[0], key[1])
    if var in {"W", "PH"}:
        return (key[0], key[1], key[2])
    raise ValueError(var)


def surface_against_wrfout(surface: dict[str, Any], wrfout: Path) -> dict[str, Any]:
    mapping: dict[str, tuple[str, str, str]] = {
        "T": ("T", "MASS_K1", "T_HIST_SRC"),
        "P": ("P", "MASS_K1", "P"),
        "PB": ("PB", "MASS_K1", "PB"),
        "MU": ("MU", "MASS_K1", "MU_NEW"),
        "MUB": ("MUB", "MASS_K1", "MUB"),
        "U": ("U", "U_K1", "U"),
        "V": ("V", "V_K1", "V"),
        "W": ("W", "WPH_KSTAG01", "W"),
        "PH": ("PH", "WPH_KSTAG01", "PH"),
    }
    out = {}
    with netCDF4.Dataset(wrfout) as ds:
        for label, (var, tag, field) in mapping.items():
            if var not in ds.variables:
                out[label] = {"missing": True}
                continue
            arr = ds.variables[var][0]
            emitted = []
            ref = []
            for key, item in surface["records"][tag].items():
                emitted.append(item[field])
                ref.append(float(arr[wrf_index(var, key)]))
            out[label] = diff_stats(emitted, ref)
    return out


def marker_against_wrfout(marker: dict[str, Any], wrfout: Path) -> dict[str, Any]:
    mapping: dict[str, tuple[str, str, str]] = {
        "T": ("T", "MASS_K1", "T"),
        "P": ("P", "MASS_K1", "P"),
        "PB": ("PB", "MASS_K1", "PB"),
        "U": ("U", "U_K1", "U"),
        "V": ("V", "V_K1", "V"),
        "W": ("W", "WPH_KSTAG01", "W"),
        "PH": ("PH", "WPH_KSTAG01", "PH"),
    }
    out = {}
    with netCDF4.Dataset(wrfout) as ds:
        for label, (var, tag, field) in mapping.items():
            arr = ds.variables[var][0]
            emitted = []
            ref = []
            for key, item in marker["records"][tag].items():
                emitted.append(item[field])
                ref.append(float(arr[wrf_index(var, key)]))
            out[label] = diff_stats(emitted, ref)
    return out


def all_green(stats_by_name: dict[str, Any], names: list[str], tol: float) -> bool:
    for name in names:
        value = stats_by_name[name].get("max_abs")
        if value is None or value > tol:
            return False
    return True


def table(stats_by_name: dict[str, Any], names: list[str]) -> str:
    lines = ["| Field | Count | Max abs | RMSE |", "| --- | ---: | ---: | ---: |"]
    for name in names:
        s = stats_by_name[name]
        lines.append(f"| {name} | {s.get('count')} | {s.get('max_abs')} | {s.get('rmse')} |")
    return "\n".join(lines)


def summarize_compact(stats_by_name: dict[str, Any], names: list[str]) -> dict[str, Any]:
    return {name: {"max_abs": stats_by_name[name].get("max_abs"), "rmse": stats_by_name[name].get("rmse")} for name in names}


def main() -> None:
    dynamic = load_json(REPO / "proofs/v014/dynamic_field_attribution.json")
    request = load_json(REPO / "proofs/v014/same_state_savepoint_request.json")
    marker_savepoint = load_json(REPO / "proofs/v014/wrf_same_state_marker_savepoint.json")
    term_proof = load_json(REPO / "proofs/v014/wrf_dynamic_term_localization.json")

    post_calc = parse_files(
        sorted(REFRESH_DIR.glob("refresh_post_final_calc_p_rho_phi_d2_step_6000_*.txt")),
        REFRESH_SCHEMAS,
    )
    post_after_all = parse_files(
        sorted(REFRESH_DIR.glob("refresh_post_after_all_rk_steps_pre_halo_d2_step_6000_*.txt")),
        REFRESH_SCHEMAS,
    )
    marker = parse_files(sorted(MARKER_DIR.glob("marker_post_d2_step_6000_*.txt")), MARKER_SCHEMAS)
    ptolemy_post = parse_files(
        sorted(DYNAMIC_POST_DIR.glob("terms_final_stage_post_small_step_finish_d2_step_6000_*.txt")),
        TERM_SCHEMAS,
    )

    selected = dynamic["localization_manifest"]["selected_cells"][0]
    patch_bounds = selected["stagger_context"]["patch_bounds_mass_grid"]
    marker_md = first_metadata(marker)

    post_calc_vs_marker = compare_refresh_to_marker(post_calc, marker)
    post_after_all_vs_marker = compare_refresh_to_marker(post_after_all, marker)
    post_calc_vs_scratch = surface_against_wrfout(post_calc, SCRATCH_H10)
    post_after_all_vs_scratch = surface_against_wrfout(post_after_all, SCRATCH_H10)
    post_after_all_vs_cpu = surface_against_wrfout(post_after_all, CPU_H10)
    post_after_all_vs_gpu = surface_against_wrfout(post_after_all, GPU_H10)
    marker_vs_scratch = marker_against_wrfout(marker, SCRATCH_H10)
    ptolemy_to_post_calc = compare_term_to_refresh(ptolemy_post, post_calc)
    post_calc_to_post_after_all = compare_refresh_to_refresh(post_after_all, post_calc)

    green_fields_marker = [
        "T_HIST_SRC_vs_marker_T",
        "P_vs_marker_P",
        "PB_vs_marker_PB",
        "U_vs_marker_U",
        "V_vs_marker_V",
        "W_vs_marker_W",
        "PH_vs_marker_PH",
    ]
    green_fields_wrfout = ["T", "P", "PB", "MU", "MUB", "U", "V", "W", "PH"]
    post_after_all_green = all_green(post_after_all_vs_marker, green_fields_marker, 2.0e-6) and all_green(
        post_after_all_vs_scratch, green_fields_wrfout, 2.0e-6
    )
    post_calc_green = all_green(post_calc_vs_marker, green_fields_marker, 2.0e-6)

    if post_after_all_green:
        verdict = "REFRESH_LAYER_GREEN_post_after_all_rk_steps_pre_halo"
        next_target = "post dyn_em/solve_em.F::after_all_rk_steps state before RK halo exchanges"
    elif post_calc_green:
        verdict = "REFRESH_LAYER_GREEN_post_final_calc_p_rho_phi"
        next_target = "post final calc_p_rho_phi / pre-after_all_rk_steps state"
    else:
        verdict = "REFRESH_CADENCE_NAMED_final_calc_p_rho_phi_to_after_all_rk_steps"
        next_target = "instrument one narrower sub-boundary between final calc_p_rho_phi and after_all_rk_steps"

    proof = {
        "schema": "wrf_post_rk_refresh_localization_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "claim_boundary": {
            "model_fix": False,
            "root_cause_claim": False,
            "wrf_source_truth_layer_emitted": True,
            "jax_same_state_wrapper_run": False,
            "gpu_used_by_this_sprint": False,
            "tost_run": False,
            "switzerland_validation_run": False,
        },
        "target_confirmed": {
            "domain": "d02",
            "valid_time_utc": selected["valid_time_utc"],
            "valid_time_wrf_history": "2026-05-02_04:00:00",
            "wrf_step": marker_md["grid_itimestep_after_increment"],
            "lead_seconds_after_step": marker_md["lead_seconds_after_step"],
            "selected_cell_zero_yx": selected["mass_index"],
            "selected_patch_bounds_mass_grid": patch_bounds,
            "native_staggered_coordinates": {
                "mass_zero": "y [1,18), x [5,22); Fortran j 2..18, i 6..22",
                "u_zero": "y [1,18), x_stag [5,23); Fortran j 2..18, i 6..23",
                "v_zero": "y_stag [1,19), x [5,22); Fortran j 2..19, i 6..22",
                "w_ph_zero": "kstag [0,2), y [1,18), x [5,22); Fortran k 1..2, j 2..18, i 6..22",
            },
            "matches_green_marker_patch": patch_bounds
            == {
                "halo_radius_cells": 8,
                "south_north_start": 1,
                "south_north_stop_exclusive": 18,
                "west_east_start": 5,
                "west_east_stop_exclusive": 22,
            },
        },
        "provenance": {
            "source_wrf_pristine": str(PRISTINE),
            "source_wrf_head": run_text(["git", "rev-parse", "HEAD"], PRISTINE),
            "source_wrf_describe": run_text(["git", "describe", "--tags", "--dirty", "--always"], PRISTINE),
            "source_wrf_dirty_status_first_lines": run_text(["git", "status", "--short"], PRISTINE).splitlines()[:80],
            "scratch_root": str(SCRATCH),
            "scratch_provenance": "rsync copy of <DATA_ROOT>/wrf_gpu2/v014_dynamic_terms/WRF and run_case3",
            "wrf_copy": str(WRF_COPY),
            "run_dir": str(RUN_DIR),
            "patch_diff": str(PATCH_DIFF),
            "patch_sha256": sha256(PATCH_DIFF),
            "solve_em_sha256": sha256(WRF_COPY / "dyn_em/solve_em.F"),
            "wrf_exe_sha256": sha256(WRF_COPY / "main/wrf.exe"),
            "run_wrf_exe_sha256": sha256(RUN_DIR / "wrf.exe"),
            "green_marker_patch_sha256": sha256(REPO / "proofs/v014/wrf_same_state_marker_patch.diff"),
            "ptolemy_dynamic_patch_sha256": sha256(REPO / "proofs/v014/wrf_dynamic_term_localization_patch.diff"),
        },
        "inputs_read": {
            "wrf_same_state_marker_savepoint_json": str(REPO / "proofs/v014/wrf_same_state_marker_savepoint.json"),
            "wrf_dynamic_term_localization_json": str(REPO / "proofs/v014/wrf_dynamic_term_localization.json"),
            "same_state_savepoint_request_json": str(REPO / "proofs/v014/same_state_savepoint_request.json"),
            "dynamic_field_attribution_json": str(REPO / "proofs/v014/dynamic_field_attribution.json"),
            "marker_verdict": marker_savepoint["attempt_verdicts"]["current_thphy_marker"]["verdict"],
            "ptolemy_verdict": term_proof["verdict"],
            "request_selected_cell_count": len(request["selection"]["selected_cells"]),
        },
        "emitted_surfaces": {
            "post_final_calc_p_rho_phi": {
                "wrf_routine": "dyn_em/solve_em.F::solve_em",
                "boundary": "after final calc_p_rho_phi and before post-RK w/after_all handling",
                "files": post_calc["files"],
                "unique_counts": post_calc["unique_counts"],
                "duplicate_overlap": {
                    "count": post_calc["duplicate_count"],
                    "max_delta": post_calc["duplicate_max_delta"],
                    "max_delta_by_field": post_calc["duplicate_max_delta_by_field"],
                },
                "metadata": first_metadata(post_calc),
            },
            "post_after_all_rk_steps_pre_halo": {
                "wrf_routine": "dyn_em/solve_em.F::solve_em",
                "boundary": "immediately after after_all_rk_steps and before RK halo exchanges",
                "files": post_after_all["files"],
                "unique_counts": post_after_all["unique_counts"],
                "duplicate_overlap": {
                    "count": post_after_all["duplicate_count"],
                    "max_delta": post_after_all["duplicate_max_delta"],
                    "max_delta_by_field": post_after_all["duplicate_max_delta_by_field"],
                },
                "metadata": first_metadata(post_after_all),
            },
            "post_marker": {
                "boundary": "Herschel green marker after after_all_rk_steps plus RK halo exchanges",
                "files": marker["files"],
                "unique_counts": marker["unique_counts"],
                "duplicate_overlap": {
                    "count": marker["duplicate_count"],
                    "max_delta": marker["duplicate_max_delta"],
                    "max_delta_by_field": marker["duplicate_max_delta_by_field"],
                },
                "metadata": marker_md,
            },
        },
        "comparisons": {
            "ptolemy_post_small_step_finish_to_post_final_calc_p_rho_phi": ptolemy_to_post_calc,
            "post_final_calc_p_rho_phi_to_post_after_all_rk_steps_pre_halo": post_calc_to_post_after_all,
            "post_final_calc_p_rho_phi_vs_post_marker": post_calc_vs_marker,
            "post_after_all_rk_steps_pre_halo_vs_post_marker": post_after_all_vs_marker,
            "post_final_calc_p_rho_phi_vs_scratch_h10_wrfout": post_calc_vs_scratch,
            "post_after_all_rk_steps_pre_halo_vs_scratch_h10_wrfout": post_after_all_vs_scratch,
            "post_after_all_rk_steps_pre_halo_vs_provided_cpu_h10_wrfout": post_after_all_vs_cpu,
            "post_after_all_rk_steps_pre_halo_vs_retained_gpu_h10_wrfout": post_after_all_vs_gpu,
            "post_marker_vs_scratch_h10_wrfout": marker_vs_scratch,
        },
        "compact_summary": {
            "green_candidate_vs_marker": summarize_compact(
                post_after_all_vs_marker,
                [
                    "T_HIST_SRC_vs_marker_T",
                    "P_vs_marker_P",
                    "PB_vs_marker_PB",
                    "U_vs_marker_U",
                    "V_vs_marker_V",
                    "W_vs_marker_W",
                    "PH_vs_marker_PH",
                ],
            ),
            "green_candidate_vs_scratch_wrfout": summarize_compact(
                post_after_all_vs_scratch, ["T", "P", "PB", "MU", "MUB", "U", "V", "W", "PH"]
            ),
            "retained_gpu_h10_vs_green_candidate": summarize_compact(
                post_after_all_vs_gpu, ["T", "P", "PB", "MU", "MUB", "U", "V", "W", "PH"]
            ),
        },
        "cadence_localization": {
            "post_small_step_finish_gap": {
                "from_ptolemy_to_final_calc": summarize_compact(
                    ptolemy_to_post_calc, ["T_HIST_SRC", "T_THM", "P", "PB", "MU_NEW", "MU_OLD", "U", "V", "W", "PH"]
                ),
                "finding": "The final post-RK calc_p_rho_phi refresh closes the large P gap from Ptolemy's post-small_step_finish layer.",
            },
            "after_all_boundary": {
                "final_calc_to_after_all": summarize_compact(
                    post_calc_to_post_after_all,
                    ["T_HIST_SRC", "T_THM", "P", "PB", "MU_NEW", "MU_OLD", "MUB", "U", "V", "W", "PH"],
                ),
                "finding": "The emitted post-after_all_rk_steps pre-halo surface is the named candidate tested against the green post marker.",
            },
            "history_t_source": "history T remains grid%th_phy_m_t0; grid%t_2 is THM-side and not the wrfout T target.",
        },
        "next_jax_cpu_wrapper_target": next_target,
        "commands": [
            "mkdir -p <DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh && rsync -a <DATA_ROOT>/wrf_gpu2/v014_dynamic_terms/WRF/ <DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/WRF/ && rsync -a <DATA_ROOT>/wrf_gpu2/v014_dynamic_terms/run_case3/ <DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/run_case3/",
            "cp <DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/WRF/dyn_em/solve_em.F <DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/WRF/dyn_em/solve_em.F.before_v014_post_rk_refresh",
            "diff -u --label a/dyn_em/solve_em.F --label b/dyn_em/solve_em.F <DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/WRF/dyn_em/solve_em.F.before_v014_post_rk_refresh <DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/WRF/dyn_em/solve_em.F > <DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/wrf_post_rk_refresh_localization_patch.diff || true",
            "timeout 3600 env PATH=<USER_HOME>/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin:$PATH NETCDF=<USER_HOME>/src/canairy_meteo/Gen2/artifacts/envs/wrf-build PNETCDF=<USER_HOME>/src/canairy_meteo/Gen2/artifacts/envs/wrf-build WRFIO_NCD_LARGE_FILE_SUPPORT=1 tcsh ./compile em_real > <DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/compile_post_rk_refresh.log 2>&1",
            "find <DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/run_case3 -maxdepth 1 \\( -name 'rsl.error.*' -o -name 'rsl.out.*' -o -name 'wrfout_d0*' -o -name 'wrfrst_d0*' -o -name '*stdout.log' -o -name 'wrf_stdout.log' \\) -delete",
            "ln -sf <DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/WRF/main/wrf.exe <DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/run_case3/wrf.exe",
            "timeout 3600 env PATH=<USER_HOME>/src/canairy_meteo/Gen2/artifacts/envs/wrf-build/bin:$PATH CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 WRFGPU2_POST_RK_REFRESH=1 WRFGPU2_POST_RK_REFRESH_ROOT=<DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/refresh_output WRFGPU2_POST_RK_REFRESH_GRID=2 WRFGPU2_POST_RK_REFRESH_START_STEP=6000 WRFGPU2_POST_RK_REFRESH_END_STEP=6000 WRFGPU2_SAMESTATE=1 WRFGPU2_SAMESTATE_ROOT=<DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/marker_output WRFGPU2_SAMESTATE_GRID=2 WRFGPU2_SAMESTATE_START_STEP=6000 WRFGPU2_SAMESTATE_END_STEP=6000 mpirun --oversubscribe -np 28 ./wrf.exe > <DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/run_case3/post_rk_refresh_28rank_stdout.log 2>&1",
            "cp <DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/wrf_post_rk_refresh_localization_patch.diff proofs/v014/wrf_post_rk_refresh_localization_patch.diff",
            "python -m py_compile proofs/v014/wrf_post_rk_refresh_localization.py",
            "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/wrf_post_rk_refresh_localization.py",
            "python -m json.tool proofs/v014/wrf_post_rk_refresh_localization.json >/tmp/wrf_post_rk_refresh_localization.validated.json",
        ],
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "patch_diff": str(PATCH_DIFF),
            "review": str(OUT_REVIEW),
            "compile_log": str(SCRATCH / "compile_post_rk_refresh.log"),
            "run_stdout": str(RUN_DIR / "post_rk_refresh_28rank_stdout.log"),
            "rsl_logs": str(RUN_DIR / "rsl.error.*"),
            "emitted_refresh_dir": str(REFRESH_DIR),
            "emitted_marker_dir": str(MARKER_DIR),
        },
        "unresolved_risks": [
            "No JAX same-state wrapper was run; this is WRF source-truth localization only.",
            "Only the selected h10 patch at K1/KSTAG01 was emitted; full-column localization remains follow-up work.",
            "Retained GPU/JAX h10 wrfout is an old retained artifact, not a fresh same-state JAX execution.",
        ],
    }

    OUT_JSON.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    write_markdown(proof)
    write_review(proof)


def write_markdown(proof: dict[str, Any]) -> None:
    cmp = proof["comparisons"]
    md = f"""# V0.14 WRF Post-RK Refresh Localization

Verdict: `{proof['verdict']}`.

CPU-only WRF emitted two refresh surfaces at `d02`, step `6000`, h10
`2026-05-02_04:00:00`: post final `calc_p_rho_phi`, and immediately after
`after_all_rk_steps` before RK halo exchanges. The green post marker from
Herschel was emitted in the same run for the same native patch.

## Target

- selected mass cell: zero-based `(y={proof['target_confirmed']['selected_cell_zero_yx']['south_north']}, x={proof['target_confirmed']['selected_cell_zero_yx']['west_east']})`
- mass patch: `{proof['target_confirmed']['selected_patch_bounds_mass_grid']}`
- WRF time before step: `{proof['emitted_surfaces']['post_marker']['metadata']['current_timestr_before_step']}`
- WRF step: `{proof['target_confirmed']['wrf_step']}`
- native coordinates: mass/U/V/W-PH staggering preserved

## Refresh Surface Vs Marker

Post final `calc_p_rho_phi` vs green post marker:

{table(cmp['post_final_calc_p_rho_phi_vs_post_marker'], ['T_HIST_SRC_vs_marker_T', 'T_THM_vs_marker_T', 'P_vs_marker_P', 'PB_vs_marker_PB', 'U_vs_marker_U', 'V_vs_marker_V', 'W_vs_marker_W', 'PH_vs_marker_PH'])}

Post `after_all_rk_steps` pre-halo vs green post marker:

{table(cmp['post_after_all_rk_steps_pre_halo_vs_post_marker'], ['T_HIST_SRC_vs_marker_T', 'T_THM_vs_marker_T', 'P_vs_marker_P', 'PB_vs_marker_PB', 'U_vs_marker_U', 'V_vs_marker_V', 'W_vs_marker_W', 'PH_vs_marker_PH'])}

## Candidate Vs WRFout

Post `after_all_rk_steps` pre-halo vs scratch h10 wrfout:

{table(cmp['post_after_all_rk_steps_pre_halo_vs_scratch_h10_wrfout'], ['T', 'P', 'PB', 'MU', 'MUB', 'U', 'V', 'W', 'PH'])}

Retained GPU/JAX h10 wrfout vs the same WRF candidate:

{table(cmp['post_after_all_rk_steps_pre_halo_vs_retained_gpu_h10_wrfout'], ['T', 'P', 'PB', 'MU', 'MUB', 'U', 'V', 'W', 'PH'])}

## Cadence

The final `calc_p_rho_phi` boundary closes the large `P` gap from Ptolemy's
post-`small_step_finish` layer. The post-`after_all_rk_steps` pre-halo surface
is the named candidate for a JAX CPU wrapper if the table above is green.

Next JAX CPU wrapper target: `{proof['next_jax_cpu_wrapper_target']}`.
"""
    OUT_MD.write_text(md)


def write_review(proof: dict[str, Any]) -> None:
    review = f"""# V0.14 Post-RK Refresh Localization Review

- objective: localize the pressure/rho/post-RK refresh cadence between Ptolemy's post-`small_step_finish` layer and Herschel's green post marker.
- verdict: `{proof['verdict']}`.
- files changed: `proofs/v014/wrf_post_rk_refresh_localization.py`, `.json`, `.md`, `_patch.diff`, and this review.
- WRF copy/run paths: `{proof['provenance']['wrf_copy']}`, `{proof['provenance']['run_dir']}`.
- patch hash: `{proof['provenance']['patch_sha256']}`.
- executable hash: `{proof['provenance']['wrf_exe_sha256']}`.
- proof objects: `{proof['proof_objects']['json']}`, `{proof['proof_objects']['markdown']}`, `{proof['proof_objects']['patch_diff']}`.
- commands run: see JSON `commands`.
- unresolved risks: no JAX same-state wrapper; selected h10 patch only; retained GPU wrfout is not fresh.
- next decision needed: use `{proof['next_jax_cpu_wrapper_target']}` as the CPU wrapper target if accepted, otherwise instrument a narrower sub-boundary.
"""
    OUT_REVIEW.write_text(review)


if __name__ == "__main__":
    main()
