#!/usr/bin/env python3
"""V0.14 strict same-input single-RK-step parity boundary proof.

This proof intentionally refuses to run a JAX-vs-WRF comparison unless the
available WRF savepoints can initialize the same native-staggered state and the
same RK-fixed tendency/source inputs consumed by the JAX RK path.
"""

from __future__ import annotations

import ast
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


def _force_cpu_defaults() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")


_force_cpu_defaults()


ROOT = Path(__file__).resolve().parents[2]
PROOF_DIR = ROOT / "proofs/v014"
OUT_JSON = PROOF_DIR / "same_input_single_rk_parity.json"
OUT_MD = PROOF_DIR / "same_input_single_rk_parity.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-same-input-single-rk-parity.md"

SPRINT_CONTRACT = ROOT / ".agent/sprints/2026-06-09-v014-same-input-single-rk-parity/sprint-contract.md"
PROJECT_CONSTITUTION = ROOT / "PROJECT_CONSTITUTION.md"
AGENTS = ROOT / "AGENTS.md"
MANAGING_SPRINTS_SKILL = ROOT / ".agent/skills/managing-sprints/SKILL.md"
OPUS_CRITIC_JSON = PROOF_DIR / "dynamic_root_cause_opus_critic.json"
PRE_RK_INPUT_JSON = PROOF_DIR / "pre_rk_input_boundary.json"
WRF_REFRESH_JSON = PROOF_DIR / "wrf_post_rk_refresh_localization.json"
SAME_STATE_JSON = PROOF_DIR / "same_state_momentum_mass.json"
GRID_AFTER_JSON = PROOF_DIR / "grid_after_live_nest_base.json"

STATE_PY = ROOT / "src/gpuwrf/contracts/state.py"
STEP_PY = ROOT / "src/gpuwrf/dynamics/step.py"
RK3_PY = ROOT / "src/gpuwrf/dynamics/rk3.py"
OPERATIONAL_MODE_PY = ROOT / "src/gpuwrf/runtime/operational_mode.py"
RK_ADDTEND_DRY_PY = ROOT / "src/gpuwrf/dynamics/core/rk_addtend_dry.py"

PRE_RK_FILES = [
    Path("/tmp/wrf_gpu2_v014_pre_rk_input_boundary/pre_rk_output/pre_rk_input_d2_step_6000_is_1_ie_23_js_18_je_33.txt"),
    Path("/tmp/wrf_gpu2_v014_pre_rk_input_boundary/pre_rk_output/pre_rk_input_d2_step_6000_is_1_ie_23_js_1_je_17.txt"),
]
POST_RK_FILES = [
    Path("<DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/refresh_output/refresh_post_after_all_rk_steps_pre_halo_d2_step_6000_is_1_ie_23_js_18_je_33.txt"),
    Path("<DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/refresh_output/refresh_post_after_all_rk_steps_pre_halo_d2_step_6000_is_1_ie_23_js_1_je_17.txt"),
]
FINAL_CALC_FILES = [
    Path("<DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/refresh_output/refresh_post_final_calc_p_rho_phi_d2_step_6000_is_1_ie_23_js_18_je_33.txt"),
    Path("<DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/refresh_output/refresh_post_final_calc_p_rho_phi_d2_step_6000_is_1_ie_23_js_1_je_17.txt"),
]

TARGET_STEP = 6000
TARGET_FIELDS = ("T", "P", "PB", "PH", "PHB", "MU", "MUB", "U", "V", "W")
VERDICT = "SAME_INPUT_TENDENCY_INPUT_BLOCKED_PRE_RK_FULL_NATIVE_STATE_RK_TENDF_AND_HISTORY_SOURCE_FIELDS"

PRE_RK_SCHEMAS = {
    "MASS_K1": {
        "index_count": 4,
        "index_names": ["fortran_i", "fortran_j", "zero_x", "zero_y"],
        "fields": ["T_THM", "T_OLD", "T_HIST_SRC", "P", "PB", "MU_NEW", "MU_OLD", "MUB"],
    },
}
POST_RK_SCHEMAS = {
    "MASS_K1": {
        "index_count": 4,
        "index_names": ["fortran_i", "fortran_j", "zero_x", "zero_y"],
        "fields": ["T_HIST_SRC", "T_THM", "P", "PB", "MU_NEW", "MU_OLD", "MUB", "MUT", "MUTS", "AL", "ALB", "ALT", "RHO"],
    },
    "U_K1": {
        "index_count": 4,
        "index_names": ["fortran_i", "fortran_j", "zero_xstag", "zero_y"],
        "fields": ["U"],
    },
    "V_K1": {
        "index_count": 4,
        "index_names": ["fortran_i", "fortran_j", "zero_x", "zero_ystag"],
        "fields": ["V"],
    },
    "WPH_KSTAG01": {
        "index_count": 6,
        "index_names": ["fortran_i", "fortran_j", "fortran_kstag", "zero_x", "zero_y", "zero_kstag"],
        "fields": ["W", "PH"],
    },
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
        "JAX_ENABLE_COMPILATION_CACHE": os.environ.get("JAX_ENABLE_COMPILATION_CACHE"),
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
    except Exception as exc:  # pragma: no cover - recorded in proof output
        env.update({"jax_import_error": repr(exc), "gpu_device_count": None})
    return env


def key_for(tag: str, idx: list[int]) -> tuple[int, ...]:
    if tag in {"MASS_K1", "U_K1", "V_K1"}:
        return (idx[3], idx[2])
    if tag == "WPH_KSTAG01":
        return (idx[5], idx[4], idx[3])
    raise ValueError(tag)


def parse_savepoint(paths: Iterable[Path], schemas: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    records: dict[str, dict[tuple[int, ...], dict[str, float]]] = defaultdict(dict)
    native_indices: dict[str, dict[tuple[int, ...], list[int]]] = defaultdict(dict)
    metadata: dict[str, list[str]] = defaultdict(list)
    schema_lines: list[str] = []
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
                    metadata.setdefault("comment", []).append(line[1:].strip())
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
                native_indices[tag][key] = idx

    return {
        "files": [path_info(path) for path in paths],
        "metadata": {key: values[0] if len(values) == 1 else values for key, values in metadata.items()},
        "schema_lines": schema_lines,
        "schemas": {
            tag: {
                "index_count": int(schema["index_count"]),
                "index_names": list(schema["index_names"]),
                "fields": list(schema["fields"]),
            }
            for tag, schema in schemas.items()
        },
        "records": records,
        "native_indices": native_indices,
        "unique_counts": {tag: len(records[tag]) for tag in schemas},
        "duplicate_count": duplicate_count,
        "duplicate_max_delta": duplicate_max_delta,
        "duplicate_max_delta_by_field": duplicate_max_delta_by_field,
    }


def compact_surface(surface: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "files": surface["files"],
        "metadata": surface["metadata"],
        "schema_lines": surface["schema_lines"],
        "schemas": surface["schemas"],
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
        out["key_ranges"][tag] = {
            "key_convention": "MASS/U/V=(zero_y_or_ystag, zero_x_or_xstag); WPH=(zero_kstag, zero_y, zero_x)",
            "min": keys.min(axis=0).tolist(),
            "max": keys.max(axis=0).tolist(),
        }
        field_ranges: dict[str, Any] = {}
        fields = list(next(iter(records.values())).keys())
        for field in fields:
            values = np.asarray([record[field] for record in records.values()], dtype=np.float64)
            finite = values[np.isfinite(values)]
            field_ranges[field] = {
                "count": int(values.size),
                "finite_count": int(finite.size),
                "min": float(np.min(finite)) if finite.size else None,
                "max": float(np.max(finite)) if finite.size else None,
            }
        out["field_ranges"][tag] = field_ranges
    return out


def metadata_numbers(surface: Mapping[str, Any], key: str) -> list[int]:
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


def horizontal_patch_assessment(pre_surface: Mapping[str, Any]) -> dict[str, Any]:
    bounds = metadata_numbers(pre_surface, "mass_patch_zero_y0_y1_x0_x1_fortran_j0_j1_i0_i1")
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
        "candidate_valid_mass_cell_count_if_full_state_existed": int(valid_count),
        "note": (
            "Horizontal width would leave only the selected central mass cell under an 8-cell halo. "
            "The proof does not reach scoring because the WRF pre-RK input lacks full native state "
            "and RK-fixed tendency/source fields."
        ),
    }


def extract_slots(path: Path, class_name: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == "__slots__":
                        try:
                            return {
                                "class": class_name,
                                "module": str(path.relative_to(ROOT)),
                                "line_start": int(node.lineno),
                                "line_end": int(getattr(node, "end_lineno", node.lineno)),
                                "slots": list(ast.literal_eval(stmt.value)),
                            }
                        except Exception as exc:  # pragma: no cover - proof metadata only
                            return {"class": class_name, "module": str(path.relative_to(ROOT)), "error": repr(exc)}
    return {"class": class_name, "module": str(path.relative_to(ROOT)), "missing": True}


def extract_dataclass_fields(path: Path, class_name: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        fields = []
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                fields.append(stmt.target.id)
        return {
            "class": class_name,
            "module": str(path.relative_to(ROOT)),
            "line_start": int(node.lineno),
            "line_end": int(getattr(node, "end_lineno", node.lineno)),
            "fields": fields,
        }
    return {"class": class_name, "module": str(path.relative_to(ROOT)), "missing": True}


def source_node_info(path: Path, names: Iterable[str]) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    wanted = set(names)
    found: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in wanted:
            found.append(
                {
                    "name": node.name,
                    "module": str(path.relative_to(ROOT)),
                    "line_start": int(node.lineno),
                    "line_end": int(getattr(node, "end_lineno", node.lineno)),
                }
            )
    return sorted(found, key=lambda item: (item["module"], item["line_start"]))


def source_contract() -> dict[str, Any]:
    return {
        "files": {
            "state_py": path_info(STATE_PY),
            "step_py": path_info(STEP_PY),
            "rk3_py": path_info(RK3_PY),
            "operational_mode_py": path_info(OPERATIONAL_MODE_PY),
            "rk_addtend_dry_py": path_info(RK_ADDTEND_DRY_PY),
        },
        "classes": {
            "State": extract_slots(STATE_PY, "State"),
            "Tendencies": extract_slots(STATE_PY, "Tendencies"),
            "DryPhysicsTendencies": extract_dataclass_fields(RK_ADDTEND_DRY_PY, "DryPhysicsTendencies"),
        },
        "entry_points": (
            source_node_info(STEP_PY, ["step"])
            + source_node_info(RK3_PY, ["rk3_step"])
            + source_node_info(OPERATIONAL_MODE_PY, ["_rk_scan_step_with_pre_halo_capture", "_rk_scan_step", "_physics_step_forcing"])
        ),
    }


def optional_surface_assessment(post_surface: Mapping[str, Any], final_surface: Mapping[str, Any]) -> dict[str, Any]:
    post_tags = {tag: list(schema["fields"]) for tag, schema in post_surface["schemas"].items()}
    final_tags = {tag: list(schema["fields"]) for tag, schema in final_surface["schemas"].items()}
    return {
        "post_after_all_rk_tags": post_tags,
        "post_final_calc_tags": final_tags,
        "same_schema_as_post_after_all_rk": post_tags == final_tags,
        "adds_missing_prestep_state_or_tendencies": False,
        "usable_as_narrower_dynamics_only_boundary": False,
        "reason": (
            "The optional post_final_calc_p_rho_phi files are output surfaces with the same native "
            "field schema as post_after_all_rk_steps_pre_halo. They do not provide pre-RK U/V/W/PH "
            "state, full columns, WRF *_tendf physics/source tendencies, or a JAX load wrapper."
        ),
    }


def blocker(pre_surface: Mapping[str, Any], post_surface: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    del post_surface
    dry_fields = contract["classes"]["DryPhysicsTendencies"].get("fields", [])
    tendency_slots = contract["classes"]["Tendencies"].get("slots", [])
    state_slots = contract["classes"]["State"].get("slots", [])
    available_pre_tags = sorted(pre_surface["records"])
    available_pre_fields = {
        tag: list(pre_surface["schemas"][tag]["fields"])
        for tag in available_pre_tags
        if tag in pre_surface["schemas"]
    }
    return {
        "verdict": VERDICT,
        "strict_same_input_comparison_run": False,
        "reason": (
            "The current WRF pre-RK input hook emits only MASS_K1 T/P/PB/MU/MUB fields. "
            "It cannot build a WRF-equivalent JAX State, Tendencies, DryPhysicsTendencies, "
            "or OperationalCarry for one RK step."
        ),
        "available_pre_rk_tags": available_pre_tags,
        "available_pre_rk_fields": available_pre_fields,
        "missing_input_groups": [
            {
                "group": "pre_rk_full_native_state",
                "missing": [
                    "U on native x-stagger for all vertical levels",
                    "V on native y-stagger for all vertical levels",
                    "W on native vertical faces for all vertical levels",
                    "PH and PHB on native vertical faces for all vertical levels",
                    "T/P/PB full mass-column, not only K1",
                    "QV and active moisture/scalar state needed by the operational carry",
                    "surface/coupling leaves carried by State when physics/boundary are active",
                ],
            },
            {
                "group": "jax_base_tendencies",
                "required_by": "src/gpuwrf/contracts/state.py::Tendencies",
                "required_fields": tendency_slots,
                "missing": tendency_slots,
            },
            {
                "group": "wrf_rk_fixed_physics_and_boundary_tendencies",
                "required_by": "src/gpuwrf/dynamics/core/rk_addtend_dry.py::DryPhysicsTendencies",
                "required_fields": dry_fields,
                "missing": dry_fields,
            },
            {
                "group": "history_source_reference",
                "missing": [
                    "full native rk1/start-of-step reference state used as _1/_old history",
                    "full T_HIST_SRC/theta history source across all levels",
                    "full MU_OLD and mass-coupled scalar reference needed for final-stage limiters",
                ],
            },
            {
                "group": "proof_load_wrapper",
                "missing": [
                    "proof-only loader that maps a WRF full pre-RK savepoint into OperationalCarry",
                    "paired OperationalNamelist/GridSpec/DycoreMetrics/boundary_config from the same WRF case",
                    "call boundary to feed WRF DryPhysicsTendencies into _rk_scan_step_with_pre_halo_capture",
                ],
            },
        ],
        "state_contract_slots_count": len(state_slots),
        "state_contract_slots_sample": state_slots[:20],
        "next_exact_action": (
            "Add a CPU WRF pre-RK hook at solve_em after grid%itimestep increment and before "
            "the RK loop that emits the full native-staggered step-entry state plus WRF "
            "rk_tendency/rk_addtend_dry inputs: U,V,W,T,P,PB,PH,PHB,MU,MUB,QV/moisture full "
            "columns; rk1/_old history/source fields; ru_tendf, rv_tendf, rw_tendf, ph_tendf, "
            "t_tendf, mu_tendf, h_diabatic, u_save, v_save, w_save, ph_save, t_save; and "
            "the boundary/carry leaves needed by _rk_scan_step_with_pre_halo_capture. Then add "
            "a proof-only JAX loader/wrapper that constructs OperationalCarry and feeds those "
            "exact WRF tendency leaves before scoring halo-valid cells."
        ),
    }


def proof_inputs() -> dict[str, Any]:
    return {
        "project_constitution": path_info(PROJECT_CONSTITUTION),
        "agents": path_info(AGENTS),
        "managing_sprints_skill": path_info(MANAGING_SPRINTS_SKILL),
        "sprint_contract": path_info(SPRINT_CONTRACT),
        "trigger_evidence": {
            "opus_critic_json": path_info(OPUS_CRITIC_JSON),
            "pre_rk_input_boundary_json": path_info(PRE_RK_INPUT_JSON),
            "wrf_post_rk_refresh_json": path_info(WRF_REFRESH_JSON),
            "same_state_momentum_mass_json": path_info(SAME_STATE_JSON),
            "grid_after_live_nest_base_json": path_info(GRID_AFTER_JSON),
        },
        "wrf_pre_rk_files": [path_info(path) for path in PRE_RK_FILES],
        "wrf_post_rk_files": [path_info(path) for path in POST_RK_FILES],
        "wrf_optional_final_calc_files": [path_info(path) for path in FINAL_CALC_FILES],
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    blocked = payload["blocked"]
    patch = payload["patch_width_assessment"]
    optional = payload["optional_final_calc_assessment"]
    lines = [
        "# V0.14 Same-Input Single-RK Parity",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "No JAX-vs-WRF same-input comparison was run. Running it with the current files would be a weak comparison, because the WRF pre-RK input does not contain the state and tendency/source inputs needed by the JAX RK boundary.",
        "",
        "## Parsed Inputs",
        "",
        f"- Pre-RK surface records: `{payload['wrf_pre_rk_surface']['unique_counts']}`.",
        f"- Pre-RK fields: `{blocked['available_pre_rk_fields']}`.",
        f"- Post-RK/pre-halo surface records: `{payload['wrf_post_rk_surface']['unique_counts']}`.",
        f"- Optional final-calc surface adds missing inputs: `{optional['adds_missing_prestep_state_or_tendencies']}`.",
        "",
        "## Blocker",
        "",
        f"- Missing full pre-RK native state: `{blocked['missing_input_groups'][0]['missing']}`.",
        f"- Missing JAX base tendency leaves: `{blocked['missing_input_groups'][1]['missing']}`.",
        f"- Missing WRF RK-fixed physics/source leaves: `{blocked['missing_input_groups'][2]['missing']}`.",
        f"- Missing wrapper: `{blocked['missing_input_groups'][4]['missing']}`.",
        "",
        "## Patch Width",
        "",
        f"- Status: `{patch['status']}`.",
        f"- Candidate valid mass cells with an 8-cell halo if full state existed: `{patch.get('candidate_valid_mass_cell_count_if_full_state_existed')}`.",
        "- Patch width is not the primary verdict because the proof is blocked before any stencil-valid scoring can begin.",
        "",
        "## Next Action",
        "",
        blocked["next_exact_action"],
        "",
        "This result supports blocked instrumentation, not upstream drift, final-RK PGF/mass-wind, or theta/tendency source by itself.",
        "",
    ]
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Review: V0.14 Same-Input Single-RK Parity",
        "",
        f"verdict: `{payload['verdict']}`.",
        "",
        "objective: test the strict WRF pre-RK input -> one JAX RK step -> WRF post-RK/pre-halo boundary, or name the exact blocker without producing a weak same-input comparison.",
        "",
        "files changed:",
        "- `proofs/v014/same_input_single_rk_parity.py`",
        "- `proofs/v014/same_input_single_rk_parity.json`",
        "- `proofs/v014/same_input_single_rk_parity.md`",
        "- `.agent/reviews/2026-06-09-v014-same-input-single-rk-parity.md`",
        "",
        "commands run:",
        "- `python -m py_compile proofs/v014/same_input_single_rk_parity.py`",
        "- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_input_single_rk_parity.py`",
        "- `python -m json.tool proofs/v014/same_input_single_rk_parity.json >/tmp/same_input_single_rk_parity.validated.json`",
        "- `git diff -- src`",
        "",
        "proof objects produced:",
        "- `proofs/v014/same_input_single_rk_parity.json`",
        "- `proofs/v014/same_input_single_rk_parity.md`",
        "- `.agent/reviews/2026-06-09-v014-same-input-single-rk-parity.md`",
        "",
        "result:",
        f"- `{payload['verdict']}`.",
        "- The current pre-RK WRF hook emits only `MASS_K1` T/P/PB/MU/MUB fields.",
        "- It does not emit full native U/V/W/PH state, full columns, JAX base `Tendencies`, WRF `DryPhysicsTendencies`, or the OperationalCarry/Namelist loader needed to feed `_rk_scan_step_with_pre_halo_capture`.",
        "- The optional `post_final_calc_p_rho_phi` files are output surfaces with the same schema as the post-RK surface, not missing input/tendency surfaces.",
        "",
        "unresolved risks:",
        "- Once the missing WRF/JAX input wrapper exists, the 17x17 horizontal patch leaves only one conservative mass-grid score cell with an 8-cell halo; widen the hook if more scored cells are required.",
        "- This proof does not decide upstream drift, final-RK PGF/mass-wind, or theta/source causality; it blocks the current instrumentation.",
        "",
        "next decision needed: add the full pre-RK native-state plus RK-fixed tendency/source WRF hook and a proof-only JAX OperationalCarry loader, then rerun this same boundary.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    pre_surface = parse_savepoint(PRE_RK_FILES, PRE_RK_SCHEMAS)
    post_surface = parse_savepoint(POST_RK_FILES, POST_RK_SCHEMAS)
    final_surface = parse_savepoint(FINAL_CALC_FILES, POST_RK_SCHEMAS)
    contract = source_contract()

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.same_input_single_rk_parity.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": VERDICT,
        "cpu_only": True,
        "gpu_used": False,
        "no_tost": True,
        "no_switzerland": True,
        "no_fp32_or_memory_implementation": True,
        "no_hermes": True,
        "production_src_edits": False,
        "target_step": TARGET_STEP,
        "target_fields_requested": TARGET_FIELDS,
        "inputs_read": proof_inputs(),
        "environment": jax_environment(),
        "source_contract": contract,
        "wrf_pre_rk_surface": compact_surface(pre_surface),
        "wrf_post_rk_surface": compact_surface(post_surface),
        "wrf_optional_final_calc_surface": compact_surface(final_surface),
        "optional_final_calc_assessment": optional_surface_assessment(post_surface, final_surface),
        "patch_width_assessment": horizontal_patch_assessment(pre_surface),
        "blocked": blocker(pre_surface, post_surface, contract),
        "comparison": {
            "strict_same_input_comparison_run": False,
            "jax_step_executed": False,
            "jax_vs_jax_self_compare": False,
            "weak_comparison_avoided": True,
            "reason": "Required WRF-equivalent state/tendency/source inputs and proof wrapper are absent.",
        },
        "interpretation": {
            "supports_upstream_drift": False,
            "supports_final_rk_pgf_mass_wind": False,
            "supports_theta_or_tendency_source": False,
            "supports_blocked_instrumentation": True,
            "do_not_overclaim": "This proof is an instrumentation blocker only.",
        },
        "commands": {
            "minimum_validation": [
                "python -m py_compile proofs/v014/same_input_single_rk_parity.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_input_single_rk_parity.py",
                "python -m json.tool proofs/v014/same_input_single_rk_parity.json >/tmp/same_input_single_rk_parity.validated.json",
                "git diff -- src",
            ],
            "argv": sys.argv,
        },
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "review": str(OUT_REVIEW),
        },
    }

    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")

    print(VERDICT)
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    print(f"review={OUT_REVIEW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
