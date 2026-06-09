#!/usr/bin/env python3
"""V0.14 JAX after-all-RK same-state wrapper proof.

This proof is intentionally read-only with respect to production source.  It
inspects the current JAX runtime boundary, parses the accepted WRF pre-halo
truth surface, and records whether a CPU-only same-state JAX wrapper can emit
that surface without adding a source hook.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import platform
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    import jax
except Exception as exc:  # pragma: no cover - recorded in proof output
    jax = None
    JAX_IMPORT_ERROR = repr(exc)
else:
    JAX_IMPORT_ERROR = None

try:
    import netCDF4
except Exception as exc:  # pragma: no cover - recorded in proof output
    netCDF4 = None
    NETCDF4_IMPORT_ERROR = repr(exc)
else:
    NETCDF4_IMPORT_ERROR = None


ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "proofs/v014/jax_after_all_rk_wrapper.json"
OUT_MD = ROOT / "proofs/v014/jax_after_all_rk_wrapper.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-jax-after-all-rk-wrapper.md"

SPRINT_CONTRACT = ROOT / ".agent/sprints/2026-06-09-v014-jax-after-all-rk-wrapper/sprint-contract.md"
HANDOFF = ROOT / ".agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md"
WRF_REFRESH_JSON = ROOT / "proofs/v014/wrf_post_rk_refresh_localization.json"
WRF_REFRESH_MD = ROOT / "proofs/v014/wrf_post_rk_refresh_localization.md"
WRF_MARKER_JSON = ROOT / "proofs/v014/wrf_same_state_marker_savepoint.json"
SAVEPOINT_REQUEST_JSON = ROOT / "proofs/v014/same_state_savepoint_request.json"
DYNAMIC_ATTRIBUTION_JSON = ROOT / "proofs/v014/dynamic_field_attribution.json"

RUNTIME_PATH = ROOT / "src/gpuwrf/runtime/operational_mode.py"
STATE_PATH = ROOT / "src/gpuwrf/contracts/state.py"
SMALL_STEP_FINISH_PATH = ROOT / "src/gpuwrf/dynamics/core/small_step_finish.py"
CALC_P_RHO_PATH = ROOT / "src/gpuwrf/dynamics/core/calc_p_rho.py"

RUN_ID = "20260501_18z_l2_72h_20260519T173026Z"
RETAINED_JAX_H10_WRFOUT = (
    Path("/tmp/v0120_powered_tost_runs")
    / f"l2_d02_{RUN_ID}"
    / "wrfout_d02_2026-05-02_04:00:00"
)

COMPARE_ORDER = ("T", "P", "PB", "U", "V", "W", "PH", "MU", "MUB")
GREEN_TOLERANCE_MAX_ABS = 2.0e-6

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


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def key_for(record_type: str, idx: list[int]) -> tuple[int, ...]:
    if record_type in {"MASS_K1", "U_K1", "V_K1"}:
        return (idx[3], idx[2])
    if record_type == "WPH_KSTAG01":
        return (idx[5], idx[4], idx[3])
    raise ValueError(record_type)


def parse_refresh_files(paths: list[Path]) -> dict[str, Any]:
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
                idx = [int(x) for x in parts[1 : 1 + nidx]]
                values = [float(x) for x in parts[1 + nidx :]]
                fields = list(schema["fields"])
                if len(values) != len(fields):
                    raise ValueError(f"{path}: {tag} expected {len(fields)} values, got {len(values)}")
                item = dict(zip(fields, values))
                key = key_for(tag, idx)
                if key in records[tag]:
                    duplicate_count += 1
                    previous = records[tag][key]
                    for name in fields:
                        label = f"{tag}.{name}"
                        delta = abs(previous[name] - item[name])
                        duplicate_max_delta_by_field[label] = max(
                            duplicate_max_delta_by_field.get(label, 0.0), delta
                        )
                        duplicate_max_delta = max(duplicate_max_delta, delta)
                records[tag][key] = item

    return {
        "files": [str(path) for path in paths],
        "records": records,
        "metadata": {name: vals[0] if len(vals) == 1 else vals for name, vals in metadata.items()},
        "unique_counts": {name: len(items) for name, items in records.items()},
        "duplicate_count": duplicate_count,
        "duplicate_max_delta": duplicate_max_delta,
        "duplicate_max_delta_by_field": duplicate_max_delta_by_field,
    }


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


def wrfout_index(var: str, key: tuple[int, ...]) -> tuple[int, ...]:
    if var in {"T", "P", "PB"}:
        return (0, key[0], key[1])
    if var in {"MU", "MUB"}:
        return (key[0], key[1])
    if var in {"U", "V"}:
        return (0, key[0], key[1])
    if var in {"W", "PH"}:
        return (key[0], key[1], key[2])
    raise ValueError(var)


def compare_surface_to_wrfout(surface: dict[str, Any], wrfout: Path) -> dict[str, Any]:
    if netCDF4 is None:
        return {"status": "BLOCKED", "reason": f"netCDF4 import failed: {NETCDF4_IMPORT_ERROR}"}
    if not wrfout.exists():
        return {"status": "MISSING", "path": str(wrfout)}

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
    out: dict[str, Any] = {}
    with netCDF4.Dataset(wrfout) as ds:
        for label, (var, tag, field) in mapping.items():
            if var not in ds.variables:
                out[label] = {"status": "MISSING_VAR", "var": var}
                continue
            arr = np.asarray(np.ma.filled(ds.variables[var][0], np.nan), dtype=np.float64)
            diffs: list[float] = []
            worst: dict[str, Any] | None = None
            for key, item in surface["records"][tag].items():
                truth = float(item[field])
                candidate = float(arr[wrfout_index(var, key)])
                diff = truth - candidate
                diffs.append(diff)
                if worst is None or abs(diff) > worst["abs_diff"]:
                    worst = {
                        "native_key": list(key),
                        "wrf_truth": truth,
                        "candidate": candidate,
                        "diff_truth_minus_candidate": diff,
                        "abs_diff": abs(diff),
                    }
            out[label] = {
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


def extract_ast_node(path: Path, name: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            start = int(node.lineno)
            end = int(getattr(node, "end_lineno", node.lineno))
            lines = text.splitlines()
            source = "\n".join(lines[start - 1 : end])
            return {
                "module_path": str(path.relative_to(ROOT)),
                "name": name,
                "line_start": start,
                "line_end": end,
                "source_sha256": sha256_text(source),
                "contains_apply_halo": "apply_halo" in source,
                "contains_carry_from_finished_stage": "_carry_from_finished_stage" in source,
                "contains_refresh_grid_p": "_refresh_grid_p_from_finished" in source,
                "contains_small_step_finish": "small_step_finish_wrf" in source,
                "contains_calc_p_rho": "calc_p_rho_wrf" in source,
                "return_apply_halo_count": len(re.findall(r"return .*apply_halo", source)),
            }
    return {"module_path": str(path.relative_to(ROOT)), "name": name, "missing": True}


def extract_state_slots(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "State":
            for item in node.body:
                if not isinstance(item, ast.Assign):
                    continue
                if not any(isinstance(target, ast.Name) and target.id == "__slots__" for target in item.targets):
                    continue
                if isinstance(item.value, ast.Tuple):
                    return [
                        str(element.value)
                        for element in item.value.elts
                        if isinstance(element, ast.Constant) and isinstance(element.value, str)
                    ]
    return []


def jax_environment() -> dict[str, Any]:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jax_import_error": JAX_IMPORT_ERROR,
    }
    if jax is not None:
        try:
            env["jax_version"] = getattr(jax, "__version__", None)
            env["jax_default_backend"] = jax.default_backend()
            env["jax_devices"] = [str(device) for device in jax.devices()]
        except Exception as exc:  # pragma: no cover - recorded in proof output
            env["jax_device_probe_error"] = repr(exc)
    return env


def load_target_surface(wrf_refresh: Mapping[str, Any]) -> dict[str, Any]:
    target = wrf_refresh["emitted_surfaces"]["post_after_all_rk_steps_pre_halo"]
    paths = [Path(path) for path in target["files"]]
    return parse_refresh_files(paths)


def inspect_runtime() -> dict[str, Any]:
    functions = [
        extract_ast_node(RUNTIME_PATH, "_rk_scan_step"),
        extract_ast_node(RUNTIME_PATH, "_acoustic_scan"),
        extract_ast_node(RUNTIME_PATH, "_carry_from_finished_stage"),
        extract_ast_node(RUNTIME_PATH, "_refresh_grid_p_from_finished"),
        extract_ast_node(RUNTIME_PATH, "_physics_boundary_step_with_limiter_diagnostics"),
        extract_ast_node(RUNTIME_PATH, "_scan_forecast_segment"),
        extract_ast_node(RUNTIME_PATH, "run_forecast_operational"),
        extract_ast_node(RUNTIME_PATH, "run_forecast_operational_segmented"),
        extract_ast_node(SMALL_STEP_FINISH_PATH, "small_step_finish_wrf"),
    ]
    state_slots = extract_state_slots(STATE_PATH)
    return {
        "source_files": {
            "runtime_operational_mode": path_info(RUNTIME_PATH),
            "contracts_state": path_info(STATE_PATH),
            "small_step_finish": path_info(SMALL_STEP_FINISH_PATH),
            "calc_p_rho": path_info(CALC_P_RHO_PATH),
        },
        "functions": functions,
        "state_slots_count": len(state_slots),
        "state_slots": state_slots,
        "field_mapping_for_target": {
            "T": "State.theta - P0_THETA_OFFSET_K (WRF history perturbation theta)",
            "P": "State.p_perturbation",
            "PB": "State.p_total - State.p_perturbation",
            "U": "State.u",
            "V": "State.v",
            "W": "State.w",
            "PH": "State.ph_perturbation",
            "MU": "State.mu_perturbation",
            "MUB": "State.mu_total - State.mu_perturbation",
        },
        "closest_jax_boundary": {
            "wrf_surface": "post dyn_em/solve_em.F::after_all_rk_steps state before RK halo exchanges",
            "jax_runtime_boundary": (
                "final RK3 _carry_from_finished_stage(..., namelist), including "
                "_refresh_grid_p_from_finished, before _acoustic_scan applies apply_halo"
            ),
            "public_api_status": "not exposed",
            "evidence": [
                "_acoustic_scan returns next_carry.replace(state=apply_halo(next_carry.state, halo_spec(...))).",
                "_rk_scan_step advance_stage returns stage_carry.replace(state=apply_halo(...)) and returns the RK3 stage result.",
                "run_forecast_operational and segmented variants return carry.state after later guard/boundary handling, not the pre-halo carry.",
            ],
        },
    }


def compact_counts(surface: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "unique_counts": surface["unique_counts"],
        "duplicate_count": surface["duplicate_count"],
        "duplicate_max_delta": surface["duplicate_max_delta"],
        "fields": {
            "T": {"record_type": "MASS_K1", "source_field": "T_HIST_SRC", "count": len(surface["records"]["MASS_K1"])},
            "P": {"record_type": "MASS_K1", "source_field": "P", "count": len(surface["records"]["MASS_K1"])},
            "PB": {"record_type": "MASS_K1", "source_field": "PB", "count": len(surface["records"]["MASS_K1"])},
            "MU": {"record_type": "MASS_K1", "source_field": "MU_NEW", "count": len(surface["records"]["MASS_K1"])},
            "MUB": {"record_type": "MASS_K1", "source_field": "MUB", "count": len(surface["records"]["MASS_K1"])},
            "U": {"record_type": "U_K1", "source_field": "U", "count": len(surface["records"]["U_K1"])},
            "V": {"record_type": "V_K1", "source_field": "V", "count": len(surface["records"]["V_K1"])},
            "W": {"record_type": "WPH_KSTAG01", "source_field": "W", "count": len(surface["records"]["WPH_KSTAG01"])},
            "PH": {"record_type": "WPH_KSTAG01", "source_field": "PH", "count": len(surface["records"]["WPH_KSTAG01"])},
        },
    }


def proof_inputs() -> dict[str, Any]:
    return {
        "sprint_contract": path_info(SPRINT_CONTRACT),
        "handoff": path_info(HANDOFF),
        "wrf_refresh_json": path_info(WRF_REFRESH_JSON),
        "wrf_refresh_md": path_info(WRF_REFRESH_MD),
        "wrf_same_state_marker_json": path_info(WRF_MARKER_JSON),
        "same_state_savepoint_request_json": path_info(SAVEPOINT_REQUEST_JSON),
        "dynamic_field_attribution_json": path_info(DYNAMIC_ATTRIBUTION_JSON),
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    comparison = payload["available_retained_jax_writer_diagnostic"]
    first = comparison.get("first_mismatch_vs_wrf_truth")
    lines = [
        "# V0.14 JAX After-All-RK Same-State Wrapper",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Target",
        "",
        "- WRF truth surface: `post dyn_em/solve_em.F::after_all_rk_steps state before RK halo exchanges`.",
        "- Domain/lead: `d02`, step `6000`, h10 `2026-05-02T04:00:00+00:00`.",
        "- Patch: mass `y [1,18), x [5,22)`, native U/V/W/PH staggering preserved.",
        "",
        "## Runtime Finding",
        "",
        "The closest JAX runtime boundary is final RK3 `_carry_from_finished_stage(..., namelist)` after "
        "`_refresh_grid_p_from_finished`, but `_acoustic_scan` immediately wraps that state in "
        "`apply_halo(...)` before returning. `_rk_scan_step` and the public forecast entries expose only "
        "that later state, followed by guard/boundary handling.",
        "",
        "## WRF Oracle",
        "",
        f"- Parsed WRF pre-halo records: `{payload['wrf_truth_surface']['counts']['unique_counts']}`.",
        f"- Duplicate overlap max delta: `{payload['wrf_truth_surface']['counts']['duplicate_max_delta']}`.",
        "",
        "## Available Diagnostic",
        "",
        "A retained JAX/GPU h10 wrfout was compared against the WRF truth surface only as a non-acceptance diagnostic. "
        "It is not a CPU internal pre-halo state and is not used for the verdict.",
    ]
    if first:
        lines.append(
            f"- First retained-writer mismatch by contract order: `{first['field']}` "
            f"max_abs `{first['max_abs']}` with tolerance `{first['tolerance']}`."
        )
    else:
        lines.append("- Retained writer diagnostic was unavailable or had no mismatch above tolerance.")
    lines.extend(
        [
            "",
            "## Missing Wrapper Prerequisite",
            "",
            "- No production API or proof-only hook exposes the final RK3 post-refresh state before RK halo exchange.",
            "- No CPU JAX h10 same-state savepoint/checkpoint exists in the inspected proof inputs or scratch areas.",
            "- A full CPU forecast through the public API would return the wrong cadence surface, so it was not run.",
            "",
            "Next decision: open a narrow source-changing/debug-hook sprint, or approve a narrower wrapper sprint that adds a "
            "CPU-only pre-halo capture around `_acoustic_scan` immediately before its `apply_halo` return.",
            "",
        ]
    )
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Review: V0.14 JAX After-All-RK Same-State Wrapper",
            "",
            f"Verdict: `{payload['verdict']}`.",
            "",
            "Objective: compare JAX CPU internals to Boole's accepted WRF post-`after_all_rk_steps` pre-halo h10 surface.",
            "",
            "Files changed:",
            "- `proofs/v014/jax_after_all_rk_wrapper.py`",
            "- `proofs/v014/jax_after_all_rk_wrapper.json`",
            "- `proofs/v014/jax_after_all_rk_wrapper.md`",
            "- `.agent/reviews/2026-06-09-v014-jax-after-all-rk-wrapper.md`",
            "",
            "Result:",
            "- WRF truth files parsed and validated for the target patch.",
            "- JAX runtime/source mapping inspected without production edits.",
            "- Same-surface JAX CPU emission is blocked because the current runtime exposes only post-halo/post-guard state.",
            "- Retained JAX wrfout mismatch was recorded as diagnostic only; it is not an accepted same-state CPU comparison.",
            "",
            "Unresolved risks:",
            "- First failing JAX operator/cadence remains unnamed until a pre-halo JAX state hook or checkpoint exists.",
            "- A CPU full-domain h10 run may be expensive, but cost is secondary to the missing same-surface API.",
            "",
            "Next decision: authorize a narrow pre-halo capture hook/source sprint or a narrower wrapper sprint with an explicit allowed source edit.",
            "",
        ]
    )


def main() -> int:
    generated_utc = datetime.now(timezone.utc).isoformat()
    wrf_refresh = load_json(WRF_REFRESH_JSON)
    savepoint_request = load_json(SAVEPOINT_REQUEST_JSON)
    dynamic_attribution = load_json(DYNAMIC_ATTRIBUTION_JSON)
    marker = load_json(WRF_MARKER_JSON)

    target_surface = load_target_surface(wrf_refresh)
    runtime = inspect_runtime()
    retained_compare = compare_surface_to_wrfout(target_surface, RETAINED_JAX_H10_WRFOUT)
    retained_first = (
        first_mismatch(retained_compare, GREEN_TOLERANCE_MAX_ABS)
        if isinstance(retained_compare, Mapping) and retained_compare.get("status") not in {"MISSING", "BLOCKED"}
        else None
    )

    selected = savepoint_request["selection"]["selected_cells"][0]
    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.jax_after_all_rk_wrapper.v1",
        "generated_utc": generated_utc,
        "verdict": "WRAPPER_BLOCKED_NO_JAX_PRE_HALO_STATE_API",
        "cpu_only": True,
        "gpu_used": False,
        "no_hermes": True,
        "production_src_edits": False,
        "wrf_source_edits": False,
        "inputs_read": proof_inputs(),
        "environment": jax_environment(),
        "target": {
            "wrf_truth_surface": wrf_refresh["next_jax_cpu_wrapper_target"],
            "accepted_wrf_verdict": wrf_refresh["verdict"],
            "domain": wrf_refresh["target_confirmed"]["domain"],
            "wrf_step": wrf_refresh["target_confirmed"]["wrf_step"],
            "lead_seconds_after_step": wrf_refresh["target_confirmed"]["lead_seconds_after_step"],
            "valid_time_utc": wrf_refresh["target_confirmed"]["valid_time_utc"],
            "selected_cell_zero_yx": wrf_refresh["target_confirmed"]["selected_cell_zero_yx"],
            "selected_patch_bounds_mass_grid": wrf_refresh["target_confirmed"]["selected_patch_bounds_mass_grid"],
            "native_staggered_coordinates": wrf_refresh["target_confirmed"]["native_staggered_coordinates"],
            "marker_mapping": marker["marker_mapping"],
            "first_selected_cell_from_request": {
                "selection_rank": selected["selection_rank"],
                "mass_index_zero_based": selected["mass_index_zero_based"],
                "mass_index_fortran_1based": selected["mass_index_fortran_1based"],
                "patch_bounds_mass_grid": selected["patch_bounds_mass_grid"],
                "valid_time_utc": selected["valid_time_utc"],
            },
            "dynamic_attribution_selected_lead_h": dynamic_attribution["localization_manifest"]["selected_lead_h"],
        },
        "wrf_truth_surface": {
            "source_files": wrf_refresh["emitted_surfaces"]["post_after_all_rk_steps_pre_halo"]["files"],
            "source_file_info": [
                path_info(Path(path))
                for path in wrf_refresh["emitted_surfaces"]["post_after_all_rk_steps_pre_halo"]["files"]
            ],
            "counts": compact_counts(target_surface),
            "metadata": wrf_refresh["emitted_surfaces"]["post_after_all_rk_steps_pre_halo"]["metadata"],
            "green_reference_comparisons": {
                "vs_scratch_h10_wrfout": wrf_refresh["comparisons"]["post_after_all_rk_steps_pre_halo_vs_scratch_h10_wrfout"],
                "vs_provided_cpu_h10_wrfout": wrf_refresh["comparisons"]["post_after_all_rk_steps_pre_halo_vs_provided_cpu_h10_wrfout"],
            },
        },
        "jax_runtime_inspection": runtime,
        "wrapper_attempt": {
            "strategy": "CPU-only read-only wrapper inspection plus WRF truth parser",
            "jax_model_executed_steps": 0,
            "same_surface_candidate_emitted": False,
            "blocked_before_model_run": True,
            "reason_not_run": (
                "The public JAX forecast APIs return post-halo/post-guard state, not the requested "
                "post-after_all_rk_steps pre-RK-halo surface. A 6000-step CPU run through those APIs "
                "would spend CPU while still missing the acceptance surface."
            ),
        },
        "available_retained_jax_writer_diagnostic": {
            "path": str(RETAINED_JAX_H10_WRFOUT),
            "exists": RETAINED_JAX_H10_WRFOUT.exists(),
            "surface_class": "retained JAX/GPU wrfout history output, not CPU internal pre-halo state",
            "used_for_verdict": False,
            "comparison_vs_wrf_truth": retained_compare,
            "first_mismatch_vs_wrf_truth": retained_first,
        },
        "missing_wrapper_prerequisite": {
            "missing_api": (
                "A CPU-callable JAX hook returning OperationalCarry/State immediately after final RK3 "
                "_carry_from_finished_stage(..., namelist) and _refresh_grid_p_from_finished, before "
                "the apply_halo call in _acoustic_scan and before _rk_scan_step returns."
            ),
            "missing_state": (
                "No CPU JAX h10 d02 step-6000 same-state savepoint/checkpoint with full-column native "
                "u/v/w/theta/p/ph/mu split fields was present in the contract inputs or inspected scratch paths."
            ),
            "missing_logs": (
                "No CPU JAX wrapper log demonstrates emission of T/P/PB/U/V/W/PH/MU/MUB at the WRF target boundary."
            ),
            "next_command": (
                "Open a narrow source-changing/debug-hook sprint to expose a CPU-only pre-halo capture in "
                "src/gpuwrf/runtime/operational_mode.py around _acoustic_scan immediately before its "
                "apply_halo return, then rerun: JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src "
                "python proofs/v014/jax_after_all_rk_wrapper.py"
            ),
        },
        "acceptance_notes": {
            "compares_against_wrf_green_target": True,
            "jax_vs_jax_self_compare": False,
            "retained_wrfout_only_verdict": False,
            "production_src_edited": False,
            "wrf_source_edited": False,
            "gpu_launched": False,
        },
        "commands": {
            "minimum_contract_commands": [
                "python -m py_compile proofs/v014/jax_after_all_rk_wrapper.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/jax_after_all_rk_wrapper.py",
                "python -m json.tool proofs/v014/jax_after_all_rk_wrapper.json >/tmp/jax_after_all_rk_wrapper.validated.json",
            ],
            "generator_argv": sys.argv,
        },
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "review": str(OUT_REVIEW),
        },
        "unresolved_risks": [
            "The first same-surface JAX operator mismatch remains unknown until the pre-halo state is capturable.",
            "A retained JAX wrfout mismatch is real but cannot distinguish dycore cadence from halo/writer/history effects.",
            "CPU h10 forward execution cost is not measured here because the available API cannot produce the target surface.",
        ],
        "next_decision": "Open a narrow pre-halo JAX capture/source-hook sprint before any source fix, GPU probe, TOST, Switzerland, or FP32 work.",
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
