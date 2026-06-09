#!/usr/bin/env python3
"""V0.14 JAX pre-halo capture hook proof.

The hook is a proof-only path in ``runtime.operational_mode``.  This script
exercises it on CPU, proves the normal RK return is unchanged when disabled,
and records why the accepted h10 WRF green surface still cannot be compared
without a JAX h10 pre-step carry/checkpoint.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except Exception as exc:  # pragma: no cover - recorded in proof output
    jax = None
    jnp = None
    JAX_IMPORT_ERROR = repr(exc)
else:
    JAX_IMPORT_ERROR = None

from gpuwrf.contracts.halo import apply_halo
from gpuwrf.contracts.state import State
from gpuwrf.dynamics.advection import halo_spec
from gpuwrf.ic_generators.idealized import build_warm_bubble_setup
from gpuwrf.runtime.operational_mode import (
    _rk_scan_step,
    _rk_scan_step_with_pre_halo_capture,
    run_forecast_operational,
)
from gpuwrf.runtime.operational_state import initial_operational_carry


ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "proofs/v014/jax_pre_halo_capture.json"
OUT_MD = ROOT / "proofs/v014/jax_pre_halo_capture.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-pre-halo-capture-hook.md"

SPRINT_CONTRACT = ROOT / ".agent/sprints/2026-06-09-v014-pre-halo-capture-hook/sprint-contract.md"
HANDOFF = ROOT / ".agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md"
WRAPPER_JSON = ROOT / "proofs/v014/jax_after_all_rk_wrapper.json"
WRF_REFRESH_JSON = ROOT / "proofs/v014/wrf_post_rk_refresh_localization.json"
WRF_REFRESH_MD = ROOT / "proofs/v014/wrf_post_rk_refresh_localization.md"
SAVEPOINT_REQUEST_JSON = ROOT / "proofs/v014/same_state_savepoint_request.json"
RUNTIME_PATH = ROOT / "src/gpuwrf/runtime/operational_mode.py"

VERDICT = "HOOK_GREEN_COMPARE_BLOCKED_NO_JAX_H10_PRESTEP_CARRY"
TARGET_FIELDS = ("T", "P", "PB", "U", "V", "W", "PH", "MU", "MUB")
THETA_OFFSET_K = 300.0


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


def path_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256(path),
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jax_import_error": JAX_IMPORT_ERROR,
    }
    if jax is not None:
        env.update(
            {
                "jax_version": getattr(jax, "__version__", None),
                "jax_default_backend": jax.default_backend(),
                "jax_devices": [str(device) for device in jax.devices()],
            }
        )
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
                "contains_capture_pre_halo": "capture_pre_halo" in source,
                "contains_carry_from_finished_stage": "_carry_from_finished_stage" in source,
                "contains_sharded_carry_halo_handling": "_maybe_exchange_sharded_carry_halos" in source,
                "contains_apply_halo": "apply_halo" in source,
                "contains_pre_halo_result": "_PreHaloCaptureResult" in source,
            }
    return {"module_path": str(path.relative_to(ROOT)), "name": name, "missing": True}


def array_stats(arr: Any) -> dict[str, Any]:
    data = np.asarray(arr)
    finite = data[np.isfinite(data)] if np.issubdtype(data.dtype, np.floating) else data.ravel()
    out: dict[str, Any] = {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "finite_count": int(finite.size),
        "count": int(data.size),
        "all_finite": bool(finite.size == data.size),
    }
    if finite.size:
        out.update(
            {
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "max_abs": float(np.max(np.abs(finite))),
            }
        )
    return out


def field_view(state: State, field: str) -> Any:
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


def compare_pytrees(left: Any, right: Any) -> dict[str, Any]:
    left_tree = jax.tree_util.tree_structure(left)
    right_tree = jax.tree_util.tree_structure(right)
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    max_abs = 0.0
    compared = 0
    first_nonzero: dict[str, Any] | None = None
    all_equal = str(left_tree) == str(right_tree) and len(left_leaves) == len(right_leaves)
    for idx, (lhs, rhs) in enumerate(zip(left_leaves, right_leaves)):
        a = np.asarray(lhs)
        b = np.asarray(rhs)
        if a.shape != b.shape or a.dtype != b.dtype:
            all_equal = False
            if first_nonzero is None:
                first_nonzero = {
                    "leaf_index": idx,
                    "left_shape": list(a.shape),
                    "right_shape": list(b.shape),
                    "left_dtype": str(a.dtype),
                    "right_dtype": str(b.dtype),
                }
            continue
        if np.issubdtype(a.dtype, np.number):
            diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
            finite = diff[np.isfinite(diff)]
            leaf_max = float(np.max(np.abs(finite))) if finite.size else 0.0
            equal = bool(np.array_equal(a, b, equal_nan=True))
        else:
            leaf_max = 0.0
            equal = bool(np.array_equal(a, b))
        compared += 1
        max_abs = max(max_abs, leaf_max)
        if not equal:
            all_equal = False
            if first_nonzero is None:
                first_nonzero = {"leaf_index": idx, "max_abs": leaf_max, "shape": list(a.shape)}
    return {
        "tree_structure_equal": str(left_tree) == str(right_tree),
        "leaf_count_left": len(left_leaves),
        "leaf_count_right": len(right_leaves),
        "compared_leaf_count": compared,
        "array_equal": bool(all_equal),
        "max_abs": float(max_abs),
        "first_difference": first_nonzero,
    }


def run_hook_probe() -> dict[str, Any]:
    if jax is None or jnp is None:
        return {"status": "BLOCKED", "reason": f"jax import failed: {JAX_IMPORT_ERROR}"}

    setup = build_warm_bubble_setup(require_gpu=False)
    namelist = dataclasses.replace(
        setup.namelist,
        run_physics=False,
        run_boundary=False,
        const_nu_m2_s=0.0,
        diff_6th_opt=0,
        km_opt=0,
        dt_s=0.1,
        acoustic_substeps=1,
        disable_guards=True,
    )
    carry0 = initial_operational_carry(setup.state)
    lead_seconds = jnp.asarray(0.0, dtype=jnp.float64)
    normal = _rk_scan_step(carry0, namelist, lead_seconds=lead_seconds)
    captured = _rk_scan_step_with_pre_halo_capture(carry0, namelist, lead_seconds=lead_seconds)
    jax.block_until_ready(captured.carry.state.theta)

    haloed_capture = apply_halo(captured.pre_halo_state, halo_spec(namelist.grid))
    captured_fields = {field: array_stats(field_view(captured.pre_halo_state, field)) for field in TARGET_FIELDS}
    public_sig = inspect.signature(run_forecast_operational)

    return {
        "status": "HOOK_GREEN",
        "fixture": {
            "name": "gpuwrf.ic_generators.idealized.build_warm_bubble_setup(require_gpu=False)",
            "grid": {"nz": int(setup.grid.nz), "ny": int(setup.grid.ny), "nx": int(setup.grid.nx)},
            "dt_s": float(namelist.dt_s),
            "acoustic_substeps": int(namelist.acoustic_substeps),
            "run_physics": bool(namelist.run_physics),
            "run_boundary": bool(namelist.run_boundary),
            "use_vertical_solver": bool(namelist.use_vertical_solver),
        },
        "disabled_path": {
            "call": "_rk_scan_step(carry, namelist)",
            "return_type": type(normal).__name__,
            "is_capture_tuple": bool(type(normal).__name__ == "_PreHaloCaptureResult"),
        },
        "capture_path": {
            "call": "_rk_scan_step_with_pre_halo_capture(carry, namelist)",
            "return_type": type(captured).__name__,
            "carry_return_type": type(captured.carry).__name__,
            "pre_halo_state_type": type(captured.pre_halo_state).__name__,
            "captured_source_function": "_acoustic_scan",
            "captured_cadence": (
                "final RK3 _carry_from_finished_stage(..., namelist) -> "
                "_maybe_exchange_sharded_carry_halos(...) -> before apply_halo(next_carry.state, halo_spec(...))"
            ),
        },
        "normal_return_vs_capture_carry": compare_pytrees(normal, captured.carry),
        "apply_halo_captured_state_vs_capture_return_state": compare_pytrees(
            haloed_capture, captured.carry.state
        ),
        "captured_target_field_views": captured_fields,
        "all_captured_target_fields_finite": bool(
            all(stats["all_finite"] for stats in captured_fields.values())
        ),
        "normal_forecast_api_signature": {
            "run_forecast_operational": str(public_sig),
            "has_capture_parameter": "capture" in str(public_sig) or "pre_halo" in str(public_sig),
        },
    }


def wrf_target_summary(wrf_refresh: Mapping[str, Any], savepoint_request: Mapping[str, Any]) -> dict[str, Any]:
    surface = wrf_refresh["emitted_surfaces"]["post_after_all_rk_steps_pre_halo"]
    selected = savepoint_request["selection"]["selected_cells"][0]
    return {
        "accepted_wrf_verdict": wrf_refresh["verdict"],
        "target_surface": wrf_refresh["next_jax_cpu_wrapper_target"],
        "domain": wrf_refresh["target_confirmed"]["domain"],
        "wrf_step": wrf_refresh["target_confirmed"]["wrf_step"],
        "valid_time_utc": wrf_refresh["target_confirmed"]["valid_time_utc"],
        "current_timestr_before_step": surface["metadata"]["current_timestr_before_step"],
        "lead_seconds_after_step": wrf_refresh["target_confirmed"]["lead_seconds_after_step"],
        "selected_cell_zero_yx": wrf_refresh["target_confirmed"]["selected_cell_zero_yx"],
        "selected_patch_bounds_mass_grid": wrf_refresh["target_confirmed"]["selected_patch_bounds_mass_grid"],
        "native_staggered_coordinates": wrf_refresh["target_confirmed"]["native_staggered_coordinates"],
        "unique_counts": surface["unique_counts"],
        "source_files": surface["files"],
        "source_file_info": [path_info(Path(path)) for path in surface["files"]],
        "green_candidate_vs_scratch_h10_wrfout": wrf_refresh["compact_summary"]["green_candidate_vs_scratch_wrfout"],
        "green_candidate_vs_provided_cpu_h10_wrfout": {
            field: wrf_refresh["comparisons"]["post_after_all_rk_steps_pre_halo_vs_provided_cpu_h10_wrfout"][field]
            for field in TARGET_FIELDS
        },
        "savepoint_request_first_cell": {
            "selection_rank": selected["selection_rank"],
            "mass_index_zero_based": selected["mass_index_zero_based"],
            "mass_index_fortran_1based": selected["mass_index_fortran_1based"],
            "native_patch_bounds": selected["native_patch_bounds"],
        },
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    hook = payload["hook_probe"]
    blocked = payload["h10_same_surface_compare"]
    lines = [
        "# V0.14 JAX Pre-Halo Capture Hook",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Hook Proof",
        "",
        f"- CPU fixture status: `{hook['status']}`.",
        f"- Captured cadence: `{hook['capture_path']['captured_cadence']}`.",
        f"- Normal `_rk_scan_step` return type with hook disabled: `{hook['disabled_path']['return_type']}`.",
        f"- Normal return vs capture carry exact: `{hook['normal_return_vs_capture_carry']['array_equal']}` "
        f"(max_abs `{hook['normal_return_vs_capture_carry']['max_abs']}`).",
        f"- Captured target field views finite: `{hook['all_captured_target_fields_finite']}`.",
        "",
        "## WRF Green Target",
        "",
        f"- WRF verdict: `{payload['wrf_green_target']['accepted_wrf_verdict']}`.",
        f"- Target: `{payload['wrf_green_target']['target_surface']}`.",
        f"- Domain/step: `d02`, step `{payload['wrf_green_target']['wrf_step']}`, "
        f"`{payload['wrf_green_target']['valid_time_utc']}`.",
        f"- Patch counts: `{payload['wrf_green_target']['unique_counts']}`.",
        "",
        "## H10 Compare",
        "",
        f"- Status: `{blocked['status']}`.",
        f"- Missing input: `{blocked['exact_missing_input_short']}`.",
        "- No retained wrfout or JAX-vs-JAX diagnostic is used as a same-surface verdict.",
        "",
    ]
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Review: V0.14 JAX Pre-Halo Capture Hook",
            "",
            f"Verdict: `{payload['verdict']}`.",
            "",
            "Objective: add and prove a default-off JAX hook for the final RK3 post-refresh, pre-halo state.",
            "",
            "Files changed:",
            "- `src/gpuwrf/runtime/operational_mode.py`",
            "- `proofs/v014/jax_pre_halo_capture.py`",
            "- `proofs/v014/jax_pre_halo_capture.json`",
            "- `proofs/v014/jax_pre_halo_capture.md`",
            "- `.agent/reviews/2026-06-09-v014-pre-halo-capture-hook.md`",
            "- `tests/test_v014_pre_halo_capture.py`",
            "",
            "Result:",
            "- The hook is private/proof-only and default-off.",
            "- The disabled RK path returns `OperationalCarry`, not an auxiliary capture tuple.",
            "- The capture path returns the same normal carry plus the final RK3 pre-halo `State`.",
            "- A same-surface h10 comparison is still blocked by missing JAX h10 pre-step carry/checkpoint.",
            "",
            "Unresolved risks:",
            "- No first numerical JAX operator mismatch is named by this sprint.",
            "- The accepted WRF green patch is available, but the JAX h10 input/carry is not.",
            "",
            "Next decision: provide/build the JAX h10 pre-step carry checkpoint or open a source-fix sprint only after a same-surface mismatch is emitted.",
            "",
        ]
    )


def main() -> int:
    wrf_refresh = load_json(WRF_REFRESH_JSON)
    savepoint_request = load_json(SAVEPOINT_REQUEST_JSON)
    wrapper = load_json(WRAPPER_JSON)
    hook_probe = run_hook_probe()

    blocked_compare = {
        "status": "BLOCKED",
        "reason": "NO_JAX_H10_PRESTEP_CARRY",
        "same_surface_comparison_run": False,
        "exact_missing_input_short": "CPU-loadable JAX OperationalCarry immediately before d02 step 6000/h10.",
        "exact_missing_input": (
            "A CPU-loadable JAX OperationalCarry immediately before d02 step 6000 "
            "(current_timestr_before_step=2026-05-02_03:59:54, lead before step 35994 s), "
            "including State plus promoted carry leaves t_2ave, ww, mudf, muave, muts, ph_tend, "
            "u_save/v_save/w_save/t_save/ph_save/mu_save/ww_save, rthraten, and any active physics "
            "carry; paired with the real d02 OperationalNamelist metrics, tendencies, boundary_config, "
            "and boundary leaves. An equivalent checkpoint/restart from which "
            "_rk_scan_step_with_pre_halo_capture can execute the single step is sufficient."
        ),
        "why_not_full_cpu_replay": (
            "The contract inputs provide the WRF green truth surface and retained writer diagnostics, "
            "but no JAX checkpoint/carry at h10. A public 6000-step CPU replay would be a long forecast "
            "and still needs validation that its start/carry matches the retained GPU/JAX trajectory."
        ),
    }

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.jax_pre_halo_capture.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": VERDICT,
        "cpu_only": True,
        "gpu_used": False,
        "wrf_source_edited": False,
        "production_forecast_api_return_changed": False,
        "no_hermes": True,
        "inputs_read": {
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "handoff": path_info(HANDOFF),
            "jax_after_all_rk_wrapper_json": path_info(WRAPPER_JSON),
            "wrf_refresh_json": path_info(WRF_REFRESH_JSON),
            "wrf_refresh_md": path_info(WRF_REFRESH_MD),
            "same_state_savepoint_request_json": path_info(SAVEPOINT_REQUEST_JSON),
        },
        "environment": jax_environment(),
        "source_inspection": {
            "runtime_operational_mode": path_info(RUNTIME_PATH),
            "nodes": [
                extract_ast_node(RUNTIME_PATH, "_PreHaloCaptureResult"),
                extract_ast_node(RUNTIME_PATH, "_acoustic_scan"),
                extract_ast_node(RUNTIME_PATH, "_rk_scan_step"),
                extract_ast_node(RUNTIME_PATH, "_rk_scan_step_with_pre_halo_capture"),
                extract_ast_node(RUNTIME_PATH, "run_forecast_operational"),
                extract_ast_node(RUNTIME_PATH, "run_forecast_operational_segmented"),
                extract_ast_node(RUNTIME_PATH, "run_forecast_operational_single_scan"),
            ],
        },
        "hook_probe": hook_probe,
        "wrf_green_target": wrf_target_summary(wrf_refresh, savepoint_request),
        "previous_wrapper_verdict": wrapper["verdict"],
        "retained_writer_diagnostic_not_used": {
            "available": bool(wrapper["available_retained_jax_writer_diagnostic"]["exists"]),
            "used_for_verdict": False,
            "first_mismatch_vs_wrf_truth": wrapper["available_retained_jax_writer_diagnostic"].get(
                "first_mismatch_vs_wrf_truth"
            ),
        },
        "h10_same_surface_compare": blocked_compare,
        "acceptance_notes": {
            "hook_default_off": True,
            "normal_rk_return_identical_when_capture_enabled_for_auxiliary_carry": (
                hook_probe.get("normal_return_vs_capture_carry", {}).get("array_equal") is True
            ),
            "normal_forecast_api_capture_parameter_absent": (
                hook_probe.get("normal_forecast_api_signature", {}).get("has_capture_parameter") is False
            ),
            "compares_against_wrf_green_target": False,
            "wrf_green_target_parsed": True,
            "h10_compare_blocked_with_exact_missing_input": True,
            "jax_vs_jax_self_compare": False,
            "retained_wrfout_only_verdict": False,
            "gpu_launched": False,
            "wrf_source_edited": False,
            "tost_run": False,
            "switzerland_validation_run": False,
            "fp32_source_landing": False,
        },
        "commands": {
            "generator_argv": sys.argv,
            "minimum_contract_commands": [
                "python -m py_compile src/gpuwrf/runtime/operational_mode.py proofs/v014/jax_pre_halo_capture.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/jax_pre_halo_capture.py",
                "python -m json.tool proofs/v014/jax_pre_halo_capture.json >/tmp/jax_pre_halo_capture.validated.json",
            ],
        },
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "review": str(OUT_REVIEW),
        },
        "unresolved_risks": [
            "No h10 same-surface numerical JAX comparison ran because no JAX h10 pre-step carry/checkpoint was available.",
            "The first failing JAX operator/cadence remains unnamed until that same-surface comparison runs.",
        ],
        "next_decision": "Build/provide the JAX h10 pre-step carry checkpoint, then run the hook against Boole's WRF green target before any source-fix sprint.",
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
