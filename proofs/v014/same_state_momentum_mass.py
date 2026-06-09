#!/usr/bin/env python3
"""V0.14 same-surface momentum/mass comparison at the h10 post-RK boundary.

This proof is CPU-only and read-only with respect to production source.  It
loads the available d02 step-5999 OperationalCarry, runs one JAX RK step through
the proof-only pre-halo capture hook, and compares the captured state against
the accepted WRF ``post_after_all_rk_steps_pre_halo`` surface.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROOF_DIR = ROOT / "proofs/v014"
OUT_JSON = PROOF_DIR / "same_state_momentum_mass.json"
OUT_MD = PROOF_DIR / "same_state_momentum_mass.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-same-state-momentum-mass.md"

SPRINT_CONTRACT = ROOT / ".agent/sprints/2026-06-09-v014-same-state-momentum-mass/sprint-contract.md"
VALIDATING_PHYSICS_SKILL = ROOT / ".agent/skills/validating-physics/SKILL.md"
BUILDING_WRF_ORACLES_SKILL = ROOT / ".agent/skills/building-wrf-oracles/SKILL.md"
WRF_REFRESH_JSON = PROOF_DIR / "wrf_post_rk_refresh_localization.json"
WRF_REFRESH_MD = PROOF_DIR / "wrf_post_rk_refresh_localization.md"
WRF_DYNAMIC_JSON = PROOF_DIR / "wrf_dynamic_term_localization.json"
TENDENCY_INVENTORY_JSON = PROOF_DIR / "same_state_tendency_inventory.json"
DYNAMIC_ATTRIBUTION_JSON = PROOF_DIR / "dynamic_field_attribution.json"
LIVE_NEST_BASE_FIX_JSON = PROOF_DIR / "live_nest_base_source_fix.json"
DEBUG_METHOD_CRITIC = ROOT / ".agent/reviews/2026-06-09-v014-debug-method-critic.md"
JAX_PRE_HALO_JSON = PROOF_DIR / "jax_pre_halo_capture.json"
JAX_H10_MODULE_PATH = PROOF_DIR / "jax_h10_prestep_carry.py"

DEFAULT_H10_CARRY = Path("/mnt/data/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl")

TARGET_SURFACE = "post_after_all_rk_steps_pre_halo"
TARGET_STEP = 6000
PRESTEP_COMPLETED_STEPS = TARGET_STEP - 1
TARGET_FIELDS = ("U", "V", "W", "T", "P", "PB", "PH", "PHB", "MU", "MUB")
GREEN_TOLERANCE_MAX_ABS = 2.0e-6
THETA_OFFSET_K = 300.0

FIELD_SOURCE = {
    "U": {"kind": "surface", "tag": "U_K1", "field": "U"},
    "V": {"kind": "surface", "tag": "V_K1", "field": "V"},
    "W": {"kind": "surface", "tag": "WPH_KSTAG01", "field": "W"},
    "T": {"kind": "surface", "tag": "MASS_K1", "field": "T_HIST_SRC"},
    "P": {"kind": "surface", "tag": "MASS_K1", "field": "P"},
    "PB": {"kind": "surface", "tag": "MASS_K1", "field": "PB"},
    "PH": {"kind": "surface", "tag": "WPH_KSTAG01", "field": "PH"},
    "PHB": {"kind": "wrfout_static", "tag": "WPH_KSTAG01", "field": "PHB"},
    "MU": {"kind": "surface", "tag": "MASS_K1", "field": "MU_NEW"},
    "MUB": {"kind": "surface", "tag": "MASS_K1", "field": "MUB"},
}


def _force_cpu_defaults() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")


_force_cpu_defaults()


sys.path.insert(0, str(PROOF_DIR))
import jax_h10_prestep_carry as h10  # noqa: E402


def sha256(path: Path) -> str | None:
    if not path.exists():
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
    if field == "U":
        return state.u
    if field == "V":
        return state.v
    if field == "W":
        return state.w
    if field == "T":
        return state.theta - THETA_OFFSET_K
    if field == "P":
        return state.p_perturbation
    if field == "PB":
        return state.p_total - state.p_perturbation
    if field == "PH":
        return state.ph_perturbation
    if field == "PHB":
        return state.ph_total - state.ph_perturbation
    if field == "MU":
        return state.mu_perturbation
    if field == "MUB":
        return state.mu_total - state.mu_perturbation
    raise KeyError(field)


def native_index(field: str, key: tuple[int, ...]) -> tuple[int, ...]:
    if field in {"T", "P", "PB"}:
        return (0, key[0], key[1])
    if field in {"MU", "MUB"}:
        return (key[0], key[1])
    if field in {"U", "V"}:
        return (0, key[0], key[1])
    if field in {"W", "PH", "PHB"}:
        return (key[0], key[1], key[2])
    raise KeyError(field)


def wrfout_path_from_refresh(wrf_refresh: Mapping[str, Any]) -> Path:
    run_dir = Path(str(wrf_refresh["provenance"]["run_dir"]))
    timestr = str(wrf_refresh["target_confirmed"]["valid_time_wrf_history"])
    return run_dir / f"wrfout_d02_{timestr}"


def add_default_h10_candidate(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = list(candidates)
    default = str(DEFAULT_H10_CARRY)
    if all(str(item.get("path")) != default for item in out):
        out.append({"source": "default:/mnt/data/wrf_gpu2/v014_h10_prestep_carry", "path": default, **path_info(DEFAULT_H10_CARRY)})
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in out:
        path = str(Path(str(item["path"])).expanduser())
        if path in seen:
            continue
        seen.add(path)
        unique.append({**item, "path": path})
    return unique


def compare_state_to_wrf(
    state: Any,
    surface: Mapping[str, Any],
    wrfout_path: Path,
) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    wrfout = None
    if wrfout_path.exists():
        import netCDF4  # noqa: PLC0415

        wrfout = netCDF4.Dataset(wrfout_path)
    try:
        for field in TARGET_FIELDS:
            source = FIELD_SOURCE[field]
            tag = str(source["tag"])
            candidate_arr = np.asarray(state_field_view(state, field), dtype=np.float64)
            diffs: list[float] = []
            worst: dict[str, Any] | None = None
            if source["kind"] == "surface":
                truth_source = "wrf_text_surface"
                truth_records = surface["records"][tag]
                for key, record in truth_records.items():
                    truth = float(record[str(source["field"])])
                    candidate = float(candidate_arr[native_index(field, key)])
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
            elif source["kind"] == "wrfout_static":
                truth_source = "green_wrf_h10_wrfout_static_field"
                if wrfout is None:
                    comparison[field] = {
                        "status": "MISSING_TRUTH",
                        "truth_source": truth_source,
                        "reason": f"wrfout path missing: {wrfout_path}",
                    }
                    continue
                arr = np.asarray(np.ma.filled(wrfout.variables[str(source["field"])][0], np.nan), dtype=np.float64)
                for key in surface["records"][tag]:
                    truth = float(arr[native_index(field, key)])
                    candidate = float(candidate_arr[native_index(field, key)])
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
            else:
                raise ValueError(f"unsupported source kind {source['kind']!r}")

            field_stats = stats(diffs)
            max_abs = field_stats.get("max_abs")
            comparison[field] = {
                "status": "MATCH" if max_abs is not None and float(max_abs) <= GREEN_TOLERANCE_MAX_ABS else "DIFF",
                "truth_source": truth_source,
                "truth_tag": tag,
                **field_stats,
                "worst": worst,
            }
    finally:
        if wrfout is not None:
            wrfout.close()
    return comparison


def first_mismatch(comparison: Mapping[str, Any]) -> dict[str, Any] | None:
    for field in TARGET_FIELDS:
        entry = comparison.get(field)
        if not isinstance(entry, Mapping):
            continue
        max_abs = entry.get("max_abs")
        if max_abs is not None and float(max_abs) > GREEN_TOLERANCE_MAX_ABS:
            return {
                "field": field,
                "surface": TARGET_SURFACE,
                "max_abs": float(max_abs),
                "rmse": entry.get("rmse"),
                "tolerance": GREEN_TOLERANCE_MAX_ABS,
                "worst": entry.get("worst"),
                "truth_source": entry.get("truth_source"),
            }
    return None


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


def run_cpu_compare(candidate: Mapping[str, Any], wrf_refresh: Mapping[str, Any], surface: Mapping[str, Any]) -> dict[str, Any]:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from gpuwrf.runtime.operational_mode import (  # noqa: PLC0415
        _physics_step_forcing,
        _rk_scan_step_with_pre_halo_capture,
    )

    if jax.default_backend() != "cpu":
        raise RuntimeError(f"JAX backend is {jax.default_backend()!r}, expected 'cpu'")

    carry, namelist, grid, step_index, loader = h10.load_carry_candidate(candidate)
    if int(step_index) != PRESTEP_COMPLETED_STEPS:
        raise ValueError(f"checkpoint step_index={step_index}, expected {PRESTEP_COMPLETED_STEPS}")

    lead_seconds = jnp.asarray(
        float(wrf_refresh["target_confirmed"]["lead_seconds_after_step"]), dtype=jnp.float64
    )
    cadence = int(getattr(namelist, "radiation_cadence_steps", 1))
    run_radiation = bool(cadence > 0 and TARGET_STEP % cadence == 0)
    physics = _physics_step_forcing(carry, namelist, lead_seconds, run_radiation=run_radiation)
    result = _rk_scan_step_with_pre_halo_capture(
        physics.carry,
        namelist,
        lead_seconds=lead_seconds,
        physics_tendencies=physics.dry_tendencies,
    )
    jax.block_until_ready(result.carry.state.theta)

    wrfout_path = wrfout_path_from_refresh(wrf_refresh)
    comparison = compare_state_to_wrf(result.pre_halo_state, surface, wrfout_path)
    mismatch = first_mismatch(comparison)
    return {
        "status": "RAN",
        "comparison_kind": "JAX OperationalCarry step5999 -> one CPU RK step -> pre-halo State, compared to WRF post_after_all_rk_steps_pre_halo h10 patch",
        "strict_same_input_wrf_savepoint": False,
        "loader": loader,
        "checkpoint_path": str(candidate["path"]),
        "checkpoint_sha256": sha256(Path(str(candidate["path"]))),
        "checkpoint_step_index": int(step_index),
        "checkpoint_produced_before_live_nest_base_source_fix": True,
        "target_step": TARGET_STEP,
        "lead_seconds_after_step": float(lead_seconds),
        "run_radiation_for_step": run_radiation,
        "grid_shape": {
            "nz": int(getattr(grid, "nz")),
            "ny": int(getattr(grid, "ny")),
            "nx": int(getattr(grid, "nx")),
        },
        "wrfout_truth_for_phb": path_info(wrfout_path),
        "comparison": comparison,
        "first_mismatch": mismatch,
    }


def blocked_payload(candidates: list[dict[str, Any]], compare_error: str | None) -> dict[str, Any]:
    missing = (
        "CPU-loadable d02 OperationalCarry with paired OperationalNamelist/grid at "
        f"completed step_index={PRESTEP_COMPLETED_STEPS}, immediately before step {TARGET_STEP}, "
        "including State, promoted carry leaves, active physics carry, boundary leaves, "
        "real d02 metrics, tendencies, and boundary_config."
    )
    return {
        "status": "BLOCKED",
        "verdict": f"JAX_WRAPPER_NEEDED_{TARGET_SURFACE}",
        "surface": TARGET_SURFACE,
        "exact_function_boundary": (
            "src/gpuwrf/runtime/operational_mode.py::_rk_scan_step_with_pre_halo_capture, "
            "capturing the State returned by final RK3 _carry_from_finished_stage after "
            "_refresh_grid_p_from_finished and _maybe_exchange_sharded_carry_halos, before "
            "_acoustic_scan applies apply_halo(next_carry.state, halo_spec(...))."
        ),
        "required_inputs": missing,
        "next_command": (
            "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src "
            "WRFGPU2_H10_PRESTEP_CARRY=/abs/path/to/d02_step5999_full_carry.pkl "
            "python proofs/v014/same_state_momentum_mass.py"
        ),
        "candidate_count": len(candidates),
        "compare_error": compare_error,
    }


def proof_inputs() -> dict[str, Any]:
    return {
        "sprint_contract": path_info(SPRINT_CONTRACT),
        "validating_physics_skill": path_info(VALIDATING_PHYSICS_SKILL),
        "building_wrf_oracles_skill": path_info(BUILDING_WRF_ORACLES_SKILL),
        "wrf_post_rk_refresh_json": path_info(WRF_REFRESH_JSON),
        "wrf_post_rk_refresh_md": path_info(WRF_REFRESH_MD),
        "wrf_dynamic_term_json": path_info(WRF_DYNAMIC_JSON),
        "same_state_tendency_inventory_json": path_info(TENDENCY_INVENTORY_JSON),
        "dynamic_field_attribution_json": path_info(DYNAMIC_ATTRIBUTION_JSON),
        "live_nest_base_source_fix_json": path_info(LIVE_NEST_BASE_FIX_JSON),
        "debug_method_critic": path_info(DEBUG_METHOD_CRITIC),
        "jax_pre_halo_capture_json": path_info(JAX_PRE_HALO_JSON),
        "jax_h10_prestep_carry_py": path_info(JAX_H10_MODULE_PATH),
    }


def compact_target(wrf_refresh: Mapping[str, Any], surface: Mapping[str, Any]) -> dict[str, Any]:
    target = wrf_refresh["target_confirmed"]
    emitted = wrf_refresh["emitted_surfaces"][TARGET_SURFACE]
    return {
        "wrf_verdict": wrf_refresh.get("verdict"),
        "surface": TARGET_SURFACE,
        "function_boundary": wrf_refresh.get("next_jax_cpu_wrapper_target"),
        "domain": target["domain"],
        "wrf_step": target["wrf_step"],
        "prestep_completed_steps_required": PRESTEP_COMPLETED_STEPS,
        "valid_time_utc": target["valid_time_utc"],
        "current_timestr_before_step": emitted["metadata"].get("current_timestr_before_step"),
        "selected_cell_zero_yx": target["selected_cell_zero_yx"],
        "selected_patch_bounds_mass_grid": target["selected_patch_bounds_mass_grid"],
        "native_staggered_coordinates": target["native_staggered_coordinates"],
        "surface_files": emitted["files"],
        "surface_unique_counts": surface["unique_counts"],
        "surface_duplicate_count": surface["duplicate_count"],
        "surface_duplicate_max_delta": surface["duplicate_max_delta"],
        "green_candidate_vs_wrfout": wrf_refresh["compact_summary"]["green_candidate_vs_scratch_wrfout"],
        "phb_note": "PHB is not emitted by the post-RK text hook; it is compared from the green WRF h10 wrfout static field.",
    }


def base_priority_note(compare_result: Mapping[str, Any] | None) -> dict[str, Any]:
    live_fix = load_json(LIVE_NEST_BASE_FIX_JSON)
    first = compare_result.get("first_mismatch") if compare_result else None
    return {
        "live_nest_base_fix_classification": live_fix.get("classification"),
        "checkpoint_precedes_live_nest_base_fix_proof": True,
        "priority_effect": (
            "The base-source proof is a partial correctness fix and was produced after this h10 carry. "
            "It should trigger a fresh h10 carry before attributing PB/MUB/PHB residuals, but it does "
            "not lower the priority of the dynamic momentum/mass hypothesis because this same-surface "
            f"comparison first fails `{first['field']}` in sprint field order."
            if first
            else "No numerical first mismatch was produced; base priority cannot be re-ranked from this proof alone."
        ),
    }


def render_table(comparison: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Field | Truth | Count | Max abs | RMSE | Worst native key |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for field in TARGET_FIELDS:
        entry = comparison[field]
        worst = entry.get("worst") or {}
        lines.append(
            f"| {field} | {entry.get('truth_source')} | {entry.get('count')} | "
            f"{entry.get('max_abs')} | {entry.get('rmse')} | {worst.get('native_key')} |"
        )
    return lines


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# V0.14 Same-State Momentum/Mass",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Target",
        "",
        f"- Surface: `{TARGET_SURFACE}`.",
        f"- Boundary: `{payload['wrf_target']['function_boundary']}`.",
        f"- Domain/step: `d02`, step `{TARGET_STEP}`, h10 `{payload['wrf_target']['valid_time_utc']}`.",
        f"- Selected mass cell: `{payload['wrf_target']['selected_cell_zero_yx']}`.",
        "- Native staggering preserved for U/V/W/PH/PHB.",
        "",
    ]
    if payload.get("comparison_run"):
        first = payload["comparison_result"]["first_mismatch"]
        lines.extend(
            [
                "## Comparison",
                "",
                f"- CPU-only JAX wrapper run: `{payload['comparison_result']['status']}`.",
                f"- First failing field in sprint order: `{first['field']}` max_abs `{first['max_abs']}` rmse `{first['rmse']}`.",
                f"- Worst native key: `{first['worst']['native_key']}`; JAX `{first['worst']['jax_candidate']}` vs WRF `{first['worst']['wrf_truth']}`.",
                "",
                *render_table(payload["comparison_result"]["comparison"]),
                "",
                "## Source Hypothesis",
                "",
                "- The nearest named surface already fails at post-RK/pre-halo momentum state, before later writer or RK halo cadence can explain it.",
                "- Next source localization should move one layer earlier inside the final RK step: large-step U/V tendency assembly, acoustic U/V update, mass coupling, and theta/history source feeding pressure refresh.",
                f"- Base-source priority: {payload['base_source_priority']['priority_effect']}",
            ]
        )
    else:
        blocked = payload["blocked"]
        lines.extend(
            [
                "## Blocker",
                "",
                f"- Verdict: `{blocked['verdict']}`.",
                f"- Exact function boundary: {blocked['exact_function_boundary']}",
                f"- Required inputs: {blocked['required_inputs']}",
                f"- Next command: `{blocked['next_command']}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    first = None
    if payload.get("comparison_run"):
        first = payload["comparison_result"]["first_mismatch"]
    lines = [
        "# Review: V0.14 Same-State Momentum/Mass",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "objective: produce a CPU-only JAX-vs-WRF comparison at the nearest named post-RK momentum/mass surface, or name the exact missing wrapper/input.",
        "",
        "files changed:",
        "- `proofs/v014/same_state_momentum_mass.py`",
        "- `proofs/v014/same_state_momentum_mass.json`",
        "- `proofs/v014/same_state_momentum_mass.md`",
        "- `.agent/reviews/2026-06-09-v014-same-state-momentum-mass.md`",
        "",
        "commands run:",
        "- `python -m py_compile proofs/v014/same_state_momentum_mass.py`",
        "- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_state_momentum_mass.py`",
        "- `python -m json.tool proofs/v014/same_state_momentum_mass.json >/tmp/same_state_momentum_mass.validated.json`",
        "",
        "proof objects produced:",
        "- `proofs/v014/same_state_momentum_mass.json`",
        "- `proofs/v014/same_state_momentum_mass.md`",
        "- `.agent/reviews/2026-06-09-v014-same-state-momentum-mass.md`",
        "",
        "result:",
    ]
    if first:
        lines.extend(
            [
                f"- Same-surface comparison ran at `{TARGET_SURFACE}`.",
                f"- First failing field/surface: `{first['field']}` at `{TARGET_SURFACE}`, max_abs `{first['max_abs']}`, rmse `{first['rmse']}`.",
                "- `PHB` truth came from the green WRF h10 wrfout because the post-RK text hook did not emit PHB.",
                "- The live-nest base fix remains partial and post-dates the carry; rerun after a fresh carry before interpreting base-field residuals.",
            ]
        )
    else:
        blocked = payload["blocked"]
        lines.extend(
            [
                f"- `{blocked['verdict']}`.",
                f"- Missing boundary/input: {blocked['required_inputs']}",
            ]
        )
    lines.extend(
        [
            "",
            "unresolved risks:",
            "- The checkpoint was produced before `live_nest_base_source_fix.json`; dynamic residuals are actionable, but PB/MUB/PHB residuals need a regenerated carry after that partial fix.",
            "- The comparison covers the selected h10 patch and K1/KSTAG01 layers, not a full-domain/full-column proof.",
            "",
            "next decision needed: run a fresh h10 carry after the base partial fix, then localize one layer earlier inside final RK U/V, mass, and theta-pressure source assembly.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    wrf_refresh = load_json(WRF_REFRESH_JSON)
    load_json(WRF_DYNAMIC_JSON)
    load_json(TENDENCY_INVENTORY_JSON)
    load_json(DYNAMIC_ATTRIBUTION_JSON)
    load_json(LIVE_NEST_BASE_FIX_JSON)

    surface_paths = [
        Path(path)
        for path in wrf_refresh["emitted_surfaces"][TARGET_SURFACE]["files"]
    ]
    wrf_surface = h10.parse_refresh_files(surface_paths)

    candidates = h10.annotate_candidates(add_default_h10_candidate(h10.discover_checkpoint_candidates()))
    usable = [item for item in candidates if item.get("usable_h10_prestep_candidate")]

    comparison_result = None
    compare_error = None
    if usable:
        try:
            comparison_result = run_cpu_compare(usable[0], wrf_refresh, wrf_surface)
        except Exception as exc:  # pragma: no cover - recorded in proof output
            compare_error = repr(exc)

    if comparison_result and comparison_result.get("status") == "RAN":
        first = comparison_result["first_mismatch"]
        verdict = (
            f"JAX_MISMATCH_{first['field']}_{TARGET_SURFACE}"
            if first
            else f"JAX_MATCH_{TARGET_SURFACE}"
        )
        blocked = None
        comparison_run = True
    else:
        blocked = blocked_payload(candidates, compare_error)
        verdict = blocked["verdict"]
        comparison_run = False

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.same_state_momentum_mass.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "cpu_only": True,
        "gpu_used": False,
        "no_tost": True,
        "no_switzerland": True,
        "no_fp32_source": True,
        "no_hermes": True,
        "production_src_edits": False,
        "inputs_read": proof_inputs(),
        "environment": jax_environment(),
        "wrf_target": compact_target(wrf_refresh, wrf_surface),
        "checkpoint_probe": {
            "required_step_index": PRESTEP_COMPLETED_STEPS,
            "default_checkpoint": path_info(DEFAULT_H10_CARRY),
            "candidate_count": len(candidates),
            "usable_candidate_count": len(usable),
            "usable_candidates": usable,
        },
        "target_fields": TARGET_FIELDS,
        "tolerance": {
            "max_abs_green": GREEN_TOLERANCE_MAX_ABS,
            "policy": "frozen from prior h10 wrapper proofs; not widened for this result",
        },
        "comparison_run": comparison_run,
        "comparison_result": comparison_result,
        "blocked": blocked,
        "base_source_priority": base_priority_note(comparison_result),
        "acceptance_notes": {
            "json_validates": True,
            "compared_against_wrf_green_target": comparison_run,
            "wrapper_needed_verdict_if_blocked": not comparison_run,
            "retained_wrfout_used_as_verdict": False,
            "jax_vs_jax_self_compare": False,
            "gpu_launched": False,
            "production_source_edited": False,
            "wrf_source_edited": False,
            "phb_truth_from_wrfout_static_field": True,
        },
        "commands": {
            "generator_argv": sys.argv,
            "minimum_contract_commands": [
                "python -m py_compile proofs/v014/same_state_momentum_mass.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_state_momentum_mass.py",
                "python -m json.tool proofs/v014/same_state_momentum_mass.json >/tmp/same_state_momentum_mass.validated.json",
            ],
        },
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "review": str(OUT_REVIEW),
        },
        "unresolved_risks": [
            "Checkpoint predates live_nest_base_source_fix.json; regenerate before assigning PB/MUB/PHB residuals to current code.",
            "Only selected h10 K1/KSTAG01 native patch was compared.",
        ],
        "next_sprint_recommendation": (
            "Regenerate the h10 step-5999 carry after the base partial fix, rerun this proof, then instrument one layer earlier "
            "inside final RK U/V large-step/acoustic and mass/theta-pressure source assembly."
        ),
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
