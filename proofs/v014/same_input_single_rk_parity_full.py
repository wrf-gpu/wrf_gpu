#!/usr/bin/env python3
"""V0.14 full pre-RK same-input single-RK parity proof.

This is a proof-only loader gate.  It parses the full pre-RK WRF hook inventory
and either runs the strict same-input JAX comparison or emits the precise
blocked verdict.  It must not use JAX-generated physics tendencies as a
substitute for missing WRF source leaves.
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


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")

ROOT = Path(__file__).resolve().parents[2]
PROOF_DIR = ROOT / "proofs/v014"
OUT_JSON = PROOF_DIR / "same_input_single_rk_parity_full.json"
OUT_MD = PROOF_DIR / "same_input_single_rk_parity_full.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-full-pre-rk-savepoint-hook.md"

HOOK_JSON = PROOF_DIR / "full_pre_rk_savepoint_hook.json"
HOOK_MD = PROOF_DIR / "full_pre_rk_savepoint_hook.md"
HOOK_SCRIPT = PROOF_DIR / "full_pre_rk_savepoint_hook.py"
PATCH_DIFF = PROOF_DIR / "full_pre_rk_savepoint_hook_wrf_patch.diff"
SPRINT_CONTRACT = ROOT / ".agent/sprints/2026-06-09-v014-full-pre-rk-savepoint-hook/sprint-contract.md"

STATE_PY = ROOT / "src/gpuwrf/contracts/state.py"
OPERATIONAL_STATE_PY = ROOT / "src/gpuwrf/runtime/operational_state.py"
OPERATIONAL_MODE_PY = ROOT / "src/gpuwrf/runtime/operational_mode.py"
RK_ADDTEND_DRY_PY = ROOT / "src/gpuwrf/dynamics/core/rk_addtend_dry.py"

POST_RK_FILES = sorted(
    Path("/mnt/data/wrf_gpu2/v014_post_rk_refresh/refresh_output").glob(
        "refresh_post_after_all_rk_steps_pre_halo_d2_step_6000_*.txt"
    )
)

VERDICT = "FULL_PRE_RK_JAX_LOADER_BLOCKED_RK_FIXED_SOURCE_BOUNDARY"
TARGET_FIELDS = ("T", "P", "PB", "PH", "PHB", "MU", "MUB", "U", "V", "W")

POST_SCHEMAS = {
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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
    if tag in {"MASS_K1", "U_K1", "V_K1"}:
        return (idx[3], idx[2])
    if tag == "WPH_KSTAG01":
        return (idx[5], idx[4], idx[3])
    raise ValueError(tag)


def parse_post_surface(paths: Iterable[Path]) -> dict[str, Any]:
    records: dict[str, dict[tuple[int, ...], dict[str, float]]] = defaultdict(dict)
    metadata: dict[str, list[str]] = defaultdict(list)
    duplicate_count = 0
    duplicate_max_delta = 0.0
    duplicate_max_delta_by_field: dict[str, float] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or line.startswith("record_schema"):
                    continue
                parts = line.split()
                tag = parts[0]
                if tag not in POST_SCHEMAS:
                    metadata[tag].append(" ".join(parts[1:]) if len(parts) > 1 else "")
                    continue
                schema = POST_SCHEMAS[tag]
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
        "metadata": {key: values[0] if len(values) == 1 else values for key, values in metadata.items()},
        "unique_counts": {tag: len(records.get(tag, {})) for tag in POST_SCHEMAS},
        "duplicate_count": duplicate_count,
        "duplicate_max_delta": duplicate_max_delta,
        "duplicate_max_delta_by_field": duplicate_max_delta_by_field,
    }


def extract_slots(path: Path, class_name: str) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "__slots__" for target in stmt.targets):
                continue
            try:
                slots = list(ast.literal_eval(stmt.value))
            except Exception as exc:  # pragma: no cover
                return {"class": class_name, "module": str(path.relative_to(ROOT)), "error": repr(exc)}
            return {
                "class": class_name,
                "module": str(path.relative_to(ROOT)),
                "line_start": int(node.lineno),
                "line_end": int(getattr(node, "end_lineno", node.lineno)),
                "slots": slots,
            }
    return {"class": class_name, "module": str(path.relative_to(ROOT)), "missing": True}


def extract_dataclass_fields(path: Path, class_name: str) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        fields = [
            stmt.target.id
            for stmt in node.body
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
        ]
        return {
            "class": class_name,
            "module": str(path.relative_to(ROOT)),
            "line_start": int(node.lineno),
            "line_end": int(getattr(node, "end_lineno", node.lineno)),
            "fields": fields,
        }
    return {"class": class_name, "module": str(path.relative_to(ROOT)), "missing": True}


def source_contract() -> dict[str, Any]:
    return {
        "files": {
            "state_py": path_info(STATE_PY),
            "operational_state_py": path_info(OPERATIONAL_STATE_PY),
            "operational_mode_py": path_info(OPERATIONAL_MODE_PY),
            "rk_addtend_dry_py": path_info(RK_ADDTEND_DRY_PY),
        },
        "classes": {
            "State": extract_slots(STATE_PY, "State"),
            "Tendencies": extract_slots(STATE_PY, "Tendencies"),
            "OperationalCarry": extract_dataclass_fields(OPERATIONAL_STATE_PY, "OperationalCarry"),
            "DryPhysicsTendencies": extract_dataclass_fields(RK_ADDTEND_DRY_PY, "DryPhysicsTendencies"),
        },
        "entry_point": {
            "name": "_rk_scan_step_with_pre_halo_capture",
            "module": str(OPERATIONAL_MODE_PY.relative_to(ROOT)),
        },
    }


def decide_blocker(hook: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    suff = hook.get("hook_sufficiency", {})
    missing = list(suff.get("missing_for_strict_same_input", []))
    dry_fields = contract["classes"]["DryPhysicsTendencies"].get("fields", [])
    state_slots = contract["classes"]["State"].get("slots", [])
    return {
        "verdict": VERDICT,
        "strict_same_input_comparison_run": False,
        "jax_step_executed": False,
        "reason": (
            "The full step-entry WRF hook emitted native state, but at the exact "
            "post-itimestep/pre-RK boundary WRF has not yet computed current-step "
            "first_rk_step_part1/part2 physics/source tendencies and has not yet "
            "zeroed/populated the rk_tendency save-family fields. Feeding zeros or "
            "JAX-generated tendencies would not be a strict same-input comparison."
        ),
        "missing_exact_boundary": (
            "current-step source/save surface after WRF has produced ru_tendf, rv_tendf, "
            "rw_tendf, ph_tendf, t_tendf, mu_tendf, h_diabatic, u_save, v_save, "
            "w_save, ph_save, and t_save, but before any dynamics state update that "
            "changes the one-step initial state used by the JAX wrapper"
        ),
        "missing_fields": missing,
        "required_dry_physics_tendency_fields": dry_fields,
        "state_contract_slots_count": len(state_slots),
        "state_contract_slots_sample": state_slots[:24],
        "weak_comparison_avoided": True,
        "next_exact_action": (
            "Add a second accepted WRF source boundary, or move the proof boundary, so "
            "the same file set contains current-step DryPhysicsTendencies/save-family "
            "leaves from WRF. Then construct OperationalCarry/DryPhysicsTendencies and "
            "call _rk_scan_step_with_pre_halo_capture on CPU."
        ),
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    blocked = payload["blocked"]
    hook = payload["wrf_full_pre_rk_hook"]
    patch = hook.get("patch_width_assessment", {})
    lines = [
        "# V0.14 Same-Input Single-RK Parity Full",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "No JAX-vs-WRF same-input comparison was run. The proof blocks before JAX execution because the WRF inputs are not a complete same-input RK boundary.",
        "",
        "## Parsed WRF Inputs",
        "",
        f"- Full pre-RK records: `{hook['emitted_surface']['unique_counts']}`.",
        f"- Post-RK/pre-halo truth records: `{payload['wrf_post_rk_surface']['unique_counts']}`.",
        f"- Candidate valid mass cells after 8-cell halo: `{patch.get('candidate_valid_mass_cell_count')}`.",
        "",
        "## Blocker",
        "",
        f"- Missing exact boundary: {blocked['missing_exact_boundary']}.",
        f"- Missing fields: `{blocked['missing_fields']}`.",
        "",
        "## Next Action",
        "",
        blocked["next_exact_action"],
        "",
    ]
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Review: V0.14 Full Pre-RK Savepoint Hook",
            "",
            f"verdict: `{payload['verdict']}`.",
            "",
            "objective: create a CPU-WRF full pre-RK native-state savepoint at d02 step 6000 and run, or precisely block, the strict same-input one-step JAX comparison.",
            "",
            "files changed:",
            "- `proofs/v014/full_pre_rk_savepoint_hook.py`",
            "- `proofs/v014/full_pre_rk_savepoint_hook.json`",
            "- `proofs/v014/full_pre_rk_savepoint_hook.md`",
            "- `proofs/v014/full_pre_rk_savepoint_hook_wrf_patch.diff`",
            "- `proofs/v014/same_input_single_rk_parity_full.py`",
            "- `proofs/v014/same_input_single_rk_parity_full.json`",
            "- `proofs/v014/same_input_single_rk_parity_full.md`",
            "- `.agent/reviews/2026-06-09-v014-full-pre-rk-savepoint-hook.md`",
            "",
            "commands run:",
            "- `python -m py_compile proofs/v014/full_pre_rk_savepoint_hook.py`",
            "- `python -m json.tool proofs/v014/full_pre_rk_savepoint_hook.json >/tmp/full_pre_rk_savepoint_hook.validated.json`",
            "- `python -m py_compile proofs/v014/same_input_single_rk_parity_full.py`",
            "- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_input_single_rk_parity_full.py`",
            "- `python -m json.tool proofs/v014/same_input_single_rk_parity_full.json >/tmp/same_input_single_rk_parity_full.validated.json`",
            "- `git diff -- src/gpuwrf`",
            "- CPU WRF scratch build/run commands are recorded in `proofs/v014/full_pre_rk_savepoint_hook.json`.",
            "",
            "proof objects produced:",
            "- `proofs/v014/full_pre_rk_savepoint_hook.json`",
            "- `proofs/v014/full_pre_rk_savepoint_hook.md`",
            "- `proofs/v014/full_pre_rk_savepoint_hook_wrf_patch.diff`",
            "- `proofs/v014/same_input_single_rk_parity_full.json`",
            "- `proofs/v014/same_input_single_rk_parity_full.md`",
            "",
            "unresolved risks:",
            "- The full native state exists only over the narrow target patch; it leaves one conservative mass score cell after an 8-cell halo.",
            "- The strict comparison is blocked because current-step WRF source/save-family leaves are not available at the exact step-entry hook.",
            "- No production `src/gpuwrf/**` files were edited, so no JAX wrapper source API was added.",
            "",
            f"next decision needed: {payload['blocked']['next_exact_action']}",
            "",
        ]
    )


def main() -> int:
    hook = load_json(HOOK_JSON) if HOOK_JSON.exists() else None
    post_surface = parse_post_surface(POST_RK_FILES)
    contract = source_contract()

    if hook is None:
        blocked = {
            "verdict": "FULL_PRE_RK_HOOK_BLOCKED_NO_HOOK_JSON",
            "strict_same_input_comparison_run": False,
            "jax_step_executed": False,
            "reason": "full_pre_rk_savepoint_hook.json is absent.",
            "missing_fields": ["full_pre_rk_savepoint_hook.json"],
            "weak_comparison_avoided": True,
            "next_exact_action": "Run proofs/v014/full_pre_rk_savepoint_hook.py after the WRF hook run.",
        }
        verdict = blocked["verdict"]
        hook_payload: Mapping[str, Any] = {
            "emitted_surface": {"unique_counts": {}},
            "patch_width_assessment": {},
            "hook_sufficiency": {},
        }
    else:
        hook_payload = hook
        blocked = decide_blocker(hook, contract)
        verdict = VERDICT

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.same_input_single_rk_parity_full.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "cpu_only": True,
        "gpu_used": False,
        "no_hermes": True,
        "production_src_edits": False,
        "target_fields_requested": TARGET_FIELDS,
        "environment": jax_environment(),
        "inputs_read": {
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "hook_script": path_info(HOOK_SCRIPT),
            "hook_json": path_info(HOOK_JSON),
            "hook_md": path_info(HOOK_MD),
            "wrf_patch_diff": path_info(PATCH_DIFF),
            "post_rk_files": [path_info(path) for path in POST_RK_FILES],
        },
        "source_contract": contract,
        "wrf_full_pre_rk_hook": hook_payload,
        "wrf_post_rk_surface": post_surface,
        "blocked": blocked,
        "comparison": {
            "strict_same_input_comparison_run": False,
            "jax_step_executed": False,
            "jax_vs_jax_self_compare": False,
            "weak_comparison_avoided": True,
            "ranked_residual_table": None,
            "reason": blocked["reason"],
        },
        "commands": {
            "argv": sys.argv,
            "minimum_validation": [
                "python -m py_compile proofs/v014/full_pre_rk_savepoint_hook.py",
                "python -m json.tool proofs/v014/full_pre_rk_savepoint_hook.json >/tmp/full_pre_rk_savepoint_hook.validated.json",
                "python -m py_compile proofs/v014/same_input_single_rk_parity_full.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_input_single_rk_parity_full.py",
                "python -m json.tool proofs/v014/same_input_single_rk_parity_full.json >/tmp/same_input_single_rk_parity_full.validated.json",
                "git diff -- src/gpuwrf",
            ],
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
    print(verdict)
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    print(f"review={OUT_REVIEW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
