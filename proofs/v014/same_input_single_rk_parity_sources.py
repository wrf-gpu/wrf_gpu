#!/usr/bin/env python3
"""V0.14 same-input single-RK proof gate using WRF source/save leaves.

This script is fail-closed by design.  It only runs JAX if the WRF hook output
is sufficient to construct the same-boundary ``State``, ``OperationalCarry``,
and ``DryPhysicsTendencies`` without mixing step-entry state with later WRF
source leaves or borrowing non-WRF-emitted state from an existing checkpoint.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")

ROOT = Path(__file__).resolve().parents[2]
PROOF_DIR = ROOT / "proofs/v014"
OUT_JSON = PROOF_DIR / "same_input_single_rk_parity_sources.json"
OUT_MD = PROOF_DIR / "same_input_single_rk_parity_sources.md"

HOOK_JSON = PROOF_DIR / "source_save_boundary_hook.json"
HOOK_MD = PROOF_DIR / "source_save_boundary_hook.md"
PATCH_DIFF = PROOF_DIR / "source_save_boundary_hook_wrf_patch.diff"
SPRINT_CONTRACT = ROOT / ".agent/sprints/2026-06-09-v014-source-save-boundary/sprint-contract.md"

STATE_PY = ROOT / "src/gpuwrf/contracts/state.py"
OPERATIONAL_STATE_PY = ROOT / "src/gpuwrf/runtime/operational_state.py"
OPERATIONAL_MODE_PY = ROOT / "src/gpuwrf/runtime/operational_mode.py"
RK_ADDTEND_DRY_PY = ROOT / "src/gpuwrf/dynamics/core/rk_addtend_dry.py"

RUN_DIR = Path("<DATA_ROOT>/wrf_gpu2/v014_source_save_boundary/run_case3")
NAMELIST = RUN_DIR / "namelist.input"
PRESTEP_CARRY = Path("<DATA_ROOT>/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl")
POST_RK_TRUTH = Path("<DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/refresh_output")

TARGET_STEP = 6000
TARGET_FIELDS = ("T", "P", "PB", "PH", "PHB", "MU", "MUB", "U", "V", "W")
READY_NO_WRAPPER_VERDICT = (
    "SOURCE_SAVE_BOUNDARY_READY_NO_JAX_WRAPPER_FULL_DOMAIN_PATCH_AND_SCALAR_OLD_LIMITER"
)


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
    except Exception as exc:  # pragma: no cover
        env.update({"jax_import_error": repr(exc), "gpu_device_count": None})
    return env


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


def parse_namelist_options(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    text = path.read_text(errors="replace")
    wanted = [
        "time_step",
        "parent_time_step_ratio",
        "rk_ord",
        "moist_adv_opt",
        "scalar_adv_opt",
        "spec_bdy_width",
        "diff_opt",
        "km_opt",
        "mp_physics",
        "bl_pbl_physics",
        "sf_surface_physics",
        "cu_physics",
    ]
    out: dict[str, Any] = {"exists": True}
    for name in wanted:
        match = re.search(rf"^\s*{re.escape(name)}\s*=\s*([^,\n/]+(?:,[^/\n]+)?)", text, re.MULTILINE)
        if match:
            raw = match.group(1).strip().rstrip(",")
            values: list[Any] = []
            for part in raw.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    values.append(int(part))
                except ValueError:
                    try:
                        values.append(float(part))
                    except ValueError:
                        values.append(part.strip("'\""))
            out[name] = values if len(values) != 1 else values[0]
    return out


def decide(hook: Mapping[str, Any]) -> dict[str, Any]:
    suff = hook.get("hook_sufficiency", {})
    emitted = hook.get("emitted_surface", {})
    missing_dry = list(suff.get("missing", []))
    unique_counts = emitted.get("unique_counts", {})
    if not emitted.get("files"):
        verdict = "SOURCE_SAVE_HOOK_BLOCKED_NO_OUTPUT"
        reason = "source/save hook files are absent"
    elif missing_dry:
        verdict = "SOURCE_SAVE_HOOK_BLOCKED_MISSING_" + "_".join(missing_dry)
        reason = "source/save hook did not emit every DryPhysicsTendencies leaf"
    else:
        verdict = READY_NO_WRAPPER_VERDICT
        reason = (
            "WRF source/save leaves are present at a consistent pre-mutation boundary, "
            "but no strict JAX wrapper can construct a full same-boundary OperationalCarry "
            "from WRF-emitted fields only."
        )
    return {
        "verdict": verdict,
        "strict_same_input_comparison_run": False,
        "jax_step_executed": False,
        "reason": reason,
        "source_save_boundary_ready": verdict == READY_NO_WRAPPER_VERDICT,
        "unique_counts": unique_counts,
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    blocker = payload["blocked"]
    ready = bool(payload["decision"]["source_save_boundary_ready"])
    comparison_note = (
        "No strict same-input JAX comparison was run. The WRF source/save boundary is consistent, "
        "but the current proof cannot build the required JAX input contract from WRF-emitted fields only."
        if ready
        else "No strict same-input JAX comparison was run because the WRF source/save hook did not pass the prerequisite gate."
    )
    lines = [
        "# V0.14 Same-Input Single-RK Parity Sources",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        comparison_note,
        "",
        "## Boundary",
        "",
        f"- Source/save boundary ready: `{payload['decision']['source_save_boundary_ready']}`.",
        f"- Hook records: `{payload['decision']['unique_counts']}`.",
        f"- Accepted ordering: `{payload['accepted_boundary']['ordering']}`.",
        "",
        "## Blocker",
        "",
        f"- Missing wrapper: {blocker['missing_wrapper']}",
        f"- Missing field surface: {blocker['missing_field_surface']}",
        f"- Patch/truth limitation: {blocker['patch_and_truth_limitation']}",
        f"- Existing JAX checkpoint used: `{blocker['existing_checkpoint_used']}`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    hook = load_json(HOOK_JSON) if HOOK_JSON.exists() else {}
    contract = source_contract()
    decision = decide(hook)
    namelist = parse_namelist_options(NAMELIST)
    dry_fields = contract["classes"]["DryPhysicsTendencies"].get("fields", [])
    carry_fields = contract["classes"]["OperationalCarry"].get("fields", [])

    blocked = {
        "missing_wrapper": (
            "proof-only loader/wrapper that maps WRF-emitted full-domain source_save_after_rk_tendency "
            "records into State, OperationalCarry, OperationalNamelist/GridSpec/DycoreMetrics, and "
            "DryPhysicsTendencies, then calls _rk_scan_step_with_pre_halo_capture"
        ),
        "missing_field_surface": (
            "full-domain same-boundary promoted carry leaves are not WRF-emitted: "
            "t_2ave, ww, mudf, muave, muts, ph_tend, mu_save, ww_save, rthraten, active physics carry, "
            "and boundary leaves; source hook emits the dry source/save family only"
        ),
        "patch_and_truth_limitation": (
            "source hook is a 17x17 patch with one 8-cell-halo-valid mass cell, while the existing "
            "post-RK/pre-halo truth emits only K1 mass/U/V and kstag 0/1 W/PH records, not a full "
            "44-level full-domain State output"
        ),
        "scalar_limiter_blocker": (
            "namelist has moist_adv_opt=1 and scalar_adv_opt=1; WRF initializes only P_QV moist_old "
            "before this boundary, and scalar_old is not valid here. A full final-stage scalar-limiter "
            "comparison needs a consistent old-field strategy or a narrower dry-only wrapper."
        ),
        "existing_checkpoint": path_info(PRESTEP_CARRY),
        "existing_checkpoint_used": False,
        "existing_checkpoint_not_used_reason": (
            "It is a JAX-produced step5999 OperationalCarry, not WRF-emitted state at the accepted "
            "source/save boundary; mixing it with WRF source leaves would violate the same-input rule."
        ),
    }

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.same_input_single_rk_parity_sources.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": decision["verdict"],
        "target": {"domain": "d02", "wrf_step": TARGET_STEP, "fields": TARGET_FIELDS},
        "cpu_only": True,
        "gpu_used": False,
        "no_hermes": True,
        "production_src_edits": False,
        "environment": jax_environment(),
        "decision": decision,
        "accepted_boundary": {
            "name": "source_save_after_rk_tendency",
            "ordering": (
                "after first_rk_step_part1/part2 and rk_tendency; before relax_bdy_dry, "
                "rk_addtend_dry, spec_bdy_dry, small_step_prep, advance_uv"
            ),
            "native_state_and_sources_same_boundary": True,
            "ordering_conflict": False,
        },
        "blocked": blocked,
        "source_contract": contract,
        "required_dry_physics_tendency_fields": dry_fields,
        "operational_carry_fields": carry_fields,
        "namelist_options": namelist,
        "inputs": {
            "source_save_hook_json": path_info(HOOK_JSON),
            "source_save_hook_md": path_info(HOOK_MD),
            "source_save_patch_diff": path_info(PATCH_DIFF),
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "post_rk_truth_dir": path_info(POST_RK_TRUTH),
            "namelist": path_info(NAMELIST),
        },
        "comparison": {
            "strict_same_input_comparison_run": False,
            "jax_step_executed": False,
            "per_field_metrics": {},
            "ranked_residuals": [],
            "weak_comparison_avoided": True,
        },
        "commands": {
            "validation": [
                "python -m py_compile proofs/v014/same_input_single_rk_parity_sources.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_input_single_rk_parity_sources.py",
                "python -m json.tool proofs/v014/same_input_single_rk_parity_sources.json >/tmp/same_input_single_rk_parity_sources.validated.json",
            ]
        },
        "proof_objects": {"json": str(OUT_JSON), "markdown": str(OUT_MD)},
    }

    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(payload["verdict"])
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
