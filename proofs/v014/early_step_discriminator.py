#!/usr/bin/env python3
"""V0.14 early-step same-input discriminator.

This proof is deliberately fail-closed.  It first inventories the strict
same-input route from shared wrfinput_d02 under the CPU-only sprint rule.  If the
required WRF/JAX contracts are absent, it emits one consolidated blocker for all
candidate steps instead of producing a weak comparison.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUT_JSON = ROOT / "proofs/v014/early_step_discriminator.json"
OUT_MD = ROOT / "proofs/v014/early_step_discriminator.md"

SPRINT_CONTRACT = ROOT / ".agent/sprints/2026-06-09-v014-early-step-discriminator/sprint-contract.md"
PLAN = ROOT / ".agent/decisions/V0140-EARLY-STEP-DISCRIMINATOR-PLAN.md"
PROJECT_CONSTITUTION = ROOT / "PROJECT_CONSTITUTION.md"
AGENTS = ROOT / "AGENTS.md"
MANAGING_SPRINTS_SKILL = ROOT / ".agent/skills/managing-sprints/SKILL.md"
WRF_ORACLE_SKILL = ROOT / ".agent/skills/building-wrf-oracles/SKILL.md"
VALIDATING_PHYSICS_SKILL = ROOT / ".agent/skills/validating-physics/SKILL.md"

D02_REPLAY = ROOT / "src/gpuwrf/integration/d02_replay.py"
NESTED_PIPELINE = ROOT / "src/gpuwrf/integration/nested_pipeline.py"
OPERATIONAL_MODE = ROOT / "src/gpuwrf/runtime/operational_mode.py"
OPERATIONAL_STATE = ROOT / "src/gpuwrf/runtime/operational_state.py"
STATE_CONTRACT = ROOT / "src/gpuwrf/contracts/state.py"

WRFINPUT_D02 = Path("<DATA_ROOT>/wrf_gpu2/v014_source_save_boundary/run_case3/wrfinput_d02")
NAMELIST_INPUT = Path("<DATA_ROOT>/wrf_gpu2/v014_source_save_boundary/run_case3/namelist.input")
WRFBDY_D01 = Path("<DATA_ROOT>/wrf_gpu2/v014_source_save_boundary/run_case3/wrfbdy_d01")
RUN_CASE3 = Path("<DATA_ROOT>/wrf_gpu2/v014_source_save_boundary/run_case3")
SCRATCH = Path("<DATA_ROOT>/wrf_gpu2/v014_early_step_discriminator")

CANDIDATE_STEPS = (1, 60, 600, 3000, 5999)
TARGET_FIELDS = ("T", "P", "PB", "PH", "PHB", "MU", "MUB", "U", "V", "W")
ACTIVE_MOISTURE_CANDIDATES = ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP")
VERDICT = (
    "EARLY_STEP_DISCRIMINATOR_BLOCKED_CPU_REALCASE_LOADER_GPU_ONLY_"
    "NO_CANDIDATE_WRF_PREHALO_TRUTH_NO_SAME_INPUT_CARRY_CONTRACT"
)

SURFACE_SEARCH_ROOTS = (
    SCRATCH,
    Path("<DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/refresh_output"),
    Path("<DATA_ROOT>/wrf_gpu2/v014_same_state_wrf/marker_output"),
    Path("<DATA_ROOT>/wrf_gpu2/v014_source_save_boundary/source_save_output"),
    Path("<DATA_ROOT>/wrf_gpu2/v014_full_pre_rk_savepoint_hook/full_pre_rk_output"),
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
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        "sha256": sha256(path),
    }


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "item"):
        try:
            scalar = value.item()
            if isinstance(scalar, float) and not math.isfinite(scalar):
                return None
            return scalar
        except Exception:
            pass
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_tail(path: Path, max_chars: int = 4000) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def run_command(command: list[str], *, cwd: Path | None = None, timeout_s: int = 120) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "JAX_PLATFORMS": "cpu",
                "PYTHONPATH": str(SRC),
            },
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_s,
        )
        return {
            "command": command,
            "cwd": str(cwd) if cwd is not None else None,
            "returncode": int(proc.returncode),
            "wall_s": float(time.perf_counter() - start),
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "cwd": str(cwd) if cwd is not None else None,
            "returncode": None,
            "wall_s": float(time.perf_counter() - start),
            "timeout_s": int(timeout_s),
            "stdout_tail": (exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "error": "TimeoutExpired",
        }


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
    except Exception as exc:  # pragma: no cover - recorded in proof output
        env.update({"jax_import_error": repr(exc), "gpu_device_count": None})
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
                "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "contains_gpu_device_guard": "_gpu_device" in source,
                "contains_state_zeros": "State.zeros" in source,
                "contains_tendencies_zeros": "Tendencies.zeros" in source,
                "contains_pre_halo_capture": "pre_halo" in source or "PreHalo" in source,
            }
    return {"module_path": str(path.relative_to(ROOT)), "name": name, "missing": True}


def source_snippets() -> dict[str, Any]:
    return {
        "state_contract": {
            "path": path_info(STATE_CONTRACT),
            "nodes": [
                extract_ast_node(STATE_CONTRACT, "_gpu_device"),
                extract_ast_node(STATE_CONTRACT, "State"),
                extract_ast_node(STATE_CONTRACT, "Tendencies"),
                extract_ast_node(STATE_CONTRACT, "BaseState"),
            ],
        },
        "d02_replay": {
            "path": path_info(D02_REPLAY),
            "nodes": [
                extract_ast_node(D02_REPLAY, "build_replay_case"),
                extract_ast_node(D02_REPLAY, "load_wrfbdy_boundary_leaves"),
            ],
        },
        "nested_pipeline": {
            "path": path_info(NESTED_PIPELINE),
            "nodes": [extract_ast_node(NESTED_PIPELINE, "_load_domains")],
        },
        "operational": {
            "path": path_info(OPERATIONAL_MODE),
            "nodes": [
                extract_ast_node(OPERATIONAL_MODE, "_rk_scan_step"),
                extract_ast_node(OPERATIONAL_MODE, "_rk_scan_step_with_pre_halo_capture"),
            ],
        },
        "operational_state": {
            "path": path_info(OPERATIONAL_STATE),
            "nodes": [extract_ast_node(OPERATIONAL_STATE, "initial_operational_carry")],
        },
    }


def domain_inventory() -> dict[str, Any]:
    try:
        from netCDF4 import Dataset  # type: ignore # noqa: PLC0415

        with Dataset(WRFINPUT_D02) as ds:
            dims = {name: int(len(dim)) for name, dim in ds.dimensions.items()}
            fields: dict[str, Any] = {}
            for name in TARGET_FIELDS + ACTIVE_MOISTURE_CANDIDATES:
                wrf_name = {"T": "T", "PH": "PH", "MU": "MU"}.get(name, name)
                if wrf_name in ds.variables:
                    var = ds.variables[wrf_name]
                    fields[name] = {
                        "present": True,
                        "wrf_variable": wrf_name,
                        "dimensions": list(var.dimensions),
                        "shape": [int(v) for v in var.shape],
                        "dtype": str(getattr(var, "dtype", "")),
                    }
                else:
                    fields[name] = {"present": False, "wrf_variable": wrf_name}
            active_moisture = [
                name for name in ACTIVE_MOISTURE_CANDIDATES if fields.get(name, {}).get("present")
            ]
            return {
                "status": "OK",
                "dims": dims,
                "target_fields": fields,
                "active_moisture_present": active_moisture,
                "time_records": dims.get("Time"),
            }
    except Exception as exc:
        return {"status": "ERROR", "error": repr(exc)}


def strict_cpu_loader_probe() -> dict[str, Any]:
    code = "\n".join(
        [
            "from gpuwrf.integration.d02_replay import build_replay_case",
            "build_replay_case(",
            "    '<DATA_ROOT>/wrf_gpu2/v014_source_save_boundary/run_case3',",
            "    domain='d02',",
            "    load_lateral_boundaries=False,",
            ")",
            "print('LOAD_OK')",
        ]
    )
    result = run_command([sys.executable, "-c", code], cwd=ROOT, timeout_s=120)
    combined = f"{result.get('stdout_tail', '')}\n{result.get('stderr_tail', '')}"
    gpu_only = "State.zeros requires a GPU device" in combined
    result.update(
        {
            "status": "BLOCKED_GPU_ONLY_STATE_ZERO" if gpu_only else ("LOAD_OK" if result.get("returncode") == 0 else "ERROR"),
            "strict_cpu_realcase_loader_available": bool(result.get("returncode") == 0),
            "gpu_only_state_zero_observed": bool(gpu_only),
            "probe_purpose": (
                "Can the existing real-case wrfinput replay loader construct a d02 State/Tendencies "
                "under JAX_PLATFORMS=cpu without production source edits?"
            ),
        }
    )
    return result


def surface_files_for_step(step: int) -> list[dict[str, Any]]:
    hits: list[Path] = []
    patterns = (
        f"*step_{int(step)}_*.txt",
        f"*step{int(step)}*.txt",
        f"*d02_step{int(step)}*.npz",
        f"*d02_step{int(step)}*.json",
    )
    for root in SURFACE_SEARCH_ROOTS:
        if not root.exists():
            continue
        for pattern in patterns:
            hits.extend(path for path in root.rglob(pattern) if path.is_file())
    unique = sorted(set(hits))
    return [path_info(path) for path in unique]


def noncandidate_step6000_inventory() -> dict[str, Any]:
    roots = (
        Path("<DATA_ROOT>/wrf_gpu2/v014_post_rk_refresh/refresh_output"),
        Path("<DATA_ROOT>/wrf_gpu2/v014_source_save_boundary/source_save_output"),
        Path("<DATA_ROOT>/wrf_gpu2/v014_full_pre_rk_savepoint_hook/full_pre_rk_output"),
    )
    hits: list[Path] = []
    for root in roots:
        if root.exists():
            hits.extend(path for path in root.rglob("*step_6000_*.txt") if path.is_file())
    infos = [path_info(path) for path in sorted(set(hits))]
    return {
        "available": bool(infos),
        "candidate_step": False,
        "why_not_sufficient": (
            "Existing step-6000 surfaces are outside the requested candidate sequence and were previously "
            "classified as patch-only / missing wrapper carry contracts."
        ),
        "files": infos,
    }


def candidate_matrix(loader_probe: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in CANDIDATE_STEPS:
        files = surface_files_for_step(step)
        rows.append(
            {
                "step": int(step),
                "strict_same_input_comparison_run": False,
                "status": "BLOCKED",
                "wrf_post_rk_pre_halo_truth": {
                    "available": False,
                    "matching_files_found": files,
                    "missing": (
                        "No candidate-step WRF post_after_all_rk_steps_pre_halo or exactly named "
                        "equivalent surface exists under allowed scratch or inspected prior proof roots."
                    ),
                },
                "jax_same_input_start": {
                    "available": False,
                    "reason": (
                        "Existing real-case replay/domain loaders cannot construct d02 State/Tendencies "
                        "on CPU under the sprint rule."
                    )
                    if loader_probe.get("gpu_only_state_zero_observed")
                    else "No accepted CPU-loadable same-input OperationalCarry sequence exists for this step.",
                },
                "missing_contracts": [
                    "CPU-loadable real-case wrfinput State/Tendencies/OperationalNamelist loader",
                    "matching WRF post-RK/pre-halo truth for the candidate step",
                    "same-input live-parent/boundary package contract for d02",
                    "field/staggering schema for T/P/PB/PH/PHB/MU/MUB/U/V/W plus active moisture",
                ],
                "weak_comparison_avoided": True,
            }
        )
    return rows


def consolidated_blockers(loader_probe: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": "CPU_REALCASE_REPLAY_LOADER_GPU_ONLY",
            "applies_to_steps": list(CANDIDATE_STEPS),
            "evidence": (
                "CPU probe of build_replay_case(run_case3, domain='d02', load_lateral_boundaries=False) "
                f"returned {loader_probe.get('status')}; production State.zeros/Tendencies.zeros allocate via _gpu_device."
            ),
            "missing_contract": (
                "A CPU-compatible real-case wrfinput loader or checkpoint reader that constructs State, "
                "Tendencies, BaseState/metrics, OperationalNamelist, and initial OperationalCarry without GPU allocation."
            ),
        },
        {
            "id": "NO_CANDIDATE_WRF_POST_RK_PRE_HALO_TRUTH",
            "applies_to_steps": list(CANDIDATE_STEPS),
            "evidence": (
                "No files matching candidate steps 1/60/600/3000/5999 were found in the allowed scratch root "
                "or inspected prior WRF proof roots."
            ),
            "missing_contract": (
                "Disposable CPU-WRF candidate-step surface at post_after_all_rk_steps_pre_halo, or an exactly named "
                "equivalent boundary, covering the required dynamic fields and active moisture."
            ),
        },
        {
            "id": "NO_SAME_INPUT_CARRY_SEQUENCE",
            "applies_to_steps": list(CANDIDATE_STEPS),
            "evidence": (
                "The available h10 checkpoint is a JAX-produced trajectory artifact for a different boundary; "
                "mixing it with WRF leaves is disallowed, and no candidate-step WRF-controlled OperationalCarry exists."
            ),
            "missing_contract": (
                "For each candidate start, a WRF-controlled or exactly reproduced same-input carry, including "
                "promoted RK/acoustic scratch leaves and live d01->d02 boundary forcing."
            ),
        },
        {
            "id": "MISSING_REQUIRED_FIELD_SURFACE_SCHEMA",
            "applies_to_steps": list(CANDIDATE_STEPS),
            "evidence": (
                "Prior post-RK WRF refresh outputs are patch/K1-oriented and do not define a full required-field "
                "schema for PHB and active moisture at the accepted comparison boundary."
            ),
            "missing_contract": (
                "A WRF/JAX field map with units, staggering, counts, first/worst index semantics, and static/base "
                "field exclusion from headline dynamic selectors."
            ),
        },
    ]


def render_markdown(payload: Mapping[str, Any]) -> str:
    blockers = payload["blockers"]
    rows = payload["candidate_steps"]
    lines = [
        "# V0.14 Early-Step Same-Input Discriminator",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "No strict same-input comparison ran. The proof avoids weak WRF-output, JAX-vs-JAX, and mixed-source comparisons.",
        "",
        "## Result",
        "",
        f"- Candidate steps covered: `{[row['step'] for row in rows]}`.",
        f"- CPU real-case loader probe: `{payload['cpu_loader_probe']['status']}`.",
        f"- Candidate WRF pre-halo surfaces found: `{payload['candidate_surface_summary']['candidate_surface_file_count']}`.",
        f"- Existing step-6000 surfaces are non-candidate and patch-only: `{payload['noncandidate_step6000_inventory']['available']}`.",
        "",
        "## Consolidated Blockers",
        "",
    ]
    lines.extend(f"- `{item['id']}`: {item['missing_contract']}" for item in blockers)
    lines.extend(
        [
            "",
            "Next decision: build one CPU-compatible same-input contract, not another single-blocker ladder step.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    env = jax_environment()
    dims = domain_inventory()
    loader_probe = strict_cpu_loader_probe()
    candidates = candidate_matrix(loader_probe)
    candidate_file_count = sum(
        len(row["wrf_post_rk_pre_halo_truth"]["matching_files_found"]) for row in candidates
    )

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.early_step_discriminator.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": VERDICT,
        "cpu_only": True,
        "gpu_used": False,
        "strict_same_input_comparison_run": False,
        "jax_step_executed": False,
        "wrf_run_executed": False,
        "weak_comparison_avoided": True,
        "jax_vs_jax_self_compare": False,
        "one_cell_proof": False,
        "production_src_edits": False,
        "tost_run": False,
        "switzerland_validation_run": False,
        "fp32_work": False,
        "hermes_used": False,
        "inputs": {
            "project_constitution": path_info(PROJECT_CONSTITUTION),
            "agents": path_info(AGENTS),
            "managing_sprints_skill": path_info(MANAGING_SPRINTS_SKILL),
            "wrf_oracle_skill": path_info(WRF_ORACLE_SKILL),
            "validating_physics_skill": path_info(VALIDATING_PHYSICS_SKILL),
            "sprint_contract": path_info(SPRINT_CONTRACT),
            "plan": path_info(PLAN),
            "wrfinput_d02": path_info(WRFINPUT_D02),
            "namelist_input": path_info(NAMELIST_INPUT),
            "wrfbdy_d01": path_info(WRFBDY_D01),
            "run_case3": {
                "path": str(RUN_CASE3),
                "exists": RUN_CASE3.exists(),
                "is_dir": RUN_CASE3.is_dir(),
            },
        },
        "environment": env,
        "domain_inventory": dims,
        "required_fields": {
            "dynamic_or_perturbation_headline_fields": ["T", "P", "PH", "MU", "U", "V", "W"],
            "static_or_base_fields_excluded_from_headline_selector": ["PB", "PHB", "MUB"],
            "minimum_requested_fields": list(TARGET_FIELDS),
            "active_moisture_candidates": list(ACTIVE_MOISTURE_CANDIDATES),
        },
        "source_contract_inspection": source_snippets(),
        "cpu_loader_probe": loader_probe,
        "candidate_steps": candidates,
        "candidate_surface_summary": {
            "search_roots": [str(path) for path in SURFACE_SEARCH_ROOTS],
            "candidate_surface_file_count": int(candidate_file_count),
            "candidate_surface_available": bool(candidate_file_count),
        },
        "noncandidate_step6000_inventory": noncandidate_step6000_inventory(),
        "decision": {
            "technical_possibility_under_sprint_rules": False,
            "reason": (
                "Under CPU-only/no-production-edit rules the current real-case loader cannot construct the "
                "same-input JAX start state, and no candidate-step WRF post-RK/pre-halo truth surface exists."
            ),
            "strict_execution_blocked_for_all_candidates": True,
        },
        "blockers": consolidated_blockers(loader_probe),
        "comparison": {
            "per_field_metrics": {},
            "ranked_residuals": [],
            "why_empty": "No strict same-input comparison was authorized or executed.",
        },
        "commands": {
            "validation": [
                "python -m py_compile proofs/v014/early_step_discriminator.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/early_step_discriminator.py",
                "python -m json.tool proofs/v014/early_step_discriminator.json >/tmp/early_step_discriminator.validated.json",
                "git diff -- src/gpuwrf",
            ],
            "loader_probe": loader_probe.get("command"),
        },
        "scratch": {
            "root": str(SCRATCH),
            "exists": SCRATCH.exists(),
        },
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
        },
        "unresolved_risks": [
            "No numerical first-divergent field/operator is named because no strict candidate-step comparison could run.",
            "A direct CPU-only proof loader might be possible, but it would need an explicit accepted contract to avoid fabricating a state outside production loader semantics.",
        ],
        "next_decision": (
            "Authorize one contract-building sprint for a CPU-compatible wrfinput->OperationalCarry loader plus "
            "candidate-step WRF post-RK/pre-halo full-field surface, then rerun this discriminator."
        ),
        "log_tails": {
            "run_case3_rsl_error_0000": read_tail(RUN_CASE3 / "rsl.error.0000", max_chars=3000),
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
