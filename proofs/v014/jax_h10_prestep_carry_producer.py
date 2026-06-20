#!/usr/bin/env python3
"""V0.14 H10 pre-step OperationalCarry checkpoint producer.

This is a proof-only producer.  It uses the existing live-nested runtime to
advance the L2 d01->d02 case to completed d02 step 5999, writes a host pickle
runtime checkpoint with the full d02 OperationalCarry, and then runs the
existing h10 pre-halo comparison in an external scratch tree so this sprint does
not rewrite prior proof files.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUT_JSON = ROOT / "proofs/v014/jax_h10_prestep_carry_producer.json"
OUT_MD = ROOT / "proofs/v014/jax_h10_prestep_carry_producer.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-09-v014-h10-prestep-carry-producer.md"

SPRINT_CONTRACT = ROOT / ".agent/sprints/2026-06-09-v014-h10-prestep-carry-producer/sprint-contract.md"
HANDOFF = ROOT / ".agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md"
VALIDATING_PHYSICS_SKILL = ROOT / ".agent/skills/validating-physics/SKILL.md"
WRF_ORACLE_SKILL = ROOT / ".agent/skills/building-wrf-oracles/SKILL.md"

JAX_H10_SCRIPT = ROOT / "proofs/v014/jax_h10_prestep_carry.py"
JAX_H10_JSON = ROOT / "proofs/v014/jax_h10_prestep_carry.json"
JAX_PRE_HALO_JSON = ROOT / "proofs/v014/jax_pre_halo_capture.json"
WRF_REFRESH_JSON = ROOT / "proofs/v014/wrf_post_rk_refresh_localization.json"
WRF_REFRESH_MD = ROOT / "proofs/v014/wrf_post_rk_refresh_localization.md"
SAVEPOINT_REQUEST_JSON = ROOT / "proofs/v014/same_state_savepoint_request.json"
DYNAMIC_ATTRIBUTION_JSON = ROOT / "proofs/v014/dynamic_field_attribution.json"

RUN_ID = "20260501_18z_l2_72h_20260519T173026Z"
DEFAULT_INPUT_ROOTS = (
    Path("/tmp/v0120_merged_run_root"),
    Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l2"),
)
TARGET_DOMAIN = "d02"
TARGET_STEP = 6000
PRESTEP_COMPLETED_STEPS = TARGET_STEP - 1
TARGET_LEAD_SECONDS_AFTER_STEP = 36000.0
TARGET_DT_S = TARGET_LEAD_SECONDS_AFTER_STEP / float(TARGET_STEP)
MAX_DOM = 2


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def artifact_root() -> Path:
    candidates: list[Path] = []
    override = os.environ.get("WRFGPU2_H10_PRESTEP_CARRY_ARTIFACT_ROOT")
    if override:
        candidates.append(Path(override))
    candidates.extend(
        [
            Path("<DATA_ROOT>/wrf_gpu2/v014_h10_prestep_carry"),
            Path("/tmp/wrf_gpu2_v014_h10_prestep_carry"),
        ]
    )
    for candidate in candidates:
        if not str(candidate):
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue
    raise OSError("could not create <DATA_ROOT> or /tmp artifact root")


ARTIFACT_ROOT = artifact_root()
CHECKPOINT_PATH = ARTIFACT_ROOT / "d02_step5999_full_carry.pkl"
CHECKPOINT_PROVENANCE_PATH = ARTIFACT_ROOT / "d02_step5999_full_carry.provenance.json"
COMPARE_ROOT = ARTIFACT_ROOT / "jax_h10_compare_tree"


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
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
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


def load_optional_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def run_command(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "command": command,
        "cwd": str(cwd) if cwd is not None else None,
        "returncode": int(proc.returncode),
        "wall_s": float(time.perf_counter() - start),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def nvidia_smi_snapshot(label: str) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = run_command(command)
    except FileNotFoundError as exc:
        return {"label": label, "status": "missing", "error": repr(exc)}
    return {"label": label, "status": "ok" if result["returncode"] == 0 else "error", **result}


def jax_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "XLA_PYTHON_CLIENT_ALLOCATOR": os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR"),
        "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
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


def input_run_dir() -> Path:
    for root in DEFAULT_INPUT_ROOTS:
        path = root / RUN_ID
        if path.is_dir() and (path / "wrfinput_d01").exists() and (path / "wrfinput_d02").exists():
            return path
    searched = ", ".join(str(root / RUN_ID) for root in DEFAULT_INPUT_ROOTS)
    raise FileNotFoundError(f"missing L2 native-init run directory; searched {searched}")


def checkpoint_load_probe(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "MISSING", **path_info(path)}
    start = time.perf_counter()
    try:
        import jax  # noqa: PLC0415
        from gpuwrf.runtime.checkpoint import read_checkpoint_with_runtime_state  # noqa: PLC0415

        state, namelist, grid, step_index, runtime_state = read_checkpoint_with_runtime_state(path)
        if runtime_state is not None:
            jax.block_until_ready(runtime_state.state.theta)
        else:
            jax.block_until_ready(state.theta)
        return {
            "status": "LOAD_OK",
            **path_info(path),
            "step_index": int(step_index),
            "has_runtime_state": runtime_state is not None,
            "runtime_state_type": type(runtime_state).__name__ if runtime_state is not None else None,
            "namelist_type": type(namelist).__name__,
            "grid_shape": {
                "nz": int(getattr(grid, "nz")),
                "ny": int(getattr(grid, "ny")),
                "nx": int(getattr(grid, "nx")),
            },
            "wall_s": float(time.perf_counter() - start),
        }
    except Exception as exc:  # pragma: no cover - recorded in proof output
        return {"status": "LOAD_ERROR", **path_info(path), "error": repr(exc), "wall_s": float(time.perf_counter() - start)}


def scalar_stats(array: Any) -> dict[str, Any]:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    data = jnp.asarray(array)
    finite = jnp.isfinite(data) if jnp.issubdtype(data.dtype, jnp.floating) else jnp.ones(data.shape, dtype=bool)
    values = {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "all_finite": bool(np.asarray(jnp.all(finite))),
        "min": float(np.asarray(jnp.nanmin(data))) if data.size else None,
        "max": float(np.asarray(jnp.nanmax(data))) if data.size else None,
        "max_abs": float(np.asarray(jnp.nanmax(jnp.abs(data)))) if data.size else None,
    }
    jax.block_until_ready(data)
    return values


def carry_summary(carry: Any) -> dict[str, Any]:
    state = carry.state
    fields = ("theta", "u", "v", "w", "p_perturbation", "ph_perturbation", "mu_perturbation")
    return {name: scalar_stats(getattr(state, name)) for name in fields}


def produce_checkpoint(path: Path) -> dict[str, Any]:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from gpuwrf.integration.nested_pipeline import (  # noqa: PLC0415
        NestedPipelineConfig,
        _load_domains,
        domain_names_for,
    )
    from gpuwrf.runtime.domain_tree import (  # noqa: PLC0415
        DomainTree,
        _operational_advance_factory,
        _operational_force,
        run_operational_domain_tree,
    )
    from gpuwrf.runtime.checkpoint import write_checkpoint  # noqa: PLC0415

    run_dir = input_run_dir()
    config = NestedPipelineConfig(
        input_dir=run_dir,
        output_dir=ARTIFACT_ROOT / "unused_wrfouts",
        proof_dir=ARTIFACT_ROOT / "nested_setup",
        hours=10,
        max_dom=MAX_DOM,
        feedback=False,
    )
    names = domain_names_for(MAX_DOM)
    start = time.perf_counter()
    hierarchy, bundles, meta, run_start, dt_by_domain = _load_domains(config, names)
    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
    parent = hierarchy.parent(TARGET_DOMAIN)
    if parent is None:
        raise ValueError(f"{TARGET_DOMAIN} has no live-nest parent")
    ratio = int(next(edge.parent_grid_ratio for edge in hierarchy.nests if edge.parent == parent and edge.child == TARGET_DOMAIN))
    target_dt = float(dt_by_domain[TARGET_DOMAIN])
    if abs(target_dt - TARGET_DT_S) > 1.0e-9:
        raise ValueError(f"{TARGET_DOMAIN} dt_s={target_dt:g}, expected {TARGET_DT_S:g} from h10 step mapping")

    full_parent_steps = PRESTEP_COMPLETED_STEPS // ratio
    child_remainder = PRESTEP_COMPLETED_STEPS - full_parent_steps * ratio
    root_segment_steps = int(os.environ.get("WRFGPU2_H10_ROOT_SEGMENT_STEPS", "200"))
    root_segment_steps = max(1, root_segment_steps)
    print(
        f"producer_start backend={jax.default_backend()} parent={parent} ratio={ratio} "
        f"full_parent_steps={full_parent_steps} child_remainder={child_remainder}",
        flush=True,
    )

    carries: dict[str, Any] | None = None
    own_steps: dict[str, int] = {name: 0 for name in names}
    segment_records: list[dict[str, Any]] = []
    completed_parent = 0
    while completed_parent < full_parent_steps:
        seg = min(root_segment_steps, full_parent_steps - completed_parent)
        seg_start = time.perf_counter()
        result = run_operational_domain_tree(
            tree,
            root_steps=int(seg),
            feedback_enabled=False,
            output=None,
            output_cadence_steps=None,
            block_between=True,
            carries=carries,
            initial_own_steps=own_steps,
        )
        jax.block_until_ready(tuple(carry.state.theta for carry in result.carries.values()))
        carries = dict(result.carries)
        own_steps = dict(result.own_steps)
        completed_parent = int(own_steps[parent])
        record = {
            "segment": len(segment_records) + 1,
            "root_steps": int(seg),
            "wall_s": float(time.perf_counter() - seg_start),
            "own_steps": dict(own_steps),
        }
        segment_records.append(record)
        print(
            f"producer_progress segment={record['segment']} d01={own_steps.get('d01')} d02={own_steps.get('d02')} "
            f"wall_s={record['wall_s']:.1f}",
            flush=True,
        )

    if carries is None:
        raise RuntimeError("no carries produced by live-nested run")
    partial_record: dict[str, Any] | None = None
    if child_remainder:
        partial_start = time.perf_counter()
        advance = _operational_advance_factory(tree)
        edge = next(edge for edge in tree.children(parent) if edge.child == TARGET_DOMAIN)
        parent_start_step = int(own_steps[parent]) + 1
        carries[parent] = advance(parent, carries[parent], parent_start_step, 1)
        jax.block_until_ready(carries[parent].state.theta)
        own_steps[parent] += 1
        carries[TARGET_DOMAIN] = _operational_force(edge, carries[parent], carries[TARGET_DOMAIN])
        child_start_step = int(own_steps[TARGET_DOMAIN]) + 1
        carries[TARGET_DOMAIN] = advance(TARGET_DOMAIN, carries[TARGET_DOMAIN], child_start_step, int(child_remainder))
        jax.block_until_ready(carries[TARGET_DOMAIN].state.theta)
        own_steps[TARGET_DOMAIN] += int(child_remainder)
        partial_record = {
            "parent_advanced_step": int(parent_start_step),
            "child_start_step": int(child_start_step),
            "child_steps": int(child_remainder),
            "own_steps_after": dict(own_steps),
            "wall_s": float(time.perf_counter() - partial_start),
        }
        print(
            f"producer_partial parent_step={parent_start_step} child_steps={child_remainder} "
            f"d02={own_steps[TARGET_DOMAIN]} wall_s={partial_record['wall_s']:.1f}",
            flush=True,
        )

    if int(own_steps[TARGET_DOMAIN]) != PRESTEP_COMPLETED_STEPS:
        raise RuntimeError(f"d02 own_steps={own_steps[TARGET_DOMAIN]}, expected {PRESTEP_COMPLETED_STEPS}")

    d02_carry = carries[TARGET_DOMAIN]
    d02_bundle = tree.domains[TARGET_DOMAIN]
    write_checkpoint(
        d02_carry.state,
        d02_bundle.namelist,
        d02_bundle.grid,
        PRESTEP_COMPLETED_STEPS,
        path,
        runtime_state=d02_carry,
    )
    load_probe = checkpoint_load_probe(path)
    if load_probe.get("status") != "LOAD_OK" or load_probe.get("step_index") != PRESTEP_COMPLETED_STEPS:
        raise RuntimeError(f"checkpoint load probe failed: {load_probe}")

    return {
        "status": "PRODUCED",
        "checkpoint": load_probe,
        "run_dir": str(run_dir),
        "run_start_utc": run_start.isoformat(),
        "dt_by_domain": {name: float(value) for name, value in dt_by_domain.items()},
        "target": {
            "domain": TARGET_DOMAIN,
            "target_step": TARGET_STEP,
            "prestep_completed_steps": PRESTEP_COMPLETED_STEPS,
            "lead_seconds_before_step": float(PRESTEP_COMPLETED_STEPS) * float(target_dt),
            "lead_seconds_after_step": TARGET_LEAD_SECONDS_AFTER_STEP,
            "dt_s": float(target_dt),
        },
        "nesting": {
            "parent": parent,
            "parent_grid_ratio": int(ratio),
            "full_parent_steps": int(full_parent_steps),
            "child_remainder_steps": int(child_remainder),
            "own_steps": dict(own_steps),
            "metadata": meta,
        },
        "segments": segment_records,
        "partial_parent_subcycle": partial_record,
        "carry_summary": carry_summary(d02_carry),
        "wall_s": float(time.perf_counter() - start),
    }


def ensure_symlink(target: Path, link: Path) -> None:
    if link.exists() or link.is_symlink():
        if link.is_symlink() and Path(os.readlink(link)) == target:
            return
        if link.is_dir() and not link.is_symlink():
            shutil.rmtree(link)
        else:
            link.unlink()
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)


def setup_compare_tree() -> dict[str, Any]:
    COMPARE_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_symlink(ROOT / "src", COMPARE_ROOT / "src")
    (COMPARE_ROOT / "proofs/v014").mkdir(parents=True, exist_ok=True)
    (COMPARE_ROOT / ".agent/reviews").mkdir(parents=True, exist_ok=True)
    (COMPARE_ROOT / ".agent/decisions").mkdir(parents=True, exist_ok=True)
    (COMPARE_ROOT / ".agent/sprints/2026-06-09-v014-h10-prestep-carry-checkpoint").mkdir(parents=True, exist_ok=True)
    (COMPARE_ROOT / ".agent/skills/validating-physics").mkdir(parents=True, exist_ok=True)

    script_target = COMPARE_ROOT / "proofs/v014/jax_h10_prestep_carry.py"
    shutil.copy2(JAX_H10_SCRIPT, script_target)
    for source in [
        JAX_PRE_HALO_JSON,
        WRF_REFRESH_JSON,
        WRF_REFRESH_MD,
        SAVEPOINT_REQUEST_JSON,
        DYNAMIC_ATTRIBUTION_JSON,
    ]:
        ensure_symlink(source, COMPARE_ROOT / "proofs/v014" / source.name)
    ensure_symlink(HANDOFF, COMPARE_ROOT / ".agent/decisions" / HANDOFF.name)
    ensure_symlink(
        ROOT / ".agent/sprints/2026-06-09-v014-h10-prestep-carry-checkpoint/sprint-contract.md",
        COMPARE_ROOT / ".agent/sprints/2026-06-09-v014-h10-prestep-carry-checkpoint/sprint-contract.md",
    )
    ensure_symlink(
        VALIDATING_PHYSICS_SKILL,
        COMPARE_ROOT / ".agent/skills/validating-physics/SKILL.md",
    )
    return {
        "compare_root": str(COMPARE_ROOT),
        "script": path_info(script_target),
        "src_symlink": str(COMPARE_ROOT / "src"),
    }


def run_h10_comparison(checkpoint_path: Path) -> dict[str, Any]:
    setup = setup_compare_tree()
    env = dict(os.environ)
    env["WRFGPU2_H10_PRESTEP_CARRY"] = str(checkpoint_path)
    env["JAX_PLATFORMS"] = "cpu"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = "src"
    command = [sys.executable, "proofs/v014/jax_h10_prestep_carry.py"]
    result = run_command(command, cwd=COMPARE_ROOT, env=env)
    output_json = COMPARE_ROOT / "proofs/v014/jax_h10_prestep_carry.json"
    comparison_payload = None
    if output_json.exists():
        try:
            comparison_payload = load_json(output_json)
        except Exception as exc:  # pragma: no cover
            comparison_payload = {"parse_error": repr(exc)}
    return {
        "status": "RAN" if result["returncode"] == 0 else "ERROR",
        "setup": setup,
        "command_result": result,
        "output_json": path_info(output_json),
        "output_md": path_info(COMPARE_ROOT / "proofs/v014/jax_h10_prestep_carry.md"),
        "output_review": path_info(COMPARE_ROOT / ".agent/reviews/2026-06-09-v014-h10-prestep-carry-checkpoint.md"),
        "verdict": comparison_payload.get("verdict") if isinstance(comparison_payload, Mapping) else None,
        "comparison": comparison_payload.get("comparison") if isinstance(comparison_payload, Mapping) else None,
        "blocked": comparison_payload.get("blocked") if isinstance(comparison_payload, Mapping) else None,
        "payload_summary": {
            "comparison_run": comparison_payload.get("comparison_run")
            if isinstance(comparison_payload, Mapping)
            else None,
            "first_mismatch": (
                comparison_payload.get("comparison", {}).get("first_mismatch")
                if isinstance(comparison_payload, Mapping) and isinstance(comparison_payload.get("comparison"), Mapping)
                else None
            ),
        },
    }


def proof_inputs() -> dict[str, Any]:
    return {
        "project_constitution": path_info(ROOT / "PROJECT_CONSTITUTION.md"),
        "agents": path_info(ROOT / "AGENTS.md"),
        "sprint_contract": path_info(SPRINT_CONTRACT),
        "handoff": path_info(HANDOFF),
        "validating_physics_skill": path_info(VALIDATING_PHYSICS_SKILL),
        "building_wrf_oracles_skill": path_info(WRF_ORACLE_SKILL),
        "jax_h10_prestep_carry_json": path_info(JAX_H10_JSON),
        "jax_pre_halo_capture_json": path_info(JAX_PRE_HALO_JSON),
        "wrf_post_rk_refresh_localization_json": path_info(WRF_REFRESH_JSON),
        "same_state_savepoint_request_json": path_info(SAVEPOINT_REQUEST_JSON),
    }


def blocked_payload(reason: str, detail: str) -> dict[str, Any]:
    return {
        "status": "BLOCKED",
        "reason": reason,
        "detail": detail,
        "exact_missing_input_or_command": (
            "Run the producer on a backend that can complete the real L2 d01->d02 h10 live-nested "
            "replay and write <DATA_ROOT>/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl, "
            "or provide an equivalent CPU-loadable gpuwrf-runtime-checkpoint with runtime_state at step_index=5999."
        ),
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    checkpoint = payload.get("checkpoint", {})
    comparison = payload.get("comparison_run") or {}
    lines = [
        "# V0.14 H10 Pre-Step Carry Producer",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Result",
        "",
        f"- Checkpoint produced: `{payload['checkpoint_produced']}`.",
        f"- Checkpoint path: `{checkpoint.get('path')}`.",
        f"- CPU-loadable: `{checkpoint.get('status') == 'LOAD_OK'}`.",
        f"- GPU used: `{payload['gpu_used']}`.",
        f"- Comparison run: `{comparison.get('status')}`.",
        f"- Comparison verdict: `{comparison.get('verdict')}`.",
    ]
    if payload.get("blocked"):
        lines.extend(["", "## Blocker", "", f"- `{payload['blocked']['reason']}`: {payload['blocked']['detail']}"])
    lines.extend(
        [
            "",
            "## Next Decision",
            "",
            payload["next_decision"],
            "",
        ]
    )
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    commands = payload.get("commands", {})
    lines = [
        "# Review: V0.14 H10 Pre-Step Carry Producer",
        "",
        f"verdict: `{payload['verdict']}`",
        "",
        "objective: produce a CPU-loadable full JAX OperationalCarry checkpoint at completed d02 step 5999 and run the h10 pre-halo comparison if produced.",
        "",
        "files changed:",
        "- `proofs/v014/jax_h10_prestep_carry_producer.py`",
        "- `proofs/v014/jax_h10_prestep_carry_producer.json`",
        "- `proofs/v014/jax_h10_prestep_carry_producer.md`",
        "- `.agent/reviews/2026-06-09-v014-h10-prestep-carry-producer.md`",
        "",
        "commands run:",
    ]
    for command in commands.get("run_or_validation", []):
        lines.append(f"- `{command}`")
    lines.extend(
        [
            "",
            "proof objects produced:",
            "- `proofs/v014/jax_h10_prestep_carry_producer.json`",
            "- `proofs/v014/jax_h10_prestep_carry_producer.md`",
            "- `.agent/reviews/2026-06-09-v014-h10-prestep-carry-producer.md`",
        ]
    )
    checkpoint_path = payload.get("checkpoint", {}).get("path")
    if checkpoint_path:
        lines.append(f"- `{checkpoint_path}`")
    lines.extend(["", "unresolved risks:"])
    for risk in payload.get("unresolved_risks", []):
        lines.append(f"- {risk}")
    lines.extend(["", f"next decision needed: {payload['next_decision']}", ""])
    return "\n".join(lines)


def decide_verdict(checkpoint: Mapping[str, Any], comparison: Mapping[str, Any] | None, blocked: Mapping[str, Any] | None) -> str:
    if blocked:
        return f"PRODUCER_BLOCKED_{blocked['reason']}"
    if comparison and comparison.get("verdict"):
        return str(comparison["verdict"])
    if checkpoint.get("status") == "LOAD_OK":
        return "PRODUCER_CHECKPOINT_WRITTEN_COMPARE_NOT_RUN"
    return "PRODUCER_BLOCKED_UNKNOWN"


def main() -> int:
    started = time.perf_counter()
    wrf_refresh = load_json(WRF_REFRESH_JSON)
    savepoint_request = load_json(SAVEPOINT_REQUEST_JSON)
    _ = load_json(JAX_PRE_HALO_JSON)
    _ = load_json(JAX_H10_JSON)

    env = jax_environment()
    backend = env.get("jax_default_backend")
    gpu_backend = backend not in {None, "cpu"}
    allow_gpu = truthy(os.environ.get("WRFGPU2_H10_PRODUCER_ALLOW_GPU"))
    force_cpu_run = truthy(os.environ.get("WRFGPU2_H10_PRODUCER_RUN_CPU"))
    skip_compare = truthy(os.environ.get("WRFGPU2_H10_PRODUCER_SKIP_COMPARE"))
    force_reproduce = truthy(os.environ.get("WRFGPU2_H10_PRODUCER_FORCE_REPRODUCE"))
    existing_probe = checkpoint_load_probe(CHECKPOINT_PATH)

    nvidia_before = nvidia_smi_snapshot("before") if gpu_backend else None
    producer: dict[str, Any] | None = None
    blocked: dict[str, Any] | None = None
    checkpoint = existing_probe
    reused_existing = (
        checkpoint.get("status") == "LOAD_OK"
        and checkpoint.get("step_index") == PRESTEP_COMPLETED_STEPS
        and not force_reproduce
    )

    if reused_existing:
        producer = {"status": "REUSED_EXISTING", "checkpoint": checkpoint}
    elif backend == "cpu" and not force_cpu_run:
        blocked = blocked_payload(
            "CPU_FULL_H10_REPLAY_INSUFFICIENT",
            (
                "No step-5999 checkpoint exists yet, and the default CPU command is kept to setup/load/proof mode. "
                "The real producer requires a 10 h live-nested L2 replay to d02 step 5999; run the minimal GPU "
                "producer with WRFGPU2_H10_PRODUCER_ALLOW_GPU=1 or set WRFGPU2_H10_PRODUCER_RUN_CPU=1 to force CPU."
            ),
        )
    elif gpu_backend and not allow_gpu:
        blocked = blocked_payload(
            "GPU_BACKEND_NOT_EXPLICITLY_ALLOWED",
            "A GPU backend is visible, but WRFGPU2_H10_PRODUCER_ALLOW_GPU=1 was not set for this minimal GPU producer run.",
        )
    else:
        try:
            producer = produce_checkpoint(CHECKPOINT_PATH)
            checkpoint = producer["checkpoint"]
        except Exception as exc:  # pragma: no cover - recorded in proof output
            blocked = blocked_payload(type(exc).__name__, repr(exc))
            checkpoint = checkpoint_load_probe(CHECKPOINT_PATH)

    comparison: dict[str, Any] | None = None
    if not blocked and checkpoint.get("status") == "LOAD_OK" and not skip_compare:
        comparison = run_h10_comparison(CHECKPOINT_PATH)
        if comparison.get("status") != "RAN":
            blocked = blocked_payload(
                "COMPARISON_COMMAND_FAILED",
                f"external h10 comparison returned {comparison.get('command_result', {}).get('returncode')}",
            )

    nvidia_after = nvidia_smi_snapshot("after") if gpu_backend else None
    if producer and producer.get("status") == "PRODUCED":
        provenance = {
            "schema": "wrfgpu2.v014.h10_prestep_carry_checkpoint_provenance.v1",
            "produced_utc": datetime.now(timezone.utc).isoformat(),
            "gpu_used": bool(gpu_backend),
            "producer_command": " ".join(sys.argv),
            "producer_env": {
                "WRFGPU2_H10_PRODUCER_ALLOW_GPU": os.environ.get("WRFGPU2_H10_PRODUCER_ALLOW_GPU"),
                "WRFGPU2_H10_PRODUCER_FORCE_REPRODUCE": os.environ.get("WRFGPU2_H10_PRODUCER_FORCE_REPRODUCE"),
                "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "PYTHONPATH": os.environ.get("PYTHONPATH"),
            },
            "environment": env,
            "nvidia_smi_before": nvidia_before,
            "nvidia_smi_after": nvidia_after,
            "checkpoint": checkpoint,
            "producer_summary": {
                "status": producer.get("status"),
                "wall_s": producer.get("wall_s"),
                "target": producer.get("target"),
                "nesting": producer.get("nesting"),
                "segment_count": len(producer.get("segments", [])),
                "partial_parent_subcycle": producer.get("partial_parent_subcycle"),
            },
        }
        write_json(CHECKPOINT_PROVENANCE_PATH, provenance)
    checkpoint_provenance = load_optional_json(CHECKPOINT_PROVENANCE_PATH)
    gpu_used = bool(
        (gpu_backend and producer and producer.get("status") == "PRODUCED")
        or (
            isinstance(checkpoint_provenance, Mapping)
            and checkpoint_provenance.get("gpu_used")
            and checkpoint.get("status") == "LOAD_OK"
        )
    )
    verdict = decide_verdict(checkpoint, comparison, blocked)
    first_mismatch = (
        comparison.get("payload_summary", {}).get("first_mismatch")
        if isinstance(comparison, Mapping)
        else None
    )
    gpu_command_display = None
    if isinstance(checkpoint_provenance, Mapping):
        gpu_command_display = checkpoint_provenance.get("producer_command_display")
    if not gpu_command_display:
        gpu_command_display = (
            "WRFGPU2_H10_PRODUCER_ALLOW_GPU=1 PYTHONPATH=src "
            "python proofs/v014/jax_h10_prestep_carry_producer.py"
        )
    next_decision = (
        "Open a T history/source-attribution sprint before any production "
        "source fix; compare JAX theta/history candidates against WRF "
        "T_HIST_SRC/grid%th_phy_m_t0 and THM-side candidates."
        if first_mismatch
        else (
            "Open a narrower producer/debug sprint for the recorded blocker."
            if blocked
            else "Escalate grid-parity investigation beyond this selected h10 patch."
        )
    )

    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.jax_h10_prestep_carry_producer.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "checkpoint_produced": bool(checkpoint.get("status") == "LOAD_OK"),
        "checkpoint_reused_existing": bool(reused_existing),
        "checkpoint": checkpoint,
        "checkpoint_provenance": checkpoint_provenance,
        "producer": producer,
        "comparison_run": comparison,
        "blocked": blocked,
        "cpu_only_default": True,
        "gpu_used": gpu_used,
        "gpu_policy": {
            "default_cpu_only": True,
            "gpu_allowed_by_env": bool(allow_gpu),
            "cpu_insufficient_reason": (
                "The full producer is a 10 h live-nested L2 replay to d02 step 5999. "
                "The required CPU validation command is used to load the host checkpoint and run the comparison; "
                "the heavy checkpoint production was allowed only with WRFGPU2_H10_PRODUCER_ALLOW_GPU=1."
                if gpu_used
                else None
            ),
            "nvidia_smi_before": nvidia_before,
            "nvidia_smi_after": nvidia_after,
        },
        "production_src_edits": False,
        "wrf_source_edits": False,
        "tost_run": False,
        "switzerland_validation_run": False,
        "fp32_source_landing": False,
        "no_hermes": True,
        "inputs_read": proof_inputs(),
        "target": {
            "domain": TARGET_DOMAIN,
            "wrf_verdict": wrf_refresh.get("verdict"),
            "valid_time_utc": wrf_refresh["target_confirmed"]["valid_time_utc"],
            "wrf_step": TARGET_STEP,
            "prestep_completed_steps": PRESTEP_COMPLETED_STEPS,
            "lead_seconds_after_step": TARGET_LEAD_SECONDS_AFTER_STEP,
            "dt_s": TARGET_DT_S,
            "run_request": savepoint_request.get("run_request"),
        },
        "environment": env,
        "artifact_root": str(ARTIFACT_ROOT),
        "commands": {
            "argv": sys.argv,
            "run_or_validation": [
                "python -m py_compile proofs/v014/jax_h10_prestep_carry_producer.py",
                *([str(gpu_command_display)] if gpu_used else []),
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/jax_h10_prestep_carry_producer.py",
                "python -m json.tool proofs/v014/jax_h10_prestep_carry_producer.json >/tmp/jax_h10_prestep_carry_producer.validated.json",
            ],
            "comparison_command": (
                f"cd {COMPARE_ROOT} && WRFGPU2_H10_PRESTEP_CARRY={CHECKPOINT_PATH} "
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src "
                "python proofs/v014/jax_h10_prestep_carry.py"
            )
            if comparison
            else None,
        },
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "review": str(OUT_REVIEW),
            "checkpoint": str(CHECKPOINT_PATH),
            "checkpoint_provenance": str(CHECKPOINT_PROVENANCE_PATH),
            "comparison_tree": str(COMPARE_ROOT),
        },
        "unresolved_risks": [
            "The comparison covers Boole's selected h10 patch, not the full grid.",
            "The producer uses private proof/runtime helpers, intentionally without landing a public checkpoint API.",
        ],
        "next_decision": next_decision,
        "wall_s": float(time.perf_counter() - started),
    }

    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")

    print(payload["verdict"])
    print(f"json={OUT_JSON}")
    print(f"checkpoint={CHECKPOINT_PATH if checkpoint.get('status') == 'LOAD_OK' else 'NONE'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
