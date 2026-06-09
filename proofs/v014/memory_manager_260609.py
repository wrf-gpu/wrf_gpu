#!/usr/bin/env python3
"""Memory-manager proof bundle for the 2026-06-09 side-manager contract.

This is intentionally a no-source-analysis proof.  It records the current
memory/FP32 collision map, the exact locks imposed by the active grid-parity
debug lane, and the post-grid-parity task queue without touching production
model code or using the GPU.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "proofs" / "v014" / "memory_manager_260609.json"
OUT_MD = ROOT / "proofs" / "v014" / "memory_manager_260609.md"


REQUIRED_INPUTS = [
    "memory_manager_contract_260609.md",
    ".agent/skills/managing-sprints/SKILL.md",
    ".agent/skills/maintaining-memory/SKILL.md",
    ".agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md",
    ".agent/decisions/V0140-MEMORY-FIX-ROADMAP.md",
    ".agent/decisions/V0140-FP32-ACOUSTIC-ROADMAP.md",
    "proofs/v014/parallel_memory_fp32_manager.json",
    "proofs/v014/parallel_memory_fp32_manager.md",
    "proofs/v014/exact_branch_memory_preflight.json",
    "proofs/v014/exact_branch_memory_preflight.md",
    "proofs/v014/empirical_memory_map.json",
    "proofs/v014/empirical_memory_map.md",
    "proofs/v014/fp32_acoustic_probes.json",
    "proofs/v014/step1_jax_start_domain_input_split.json",
    "proofs/v014/step1_jax_start_domain_input_split.md",
    ".agent/reviews/2026-06-09-v014-parallel-memory-fp32-manager.md",
    ".agent/sprints/2026-06-09-v014-step1-base-state-boundary/sprint-contract.md",
]

HARD_LOCKS = [
    "src/gpuwrf/dynamics/**",
    "src/gpuwrf/runtime/operational_mode.py",
    "src/gpuwrf/integration/d02_replay.py",
    "src/gpuwrf/nesting/**",
    "src/gpuwrf/boundary*",
    "src/gpuwrf/contracts/state.py",
    "boundary, carry, restart, init, wrfout, live-nest, and base-state files",
    "any file currently touched by the active grid-parity sprint",
]

MEMORY_CANDIDATES = [
    {
        "item": "exact_branch_memory_preflight",
        "status": "defer_gpu_run_until_grid_parity_branch_stabilizes",
        "source_files": ["proofs/v014/exact_branch_memory_preflight.py"],
        "locked_files": [],
        "current_action": "prepared_command_only",
        "resume_gate": "primary manager releases GPU and selected post-grid-parity branch",
        "proof_gate": (
            "scripts/run_gpu_lowprio.sh --cores 0-23 -- python "
            "proofs/v014/exact_branch_memory_preflight.py --run-gpu "
            "--nested-input <selected-long-validation-input> --max-dom 3 "
            "--hours 1 --timeout-s 600"
        ),
    },
    {
        "item": "moisture_transport_velocity_reuse",
        "status": "blocked_by_grid_parity_source_lock",
        "source_files": [
            "src/gpuwrf/runtime/operational_mode.py",
            "src/gpuwrf/dynamics/flux_advection.py",
        ],
        "locked_files": [
            "src/gpuwrf/runtime/operational_mode.py",
            "src/gpuwrf/dynamics/flux_advection.py",
        ],
        "current_action": "write_ready_to_run_plan_only",
        "resume_gate": "runtime/dynamics locks released and active moisture advection is on the validation path",
        "proof_gate": "default moist_adv_opt=0 exactness plus active-path conservation, positivity, no-transfer audit",
    },
    {
        "item": "non_radiation_physics_column_tiling_pilot",
        "status": "measure_first_after_grid_parity",
        "source_files": [
            "src/gpuwrf/coupling/scan_adapters.py",
            "src/gpuwrf/coupling/physics_couplers.py",
            "src/gpuwrf/physics/*",
        ],
        "locked_files": ["active-grid-debug-touched files if any overlap at dispatch time"],
        "current_action": "defer_source_work",
        "resume_gate": "exact branch preflight or HLO/RSS evidence names a measured non-radiation offender",
        "proof_gate": "one scheme CPU exact tile-vs-untiled proof, short GPU VRAM suite, real-case smoke",
    },
    {
        "item": "moisture_limiter_workspace_reduction",
        "status": "blocked_by_grid_parity_source_lock",
        "source_files": [
            "src/gpuwrf/dynamics/flux_advection.py",
            "src/gpuwrf/runtime/operational_mode.py",
        ],
        "locked_files": [
            "src/gpuwrf/dynamics/flux_advection.py",
            "src/gpuwrf/runtime/operational_mode.py",
        ],
        "current_action": "defer_source_work",
        "resume_gate": "dycore/runtime locks released and limiter liveness measured as material",
        "proof_gate": "per-species parity, water conservation, positivity/monotonicity, active real-case smoke",
    },
    {
        "item": "acoustic_carry_split_or_pad_cleanup",
        "status": "blocked_by_same_fault_surface",
        "source_files": [
            "src/gpuwrf/dynamics/core/acoustic.py",
            "src/gpuwrf/dynamics/core/small_step_prep.py",
            "src/gpuwrf/dynamics/core/small_step_finish.py",
            "src/gpuwrf/runtime/operational_mode.py",
        ],
        "locked_files": [
            "src/gpuwrf/dynamics/**",
            "src/gpuwrf/runtime/operational_mode.py",
        ],
        "current_action": "defer_to_post_grid_parity_or_fp32_co_design",
        "resume_gate": "P/MU/W live-nest/base-state grid-parity lock released",
        "proof_gate": "fp64 default bit identity, acoustic savepoint parity, warm-bubble/Straka/terrain-rest gates",
    },
    {
        "item": "state_alias_reduction",
        "status": "blocked_by_state_contract_lock_and_adr",
        "source_files": [
            "src/gpuwrf/contracts/state.py",
            "init/restart/wrfout/boundary compatibility paths",
        ],
        "locked_files": [
            "src/gpuwrf/contracts/state.py",
            "boundary, restart, init, wrfout compatibility files",
        ],
        "current_action": "do_not_start",
        "resume_gate": "ADR approved after grid-parity branch stabilizes",
        "proof_gate": "restart roundtrip, wrfout compatibility, boundary and savepoint parity",
    },
]

FP32_CANDIDATES = [
    {
        "item": "R0 precision-mode contract",
        "status": "review_only_until_grid_lock_released",
        "source_files": ["runtime/config/cache/static aux surfaces"],
        "locked_files": ["src/gpuwrf/runtime/operational_mode.py"],
        "current_action": "keep ADR/probe evidence only",
        "proof_gate": "default-off fp64 bit identity and cache-key/report-label tests",
    },
    {
        "item": "R1 explicit base-state plumbing",
        "status": "blocked_by_active_base_state_boundary_debug",
        "source_files": [
            "src/gpuwrf/dynamics/core/small_step_prep.py",
            "src/gpuwrf/dynamics/core/small_step_finish.py",
            "src/gpuwrf/runtime/operational_mode.py",
            "boundary/restart/init/carry staging",
        ],
        "locked_files": [
            "src/gpuwrf/dynamics/**",
            "src/gpuwrf/runtime/operational_mode.py",
            "boundary/restart/init/carry staging",
        ],
        "current_action": "defer_source_work",
        "proof_gate": "focused acoustic prep/finish exactness plus one-step operational carry test",
    },
    {
        "item": "R2 perturbation-authoritative acoustic state",
        "status": "blocked_by_dycore_lock",
        "source_files": [
            "src/gpuwrf/dynamics/core/acoustic.py",
            "src/gpuwrf/dynamics/core/advance_w.py",
            "src/gpuwrf/dynamics/core/calc_p_rho.py",
        ],
        "locked_files": ["src/gpuwrf/dynamics/**"],
        "current_action": "defer_source_work",
        "proof_gate": "small-step WRF savepoint parity, idealized dry gates, transfer audit, VRAM proof",
    },
    {
        "item": "R3 CPU scalar and one-column probes",
        "status": "already_available_proof_only",
        "source_files": ["proofs/v014/fp32_acoustic_probes.py"],
        "locked_files": [],
        "current_action": "carry evidence forward; no new source",
        "proof_gate": "refresh only if ADR or precision formulas change",
    },
]

TOP_NEXT_TASKS = [
    {
        "rank": 1,
        "task": "Run exact-branch memory preflight on the stabilized post-grid-parity branch",
        "why": "Confirms selected long-validation config fits with actual branch code and GPU allocator settings.",
        "gate": "repo GPU lock wrapper, peak VRAM, allocator mode, output count, finiteness record",
    },
    {
        "rank": 2,
        "task": "Resume FP32 R0/R1 only after the live-nest/base-state lock is released",
        "why": "The mixed acoustic lane touches the same P/MU/W/base-state surfaces now under debug.",
        "gate": "default-off fp64 bit identity and explicit base-state plumbing proof",
    },
    {
        "rank": 3,
        "task": "Moisture transport velocity reuse if active moisture advection is in the validation path",
        "why": "It is the only material bit-identical memory cleanup left, but it touches locked runtime/dynamics files.",
        "gate": "default exactness plus active-path conservation, positivity, no-transfer audit",
    },
    {
        "rank": 4,
        "task": "Measurement-led non-radiation column-tiling pilot",
        "why": "Potential GiB-scale gains need HLO/RSS or short GPU peak evidence before source work.",
        "gate": "one scheme tile-vs-untiled exact proof, short GPU VRAM suite, real-case smoke",
    },
    {
        "rank": 5,
        "task": "Co-design acoustic carry split with FP32 acoustic after fp64 parity is repaired",
        "why": "It has material memory upside but previously-reverted dycore liveness risk.",
        "gate": "fp64 exactness, acoustic savepoint parity, warm-bubble/Straka/terrain-rest gates",
    },
]


def run_cmd(cmd: list[str], *, timeout_s: float | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "cmd": cmd,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
            "timeout_s": timeout_s,
        }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: str) -> dict[str, Any]:
    full = ROOT / path
    return json.loads(full.read_text(encoding="utf-8"))


def input_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel in REQUIRED_INPUTS:
        full = ROOT / rel
        row: dict[str, Any] = {"path": rel, "exists": full.exists()}
        if full.exists() and full.is_file():
            row["sha256"] = sha256_file(full)
            row["size_bytes"] = full.stat().st_size
        rows.append(row)
    return rows


def git_snapshot() -> dict[str, Any]:
    status = run_cmd(["git", "status", "--short", "--branch"])
    return {
        "branch": run_cmd(["git", "branch", "--show-current"])["stdout"].strip(),
        "head": run_cmd(["git", "rev-parse", "HEAD"])["stdout"].strip(),
        "ee6cbbe1_is_ancestor": run_cmd(
            ["git", "merge-base", "--is-ancestor", "ee6cbbe1", "HEAD"]
        )["returncode"]
        == 0,
        "base_131b27cd_is_ancestor": run_cmd(
            ["git", "merge-base", "--is-ancestor", "131b27cd", "HEAD"]
        )["returncode"]
        == 0,
        "status_short": status["stdout"].splitlines(),
        "dirty": any(line and not line.startswith("##") for line in status["stdout"].splitlines()),
    }


def tmux_snapshot() -> dict[str, Any]:
    if shutil.which("tmux") is None:
        return {"tmux_available": False, "windows": [], "active_grid_worker_seen": False}
    windows_cmd = run_cmd(["tmux", "list-windows", "-a"])
    windows = windows_cmd["stdout"].splitlines() if windows_cmd["returncode"] == 0 else []
    active = [line for line in windows if "gpt-base-boundary" in line or "gpt-" in line]
    return {
        "tmux_available": True,
        "list_windows_returncode": windows_cmd["returncode"],
        "windows": windows,
        "active_grid_worker_seen": any("gpt-base-boundary" in line for line in windows),
        "grid_worker_windows": active,
    }


def summarize_inputs() -> dict[str, Any]:
    parallel = read_json("proofs/v014/parallel_memory_fp32_manager.json")
    empirical = read_json("proofs/v014/empirical_memory_map.json")
    preflight = read_json("proofs/v014/exact_branch_memory_preflight.json")
    fp32 = read_json("proofs/v014/fp32_acoustic_probes.json")
    latest_grid = read_json("proofs/v014/step1_jax_start_domain_input_split.json")
    return {
        "parallel_memory_fp32_manager": {
            "recommendation": parallel.get("recommendation"),
            "memory_verdict": parallel.get("memory_fix", {}).get("verdict"),
            "source_edits": parallel.get("source_edits"),
            "saved_mib": parallel.get("memory_fix", {})
            .get("target_641x321x50_fp64_bytes", {})
            .get("saved_mib"),
            "fp32_source_verdict": parallel.get("fp32_refresh", {}).get("source_work_verdict"),
        },
        "empirical_memory_map": {
            "verdict": empirical.get("recommendation", {}).get("verdict"),
            "smallest_safe_memory_source_sprint": empirical.get("recommendation", {}).get(
                "smallest_safe_memory_source_sprint"
            ),
            "only_material_bit_identical_cleanup": empirical.get("recommendation", {}).get(
                "only_material_bit_identical_cleanup"
            ),
            "do_not_start_before_grid_parity": empirical.get("recommendation", {}).get(
                "do_not_start_before_grid_parity"
            ),
        },
        "exact_branch_memory_preflight": {
            "verdict": preflight.get("verdict"),
            "gpu_attempted": preflight.get("gpu_run", {}).get("attempted"),
            "planned_command": preflight.get("gpu_run", {}).get("planned_command"),
            "rrtmg_column_tiling_present": preflight.get("branch_controls", {}).get(
                "rrtmg_column_tiling_present"
            ),
            "nested_allocator_controls_present": preflight.get("branch_controls", {}).get(
                "nested_allocator_controls_present"
            ),
        },
        "fp32_acoustic_probes": {
            "recommendation": fp32.get("recommendation"),
            "absolute_total_millipascal_delta_pa": fp32.get("absolute_total_cancellation", {}).get(
                "millipascal_fresh_recovered_delta_pa"
            ),
            "perturbation_millipascal_delta_pa": fp32.get(
                "perturbation_form_preservation", {}
            ).get("millipascal_recovered_delta_pa"),
            "one_column_error_ratios": fp32.get("one_column_recurrence_sensitivity", {}).get(
                "errors_vs_fp64_reference", {}
            ),
        },
        "latest_grid_lock_evidence": {
            "proof": latest_grid.get("proof"),
            "verdict": latest_grid.get("verdict"),
            "next_decision": latest_grid.get("next_decision"),
            "unresolved_risks": latest_grid.get("unresolved_risks"),
            "no_memory_source_work": latest_grid.get("no_memory_source_work"),
            "no_fp32_source_work": latest_grid.get("no_fp32_source_work"),
            "gpu_used": latest_grid.get("gpu_used"),
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    latest = payload["input_summaries"]["latest_grid_lock_evidence"]
    parallel = payload["input_summaries"]["parallel_memory_fp32_manager"]
    preflight = payload["input_summaries"]["exact_branch_memory_preflight"]

    lines = [
        "# Memory Manager 2026-06-09",
        "",
        f"- Recommendation: `{payload['recommendation']}`",
        f"- Branch: `{payload['git']['branch']}`",
        f"- HEAD: `{payload['git']['head']}`",
        f"- CPU-only: `{payload['cpu_only']}`",
        f"- GPU used: `{payload['gpu_usage']['used']}`",
        "- Production source edits: `False`",
        "",
        "## Current Lock",
        "",
        f"- Active grid verdict: `{latest['verdict']}`",
        "- Active grid worker observed: "
        f"`{payload['active_primary_debug']['tmux'].get('active_grid_worker_seen')}`",
        "- Current blocker: exact WRF start-domain base-state source boundary before the "
        "hypsometric `AL/ALT` pass.",
        "- Resulting lock: no memory/FP32 source work, no GPU, no `d02_replay.py`, "
        "runtime, dycore, state-contract, boundary, restart, init, wrfout, or live-nest/base-state edits.",
        "",
        "## Prior Closed Work",
        "",
        f"- Prior side-manager recommendation: `{parallel['recommendation']}`.",
        f"- Closed memory fix: `{parallel['memory_verdict']}`.",
        f"- WDM6 target saving: `{parallel['saved_mib']}` MiB at 641x321x50.",
        f"- FP32 source verdict: `{parallel['fp32_source_verdict']}`.",
        "",
        "## Collision Map",
        "",
        "| Item | Status | Blocking lock | Resume gate |",
        "|---|---|---|---|",
    ]
    for row in payload["memory_candidates"] + payload["fp32_candidates"]:
        locks = ", ".join(row["locked_files"]) if row["locked_files"] else "none"
        lines.append(
            f"| `{row['item']}` | `{row['status']}` | {locks} | {row.get('resume_gate') or row.get('proof_gate')} |"
        )

    lines.extend(
        [
            "",
            "## Exact-Branch Preflight",
            "",
            f"- Current status: `{preflight['verdict']}`.",
            f"- GPU attempted in this lane: `{preflight['gpu_attempted']}`.",
            f"- Prepared command: `{preflight['planned_command']}`.",
            "- This is deliberately not a long validation substitute.",
            "",
            "## GPU Usage",
            "",
            "- GPU used: `False`.",
            "- Peak VRAM: `null`.",
            "- Lock protocol evidence: no GPU command was launched; the queued preflight command uses "
            "`scripts/run_gpu_lowprio.sh`; active grid worker `gpt-base-boundary` was visible in tmux.",
            "",
            "## Top 5 Next Tasks After Grid Parity",
            "",
        ]
    )
    for row in payload["top_5_next_tasks"]:
        lines.append(f"{row['rank']}. {row['task']} - gate: {row['gate']}.")

    lines.extend(
        [
            "",
            "## Files Changed",
            "",
            "- `proofs/v014/memory_manager_260609.py`",
            "- `proofs/v014/memory_manager_260609.json`",
            "- `proofs/v014/memory_manager_260609.md`",
            "- `.agent/reviews/2026-06-09-v014-memory-manager-260609.md`",
            "",
            "## Commands Run",
            "",
            "- `python -m py_compile proofs/v014/memory_manager_260609.py`",
            "- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/memory_manager_260609.py`",
            "- `python -m json.tool proofs/v014/memory_manager_260609.json >/tmp/memory_manager_260609.validated.json`",
            "- `git diff --check`",
            "- `git diff -- src/gpuwrf`",
            "",
            "## Proof Objects",
            "",
            "- `proofs/v014/memory_manager_260609.json`",
            "- `proofs/v014/memory_manager_260609.md`",
            "- `.agent/reviews/2026-06-09-v014-memory-manager-260609.md`",
            "",
            "## Recommendation",
            "",
            "`REVIEW_ONLY`: there is no production source change to merge from this lane. Keep the "
            "report as the current memory/FP32 side-manager handoff, and resume source work only after "
            "the primary grid-parity manager releases the relevant locks.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_payload() -> dict[str, Any]:
    tmux = tmux_snapshot()
    return {
        "proof": "memory_manager_260609",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "recommendation": "REVIEW_ONLY",
        "recommendation_reason": (
            "No safe non-conflicting source edit remains after ee6cbbe1; active "
            "grid-parity/base-state debugging owns the source surfaces needed by "
            "larger memory and FP32 work."
        ),
        "cpu_only": True,
        "production_source_edits": False,
        "git": git_snapshot(),
        "inputs": input_manifest(),
        "input_summaries": summarize_inputs(),
        "active_primary_debug": {
            "active_sprint": ".agent/sprints/2026-06-09-v014-step1-base-state-boundary",
            "active_worker": "tmux 0:4 gpt-base-boundary, if still running",
            "priority": "primary grid-parity/base-state debug takes priority over this lane",
            "tmux": tmux,
        },
        "hard_locks_respected": HARD_LOCKS,
        "memory_candidates": MEMORY_CANDIDATES,
        "fp32_candidates": FP32_CANDIDATES,
        "deferred_items": MEMORY_CANDIDATES + FP32_CANDIDATES,
        "gpu_usage": {
            "used": False,
            "peak_vram_mib": None,
            "long_gpu_validation_run": False,
            "lock_protocol_evidence": (
                "No GPU command launched. The only prepared GPU path is the exact-branch "
                "preflight through scripts/run_gpu_lowprio.sh after grid parity stabilizes."
            ),
        },
        "files_changed": [
            "proofs/v014/memory_manager_260609.py",
            "proofs/v014/memory_manager_260609.json",
            "proofs/v014/memory_manager_260609.md",
            ".agent/reviews/2026-06-09-v014-memory-manager-260609.md",
        ],
        "commands_run": [
            "python -m py_compile proofs/v014/memory_manager_260609.py",
            "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/memory_manager_260609.py",
            "python -m json.tool proofs/v014/memory_manager_260609.json >/tmp/memory_manager_260609.validated.json",
            "git diff --check",
            "git diff -- src/gpuwrf",
        ],
        "proof_objects": [
            "proofs/v014/memory_manager_260609.json",
            "proofs/v014/memory_manager_260609.md",
            ".agent/reviews/2026-06-09-v014-memory-manager-260609.md",
        ],
        "source_locks_blocking_deferred_items": {
            row["item"]: row["locked_files"] for row in MEMORY_CANDIDATES + FP32_CANDIDATES
        },
        "top_5_next_tasks": TOP_NEXT_TASKS,
        "unresolved_risks": [
            "Exact-branch memory preflight is prepared but not a GPU memory-fit proof for the final branch.",
            "FP32 acoustic remains feasible in principle, but any source integration now risks contaminating fp64 grid-parity diagnosis.",
            "The active base-state boundary sprint may change the exact source lock list; this report should be refreshed if it lands a production patch.",
        ],
    }


def main() -> int:
    payload = build_payload()
    write_json(OUT_JSON, payload)
    write_markdown(OUT_MD, payload)
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")
    print(f"Recommendation: {payload['recommendation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
