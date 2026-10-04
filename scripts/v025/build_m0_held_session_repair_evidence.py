#!/usr/bin/env python3
"""Build and independently re-derive the compact Amendment-6 CPU repair proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from xml.etree import ElementTree


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import m0_core_session_protocol as core_session  # noqa: E402
import m0_exact_boundary_contract as exact_contract  # noqa: E402
import m0_postlock_census as postlock  # noqa: E402
import m0_three_window_executor as executor  # noqa: E402


SCHEMA = "wrf_gpu2.v025.m0.amendment6_repair_cpu_proof.v1"
OUTCOME = "CPU_AMENDMENT6_REPAIR_GREEN_READY_FOR_NARROW_CLOSURE"
FROZEN_SRC_TREE = "a6885ceded260df2f5777d7366d75a5d38947cb7"
DEFAULT_JUNIT = REPO / "proofs/v025/m0/m0_held_session_repair_tests.xml"
DEFAULT_OUTPUT = REPO / "proofs/v025/m0/m0_held_session_repair_evidence.json"

REVIEW_SOURCES = (
    "scripts/v025/build_m0_held_session_repair_evidence.py",
    "scripts/v025/m0_c1_c2_cpu_proofs.py",
    "scripts/v025/m0_core_session_protocol.py",
    "scripts/v025/m0_exact_boundary_contract.py",
    "scripts/v025/m0_postlock_census.py",
    "scripts/v025/m0_three_window_executor.py",
    "scripts/v025/m0_w1_fast_pair.py",
    "scripts/v025/m0_window_parent.py",
    "scripts/v025/nsys_export.py",
    "scripts/v025/nvtx_exclusive.py",
    "scripts/v025/parse_profiler.py",
    "scripts/v025/prepare_m0_matched_pair.py",
    "scripts/v025/run_cpu_arm.py",
    "scripts/v025/run_gpu_arm.py",
    "scripts/v025/step1_driver.py",
    "scripts/v025/step1_stub.py",
    "tests/v025/test_m0_exact_boundary.py",
    "tests/v025/test_m0_held_session_repair.py",
    "tests/v025/test_m0_step1_evidence_boundary.py",
    "tests/v025/test_m0_three_window_executor.py",
)

# Each prior independent-review attack is tied to current JUnit cases.  The
# builder rejects a missing, skipped, failed, or errored match.
ATTACK_TESTS: dict[str, tuple[str, ...]] = {
    "B1_real_callback_and_canonical_w1": (
        "test_real_w1_c1_w2_w3_callbacks_reach_w3_with_only_children_injected",
        "test_separate_pair_is_prepared_as_exact_held_session_callbacks",
        "test_c1_failure_makes_w2_and_w3_mechanically_unreachable",
    ),
    "B2_outer_owner_and_terminal_path": (
        "test_frozen_amendment6_commands_parse_byte_for_byte",
        "test_cpu_preflight_failure_never_launches_wrapper_or_spends_receipt",
        "test_cpu_preflight_binds_the_exact_mpi_and_wrf_executables",
        "test_outer_owner_source_mechanically_reaches_release_posts_and_finalizer",
        "test_legacy_outer_graph_refuses_without_reading_receipt",
    ),
    "M1_deferred_pallas_and_dump_unreachable": (
        "test_census_hlo_gate_defers_peak_live_and_ignores_the_future_sidecar",
        "test_m0_core_gate_subset_excludes_the_deferred_diagnostic",
        "test_deferred_evidence_cannot_be_waived_or_promoted",
    ),
    "M2_exact_lock_budget_and_absolute_deadline": (
        "test_wrong_session_lock_label_is_refused_before_receipt_spend",
        "test_global_deadline_is_one_monotonic_4500s_budget",
        "test_worst_branch_budget_mutation_is_refused_before_coordination",
        "test_deadline_exhaustion_suppresses_every_later_stage",
    ),
    "M3_process_group_kill_reap_and_zero_orphan": (
        "test_registered_process_group_kills_and_reaps_descendants",
        "test_preemption_or_stage_exception_still_sweeps_and_fails_closed",
    ),
    "M4_stale_live_buffer_cannot_promote_m0_core": (
        "test_census_hlo_gate_defers_peak_live_and_ignores_the_future_sidecar",
        "test_aggregate_temporary_bytes_can_never_be_relabelled_as_peak",
        "test_m0_core_verdict_is_reachable_while_deferred_evidence_is_missing",
    ),
    "M5_session_root_and_raw_identity": (
        "test_m0_core_manifest_mutations_fail_closed",
        "test_finalizer_stale_hash_is_refused",
        "test_finalizer_source_rehashes_session_and_all_retained_raw_artifacts",
        "test_mutated_prelock_revalidation_is_refused_before_receipt_spend",
        "test_prelock_revalidation_accepts_symlinked_tools_but_rejects_retarget",
    ),
    "M6_one_sqlite_and_exact_preambles": (
        "test_stale_sqlite_notice_is_refused_by_the_parsers",
        "test_verified_preamble_is_tolerated_but_unknown_preamble_is_not",
        "test_export_forces_a_fresh_private_sqlite_and_refuses_a_reused_one",
    ),
    "M7_current_plan_junit_and_normalized_commands": (
        "test_normalized_commands_are_identical_across_worktrees_and_python_paths",
        "test_executable_dry_run_of_the_whole_session_in_a_fresh_process",
        "test_session_source_contract_still_holds",
    ),
}

REQUIRED_NON_ATTACK_TESTS = (
    "test_prepared_cache_uses_zero_payload_copy_hardlinks_and_read_on_hit_policy",
    "test_session_modules_import_no_accelerator_in_a_hostile_environment",
    "test_plan_freezes_three_unique_capped_windows_and_exact_commands",
)


class EvidenceError(RuntimeError):
    """The compact proof cannot be built or reproduced honestly."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _git(*args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
        timeout=30.0,
    )
    if check and completed.returncode != 0:
        raise EvidenceError(
            f"git {' '.join(args)} failed: {completed.stderr[-500:]}"
        )
    return completed.stdout.strip()


def _accelerator_imports() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if name.split(".", 1)[0] in {"jax", "jaxlib", "gpuwrf"}
    )


def _junit(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError) as exc:
        raise EvidenceError(f"cannot parse current JUnit: {exc}") from exc
    cases: list[dict[str, str]] = []
    for case in root.iter("testcase"):
        if case.find("failure") is not None:
            status = "FAILED"
        elif case.find("error") is not None:
            status = "ERROR"
        elif case.find("skipped") is not None:
            status = "SKIPPED"
        else:
            status = "PASSED"
        cases.append(
            {
                "classname": case.attrib.get("classname", ""),
                "name": case.attrib.get("name", ""),
                "status": status,
            }
        )
    summary = {
        "path": str(Path(path).resolve()),
        "sha256": _sha256_file(path),
        "tests": len(cases),
        "passed": sum(case["status"] == "PASSED" for case in cases),
        "failures": sum(case["status"] == "FAILED" for case in cases),
        "errors": sum(case["status"] == "ERROR" for case in cases),
        "skipped": sum(case["status"] == "SKIPPED" for case in cases),
        "skipped_cases": [
            f"{case['classname']}::{case['name']}"
            for case in cases
            if case["status"] == "SKIPPED"
        ],
    }
    summary["green"] = (
        summary["failures"] == 0
        and summary["errors"] == 0
        and summary["passed"] + summary["skipped"] == summary["tests"]
    )
    return summary, cases


def _base_test_name(name: str) -> str:
    return name.split("[", 1)[0]


def _required_outcomes(
    cases: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    requirements = {
        **ATTACK_TESTS,
        "required_non_attack_controls": REQUIRED_NON_ATTACK_TESTS,
    }
    outcomes: dict[str, dict[str, Any]] = {}
    for category, test_names in requirements.items():
        category_cases: dict[str, list[dict[str, str]]] = {}
        for test_name in test_names:
            matches = [
                case
                for case in cases
                if _base_test_name(case["name"]) == test_name
            ]
            if not matches:
                raise EvidenceError(
                    f"current JUnit has no case for {category}: {test_name}"
                )
            if any(case["status"] != "PASSED" for case in matches):
                raise EvidenceError(
                    f"required current-run mutation/control is not green: "
                    f"{category}: {test_name}: {matches}"
                )
            category_cases[test_name] = matches
        outcomes[category] = {
            "status": "REPRODUCED_AND_REJECTED",
            "tests": category_cases,
            "case_count": sum(len(value) for value in category_cases.values()),
        }
    return outcomes


def _current_plan() -> dict[str, Any]:
    plan = executor.build_plan()
    session_plan = executor.held_session_plan()
    return {
        "plan_gate": exact_contract.validate_plan(plan),
        "source_gate": exact_contract.validate_sources(),
        "manager_session_command_normalized":
            session_plan["manager_session_command_normalized"],
        "manager_session_command_sha256":
            session_plan["manager_session_command_sha256"],
        "held_wrapper_command_normalized":
            session_plan["held_wrapper_command_normalized"],
        "held_wrapper_command_sha256":
            session_plan["held_wrapper_command_sha256"],
        "held_budget": session_plan["held_budget"],
        "stage_order": session_plan["stage_order"],
        "receipt_spends": session_plan["receipt_spends"],
        "lock_acquisitions": session_plan["lock_acquisitions"],
        "compiler_peak_live_buffers":
            session_plan["m0_core_evidence_prerequisites"][
                "compiler_peak_live_buffers"
            ],
    }


def _preflight_identity_probe() -> dict[str, Any]:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix="m0-amendment6-preflight-proof-"
    ) as temporary:
        identity_path = Path(temporary) / "identity.json"
        identity = executor.build_session_preflight_identity(
            output_path=identity_path,
            receipt_path=REPO / executor.SESSION_RECEIPT,
        )
        revalidation = executor.revalidate_session_preflight_inputs(identity)
    return {
        "status": identity["status"],
        "revalidation_status": revalidation["status"],
        "src_gpuwrf_tree": identity["src_gpuwrf_tree"],
        "source_inputs_sha256": identity["source_inputs_sha256"],
        "config": identity["config"],
        "input_manifest_sha256": identity["input_manifest_sha256"],
        "cache": identity["cache"],
        "tools": identity["tools"],
        "device_action": (
            identity["device_action"] or revalidation["device_action"]
        ),
        "wallclock_seconds": time.monotonic() - started,
    }


def _stable_preflight(probe: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in probe.items()
        if key != "wallclock_seconds"
    }


def _production_tree() -> dict[str, Any]:
    tree = _git("rev-parse", "HEAD:src/gpuwrf")
    dirty = _git("status", "--porcelain", "--", "src/gpuwrf")
    if tree != FROZEN_SRC_TREE or dirty:
        raise EvidenceError(
            f"src/gpuwrf is not frozen: tree={tree}, dirty={dirty!r}"
        )
    return {
        "src_gpuwrf_tree": tree,
        "src_gpuwrf_dirty": dirty,
        "frozen": True,
    }


def _atomic_json_replace(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def build(*, junit_path: Path, output_path: Path) -> dict[str, Any]:
    imports = _accelerator_imports()
    if imports:
        raise EvidenceError(
            f"proof builder imported accelerator roots: {imports[:8]}"
        )
    junit, cases = _junit(junit_path)
    if not junit["green"] or junit["skipped"] != 2:
        raise EvidenceError(
            f"full CPU JUnit is not 0-failure/0-error/2-skip: {junit}"
        )
    payload = {
        "schema": SCHEMA,
        "status": "PASS",
        "outcome": OUTCOME,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "review_scope": "Amendment-6 session-fatal CPU repair only",
        "device_policy": {
            "device_touched": False,
            "device_queries": [],
            "device_imports": [],
            "gpu_compiles": [],
            "gpu_runs": [],
            "coordination_requests": [],
            "locks_acquired": [],
            "receipts_read": [],
            "receipts_spent": [],
            "full_cpu_oracles": [],
        },
        "cpu_test_environment": {
            "affinity": "16-27",
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "junit": junit,
        "attack_outcomes": _required_outcomes(cases),
        "plan": _current_plan(),
        "preflight_identity_probe": _preflight_identity_probe(),
        "production_tree": _production_tree(),
        "source_hashes": {
            name: _sha256_file(REPO / name) for name in REVIEW_SOURCES
        },
        "performance_repairs": {
            "runtime_primary": True,
            "cpu_wrf_preflight": {
                "outside_lock": True,
                "hard_cap_seconds": core_session.CPU_PREFLIGHT_TIMEOUT_SECONDS,
            },
            "held_budget": core_session.validate_held_budget(),
            "cache_pair": {
                "payload_copies": 0,
                "method": "same-filesystem hard links plus read-on-hit policy",
                "ram_scaling": "O(1)",
                "disk_scaling": "O(seed bytes + per-arm misses)",
            },
            "nsys_sqlite_exports_per_w2_analysis": 1,
            "compiler_dump_flags_in_m0_core": [],
            "native_pallas_reachable_from_m0_core": False,
            "compiler_peak_live_buffers": {
                "status": "MISSING",
                "waived": False,
                "owner": "M1 same-set candidate diagnosis / M2 backend bake-off",
            },
        },
        "m0_core_partition": {
            "blocking_gates": list(postlock.M0_CORE_CENSUS_GATES),
            "deferred_gates": list(postlock.M0_DEFERRED_CENSUS_GATES),
            "deferred_fields": dict(postlock.M0_DEFERRED_FIELDS),
            "may_open_m1": False,
            "authorizes_gpu_work": False,
        },
        "remaining_device_evidence": {
            "W1_W2_W3_runtime_correctness_and_safety": "MISSING",
            "GPU_VRAM_and_host_RSS": "MISSING",
            "profiler_perturbation": "MISSING",
            "multi_size_VRAM_scaling": "MISSING_DEFERRED_RELEASE_GATE",
        },
    }
    payload["proof_sha256"] = _canonical_sha256(payload)
    _atomic_json_replace(output_path, payload)
    return payload


def validate(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("outcome") != OUTCOME
    ):
        raise EvidenceError("proof schema/status/outcome changed")
    expected_hash = _canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "proof_sha256"
        }
    )
    if payload.get("proof_sha256") != expected_hash:
        raise EvidenceError("compact proof content hash changed")
    policy = payload.get("device_policy") or {}
    if policy.get("device_touched") is not False or any(
        policy.get(field)
        for field in (
            "device_queries",
            "device_imports",
            "gpu_compiles",
            "gpu_runs",
            "coordination_requests",
            "locks_acquired",
            "receipts_read",
            "receipts_spent",
            "full_cpu_oracles",
        )
    ):
        raise EvidenceError("CPU proof claims a forbidden device/coordination action")

    junit_path = Path(str((payload.get("junit") or {}).get("path", "")))
    junit, cases = _junit(junit_path)
    if junit != payload.get("junit"):
        raise EvidenceError("current JUnit hash/counts/skips changed")
    outcomes = _required_outcomes(cases)
    if outcomes != payload.get("attack_outcomes"):
        raise EvidenceError("current mutation outcomes changed")
    if _current_plan() != payload.get("plan"):
        raise EvidenceError("current normalized plan/source gate changed")
    current_preflight = _preflight_identity_probe()
    if _stable_preflight(current_preflight) != _stable_preflight(
        payload.get("preflight_identity_probe") or {}
    ):
        raise EvidenceError("current source/input/config/cache/tool identity changed")
    if _production_tree() != payload.get("production_tree"):
        raise EvidenceError("frozen production tree changed")
    for name, digest in (payload.get("source_hashes") or {}).items():
        if name not in REVIEW_SOURCES or _sha256_file(REPO / name) != digest:
            raise EvidenceError(f"review source changed after proof: {name}")
    if set((payload.get("source_hashes") or {})) != set(REVIEW_SOURCES):
        raise EvidenceError("review source inventory changed")
    if _accelerator_imports():
        raise EvidenceError("validator imported an accelerator root")
    return {
        "status": "PASS",
        "outcome": OUTCOME,
        "proof_sha256": payload["proof_sha256"],
        "junit_tests": junit["tests"],
        "junit_skipped": junit["skipped"],
        "attack_categories": len(ATTACK_TESTS),
        "sources_verified": len(REVIEW_SOURCES),
        "current_plan_rederived": True,
        "current_mutations_rederived": True,
        "current_junit_rederived": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path, default=DEFAULT_JUNIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.validate_only:
            summary = validate(args.output)
        else:
            payload = build(
                junit_path=args.junit,
                output_path=args.output,
            )
            summary = {
                "status": "PASS",
                "outcome": payload["outcome"],
                "proof_sha256": payload["proof_sha256"],
                "output": str(args.output),
                "junit_tests": payload["junit"]["tests"],
                "junit_skipped": payload["junit"]["skipped"],
            }
    except Exception as exc:  # noqa: BLE001 - terminal proof is fail closed
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "reason": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
