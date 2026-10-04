#!/usr/bin/env python3
"""Assemble the terminal CPU proof object for ADR-036 implementation."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/v025"))

import cpu_guard  # noqa: E402,F401
import m0_vram_sampler as mvs  # noqa: E402
import step1_driver as drv  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"proof input is not an object: {path}")
    return value


def _test_summary(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals = {
        key: sum(int(float(suite.attrib.get(key, "0"))) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    if totals["tests"] <= 0 or totals["failures"] or totals["errors"]:
        raise RuntimeError(f"CPU test gate is not green: {totals}")
    return totals


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {completed.stderr[-500:]}")
    return completed.stdout.strip()


def build(
    *,
    test_junit: Path,
    diff_validation: Path,
    overhead: Path,
    pair_plan: Path,
    output: Path,
) -> dict[str, Any]:
    test_junit = test_junit if test_junit.is_absolute() else REPO / test_junit
    diff_validation = (
        diff_validation
        if diff_validation.is_absolute()
        else REPO / diff_validation
    )
    overhead = overhead if overhead.is_absolute() else REPO / overhead
    pair_plan = pair_plan if pair_plan.is_absolute() else REPO / pair_plan
    output = output if output.is_absolute() else REPO / output
    tests = _test_summary(test_junit)
    semantic = _json(diff_validation)
    performance = _json(overhead)
    pair = _json(pair_plan)
    if semantic.get("status") != "PASS":
        raise RuntimeError("semantic diff validation did not pass")
    if performance.get("status") != "PASS":
        raise RuntimeError("default-path overhead gate did not pass")
    if pair.get("status") != "PREPARED_NOT_AUTHORISED_NOT_RUN":
        raise RuntimeError("matched profiler pair is not in the prepared state")
    if pair.get("profiler_gate_evidence_status") != "MISSING":
        raise RuntimeError("matched profiler pair must remain MISSING in this sprint")
    pair_cache = pair.get("prepared_cache") or {}
    cache_seed_identity = pair_cache.get("seed_identity")
    if (
        pair_cache.get("status")
        != "BYTE_IDENTICAL_SNAPSHOTS_PREPARED_WARM_HIT_UNPROVEN"
        or not isinstance(cache_seed_identity, dict)
    ):
        raise RuntimeError("matched pair has no byte-bound cache snapshots")
    external_caches: dict[str, dict[str, Any]] = {}
    for arm in ("profiled", "clean"):
        path = Path(pair[arm]["prepared_cache_path"])
        observed = mvs.directory_tree_identity(path)
        if (
            observed != cache_seed_identity
            or pair[arm]["prepared_cache_sha256"] != observed["sha256"]
        ):
            raise RuntimeError(f"{arm} prepared cache snapshot changed after preparation")
        external_caches[arm] = {"path": str(path), **observed}

    schema_path = REPO / "scripts/v025/m0_evidence_v1.schema.json"
    source_path = REPO / "src/gpuwrf/runtime/operational_mode.py"
    sampler_path = REPO / "scripts/v025/m0_vram_sampler.py"
    validator_path = REPO / "scripts/v025/validate_m0_evidence.py"
    changed_src = [
        line for line in _git(
            "diff", "--name-only", "a1fa459c", "--", "src/gpuwrf"
        ).splitlines() if line
    ]
    if changed_src != ["src/gpuwrf/runtime/operational_mode.py"]:
        raise RuntimeError(f"production source allowlist violated: {changed_src}")

    artifacts = {
        str(path.relative_to(REPO)): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in (
            test_junit,
            diff_validation,
            overhead,
            pair_plan,
            schema_path,
            source_path,
            sampler_path,
            validator_path,
        )
    }
    proof = {
        "schema": "wrf_gpu2.v025.m0.instrumentation_cpu_evidence.v1",
        "status": "CPU_IDENTITY_GREEN_GPU_EVIDENCE_MISSING",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manager_decision": "45b513aa",
        "adr": "ADR-036",
        "objective": (
            "default-off whole-integration evidence hook, same-process allocator "
            "sidecar, external total-residency sampler, semantic diff/schema gates"
        ),
        "device_policy": {
            "worker": "DevicePolicy=closed",
            "gpu_import_query_lock_request_test": "NONE",
            "gpu_commands_executed": [],
            "profiler_pair": "MISSING_NOT_RUN",
            "runtime_vram_measurement": "MISSING_NOT_RUN",
        },
        "production_diff": {
            "base": "a1fa459c",
            "changed_files": changed_src,
            "existing_files_changed": 1,
            "allowed_max_existing_files": 2,
            "physics_dycore_state_contract_output_public_api_changes": 0,
            "semantic_validation": semantic,
        },
        "cpu_tests": {
            **tests,
            "status": "PASS",
            "command": "python -m pytest -q tests/v025",
            "environment": (
                "taskset -c 0-3; JAX_PLATFORMS=cpu; CUDA_VISIBLE_DEVICES=''; "
                "OMP/OPENBLAS/MKL/NUMEXPR=1; XLA CPU Eigen threading disabled"
            ),
        },
        "gates": {
            "default_off_original_direct_call": "PASS",
            "default_off_no_range_or_artifact_or_allocator_query": "PASS",
            "enabled_default_numerical_digest_identity": "PASS",
            "range_after_staging_through_final_sync": "PASS",
            "allocator_read_after_range_same_result_device": "PASS",
            "atomic_no_replace_sidecar": "PASS",
            "run_pid_source_config_input_device_identity": "PASS",
            "monotonic_and_utc_enclosure": "PASS",
            "external_sampler_no_jax": "PASS",
            "absolute_and_baseline_subtracted_residency_schema": "PASS",
            "cadence_misses_process_tree_competing_contexts": "PASS",
            "schema_mutations_fail_closed": "PASS",
            "semantic_numerical_edit_mutation_rejected": "PASS",
            "hot_loop_callback_mutation_rejected": "PASS",
            "default_off_jitted_body_ast_identity": "PASS",
            "default_path_runtime_overhead": "PASS",
            "sampler_ram_scaling": "PASS_O_UNIQUE_PIDS",
            "compile_latency_program_identity": "PASS_JITTED_BODY_UNCHANGED",
            "profiler_matched_pair": "MISSING_PREPARED_SEPARATE_WINDOW",
            "real_vram_measurement": "MISSING_MANAGER_EXECUTOR_REQUIRED",
        },
        "performance": {
            "primary_runtime_gate": performance,
            "compile_latency": {
                "status": "PASS_BY_IDENTITY",
                "basis": (
                    "the entire pre-existing _run_forecast_operational_jit AST hash "
                    "is unchanged; the evidence envelope is outside the jitted body"
                ),
                "jitted_body_ast_sha256": semantic[
                    "default_off_jitted_body_ast_sha256"
                ],
            },
            "ram_scaling": {
                "status": "PASS",
                "sampler": (
                    "samples are aggregated online; memory is O(unique observed PIDs), "
                    "not O(samples or forecast length)"
                ),
                "forecast_default": "no sidecar/range/stats object is created",
            },
            "vram_scaling": {
                "status": "UNCHANGED_DEFAULT_PROGRAM",
                "basis": "jitted integration and original direct call are mechanically identical",
            },
        },
        "separate_profiler_pair": pair,
        "external_prepared_cache_snapshots": external_caches,
        "artifacts": artifacts,
        "unresolved": {
            "gpu_runtime_evidence": (
                "intentionally absent under DevicePolicy=closed; manager executor follows critic"
            ),
            "profiler_clean_clock": pair["timing_boundary"]["known_gap"],
        },
        "terminal_sprint_verdict": (
            "implementation CPU-terminal and critic-ready; M0 remains open because "
            "the real profiler/VRAM/device gates were intentionally not run"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    drv._atomic_json(output, proof)
    return proof


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-junit", required=True, type=Path)
    parser.add_argument("--diff-validation", required=True, type=Path)
    parser.add_argument("--overhead", required=True, type=Path)
    parser.add_argument("--pair-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    proof = build(
        test_junit=args.test_junit,
        diff_validation=args.diff_validation,
        overhead=args.overhead,
        pair_plan=args.pair_plan,
        output=args.output,
    )
    print(json.dumps({
        "status": proof["status"],
        "output": str(args.output),
        "sha256": _sha256(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
