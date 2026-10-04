#!/usr/bin/env python3
"""Build compact CPU-only proof objects for the ADR-036 gate repair."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import m0_stale_pair as stale_pair  # noqa: E402
import m0_three_window_executor as executor  # noqa: E402
import m0_vram_sampler as sampler  # noqa: E402
import validate_m0_instrumentation_diff as diff_validator  # noqa: E402


PROOF_ROOT = REPO / "proofs/v025/m0"
PLAN_PATH = PROOF_ROOT / "autotune0_three_window_plan.json"
DRY_PATH = PROOF_ROOT / "autotune0_three_window_dry_run.json"
FAILURE_PATH = PROOF_ROOT / "autotune0_three_window_failure_matrix.json"
VALIDATOR_PATH = PROOF_ROOT / "m0_instrumentation_diff_validation.json"
BENCHMARK_PATH = PROOF_ROOT / "m0_default_path_overhead.json"
JUNIT_PATH = PROOF_ROOT / "adr036_gate_repair_cpu_tests.xml"
DEFAULT_OUTPUT = PROOF_ROOT / "adr036_gate_repair_cpu_evidence.json"
ADR_PATH = REPO / ".agent/decisions/ADR-036-v025-m0-evidence-hook.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"proof is not an object: {path}")
    return value


def _load_attack_module():
    path = REPO / "tests/v025/test_m0_instrumentation_diff_attacks.py"
    spec = importlib.util.spec_from_file_location("adr036_attacks", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import attack module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _attack_matrix() -> dict[str, Any]:
    module = _load_attack_module()
    base = diff_validator._git_file(
        "a1fa459c", diff_validator.EXPECTED_CHANGED_FILE
    )
    current = (
        REPO / diff_validator.EXPECTED_CHANGED_FILE
    ).read_text(encoding="utf-8")
    attacks = module._mutations(current)
    results = []
    for attack_id, mutated in attacks.items():
        try:
            diff_validator.validate_sources(base, mutated)
        except diff_validator.DiffValidationError as exc:
            results.append(
                {
                    "attack_id": attack_id,
                    "status": "REJECTED",
                    "reason": str(exc),
                }
            )
        else:
            results.append(
                {
                    "attack_id": attack_id,
                    "status": "SURVIVED",
                    "reason": None,
                }
            )
    return {
        "required_attacks": [
            "A1",
            "A2",
            "A3",
            "A4",
            "A6",
            "A11",
            "A12r",
            "A14",
            "A15",
        ],
        "total": len(results),
        "rejected": sum(item["status"] == "REJECTED" for item in results),
        "rejection_fraction": (
            sum(item["status"] == "REJECTED" for item in results) / len(results)
        ),
        "results": results,
    }


def _stale_matrix() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def rejected(name: str, **kwargs: Any) -> None:
        try:
            stale_pair.reject_stale_prepared_pair(**kwargs)
        except stale_pair.StalePairError as exc:
            checks.append({"name": name, "status": "REJECTED", "reason": str(exc)})
        else:
            checks.append({"name": name, "status": "SURVIVED", "reason": None})

    for run_id in sorted(stale_pair.STALE_RUN_IDS):
        rejected(f"run_id:{run_id}", run_id=run_id)
    rejected(
        "cache_sha256",
        prepared_cache_sha256=next(iter(stale_pair.STALE_CACHE_SHA256)),
    )
    with tempfile.TemporaryDirectory(prefix="adr036-stale-proof-") as temporary:
        root = Path(temporary)
        for label, relative in (
            (
                "profiled_identity",
                "proofs/v025/m0/prepared_profiler_pair/profiled_identity.json",
            ),
            (
                "clean_identity",
                "proofs/v025/m0/prepared_profiler_pair/clean_identity.json",
            ),
            ("plan", "proofs/v025/m0/prepared_profiler_pair/plan.json"),
        ):
            completed = _run(["git", "show", f"5638bedd:{relative}"])
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr)
            path = root / f"{label}.json"
            path.write_text(completed.stdout, encoding="utf-8")
            if label == "plan":
                rejected(label, plan_path=path)
            else:
                rejected(label, identity_path=path)
    return {
        "checks": checks,
        "total": len(checks),
        "rejected": sum(item["status"] == "REJECTED" for item in checks),
        "all_refused_before_receipt_spend": all(
            item["status"] == "REJECTED"
            and "before receipt consumption" in item["reason"]
            for item in checks
        ),
    }


def _failure_matrix(plan: dict[str, Any]) -> dict[str, Any]:
    ordered = executor.dry_run_all()["executed_stub_stage_keys"]
    scenarios = []
    for index, failure_key in enumerate(ordered):
        payload = executor.dry_run_all(inject_failure=failure_key)
        observed: dict[str, str] = {}
        for phase in payload["results"]:
            for stage in phase["result"]["stages"]:
                observed[f"{phase['phase']}:{stage['name']}"] = stage["status"]
        later = ordered[index + 1 :]
        scenarios.append(
            {
                "failure_key": failure_key,
                "executed_stub_stage_keys": payload["executed_stub_stage_keys"],
                "failed_stage_status": observed[failure_key],
                "later_stage_count": len(later),
                "later_stages_all_suppressed": all(
                    observed[key] == "SUPPRESSED" for key in later
                ),
                "device_touched": payload["device_touched"],
                "receipts_consumed": payload["receipts_consumed"],
            }
        )
    return {
        "schema": "wrf_gpu2.v025.m0.autotune0_failure_matrix.v1",
        "status": executor.STATUS,
        "plan_sha256": _canonical_sha256(plan),
        "scenarios": scenarios,
        "all_failures_suppress_every_later_stage": all(
            item["failed_stage_status"] == "FAILED"
            and item["later_stages_all_suppressed"]
            and item["device_touched"] is False
            and item["receipts_consumed"] == []
            for item in scenarios
        ),
    }


def _deferred_path_state(plan: dict[str, Any]) -> dict[str, Any]:
    records = {}
    for name, record in plan["identity_and_cache_provenance"].items():
        if not isinstance(record, dict) or "path" not in record:
            continue
        path = Path(record["path"])
        resolved = path if path.is_absolute() else REPO / path
        records[name] = {
            "path": str(path),
            "exists": os.path.lexists(resolved),
            "required_status": record["status"]
            if "status" in record
            else record["pre_w1_required_state"],
        }
    return {
        "records": records,
        "all_deferred_objects_absent": all(
            item["exists"] is False for item in records.values()
        ),
    }


def _junit_summary(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    summary = {
        name: sum(int(suite.attrib.get(name, "0")) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    summary["path"] = str(path.relative_to(REPO))
    summary["sha256"] = _sha256(path)
    summary["green"] = summary["failures"] == 0 and summary["errors"] == 0
    return summary


def build() -> dict[str, Any]:
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("proof builder requires JAX_PLATFORMS=cpu")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", None):
        raise RuntimeError("proof builder requires CUDA_VISIBLE_DEVICES empty")

    plan = executor.build_plan()
    dry = executor.dry_run_all()
    failures = _failure_matrix(plan)
    _atomic_json(PLAN_PATH, plan)
    _atomic_json(DRY_PATH, dry)
    _atomic_json(FAILURE_PATH, failures)

    validation = diff_validator.validate_repository("a1fa459c", ADR_PATH)
    _atomic_json(VALIDATOR_PATH, validation)
    benchmark = _load_json(BENCHMARK_PATH)
    attacks = _attack_matrix()
    stale = _stale_matrix()
    deferred = _deferred_path_state(plan)
    junit = _junit_summary(JUNIT_PATH)

    production_diff = _run(
        [
            "git",
            "diff",
            "--exit-code",
            diff_validator.CANDIDATE_PRODUCTION_COMMIT,
            "--",
            "src/gpuwrf",
        ]
    )
    candidate_tree = _run(
        [
            "git",
            "rev-parse",
            f"{diff_validator.CANDIDATE_PRODUCTION_COMMIT}:src/gpuwrf",
        ]
    )
    tombstones = {
        name: _load_json(
            REPO / f"proofs/v025/m0/prepared_profiler_pair/{name}.json"
        )
        for name in ("plan", "profiled_identity", "clean_identity")
    }
    checks = {
        "cpu_pins_active": (
            os.environ.get("JAX_PLATFORMS") == "cpu"
            and os.environ.get("CUDA_VISIBLE_DEVICES") in ("", None)
        ),
        "full_v025_pytest_green": junit["green"] and junit["tests"] > 0,
        "validator_bound_pass": (
            validation["status"] == "PASS"
            and validation["candidate_production_commit"]
            == diff_validator.CANDIDATE_PRODUCTION_COMMIT
            and validation["candidate_src_tree"]
            == diff_validator.CANDIDATE_SRC_TREE
            and len(validation["validator_sha256"]) == 64
        ),
        "mandatory_attacks_100_percent_rejected": (
            attacks["total"] == 9
            and attacks["rejected"] == 9
            and attacks["rejection_fraction"] == 1.0
        ),
        "sampler_miss_threshold_five_percent": (
            sampler.MAX_SAMPLING_MISS_FRACTION == 0.05
        ),
        "sampler_limitations_machine_visible": len(sampler.SAMPLING_LIMITATIONS) >= 3,
        "benchmark_without_external_pythonpath_green": (
            benchmark.get("status") == "PASS"
        ),
        "production_tree_exact_candidate": (
            production_diff.returncode == 0
            and candidate_tree.returncode == 0
            and candidate_tree.stdout.strip() == diff_validator.CANDIDATE_SRC_TREE
        ),
        "stale_pair_fully_refused": stale["all_refused_before_receipt_spend"],
        "stale_tombstones_not_authorisable": all(
            item.get("status") == "MECHANICALLY_SUPERSEDED"
            and item.get("authorisation_eligible") is False
            for item in tombstones.values()
        ),
        "three_windows_capped": (
            len(plan["windows"]) == 3
            and all(
                window["deadline_seconds"] <= 1500.0
                and window["budget"]["headroom_seconds"] >= 0.0
                for window in plan["windows"]
            )
        ),
        "dry_windows_end_to_end": (
            dry["scenario"] == "ALL_STUBS_OK"
            and all(
                phase["result"]["ok"] is True for phase in dry["results"]
            )
        ),
        "failure_matrix_suppresses_later_stages": failures[
            "all_failures_suppress_every_later_stage"
        ],
        "fresh_pair_objects_absent_before_w1": deferred[
            "all_deferred_objects_absent"
        ],
        "no_device_or_receipt_action": (
            dry["device_touched"] is False
            and dry["device_queries"] == []
            and dry["device_imports"] == []
            and dry["receipts_read"] == []
            and dry["receipts_consumed"] == []
        ),
    }
    source_paths = [
        "scripts/v025/validate_m0_instrumentation_diff.py",
        "scripts/v025/m0_vram_sampler.py",
        "scripts/v025/m0_stale_pair.py",
        "scripts/v025/m0_three_window_executor.py",
        "scripts/v025/prepare_m0_matched_pair.py",
        "scripts/v025/benchmark_m0_default_path.py",
        "scripts/v025/validate_m0_evidence.py",
        "scripts/v025/baseline_census.py",
        "tests/v025/test_m0_instrumentation_diff_attacks.py",
        "tests/v025/test_m0_vram_sampler.py",
        "tests/v025/test_m0_three_window_executor.py",
        "tests/v025/test_m0_step1_evidence_boundary.py",
    ]
    proof_paths = [
        PLAN_PATH,
        DRY_PATH,
        FAILURE_PATH,
        VALIDATOR_PATH,
        BENCHMARK_PATH,
        JUNIT_PATH,
    ]
    return {
        "schema": "wrf_gpu2.v025.m0.adr036_gate_repair_cpu_evidence.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            executor.STATUS if all(checks.values()) else "CPU_REPAIR_FAILED"
        ),
        "gpu_evidence_status": "MISSING",
        "device_policy": {
            "policy": "closed",
            "device_actions": [],
            "coordination_requests": [],
            "receipt_reads": [],
            "receipt_consumptions": [],
        },
        "checks": checks,
        "pytest": junit,
        "attack_matrix": attacks,
        "sampler_policy": {
            "maximum_miss_fraction": sampler.MAX_SAMPLING_MISS_FRACTION,
            "baseline_counted_as_stream_sample": False,
            "jax_import_check": "sys.modules prefix scan before baseline or stream",
            "limitations": list(sampler.SAMPLING_LIMITATIONS),
        },
        "stale_pair": stale,
        "deferred_identity_cache_state": deferred,
        "three_window_plan": {
            "path": str(PLAN_PATH.relative_to(REPO)),
            "sha256": _sha256(PLAN_PATH),
            "canonical_sha256": _canonical_sha256(plan),
            "windows": [
                {
                    "id": window["id"],
                    "label": window["label"],
                    "run_id": window["run_id"],
                    "receipt": window["receipt"],
                    "budget": window["budget"],
                }
                for window in plan["windows"]
            ],
            "live_execution_status": plan["live_execution_status"],
        },
        "unresolved_gpu_measurement_boundaries": plan["timing_boundaries"],
        "production_identity": {
            "candidate_commit": diff_validator.CANDIDATE_PRODUCTION_COMMIT,
            "candidate_src_tree": candidate_tree.stdout.strip(),
            "working_diff_returncode": production_diff.returncode,
        },
        "benchmark": benchmark,
        "source_sha256": {
            path: _sha256(REPO / path) for path in source_paths
        },
        "proof_sha256": {
            str(path.relative_to(REPO)): _sha256(path) for path in proof_paths
        },
        "reproduce": {
            "environment": (
                "taskset -c 0-3 env JAX_PLATFORMS=cpu "
                "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
                "OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
                "XLA_FLAGS=--xla_cpu_multi_thread_eigen=false"
            ),
            "tests": (
                "<USER_HOME>/miniconda3/bin/python -m pytest -q tests/v025 "
                "--junitxml=proofs/v025/m0/adr036_gate_repair_cpu_tests.xml"
            ),
            "builder": (
                "<USER_HOME>/miniconda3/bin/python "
                "scripts/v025/build_adr036_gate_repair_evidence.py"
            ),
            "external_pythonpath_required": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    value = build()
    _atomic_json(args.output, value)
    print(
        json.dumps(
            {
                "path": str(args.output),
                "status": value["status"],
                "sha256": _sha256(args.output),
            },
            indent=2,
        )
    )
    return 0 if value["status"] == executor.STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
