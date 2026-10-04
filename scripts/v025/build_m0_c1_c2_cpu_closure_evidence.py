#!/usr/bin/env python3
"""Build the terminal, CPU-only M0 C1/C2 closure proof and manifest.

This process is deliberately JAX/gpuwrf-free.  It consumes the expensive exact
CPU oracle, verifies the exact result-to-wrfout binding, compares that output to
one fresh CPU-WRF arm with the existing FAST comparator, and executes the real
post-lock exporter/census path over the incomplete W1b report.  It never turns
CPU observations or adversarial fixtures into device evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
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

import m0_c1_c2_cpu_proofs as cpu_proofs  # noqa: E402
import m0_exact_boundary_contract as exact_contract  # noqa: E402
import m0_three_window_executor as executor  # noqa: E402
import run_fast_pair  # noqa: E402


STATUS = "CPU_C1_C2_GREEN_GPU_WINDOWS_MISSING"
EXACT_STATUS = "CPU_C1_GREEN_C2_PENDING"
PRODUCTION_TREE = "a6885ceded260df2f5777d7366d75a5d38947cb7"
C1_SCHEMA = "wrf_gpu2.v025.m0.c1_cpu_closure.v1"
C2_SCHEMA = "wrf_gpu2.v025.m0.c2_cpu_closure.v1"
CLOSURE_SCHEMA = "wrf_gpu2.v025.m0.c1_c2_cpu_closure.v1"
MANIFEST_SCHEMA = "wrf_gpu2.v025.m0.c1_c2_cpu_closure_manifest.v1"
DEFAULT_EXACT = REPO / "proofs/v025/m0/m0_exact_boundary_cpu_evidence.json"
DEFAULT_CPU_ARM = REPO / "proofs/v025/m0/m0_c1_fresh_cpu_wrf_arm.json"
DEFAULT_C1 = REPO / "proofs/v025/m0/m0_c1_cpu_evidence.json"
DEFAULT_C2 = REPO / "proofs/v025/m0/m0_c2_cpu_evidence.json"
DEFAULT_CLOSURE = REPO / "proofs/v025/m0/m0_c1_c2_cpu_closure_evidence.json"
DEFAULT_MANIFEST = (
    REPO / "proofs/v025/m0/m0_c1_c2_cpu_closure_manifest.json"
)


class ClosureError(RuntimeError):
    """A frozen C1/C2 acceptance condition was absent or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ClosureError(message)


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    path = Path(path)
    _require(
        not path.is_symlink() and path.is_file(),
        f"proof dependency is absent/not a regular file: {path}",
    )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClosureError(f"cannot read JSON proof {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"JSON proof is not an object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ClosureError(
            f"git {' '.join(arguments)} failed rc={completed.returncode}: "
            f"{completed.stderr[-1000:]}"
        )
    return completed.stdout.strip()


def _accelerator_modules() -> list[str]:
    roots = ("jax", "jaxlib", "gpuwrf")
    return sorted(
        name
        for name in sys.modules
        if any(name == root or name.startswith(f"{root}.") for root in roots)
    )


def _machine_identity() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "JAX_PLATFORMS",
                "CUDA_VISIBLE_DEVICES",
                "XLA_FLAGS",
                "GPUWRF_JAX_CACHE",
                "GPUWRF_JAX_CACHE_DIR",
                "GPUWRF_JAX_CACHE_LOCK",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "accelerator_modules_imported": _accelerator_modules(),
    }


def _production_identity() -> dict[str, Any]:
    status = _git("status", "--porcelain", "--", "src/gpuwrf")
    tree = _git("rev-parse", "HEAD:src/gpuwrf")
    _require(not status, f"src/gpuwrf has local changes: {status}")
    _require(
        tree == PRODUCTION_TREE,
        f"src/gpuwrf tree changed: expected {PRODUCTION_TREE}, observed {tree}",
    )
    return {
        "status": "PASS",
        "src_gpuwrf_tree": tree,
        "working_tree_changes": [],
    }


def _source_contract_matches(
    exact_proof: dict[str, Any],
) -> dict[str, Any]:
    current = exact_contract.validate_sources()
    observed = (exact_proof.get("source_contract") or {}).get(
        "source_sha256"
    )
    _require(
        isinstance(observed, dict) and observed == current["source_sha256"],
        "exact proof source manifest differs from the closure source manifest",
    )
    return current


def _validate_exact_proof(
    path: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    _require(
        payload.get("schema")
        == "wrf_gpu2.v025.m0.exact_boundary_cpu_evidence.v1"
        and payload.get("status") == EXACT_STATUS,
        "exact CPU oracle schema/status is not the C1 terminal observation",
    )
    _require(
        payload.get("device_policy") == "closed"
        and payload.get("device_touched") is False
        and payload.get("device_queries") == []
        and payload.get("receipts_read") == []
        and payload.get("receipts_consumed") == [],
        "exact CPU oracle carries device action or coordination",
    )
    _require(
        payload.get("gpu_windows")
        == {"W1": "MISSING", "W2": "MISSING", "W3": "MISSING"}
        and payload.get("native_pallas_verdict") == "MISSING",
        "exact CPU oracle overclaims a device/Pallas verdict",
    )
    _require(
        (payload.get("production_identity") or {}).get("src_gpuwrf_tree")
        == PRODUCTION_TREE,
        "exact CPU oracle production tree differs",
    )
    _require(
        (payload.get("clock_accounting") or {}).get("status") == "PASS",
        "exact CPU oracle pure-readiness clock repair is not proven",
    )
    identity = payload.get("real_fast_cpu_identity") or {}
    checks = identity.get("checks") or {}
    required_checks = {
        "arguments_identical",
        "lowered_program_identical",
        "called_function_identical",
        "output_digest_identical",
        "returned_state_semantics_identical",
        "hours_identical",
    }
    _require(
        set(checks) == required_checks
        and all(checks[name] is True for name in required_checks),
        "one or more of the six exact real-FAST identities failed",
    )
    local = identity.get("local_exact_compiled") or {}
    backend = local.get("backend") or {}
    _require(
        local.get("status") == "OK"
        and backend.get("expected_platform") == "cpu"
        and backend.get("observed_platforms") == ["cpu"]
        and local.get("device_touched") is False,
        "exact local candidate is not the completed CPU-only invocation",
    )
    binding = exact_contract.validate_wrfout_binding(
        local, verify_file=True
    )
    matrix = payload.get("mutation_matrix") or {}
    _require(
        matrix.get("status") == "PASS"
        and isinstance(matrix.get("total"), int)
        and matrix["total"] >= 18
        and matrix.get("rejected") == matrix["total"],
        "exact-boundary mutation matrix is incomplete",
    )
    _source_contract_matches(payload)
    return {
        "status": "PASS",
        "proof": _artifact(path),
        "run_id": local["run_id"],
        "six_identity_checks": checks,
        "mutation_matrix": matrix,
        "wrfout_binding": binding,
        "local": local,
    }


def _validate_cpu_arm(
    path: Path,
    payload: dict[str, Any],
    *,
    exact_generated_at_utc: str,
) -> dict[str, Any]:
    _require(payload.get("status") == "OK", "fresh CPU-WRF arm failed")
    _require(
        payload.get("max_dom") == 1
        and payload.get("hours") == 1
        and payload.get("ranks") == 12
        and payload.get("cpu_list") == "16-27",
        "fresh CPU-WRF arm does not use the frozen 1h/12-rank envelope",
    )
    _require(
        (payload.get("finiteness") or {}).get("pass") is True
        and (payload.get("finiteness") or {}).get("non_finite_total") == 0,
        "fresh CPU-WRF arm has non-finite output",
    )
    try:
        exact_finished = datetime.fromisoformat(exact_generated_at_utc)
        cpu_started = datetime.fromisoformat(payload["started_at_utc"])
        cpu_finished = datetime.fromisoformat(payload["finished_at_utc"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ClosureError("fresh CPU-WRF arm timestamps are malformed") from exc
    _require(
        exact_finished.tzinfo is not None
        and cpu_started.tzinfo is not None
        and cpu_finished.tzinfo is not None
        and exact_finished < cpu_started < cpu_finished,
        "fresh CPU-WRF arm was not run after the exact oracle",
    )
    final = Path(payload["run_dir"]) / payload["final_wrfout"]
    observed_hash = _sha256_file(final)
    _require(
        not final.is_symlink()
        and final.is_file()
        and observed_hash == payload.get("final_wrfout_sha256"),
        "fresh CPU-WRF final output path/hash changed",
    )
    return {
        "status": "PASS",
        "proof": _artifact(path),
        "final_wrfout_path": str(final),
        "final_wrfout_bytes": final.stat().st_size,
        "final_wrfout_sha256": observed_hash,
    }


def _fixed_output_overhead(local: dict[str, Any]) -> dict[str, Any]:
    timing = local["wrfout"]["timing"]

    def seconds(start: str, end: str) -> float:
        return (
            timing[f"{end}_monotonic_ns"]
            - timing[f"{start}_monotonic_ns"]
        ) / 1e9

    return {
        "status": "PASS",
        "outside_readiness_and_integration_clocks": timing[
            "outside_readiness_and_integration_clocks"
        ],
        "materialization_seconds": seconds(
            "materialization_start", "materialization_end"
        ),
        "production_prepare_seconds": seconds(
            "prepare_start", "prepare_end"
        ),
        "netcdf_write_seconds": seconds("write_start", "write_end"),
        "inspection_and_hash_seconds": seconds(
            "inspection_start", "inspection_end"
        ),
        "total_added_fixed_seconds": seconds(
            "materialization_start", "inspection_end"
        ),
        "integration_end_precedes_output_start": (
            local["timing"]["integration_end_monotonic_ns"]
            <= timing["materialization_start_monotonic_ns"]
        ),
    }


def _compare_outputs(
    *,
    exact_proof: dict[str, Any],
    cpu_arm: dict[str, Any],
) -> dict[str, Any]:
    local = exact_proof["real_fast_cpu_identity"]["local_exact_compiled"]
    cpu_payload = dict(cpu_arm)
    cpu_payload["final_wrfout_path"] = str(
        Path(cpu_arm["run_dir"]) / cpu_arm["final_wrfout"]
    )
    cpu_record = run_fast_pair.ArmRecord(
        kind="cpu",
        run_id=f"{cpu_arm['label']}-fresh-cpu-wrf",
        started_at_utc=cpu_arm["started_at_utc"],
        finished_at_utc=cpu_arm["finished_at_utc"],
        status="OK",
        payload=cpu_payload,
    )
    # ``compare_arms`` is the existing production comparator and names its
    # candidate slot "gpu".  Here that slot is intentionally populated by the
    # exact CPU oracle solely to record the C1 delta atlas; this is not W1 or
    # device evidence.
    candidate_record = run_fast_pair.ArmRecord(
        kind="gpu",
        run_id=local["run_id"],
        started_at_utc=exact_proof["generated_at_utc"],
        finished_at_utc=exact_proof["generated_at_utc"],
        status="OK",
        payload={
            "wrfout": local["wrfout"],
            "execution_platform": "cpu",
            "evidence_role": "exact-compiled-CPU-candidate",
        },
    )
    raw = run_fast_pair.compare_arms(cpu_record, candidate_record)
    _require(
        raw.get("variables_compared", 0) > 0
        and raw.get("total_nonfinite") == 0,
        "existing FAST comparator produced no finite delta atlas",
    )
    metrics = raw["per_variable"]

    def top(field: str) -> list[dict[str, Any]]:
        return [
            {"variable": name, field: values[field]}
            for name, values in sorted(
                metrics.items(),
                key=lambda item: float(item[1][field]),
                reverse=True,
            )[:20]
        ]

    return {
        "status": "PASS",
        "comparator": "scripts/v025/run_fast_pair.py:compare_arms",
        "candidate_parameter_name_in_existing_api": "gpu",
        "candidate_actual_execution_platform": "cpu",
        "candidate_evidence_role": "exact-compiled-CPU-output",
        "is_gpu_or_w1_evidence": False,
        "tolerance_applied": None,
        "interpretation": (
            "raw M0 delta atlas only; no production-accuracy conclusion"
        ),
        "delta_atlas": raw,
        "top_20_by_rmse": top("rmse"),
        "top_20_by_max_abs": top("max_abs"),
        "top_20_by_max_rel": top("max_rel"),
    }


def _optional_rss(path: Path | None, *, expected_run_id: str) -> Any:
    if path is None:
        return "MISSING"
    payload = _load_object(path)
    _require(
        payload.get("schema") == "wrf_gpu2.v025.m0.process_tree_rss.v1"
        and payload.get("status") == "PASS"
        and payload.get("run_id") == expected_run_id
        and payload.get("device_action") is False
        and payload.get("peak_process_tree_rss_bytes", 0) > 0,
        f"host RSS proof is invalid: {path}",
    )
    return {"artifact": _artifact(path), "measurement": payload}


def _junit(path: Path | None) -> Any:
    if path is None:
        return "MISSING"
    artifact = _artifact(path)
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ClosureError(f"JUnit XML is malformed: {path}") from exc
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals = {
        name: sum(int(float(suite.attrib.get(name, "0"))) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    _require(
        totals["tests"] > 0
        and totals["failures"] == 0
        and totals["errors"] == 0,
        f"full tests/v025 JUnit is not green: {totals}",
    )
    return {"artifact": artifact, "totals": totals}


def validate_terminal_artifacts(
    *,
    closure_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Independently re-hash the terminal closure and every manifest member."""

    _require(
        not _accelerator_modules(),
        "terminal validator imported accelerator/model roots",
    )
    closure = _load_object(closure_path)
    _require(
        closure.get("schema") == CLOSURE_SCHEMA
        and closure.get("status") == STATUS
        and closure.get("device_touched") is False
        and closure.get("device_queries") == []
        and closure.get("receipts_read") == []
        and closure.get("receipts_consumed") == []
        and closure.get("gpu_windows")
        == {"W1": "MISSING", "W2": "MISSING", "W3": "MISSING"}
        and closure.get("m0_complete") is False,
        "terminal closure status/device policy is ineligible",
    )
    expected_closure_hash = _canonical_sha256(
        {
            key: value
            for key, value in closure.items()
            if key != "closure_sha256"
        }
    )
    _require(
        closure.get("closure_sha256") == expected_closure_hash,
        "terminal closure content hash changed",
    )
    _require(
        closure.get("source_contract") == exact_contract.validate_sources(),
        "terminal closure source manifest differs from current source",
    )
    _require(
        (closure.get("production_identity") or {}).get("src_gpuwrf_tree")
        == PRODUCTION_TREE,
        "terminal closure production tree changed",
    )
    for name in ("c1", "c2"):
        record = closure.get(name) or {}
        artifact = record.get("artifact") or {}
        path = Path(str(artifact.get("path", "")))
        _require(
            record.get("status") == "PASS"
            and _artifact(path) == artifact
            and _load_object(path).get("status") == "PASS",
            f"terminal {name.upper()} artifact changed or is not passing",
        )
    tests = closure.get("tests_v025") or {}
    test_totals = tests.get("totals") or {}
    _require(
        test_totals.get("tests", 0) > 0
        and test_totals.get("failures") == 0
        and test_totals.get("errors") == 0
        and _artifact(Path(tests["artifact"]["path"])) == tests["artifact"],
        "terminal full-suite proof changed or is not green",
    )

    manifest = _load_object(manifest_path)
    _require(
        manifest.get("schema") == MANIFEST_SCHEMA
        and manifest.get("status") == STATUS
        and manifest.get("device_action") is False,
        "terminal manifest schema/status changed",
    )
    expected_manifest_hash = _canonical_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key != "manifest_payload_sha256"
        }
    )
    _require(
        manifest.get("manifest_payload_sha256")
        == expected_manifest_hash,
        "terminal manifest payload hash changed",
    )
    files = manifest.get("files")
    _require(
        isinstance(files, list) and files,
        "terminal manifest file inventory is missing",
    )
    for artifact in files:
        _require(
            isinstance(artifact, dict)
            and _artifact(Path(str(artifact.get("path", "")))) == artifact,
            f"terminal manifest member changed: {artifact}",
        )
    _require(
        any(
            Path(record["path"]).resolve() == Path(closure_path).resolve()
            for record in files
        ),
        "terminal manifest does not bind the closure proof",
    )
    return {
        "status": "PASS",
        "closure_sha256": _sha256_file(closure_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "files_verified": len(files),
        "device_touched": False,
    }


def build(
    *,
    exact_path: Path,
    cpu_arm_path: Path,
    c1_output: Path,
    c2_output: Path,
    closure_output: Path,
    manifest_output: Path,
    w1b_report: Path,
    w1b_run_root: Path,
    exact_rss_path: Path | None,
    cpu_rss_path: Path | None,
    junit_path: Path | None,
) -> dict[str, Any]:
    _require(
        not _accelerator_modules(),
        "closure builder imported accelerator/model roots before analysis",
    )
    production = _production_identity()
    exact_payload = _load_object(exact_path)
    exact = _validate_exact_proof(exact_path, exact_payload)
    cpu_arm_payload = _load_object(cpu_arm_path)
    cpu_arm = _validate_cpu_arm(
        cpu_arm_path,
        cpu_arm_payload,
        exact_generated_at_utc=exact_payload["generated_at_utc"],
    )
    comparison = _compare_outputs(
        exact_proof=exact_payload,
        cpu_arm=cpu_arm_payload,
    )
    local = exact["local"]
    output_overhead = _fixed_output_overhead(local)
    _require(
        output_overhead["outside_readiness_and_integration_clocks"] is True
        and output_overhead["integration_end_precedes_output_start"] is True,
        "C1 output work entered the readiness/integration clock",
    )
    exact_rss = _optional_rss(
        exact_rss_path, expected_run_id=local["run_id"]
    )
    cpu_rss = _optional_rss(
        cpu_rss_path,
        expected_run_id=f"{cpu_arm_payload['label']}-fresh-cpu-wrf",
    )
    c1 = {
        "schema": C1_SCHEMA,
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device_action": False,
        "is_gpu_or_w1_evidence": False,
        "exact_boundary": {
            "proof": exact["proof"],
            "run_id": exact["run_id"],
            "six_identity_checks": exact["six_identity_checks"],
            "mutation_matrix": exact["mutation_matrix"],
            "case": local["case"],
            "call_identity": {
                "argument_identity": local["call"]["argument_identity"],
                "lowered_program_sha256": local["call"][
                    "lowered_program_sha256"
                ],
                "production_jit": local["production_binding"]["jit_identity"],
            },
            "result": local["result"],
            "wrfout": local["wrfout"],
        },
        "fresh_cpu_wrf": {
            **cpu_arm,
            "run": cpu_arm_payload,
        },
        "comparison": comparison,
        "performance_resource_audit": {
            "cpu_readiness_seconds": local["timing"]["readiness_seconds"],
            "cpu_integration_seconds": local["timing"]["integration_seconds"],
            "cpu_times_are_not_gpu_economy_evidence": True,
            "output_fixed_overhead": output_overhead,
            "stablehlo_encoding": exact_payload[
                "performance_implications"
            ]["stablehlo_encoding_ram_scaling"],
            "proof_hashing_ram_scaling": exact_payload[
                "performance_implications"
            ]["proof_hashing_ram_scaling"],
            "exact_oracle_process_tree_rss": exact_rss,
            "fresh_cpu_wrf_process_tree_rss": cpu_rss,
        },
        "frozen_gate_inventory": {
            "same_synchronized_exact_result": "PASS",
            "production_wrfout_adapter": "PASS",
            "atomic_no_replace": "PASS",
            "result_run_case_output_hash_binding": "PASS",
            "fresh_12_rank_cpu_wrf": "PASS",
            "existing_comparator_delta_atlas": "PASS",
            "gpu_window": "MISSING",
        },
    }
    _atomic_json(c1_output, c1)

    complete = cpu_proofs.complete_census_proof()
    attacks = cpu_proofs.census_attack_matrix()
    _require(
        complete.get("status") == "PASS"
        and complete.get("fixture_is_device_evidence") is False
        and attacks.get("status") == "PASS"
        and attacks.get("rejected") == attacks.get("total")
        and attacks.get("total", 0) >= 15,
        "C2 complete arithmetic or fail-closed matrix failed",
    )
    incomplete = cpu_proofs.run_incomplete_w1b_postlock(
        report_path=w1b_report,
        run_root=w1b_run_root,
    )
    _require(
        incomplete.get("status") == "PASS"
        and incomplete.get("path_status") == "BLOCKED_AS_REQUIRED"
        and (incomplete.get("analysis") or {}).get("status") == "BLOCKED"
        and (incomplete.get("analysis") or {}).get(
            "release_before_analysis"
        )
        is True,
        "real incomplete W1b post-lock path did not fail closed",
    )
    c2 = {
        "schema": C2_SCHEMA,
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device_action": False,
        "complete_fixture_is_device_evidence": False,
        "complete_adversarial_census": complete,
        "fail_closed_matrix": attacks,
        "real_incomplete_w1b_postlock_path": incomplete,
        "frozen_gate_inventory": {
            "release_before_analysis": "PASS",
            "fresh_jax_free_cpu_process": "PASS",
            "hash_bound_real_nsys_export_path": "PASS",
            "complete_49_operator_a6_census_arithmetic": "PASS",
            "launch_and_device_time_thresholds_independent": "PASS",
            "timestamped_in_range_transfer_audit": "PASS",
            "incomplete_real_w1b_remains_blocked": "PASS",
            "real_w2_device_census": "MISSING",
            "real_w3_clean_timing": "MISSING",
        },
    }
    _atomic_json(c2_output, c2)

    dry_graph = executor.dry_run_all()
    _require(
        dry_graph.get("status") == STATUS
        and dry_graph.get("device_touched") is False
        and dry_graph.get("gpu_evidence_status") == "MISSING"
        and dry_graph.get("receipts_consumed") == [],
        "CPU dry W1/W2/W3 graph overclaimed or touched device state",
    )
    tests = _junit(junit_path)
    _require(tests != "MISSING", "terminal closure requires full tests/v025 JUnit")
    source_contract = exact_contract.validate_sources()
    machine = _machine_identity()
    _require(
        machine["accelerator_modules_imported"] == [],
        "closure analysis process imported accelerator/model roots",
    )
    c1_artifact = _artifact(c1_output)
    c2_artifact = _artifact(c2_output)
    closure = {
        "schema": CLOSURE_SCHEMA,
        "status": STATUS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "CPU-only executable closure of C1 exact-result WRF output and "
            "C2 release-before-analysis export/census path"
        ),
        "device_policy": "closed",
        "device_touched": False,
        "device_queries": [],
        "receipts_read": [],
        "receipts_consumed": [],
        "gpu_windows": {"W1": "MISSING", "W2": "MISSING", "W3": "MISSING"},
        "native_pallas_verdict": "MISSING",
        "m0_complete": False,
        "production_identity": production,
        "machine": machine,
        "source_contract": source_contract,
        "c1": {"status": "PASS", "artifact": c1_artifact},
        "c2": {"status": "PASS", "artifact": c2_artifact},
        "cpu_dry_end_to_end_graph": dry_graph,
        "tests_v025": tests,
        "unresolved_device_evidence": [
            "W1 cached GPU arm and fresh CPU paired qualification",
            "W2 real complete profiled census and resource evidence",
            "W3 clean warm timing and profiler perturbation pair",
            "native Pallas result",
        ],
        "next_decision": (
            "independent different-model CPU review; only the manager may "
            "subsequently authorize W1"
        ),
        "closure_sha256": "COMPUTED_OVER_PAYLOAD_WITHOUT_THIS_FIELD",
    }
    closure["closure_sha256"] = _canonical_sha256(
        {key: value for key, value in closure.items() if key != "closure_sha256"}
    )
    _atomic_json(closure_output, closure)

    manifest_paths = [
        exact_path,
        cpu_arm_path,
        c1_output,
        c2_output,
        closure_output,
    ]
    if junit_path is not None:
        manifest_paths.append(junit_path)
    if exact_rss_path is not None:
        manifest_paths.append(exact_rss_path)
    if cpu_rss_path is not None:
        manifest_paths.append(cpu_rss_path)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": STATUS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [_artifact(path) for path in manifest_paths],
        "device_action": False,
    }
    manifest["manifest_payload_sha256"] = _canonical_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key != "manifest_payload_sha256"
        }
    )
    _atomic_json(manifest_output, manifest)
    return {
        "status": STATUS,
        "c1": c1_artifact,
        "c2": c2_artifact,
        "closure": _artifact(closure_output),
        "manifest": _artifact(manifest_output),
        "variables_compared": comparison["delta_atlas"][
            "variables_compared"
        ],
        "c2_attacks_rejected": attacks["rejected"],
        "c2_attacks_total": attacks["total"],
        "device_touched": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-proof", type=Path, default=DEFAULT_EXACT)
    parser.add_argument("--cpu-arm", type=Path, default=DEFAULT_CPU_ARM)
    parser.add_argument("--c1-output", type=Path, default=DEFAULT_C1)
    parser.add_argument("--c2-output", type=Path, default=DEFAULT_C2)
    parser.add_argument("--closure-output", type=Path, default=DEFAULT_CLOSURE)
    parser.add_argument(
        "--manifest-output", type=Path, default=DEFAULT_MANIFEST
    )
    parser.add_argument(
        "--w1b-report", type=Path, default=cpu_proofs.W1B_REPORT
    )
    parser.add_argument("--w1b-run-root", type=Path, default=None)
    parser.add_argument("--exact-rss", type=Path, default=None)
    parser.add_argument("--cpu-rss", type=Path, default=None)
    parser.add_argument("--junit", type=Path, default=None)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="re-hash an already-built closure/manifest without rebuilding",
    )
    args = parser.parse_args()
    try:
        if args.validate_only:
            summary = validate_terminal_artifacts(
                closure_path=args.closure_output,
                manifest_path=args.manifest_output,
            )
        else:
            if args.w1b_run_root is None or args.junit is None:
                parser.error(
                    "--w1b-run-root and --junit are required when building"
                )
            summary = build(
                exact_path=args.exact_proof,
                cpu_arm_path=args.cpu_arm,
                c1_output=args.c1_output,
                c2_output=args.c2_output,
                closure_output=args.closure_output,
                manifest_output=args.manifest_output,
                w1b_report=args.w1b_report,
                w1b_run_root=args.w1b_run_root,
                exact_rss_path=args.exact_rss,
                cpu_rss_path=args.cpu_rss,
                junit_path=args.junit,
            )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "error": f"{type(exc).__name__}: {exc}",
                    "device_touched": False,
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
