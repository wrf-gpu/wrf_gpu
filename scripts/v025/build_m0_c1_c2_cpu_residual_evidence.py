#!/usr/bin/env python3
"""Build the amendment-authorized terminal C1/C2 CPU residual handoff.

The final admissible full CPU oracle (r8) reached the post-integration
publication boundary and then correctly refused a harness-precreated output
directory.  Manager commit ``78aa07a9`` forbids another oracle.  Consequently
this builder records C1 as a blocking runtime residual, proves the repaired
protocol with static/fixture/mutation gates, closes C2's CPU path, and emits the
exact terminal handoff status without claiming C1, M0, or any GPU window green.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_m0_c1_c2_cpu_closure_evidence as common  # noqa: E402
import build_m0_exact_boundary_evidence as exact_builder  # noqa: E402
import m0_c1_c2_cpu_proofs as cpu_proofs  # noqa: E402
import m0_core_session_protocol as core_session  # noqa: E402
import m0_exact_boundary_contract as exact_contract  # noqa: E402
import m0_long_run_controls as longrun  # noqa: E402
import m0_three_window_executor as executor  # noqa: E402


STATUS = "CPU_C1_C2_GREEN_GPU_WINDOWS_MISSING"
C1_STATUS = "BLOCKED_RUNTIME_RESIDUAL"
C2_STATUS = "PASS"
R8_RUN_ID = "m0-exact-boundary-real-fast-cpu-20260728-r8"
R8_ROOT = Path(
    "<DATA_ROOT>/wrf_gpu2/v025/m0/cpu_c1_c2_closure_20260728_r8"
)
R8_STDERR = R8_ROOT / "exact_oracle.stderr.log"
R8_STDOUT = R8_ROOT / "exact_oracle.stdout.log"
R8_TIME = R8_ROOT / "exact_oracle.time"
R8_RSS = R8_ROOT / "ineligible_precreated_wrfout_process_tree_rss.json"
ACCEPTED_EXACT = REPO / "proofs/v025/m0/m0_exact_boundary_cpu_evidence.json"
ACCEPTED_EXACT_SHA256 = (
    "ac596a79680f1ce41b88e311d97507ed199bdb1feb587f323ca89baf7cfcd16e"
)
MANAGER_AMENDMENT_COMMIT = "9c53ce62"
MANAGER_STOP_COMMIT = "78aa07a9"
PRODUCTION_TREE = "a6885ceded260df2f5777d7366d75a5d38947cb7"
DEFAULT_C1 = REPO / "proofs/v025/m0/m0_c1_cpu_evidence.json"
DEFAULT_C2 = REPO / "proofs/v025/m0/m0_c2_cpu_evidence.json"
DEFAULT_CLOSURE = (
    REPO / "proofs/v025/m0/m0_c1_c2_cpu_closure_evidence.json"
)
DEFAULT_MANIFEST = (
    REPO / "proofs/v025/m0/m0_c1_c2_cpu_closure_manifest.json"
)


class ResidualError(RuntimeError):
    """A residual, amendment, or proof binding failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ResidualError(message)


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResidualError(f"cannot read JSON {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"JSON is not an object: {path}")
    return payload


def _manager_object(commit: str, path: str) -> dict[str, Any]:
    common._git("cat-file", "-e", f"{commit}^{{commit}}")
    text = common._git("show", f"{commit}:{path}")
    return {
        "commit": common._git("rev-parse", commit),
        "path": path,
        "git_blob_oid": common._git("rev-parse", f"{commit}:{path}"),
        "sha256": common.hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def accepted_exact_boundary(path: Path = ACCEPTED_EXACT) -> dict[str, Any]:
    payload = _load(path)
    artifact = common._artifact(path)
    _require(
        artifact["sha256"] == ACCEPTED_EXACT_SHA256,
        "accepted pre-C1 exact proof hash changed",
    )
    _require(
        payload.get("schema")
        == "wrf_gpu2.v025.m0.exact_boundary_cpu_evidence.v1"
        and payload.get("status")
        == "CPU_EXACT_BOUNDARY_GREEN_GPU_WINDOWS_MISSING",
        "accepted pre-C1 exact proof schema/status changed",
    )
    checks = (payload.get("real_fast_cpu_identity") or {}).get("checks") or {}
    _require(
        len(checks) == 6 and all(value is True for value in checks.values()),
        "accepted pre-C1 exact proof lost a six-way identity",
    )
    mutation = payload.get("mutation_matrix") or {}
    _require(
        mutation.get("status") == "PASS"
        and mutation.get("rejected") == mutation.get("total")
        and mutation.get("total", 0) >= 18,
        "accepted pre-C1 exact mutation matrix changed",
    )
    _require(
        (payload.get("production_identity") or {}).get("src_gpuwrf_tree")
        == PRODUCTION_TREE,
        "accepted pre-C1 proof binds a different production tree",
    )
    return {
        "status": "PASS_ACCEPTED_PRE_C1",
        "artifact": artifact,
        "six_identity_checks": checks,
        "mutation_matrix": mutation,
        "readiness_seconds": (
            (payload.get("real_fast_cpu_identity") or {})
            .get("local_exact_compiled", {})
            .get("timing", {})
            .get("readiness_seconds")
        ),
    }


def collect_r8_residual(root: Path = R8_ROOT) -> dict[str, Any]:
    stderr = root / R8_STDERR.name
    stdout = root / R8_STDOUT.name
    timing = root / R8_TIME.name
    rss_path = root / R8_RSS.name
    artifacts = {
        name: common._artifact(path)
        for name, path in {
            "stderr": stderr,
            "stdout": stdout,
            "time": timing,
            "process_tree_rss": rss_path,
        }.items()
    }
    stderr_text = stderr.read_text(encoding="utf-8", errors="replace")
    time_text = timing.read_text(encoding="utf-8", errors="replace")
    stdout_payload = _load(stdout)
    rss = _load(rss_path)
    refusal = (
        "BoundaryRefusal: refusing stale/pre-existing wrfout output directory: "
        f"{root / 'wrfout'}"
    )
    wrfout_dir = root / "wrfout"
    _require(
        wrfout_dir.is_dir() and not any(wrfout_dir.iterdir()),
        "r8 precreated WRF output directory is no longer empty",
    )
    _require(refusal in stderr_text, "r8 lacks the exact publication refusal")
    _require(
        "_publish_exact_wrfout" in stderr_text
        and "wrfout, _host_result, result_digest = _publish_exact_wrfout("
        in stderr_text
        and "isolated CPU arm local-exact failed rc=1" in stderr_text,
        "r8 traceback does not prove the failure boundary/return code",
    )
    _require(
        "Elapsed (wall clock) time (h:mm:ss or m:ss): 1:16:44" in time_text
        and "Maximum resident set size (kbytes): 20899988" in time_text
        and "Exit status: 1" in time_text,
        "r8 /usr/bin/time terminal facts changed",
    )
    _require(
        stdout_payload.get("command_returncode") == 1
        and stdout_payload.get("device_action") is False
        and stdout_payload.get("peak_process_tree_rss_bytes")
        == 21_454_475_264,
        "r8 launcher summary changed",
    )
    expected_environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "GPUWRF_JAX_CACHE": "1",
        "GPUWRF_JAX_CACHE_DIR":
            "<DATA_ROOT>/wrf_gpu2/v025/m0/cpu_exact_boundary_cache_r1",
        "GPUWRF_JAX_CACHE_LOCK": "1",
        "JAX_PLATFORMS": "cpu",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
    }
    _require(
        rss.get("schema") == "wrf_gpu2.v025.m0.process_tree_rss.v1"
        and rss.get("status") == "PASS"
        and rss.get("run_id") == R8_RUN_ID
        and rss.get("device_action") is False
        and rss.get("jax_imported_by_sampler") is False
        and rss.get("cpu_affinity") == [4]
        and rss.get("environment") == expected_environment
        and rss.get("cache_path")
        == "<DATA_ROOT>/wrf_gpu2/v025/m0/cpu_exact_boundary_cache_r1"
        and rss.get("peak_process_tree_rss_bytes") == 21_454_475_264,
        "r8 machine/affinity/environment/cache/RSS identity changed",
    )
    return {
        "status": "RESIDUAL_EXACT_COMPUTE_COMPLETE_PUBLICATION_REFUSED",
        "accepted_as_c1_proof": False,
        "reason": refusal,
        "failure_boundary": (
            "after synchronized exact invocation, at the first "
            "_publish_exact_wrfout absent-directory guard"
        ),
        "science_computation_failure_observed": False,
        "wrfout_published": False,
        "fresh_cpu_wrf_comparator_run": False,
        "elapsed_wallclock": "1:16:44",
        "max_rss_time_kib": 20_899_988,
        "process_tree_peak_rss_bytes": 21_454_475_264,
        "machine_identity": {
            key: rss[key]
            for key in (
                "hostname",
                "platform",
                "python",
                "cpu_affinity",
                "environment",
                "cache_path",
                "observed_process_tree_pids",
            )
        },
        "artifacts": artifacts,
    }


def protocol_repairs() -> dict[str, Any]:
    fixture = exact_contract.validate_representative_long_run_fixtures()
    source = exact_contract.validate_sources()
    attacks = exact_builder._source_attack_matrix()
    _require(
        fixture.get("status") == "PASS"
        and len(fixture.get("validators_exercised") or []) == 6,
        "representative long-run validator preflight failed",
    )
    _require(
        attacks.get("status") == "PASS"
        and attacks.get("rejected") == attacks.get("total")
        and attacks.get("total", 0) >= 35,
        "source/static/mutation protocol repair is incomplete",
    )
    plan = executor.build_plan()
    plan_gate = exact_contract.validate_plan(plan)
    cold_sequences = {}
    for name, outcomes in {
        "pass_two": (True, True),
        "fail_two": (False, False),
        "pass_third": (True, False, True),
        "fail_third": (False, True, False),
    }.items():
        attempts = [
            {
                "process_id": 50_000 + index,
                "cache_path": f"/prospective-empty-cache-{name}-{index}",
                "readiness_seconds": 599.0 if passed else None,
                "threshold_stop": not passed,
            }
            for index, passed in enumerate(outcomes, start=1)
        ]
        cold_sequences[name] = core_session.classify_cold_attempts(attempts)
    memory = longrun.memory_preflight()
    clock_start = longrun.boot_clock_sample()
    clock_end = longrun.boot_clock_sample()
    continuity = longrun.validate_clock_continuity(clock_start, clock_end)
    inhibited, inhibitor = longrun.inhibited_command(
        ["FUTURE_EXACT_ORACLE_COMMAND"],
        estimated_seconds=5_400.0,
    )
    return {
        "status": "PASS_FIXTURE_STATIC_MUTATION_ONLY",
        "runtime_or_device_evidence": False,
        "representative_long_run_validators": fixture,
        "source_contract": source,
        "source_mutation_matrix": attacks,
        "output_directory_protocol": {
            "early_absent-target_preflight": "PASS",
            "child_atomic_no_replace_guard": "PASS",
            "r8_failure_reproduced_by_fixture": "PASS",
        },
        "long_run_controls": {
            "memory_preflight": memory,
            "clock_continuity_smoke": continuity,
            "future_inhibited_command": inhibited,
            "inhibitor": inhibitor,
        },
        "amendment5_session": {
            "plan_gate": plan_gate,
            "contract": core_session.session_contract(),
            "cold_sequence_proofs": cold_sequences,
            "device_execution": "MISSING_NOT_AUTHORIZED",
        },
    }


def milestone_partition() -> dict[str, Any]:
    return {
        "M0_CORE": {
            "status": "BLOCKED",
            "required_before": "M1_MECHANISM_WORK",
            "fields": {
                "accepted_exact_cpu_boundary": "PASS",
                "c1_same_result_wrfout_runtime": "BLOCKED_R8_PUBLICATION_REFUSED",
                "c1_fresh_cpu_wrf_comparator": "MISSING_NOT_RUN",
                "c2_jax_free_postlock_cpu_path": "PASS",
                "frozen_command_source_input_cache_tool_manifest": "PASS_CPU_SCOPE",
                "one_session_w1_w2_w3": "MISSING_NOT_AUTHORIZED",
                "fast_correctness_and_resource_gates": "MISSING_GPU_WINDOWS",
                "different_model_review": "MISSING_MANAGER_OWNED",
            },
            "may_open_m1": False,
        },
        "M0_DEFERRED": {
            "status": "MISSING_BY_NAMED_FUTURE_MILESTONE",
            "waived": False,
            "fields": {
                "instruction_counter_E_baseline_T_ceiling": "M2_ENTRY",
                "native_pallas_sm120": "M2_BACKEND_BAKEOFF_ENTRY",
                "ncu_registers_occupancy_stalls_dram_l2": "M2",
                "two_domain_rho_family_coverage_trace":
                    "M1_M2_PRODUCTION_REPRESENTATIVENESS",
                "multi_size_vram_scaling_fit": "RELEVANT_MILESTONE_AND_RELEASE",
            },
        },
    }


def validate_residual_closure(
    *,
    closure_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    closure = _load(closure_path)
    _require(
        closure.get("status") == STATUS
        and closure.get("terminal_status_semantics")
        == "CPU_HANDOFF_COMPLETE_NOT_C1_OR_M0_ACCEPTANCE"
        and closure.get("cpu_closure_gate") == "BLOCKED"
        and closure.get("m0_complete") is False
        and closure.get("device_touched") is False
        and closure.get("gpu_windows")
        == {"W1": "MISSING", "W2": "MISSING", "W3": "MISSING"}
        and (closure.get("c1") or {}).get("status") == C1_STATUS
        and (closure.get("c2") or {}).get("status") == C2_STATUS
        and (closure.get("milestone_partition") or {})
        .get("M0_CORE", {})
        .get("status")
        == "BLOCKED"
        and (closure.get("milestone_partition") or {})
        .get("M0_DEFERRED", {})
        .get("waived")
        is False,
        "residual closure overclaims C1/M0/device acceptance",
    )
    expected = common._canonical_sha256(
        {key: value for key, value in closure.items() if key != "closure_sha256"}
    )
    _require(closure.get("closure_sha256") == expected, "closure hash changed")
    manifest = _load(manifest_path)
    _require(
        manifest.get("status") == STATUS
        and manifest.get("cpu_closure_gate") == "BLOCKED"
        and manifest.get("device_action") is False
        and (manifest.get("milestone_partition") or {})
        .get("M0_CORE", {})
        .get("status")
        == "BLOCKED",
        "residual manifest overclaims the gate",
    )
    manifest_expected = common._canonical_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key != "manifest_payload_sha256"
        }
    )
    _require(
        manifest.get("manifest_payload_sha256") == manifest_expected,
        "residual manifest hash changed",
    )
    for artifact in manifest.get("files") or []:
        _require(
            common._artifact(Path(artifact["path"])) == artifact,
            f"manifest member changed: {artifact['path']}",
        )
    return {
        "status": "PASS",
        "terminal_status": STATUS,
        "cpu_closure_gate": "BLOCKED",
        "closure_sha256": common._sha256_file(closure_path),
        "manifest_sha256": common._sha256_file(manifest_path),
        "files_verified": len(manifest["files"]),
        "device_touched": False,
    }


def build(
    *,
    c1_output: Path,
    c2_output: Path,
    closure_output: Path,
    manifest_output: Path,
    junit_path: Path,
    w1b_report: Path,
    w1b_run_root: Path,
) -> dict[str, Any]:
    _require(not common._accelerator_modules(), "builder imported accelerator roots")
    production = common._production_identity()
    accepted = accepted_exact_boundary()
    r8 = collect_r8_residual()
    repairs = protocol_repairs()
    tests = common._junit(junit_path)
    _require(tests != "MISSING", "full tests/v025 JUnit is required")

    c1 = {
        "schema": common.C1_SCHEMA,
        "status": C1_STATUS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device_action": False,
        "accepted_exact_boundary": accepted,
        "final_admissible_attempt_r8": r8,
        "fixture_static_mutation_repairs": repairs,
        "frozen_gate_inventory": {
            "accepted_pre_c1_exact_boundary": "PASS",
            "same_synchronized_exact_result_adapter_source": "PASS_STATIC",
            "production_wrfout_adapter_fixture": "PASS_FIXTURE",
            "atomic_no_replace_and_absent_target_preflight": "PASS",
            "r8_same_run_wrfout": "MISSING_PUBLICATION_REFUSED",
            "fresh_12_rank_cpu_wrf_after_r8": "MISSING_NOT_RUN",
            "existing_comparator_delta_atlas": "MISSING_NOT_RUN",
        },
        "residual_risk": (
            "The repaired early absent-directory protocol is fixture/static/"
            "mutation proven but was not exercised by another full real-FAST "
            "oracle because manager commit 78aa07a9 forbids one."
        ),
    }
    cpu_proofs.write_json_no_replace(c1_output, c1)

    complete = cpu_proofs.complete_census_proof()
    attacks = cpu_proofs.census_attack_matrix()
    incomplete = cpu_proofs.run_incomplete_w1b_postlock(
        report_path=w1b_report,
        run_root=w1b_run_root,
    )
    _require(
        complete.get("status") == "PASS"
        and attacks.get("status") == "PASS"
        and attacks.get("rejected") == attacks.get("total")
        and attacks.get("total", 0) >= 20
        and incomplete.get("path_status") == "BLOCKED_AS_REQUIRED"
        and (incomplete.get("analysis") or {}).get("release_before_analysis")
        is True,
        "C2 CPU closure gates failed",
    )
    c2 = {
        "schema": common.C2_SCHEMA,
        "status": C2_STATUS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device_action": False,
        "complete_fixture_is_device_evidence": False,
        "complete_adversarial_census": complete,
        "fail_closed_matrix": attacks,
        "real_incomplete_w1b_postlock_path": incomplete,
        "measurement_partition":
            complete["census"]["measurement_partition"],
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
            "ncu_only_m2_fields": "MISSING_UNTIL_M2_NOT_WAIVED",
        },
    }
    cpu_proofs.write_json_no_replace(c2_output, c2)

    amendments = {
        "manager_amendment_01": _manager_object(
            MANAGER_AMENDMENT_COMMIT,
            ".agent/sprints/2026-07-27-v0250-m0-setup/"
            "MANAGER_C1_C2_AMENDMENT_01.md",
        ),
        "amendment_5": _manager_object(
            MANAGER_AMENDMENT_COMMIT,
            ".agent/sprints/2026-07-27-v0250-m0-setup/"
            "AMENDMENT_5_M0_CORE_EXIT.md",
        ),
        "r8_r9_adjudication": _manager_object(
            MANAGER_STOP_COMMIT,
            ".agent/decisions/V0250-ROADMAP.md",
        ),
    }
    attempt_provenance = {
        "r6": {
            "status": "INVALID_HIBERNATION_SPANNED",
            "exit_provenance": "manager SIGINT to dedicated PGID 3275882",
            "runtime_or_session_limit": False,
            "authority": amendments["manager_amendment_01"],
        },
        "r7": {
            "status": "INVALIDATED_DEADLOCK_DISCRIMINATOR",
            "repair": "single-core pre-JAX XLA guard",
        },
        "r8": r8,
        "r9": {
            "status": "INVALID_MANAGER_STOPPED_NO_PUBLICATION",
            "manager_signals": ["SIGINT", "SIGTERM"],
            "dedicated_pgid": 3_403_883,
            "verified_gone_pids": [
                3_403_883,
                3_403_887,
                3_403_888,
                3_403_900,
            ],
            "authority": amendments["r8_r9_adjudication"],
        },
    }
    partition = milestone_partition()
    c1_artifact = common._artifact(c1_output)
    c2_artifact = common._artifact(c2_output)
    closure = {
        "schema": common.CLOSURE_SCHEMA,
        "status": STATUS,
        "terminal_status_semantics":
            "CPU_HANDOFF_COMPLETE_NOT_C1_OR_M0_ACCEPTANCE",
        "cpu_closure_gate": "BLOCKED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device_policy": "closed",
        "device_touched": False,
        "device_queries": [],
        "receipts_read": [],
        "receipts_consumed": [],
        "gpu_windows": {"W1": "MISSING", "W2": "MISSING", "W3": "MISSING"},
        "native_pallas_verdict": "MISSING_DEFERRED_TO_M2",
        "m0_complete": False,
        "production_identity": production,
        "machine": common._machine_identity(),
        "source_contract": repairs["source_contract"],
        "amendments": amendments,
        "attempt_provenance": attempt_provenance,
        "kimi_f2_machine_affinity_environment_cache_identity": {
            "status": "PASS",
            "proof": r8["machine_identity"],
        },
        "c1": {"status": C1_STATUS, "artifact": c1_artifact},
        "c2": {"status": C2_STATUS, "artifact": c2_artifact},
        "milestone_partition": partition,
        "tests_v025": tests,
        "unresolved_risks": [
            "C1 lacks an accepted same-run WRF output and fresh CPU-WRF comparator.",
            "W1/W2/W3 and every device/resource measurement remain missing.",
            "A different-model review is manager-owned and still missing.",
        ],
        "next_decision": (
            "manager assigns different-model review; W1 remains forbidden "
            "until the residual is adjudicated and M0-CORE gates permit it"
        ),
    }
    closure["closure_sha256"] = common._canonical_sha256(closure)
    cpu_proofs.write_json_no_replace(closure_output, closure)

    files = [
        ACCEPTED_EXACT,
        c1_output,
        c2_output,
        closure_output,
        junit_path,
        R8_STDERR,
        R8_STDOUT,
        R8_TIME,
        R8_RSS,
        Path(incomplete["manager_result"]["path"]),
        Path(incomplete["release_proof"]["path"]),
        Path(incomplete["analysis"]["path"]),
        w1b_report,
    ]
    manifest = {
        "schema": common.MANIFEST_SCHEMA,
        "status": STATUS,
        "cpu_closure_gate": "BLOCKED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device_action": False,
        "production_tree": PRODUCTION_TREE,
        "command_identity": r8["artifacts"]["stdout"],
        "source_identity": repairs["source_contract"]["source_sha256"],
        "input_cache_machine_identity": r8["machine_identity"],
        "tool_identity": {
            "systemd_inhibit":
                repairs["long_run_controls"]["inhibitor"],
            "nsys_export":
                incomplete["analysis"]["export"].get("tool"),
        },
        "milestone_partition": partition,
        "files": [common._artifact(path) for path in files],
    }
    manifest["manifest_payload_sha256"] = common._canonical_sha256(manifest)
    cpu_proofs.write_json_no_replace(manifest_output, manifest)
    validated = validate_residual_closure(
        closure_path=closure_output,
        manifest_path=manifest_output,
    )
    return {
        **validated,
        "c1_status": C1_STATUS,
        "c2_status": C2_STATUS,
        "c1": c1_artifact,
        "c2": c2_artifact,
        "closure": common._artifact(closure_output),
        "manifest": common._artifact(manifest_output),
        "c1_source_attacks_rejected":
            repairs["source_mutation_matrix"]["rejected"],
        "c2_attacks_rejected": attacks["rejected"],
        "c2_attacks_total": attacks["total"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c1-output", type=Path, default=DEFAULT_C1)
    parser.add_argument("--c2-output", type=Path, default=DEFAULT_C2)
    parser.add_argument("--closure-output", type=Path, default=DEFAULT_CLOSURE)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--w1b-report", type=Path, default=cpu_proofs.W1B_REPORT)
    parser.add_argument("--w1b-run-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.validate_only:
            summary = validate_residual_closure(
                closure_path=args.closure_output,
                manifest_path=args.manifest_output,
            )
        else:
            if args.junit is None or args.w1b_run_root is None:
                parser.error("--junit and --w1b-run-root are required")
            summary = build(
                c1_output=args.c1_output,
                c2_output=args.c2_output,
                closure_output=args.closure_output,
                manifest_output=args.manifest_output,
                junit_path=args.junit,
                w1b_report=args.w1b_report,
                w1b_run_root=args.w1b_run_root,
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
