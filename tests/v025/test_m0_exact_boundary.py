"""CPU-only closure gates for the Amendment-4 exact executable boundary."""

from __future__ import annotations

import copy
import json
import os
import pickle
import signal
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts/v025"
sys.path.insert(0, str(SCRIPTS))

import build_m0_exact_boundary_evidence as builder  # noqa: E402
import m0_exact_boundary_child as child  # noqa: E402
import m0_exact_boundary_contract as contract  # noqa: E402
import m0_three_window_executor as executor  # noqa: E402
import m0_window_parent as parent  # noqa: E402


class _AddressBearingDefinition:
    """Pickle-stable test double whose debug repr is intentionally unstable."""

    num_nodes = 7
    num_leaves = 3

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"<definition value={self.value!r} at {hex(id(self))}>"


def _sources():
    return {
        "child_source": contract.CHILD.read_text(),
        "parent_source": contract.PARENT.read_text(),
        "parent_death_guard_source": contract.PARENT_DEATH_GUARD.read_text(),
        "executor_source": contract.EXECUTOR.read_text(),
        "pallas_source": contract.PALLAS.read_text(),
        "evidence_builder_source": contract.EVIDENCE_BUILDER.read_text(),
        "long_run_controls_source": contract.LONG_RUN_CONTROLS.read_text(),
        "core_session_source": contract.CORE_SESSION_PROTOCOL.read_text(),
    }


def test_cpu_reference_cache_contract_is_hash_bound(tmp_path, monkeypatch):
    cache = tmp_path / "matched-cache"
    cache.mkdir()
    entry_name = "jit__run_forecast_operational_jit-test-cache"
    entry = cache / entry_name
    entry.write_bytes(b"matched historical executable")
    monkeypatch.setattr(builder, "REFERENCE_CPU_CACHE_DIR", cache)
    monkeypatch.setattr(builder, "REFERENCE_CPU_CACHE_ENTRY", entry_name)
    monkeypatch.setattr(
        builder, "REFERENCE_CPU_CACHE_ENTRY_BYTES", entry.stat().st_size
    )
    monkeypatch.setattr(
        builder,
        "REFERENCE_CPU_CACHE_ENTRY_SHA256",
        builder._sha256_file(entry),
    )
    monkeypatch.setenv("GPUWRF_JAX_CACHE", "1")
    monkeypatch.setenv("GPUWRF_JAX_CACHE_DIR", str(cache))
    monkeypatch.setenv("GPUWRF_JAX_CACHE_LOCK", "1")

    result = builder._cpu_reference_cache_contract()

    assert result["status"] == "PASS"
    assert result["top_level_entry_sha256"] == builder._sha256_file(entry)
    assert result["device_touched"] is False


def test_cpu_reference_cache_contract_rejects_unlike_cold_run(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("GPUWRF_JAX_CACHE", "0")
    monkeypatch.setenv("GPUWRF_JAX_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GPUWRF_JAX_CACHE_LOCK", "1")

    with pytest.raises(RuntimeError, match="cold-vs-cached"):
        builder._cpu_reference_cache_contract()


def _valid_result(mode: str) -> dict:
    is_compile = mode == "compile-only"
    is_profiled = mode == "profiled"
    run_id = f"cpu-fixture-{mode}"
    payload = {
        "schema": "wrf_gpu2.v025.m0.exact_executable_boundary.v1",
        "status": "OK",
        "run_id": run_id,
        "case": {
            "domain": "d01",
            "hours": 1.0,
            "case_metadata": {"fixture": "real-case-shaped"},
        },
        "production_binding": {
            "wrapper_preparation_sequence": list(child.WRAPPER_PREPARATION_SEQUENCE),
            "wrapper_preparation_identity": [
                {
                    "fqname": name,
                    "ast_sha256": child.EXPECTED_FUNCTION_AST_SHA256[name],
                }
                for name in child.WRAPPER_PREPARATION_SEQUENCE
            ],
            "jit_identity": {
                "fqname": child.JIT_FUNCTION,
                "ast_sha256": child.EXPECTED_FUNCTION_AST_SHA256[child.JIT_FUNCTION],
            },
            "case_preparation_sequence": list(child.CASE_PREPARATION_SEQUENCE),
        },
        "call": {
            "argument_identity": {"exact_value_sha256": "1" * 64},
            "lower_api":
                "_run_forecast_operational_jit.lower(*prepared_arguments).compile()",
            "compiled_invocation": (
                "_call_compiled_with_exact_arguments("
                "executable, prepared_arguments, lowered_static_hours=hours)"
            ),
            "jax_compiled_dynamic_invocation":
                "executable(prepared_state, namelist)",
            "static_argument_binding": {
                "name": "hours",
                "value": 1.0,
                "runtime_revalidated_before_dynamic_call": True,
            },
            "lowered_program_sha256": "2" * 64,
            "hours": 1.0,
            "compiled_invocation_count": 0 if is_compile else 1,
        },
        "timing": {
            "parent_process_launch_monotonic_ns": 100_000_000,
            "child_executable_ready_monotonic_ns": 200_000_000,
            "readiness_seconds": 0.1,
            "argument_identity_start_monotonic_ns": 210_000_000,
            "argument_identity_end_monotonic_ns": 220_000_000,
            "argument_identity_seconds": 0.01,
            "lowered_identity_start_monotonic_ns": 230_000_000,
            "lowered_identity_end_monotonic_ns": 240_000_000,
            "lowered_identity_seconds": 0.01,
            "proof_identity_seconds": 0.03,
            "proof_identity_definition": (
                "after executable readiness -> exact argument and lowered "
                "StableHLO identities -> before compiled invocation"
            ),
            "integration_start_monotonic_ns":
                None if is_compile else 300_000_000,
            "integration_end_monotonic_ns":
                None if is_compile else 400_000_000,
            "integration_seconds": None if is_compile else 0.1,
            "derived_by_phase_subtraction": False,
            "first_lazy_call_used_as_readiness": False,
        },
        "instrumentation": {
            "mode": mode,
            "nvtx_range": child.RANGE_NAME if is_profiled else None,
            "allocator_sidecar": "/tmp/allocator.json" if is_profiled else None,
            "invocation_helper":
                "m0_exact_boundary_child._invoke_and_synchronize",
        },
        "allocator": {"peak_bytes_in_use": 1} if is_profiled else None,
        "result": {
            "semantics": None if is_compile else {"treedef": "State"},
            "exact_value_sha256": None if is_compile else "3" * 64,
            "synchronization":
                None if is_compile else "jax.block_until_ready(result)",
        },
    }
    if not is_compile:
        exact_identity = {
            "case_sha256": contract.canonical_sha256(payload["case"]),
            "hours": 1.0,
            "fixture": "exact-boundary",
        }
        input_binding = {
            "config_sha256": "7" * 64,
            "namelist_input_sha256": "6" * 64,
            "input_manifest_sha256": "8" * 64,
            "case_sha256": contract.canonical_sha256(payload["case"]),
        }
        config_identity = {"run_id": run_id, "domain": "d01", "hours": 1}
        input_binding["config_sha256"] = contract.canonical_sha256(
            config_identity
        )
        variable_inventory = [
            {
                "name": "T",
                "dtype": "float32",
                "dimensions": [
                    "Time",
                    "bottom_top",
                    "south_north",
                    "west_east",
                ],
                "shape": [1, 44, 70, 120],
            }
        ]
        dimensions = {
            "Time": 1,
            "bottom_top": 44,
            "south_north": 70,
            "west_east": 120,
        }
        wrfout = {
            "schema": "wrf_gpu2.v025.m0.exact_result_wrfout.v1",
            "status": "PASS",
            "run_id": run_id,
            "result_exact_value_sha256": "3" * 64,
            "exact_boundary_identity": exact_identity,
            "exact_boundary_identity_sha256":
                contract.canonical_sha256(exact_identity),
            "input_binding": input_binding,
            "adapter_identity": [
                {
                    "fqname": name,
                    "ast_sha256": "9" * 64,
                }
                for name in (
                    "gpuwrf.integration.daily_pipeline."
                    "_surface_diagnostics_for_output",
                    "gpuwrf.integration.daily_pipeline."
                    "_merge_output_diagnostics",
                    "gpuwrf.integration.daily_pipeline._wrfout_name",
                    "gpuwrf.integration.daily_pipeline."
                    "build_wrfout_inventory",
                    "gpuwrf.io.wrfout_writer.prepare_wrfout_payload",
                    "gpuwrf.io.wrfout_writer.write_prepared_wrfout",
                )
            ],
            "adapter_source": {
                "src_gpuwrf_tree":
                    "a6885ceded260df2f5777d7366d75a5d38947cb7",
                "child_path": str(contract.CHILD.resolve()),
                "child_sha256": contract.sha256_file(contract.CHILD),
            },
            "final_wrfout_path": f"/fixture/{run_id}.nc",
            "final_wrfout_bytes": 123,
            "final_wrfout_sha256": "4" * 64,
            "domain": "d01",
            "domain_authority_sha256": "a" * 64,
            "run_start_utc": "2026-07-28T00:00:00+00:00",
            "valid_time_utc": "2026-07-28T01:00:00+00:00",
            "lead_hours": 1.0,
            "operational_variable_set": True,
            "full_variable_set": False,
            "inventory": {"status": "PASS"},
            "finiteness": {"status": "PASS"},
            "config_identity": config_identity,
            "variable_inventory": variable_inventory,
            "variable_inventory_sha256":
                contract.canonical_sha256(variable_inventory),
            "dimensions": dimensions,
            "dimensions_sha256": contract.canonical_sha256(dimensions),
            "inventory_sha256":
                contract.canonical_sha256({"status": "PASS"}),
            "finiteness_sha256":
                contract.canonical_sha256({"status": "PASS"}),
            "timing": {
                "materialization_start_monotonic_ns": 410_000_000,
                "materialization_end_monotonic_ns": 420_000_000,
                "prepare_start_monotonic_ns": 430_000_000,
                "prepare_end_monotonic_ns": 440_000_000,
                "write_start_monotonic_ns": 450_000_000,
                "write_end_monotonic_ns": 460_000_000,
                "inspection_start_monotonic_ns": 470_000_000,
                "inspection_end_monotonic_ns": 480_000_000,
                "outside_readiness_and_integration_clocks": True,
            },
            "publication": {"atomic": True, "replacement": False},
        }
        wrfout["adapter_identity_sha256"] = contract.canonical_sha256(
            wrfout["adapter_identity"]
        )
        wrfout["adapter_source_sha256"] = contract.canonical_sha256(
            wrfout["adapter_source"]
        )
        wrfout["binding_sha256"] = contract.canonical_sha256(
            {
                "run_id": run_id,
                "result_exact_value_sha256": "3" * 64,
                "exact_boundary_identity_sha256":
                    wrfout["exact_boundary_identity_sha256"],
                **input_binding,
                "final_wrfout_path": wrfout["final_wrfout_path"],
                "final_wrfout_bytes": wrfout["final_wrfout_bytes"],
                "final_wrfout_sha256": wrfout["final_wrfout_sha256"],
                "domain": "d01",
                "domain_authority_sha256":
                    wrfout["domain_authority_sha256"],
                "run_start_utc": wrfout["run_start_utc"],
                "valid_time_utc": wrfout["valid_time_utc"],
                "lead_hours": 1.0,
                "operational_variable_set": True,
                "full_variable_set": False,
                "variable_inventory_sha256":
                    wrfout["variable_inventory_sha256"],
                "dimensions_sha256": wrfout["dimensions_sha256"],
                "inventory_sha256": wrfout["inventory_sha256"],
                "finiteness_sha256": wrfout["finiteness_sha256"],
                "adapter_identity_sha256":
                    wrfout["adapter_identity_sha256"],
                "adapter_source_sha256":
                    wrfout["adapter_source_sha256"],
            }
        )
        payload["wrfout"] = wrfout
    else:
        payload["wrfout"] = None
    return payload


def test_importing_plan_parent_and_children_is_accelerator_free():
    code = f"""
import json
import sys
sys.path.insert(0, {str(SCRIPTS)!r})
before = set(sys.modules)
import m0_exact_boundary_child
import m0_three_window_executor
import m0_window_parent
import pallas_sm120_spike
after = set(sys.modules)
print(json.dumps({{
    "jax": "jax" in after - before,
    "jaxlib": "jaxlib" in after - before,
    "gpuwrf": "gpuwrf" in after - before,
}}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "jax": False,
        "jaxlib": False,
        "gpuwrf": False,
    }


def test_static_exact_boundary_contract_passes():
    result = contract.validate_sources()
    assert result["status"] == "PASS"
    assert result["exact_lower_compile"] is True
    assert result["clean_instrumentation_absent"] is True
    assert result["sequential_fresh_cpu_arm_processes"] is True
    assert result["cpu_arm_process_group_attached"] is True
    assert result["single_core_cpu_arm_before_jax"] is True
    assert result["long_run_preflight_and_inhibition"] is True
    assert result["amendment5_one_session_two_of_three"] is True


def test_treedef_identity_uses_content_not_address_bearing_debug_repr():
    left = _AddressBearingDefinition({"static": None})
    right = _AddressBearingDefinition({"static": None})
    assert repr(left) != repr(right)
    assert pickle.dumps(left, protocol=5) == pickle.dumps(right, protocol=5)
    assert child._treedef_identity(left) == child._treedef_identity(right)


def _mutate_prep_order(sources):
    source = sources["child_source"]
    sources["child_source"] = source.replace(
        "prepared_state = scan_state(state, namelist)\n"
        "    prepared_state = dealias(prepared_state)",
        "prepared_state = dealias(state)\n"
        "    prepared_state = scan_state(prepared_state, namelist)",
    )


def _mutate_function_identity(sources):
    sources["child_source"] = sources["child_source"].replace(
        child.EXPECTED_FUNCTION_AST_SHA256[
            "gpuwrf.runtime.operational_mode._operational_scan_state"
        ],
        "0" * 64,
        1,
    )


def _mutate_arguments(sources):
    sources["child_source"] = sources["child_source"].replace(
        "lowered = exact_jit.lower(*prepared_arguments)",
        "lowered = exact_jit.lower(prepared_arguments[0], "
        "prepared_arguments[1], prepared_arguments[2] + 1.0)",
    )


def _mutate_readiness(sources):
    sources["child_source"] = sources["child_source"].replace(
        "executable = lowered.compile()\n"
        "    executable_ready_monotonic_ns = time.monotonic_ns()",
        "executable_ready_monotonic_ns = time.monotonic_ns()\n"
        "    executable = lowered.compile()",
    )


def _mutate_argument_identity_into_readiness(sources):
    source = sources["child_source"]
    identity = (
        "    argument_identity = "
        "_structural_argument_identity(jax, prepared_arguments)\n"
    )
    without_identity = source.replace(identity, "", 1)
    sources["child_source"] = without_identity.replace(
        "    executable = lowered.compile()\n",
        identity + "    executable = lowered.compile()\n",
        1,
    )


def _mutate_lowered_identity_into_readiness(sources):
    source = sources["child_source"]
    identity = "    stablehlo = _stablehlo_identity(lowered)\n"
    without_identity = source.replace(identity, "", 1)
    sources["child_source"] = without_identity.replace(
        "    executable = lowered.compile()\n",
        identity + "    executable = lowered.compile()\n",
        1,
    )


def _mutate_sync(sources):
    sources["child_source"] = sources["child_source"].replace(
        "jax.block_until_ready(result)",
        "jax.tree_util.tree_leaves(result)",
        1,
    )


def _mutate_range(sources):
    sources["child_source"] = sources["child_source"].replace(
        'RANGE_NAME = "GPUWRF_M0_FORECAST_INTEGRATION"',
        'RANGE_NAME = "MUTATED_RANGE"',
    )


def _mutate_clean_instrumentation(sources):
    sources["child_source"] = sources["child_source"].replace(
        'elif mode == "clean":\n'
        "        result, integration_start_ns, integration_end_ns",
        'elif mode == "clean":\n'
        "        _forbidden_profiler = jax.profiler\n"
        "        result, integration_start_ns, integration_end_ns",
    )


def _mutate_parent_jax_import(sources):
    sources["parent_source"] = sources["parent_source"].replace(
        "import hashlib", "import jax\nimport hashlib", 1
    )


def _mutate_parent_death_arm(sources):
    sources["parent_death_guard_source"] = sources[
        "parent_death_guard_source"
    ].replace(
        "arm_parent_death_signal(expected_lock_owner_pid, signal.SIGTERM)",
        "arm_parent_death_signal_disabled("
        "expected_lock_owner_pid, signal.SIGTERM)",
        1,
    )


def _mutate_parent_death_group_kill(sources):
    sources["parent_death_guard_source"] = sources[
        "parent_death_guard_source"
    ].replace(
        "os.killpg(os.getpgrp(), signal.SIGKILL)",
        "os.killpg(os.getpgrp(), signal.SIGTERM)",
        1,
    )


def _mutate_pallas_auth_order(sources):
    sources["pallas_source"] = sources["pallas_source"].replace(
        "payload = run_native(args, authorization)",
        "payload = run_native(args, authorization)\n"
        "            authorization = exact_child.consume_authorization_handoff("
        "args.handoff, expected_window=args.window, "
        "expected_stage=args.stage, expected_run_id=args.run_id)",
        1,
    )


def _mutate_pallas_pair_gate(sources):
    sources["parent_source"] = sources["parent_source"].replace(
        "pair_gate = {",
        "pair_gate_deferred = {",
        1,
    )


def _mutate_profiled_capture_integrity(sources):
    sources["parent_source"] = sources["parent_source"].replace(
        "artifacts = _require_profiled_capture_artifacts(profiled)",
        "artifacts = {}",
        1,
    )


def _mutate_w1_fast_pair_binding(sources):
    sources["parent_source"] = sources["parent_source"].replace(
        "if binding != expected_binding:",
        "if False:",
        1,
    )


def _mutate_substituted_wrfout_result(sources):
    sources["child_source"] = sources["child_source"].replace(
        "result=result,",
        "result=prepared_arguments[0],",
        1,
    )


def _mutate_output_clock_before_synchronization(sources):
    sources["child_source"] = sources["child_source"].replace(
        "jax.block_until_ready(result)\n"
        "    integration_end_ns = time.monotonic_ns()",
        "integration_end_ns = time.monotonic_ns()\n"
        "    jax.block_until_ready(result)",
        1,
    )


def _mutate_stale_output_reuse(sources):
    sources["child_source"] = sources["child_source"].replace(
        "if os.path.lexists(target_dir):",
        "if False and os.path.lexists(target_dir):",
        1,
    )


def _mutate_output_domain(sources):
    sources["child_source"] = sources["child_source"].replace(
        "domain=config.domain,",
        'domain="d02",',
        1,
    )


def _mutate_output_variable_set(sources):
    sources["child_source"] = sources["child_source"].replace(
        "full_variable_set=full_variable_set,",
        "full_variable_set=False,",
        1,
    )


def _mutate_output_variable_inventory_binding(sources):
    sources["child_source"] = sources["child_source"].replace(
        '"variable_inventory_sha256": canonical_sha256(variable_inventory),',
        '"variable_inventory_sha256": "0" * 64,',
        1,
    )


def _mutate_same_process_double_integration(sources):
    sources["evidence_builder_source"] = sources[
        "evidence_builder_source"
    ].replace(
        "local_envelope, local_producer = _run_isolated_arm(\n"
        '            kind="local-exact",',
        "local_envelope, local_producer = _run_isolated_arm(\n"
        '            kind="public-wrapper",',
        1,
    )


def _mutate_isolated_arm_detached_process_group(sources):
    sources["evidence_builder_source"] = sources[
        "evidence_builder_source"
    ].replace(
        "completed = subprocess.run(\n"
        "        inhibited,\n"
        "        cwd=REPO,\n"
        "        check=False,\n"
        "    )",
        "completed = subprocess.run(\n"
        "        inhibited,\n"
        "        cwd=REPO,\n"
        "        check=False,\n"
        "        start_new_session=True,\n"
        "    )",
        1,
    )


def _mutate_cpu_arm_gpu_timing_gate(sources):
    sources["evidence_builder_source"] = sources[
        "evidence_builder_source"
    ].replace(
        "contract.validate_wrfout_binding(local, verify_file=True)",
        'contract.validate_boundary_result(local, expected_mode="clean")\n'
        "        contract.validate_wrfout_binding(local, verify_file=True)",
        1,
    )


def _mutate_single_core_cpu_guard(sources):
    sources["evidence_builder_source"] = sources[
        "evidence_builder_source"
    ].replace(
        "os.sched_setaffinity(0, {inherited_affinity[0]})",
        "single_core_affinity_not_applied = inherited_affinity[0]",
    )


@pytest.mark.parametrize(
    "mutation",
    [
        _mutate_prep_order,
        _mutate_function_identity,
        _mutate_arguments,
        _mutate_readiness,
        _mutate_argument_identity_into_readiness,
        _mutate_lowered_identity_into_readiness,
        _mutate_sync,
        _mutate_range,
        _mutate_clean_instrumentation,
        _mutate_parent_jax_import,
        _mutate_parent_death_arm,
        _mutate_parent_death_group_kill,
        _mutate_pallas_auth_order,
        _mutate_pallas_pair_gate,
        _mutate_profiled_capture_integrity,
        _mutate_w1_fast_pair_binding,
        _mutate_substituted_wrfout_result,
        _mutate_output_clock_before_synchronization,
        _mutate_stale_output_reuse,
        _mutate_output_domain,
        _mutate_output_variable_set,
        _mutate_output_variable_inventory_binding,
        _mutate_same_process_double_integration,
        _mutate_isolated_arm_detached_process_group,
        _mutate_cpu_arm_gpu_timing_gate,
        _mutate_single_core_cpu_guard,
    ],
    ids=lambda function: function.__name__,
)
def test_all_source_boundary_mutations_fail_closed(mutation):
    sources = _sources()
    mutation(sources)
    with pytest.raises((contract.ContractViolation, SyntaxError)):
        contract.validate_source_texts(**sources)


def test_prepare_exact_call_uses_exact_functions_in_exact_order():
    calls = []

    def assert_nonzero(state):
        calls.append(("assert", state))

    def scan(state, namelist):
        calls.append(("scan", state, namelist))
        return f"{state}:scan"

    def dealias(state):
        calls.append(("dealias", state))
        return f"{state}:dealias"

    functions = {
        child.WRAPPER_PREPARATION_SEQUENCE[0]: assert_nonzero,
        child.WRAPPER_PREPARATION_SEQUENCE[1]: scan,
        child.WRAPPER_PREPARATION_SEQUENCE[2]: dealias,
    }
    identities = {name: {"fqname": name} for name in functions}
    prepared, sequence = child.prepare_exact_call(
        ("state", "namelist", 1.0),
        {"functions": functions, "identities": identities},
    )

    assert calls == [
        ("assert", "state"),
        ("scan", "state", "namelist"),
        ("dealias", "state:scan"),
    ]
    assert prepared == ("state:scan:dealias", "namelist", 1.0)
    assert sequence == [identities[name] for name in child.WRAPPER_PREPARATION_SEQUENCE]


@pytest.mark.parametrize("mode", ["compile-only", "clean", "profiled"])
def test_valid_boundary_artifact_passes(mode):
    assert contract.validate_boundary_result(
        _valid_result(mode), expected_mode=mode
    )["status"] == "PASS"


@pytest.mark.parametrize(
    "path,replacement",
    [
        (("production_binding", "wrapper_preparation_sequence"), ["wrong"]),
        (("call", "compiled_invocation"), "executable(state, namelist, 2.0)"),
        (("timing", "child_executable_ready_monotonic_ns"), 99_000_000),
        (("timing", "argument_identity_start_monotonic_ns"), 199_000_000),
        (("timing", "lowered_identity_end_monotonic_ns"), 301_000_000),
        (("timing", "integration_start_monotonic_ns"), 199_000_000),
        (("timing", "derived_by_phase_subtraction"), True),
        (("result", "synchronization"), "none"),
        (("instrumentation", "nvtx_range"), "WRONG"),
        (("wrfout", "result_exact_value_sha256"), "f" * 64),
        (("wrfout", "domain"), "d02"),
        (("wrfout", "valid_time_utc"), "2026-07-28T02:00:00+00:00"),
        (("wrfout", "lead_hours"), 2.0),
        (("wrfout", "binding_sha256"), "f" * 64),
        (("wrfout", "variable_inventory_sha256"), "f" * 64),
        (("wrfout", "adapter_identity_sha256"), "f" * 64),
        (("wrfout", "operational_variable_set"), False),
        (("wrfout", "publication"), {"atomic": False, "replacement": True}),
    ],
)
def test_boundary_artifact_mutations_fail_closed(path, replacement):
    payload = _valid_result("profiled")
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    with pytest.raises(contract.ContractViolation):
        contract.validate_boundary_result(payload, expected_mode="profiled")


def test_profiled_and_clean_differ_only_in_instrumentation():
    profiled = _valid_result("profiled")
    clean = _valid_result("clean")
    assert contract.compare_profiled_clean(profiled, clean)["status"] == "PASS"
    clean["call"]["argument_identity"]["exact_value_sha256"] = "f" * 64
    with pytest.raises(contract.ContractViolation, match="argument_identity"):
        contract.compare_profiled_clean(profiled, clean)


def test_jax_free_postlock_profiler_pair_is_exactly_bound_and_fail_closed():
    profiled = _valid_result("profiled")
    clean = _valid_result("clean")
    for run_id, result in (
        ("m0-profiled-post-0001", profiled),
        ("m0-clean-post-0001", clean),
    ):
        result["run_id"] = run_id
        result["call"]["hours"] = 1.0
        result["timing"]["integration_definition"] = (
            "immediately before compiled invocation -> after "
            "jax.block_until_ready(result)"
        )
        result["case"] = {
            "domain": "d01",
            "hours": 1.0,
            "boundary_record_cadence_s": 21600.0,
            "boundary_window_cadence_s": 21600.0,
        }

    accelerator_modules_before = {
        name
        for name in sys.modules
        if name == "jax"
        or name.startswith("jax.")
        or name == "gpuwrf"
        or name.startswith("gpuwrf.")
    }
    payload = parent._matched_profiler_post(
        profiled=profiled,
        clean=clean,
    )
    assert payload["profiler_perturbation"]["status"] == "OK"
    assert payload["profiler_perturbation"]["overhead_fraction"] == 0.0
    assert payload["matched_pair"]["order"] == [
        "m0-profiled-post-0001",
        "m0-clean-post-0001",
    ]
    assert {
        name
        for name in sys.modules
        if name == "jax"
        or name.startswith("jax.")
        or name == "gpuwrf"
        or name.startswith("gpuwrf.")
    } == accelerator_modules_before

    mutation = copy.deepcopy(clean)
    mutation["case"]["boundary_window_cadence_s"] = 54.0
    with pytest.raises(parent.WindowRefusal, match="bindings differ"):
        parent._matched_profiler_post(profiled=profiled, clean=mutation)


def test_w1_post_qualification_rejects_unbound_green_label(tmp_path):
    w1 = tmp_path / "w1.json"
    pair = tmp_path / "pair.json"
    output = tmp_path / "qualification.json"
    w1.write_text(json.dumps({"status": "OK", "stages": []}))
    pair.write_text(json.dumps({"status": "PASS"}))

    with pytest.raises(parent.WindowRefusal, match="qualification is not green"):
        parent._post_w1_qualification(
            w1_result_path=w1,
            fast_pair_path=pair,
            output_path=output,
        )
    assert not output.exists()


def test_plan_fresh_process_cache_budget_and_profiled_first_contract():
    plan = executor.build_plan()
    assert contract.validate_plan(plan)["status"] == "PASS"

    mutation = copy.deepcopy(plan)
    mutation["manager_window_inspection"]["W3"]["stages"][0][
        "fresh_process"
    ] = False
    with pytest.raises(contract.ContractViolation, match="non-fresh"):
        contract.validate_plan(mutation)

    mutation = copy.deepcopy(plan)
    mutation["profiled_pair_order"].reverse()
    with pytest.raises(contract.ContractViolation, match="profiled-first"):
        contract.validate_plan(mutation)

    mutation = copy.deepcopy(plan)
    mutation["identity_and_cache_provenance"]["profiled_identity"]["path"] = (
        "wrong.json"
    )
    with pytest.raises(contract.ContractViolation, match="paths consumed"):
        contract.validate_plan(mutation)

    mutation = copy.deepcopy(plan)
    mutation["maximum_lock_acquisitions"] = 2
    with pytest.raises(contract.ContractViolation, match="more than one"):
        contract.validate_plan(mutation)

    mutation = copy.deepcopy(plan)
    mutation["manager_session_command"][2] = "30"
    with pytest.raises(contract.ContractViolation, match="one-session"):
        contract.validate_plan(mutation)


def _process_is_running(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return False
    fields = stat[stat.rfind(")") + 2 :].split()
    return bool(fields) and fields[0] != "Z"


def _wait_for_path(path: Path, timeout_seconds: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {path}")


@pytest.mark.parametrize(
    "parent_death_signal",
    [signal.SIGTERM, signal.SIGINT, signal.SIGKILL],
    ids=["sigterm", "sigint", "abrupt-sigkill"],
)
def test_parent_death_kills_trapping_child_grandchild_and_spends_receipt(
    tmp_path,
    parent_death_signal,
):
    import wrf_source_authority as wsa

    authority = wsa.build_source_authority(
        namelist_path=parent.FAST_RUN_DIR / "namelist.input",
        environ={
            variable: str(wsa.CANONICAL_ROOT)
            for variable in wsa.ROOT_ENV_VARS
        },
    )
    authority_path = tmp_path / "wrf_source_authority.json"
    wsa.write_authority(authority_path, authority)
    """A dead lock owner cannot leave device work or authorize a later stage."""

    now = datetime.now(timezone.utc).isoformat()
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "window": parent.WINDOWS["W1"]["label"],
                "requested_at_utc": now,
                "request_text": "CPU-only parent-death test",
                "replies": {
                    "0:2": {
                        "affirmative": True,
                        "verbatim": "CPU-only fixture approval 0:2",
                        "received_at_utc": now,
                    },
                    "0:3": {
                        "affirmative": True,
                        "verbatim": "CPU-only fixture approval 0:3",
                        "received_at_utc": now,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    token = "cpu-parent-death-token"
    holder = tmp_path / "holder"
    holder.write_text(
        f"holder={parent.WINDOWS['W1']['label']} "
        f"pid={os.getpid()} token={token} cmd=cpu-only\n",
        encoding="utf-8",
    )
    ledger = tmp_path / "spent.json"
    ready = tmp_path / "descendants-ready"
    child_pid_path = tmp_path / "child.pid"
    grandchild_pid_path = tmp_path / "grandchild.pid"
    later_stage = tmp_path / "later-stage-ran"
    authorized = tmp_path / "receipt-spent"
    run_root = tmp_path / "run"

    spawner = tmp_path / "sigterm_trapping_tree.py"
    spawner.write_text(
        textwrap.dedent(
            """\
            import os
            import signal
            import subprocess
            import sys
            import time
            from pathlib import Path

            for caught in (signal.SIGTERM, signal.SIGINT):
                signal.signal(caught, lambda *_args: None)
            grandchild = subprocess.Popen([
                sys.executable,
                "-c",
                (
                    "import signal,time;"
                    "signal.signal(signal.SIGTERM,lambda *_:None);"
                    "signal.signal(signal.SIGINT,lambda *_:None);"
                    "time.sleep(300)"
                ),
            ])
            Path(sys.argv[2]).write_text(str(os.getpid()), encoding="utf-8")
            Path(sys.argv[3]).write_text(str(grandchild.pid), encoding="utf-8")
            Path(sys.argv[1]).write_text("ready", encoding="utf-8")
            while True:
                time.sleep(1)
            """
        ),
        encoding="utf-8",
    )
    driver = tmp_path / "lock_owner.py"
    driver.write_text(
        textwrap.dedent(
            f"""\
            import json
            import sys
            from pathlib import Path

            sys.path.insert(0, {str(SCRIPTS)!r})
            import m0_window_parent as parent

            receipt = Path({str(receipt)!r})
            ledger = Path({str(ledger)!r})
            authorization = parent.authorise_window(
                "W1",
                receipt_path=receipt,
                ledger_path=ledger,
            )
            Path({str(authorized)!r}).write_text(
                json.dumps(authorization), encoding="utf-8"
            )

            def command_builder(_handoff, _result):
                return [
                    sys.executable,
                    {str(spawner)!r},
                    {str(ready)!r},
                    {str(child_pid_path)!r},
                    {str(grandchild_pid_path)!r},
                ]

            parent._launch_authorized_child(
                window_id="W1",
                stage="parent_death_cpu_test",
                run_id=parent.WINDOWS["W1"]["run_id"],
                mode="clean",
                cache_path=Path({str(tmp_path / "cache")!r}),
                run_root=Path({str(run_root)!r}),
                authorization=authorization,
                timeout_seconds=300.0,
                command_builder=command_builder,
            )
            Path({str(later_stage)!r}).write_text("unsafe", encoding="utf-8")
            """
        ),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.update(
        {
            "GPUWRF_GPU_LOCK_HELD": "1",
            "GPUWRF_GPU_LOCK_TOKEN": token,
            "GPUWRF_GPU_LOCK_HOLDER_FILE": str(holder),
            "GPUWRF_GPU_LOCK_LABEL": parent.WINDOWS["W1"]["label"],
            "JAX_PLATFORMS": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            wsa.AUTHORITY_ENV_VAR: str(authority_path),
            **wsa.child_environment_binding(authority),
        }
    )
    lock_owner = subprocess.Popen(
        [sys.executable, str(driver)],
        cwd=REPO,
        env=environment,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    child_pid = None
    grandchild_pid = None
    try:
        _wait_for_path(ready)
        _wait_for_path(authorized)
        child_pid = int(child_pid_path.read_text())
        grandchild_pid = int(grandchild_pid_path.read_text())
        assert _process_is_running(child_pid)
        assert _process_is_running(grandchild_pid)

        lock_owner.send_signal(parent_death_signal)
        lock_owner.wait(timeout=10.0)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and (
            _process_is_running(child_pid)
            or _process_is_running(grandchild_pid)
        ):
            time.sleep(0.02)

        assert not _process_is_running(child_pid)
        assert not _process_is_running(grandchild_pid)
        assert not later_stage.exists()
        assert not (
            run_root / "parent_death_cpu_test" / "exact_boundary.json"
        ).exists()
        spent = parent.gpu_auth.CoordinationReceipt.load(receipt)
        with pytest.raises(parent.gpu_auth.WindowNotAuthorised, match="SPENT"):
            spent.check(parent.WINDOWS["W1"]["label"], ledger_path=ledger)
    finally:
        if lock_owner.poll() is None:
            lock_owner.kill()
            lock_owner.wait(timeout=5.0)
        for pid in (child_pid, grandchild_pid):
            if pid is not None and _process_is_running(pid):
                os.kill(pid, signal.SIGKILL)


def test_forecast_child_refuses_missing_handoff_before_import(tmp_path):
    result = tmp_path / "must-not-exist.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(contract.CHILD),
            "--session-label",
            parent.WINDOWS["W1"]["label"],
            "--window",
            "W1",
            "--stage",
            "cold_empty_cache_readiness",
            "--run-id",
            parent.WINDOWS["W1"]["run_id"],
            "--mode",
            "compile-only",
            "--handoff",
            str(tmp_path / "absent.json"),
            "--result",
            str(result),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(completed.stdout)
    assert completed.returncode == 2
    assert payload["status"] == "REFUSED_PRE_DEVICE_IMPORT"
    assert payload["jax_imported"] is False
    assert payload["gpuwrf_imported"] is False
    assert result.exists() is False


def test_pallas_native_refuses_missing_handoff_before_import(tmp_path):
    result = tmp_path / "must-not-exist.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(contract.PALLAS),
            "--native",
            "--window",
            "W3",
            "--stage",
            "native_pallas_if_suppressed_in_w1",
            "--run-id",
            "missing-auth-0002",
            "--handoff",
            str(tmp_path / "absent.json"),
            "--out",
            str(result),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(completed.stdout)
    assert completed.returncode == 2
    assert payload["status"] == "REFUSED_PRE_DEVICE_IMPORT"
    assert payload["jax_imported"] is False
    assert payload["gpuwrf_imported"] is False
    assert result.exists() is False


def test_inspection_is_real_graph_but_reads_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = list(tmp_path.iterdir())
    for window_id in ("W1", "W2", "W3"):
        graph = parent.inspect_window(window_id)
        assert graph["inspection_only"] is True
        assert graph["files_read"] == []
        assert graph["files_written"] == []
        assert graph["receipts_consumed"] == []
        assert all(
            stage.get("fresh_process") is True
            for stage in graph["stages"]
            if stage.get("mode") not in {"post-analysis", "parent-gate"}
        )
    assert list(tmp_path.iterdir()) == before


def test_static_refresh_refuses_old_exact_status_after_c1_source_change(
    tmp_path,
):
    output = tmp_path / "refreshed.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "refresh_m0_exact_boundary_evidence.py"),
            "--input",
            str(
                REPO
                / "proofs/v025/m0/m0_exact_boundary_cpu_evidence.json"
            ),
            "--output",
            str(output),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "terminal proof schema/status changed" in completed.stderr
    assert not output.exists()
