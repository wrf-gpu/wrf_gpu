#!/usr/bin/env python3
"""JAX-free validators for the Amendment-4 source, plan, and artifacts."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[2]
CONTRACT = Path(__file__).resolve()
CHILD = REPO / "scripts/v025/m0_exact_boundary_child.py"
PARENT = REPO / "scripts/v025/m0_window_parent.py"
PARENT_DEATH_GUARD = REPO / "scripts/v025/m0_parent_death_guard.py"
EXECUTOR = REPO / "scripts/v025/m0_three_window_executor.py"
PALLAS = REPO / "scripts/v025/pallas_sm120_spike.py"
SAMPLER = REPO / "scripts/v025/m0_vram_sampler.py"
PAIR_PREPARER = REPO / "scripts/v025/prepare_m0_matched_pair.py"
PALLAS_PREPARER = REPO / "scripts/v025/prepare_pallas_fast_savepoint.py"
EVIDENCE_BUILDER = REPO / "scripts/v025/build_m0_exact_boundary_evidence.py"
EVIDENCE_REFRESHER = REPO / "scripts/v025/refresh_m0_exact_boundary_evidence.py"
POSTLOCK_CENSUS = REPO / "scripts/v025/m0_postlock_census.py"
HOST_RSS_SAMPLER = REPO / "scripts/v025/m0_host_rss_sampler.py"
W1_FAST_PAIR = REPO / "scripts/v025/m0_w1_fast_pair.py"
C1_C2_PROOFS = REPO / "scripts/v025/m0_c1_c2_cpu_proofs.py"
NSYS_EXPORT = REPO / "scripts/v025/nsys_export.py"
RUN_FAST_PAIR = REPO / "scripts/v025/run_fast_pair.py"
CLOSURE_BUILDER = (
    REPO / "scripts/v025/build_m0_c1_c2_cpu_closure_evidence.py"
)
RESIDUAL_CLOSURE_BUILDER = (
    REPO / "scripts/v025/build_m0_c1_c2_cpu_residual_evidence.py"
)
LONG_RUN_CONTROLS = REPO / "scripts/v025/m0_long_run_controls.py"
CORE_SESSION_PROTOCOL = REPO / "scripts/v025/m0_core_session_protocol.py"
STATUS = "CPU_C1_C2_GREEN_GPU_WINDOWS_MISSING"
RANGE_NAME = "GPUWRF_M0_FORECAST_INTEGRATION"
ACCELERATOR_IMPORT_ROOTS = {"jax", "jaxlib", "gpuwrf"}
FROZEN_FUNCTION_HASHES = {
    "gpuwrf.runtime.operational_mode._assert_nonzero_initial_mu_total":
        "4dc8690a0f6181a24f0817242501d9a7599db372bca7677b013327a1ea96ecb1",
    "gpuwrf.runtime.operational_mode._operational_scan_state":
        "f0609a83ad8c7ee7421e4155a13632299495c9466ab38109ff2db2326785e1c1",
    "gpuwrf.runtime.operational_mode._dealias_pytree_buffers":
        "9e3bc0fd9be3ad694805578db05fc9bb7409646649cee82c9ff902e2f1257265",
    "gpuwrf.runtime.operational_mode._run_forecast_operational_jit":
        "85712542d47ada3716ea6343d5a46409957f5f70c8608f6a9e8f9f77fa0ae123",
    "gpuwrf.integration.daily_pipeline._build_real_case":
        "aff14597c9760987836e08382bf6f9d4a0f240ba69ae6caaceb4a6efc909a143",
    "gpuwrf.integration.daily_pipeline._capture_boundary_leaves":
        "e02ce49fe0c62fff014f7679dc8b3e709e8b9dc6c6f746acbdb01b5f49266613",
    "gpuwrf.integration.daily_pipeline._boundary_window_cadence_s":
        "23702a1d4d5154f523ed3dd9ec19d43fc11376ffe600b1f959595ea4c75e72db",
    "gpuwrf.integration.daily_pipeline._rewindow_boundary_leaves":
        "c85f42ce6e33237f3df20613ef8cc9a78cd95589ebaa880b65e86806726e83d4",
}


class ContractViolation(RuntimeError):
    """One frozen source or artifact invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractViolation(message)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    _require(len(found) == 1, f"expected exactly one function {name}, found {len(found)}")
    return found[0]


def _source_segment(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    _require(segment is not None, "AST source segment is unavailable")
    return str(segment)


def _import_roots(tree: ast.AST, *, top_level_only: bool = False) -> list[str]:
    nodes: Iterable[ast.AST]
    if top_level_only and isinstance(tree, ast.Module):
        nodes = tree.body
    else:
        nodes = ast.walk(tree)
    roots = []
    for node in nodes:
        if isinstance(node, ast.Import):
            roots.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.append(node.module.split(".")[0])
    return roots


def _literal_assignment(tree: ast.AST, name: str) -> Any:
    for node in getattr(tree, "body", []):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return ast.literal_eval(node.value)
    raise ContractViolation(f"no literal assignment for {name}")


def _assignment_expression(tree: ast.AST, name: str) -> str:
    for node in getattr(tree, "body", []):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            return ast.unparse(node.value)
    raise ContractViolation(f"no assignment for {name}")


def validate_source_texts(
    *,
    child_source: str,
    parent_source: str,
    parent_death_guard_source: str,
    executor_source: str,
    pallas_source: str,
    evidence_builder_source: str,
    long_run_controls_source: str,
    core_session_source: str,
) -> dict[str, Any]:
    """Reject source mutations at every security/measurement boundary."""

    child_tree = ast.parse(child_source)
    parent_tree = ast.parse(parent_source)
    parent_death_guard_tree = ast.parse(parent_death_guard_source)
    executor_tree = ast.parse(executor_source)
    pallas_tree = ast.parse(pallas_source)
    evidence_builder_tree = ast.parse(evidence_builder_source)
    long_run_controls_tree = ast.parse(long_run_controls_source)
    core_session_tree = ast.parse(core_session_source)

    # Parent and executor are entirely JAX/gpuwrf-free, including nested
    # post-analysis helpers. Device-capable children may import them only in
    # nested functions, never while the module is imported for inspection.
    for label, tree in (
        ("parent", parent_tree),
        ("parent-death guard", parent_death_guard_tree),
        ("executor", executor_tree),
    ):
        roots = set(_import_roots(tree))
        _require(
            not roots & ACCELERATOR_IMPORT_ROOTS,
            f"{label} imports accelerator roots {sorted(roots & ACCELERATOR_IMPORT_ROOTS)}",
        )
    for label, tree in (("forecast child", child_tree), ("Pallas child", pallas_tree)):
        roots = set(_import_roots(tree, top_level_only=True))
        _require(
            not roots & ACCELERATOR_IMPORT_ROOTS,
            f"{label} imports accelerator roots at module scope",
        )
    evidence_builder_imports = set(
        _import_roots(evidence_builder_tree, top_level_only=True)
    )
    _require(
        not evidence_builder_imports & ACCELERATOR_IMPORT_ROOTS,
        "exact evidence builder imports accelerator roots at module scope",
    )
    long_run_imports = set(_import_roots(long_run_controls_tree))
    _require(
        not long_run_imports & ACCELERATOR_IMPORT_ROOTS,
        "long-run controls import accelerator roots",
    )
    core_session_imports = set(_import_roots(core_session_tree))
    _require(
        not core_session_imports & ACCELERATOR_IMPORT_ROOTS,
        "Amendment-5 session protocol imports accelerator roots",
    )
    _require(
        _assignment_expression(core_session_tree, "SESSION_LABEL")
        == "window_registry.SESSION_LABEL"
        and _literal_assignment(core_session_tree, "GLOBAL_DEADLINE_SECONDS")
        == 4500.0
        and _literal_assignment(core_session_tree, "COLD_THRESHOLD_SECONDS")
        == 600.0
        and _literal_assignment(core_session_tree, "MAX_COLD_PROCESSES") == 3
        and _literal_assignment(core_session_tree, "REQUIRED_MANAGERS")
        == ("0:2", "0:3"),
        "Amendment-5 session/cold constants changed",
    )
    cold_classifier = _source_segment(
        core_session_source,
        _function(core_session_tree, "classify_cold_attempts"),
    )
    session_graph = _source_segment(
        core_session_source,
        _function(core_session_tree, "execute_session_graph"),
    )
    _require(
        'decision = "PASS"' in cold_classifier
        and 'decision = "FAIL"' in cold_classifier
        and "classifications[0] == classifications[1]" in cold_classifier
        and "unique processes" in cold_classifier
        and "unique empty caches" in cold_classifier,
        "two-of-three cold classifier no longer enforces its frozen denominator",
    )
    _require(
        "for index in range(1, MAX_COLD_PROCESSES + 1):" in session_graph
        and 'if decision["decision"] != "CONTINUE":' in session_graph
        and 'if decision["decision"] == "FAIL":' in session_graph
        and '"W2_PROFILED_CAPTURE"' in session_graph
        and '"W3_CLEAN_MATCHED_ARM"' in session_graph,
        "one-session graph no longer preserves cold continuation/suppression",
    )
    _require(
        _literal_assignment(
            long_run_controls_tree, "LONG_RUN_THRESHOLD_SECONDS"
        )
        == 600.0
        and _literal_assignment(
            long_run_controls_tree, "MEMORY_HEADROOM_RATIO"
        )
        == 2.0
        and _literal_assignment(
            long_run_controls_tree,
            "FOREIGN_PROCESS_PHYSICAL_RAM_RATIO",
        )
        == 0.25
        and _literal_assignment(long_run_controls_tree, "INHIBIT_WHAT")
        == "sleep:idle:handle-lid-switch",
        "long-run duration/memory/inhibitor constants changed",
    )
    memory_preflight = _source_segment(
        long_run_controls_source,
        _function(long_run_controls_tree, "memory_preflight"),
    )
    continuity = _source_segment(
        long_run_controls_source,
        _function(long_run_controls_tree, "validate_clock_continuity"),
    )
    inhibition = _source_segment(
        long_run_controls_source,
        _function(long_run_controls_tree, "inhibited_command"),
    )
    for fragment, message in (
        (
            "memory[\"MemAvailable\"] < required_available_bytes",
            "long-run MemAvailable refusal is absent",
        ),
        (
            "int(record[\"rss_bytes\"]) >= foreign_limit_bytes",
            "long-run foreign-process RSS refusal is absent",
        ),
        (
            "start.get(\"boot_id\") != end.get(\"boot_id\")",
            "long-run boot-ID invalidation is absent",
        ),
        (
            "abs(offset_change_ns) > tolerance_ns",
            "long-run suspend discontinuity invalidation is absent",
        ),
        (
            "f\"--what={INHIBIT_WHAT}\"",
            "long-run systemd inhibitor scope is absent",
        ),
    ):
        source = (
            memory_preflight
            if "memory" in fragment or "rss_bytes" in fragment
            else continuity
            if "boot_id" in fragment or "offset_change" in fragment
            else inhibition
        )
        _require(fragment in source, message)

    frozen = _literal_assignment(child_tree, "EXPECTED_FUNCTION_AST_SHA256")
    _require(frozen == FROZEN_FUNCTION_HASHES, "production function identities changed")
    _require(
        _literal_assignment(child_tree, "RANGE_NAME") == RANGE_NAME,
        "profiled range identity changed",
    )
    wrapper_sequence = _literal_assignment(child_tree, "WRAPPER_PREPARATION_SEQUENCE")
    _require(
        wrapper_sequence
        == (
            "gpuwrf.runtime.operational_mode._assert_nonzero_initial_mu_total",
            "gpuwrf.runtime.operational_mode._operational_scan_state",
            "gpuwrf.runtime.operational_mode._dealias_pytree_buffers",
        ),
        "wrapper preparation function identity/order changed",
    )

    prepare = _source_segment(
        child_source, _function(child_tree, "prepare_exact_call")
    )
    ordered_fragments = (
        "assert_nonzero(state)",
        "prepared_state = scan_state(state, namelist)",
        "prepared_state = dealias(prepared_state)",
        "return (prepared_state, namelist, hours), sequence",
    )
    positions = [prepare.find(fragment) for fragment in ordered_fragments]
    _require(all(position >= 0 for position in positions), "exact preparation call/args changed")
    _require(positions == sorted(positions), "exact preparation order changed")

    run_boundary = _source_segment(
        child_source, _function(child_tree, "run_boundary")
    )
    lower_fragment = "lowered = exact_jit.lower(*prepared_arguments)"
    compile_fragment = "executable = lowered.compile()"
    ready_fragment = "executable_ready_monotonic_ns = time.monotonic_ns()"
    argument_start_fragment = "argument_identity_start_ns = time.monotonic_ns()"
    argument_identity_fragment = (
        "argument_identity = _structural_argument_identity(jax, prepared_arguments)"
    )
    argument_end_fragment = "argument_identity_end_ns = time.monotonic_ns()"
    lowered_start_fragment = "lowered_identity_start_ns = time.monotonic_ns()"
    lowered_identity_fragment = "stablehlo = _stablehlo_identity(lowered)"
    lowered_hash_fragment = 'lowered_sha256 = stablehlo["sha256"]'
    lowered_end_fragment = "lowered_identity_end_ns = time.monotonic_ns()"
    invocation_branch_fragment = 'if mode == "profiled":'
    for fragment in (
        lower_fragment,
        compile_fragment,
        ready_fragment,
        argument_start_fragment,
        argument_identity_fragment,
        argument_end_fragment,
        lowered_start_fragment,
        lowered_identity_fragment,
        lowered_hash_fragment,
        lowered_end_fragment,
        invocation_branch_fragment,
    ):
        _require(fragment in run_boundary, f"missing exact boundary fragment {fragment!r}")
    boundary_positions = [
        run_boundary.index(lower_fragment),
        run_boundary.index(compile_fragment),
        run_boundary.index(ready_fragment),
        run_boundary.index(argument_start_fragment),
        run_boundary.index(argument_identity_fragment),
        run_boundary.index(argument_end_fragment),
        run_boundary.index(lowered_start_fragment),
        run_boundary.index(lowered_identity_fragment),
        run_boundary.index(lowered_hash_fragment),
        run_boundary.index(lowered_end_fragment),
        run_boundary.index(invocation_branch_fragment),
    ]
    _require(
        boundary_positions == sorted(boundary_positions),
        "proof identities are not strictly after readiness and before invocation",
    )
    _require(
        run_boundary.count("_invoke_and_synchronize(") == 2,
        "profiled and clean must call one shared invocation helper",
    )
    _require(
        f"jax.profiler.TraceAnnotation(RANGE_NAME)" in run_boundary,
        "profiled exact range is absent",
    )
    _require(
        "allocator = _allocator_stats(jax, result)" in run_boundary,
        "same-process profiled allocator read is absent",
    )
    _require(
        "_publish_exact_wrfout(" in run_boundary
        and "_host_pytree_and_digest(jax, result)" in run_boundary,
        "same-result post-integration wrfout adapter is absent",
    )

    isolated_child = _source_segment(
        evidence_builder_source,
        _function(evidence_builder_tree, "_isolated_arm_child"),
    )
    _require(
        'if kind == "local-exact":' in isolated_child
        and "payload = child.run_boundary(" in isolated_child
        and "payload = _public_wrapper_observation(" in isolated_child,
        "fresh CPU child does not execute both exact/public arm identities",
    )
    affinity_fragments = (
        "inherited_affinity = sorted(os.sched_getaffinity(0))",
        "os.sched_setaffinity(0, {inherited_affinity[0]})",
        "cpu_platform = assert_cpu_only()",
        "import jax",
    )
    affinity_positions = [
        isolated_child.find(fragment) for fragment in affinity_fragments
    ]
    _require(
        all(position >= 0 for position in affinity_positions)
        and affinity_positions == sorted(affinity_positions),
        "single-core CPU deadlock guard is not applied before JAX import",
    )
    _require(
        '"single_core_xla_cpu_deadlock_guard": True' in isolated_child,
        "isolated CPU arm does not record its single-core deadlock guard",
    )
    _require(
        "child._atomic_json_no_replace(output, envelope)" in isolated_child,
        "fresh CPU child result is not published atomically without replacement",
    )
    isolated_launcher = _source_segment(
        evidence_builder_source,
        _function(evidence_builder_tree, "_run_isolated_arm"),
    )
    _require(
        "completed = subprocess.run(" in isolated_launcher
        and '"--isolated-arm"' in isolated_launcher
        and "check=False" in isolated_launcher,
        "CPU arm launcher is not a real synchronous subprocess",
    )
    _require(
        "start_new_session" not in isolated_launcher,
        "CPU arm launcher detaches from the caller process group",
    )
    evidence_build = _source_segment(
        evidence_builder_source,
        _function(evidence_builder_tree, "build"),
    )
    long_run_preflight_fragments = (
        "output_preflight = longrun.output_directory_preflight(",
        "memory_preflight = longrun.memory_preflight()",
        "contract.validate_representative_long_run_fixtures()",
        "local_envelope, local_producer = _run_isolated_arm(",
    )
    long_run_preflight_positions = [
        evidence_build.find(fragment)
        for fragment in long_run_preflight_fragments
    ]
    _require(
        all(position >= 0 for position in long_run_preflight_positions)
        and long_run_preflight_positions
        == sorted(long_run_preflight_positions),
        "long-run output/memory/validator preflights are not before execution",
    )
    local_arm = 'kind="local-exact"'
    public_arm = 'kind="public-wrapper"'
    _require(
        evidence_build.count("_run_isolated_arm(") == 2
        and local_arm in evidence_build
        and public_arm in evidence_build,
        "exact/public CPU arms are not each launched in a fresh process",
    )
    _require(
        evidence_build.index(local_arm) < evidence_build.index(public_arm),
        "exact/public CPU arms are not sequential in frozen order",
    )
    _require(
        "child.run_boundary(" not in evidence_build
        and "_public_wrapper_observation(" not in evidence_build,
        "top-level evidence builder executes an expensive JAX arm in-process",
    )
    _require(
        "contract.validate_boundary_result(" not in evidence_build,
        "CPU semantic oracle is incorrectly subjected to GPU timing gates",
    )
    publish_position = run_boundary.find(
        "wrfout, _host_result, result_digest = _publish_exact_wrfout("
    )
    _require(
        publish_position > run_boundary.find(
            "if integration_start_ns is not None:"
        )
        and "result=result," in run_boundary[publish_position:],
        "wrfout publication can consume a substituted or unsynchronized result",
    )
    clean_branch = run_boundary.split('elif mode == "clean":', 1)
    _require(len(clean_branch) == 2, "clean branch is absent")
    clean_body = clean_branch[1].split("if integration_start_ns is not None:", 1)[0]
    _require(
        "profiler" not in clean_body and "_allocator_stats" not in clean_body,
        "clean branch contains profiler/resource instrumentation",
    )

    invoke = _source_segment(
        child_source, _function(child_tree, "_invoke_and_synchronize")
    )
    invoke_fragments = (
        "integration_start_ns = time.monotonic_ns()",
        "result = _call_compiled_with_exact_arguments(",
        "jax.block_until_ready(result)",
        "integration_end_ns = time.monotonic_ns()",
    )
    invoke_positions = [invoke.find(fragment) for fragment in invoke_fragments]
    _require(all(position >= 0 for position in invoke_positions), "compiled call/sync changed")
    _require(invoke_positions == sorted(invoke_positions), "compiled call/sync order changed")
    compiled_adapter = _source_segment(
        child_source, _function(child_tree, "_call_compiled_with_exact_arguments")
    )
    _require(
        "if float(hours) != float(lowered_static_hours):" in compiled_adapter
        and "return executable(prepared_state, namelist)" in compiled_adapter,
        "static-hours binding or exact dynamic compiled call changed",
    )
    leaf_semantics = _source_segment(
        child_source, _function(child_tree, "_leaf_semantics")
    )
    pytree_digest = _source_segment(
        child_source, _function(child_tree, "_host_pytree_and_digest")
    )
    serialized_treedef = _source_segment(
        child_source, _function(child_tree, "_serialized_treedef")
    )
    _require(
        '"treedef": _treedef_identity(treedef)' in leaf_semantics
        and "hashlib.sha256(_serialized_treedef(treedef))" in pytree_digest
        and "pickle.dumps(treedef, protocol=5)" in serialized_treedef,
        "argument identity uses an unstable PyTree definition representation",
    )
    _require(
        "str(treedef)" not in leaf_semantics
        and "str(treedef)" not in pytree_digest,
        "address-bearing PyTree debug text is forbidden",
    )

    child_main = _source_segment(child_source, _function(child_tree, "main"))
    _require(
        child_main.index("consume_authorization_handoff(")
        < child_main.index("run_boundary("),
        "forecast child can import/run before consuming authorization",
    )
    pallas_main = _source_segment(pallas_source, _function(pallas_tree, "main"))
    _require(
        pallas_main.count("consume_authorization_handoff(") == 1,
        "Pallas child authorization must be consumed exactly once",
    )
    _require(
        pallas_main.index("consume_authorization_handoff(")
        < pallas_main.index("run_native(args, authorization)"),
        "Pallas child can import/run before consuming authorization",
    )

    parent_launch = _source_segment(
        parent_source, _function(parent_tree, "_launch_authorized_child")
    )
    launch_fragments = (
        "launch_ns = time.monotonic_ns()",
        'environment["GPUWRF_M0_PARENT_LAUNCH_NS"] = str(launch_ns)',
        "process = subprocess.Popen(",
    )
    launch_positions = [parent_launch.find(fragment) for fragment in launch_fragments]
    _require(all(position >= 0 for position in launch_positions), "parent launch endpoint changed")
    _require(launch_positions == sorted(launch_positions), "launch endpoint is not pre-Popen")
    sampler_start_position = parent_launch.find("sampler.start()")
    process_launch_position = parent_launch.find("process = subprocess.Popen(")
    _require(
        sampler_start_position >= 0
        and process_launch_position >= 0
        and sampler_start_position < process_launch_position,
        "profiled residency baseline/streams do not start before forecast launch",
    )
    parent_death_fragments = (
        "environment[pdg.LOCK_OWNER_PID_ENV] = str(lock_owner_pid)",
        "guarded_command = [",
        "process = subprocess.Popen(",
        "start_new_session=True",
    )
    _require(
        all(fragment in parent_launch for fragment in parent_death_fragments),
        "forecast/Pallas launch is not parent-death guarded",
    )
    guard_arm = _source_segment(
        parent_death_guard_source,
        _function(parent_death_guard_tree, "arm_parent_death_signal"),
    )
    _require(
        guard_arm.count("if os.getppid() != expected_parent_pid:") == 2
        and (
            "libc.prctl(PR_SET_PDEATHSIG, int(death_signal), 0, 0, 0)"
            in guard_arm
        )
        and guard_arm.count("_die_now()") == 2,
        "parent-death arm does not close the fork-to-prctl race",
    )
    guard_kill = _source_segment(
        parent_death_guard_source,
        _function(parent_death_guard_tree, "_kill_guard_process_group"),
    )
    _require(
        "os.killpg(os.getpgrp(), signal.SIGKILL)" in guard_kill,
        "parent-death handler does not unconditionally kill the child group",
    )
    guard_run = _source_segment(
        parent_death_guard_source,
        _function(parent_death_guard_tree, "run_guard"),
    )
    guard_self_arm_position = guard_run.find(
        "arm_parent_death_signal(expected_lock_owner_pid, signal.SIGTERM)"
    )
    handler_position = guard_run.find(
        "signal.signal(signal_number, _kill_guard_process_group)"
    )
    child_position = guard_run.find("child = subprocess.Popen(")
    _require(
        guard_self_arm_position >= 0
        and handler_position >= 0
        and child_position >= 0
        and guard_self_arm_position < child_position
        and handler_position < child_position
        and "start_new_session=False" in guard_run
        and "signal.SIGKILL," in guard_run,
        "guard does not arm group cleanup before launching its child",
    )
    _require(
        "preexec_fn" not in parent_launch,
        "threaded manager must not run Python in a forecast Popen preexec_fn",
    )
    _require(
        "process_group = process.pid" in parent_launch
        and parent_launch.count(
            "_kill_process_group(process, process_group=process_group)"
        )
        >= 2,
        "nonzero/exception paths do not sweep the captured child process group",
    )
    _require(
        "exact_contract.validate_wrfout_binding(" in parent_launch
        and "verify_file=True" in parent_launch,
        "parent does not mechanically validate the complete wrfout binding",
    )
    run_w3 = _source_segment(parent_source, _function(parent_tree, "_run_w3"))
    _require(
        "pair_gate = {" in run_w3
        and '"status": "PASS"' in run_w3
        and "pallas" not in run_w3.lower(),
        "Amendment-6 W3 lost its exact pair gate or regained native Pallas",
    )
    run_w2 = _source_segment(parent_source, _function(parent_tree, "_run_w2"))
    _require(
        "artifacts = _require_profiled_capture_artifacts(profiled)" in run_w2
        and '"postlock_export_and_census_status": "MISSING"' in run_w2,
        "W2 does not fail closed on profiled capture integrity/MISSING census",
    )
    post_w1 = _source_segment(
        parent_source, _function(parent_tree, "_post_w1_qualification")
    )
    _require(
        'pair.get("schema") != "wrf_gpu2.v025.m0.fast_case_qualification.v1"'
        in post_w1
        and "if binding != expected_binding:" in post_w1
        and 'completeness.get("completeness_percent") != 100' in post_w1,
        "W1 qualification can accept an unbound or incomplete FAST pair",
    )
    post_analysis = _source_segment(
        parent_source, _function(parent_tree, "post_analysis")
    )
    _require(
        "validate_lock_release_proof(" in post_analysis
        and "postlock.analyze_w2(" in post_analysis,
        "post-analysis is not release-proven or W2 census-enabled",
    )
    held_session = _source_segment(
        executor_source, _function(executor_tree, "run_held_session")
    )
    outer_owner = _source_segment(
        executor_source, _function(executor_tree, "run_session_owner")
    )
    _require(
        "pallas_sm120_spike" not in held_session
        and "PALLAS_CHILD" not in held_session
        and "hlo_dump" not in held_session.lower()
        and "_held_session_wrapper_command(" in outer_owner
        and "CPU_PREFLIGHT" in outer_owner
        and "build_session_lock_release_proof(" in outer_owner,
        "M0-CORE regained deferred microscopy/Pallas or lost its outer owner",
    )
    outer_graph = _source_segment(
        executor_source, _function(executor_tree, "run_outer_window_graph")
    )
    outer_fragments = (
        "process = subprocess.Popen(",
        "returncode = process.wait(",
        "returned_ns = time.monotonic_ns()",
        "postlock.build_lock_release_proof(",
        "post = subprocess.run(",
    )
    outer_positions = [outer_graph.find(fragment) for fragment in outer_fragments]
    _require(
        all(position >= 0 for position in outer_positions)
        and outer_positions == sorted(outer_positions),
        "outer graph does not prove wrapper return before post process launch",
    )
    output_adapter = _source_segment(
        child_source, _function(child_tree, "_publish_exact_wrfout")
    )
    adapter_fragments = (
        "if os.path.lexists(target_dir):",
        "materialization_start_ns = time.monotonic_ns()",
        "if materialization_start_ns < integration_end_ns:",
        "daily._surface_diagnostics_for_output(",
        "full_variable_set = daily._full_wrfout_variables_enabled(config)",
        "host_result, result_digest = _host_pytree_and_digest(jax, result)",
        "wrfout_writer.prepare_wrfout_payload(",
        "wrfout_writer.write_prepared_wrfout(",
        "daily.build_wrfout_inventory([target])",
        "output_sha256 = sha256_file(target)",
    )
    positions = [output_adapter.find(fragment) for fragment in adapter_fragments]
    _require(
        all(position >= 0 for position in positions)
        and positions == sorted(positions),
        "exact-result wrfout adapter order/source changed",
    )
    _require(
        "run_forecast" not in output_adapter
        and "os.replace" not in output_adapter,
        "wrfout adapter can recompute a forecast or replace stale output",
    )
    for exact_argument in (
        "host_result,",
        "case.grid,",
        "case.namelist,",
        "domain_authority=case.writer_domain_authority,",
        "valid_time=valid_time,",
        "lead_hours=float(hours),",
        "run_start=case.run_start,",
        "diagnostics=diagnostics,",
        "full_variable_set=full_variable_set,",
    ):
        _require(
            exact_argument in output_adapter,
            f"wrfout adapter production argument changed: {exact_argument}",
        )
    _require(
        "\n        domain=config.domain," in output_adapter,
        "wrfout adapter production domain changed",
    )
    for binding_fragment in (
        "config_identity = json.loads(",
        '"namelist_input_sha256": sha256_file(config_path),',
        '"variable_inventory_sha256": canonical_sha256(variable_inventory),',
        '"dimensions_sha256": canonical_sha256(dimensions),',
        '"inventory_sha256": canonical_sha256(inventory),',
        '"finiteness_sha256": canonical_sha256(finiteness),',
        '"adapter_identity_sha256": canonical_sha256(adapter_identity),',
        '"adapter_source_sha256": canonical_sha256(adapter_source),',
        '"operational_variable_set": True,',
        '"full_variable_set": bool(full_variable_set),',
    ):
        _require(
            binding_fragment in output_adapter,
            f"wrfout proof binding changed: {binding_fragment}",
        )
    stablehlo_identity = _source_segment(
        child_source, _function(child_tree, "_stablehlo_identity")
    )
    _require(
        '.encode("utf-8")' in stablehlo_identity
        and "for offset in range(" in stablehlo_identity
        and '"full_encoded_copy_materialized": False' in stablehlo_identity,
        "StableHLO identity is not streamed in bounded chunks",
    )

    return {
        "status": "PASS",
        "parent_accelerator_imports": [],
        "parent_death_guard_accelerator_imports": [],
        "executor_accelerator_imports": [],
        "forecast_child_top_level_accelerator_imports": [],
        "pallas_child_top_level_accelerator_imports": [],
        "evidence_builder_top_level_accelerator_imports": [],
        "production_function_hashes": frozen,
        "wrapper_preparation_sequence": list(wrapper_sequence),
        "exact_lower_compile": True,
        "direct_synchronized_invocation": True,
        "profiled_clean_shared_helper": True,
        "profiled_exact_range": RANGE_NAME,
        "clean_instrumentation_absent": True,
        "authorization_before_child_import": True,
        "parent_death_kills_child_process_group": True,
        "nonzero_child_final_group_sweep": True,
        "exact_result_wrfout_adapter": True,
        "stablehlo_streamed_encoding": True,
        "sequential_fresh_cpu_arm_processes": True,
        "cpu_arm_process_group_attached": True,
        "single_core_cpu_arm_before_jax": True,
        "long_run_preflight_and_inhibition": True,
        "amendment5_one_session_two_of_three": True,
    }


def validate_sources(
    *,
    child_path: Path = CHILD,
    parent_path: Path = PARENT,
    parent_death_guard_path: Path = PARENT_DEATH_GUARD,
    executor_path: Path = EXECUTOR,
    pallas_path: Path = PALLAS,
) -> dict[str, Any]:
    result = validate_source_texts(
        child_source=Path(child_path).read_text(encoding="utf-8"),
        parent_source=Path(parent_path).read_text(encoding="utf-8"),
        parent_death_guard_source=Path(parent_death_guard_path).read_text(
            encoding="utf-8"
        ),
        executor_source=Path(executor_path).read_text(encoding="utf-8"),
        pallas_source=Path(pallas_path).read_text(encoding="utf-8"),
        evidence_builder_source=EVIDENCE_BUILDER.read_text(encoding="utf-8"),
        long_run_controls_source=LONG_RUN_CONTROLS.read_text(
            encoding="utf-8"
        ),
        core_session_source=CORE_SESSION_PROTOCOL.read_text(
            encoding="utf-8"
        ),
    )
    result["source_sha256"] = {
        str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(child_path),
            Path(parent_path),
            Path(parent_death_guard_path),
            Path(executor_path),
            Path(pallas_path),
            SAMPLER,
            PAIR_PREPARER,
            PALLAS_PREPARER,
            CONTRACT,
            EVIDENCE_BUILDER,
            EVIDENCE_REFRESHER,
            POSTLOCK_CENSUS,
            HOST_RSS_SAMPLER,
            W1_FAST_PAIR,
            C1_C2_PROOFS,
            NSYS_EXPORT,
            RUN_FAST_PAIR,
            CLOSURE_BUILDER,
            RESIDUAL_CLOSURE_BUILDER,
            LONG_RUN_CONTROLS,
            CORE_SESSION_PROTOCOL,
        )
    }
    return result


def _positive_bounded(value: Any, maximum: float, name: str) -> float:
    _require(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and 0.0 < float(value) <= maximum,
        f"{name}={value!r} is not in (0,{maximum}]",
    )
    return float(value)


def validate_wrfout_binding(
    payload: dict[str, Any],
    *,
    verify_file: bool = False,
) -> dict[str, Any]:
    """Validate the exact result/case/output binding and optional file bytes."""

    wrfout = payload.get("wrfout") or {}
    result = payload.get("result") or {}
    case = payload.get("case") or {}
    call = payload.get("call") or {}
    _require(
        wrfout.get("schema") == "wrf_gpu2.v025.m0.exact_result_wrfout.v1"
        and wrfout.get("status") == "PASS",
        "same-result wrfout proof is missing",
    )
    _require(
        wrfout.get("run_id") == payload.get("run_id")
        and wrfout.get("result_exact_value_sha256")
        == result.get("exact_value_sha256"),
        "wrfout run/result binding mismatch",
    )
    exact_identity = wrfout.get("exact_boundary_identity")
    _require(
        isinstance(exact_identity, dict)
        and wrfout.get("exact_boundary_identity_sha256")
        == canonical_sha256(exact_identity),
        "wrfout exact-boundary identity hash mismatch",
    )
    input_binding = wrfout.get("input_binding") or {}
    for field in (
        "config_sha256",
        "namelist_input_sha256",
        "input_manifest_sha256",
        "case_sha256",
    ):
        _require(
            isinstance(input_binding.get(field), str)
            and len(input_binding[field]) == 64,
            f"wrfout input binding {field} is missing",
        )
    _require(
        input_binding["case_sha256"] == canonical_sha256(case)
        and exact_identity.get("case_sha256") == input_binding["case_sha256"],
        "wrfout case identity differs from the exact call",
    )
    hours = float(call.get("hours", case.get("hours", -1)))
    _require(
        math.isfinite(hours)
        and hours > 0
        and math.isclose(float(case.get("hours")), hours)
        and math.isclose(float(wrfout.get("lead_hours")), hours)
        and math.isclose(float(exact_identity.get("hours")), hours),
        "wrfout lead differs from the exact call",
    )
    _require(
        wrfout.get("domain") == case.get("domain"),
        "wrfout domain differs from the exact case",
    )
    try:
        from datetime import datetime
        run_start = datetime.fromisoformat(str(wrfout["run_start_utc"]))
        valid_time = datetime.fromisoformat(str(wrfout["valid_time_utc"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractViolation("wrfout time authority is malformed") from exc
    _require(
        run_start.tzinfo is not None
        and valid_time.tzinfo is not None
        and math.isclose(
            (valid_time - run_start).total_seconds(),
            hours * 3600.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ),
        "wrfout valid time differs from run start plus exact lead",
    )
    for field in (
        "final_wrfout_sha256",
        "domain_authority_sha256",
    ):
        _require(
            isinstance(wrfout.get(field), str)
            and len(wrfout[field]) == 64,
            f"wrfout {field} is missing",
        )
    _require(
        isinstance(wrfout.get("final_wrfout_path"), str)
        and wrfout.get("final_wrfout_path")
        and isinstance(wrfout.get("final_wrfout_bytes"), int)
        and not isinstance(wrfout.get("final_wrfout_bytes"), bool)
        and wrfout["final_wrfout_bytes"] > 0,
        "wrfout path/size is invalid",
    )
    _require(
        wrfout.get("operational_variable_set") is True
        and (wrfout.get("inventory") or {}).get("status") == "PASS"
        and (wrfout.get("finiteness") or {}).get("status") == "PASS",
        "wrfout operational inventory/finiteness is not proven",
    )
    binding_details = {
        "variable_inventory_sha256": wrfout.get(
            "variable_inventory_sha256"
        ),
        "dimensions_sha256": wrfout.get("dimensions_sha256"),
        "inventory_sha256": wrfout.get("inventory_sha256"),
        "finiteness_sha256": wrfout.get("finiteness_sha256"),
        "adapter_identity_sha256": wrfout.get(
            "adapter_identity_sha256"
        ),
        "adapter_source_sha256": wrfout.get("adapter_source_sha256"),
    }
    _require(
        isinstance(wrfout.get("config_identity"), dict)
        and input_binding["config_sha256"]
        == canonical_sha256(wrfout["config_identity"])
        and isinstance(wrfout.get("variable_inventory"), list)
        and wrfout["variable_inventory"]
        and binding_details["variable_inventory_sha256"]
        == canonical_sha256(wrfout["variable_inventory"])
        and isinstance(wrfout.get("dimensions"), dict)
        and wrfout["dimensions"]
        and binding_details["dimensions_sha256"]
        == canonical_sha256(wrfout["dimensions"])
        and binding_details["inventory_sha256"]
        == canonical_sha256(wrfout["inventory"])
        and binding_details["finiteness_sha256"]
        == canonical_sha256(wrfout["finiteness"]),
        "wrfout config/variable/dimension/inventory/finiteness binding changed",
    )
    required_adapters = {
        "gpuwrf.integration.daily_pipeline._surface_diagnostics_for_output",
        "gpuwrf.integration.daily_pipeline._merge_output_diagnostics",
        "gpuwrf.integration.daily_pipeline._wrfout_name",
        "gpuwrf.integration.daily_pipeline.build_wrfout_inventory",
        "gpuwrf.io.wrfout_writer.prepare_wrfout_payload",
        "gpuwrf.io.wrfout_writer.write_prepared_wrfout",
    }
    adapter_identity = wrfout.get("adapter_identity")
    _require(
        isinstance(adapter_identity, list)
        and {record.get("fqname") for record in adapter_identity}
        == required_adapters
        and all(
            isinstance(record.get("ast_sha256"), str)
            and len(record["ast_sha256"]) == 64
            for record in adapter_identity
        ),
        "wrfout production adapter identity is incomplete",
    )
    _require(
        (wrfout.get("adapter_source") or {}).get("src_gpuwrf_tree")
        == "a6885ceded260df2f5777d7366d75a5d38947cb7"
        and (wrfout.get("adapter_source") or {}).get("child_path")
        == str(CHILD.resolve())
        and (wrfout.get("adapter_source") or {}).get("child_sha256")
        == sha256_file(CHILD),
        "wrfout production source tree changed",
    )
    _require(
        binding_details["adapter_identity_sha256"]
        == canonical_sha256(adapter_identity)
        and binding_details["adapter_source_sha256"]
        == canonical_sha256(wrfout["adapter_source"]),
        "wrfout adapter identity/source hash changed",
    )
    binding_payload = {
        "run_id": wrfout["run_id"],
        "result_exact_value_sha256": wrfout[
            "result_exact_value_sha256"
        ],
        "exact_boundary_identity_sha256": wrfout[
            "exact_boundary_identity_sha256"
        ],
        **input_binding,
        "final_wrfout_path": wrfout["final_wrfout_path"],
        "final_wrfout_bytes": wrfout["final_wrfout_bytes"],
        "final_wrfout_sha256": wrfout["final_wrfout_sha256"],
        "domain": wrfout["domain"],
        "domain_authority_sha256": wrfout[
            "domain_authority_sha256"
        ],
        "run_start_utc": wrfout["run_start_utc"],
        "valid_time_utc": wrfout["valid_time_utc"],
        "lead_hours": float(wrfout["lead_hours"]),
        "operational_variable_set": True,
        "full_variable_set": bool(wrfout.get("full_variable_set")),
        **binding_details,
    }
    _require(
        wrfout.get("binding_sha256") == canonical_sha256(binding_payload),
        "wrfout canonical binding hash mismatch",
    )
    if verify_file:
        path = Path(wrfout["final_wrfout_path"])
        _require(
            not path.is_symlink()
            and path.is_file()
            and path.stat().st_size == wrfout["final_wrfout_bytes"]
            and sha256_file(path) == wrfout["final_wrfout_sha256"],
            "wrfout file bytes/hash changed",
        )
    return {
        "status": "PASS",
        "run_id": wrfout["run_id"],
        "binding_sha256": wrfout["binding_sha256"],
    }


def representative_boundary_fixture(mode: str) -> dict[str, Any]:
    """Return a compact real-case-shaped fixture for every payload validator.

    This fixture is deliberately not device evidence.  It exists so an
    expensive entrypoint can prove that every validator it will invoke later
    accepts one internally consistent payload before spending long-run time.
    """

    if mode not in {"compile-only", "profiled", "clean"}:
        raise ContractViolation(f"unknown representative fixture mode {mode!r}")
    is_compile = mode == "compile-only"
    is_profiled = mode == "profiled"
    run_id = f"cpu-validator-fixture-{mode}"
    wrapper_sequence = [
        "gpuwrf.runtime.operational_mode._assert_nonzero_initial_mu_total",
        "gpuwrf.runtime.operational_mode._operational_scan_state",
        "gpuwrf.runtime.operational_mode._dealias_pytree_buffers",
    ]
    case = {
        "domain": "d01",
        "hours": 1.0,
        "case_metadata": {"fixture": "real-case-shaped-validator"},
    }
    payload: dict[str, Any] = {
        "schema": "wrf_gpu2.v025.m0.exact_executable_boundary.v1",
        "status": "OK",
        "run_id": run_id,
        "case": case,
        "production_binding": {
            "wrapper_preparation_sequence": wrapper_sequence,
            "wrapper_preparation_identity": [
                {
                    "fqname": name,
                    "ast_sha256": FROZEN_FUNCTION_HASHES[name],
                }
                for name in wrapper_sequence
            ],
            "jit_identity": {
                "fqname":
                    "gpuwrf.runtime.operational_mode."
                    "_run_forecast_operational_jit",
                "ast_sha256": FROZEN_FUNCTION_HASHES[
                    "gpuwrf.runtime.operational_mode."
                    "_run_forecast_operational_jit"
                ],
            },
            "case_preparation_sequence": [
                "gpuwrf.integration.daily_pipeline._build_real_case",
                "gpuwrf.integration.daily_pipeline._capture_boundary_leaves",
                "gpuwrf.integration.daily_pipeline."
                "_boundary_window_cadence_s",
                "gpuwrf.integration.daily_pipeline._rewindow_boundary_leaves",
            ],
        },
        "call": {
            "argument_identity": {"exact_value_sha256": "1" * 64},
            "lower_api":
                "_run_forecast_operational_jit.lower("
                "*prepared_arguments).compile()",
            "compiled_invocation": (
                "_call_compiled_with_exact_arguments("
                "executable, prepared_arguments, "
                "lowered_static_hours=hours)"
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
            "nvtx_range": RANGE_NAME if is_profiled else None,
            "allocator_sidecar":
                "/fixture/allocator.json" if is_profiled else None,
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
        "wrfout": None,
    }
    if is_compile:
        return payload

    exact_identity = {
        "case_sha256": canonical_sha256(case),
        "hours": 1.0,
        "fixture": "exact-boundary-validator",
    }
    config_identity = {
        "run_id": run_id,
        "domain": "d01",
        "hours": 1,
    }
    input_binding = {
        "config_sha256": canonical_sha256(config_identity),
        "namelist_input_sha256": "6" * 64,
        "input_manifest_sha256": "8" * 64,
        "case_sha256": canonical_sha256(case),
    }
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
    inventory = {"status": "PASS"}
    finiteness = {"status": "PASS"}
    adapters = [
        {
            "fqname": name,
            "ast_sha256": "9" * 64,
        }
        for name in (
            "gpuwrf.integration.daily_pipeline."
            "_surface_diagnostics_for_output",
            "gpuwrf.integration.daily_pipeline._merge_output_diagnostics",
            "gpuwrf.integration.daily_pipeline._wrfout_name",
            "gpuwrf.integration.daily_pipeline.build_wrfout_inventory",
            "gpuwrf.io.wrfout_writer.prepare_wrfout_payload",
            "gpuwrf.io.wrfout_writer.write_prepared_wrfout",
        )
    ]
    adapter_source = {
        "src_gpuwrf_tree":
            "a6885ceded260df2f5777d7366d75a5d38947cb7",
        "child_path": str(CHILD.resolve()),
        "child_sha256": sha256_file(CHILD),
    }
    wrfout: dict[str, Any] = {
        "schema": "wrf_gpu2.v025.m0.exact_result_wrfout.v1",
        "status": "PASS",
        "run_id": run_id,
        "result_exact_value_sha256": "3" * 64,
        "exact_boundary_identity": exact_identity,
        "exact_boundary_identity_sha256":
            canonical_sha256(exact_identity),
        "input_binding": input_binding,
        "adapter_identity": adapters,
        "adapter_identity_sha256": canonical_sha256(adapters),
        "adapter_source": adapter_source,
        "adapter_source_sha256": canonical_sha256(adapter_source),
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
        "inventory": inventory,
        "finiteness": finiteness,
        "config_identity": config_identity,
        "variable_inventory": variable_inventory,
        "variable_inventory_sha256":
            canonical_sha256(variable_inventory),
        "dimensions": dimensions,
        "dimensions_sha256": canonical_sha256(dimensions),
        "inventory_sha256": canonical_sha256(inventory),
        "finiteness_sha256": canonical_sha256(finiteness),
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
    binding_details = {
        "variable_inventory_sha256":
            wrfout["variable_inventory_sha256"],
        "dimensions_sha256": wrfout["dimensions_sha256"],
        "inventory_sha256": wrfout["inventory_sha256"],
        "finiteness_sha256": wrfout["finiteness_sha256"],
        "adapter_identity_sha256":
            wrfout["adapter_identity_sha256"],
        "adapter_source_sha256": wrfout["adapter_source_sha256"],
    }
    wrfout["binding_sha256"] = canonical_sha256(
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
            **binding_details,
        }
    )
    payload["wrfout"] = wrfout
    return payload


def validate_representative_long_run_fixtures() -> dict[str, Any]:
    """Exercise every exact long-run payload validator before eligibility."""

    compile_only = representative_boundary_fixture("compile-only")
    profiled = representative_boundary_fixture("profiled")
    clean = representative_boundary_fixture("clean")
    results = {
        "compile_only": validate_boundary_result(
            compile_only, expected_mode="compile-only"
        ),
        "profiled": validate_boundary_result(
            profiled, expected_mode="profiled"
        ),
        "clean": validate_boundary_result(clean, expected_mode="clean"),
        "profiled_wrfout": validate_wrfout_binding(profiled),
        "clean_wrfout": validate_wrfout_binding(clean),
        "profiled_clean": compare_profiled_clean(profiled, clean),
    }
    return {
        "status": "PASS",
        "fixture_is_runtime_or_device_evidence": False,
        "validators_exercised": [
            "validate_boundary_result:compile-only",
            "validate_boundary_result:profiled",
            "validate_boundary_result:clean",
            "validate_wrfout_binding:profiled",
            "validate_wrfout_binding:clean",
            "compare_profiled_clean",
        ],
        "fixture_sha256": {
            "compile-only": canonical_sha256(compile_only),
            "profiled": canonical_sha256(profiled),
            "clean": canonical_sha256(clean),
        },
        "results": results,
    }


def validate_boundary_result(payload: dict[str, Any], *, expected_mode: str) -> dict[str, Any]:
    _require(payload.get("status") == "OK", "boundary status is not OK")
    _require(payload.get("schema") == "wrf_gpu2.v025.m0.exact_executable_boundary.v1",
             "boundary schema changed")
    production = payload.get("production_binding") or {}
    _require(
        production.get("wrapper_preparation_sequence")
        == [
            "gpuwrf.runtime.operational_mode._assert_nonzero_initial_mu_total",
            "gpuwrf.runtime.operational_mode._operational_scan_state",
            "gpuwrf.runtime.operational_mode._dealias_pytree_buffers",
        ],
        "artifact preparation order changed",
    )
    identities = production.get("wrapper_preparation_identity") or []
    _require(
        [record.get("ast_sha256") for record in identities]
        == [
            FROZEN_FUNCTION_HASHES[name]
            for name in production["wrapper_preparation_sequence"]
        ],
        "artifact function identities changed",
    )
    call = payload.get("call") or {}
    _require(
        call.get("lower_api")
        == "_run_forecast_operational_jit.lower(*prepared_arguments).compile()",
        "artifact lower/compile arguments changed",
    )
    _require(
        call.get("compiled_invocation")
        == (
            "_call_compiled_with_exact_arguments("
            "executable, prepared_arguments, lowered_static_hours=hours)"
        ),
        "artifact compiled invocation arguments changed",
    )
    _require(
        call.get("jax_compiled_dynamic_invocation")
        == "executable(prepared_state, namelist)",
        "JAX dynamic compiled invocation changed",
    )
    static = call.get("static_argument_binding") or {}
    _require(
        static.get("name") == "hours"
        and static.get("runtime_revalidated_before_dynamic_call") is True,
        "static hours binding is not fail-closed",
    )
    _require(
        isinstance(call.get("lowered_program_sha256"), str)
        and len(call["lowered_program_sha256"]) == 64,
        "artifact lowered-program identity missing",
    )
    timing = payload.get("timing") or {}
    _positive_bounded(timing.get("readiness_seconds"), 600.0, "readiness_seconds")
    _require(
        timing.get("child_executable_ready_monotonic_ns")
        >= timing.get("parent_process_launch_monotonic_ns"),
        "readiness precedes parent launch",
    )
    _require(
        math.isclose(
            float(timing["readiness_seconds"]),
            (
                timing["child_executable_ready_monotonic_ns"]
                - timing["parent_process_launch_monotonic_ns"]
            )
            / 1e9,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "readiness seconds do not match the direct endpoints",
    )
    proof_endpoints = [
        timing.get("child_executable_ready_monotonic_ns"),
        timing.get("argument_identity_start_monotonic_ns"),
        timing.get("argument_identity_end_monotonic_ns"),
        timing.get("lowered_identity_start_monotonic_ns"),
        timing.get("lowered_identity_end_monotonic_ns"),
    ]
    _require(
        all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in proof_endpoints
        )
        and proof_endpoints == sorted(proof_endpoints),
        "proof identity endpoints are not after executable readiness",
    )
    _positive_bounded(
        timing.get("argument_identity_seconds"),
        600.0,
        "argument_identity_seconds",
    )
    _positive_bounded(
        timing.get("lowered_identity_seconds"),
        600.0,
        "lowered_identity_seconds",
    )
    _positive_bounded(
        timing.get("proof_identity_seconds"),
        1200.0,
        "proof_identity_seconds",
    )
    _require(
        math.isclose(
            float(timing["argument_identity_seconds"]),
            (
                timing["argument_identity_end_monotonic_ns"]
                - timing["argument_identity_start_monotonic_ns"]
            )
            / 1e9,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(timing["lowered_identity_seconds"]),
            (
                timing["lowered_identity_end_monotonic_ns"]
                - timing["lowered_identity_start_monotonic_ns"]
            )
            / 1e9,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(timing["proof_identity_seconds"]),
            (
                timing["lowered_identity_end_monotonic_ns"]
                - timing["argument_identity_start_monotonic_ns"]
            )
            / 1e9,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "proof identity seconds do not match the direct endpoints",
    )
    _require(
        timing.get("proof_identity_definition")
        == (
            "after executable readiness -> exact argument and lowered "
            "StableHLO identities -> before compiled invocation"
        ),
        "proof identity clock definition changed",
    )
    _require(timing.get("derived_by_phase_subtraction") is False,
             "phase subtraction is forbidden")
    _require(timing.get("first_lazy_call_used_as_readiness") is False,
             "first lazy call is forbidden")
    instrumentation = payload.get("instrumentation") or {}
    _require(instrumentation.get("mode") == expected_mode, "mode mismatch")
    result = payload.get("result") or {}
    if expected_mode == "compile-only":
        _require(call.get("compiled_invocation_count") == 0, "compile-only invoked")
        _require(timing.get("integration_start_monotonic_ns") is None,
                 "compile-only has integration start")
        _require(result.get("synchronization") is None, "compile-only synchronized result")
    else:
        _require(call.get("compiled_invocation_count") == 1, "not exactly one invocation")
        _positive_bounded(timing.get("integration_seconds"), 300.0, "integration_seconds")
        _require(
            timing["integration_start_monotonic_ns"]
            >= timing["lowered_identity_end_monotonic_ns"],
            "warm call began before proof identities completed",
        )
        _require(
            timing["integration_end_monotonic_ns"]
            >= timing["integration_start_monotonic_ns"],
            "integration endpoints reversed",
        )
        _require(
            math.isclose(
                float(timing["integration_seconds"]),
                (
                    timing["integration_end_monotonic_ns"]
                    - timing["integration_start_monotonic_ns"]
                )
                / 1e9,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "integration seconds do not match the direct endpoints",
        )
        _require(result.get("synchronization") == "jax.block_until_ready(result)",
                 "final synchronization changed")
        _require(
            isinstance(result.get("exact_value_sha256"), str)
            and len(result["exact_value_sha256"]) == 64,
            "result digest missing",
        )
        wrfout = payload.get("wrfout")
        validate_wrfout_binding(payload)
        for field in (
            "final_wrfout_sha256",
            "binding_sha256",
            "exact_boundary_identity_sha256",
        ):
            _require(
                isinstance(wrfout.get(field), str)
                and len(wrfout[field]) == 64,
                f"wrfout {field} is missing",
            )
        wrfout_timing = wrfout.get("timing") or {}
        endpoints = [
            timing["integration_end_monotonic_ns"],
            wrfout_timing.get("materialization_start_monotonic_ns"),
            wrfout_timing.get("materialization_end_monotonic_ns"),
            wrfout_timing.get("prepare_start_monotonic_ns"),
            wrfout_timing.get("prepare_end_monotonic_ns"),
            wrfout_timing.get("write_start_monotonic_ns"),
            wrfout_timing.get("write_end_monotonic_ns"),
            wrfout_timing.get("inspection_start_monotonic_ns"),
            wrfout_timing.get("inspection_end_monotonic_ns"),
        ]
        _require(
            all(isinstance(value, int) and not isinstance(value, bool)
                for value in endpoints)
            and endpoints == sorted(endpoints)
            and wrfout_timing.get(
                "outside_readiness_and_integration_clocks"
            )
            is True,
            "wrfout work is not directly proven after integration",
        )
        _require(
            (wrfout.get("publication") or {}).get("atomic") is True
            and (wrfout.get("publication") or {}).get("replacement") is False,
            "wrfout publication is not atomic no-replace",
        )
    if expected_mode == "profiled":
        _require(instrumentation.get("nvtx_range") == RANGE_NAME,
                 "profiled range changed")
        _require(instrumentation.get("allocator_sidecar"), "allocator sidecar missing")
        _require(payload.get("allocator") is not None, "allocator read missing")
    else:
        _require(instrumentation.get("nvtx_range") is None,
                 "clean/compile-only has a profiler range")
        _require(instrumentation.get("allocator_sidecar") is None,
                 "clean/compile-only has allocator sidecar")
        _require(payload.get("allocator") is None,
                 "clean/compile-only read allocator")
    return {"status": "PASS", "mode": expected_mode}


def compare_profiled_clean(profiled: dict[str, Any], clean: dict[str, Any]) -> dict[str, Any]:
    validate_boundary_result(profiled, expected_mode="profiled")
    validate_boundary_result(clean, expected_mode="clean")
    for path in (
        ("call", "argument_identity"),
        ("call", "lowered_program_sha256"),
        ("production_binding", "jit_identity"),
        ("production_binding", "case_preparation_sequence"),
        ("production_binding", "wrapper_preparation_sequence"),
        ("result", "semantics"),
        ("result", "exact_value_sha256"),
    ):
        left: Any = profiled
        right: Any = clean
        for key in path:
            left = left[key]
            right = right[key]
        _require(left == right, f"profiled/clean identity differs at {'.'.join(path)}")
    _require(
        profiled["instrumentation"]["invocation_helper"]
        == clean["instrumentation"]["invocation_helper"],
        "profiled/clean invocation helpers differ",
    )
    return {
        "status": "PASS",
        "only_allowed_difference": "profiler/resource instrumentation",
    }


def _normalized_plan_command(command: list[str]) -> list[str]:
    normalized = []
    prefix = f"{REPO.resolve()}{os.sep}"
    for raw in command:
        value = str(raw)
        if Path(value) == Path(sys.executable):
            normalized.append("<PYTHON>")
        elif value == str(REPO.resolve()):
            normalized.append("<REPO>")
        elif value.startswith(prefix):
            normalized.append(f"<REPO>/{value[len(prefix):]}")
        else:
            normalized.append(value)
    return normalized


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    _require(plan.get("status") == STATUS, "plan status changed")
    windows = plan.get("windows") or []
    _require([window.get("id") for window in windows] == ["W1", "W2", "W3"],
             "window order/count changed")
    _require(len({window["run_id"] for window in windows}) == 3, "run IDs not unique")
    session_receipt = plan.get("session_receipt")
    _require(
        isinstance(session_receipt, str)
        and session_receipt
        and {window["receipt"] for window in windows} == {session_receipt},
        "windows do not share the one Amendment-5 session receipt",
    )
    _require(
        plan.get("maximum_lock_acquisitions") == 1,
        "Amendment-5 plan permits more than one canonical lock acquisition",
    )
    session_protocol = plan.get("amendment_5_session_protocol") or {}
    _require(
        session_protocol.get("label") == "m0-core-w1-w2-w3-session"
        and session_protocol.get("canonical_lock_acquisitions") == 1
        and (session_protocol.get("cold_protocol") or {}).get(
            "threshold_seconds"
        )
        == 600.0
        and (session_protocol.get("cold_protocol") or {}).get(
            "maximum_processes"
        )
        == 3
        and (session_protocol.get("cold_protocol") or {}).get(
            "third_process"
        )
        == "only if first two disagree",
        "Amendment-5 session/cold protocol is incomplete",
    )
    expected_deadlines = {"W1": 2160.0, "W2": 1400.0, "W3": 600.0}
    _require(
        {window["id"]: float(window["deadline_seconds"]) for window in windows}
        == expected_deadlines,
        "Amendment-6 per-stage reservations changed",
    )
    for window in windows:
        budget = window.get("budget") or {}
        _require(
            budget.get("stage_budget_seconds", 0)
            + budget.get("overhead_seconds", 0)
            <= budget.get("deadline_seconds", 0)
            <= expected_deadlines[window["id"]],
            f"{window['id']} budget does not fit",
        )
    inspections = plan.get("manager_window_inspection") or {}
    _require(list(inspections) == ["W1", "W2", "W3"], "real manager graphs missing")
    manager_commands = plan.get("manager_window_commands") or {}
    _require(
        manager_commands.get("status") == "SUPERSEDED_BY_ONE_SESSION_COMMAND"
        and manager_commands.get("independent_authorisation_allowed") is False,
        "superseded per-window authorisation remains eligible",
    )
    session_command = plan.get("manager_session_command") or []
    normalized_session = _normalized_plan_command(session_command)
    _require(
        session_command == manager_commands.get("command")
        and plan.get("manager_session_command_sha256")
        == hashlib.sha256(
            "\0".join(normalized_session).encode("utf-8")
        ).hexdigest()
        and session_command[:3]
        == [
            sys.executable,
            str(REPO / "scripts/v025/m0_three_window_executor.py"),
            "--manager-core-session-owner",
        ]
        and session_command[3] == "--receipt"
        and len(session_command) == 5,
        "one-session outer-owner command is not exact/receipt-bound",
    )
    held = plan.get("amendment_5_held_session") or {}
    wrapper = held.get("held_wrapper_command") or []
    _require(
        wrapper[:6]
        == [
            str(REPO / "scripts/with_gpu_lock.sh"),
            "--timeout",
            "0",
            "--label",
            "m0-core-w1-w2-w3-session",
            "--",
        ]
        and "--manager-core-session-held" in wrapper
        and len(wrapper) == 11
        and (held.get("held_budget") or {}).get("reservation_seconds") == 4220.0
        and (held.get("held_budget") or {}).get("headroom_seconds") == 280.0,
        "outer owner does not freeze the exact 4220/4500 held wrapper",
    )
    post_commands = plan.get("manager_post_commands") or {}
    _require(
        list(post_commands) == ["W1", "W2", "W3"],
        "JAX-free post-lock commands are missing",
    )
    legacy_outer = plan.get("legacy_amendment4_outer_graph_commands") or {}
    _require(
        legacy_outer.get("status") == "NON_AUTHORITATIVE_SUPERSEDED"
        and list((legacy_outer.get("commands") or {})) == ["W1", "W2", "W3"],
        "release-owning legacy graphs are not explicitly superseded",
    )
    for window_id, inspection in inspections.items():
        _require(inspection.get("inspection_only") is True, "inspection touched state")
        _require(inspection.get("device_imports") == [], "inspection imported device")
        device_stages = [
            stage for stage in inspection["stages"]
            if stage.get("mode") not in {"post-analysis", "parent-gate"}
        ]
        _require(
            all(stage.get("fresh_process") is True for stage in device_stages),
            f"{window_id} has a non-fresh device child",
        )
        post_command = post_commands[window_id]
        try:
            post_marker = post_command.index("--manager-post-stage")
        except ValueError as exc:
            raise ContractViolation(
                f"{window_id} post-lock command marker is absent"
            ) from exc
        _require(
            post_command[post_marker + 1] == window_id
            and "--output" in post_command
            and "--lock-release-proof" in post_command
            and post_command[:2] == ["taskset", "-c"],
            f"{window_id} post-lock command is not identity/output bound",
        )
        outer = legacy_outer["commands"][window_id]
        _require(
            "--manager-window-graph" in outer
            and outer[outer.index("--manager-window-graph") + 1] == window_id
            and "--receipt" in outer,
            f"{window_id} outer graph is not window/receipt bound",
        )
    for window in windows:
        command = window.get("manager_command") or []
        _require(
            command == session_command
            and window.get("manager_command_invocations") == 0,
            f"{window['id']} remains independently invocable",
        )
        _require(command == session_command, f"{window['id']} escaped the outer owner")
        _require(
            all(
                stage.get("command") == command
                and stage.get("command_scope")
                == "ONE_INVOCATION_FOR_W1_W2_W3_SESSION"
                and stage.get("dry_stub_command")
                == ["CPU_DRY_STUB_ONLY", window["id"], stage["name"]]
                for stage in window["stages"]
            ),
            f"{window['id']} mixes live commands with CPU dry stubs",
        )
    _require(
        plan.get("profiled_pair_order")
        == [
            "m0-autotune0-profiled-20260728-r3",
            "m0-autotune0-clean-20260728-r3",
        ],
        "profiled-first order changed",
    )
    _require(
        "IMPLEMENTED" in str(
            (plan.get("prelock_cpu_comparator") or {}).get(
                "exact_result_wrfout_adapter", ""
            )
        ),
        "plan does not expose the implemented exact-result wrfout adapter",
    )
    provenance = plan.get("identity_and_cache_provenance") or {}
    _require(
        (provenance.get("profiled_identity") or {}).get("path")
        == (
            "proofs/v025/m0/autotune0_three_window_r5/"
            "prepared_profiler_pair_r3/profiled_identity.json"
        )
        and (provenance.get("clean_identity") or {}).get("path")
        == (
            "proofs/v025/m0/autotune0_three_window_r5/"
            "prepared_profiler_pair_r3/clean_identity.json"
        ),
        "plan identities do not bind the paths consumed by W2/W3",
    )
    _require(
        all(
            "pallas" not in stage.get("name", "").lower()
            for inspection in inspections.values()
            for stage in inspection.get("stages", [])
        ),
        "native Pallas is still reachable from an M0 window",
    )
    _require(
        plan.get("timing_boundaries", {}).get("current_gate_eligibility")
        == "CPU_PROVEN_GPU_MEASUREMENTS_MISSING",
        "plan overclaims device evidence",
    )
    return {
        "status": "PASS",
        "window_count": 3,
        "max_deadline_seconds": 2160.0,
        "lock_acquisitions": 1,
        "session_receipts": 1,
    }


def mutate_copy(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> dict[str, Any]:
    out = copy.deepcopy(value)
    cursor = out
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    return out


__all__ = [
    "ContractViolation",
    "FROZEN_FUNCTION_HASHES",
    "STATUS",
    "compare_profiled_clean",
    "mutate_copy",
    "representative_boundary_fixture",
    "validate_boundary_result",
    "validate_plan",
    "validate_representative_long_run_fixtures",
    "validate_source_texts",
    "validate_sources",
    "validate_wrfout_binding",
]
