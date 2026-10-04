#!/usr/bin/env python3
"""Build the terminal CPU proof for Amendment-4 exact executable identity."""

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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cpu_guard  # noqa: E402,F401  MUST precede jax/gpuwrf imports

import m0_exact_boundary_child as child  # noqa: E402
import m0_exact_boundary_contract as contract  # noqa: E402
import m0_long_run_controls as longrun  # noqa: E402
import m0_three_window_executor as executor  # noqa: E402
from real_state import assert_cpu_only  # noqa: E402


STATUS = "CPU_C1_GREEN_C2_PENDING"
SCHEMA = "wrf_gpu2.v025.m0.exact_boundary_cpu_evidence.v1"
ISOLATED_ARM_SCHEMA = "wrf_gpu2.v025.m0.exact_cpu_isolated_arm.v1"
ISOLATED_ARMS = frozenset({"local-exact", "public-wrapper"})
DEFAULT_OUTPUT = REPO / "proofs/v025/m0/m0_exact_boundary_cpu_evidence.json"
START_COMMIT = "dedff5bbb4ac4fb852bdd9d42b50c1da25a61dfb"
PRODUCTION_TREE = "a6885ceded260df2f5777d7366d75a5d38947cb7"
PRE_REPAIR_READINESS_SECONDS = 81.242206654
PRE_REPAIR_PROOF_SHA256 = (
    "3f07f72db80addf4c706fe39f17178c36741cd408f891daa9236f6ccd5c68646"
)
PRE_REPAIR_PROOF_ARCHIVE = (
    REPO
    / "proofs/v025/m0/archive/"
    f"m0_exact_boundary_cpu_evidence.{PRE_REPAIR_PROOF_SHA256}.json"
)
REFERENCE_CPU_CACHE_DIR = Path(
    "<DATA_ROOT>/wrf_gpu2/v025/m0/cpu_exact_boundary_cache_r1"
)
REFERENCE_CPU_CACHE_ENTRY = (
    "jit__run_forecast_operational_jit-"
    "5ca1a338cdbae726ba84ea3ae9d8d37c2b382165acd4310303acac4d4f3733ee-cache"
)
REFERENCE_CPU_CACHE_ENTRY_BYTES = 226_417_734
REFERENCE_CPU_CACHE_ENTRY_SHA256 = (
    "d05078bcb8dcd03ff045199975e7f7d1d848f1b8b42e76c06948d7ad254ae0b4"
)
ISOLATED_ARM_ESTIMATED_SECONDS = 5_400.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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
        raise RuntimeError(
            f"git {' '.join(arguments)} failed rc={completed.returncode}: "
            f"{completed.stderr[-1000:]}"
        )
    return completed.stdout.strip()


def _production_identity() -> dict[str, Any]:
    diff = subprocess.run(
        ["git", "diff", "--exit-code", START_COMMIT, "--", "src/gpuwrf"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    tree = _git("rev-parse", f"{START_COMMIT}:src/gpuwrf")
    if diff.returncode != 0 or tree != PRODUCTION_TREE:
        raise RuntimeError(
            "production source differs from the frozen starting tree: "
            f"diff_rc={diff.returncode}, tree={tree}"
        )
    return {
        "starting_commit": START_COMMIT,
        "src_gpuwrf_tree": tree,
        "diff_exit_code": diff.returncode,
        "unchanged": True,
    }


def _cpu_reference_cache_contract() -> dict[str, Any]:
    """Bind the post-repair clock to the cache used by the 81.24 s baseline."""

    enabled = os.environ.get("GPUWRF_JAX_CACHE", "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        raise RuntimeError(
            "R1 clock comparison requires GPUWRF_JAX_CACHE=1; refusing an "
            "unlike cold-vs-cached readiness comparison"
        )
    configured_text = os.environ.get("GPUWRF_JAX_CACHE_DIR", "").strip()
    if not configured_text:
        raise RuntimeError(
            "R1 clock comparison requires the explicit historical "
            "GPUWRF_JAX_CACHE_DIR"
        )
    configured = Path(configured_text).expanduser().resolve()
    expected = REFERENCE_CPU_CACHE_DIR.resolve()
    if configured != expected:
        raise RuntimeError(
            "R1 clock comparison requires the historical CPU cache "
            f"{expected}, observed {configured}"
        )
    lock = os.environ.get("GPUWRF_JAX_CACHE_LOCK", "").strip().lower()
    if lock not in {"1", "true", "yes", "on"}:
        raise RuntimeError(
            "R1 clock comparison requires GPUWRF_JAX_CACHE_LOCK=1"
        )
    entry = configured / REFERENCE_CPU_CACHE_ENTRY
    if entry.is_symlink() or not entry.is_file():
        raise RuntimeError(f"historical top-level CPU cache entry is missing: {entry}")
    size = entry.stat().st_size
    digest = _sha256_file(entry)
    if (
        size != REFERENCE_CPU_CACHE_ENTRY_BYTES
        or digest != REFERENCE_CPU_CACHE_ENTRY_SHA256
    ):
        raise RuntimeError(
            "historical top-level CPU cache entry identity changed: "
            f"bytes={size}, sha256={digest}"
        )
    files = [path for path in configured.iterdir() if path.is_file()]
    return {
        "status": "PASS",
        "comparison_basis": (
            "same populated persistent CPU cache as the recorded "
            "81.242206654 s pre-repair readiness"
        ),
        "enabled": True,
        "locked": True,
        "directory": str(configured),
        "top_level_entry": str(entry),
        "top_level_entry_bytes": size,
        "top_level_entry_sha256": digest,
        "directory_file_count_observed": len(files),
        "device_touched": False,
    }


def _source_attack_matrix() -> dict[str, Any]:
    pristine = {
        "child_source": contract.CHILD.read_text(encoding="utf-8"),
        "parent_source": contract.PARENT.read_text(encoding="utf-8"),
        "parent_death_guard_source":
            contract.PARENT_DEATH_GUARD.read_text(encoding="utf-8"),
        "executor_source": contract.EXECUTOR.read_text(encoding="utf-8"),
        "pallas_source": contract.PALLAS.read_text(encoding="utf-8"),
        "evidence_builder_source":
            contract.EVIDENCE_BUILDER.read_text(encoding="utf-8"),
        "long_run_controls_source":
            contract.LONG_RUN_CONTROLS.read_text(encoding="utf-8"),
        "core_session_source":
            contract.CORE_SESSION_PROTOCOL.read_text(encoding="utf-8"),
    }

    def attack(
        name: str,
        target: str,
        old: str,
        new: str,
        *,
        count: int = -1,
    ) -> tuple[str, bool, str]:
        mutated = dict(pristine)
        if count < 0:
            mutated[target] = mutated[target].replace(old, new)
        else:
            mutated[target] = mutated[target].replace(old, new, count)
        if mutated[target] == pristine[target]:
            return name, False, "mutation anchor not found"
        try:
            contract.validate_source_texts(**mutated)
        except (contract.ContractViolation, SyntaxError) as exc:
            return name, True, f"{type(exc).__name__}: {exc}"
        return name, False, "unsafe mutation was accepted"

    scan_hash = child.EXPECTED_FUNCTION_AST_SHA256[
        "gpuwrf.runtime.operational_mode._operational_scan_state"
    ]
    attacks = [
        attack(
            "preparation_order",
            "child_source",
            "prepared_state = scan_state(state, namelist)\n"
            "    prepared_state = dealias(prepared_state)",
            "prepared_state = dealias(state)\n"
            "    prepared_state = scan_state(prepared_state, namelist)",
        ),
        attack(
            "production_function_identity",
            "child_source",
            scan_hash,
            "0" * 64,
            count=1,
        ),
        attack(
            "lower_call_arguments",
            "child_source",
            "lowered = exact_jit.lower(*prepared_arguments)",
            "lowered = exact_jit.lower(prepared_arguments[0], "
            "prepared_arguments[1], prepared_arguments[2] + 1.0)",
        ),
        attack(
            "readiness_event_before_compile",
            "child_source",
            "executable = lowered.compile()\n"
            "    executable_ready_monotonic_ns = time.monotonic_ns()",
            "executable_ready_monotonic_ns = time.monotonic_ns()\n"
            "    executable = lowered.compile()",
        ),
        attack(
            "argument_identity_inside_readiness_clock",
            "child_source",
            "executable = lowered.compile()",
            "argument_identity = _structural_argument_identity("
            "jax, prepared_arguments)\n"
            "    executable = lowered.compile()",
            count=1,
        ),
        attack(
            "lowered_identity_inside_readiness_clock",
            "child_source",
            "executable = lowered.compile()",
            "stablehlo = _stablehlo_identity(lowered)\n"
            "    executable = lowered.compile()",
            count=1,
        ),
        attack(
            "final_synchronization_removed",
            "child_source",
            "jax.block_until_ready(result)",
            "jax.tree_util.tree_leaves(result)",
            count=1,
        ),
        attack(
            "profile_range_identity",
            "child_source",
            'RANGE_NAME = "GPUWRF_M0_FORECAST_INTEGRATION"',
            'RANGE_NAME = "MUTATED_RANGE"',
        ),
        attack(
            "clean_instrumentation_added",
            "child_source",
            'elif mode == "clean":\n'
            "        result, integration_start_ns, integration_end_ns",
            'elif mode == "clean":\n'
            "        _forbidden_profiler = jax.profiler\n"
            "        result, integration_start_ns, integration_end_ns",
        ),
        attack(
            "parent_jax_import",
            "parent_source",
            "import hashlib",
            "import jax\nimport hashlib",
            count=1,
        ),
        attack(
            "residency_sampler_after_forecast_launch",
            "parent_source",
            "sampler.start()",
            "sampler_start_deferred = True",
            count=1,
        ),
        attack(
            "parent_death_arm_removed",
            "parent_death_guard_source",
            "arm_parent_death_signal(expected_lock_owner_pid, signal.SIGTERM)",
            "arm_parent_death_signal_disabled("
            "expected_lock_owner_pid, signal.SIGTERM)",
            count=1,
        ),
        attack(
            "parent_death_group_kill_weakened",
            "parent_death_guard_source",
            "os.killpg(os.getpgrp(), signal.SIGKILL)",
            "os.killpg(os.getpgrp(), signal.SIGTERM)",
            count=1,
        ),
        attack(
            "pallas_second_authorization",
            "pallas_source",
            "payload = run_native(args, authorization)",
            "payload = run_native(args, authorization)\n"
            "            authorization = "
            "exact_child.consume_authorization_handoff("
            "args.handoff, expected_window=args.window, "
            "expected_stage=args.stage, expected_run_id=args.run_id)",
            count=1,
        ),
        attack(
            "pallas_blocked_verdict_accepted",
            "parent_source",
            'if verdict == "PALLAS_BLOCKED":',
            'if verdict == "NEVER_BLOCK":',
            count=1,
        ),
        attack(
            "pallas_before_profiled_clean_identity_gate",
            "parent_source",
            "pair_gate = {",
            "pair_gate_deferred = {",
            count=1,
        ),
        attack(
            "profiled_capture_integrity_removed",
            "parent_source",
            "artifacts = _require_profiled_capture_artifacts(profiled)",
            "artifacts = {}",
            count=1,
        ),
        attack(
            "w1_fast_pair_binding_removed",
            "parent_source",
            "if binding != expected_binding:",
            "if False:",
            count=1,
        ),
        attack(
            "substituted_wrfout_result",
            "child_source",
            "result=result,",
            "result=prepared_arguments[0],",
            count=1,
        ),
        attack(
            "output_clock_before_synchronization",
            "child_source",
            "jax.block_until_ready(result)\n"
            "    integration_end_ns = time.monotonic_ns()",
            "integration_end_ns = time.monotonic_ns()\n"
            "    jax.block_until_ready(result)",
            count=1,
        ),
        attack(
            "stale_output_reuse",
            "child_source",
            "if os.path.lexists(target_dir):",
            "if False and os.path.lexists(target_dir):",
            count=1,
        ),
        attack(
            "wrong_wrfout_domain",
            "child_source",
            "domain=config.domain,",
            'domain="d02",',
            count=1,
        ),
        attack(
            "missing_production_variable_set",
            "child_source",
            "full_variable_set=full_variable_set,",
            "full_variable_set=False,",
            count=1,
        ),
        attack(
            "variable_inventory_hash_unbound",
            "child_source",
            '"variable_inventory_sha256": canonical_sha256(variable_inventory),',
            '"variable_inventory_sha256": "0" * 64,',
            count=1,
        ),
        attack(
            "same_process_double_integration",
            "evidence_builder_source",
            "local_envelope, local_producer = _run_isolated_arm(\n"
            '            kind="local-exact",',
            "local_envelope, local_producer = _run_isolated_arm(\n"
            '            kind="public-wrapper",',
            count=1,
        ),
        attack(
            "isolated_arm_detached_process_group",
            "evidence_builder_source",
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
            count=1,
        ),
        attack(
            "cpu_arm_mislabelled_with_gpu_timing_gate",
            "evidence_builder_source",
            "contract.validate_wrfout_binding(local, verify_file=True)",
            'contract.validate_boundary_result(local, expected_mode="clean")\n'
            "        contract.validate_wrfout_binding(local, verify_file=True)",
            count=1,
        ),
        attack(
            "single_core_cpu_deadlock_guard_removed",
            "evidence_builder_source",
            "os.sched_setaffinity(0, {inherited_affinity[0]})",
            "single_core_affinity_not_applied = inherited_affinity[0]",
        ),
        attack(
            "early_output_directory_preflight_removed",
            "evidence_builder_source",
            "output_preflight = longrun.output_directory_preflight(\n"
            "        wrfout_output_dir\n"
            "    )",
            'output_preflight = {"status": "SKIPPED"}',
        ),
        attack(
            "representative_validator_preflight_removed",
            "evidence_builder_source",
            "fixture_preflight = (\n"
            "        contract.validate_representative_long_run_fixtures()\n"
            "    )",
            'fixture_preflight = {"status": "SKIPPED"}',
        ),
        attack(
            "long_run_inhibitor_removed",
            "evidence_builder_source",
            "inhibited, inhibitor = longrun.inhibited_command(",
            "inhibited, inhibitor = command, "
            '{"status": "SKIPPED"}  # ',
        ),
        attack(
            "long_run_continuity_gate_removed",
            "evidence_builder_source",
            "continuity = longrun.validate_clock_continuity(",
            'continuity = {"status": "SKIPPED"}  # ',
        ),
        attack(
            "memory_headroom_ratio_weakened",
            "long_run_controls_source",
            "MEMORY_HEADROOM_RATIO = 2.0",
            "MEMORY_HEADROOM_RATIO = 1.0",
        ),
        attack(
            "foreign_process_limit_weakened",
            "long_run_controls_source",
            "FOREIGN_PROCESS_PHYSICAL_RAM_RATIO = 0.25",
            "FOREIGN_PROCESS_PHYSICAL_RAM_RATIO = 0.75",
        ),
        attack(
            "systemd_inhibit_scope_weakened",
            "long_run_controls_source",
            'INHIBIT_WHAT = "sleep:idle:handle-lid-switch"',
            'INHIBIT_WHAT = "sleep"',
        ),
        attack(
            "session_receipt_label_weakened",
            "core_session_source",
            'SESSION_LABEL = "m0-core-w1-w2-w3-session"',
            'SESSION_LABEL = "per-window-session"',
        ),
        attack(
            "session_cold_threshold_weakened",
            "core_session_source",
            "COLD_THRESHOLD_SECONDS = 600.0",
            "COLD_THRESHOLD_SECONDS = 601.0",
        ),
        attack(
            "session_third_process_always_runs",
            "core_session_source",
            'if decision["decision"] != "CONTINUE":',
            'if decision["decision"] == "NEVER":',
            count=1,
        ),
    ]
    return {
        "status": "PASS" if all(rejected for _, rejected, _ in attacks) else "FAIL",
        "rejected": sum(rejected for _, rejected, _ in attacks),
        "total": len(attacks),
        "attacks": [
            {"name": name, "rejected": rejected, "detail": detail}
            for name, rejected, detail in attacks
        ],
    }


def _public_wrapper_observation(
    *,
    run_dir: Path,
    hours: float,
) -> dict[str, Any]:
    """Run the real public wrapper while recording its exact private-JIT call."""

    import jax
    from gpuwrf.runtime import operational_mode as operational

    raw_arguments, case_metadata, verified = child.load_fast_call_arguments(
        run_dir=run_dir,
        hours=hours,
        cpu_device_adapter=True,
    )
    original_jit = operational._run_forecast_operational_jit
    captured: dict[str, Any] = {}

    def recording_jit(*arguments):
        captured["argument_identity"] = child._structural_argument_identity(
            jax, arguments
        )
        lowered = original_jit.lower(*arguments)
        stablehlo = child._stablehlo_identity(lowered)
        captured["lowered_program_sha256"] = stablehlo["sha256"]
        captured["lowered_program_bytes"] = stablehlo["bytes"]
        captured["called_function"] = {
            "fqname": child.JIT_FUNCTION,
            "ast_sha256": child.callable_ast_sha256(original_jit),
        }
        return original_jit(*arguments)

    operational._run_forecast_operational_jit = recording_jit
    try:
        result = operational.run_forecast_operational(*raw_arguments)
        jax.block_until_ready(result)
    finally:
        operational._run_forecast_operational_jit = original_jit
    if not captured:
        raise RuntimeError("public wrapper never called the private production JIT")
    return {
        "case": case_metadata,
        "captured_call": captured,
        "result": {
            "semantics": child._leaf_semantics(jax, result),
            "exact_value_sha256": child._pytree_digest(jax, result),
            "synchronization": "jax.block_until_ready(result)",
        },
        "returned_state_semantics": (
            f"{type(result).__module__}.{type(result).__qualname__}"
        ),
        "wrapper_function": (
            "gpuwrf.runtime.operational_mode.run_forecast_operational"
        ),
    }


def _accelerator_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if name == "jax"
        or name.startswith("jax.")
        or name == "jaxlib"
        or name.startswith("jaxlib.")
        or name == "gpuwrf"
        or name.startswith("gpuwrf.")
    )


def _isolated_arm_child(
    *,
    kind: str,
    output: Path,
    run_id: str,
    run_dir: Path,
    hours: float,
    wrfout_output_dir: Path,
    parent_launch_monotonic_ns: int,
) -> dict[str, Any]:
    """Execute one expensive arm in a fresh process and publish no-replace JSON."""

    if kind not in ISOLATED_ARMS:
        raise RuntimeError(f"unknown isolated CPU arm {kind!r}")
    inherited_affinity = sorted(os.sched_getaffinity(0))
    if not inherited_affinity:
        raise RuntimeError("isolated CPU arm inherited an empty CPU affinity")
    os.sched_setaffinity(0, {inherited_affinity[0]})
    cpu_platform = assert_cpu_only()
    import jax

    if kind == "local-exact":
        payload = child.run_boundary(
            mode="clean",
            run_id=run_id,
            parent_launch_monotonic_ns=parent_launch_monotonic_ns,
            run_dir=run_dir,
            hours=hours,
            expected_platform="cpu",
            cpu_device_adapter=True,
            allocator_sidecar=None,
            wrfout_output_dir=wrfout_output_dir,
        )
    else:
        payload = _public_wrapper_observation(run_dir=run_dir, hours=hours)

    envelope = {
        "schema": ISOLATED_ARM_SCHEMA,
        "status": "PASS",
        "kind": kind,
        "run_id": run_id,
        "pid": os.getpid(),
        "device_touched": False,
        "cpu_platform": cpu_platform,
        "inherited_cpu_affinity": inherited_affinity,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "single_core_xla_cpu_deadlock_guard": True,
        "jax_version": jax.__version__,
        "python": sys.version.split()[0],
        "producer_argv": list(sys.argv),
        "payload": payload,
    }
    child._atomic_json_no_replace(output, envelope)
    return envelope


def _run_isolated_arm(
    *,
    kind: str,
    output: Path,
    run_id: str,
    run_dir: Path,
    hours: float,
    wrfout_output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one arm synchronously without detaching it from the caller's group."""

    if kind not in ISOLATED_ARMS:
        raise RuntimeError(f"unknown isolated CPU arm {kind!r}")
    if os.path.lexists(output):
        raise RuntimeError(f"refusing stale isolated-arm output: {output}")
    if kind == "local-exact":
        longrun.output_directory_preflight(wrfout_output_dir)
    before = _accelerator_modules()
    if before:
        raise RuntimeError(
            "top-level exact evidence builder imported accelerator modules "
            f"before isolated arm launch: {before}"
        )

    parent_launch_ns = time.monotonic_ns()
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--isolated-arm",
        kind,
        "--arm-output",
        str(output),
        "--run-id",
        run_id,
        "--run-dir",
        str(run_dir),
        "--hours",
        str(float(hours)),
        "--wrfout-output-dir",
        str(wrfout_output_dir),
        "--parent-launch-monotonic-ns",
        str(parent_launch_ns),
    ]
    inhibited, inhibitor = longrun.inhibited_command(
        command,
        estimated_seconds=ISOLATED_ARM_ESTIMATED_SECONDS,
        who=f"wrf_gpu2-v025-{kind}",
        why=f"protect the M0 {kind} exact CPU evidence arm",
    )
    clock_start = longrun.boot_clock_sample()
    completed = subprocess.run(
        inhibited,
        cwd=REPO,
        check=False,
    )
    clock_end = longrun.boot_clock_sample()
    continuity = longrun.validate_clock_continuity(
        clock_start, clock_end
    )
    finished_ns = time.monotonic_ns()
    after = _accelerator_modules()
    if after != before:
        raise RuntimeError(
            "isolated CPU arm contaminated the JAX-free manager process: "
            f"before={before}, after={after}"
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"isolated CPU arm {kind} failed rc={completed.returncode}"
        )
    if output.is_symlink() or not output.is_file():
        raise RuntimeError(f"isolated CPU arm {kind} published no regular result")
    envelope = json.loads(output.read_text(encoding="utf-8"))
    if (
        envelope.get("schema") != ISOLATED_ARM_SCHEMA
        or envelope.get("status") != "PASS"
        or envelope.get("kind") != kind
        or envelope.get("run_id") != run_id
        or envelope.get("device_touched") is not False
        or not isinstance(envelope.get("inherited_cpu_affinity"), list)
        or not envelope["inherited_cpu_affinity"]
        or envelope.get("cpu_affinity")
        != [min(envelope["inherited_cpu_affinity"])]
        or envelope.get("single_core_xla_cpu_deadlock_guard") is not True
        or not isinstance(envelope.get("payload"), dict)
    ):
        raise RuntimeError(f"isolated CPU arm {kind} envelope is malformed")
    producer = {
        "kind": kind,
        "command": command,
        "executed_command": inhibited,
        "systemd_inhibitor": inhibitor,
        "clock_continuity": continuity,
        "returncode": completed.returncode,
        "parent_launch_monotonic_ns": parent_launch_ns,
        "finished_monotonic_ns": finished_ns,
        "wall_seconds": (finished_ns - parent_launch_ns) / 1e9,
        "child_pid": envelope["pid"],
        "inherited_cpu_affinity": envelope["inherited_cpu_affinity"],
        "cpu_affinity": envelope["cpu_affinity"],
        "single_core_xla_cpu_deadlock_guard":
            envelope["single_core_xla_cpu_deadlock_guard"],
        "output_path": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": _sha256_file(output),
        "payload_sha256": contract.canonical_sha256(envelope["payload"]),
        "process_group": "inherited; no detached session",
    }
    return envelope, producer


def build(
    *,
    output: Path,
    run_dir: Path,
    hours: float,
    wrfout_output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    if _accelerator_modules():
        raise RuntimeError(
            "top-level exact evidence builder must remain accelerator-free"
        )
    output_preflight = longrun.output_directory_preflight(
        wrfout_output_dir
    )
    memory_preflight = longrun.memory_preflight()
    fixture_preflight = (
        contract.validate_representative_long_run_fixtures()
    )
    if (
        not PRE_REPAIR_PROOF_ARCHIVE.is_file()
        or _sha256_file(PRE_REPAIR_PROOF_ARCHIVE) != PRE_REPAIR_PROOF_SHA256
    ):
        raise RuntimeError(
            "the prior expensive proof is not archived under its content hash"
        )
    cache_before = _cpu_reference_cache_contract()
    source_validation = contract.validate_sources()
    plan = executor.build_plan()
    plan_validation = contract.validate_plan(plan)
    mutation_matrix = _source_attack_matrix()
    if mutation_matrix["status"] != "PASS":
        raise RuntimeError("one or more exact-boundary mutations were accepted")

    arm_parent = wrfout_output_dir.resolve().parent
    if not arm_parent.is_dir():
        raise RuntimeError(
            f"isolated CPU arm parent directory is missing: {arm_parent}"
        )
    with tempfile.TemporaryDirectory(
        dir=arm_parent,
        prefix=".m0-exact-cpu-arms-",
    ) as arm_directory_text:
        arm_directory = Path(arm_directory_text)
        local_envelope, local_producer = _run_isolated_arm(
            kind="local-exact",
            output=arm_directory / "local-exact.json",
            run_id=run_id,
            run_dir=run_dir,
            hours=hours,
            wrfout_output_dir=wrfout_output_dir,
        )
        local = local_envelope["payload"]
        contract.validate_wrfout_binding(local, verify_file=True)
    # The 600/60/300 numeric gates belong to device windows. The real CPU run
    # proves boundary semantics and records its own times without relabelling
    # them as a GPU economy pass.
    if local["status"] != "OK":
        raise RuntimeError("local exact compiled CPU path did not complete")
    if (local.get("wrfout") or {}).get("status") != "PASS":
        raise RuntimeError("local exact compiled CPU path did not publish wrfout")
    local_timing = local["timing"]
    new_readiness_seconds = float(local_timing["readiness_seconds"])
    measured_proof_seconds = float(local_timing["proof_identity_seconds"])
    readiness_reduction_seconds = (
        PRE_REPAIR_READINESS_SECONDS - new_readiness_seconds
    )
    clock_accounting = {
        "status": (
            "PASS"
            if (
                new_readiness_seconds < PRE_REPAIR_READINESS_SECONDS
                and readiness_reduction_seconds >= measured_proof_seconds
            )
            else "BLOCKED"
        ),
        "pre_repair_proof_archive": str(
            PRE_REPAIR_PROOF_ARCHIVE.relative_to(REPO)
        ),
        "pre_repair_proof_sha256": PRE_REPAIR_PROOF_SHA256,
        "pre_repair_readiness_seconds": PRE_REPAIR_READINESS_SECONDS,
        "post_repair_readiness_seconds": new_readiness_seconds,
        "readiness_reduction_seconds": readiness_reduction_seconds,
        "argument_identity_seconds": float(
            local_timing["argument_identity_seconds"]
        ),
        "lowered_identity_seconds": float(
            local_timing["lowered_identity_seconds"]
        ),
        "measured_proof_identity_seconds": measured_proof_seconds,
        "required_reduction_seconds": measured_proof_seconds,
        "comparison_tolerance_seconds": 0.0,
        "proof_work_outside_readiness_clock": True,
        "proof_work_outside_integration_clock": True,
    }
    if clock_accounting["status"] != "PASS":
        raise RuntimeError(
            "BLOCKED: CPU readiness did not fall below 81.242206654 s by "
            "the measured proof-work delta: "
            f"{json.dumps(clock_accounting, sort_keys=True)}"
        )
    print(
        json.dumps(
            {
                "stage": "local_exact_compiled_complete",
                "readiness_seconds": local["timing"]["readiness_seconds"],
                "proof_identity_seconds":
                    local["timing"]["proof_identity_seconds"],
                "readiness_reduction_seconds": readiness_reduction_seconds,
                "integration_seconds": local["timing"]["integration_seconds"],
                "result_sha256": local["result"]["exact_value_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    with tempfile.TemporaryDirectory(
        dir=arm_parent,
        prefix=".m0-public-cpu-arm-",
    ) as public_arm_directory_text:
        public_arm_directory = Path(public_arm_directory_text)
        public_envelope, public_producer = _run_isolated_arm(
            kind="public-wrapper",
            output=public_arm_directory / "public-wrapper.json",
            run_id=run_id,
            run_dir=run_dir,
            hours=hours,
            wrfout_output_dir=wrfout_output_dir,
        )
        public = public_envelope["payload"]
    if (
        local_envelope["cpu_platform"] != public_envelope["cpu_platform"]
        or local_envelope["jax_version"] != public_envelope["jax_version"]
        or local_envelope["python"] != public_envelope["python"]
    ):
        raise RuntimeError("isolated exact/public CPU arm environments differ")
    cpu_platform = local_envelope["cpu_platform"]
    print(
        json.dumps(
            {
                "stage": "public_wrapper_complete",
                "result_sha256": public["result"]["exact_value_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    identity_checks = {
        "arguments_identical": (
            public["captured_call"]["argument_identity"]
            == local["call"]["argument_identity"]
        ),
        "lowered_program_identical": (
            public["captured_call"]["lowered_program_sha256"]
            == local["call"]["lowered_program_sha256"]
        ),
        "called_function_identical": (
            public["captured_call"]["called_function"]["ast_sha256"]
            == local["production_binding"]["jit_identity"]["ast_sha256"]
            and public["captured_call"]["called_function"]["fqname"]
            == local["production_binding"]["jit_identity"]["fqname"]
        ),
        "output_digest_identical": (
            public["result"]["exact_value_sha256"]
            == local["result"]["exact_value_sha256"]
        ),
        "returned_state_semantics_identical": (
            public["result"]["semantics"] == local["result"]["semantics"]
        ),
        "hours_identical": (
            float(hours)
            == float(public["case"]["hours"])
            == float(local["case"]["hours"])
            == float(local["call"]["hours"])
        ),
    }
    if not all(identity_checks.values()):
        raise RuntimeError(f"real FAST exact-boundary identity failed: {identity_checks}")
    cache_after = _cpu_reference_cache_contract()
    if (
        cache_after["top_level_entry_sha256"]
        != cache_before["top_level_entry_sha256"]
        or cache_after["top_level_entry_bytes"]
        != cache_before["top_level_entry_bytes"]
    ):
        raise RuntimeError("historical top-level CPU cache entry changed during oracle")

    pallas_manifest_path = (
        REPO / "proofs/v025/m0/pallas_fast_savepoint_manifest.json"
    )
    pallas_manifest = json.loads(pallas_manifest_path.read_text())
    if (
        pallas_manifest.get("native_pallas_verdict") != "MISSING"
        or pallas_manifest.get("device_touched") is not False
    ):
        raise RuntimeError("CPU proof may not carry a native Pallas verdict")

    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device_policy": "closed",
        "device_touched": False,
        "device_queries": [],
        "receipts_read": [],
        "receipts_consumed": [],
        "gpu_windows": {
            "W1": "MISSING",
            "W2": "MISSING",
            "W3": "MISSING",
        },
        "native_pallas_verdict": "MISSING",
        "cpu_platform": cpu_platform,
        "cpu_arm_isolation": {
            "status": "PASS",
            "manager_accelerator_modules": _accelerator_modules(),
            "sequential": True,
            "fresh_process_per_arm": True,
            "same_process_double_integration": False,
            "process_group_attached": True,
            "single_core_xla_cpu_deadlock_guard": True,
            "local_exact": local_producer,
            "public_wrapper": public_producer,
        },
        "long_run_preflight": {
            "status": "PASS",
            "output_directory": output_preflight,
            "memory": memory_preflight,
            "representative_payload_validators": fixture_preflight,
            "estimated_seconds_per_isolated_arm":
                ISOLATED_ARM_ESTIMATED_SECONDS,
            "systemd_inhibit_required": True,
            "clock_discontinuity_self_invalidating": True,
        },
        "machine": {
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
            "producer_argv": list(sys.argv),
        },
        "production_identity": _production_identity(),
        "source_contract": source_validation,
        "plan_contract": plan_validation,
        "plan_sha256": hashlib.sha256(
            json.dumps(
                plan, sort_keys=True, separators=(",", ":"), default=str
            ).encode("utf-8")
        ).hexdigest(),
        "mutation_matrix": mutation_matrix,
        "clock_accounting": clock_accounting,
        "cpu_compile_cache": {
            **cache_after,
            "unchanged_during_oracle": True,
        },
        "real_fast_cpu_identity": {
            "status": "PASS",
            "run_dir": str(run_dir.resolve()),
            "hours": float(hours),
            "cpu_state_zeros_adapter": {
                "enabled": True,
                "scope":
                    "gpuwrf.contracts.state._gpu_device during _build_real_case only",
                "reason":
                    "production State.zeros explicitly rejects a CPU backend",
            },
            "checks": identity_checks,
            "local_exact_compiled": local,
            "public_wrapper": public,
        },
        "c1_exact_result_wrfout": local["wrfout"],
        "pallas_wiring": {
            "status": "CPU_REAL_FAST_SAVEPOINT_GREEN_NATIVE_PALLAS_MISSING",
            "manifest": str(pallas_manifest_path.relative_to(REPO)),
            "manifest_sha256": _sha256_file(pallas_manifest_path),
            "payload_path": pallas_manifest["payload_path"],
            "payload_sha256": pallas_manifest["payload_sha256"],
            "shape": pallas_manifest["arrays"]["rhs"]["shape"],
            "warm_iterations_future": 100,
            "alternating_pairs_future": 5,
            "transfer_audit_future": True,
            "compile_separated_future": True,
        },
        "performance_implications": {
            "readiness_is_direct_endpoint": True,
            "integration_is_direct_synchronized_endpoint": True,
            "phase_subtraction": False,
            "profiled_clean_share_one_invocation_helper": True,
            "clean_instrumentation_absent": True,
            "cpu_local_readiness_seconds":
                local["timing"]["readiness_seconds"],
            "cpu_proof_identity_seconds":
                local["timing"]["proof_identity_seconds"],
            "readiness_reduction_seconds": readiness_reduction_seconds,
            "readiness_reduction_covers_measured_proof_work": True,
            "cpu_readiness_cache_matched_to_pre_repair": True,
            "cpu_local_integration_seconds":
                local["timing"]["integration_seconds"],
            "cpu_times_are_not_gpu_economy_evidence": True,
            "cpu_oracle_single_core_deadlock_guard": True,
            "cache_disk_scaling":
                "two private snapshots; O(2 * qualified cache bytes)",
            "proof_hashing_ram_scaling":
                "streamed per leaf; O(largest state leaf)",
            "stablehlo_encoding_ram_scaling": (
                "one StableHLO text plus <=1 Mi-character UTF-8 chunk; "
                "no second full encoded byte copy"
            ),
            "wrfout_fixed_overhead_seconds": (
                local["wrfout"]["timing"]["inspection_end_monotonic_ns"]
                - local["wrfout"]["timing"][
                    "materialization_start_monotonic_ns"
                ]
            )
            / 1e9,
        },
        "jax_version": local_envelope["jax_version"],
        "python": sys.version.split()[0],
    }
    _atomic_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-dir", type=Path, default=child.FAST_RUN_DIR)
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--wrfout-output-dir", type=Path)
    parser.add_argument(
        "--run-id",
        default="m0-exact-boundary-real-fast-cpu-20260728-r1",
    )
    parser.add_argument("--isolated-arm", choices=sorted(ISOLATED_ARMS))
    parser.add_argument("--arm-output", type=Path)
    parser.add_argument("--parent-launch-monotonic-ns", type=int)
    args = parser.parse_args()
    if args.isolated_arm is not None:
        if (
            args.arm_output is None
            or args.wrfout_output_dir is None
            or args.parent_launch_monotonic_ns is None
            or args.parent_launch_monotonic_ns <= 0
        ):
            parser.error(
                "isolated arm requires --arm-output, --wrfout-output-dir, "
                "and positive --parent-launch-monotonic-ns"
            )
        _isolated_arm_child(
            kind=args.isolated_arm,
            output=args.arm_output,
            run_id=args.run_id,
            run_dir=args.run_dir,
            hours=args.hours,
            wrfout_output_dir=args.wrfout_output_dir,
            parent_launch_monotonic_ns=args.parent_launch_monotonic_ns,
        )
        return 0
    if args.wrfout_output_dir is None:
        parser.error("--wrfout-output-dir is required")
    payload = build(
        output=args.output,
        run_dir=args.run_dir,
        hours=args.hours,
        wrfout_output_dir=args.wrfout_output_dir,
        run_id=args.run_id,
    )
    print(json.dumps({
        "status": payload["status"],
        "output": str(args.output),
        "identity_checks": payload["real_fast_cpu_identity"]["checks"],
        "cpu_local_readiness_seconds":
            payload["performance_implications"]["cpu_local_readiness_seconds"],
        "cpu_local_integration_seconds":
            payload["performance_implications"]["cpu_local_integration_seconds"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
