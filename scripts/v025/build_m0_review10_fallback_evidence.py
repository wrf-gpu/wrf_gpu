#!/usr/bin/env python3
"""Build terminal CPU evidence for the bounded Review-10 fallback assembly."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import gpu_window_registry as registry  # noqa: E402
import m0_cpu_boundary_preflight as cpu_preflight  # noqa: E402
import m0_review10_fallback_capability as capability  # noqa: E402
import run_gpu_arm as arm  # noqa: E402
import wrf_source_authority as wsa  # noqa: E402


SCHEMA = "wrf_gpu2.v025.m0.review10_fallback_cpu_evidence.v1"
VERDICT_GREEN = "CPU_REVIEW10_FALLBACK_ASSEMBLY_GREEN"
CANDIDATE = "6ff0984984eddf697be2b675784ad18003770c01"
CANDIDATE_TERMINAL_IMPLEMENTATION = (
    "7d0bf38188feea37f0055d86fee55255454d8968"
)
FAST = Path("<DATA_ROOT>/wrf_gpu2/v025/m0/cpu_arms/fastbind_r1")
LEDGER = (
    REPO
    / ".agent/sprints/2026-07-27-v0250-m0-setup/"
    "gpu_coordination_spent.json"
)
RETAINED_ARTIFACTS = {
    "receipt": (
        REPO
        / ".agent/sprints/2026-07-27-v0250-m0-setup/"
        "M0_CORE_W1_W2_W3_SESSION_RECEIPT.json"
    ),
    "held_wrapper_log": (
        REPO
        / "proofs/v025/m0/autotune0_three_window/held_session/"
        "held_wrapper.log"
    ),
    "session": (
        REPO
        / "proofs/v025/m0/autotune0_three_window/held_session/"
        "m0_core_session.json"
    ),
    "terminal": (
        REPO
        / "proofs/v025/m0/autotune0_three_window/held_session/"
        "m0_core_terminal.json"
    ),
    "lock_release": (
        REPO
        / "proofs/v025/m0/autotune0_three_window/held_session/"
        "session_lock_release.json"
    ),
    "owner": (
        REPO
        / "proofs/v025/m0/autotune0_three_window/held_session/"
        "session_owner.json"
    ),
    "session_identity": (
        REPO
        / "proofs/v025/m0/autotune0_three_window/held_session/"
        "session_preflight_identity.json"
    ),
    "cpu_preflight": (
        REPO
        / "proofs/v025/m0/autotune0_three_window/held_session/"
        "cpu_preflight.json"
    ),
    "prelock_revalidation": (
        REPO
        / "proofs/v025/m0/autotune0_three_window/held_session/"
        "prelock_revalidation.json"
    ),
}

IMPLEMENTATION_PATHS = {
    "scripts/v025/build_m0_review10_fallback_evidence.py",
    "scripts/v025/gpu_window_registry.py",
    "scripts/v025/m0_core_session_protocol.py",
    "scripts/v025/m0_cpu_boundary_preflight.py",
    "scripts/v025/m0_exact_boundary_child.py",
    "scripts/v025/m0_exact_boundary_contract.py",
    "scripts/v025/m0_review10_fallback_capability.py",
    "scripts/v025/m0_three_window_executor.py",
    "scripts/v025/m0_window_parent.py",
    "scripts/v025/real_state.py",
    "scripts/v025/run_gpu_arm.py",
    "scripts/v025/run_shape_sweep.py",
    "scripts/v025/step1_dryrun.py",
    "scripts/v025/wrf_source_authority.py",
    "tests/v025/test_gpu_arm.py",
    "tests/v025/test_m0_c1_c2_cpu_closure.py",
    "tests/v025/test_m0_exact_boundary.py",
    "tests/v025/test_m0_held_session_repair.py",
    "tests/v025/test_m0_review10_fallback_assembly.py",
    "tests/v025/test_m0_step1_evidence_boundary.py",
    "tests/v025/test_receipt_spending.py",
    "tests/v025/test_step1_boundary.py",
    "tests/v025/test_step1_driver.py",
}
EXCLUDED_CANDIDATE_R3_R4 = (
    "scripts/v025/m0_w2_attribution.py",
    "scripts/v025/m0_postlock_census.py",
    "scripts/v025/build_m0_final_r1_r4_evidence.py",
    "tests/v025/test_m0_final_r1_r4_repair.py",
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _git_optional_object(spec: str) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", spec],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _json_write_no_replace(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _text_write_no_replace(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _prepare_raw(raw_root: Path) -> dict[str, Any]:
    if os.path.lexists(raw_root):
        raise RuntimeError(f"raw evidence root already exists: {raw_root}")
    raw_root.mkdir(parents=True, exist_ok=False)
    authority_environment = dict(os.environ)
    for variable in wsa.ROOT_ENV_VARS:
        authority_environment[variable] = str(wsa.CANONICAL_ROOT)
    authority = wsa.build_source_authority(
        namelist_path=FAST / "namelist.input",
        environ=authority_environment,
    )
    authority_path = raw_root / "wrf_source_authority.json"
    wsa.write_authority(authority_path, authority)

    boundary_path = raw_root / "cpu_real_boundary_preflight.json"
    boundary_environment = {
        key: value
        for key, value in authority_environment.items()
        if not key.startswith("GPUWRF_GPU_LOCK_")
        and key
        not in {
            "GPUWRF_M0_EVIDENCE",
            "GPUWRF_M0_EVIDENCE_PATH",
            "GPUWRF_M0_RUN_ID",
            "GPUWRF_M0_SOURCE_SHA256",
            "GPUWRF_M0_CONFIG_SHA256",
            "GPUWRF_M0_INPUT_MANIFEST_SHA256",
            "GPUWRF_M0_DEVICE_UUID",
        }
    }
    boundary_environment.update(wsa.child_environment_binding(authority))
    boundary_environment.update(
        {
            wsa.AUTHORITY_ENV_VAR: str(authority_path),
            "JAX_PLATFORMS": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
            "GPUWRF_JAX_CACHE": "0",
            "GPUWRF_JAX_CACHE_LOCK": "0",
            "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false",
        }
    )
    command = [
        sys.executable,
        str(SCRIPT_DIR / "m0_cpu_boundary_preflight.py"),
        "--authority",
        str(authority_path),
        "--run-dir",
        str(FAST),
        "--output",
        str(boundary_path),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO,
        env=boundary_environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=300.0,
    )
    if completed.returncode != 0 or not boundary_path.is_file():
        raise RuntimeError(
            "real CPU boundary failed: "
            + (completed.stderr or completed.stdout)[-2000:]
        )
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    capability.validate_real_boundary(
        boundary,
        expected_authority_sha256=authority["authority_sha256"],
    )
    declaration = capability.build_declaration(
        boundary,
        boundary_path=boundary_path,
        environ=boundary_environment,
    )
    declaration_path = raw_root / "capture_capability_declaration.json"
    _json_write_no_replace(declaration_path, declaration)
    for path in (authority_path, boundary_path, declaration_path):
        os.chmod(path, 0o444)
    return {
        "command": command,
        "authority": authority,
        "authority_path": authority_path,
        "boundary": boundary,
        "boundary_path": boundary_path,
        "declaration": declaration,
        "declaration_path": declaration_path,
        "subprocess_returncode": completed.returncode,
    }


def _junit(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals = {
        name: sum(int(suite.attrib.get(name, "0")) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    cases = []
    for suite in suites:
        for case in suite.findall("testcase"):
            cases.append(
                {
                    "classname": case.attrib.get("classname"),
                    "name": case.attrib.get("name"),
                    "time_seconds": float(case.attrib.get("time", "0")),
                    "status": (
                        "FAILED"
                        if case.find("failure") is not None
                        else "ERROR"
                        if case.find("error") is not None
                        else "SKIPPED"
                        if case.find("skipped") is not None
                        else "PASSED"
                    ),
                }
            )
    if totals["failures"] or totals["errors"]:
        raise RuntimeError(f"JUnit is not green: {path}: {totals}")
    return {
        "path": str(path.relative_to(REPO.resolve())),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        **totals,
        "cases_sha256": canonical_sha256(cases),
    }


def _resign(declaration: dict[str, Any]) -> dict[str, Any]:
    unsigned = {
        key: value
        for key, value in declaration.items()
        if key not in {"content_address", "signature"}
    }
    digest = capability.canonical_sha256(unsigned)
    declaration["content_address"] = {
        "algorithm": "sha256",
        "sha256": digest,
    }
    declaration["signature"] = {
        "scheme": "sha256-canonical-json-content-signature",
        "signed_sha256": digest,
    }
    return declaration


def _refusal(name: str, action: Callable[[], Any]) -> tuple[str, dict[str, Any]]:
    try:
        action()
    except Exception as exc:  # noqa: BLE001 - mutation transcript
        return name, {
            "status": "REFUSED",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "device_action": False,
            "coordination_action": False,
            "real_receipt_or_lock_touched": False,
        }
    return name, {
        "status": "UNEXPECTED_ACCEPT",
        "device_action": False,
        "coordination_action": False,
        "real_receipt_or_lock_touched": False,
    }


def _synthetic_receipt() -> arm.CoordinationReceipt:
    stamp = datetime.now(timezone.utc).isoformat()
    return arm.CoordinationReceipt(
        window=registry.SESSION_LABEL,
        requested_at_utc=stamp,
        replies={
            manager: {
                "affirmative": True,
                "verbatim": f"{manager} CPU mutation affirmative",
                "received_at_utc": stamp,
            }
            for manager in arm.REQUIRED_MANAGERS
        },
    )


def _r1_mutations(
    authority: dict[str, Any],
    authority_path: Path,
    boundary: dict[str, Any],
) -> dict[str, Any]:
    outcomes: dict[str, Any] = {}
    roots = {
        variable: str(wsa.CANONICAL_ROOT)
        for variable in wsa.ROOT_ENV_VARS
    }
    actions: dict[str, Callable[[], Any]] = {
        "unset_wrf_roots": lambda: wsa.resolve_authority_root(environ={}),
        "split_wrf_roots": lambda: wsa.resolve_authority_root(
            environ={
                wsa.ENV_VAR: str(wsa.CANONICAL_ROOT),
                wsa.SRC_ENV_VAR: str(REPO),
            }
        ),
        "non_cpu_preflight_platform": lambda: (
            cpu_preflight.validate_device_free_environment(
                {"JAX_PLATFORMS": "gpu", "CUDA_VISIBLE_DEVICES": ""}
            )
        ),
        "visible_cuda_device_in_preflight": lambda: (
            cpu_preflight.validate_device_free_environment(
                {"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "0"}
            )
        ),
        "lock_environment_leaks_into_preflight": lambda: (
            cpu_preflight.validate_device_free_environment(
                {
                    "JAX_PLATFORMS": "cpu",
                    "CUDA_VISIBLE_DEVICES": "",
                    "GPUWRF_GPU_LOCK_TOKEN": "leaked",
                }
            )
        ),
    }

    content_mutation = copy.deepcopy(authority)
    content_mutation["files"][0]["sha256"] = "0" * 64
    actions["mutated_authority_content_hash"] = lambda: (
        wsa.validate_source_authority(content_mutation, environ=roots)
    )
    source_mutation = copy.deepcopy(content_mutation)
    source_mutation["authority_sha256"] = wsa.canonical_sha256(
        {
            key: value
            for key, value in source_mutation.items()
            if key != "authority_sha256"
        }
    )
    actions["mutated_authoritative_source_file"] = lambda: (
        wsa.validate_source_authority(source_mutation, environ=roots)
    )

    def imported_before_authority() -> None:
        sentinel = object()
        previous = sys.modules.get("jax", sentinel)
        sys.modules["jax"] = object()
        try:
            wsa.assert_child_binding(
                environ={
                    **roots,
                    wsa.AUTHORITY_ENV_VAR: str(authority_path),
                }
            )
        finally:
            if previous is sentinel:
                sys.modules.pop("jax", None)
            else:
                sys.modules["jax"] = previous

    actions["product_import_before_authority"] = imported_before_authority
    boundary_mutations = {
        "native_loader_not_executed_once": ("native_loader_calls", 0),
        "exact_lower_not_executed_once": ("exact_lower_calls", 0),
        "compile_or_device_call_during_preflight": ("compile_calls", 1),
    }
    for name, (field, value) in boundary_mutations.items():
        mutated = copy.deepcopy(boundary)
        mutated["observations"][field] = value
        actions[name] = lambda mutated=mutated: (
            capability.validate_real_boundary(mutated)
        )
    for name in ("authority_drift_before_spend", "authority_drift_at_stage_entry"):
        actions[name] = lambda: wsa.validate_source_authority(
            source_mutation, environ=roots
        )

    for name in capability.HARD_GATE_MUTATIONS["R1"]:
        mutation, result = _refusal(name, actions[name])
        outcomes[mutation] = result
    return outcomes


def _r2_mutations() -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _synthetic_receipt()
    outcomes: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="m0-r2-cpu-") as temporary:
        root = Path(temporary)

        def environment(
            *,
            env_token: str = "exact-token",
            holder_token: str = "exact-token",
            holder_label: str = registry.SESSION_LABEL,
            exported_label: str | None = registry.SESSION_LABEL,
            command: str = "cpu-only",
            suffix: str = "",
        ) -> dict[str, str]:
            holder = root / f"holder{suffix}"
            holder.write_text(
                f"holder={holder_label} pid=1 token={holder_token} "
                f"cmd={command}\n",
                encoding="utf-8",
            )
            result = {
                "GPUWRF_GPU_LOCK_HELD": "1",
                "GPUWRF_GPU_LOCK_TOKEN": env_token,
                "GPUWRF_GPU_LOCK_HOLDER_FILE": str(holder),
            }
            if exported_label is not None:
                result["GPUWRF_GPU_LOCK_LABEL"] = exported_label
            return result

        positive = arm.validate_spending_identity(
            window=registry.SESSION_LABEL,
            receipt=receipt,
            env=environment(suffix="-positive"),
            ledger_path=root / "absent-positive-ledger.json",
        )
        positive_control = {
            "status": "PASS",
            "token_matched": positive["token_matched"],
            "label_matched": positive["label_matched"],
            "ledger_created": (root / "absent-positive-ledger.json").exists(),
            "receipt_spent": False,
        }
        actions: dict[str, Callable[[], Any]] = {
            "one_character_token": lambda: arm.validate_spending_identity(
                window=registry.SESSION_LABEL,
                receipt=receipt,
                env=environment(
                    holder_token="exact-tokem", suffix="-character"
                ),
                ledger_path=root / "ledger-character.json",
            ),
            "strict_prefix_token": lambda: arm.validate_spending_identity(
                window=registry.SESSION_LABEL,
                receipt=receipt,
                env=environment(
                    holder_token="exact-token-extra", suffix="-prefix"
                ),
                ledger_path=root / "ledger-prefix.json",
            ),
            "strict_suffix_token": lambda: arm.validate_spending_identity(
                window=registry.SESSION_LABEL,
                receipt=receipt,
                env=environment(
                    holder_token="prefix-exact-token", suffix="-suffix"
                ),
                ledger_path=root / "ledger-suffix.json",
            ),
            "token_only_inside_cmd": lambda: arm.validate_spending_identity(
                window=registry.SESSION_LABEL,
                receipt=receipt,
                env=environment(
                    holder_token="different-token",
                    command="cpu token=exact-token",
                    suffix="-cmd",
                ),
                ledger_path=root / "ledger-cmd.json",
            ),
            "wrong_holder_label": lambda: arm.validate_spending_identity(
                window=registry.SESSION_LABEL,
                receipt=receipt,
                env=environment(
                    holder_label="baseline-census", suffix="-label"
                ),
                ledger_path=root / "ledger-label.json",
            ),
            "missing_exported_label": lambda: arm.validate_spending_identity(
                window=registry.SESSION_LABEL,
                receipt=receipt,
                env=environment(exported_label=None, suffix="-missing-label"),
                ledger_path=root / "ledger-missing-label.json",
            ),
        }

        def mutated_registry() -> None:
            saved = registry.REGISTERED_WINDOWS[registry.SESSION_LABEL]
            del registry.REGISTERED_WINDOWS[registry.SESSION_LABEL]
            try:
                arm.validate_spending_identity(
                    window=registry.SESSION_LABEL,
                    receipt=receipt,
                    env=environment(suffix="-registry"),
                    ledger_path=root / "ledger-registry.json",
                )
            finally:
                registry.REGISTERED_WINDOWS[registry.SESSION_LABEL] = saved

        actions["mutated_registry"] = mutated_registry
        stale = root / "stale-ledger.json"
        stale.write_text(
            json.dumps(
                {
                    "spent": {
                        receipt.fingerprint(): {
                            "spent_at_utc": datetime.now(
                                timezone.utc
                            ).isoformat(),
                            "reason": "CPU mutation",
                        }
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )
        actions["stale_spend_ledger"] = lambda: (
            arm.validate_spending_identity(
                window=registry.SESSION_LABEL,
                receipt=receipt,
                env=environment(suffix="-stale"),
                ledger_path=stale,
            )
        )
        for name in capability.HARD_GATE_MUTATIONS["R2"]:
            mutation, result = _refusal(name, actions[name])
            outcomes[mutation] = result
    return outcomes, positive_control


def _c0_mutations(
    boundary: dict[str, Any],
    boundary_path: Path,
    declaration: dict[str, Any],
) -> dict[str, Any]:
    outcomes: dict[str, Any] = {}
    actions: dict[str, Callable[[], Any]] = {}
    for name in (
        "missing_lowered_trip_count",
        "lowered_configured_count_mismatch",
        "wrong_denominator_method",
        "stablehlo_hash_mismatch",
    ):
        mutated = copy.deepcopy(boundary)
        trip = mutated["lowered_program"]["integration_trip_count"]
        if name == "missing_lowered_trip_count":
            del mutated["lowered_program"]["integration_trip_count"]
        elif name == "lowered_configured_count_mismatch":
            trip["configured_cross_check"]["exact_steps"] -= 1
        elif name == "wrong_denominator_method":
            trip["method"] = "configured-count-observed"
        else:
            trip["stablehlo_sha256"] = "0" * 64
        actions[name] = lambda mutated=mutated: (
            capability.build_declaration(
                mutated, boundary_path=boundary_path, environ={}
            )
        )
    actions["duplicate_production_integration_range"] = lambda: (
        capability.build_declaration(
            boundary,
            boundary_path=boundary_path,
            environ={"GPUWRF_M0_EVIDENCE": "1"},
        )
    )

    declaration_mutations = {
        "available_set_drift": lambda value: value["available"].pop(
            next(iter(value["available"]))
        ),
        "missing_set_drift": lambda value: value["missing"].pop(
            next(iter(value["missing"]))
        ),
        "integration_clip_wrong_emitter": lambda value: (
            value["integration_clip"].update(
                {
                    "emitter": "src/gpuwrf/runtime/operational_mode.py",
                    "production_operational_mode_emitter": True,
                }
            )
        ),
    }
    invariant_mutations = {
        "raw_capture_hash_binding_removed":
            "raw_capture_hash_bound_before_postlock_analysis",
        "postlock_order_removed":
            "postcapture_analysis_after_lock_release",
        "postcapture_readonly_preservation_removed":
            "postcapture_failure_preserves_raw_capture_and_readonly_sqlite",
        "w2_w3_reservation_removed":
            "insufficient_w2_w3_time_stops_before_w2",
        "extra_device_arm_after_w3":
            "no_extra_device_arm_after_w2_hash_and_w3_stop",
    }
    for name, mutate in declaration_mutations.items():
        changed = copy.deepcopy(declaration)
        mutate(changed)
        actions[name] = lambda changed=changed: (
            capability.validate_declaration(_resign(changed))
        )
    for name, invariant in invariant_mutations.items():
        changed = copy.deepcopy(declaration)
        changed["fallback_invariants"][invariant]["status"] = "FAIL"
        actions[name] = lambda changed=changed: (
            capability.validate_declaration(_resign(changed))
        )
    for name in capability.HARD_GATE_MUTATIONS["C0"]:
        mutation, result = _refusal(name, actions[name])
        outcomes[mutation] = result
    return outcomes


def _changed_tree(
    manager_base: str, implementation_commit: str
) -> dict[str, Any]:
    rows = []
    changed_paths = set()
    output = _git(
        "diff", "--name-status", manager_base, implementation_commit, "--"
    )
    for line in output.splitlines():
        if not line:
            continue
        status, path = line.split("\t", 1)
        changed_paths.add(path)
        blob = (
            None
            if status.startswith("D")
            else _git("rev-parse", f"{implementation_commit}:{path}")
        )
        rows.append({"status": status, "path": path, "blob_oid": blob})
    outside = sorted(changed_paths - IMPLEMENTATION_PATHS)
    missing_expected = sorted(IMPLEMENTATION_PATHS - changed_paths)
    if outside:
        raise RuntimeError(f"implementation changed out-of-scope paths: {outside}")
    if missing_expected:
        raise RuntimeError(
            f"implementation manifest is missing expected paths: {missing_expected}"
        )
    return {
        "manager_base": manager_base,
        "terminal_implementation_commit": implementation_commit,
        "terminal_implementation_tree": _git(
            "rev-parse", f"{implementation_commit}^{{tree}}"
        ),
        "entries": sorted(rows, key=lambda row: row["path"]),
        "changed_tree_sha256": canonical_sha256(
            sorted(rows, key=lambda row: row["path"])
        ),
        "set_equality_with_frozen_manifest": changed_paths
        == IMPLEMENTATION_PATHS,
    }


def _candidate_attribution(
    manager_base: str, implementation_commit: str
) -> dict[str, Any]:
    replayed = {}
    for path in sorted(
        IMPLEMENTATION_PATHS
        & {
            "scripts/v025/gpu_window_registry.py",
            "scripts/v025/m0_core_session_protocol.py",
            "scripts/v025/m0_cpu_boundary_preflight.py",
            "scripts/v025/m0_exact_boundary_child.py",
            "scripts/v025/m0_exact_boundary_contract.py",
            "scripts/v025/m0_three_window_executor.py",
            "scripts/v025/m0_window_parent.py",
            "scripts/v025/real_state.py",
            "scripts/v025/run_gpu_arm.py",
            "scripts/v025/run_shape_sweep.py",
            "scripts/v025/step1_dryrun.py",
            "scripts/v025/wrf_source_authority.py",
        }
    ):
        candidate_blob = _git(
            "rev-parse", f"{CANDIDATE_TERMINAL_IMPLEMENTATION}:{path}"
        )
        implementation_blob = _git(
            "rev-parse", f"{implementation_commit}:{path}"
        )
        replayed[path] = {
            "candidate_blob_oid": candidate_blob,
            "implementation_blob_oid": implementation_blob,
            "exact_candidate_blob": candidate_blob == implementation_blob,
            "attribution": (
                "exact-R1/R2-replay"
                if candidate_blob == implementation_blob
                else "R1/R2-replay-with-Review11-or-C0-repair"
            ),
        }
    excluded = {}
    for path in EXCLUDED_CANDIDATE_R3_R4:
        base_blob = _git_optional_object(f"{manager_base}:{path}")
        implementation_blob = _git_optional_object(
            f"{implementation_commit}:{path}"
        )
        excluded[path] = {
            "manager_base_blob_oid": base_blob,
            "implementation_blob_oid": implementation_blob,
            "unchanged_or_absent_from_manager_base":
                base_blob == implementation_blob,
        }
    if not all(
        record["unchanged_or_absent_from_manager_base"]
        for record in excluded.values()
    ):
        raise RuntimeError("candidate R3/R4 subset leaked into implementation")
    return {
        "candidate_terminal_commit": CANDIDATE,
        "candidate_terminal_implementation_commit":
            CANDIDATE_TERMINAL_IMPLEMENTATION,
        "manager_candidate_merge_base": _git(
            "merge-base", manager_base, CANDIDATE
        ),
        "replayed_r1_r2": replayed,
        "excluded_r3_r4": excluded,
        "candidate_merged_wholesale": False,
    }


def _system_policy() -> dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0))
    cgroup_line = next(
        (
            line
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        ),
        "",
    )
    relative = cgroup_line.partition("0::")[2]
    memory_path = Path("/sys/fs/cgroup") / relative.lstrip("/") / "memory.max"
    memory_max = (
        memory_path.read_text(encoding="utf-8").strip()
        if memory_path.is_file()
        else None
    )
    unit = Path(relative).name if relative else ""
    properties: dict[str, str] = {}
    if unit:
        completed = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                unit,
                "--property=DevicePolicy",
                "--property=MemoryMax",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            properties = dict(
                line.split("=", 1)
                for line in completed.stdout.splitlines()
                if "=" in line
            )
    policy = {
        "affinity": affinity,
        "required_affinity": [0, 1, 2, 3],
        "cgroup": relative,
        "unit": unit,
        "memory_max_cgroup": memory_max,
        "unit_properties": properties,
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "lock_environment_absent": not any(
            key.startswith("GPUWRF_GPU_LOCK_") for key in os.environ
        ),
    }
    if (
        affinity != [0, 1, 2, 3]
        or memory_max != str(16 * 1024**3)
        or properties.get("DevicePolicy") != "closed"
        or properties.get("MemoryMax") != str(16 * 1024**3)
        or policy["JAX_PLATFORMS"] != "cpu"
        or policy["CUDA_VISIBLE_DEVICES"] != ""
        or policy["lock_environment_absent"] is not True
    ):
        raise RuntimeError(f"CPU evidence policy is not frozen: {policy}")
    return policy


def _artifact_identity(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {
        "path": str(path.relative_to(REPO.resolve())),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _report_text(proof: Mapping[str, Any]) -> str:
    denominator = proof["gates"]["C0"]["declaration"]["denominator"]
    focused = proof["tests"]["focused"]
    relevant = proof["tests"]["relevant_v025"]
    return f"""# Review-10 fallback assembly report

## Terminal verdict

`{proof['verdict']}`

This is CPU-only, critic-ready evidence. It does not authorize coordination or
a GPU run.

## Objective

Replay and re-prove only the retained candidate's R1 real-boundary and R2
spending-identity subset, repair Review-11 exact-token parsing, and prove the
frozen W2 capture capability before any spend.

## Result

- R1: the fresh CPU child bound the canonical WRF authority, executed the real
  native RRTMG-LW loader, built the FAST call, and lowered the exact production
  JIT. `lower()` does not prove later device compile/link.
- R2: registry, receipt, exported label, holder label, and the parsed holder
  `token=` field are exact-equality gates before the sole production spend
  caller. Strict prefix/suffix, command-only token, label, registry, and stale
  ledger mutations refused.
- C0: `{denominator['method']}` recovered
  `{denominator['segment_trip_counts']}` = `{denominator['steps']}` contiguous
  steps and matched the exact configured interval/timestep derivation.
- Capability sets are exact: nine AVAILABLE fields and three MISSING fields.
  The integration clip is emitted only by
  `scripts/v025/m0_exact_boundary_child.py`; `GPUWRF_M0_EVIDENCE` remained
  unset.
- The W1→C1→W2→W3 held graph and `4,220 <= 4,500 s` budget remain frozen. W2
  now refuses before launch unless the combined W2+W3 reservation remains.

## Validation

- Focused fallback suite: {focused['tests']} passed, {focused['failures']}
  failures, {focused['errors']} errors.
- Relevant retained v0.25 suite: {relevant['tests']} passed,
  {relevant['failures']} failures, {relevant['errors']} errors.
- Device policy: closed; memory ceiling: 16 GiB; CPU affinity: 0-3.
- No GPU, real receipt check/spend, canonical lock, coordination, or resource
  lock action occurred.

## Proof objects

- `proofs/v025/m0/m0_review10_fallback_cpu_evidence.json`
- `proofs/v025/m0/m0_review10_fallback_cpu_evidence.json.sha256`
- raw authority, real boundary, capability declaration, and JUnit XML under
  `.agent/sprints/2026-07-30-v0250-m0-review10-fallback-assembly/`

## Residual risk

Device compile/link and all measured W2 values remain unproved until a separate
critic accepts this CPU gate and the manager later obtains fresh coordination.
The declaration intentionally keeps achieved DRAM GB/s, occupancy/stall
counters, and `run_physics=False` MISSING.
"""


def _patch_text() -> str:
    return """# Patch proposals

No roadmap, contract, stable-memory, or R3/R4 patch is proposed by this sprint.

The bounded implementation repairs only Review-11 R1/R2/C0 requirements:
canonical source authority, exact holder token/label identity, exact lowered
trip-count capability, immutable pre-spend declaration, and the combined W2/W3
remaining-time refusal. The manager should dispatch the one contracted
different-model critic before any coordination decision.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manager-base", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--focused-junit", type=Path, required=True)
    parser.add_argument("--relevant-junit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--patch-proposals", type=Path, required=True)
    args = parser.parse_args()

    manager_base = _git("rev-parse", args.manager_base)
    implementation_commit = _git("rev-parse", args.implementation_commit)
    if implementation_commit != _git("rev-parse", "HEAD"):
        raise RuntimeError("implementation commit must be current HEAD")
    if _git("status", "--short", "--untracked-files=no"):
        raise RuntimeError("tracked worktree changed after implementation commit")
    policy = _system_policy()
    raw = _prepare_raw(args.raw_root)
    authority = raw["authority"]
    boundary = raw["boundary"]
    declaration = raw["declaration"]
    capability.validate_declaration(declaration)

    mutations = {
        "R1": _r1_mutations(
            authority, raw["authority_path"], boundary
        ),
    }
    mutations["R2"], positive_r2 = _r2_mutations()
    mutations["C0"] = _c0_mutations(
        boundary, raw["boundary_path"], declaration
    )
    expected_mutations = {
        gate: list(names)
        for gate, names in capability.HARD_GATE_MUTATIONS.items()
    }
    observed_mutations = {
        gate: list(outcomes)
        for gate, outcomes in mutations.items()
    }
    set_equality = {
        gate: set(expected_mutations[gate]) == set(observed_mutations[gate])
        for gate in expected_mutations
    }
    all_refused = all(
        result["status"] == "REFUSED"
        for outcomes in mutations.values()
        for result in outcomes.values()
    )
    if not all(set_equality.values()) or not all_refused:
        raise RuntimeError(
            "hard-gate mutation set equality/refusal failed: "
            f"{set_equality=}, {all_refused=}"
        )

    tests = {
        "focused": _junit(args.focused_junit),
        "relevant_v025": _junit(args.relevant_junit),
    }
    changed_tree = _changed_tree(manager_base, implementation_commit)
    attribution = _candidate_attribution(
        manager_base, implementation_commit
    )
    retained = {
        name: _artifact_identity(path)
        for name, path in RETAINED_ARTIFACTS.items()
    }
    raw_artifacts = {
        "wrf_source_authority": _artifact_identity(raw["authority_path"]),
        "cpu_real_boundary_preflight":
            _artifact_identity(raw["boundary_path"]),
        "capture_capability_declaration":
            _artifact_identity(raw["declaration_path"]),
    }
    proof: dict[str, Any] = {
        "schema": SCHEMA,
        "verdict": VERDICT_GREEN,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device_action": False,
        "coordination_action": False,
        "real_receipt_check_or_spend": False,
        "canonical_lock_action": False,
        "gpu_receipt_lock_or_coordination_prohibited_and_absent": True,
        "manager_base": manager_base,
        "terminal_implementation_commit": implementation_commit,
        "complete_changed_tree": changed_tree,
        "candidate_source_attribution": attribution,
        "production_tree": {
            "src_gpuwrf_tree": _git(
                "rev-parse", f"{implementation_commit}:src/gpuwrf"
            ),
            "expected": "a6885ceded260df2f5777d7366d75a5d38947cb7",
            "read_only": True,
        },
        "system_policy": policy,
        "gates": {
            "R1": {
                "status": "PASS",
                "authority_sha256": authority["authority_sha256"],
                "boundary_content_sha256": boundary["boundary_sha256"],
                "observations": boundary["observations"],
                "platforms": boundary["platforms"],
                "native_bundle_field_count":
                    boundary["native_bundle"]["field_count"],
                "accepted_residual": (
                    "lower() cannot prove later device compile/link"
                ),
            },
            "R2": {
                "status": "PASS",
                "registry_schema": registry.SCHEMA,
                "registry_fingerprint": registry.registry_fingerprint(),
                "registered_session_label": registry.SESSION_LABEL,
                "positive_control": positive_r2,
                "single_production_spend_caller": "run_gpu_arm.authorise",
                "committed_ledger": _artifact_identity(LEDGER),
            },
            "C0": {
                "status": "PASS",
                "declaration_content_sha256":
                    declaration["content_address"]["sha256"],
                "declaration": declaration,
            },
        },
        "hard_gate_mutations": {
            "expected": expected_mutations,
            "observed": observed_mutations,
            "set_equality": set_equality,
            "all_refused": all_refused,
            "outcomes": mutations,
        },
        "tests": tests,
        "raw_artifacts": raw_artifacts,
        "retained_first_session_read_only_bindings": retained,
        "source_test_command_manifest": {
            "source_manifest": sorted(IMPLEMENTATION_PATHS),
            "focused_command": (
                "systemd-run --user --scope -p DevicePolicy=closed "
                "-p MemoryMax=16G taskset -c 0-3 env JAX_PLATFORMS=cpu "
                "CUDA_VISIBLE_DEVICES= GPUWRF_JAX_CACHE=0 "
                "GPUWRF_JAX_CACHE_LOCK=0 "
                "XLA_FLAGS=--xla_cpu_multi_thread_eigen=false pytest -q "
                "tests/v025/test_m0_review10_fallback_assembly.py"
            ),
            "relevant_command": (
                "systemd-run --user --scope -p DevicePolicy=closed "
                "-p MemoryMax=16G taskset -c 0-3 env JAX_PLATFORMS=cpu "
                "CUDA_VISIBLE_DEVICES= GPUWRF_JAX_CACHE=0 "
                "GPUWRF_JAX_CACHE_LOCK=0 "
                "XLA_FLAGS=--xla_cpu_multi_thread_eigen=false pytest -q "
                "tests/v025/test_gpu_arm.py "
                "tests/v025/test_receipt_spending.py "
                "tests/v025/test_step1_boundary.py "
                "tests/v025/test_step1_driver.py "
                "tests/v025/test_m0_step1_evidence_boundary.py "
                "tests/v025/test_m0_exact_boundary.py "
                "tests/v025/test_m0_c1_c2_cpu_closure.py "
                "tests/v025/test_m0_held_session_repair.py"
            ),
            "proof_command": " ".join(sys.argv),
            "raw_boundary_command": raw["command"],
        },
        "known_preterminal_falsifications": [
            {
                "failure": (
                    "initial exact-trip extractor required direct iota carry"
                ),
                "repair": (
                    "bound iota through static broadcast/add index vector into "
                    "the while"
                ),
            },
            {
                "failure": "initial condition parser accepted only i32 bounds",
                "repair": "bound the emitted i64 scan-counter condition",
            },
            {
                "failure": (
                    "JSON sort order invalidated tuple-order set validation"
                ),
                "repair": "enforced mathematical set equality",
            },
        ],
        "terminal_disposition": (
            "critic-ready CPU evidence only; no GPU coordination authorized"
        ),
    }
    if (
        proof["production_tree"]["src_gpuwrf_tree"]
        != proof["production_tree"]["expected"]
    ):
        raise RuntimeError("src/gpuwrf tree changed")
    proof["proof_content_sha256"] = canonical_sha256(proof)
    _json_write_no_replace(args.output, proof)
    proof_file_hash = sha256_file(args.output)
    _text_write_no_replace(
        Path(str(args.output) + ".sha256"),
        f"{proof_file_hash}  {args.output.name}\n",
    )
    _text_write_no_replace(args.report, _report_text(proof))
    _text_write_no_replace(args.patch_proposals, _patch_text())
    print(
        json.dumps(
            {
                "verdict": proof["verdict"],
                "proof": str(args.output),
                "proof_file_sha256": proof_file_hash,
                "implementation_commit": implementation_commit,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
