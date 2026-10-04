#!/usr/bin/env python3
"""One-shot fixed-tree d03 pre-PBL capture under the canonical lock-v2.

The proven pristine capture engine remains the implementation.  This profile
binds a new contract amendment, nonce, output namespace, and fixed model tree.
Import and every pre-arm mode are backend-dark.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-19-v0234-gpt-single-authority-attribution"
CONTRACT = SPRINT / "CONTRACT.md"
CONTRACT_SHA256 = "08a2db93015a056133520bf1132acb4b0868969ff2eb215b297b924da6db2ed1"
CONTRACT_COMMIT = "68d30d515d7cc4ce157de33528d97364d85dad55"
AMENDMENT = REPO / ".agent/patches/2026-07-19-v0234-gpt-postfix-sp2-corrected-arm.md"
AMENDMENT_SHA256 = "d60a18f69e7f0d5a231ce047483924ca3f24a217b67ff1d9ecb5d5cc3365fd71"
AMENDMENT_COMMIT = "b7ef240c0c4f9916419f4dfa660c4a04fde51275"
ACCEPTED_COMPONENT_HEAD = "672f55c0e6ac5ede82c288cdcaed63fd71a5e9a6"
FIXED_COUPLER = REPO / "src/gpuwrf/coupling/physics_couplers.py"
FIXED_COUPLER_SHA256 = "694e51f8e9df30d7c5abd3729c8eb756b6dbfd45c83bf774f95a23747b0b2283"

NONCE = "a73aef522fbdca9c88ec1c50464db6e51819f47aa00cc9493a1fe2616cd2ebfe"
PREFIX = NONCE[:16]
AUTHORITY_ROOT = Path(f"<DATA_ROOT>/wrf_gpu2/v0234_gpt_postfix_sp2_{PREFIX}")
CAPTURE_ROOT = AUTHORITY_ROOT / f"capture/authentic-{PREFIX}"
RESERVATION = AUTHORITY_ROOT / "control/authority-reservation.json"
EXECUTION_RECEIPT = SPRINT / "postfix-corrected-gpu-execution-receipt.json"
LOCK_LABEL = f"v0234-gpt-postfix-sp2-{PREFIX}"
CONSUMED_PREFIXES = (
    "0b18530a1dc9cac2", "ac6712170cbe5084", "2e1ad51a081af145",
)

BASE_SCRIPT = REPO / "scripts/v0234_pristine_pbl_capture.py"
BASE_SHA256 = "e3b9c7cbdfc498613e44ccbbe854c198ae39739f8d695c37cb4694756161cebc"
SPEC = importlib.util.spec_from_file_location("v0234_postfix_capture_base", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("capture base import unavailable")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


def configure_base() -> None:
    """Bind the historical engine to this amendment without editing it."""

    base.SPRINT = SPRINT
    base.CONTRACT = CONTRACT
    base.CONTRACT_SHA256 = CONTRACT_SHA256
    base.CONTRACT_COMMIT = CONTRACT_COMMIT
    base.CONTRACT_PATCH = AMENDMENT
    base.CONTRACT_PATCH_SHA256 = AMENDMENT_SHA256
    base.CONTRACT_PATCH_COMMIT = AMENDMENT_COMMIT
    base.COORDINATION_COMMIT = ACCEPTED_COMPONENT_HEAD
    base.OUTPUT_AUTHORITY_ROOT = AUTHORITY_ROOT
    base.NONCE = NONCE
    base.OUTPUT_DIR = CAPTURE_ROOT
    base.EXECUTION_RECEIPT = EXECUTION_RECEIPT
    base.LOCK_LABEL = LOCK_LABEL
    base.build_lock_command = build_lock_command


def git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args], check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise base.GateError(f"GIT:{' '.join(args)}:{result.stderr}")
    return result.stdout.strip()


def fixed_tree_gate(approved_head: str) -> dict[str, Any]:
    if git_text("rev-parse", "HEAD") != approved_head:
        raise base.GateError("APPROVED_HEAD_DRIFT")
    if git_text("status", "--porcelain"):
        raise base.GateError("WORKTREE_NOT_CLEAN")
    for commit in (AMENDMENT_COMMIT, ACCEPTED_COMPONENT_HEAD):
        if subprocess.run(
            ["git", "-C", str(REPO), "merge-base", "--is-ancestor", commit, approved_head],
            check=False,
        ).returncode:
            raise base.GateError(f"ANCESTRY:{commit}")
    production_diff = subprocess.run(
        [
            "git", "-C", str(REPO), "diff", "--quiet",
            f"{ACCEPTED_COMPONENT_HEAD}..{approved_head}", "--", "src/gpuwrf",
        ],
        check=False,
    )
    if production_diff.returncode != 0:
        raise base.GateError("PRODUCTION_TREE_CHANGED_AFTER_ACCEPTANCE")
    return {
        "approved_head": approved_head,
        "approved_tree": git_text("rev-parse", f"{approved_head}^{{tree}}"),
        "accepted_component_head": ACCEPTED_COMPONENT_HEAD,
        "production_tree_unchanged_since_acceptance": True,
        "fixed_coupler": base.require_file(FIXED_COUPLER, FIXED_COUPLER_SHA256),
    }


def reservation_value(approved_head: str) -> dict[str, Any]:
    return {
        "schema": "wrfgpu2-v0234-gpt-postfix-sp2-authority-reservation-v1",
        "nonce": NONCE,
        "nonce_prefix": PREFIX,
        "single_use": True,
        "authority_root": str(AUTHORITY_ROOT),
        "capture_root": str(CAPTURE_ROOT),
        "approved_head": approved_head,
        "accepted_component_head": ACCEPTED_COMPONENT_HEAD,
        "amendment": base.require_file(AMENDMENT, AMENDMENT_SHA256),
        "consumed_prefixes_forbidden": list(CONSUMED_PREFIXES),
        "gpu_arm_limit": 1,
        "retry_permitted": False,
        "gpu_query_permitted": False,
    }


def reserve_authority(approved_head: str, root: Path = AUTHORITY_ROOT) -> dict[str, Any]:
    fixed_tree_gate(approved_head)
    if root != AUTHORITY_ROOT:
        raise base.GateError("AUTHORITY_ROOT_OVERRIDE")
    if root.exists() or root.is_symlink():
        raise base.GateError("AUTHORITY_ROOT_NOT_FRESH")
    root.mkdir(mode=0o755)
    for relative in ("capture", "cpu-adapter", "cpu-reference", "comparator", "proofs", "control"):
        (root / relative).mkdir(mode=0o755)
    value = reservation_value(approved_head)
    value["canonical_payload_sha256"] = base.canonical_without_self(value)
    base.atomic_json(RESERVATION, value, replace=False)
    return value


def load_reservation() -> dict[str, Any]:
    value, record = base.load_canonical_json(RESERVATION)
    expected = reservation_value(value.get("approved_head", ""))
    if {
        key: item for key, item in value.items() if key != "canonical_payload_sha256"
    } != expected:
        raise base.GateError("RESERVATION_DRIFT")
    return {"value": value, "record": record}


def load_execution_receipt() -> dict[str, Any]:
    value, record = base.load_canonical_json(EXECUTION_RECEIPT)
    command = value.get("lock_command")
    if not (
        value.get("schema")
        == "wrfgpu2-v0234-single-capture-execution-receipt-v1"
        and value.get("nonce") == NONCE
        and value.get("nonce_consumed") is True
        and value.get("lock_invocation_count") == 1
        and value.get("returncode") == 0
        and value.get("payload_or_wrapper_returncode") == 0
        and value.get("retry_performed") is False
        and value.get("alternate_arm_performed") is False
        and value.get("gpu_query_performed") is False
        and isinstance(command, list)
        and command == build_lock_command(value.get("approved_head", ""))
        and value.get("lock_release", {}).get("acquired") is True
        and value.get("lock_release", {}).get("release_line_count") == 1
        and value.get("lock_release", {}).get("own_holder_absent") is True
    ):
        raise base.GateError("EXECUTION_RECEIPT_STATUS")
    return {"value": value, "record": record}


def profile_preflight(approved_head: str) -> dict[str, Any]:
    base.assert_no_backend_imported()
    fixed = fixed_tree_gate(approved_head)
    reservation = load_reservation()
    engine = base.static_preflight(approved_head, require_output_absent=True)
    result = {
        **engine,
        "schema": "wrfgpu2-v0234-gpt-postfix-sp2-capture-preflight-v1",
        "fixed_tree": fixed,
        "profile": base.require_file(Path(__file__).resolve(), base.sha256_file(Path(__file__).resolve())),
        "base_harness": base.require_file(BASE_SCRIPT, BASE_SHA256),
        "authority_reservation": reservation["record"],
        "nonce": NONCE,
        "one_arm": True,
        "retry_permitted": False,
        "gpu_query_permitted": False,
        "pbl_invocations_permitted": 0,
        "passed": True,
    }
    return result


def build_lock_command(approved_head: str) -> list[str]:
    return [
        str(base.LOCK_WRAPPER),
        "--timeout", "0",
        "--label", LOCK_LABEL,
        "--intent", "production-preemptible",
        "--",
        "/usr/bin/taskset", "-c", "13,14,15,29,30,31",
        "/usr/bin/nice", "-n", "15",
        "/usr/bin/ionice", "-c", "3",
        str(base.PYTHON_EXECUTABLE), str(Path(__file__).resolve()),
        "--execute", "--approved-head", approved_head,
    ]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    modes = value.add_mutually_exclusive_group(required=True)
    modes.add_argument("--reserve", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--launch", action="store_true")
    modes.add_argument("--validate-output", action="store_true")
    value.add_argument("--approved-head")
    value.add_argument("--proof", type=Path)
    value.add_argument("--preflight-path", type=Path)
    value.add_argument("--preflight-sha256")
    value.add_argument("--receipt", type=Path)
    return value


def main(argv: list[str] | None = None) -> int:
    configure_base()
    args = parser().parse_args(argv)
    try:
        if not args.approved_head:
            raise base.GateError("APPROVED_HEAD_REQUIRED")
        if args.reserve:
            result = reserve_authority(args.approved_head)
            if args.proof is not None:
                base.atomic_json(args.proof.resolve(), result, replace=False)
        elif args.preflight:
            if args.proof is None:
                raise base.GateError("PREFLIGHT_PROOF_REQUIRED")
            result = profile_preflight(args.approved_head)
            result["canonical_payload_sha256"] = base.canonical_without_self(result)
            base.atomic_json(args.proof.resolve(), result, replace=False)
        elif args.execute:
            fixed_tree_gate(args.approved_head)
            return base.execute_payload(args.approved_head)
        elif args.launch:
            if not all((args.preflight_path, args.preflight_sha256, args.receipt)):
                raise base.GateError("LAUNCH_ARGS")
            if args.receipt.resolve() != EXECUTION_RECEIPT.resolve():
                raise base.GateError("EXECUTION_RECEIPT_SCOPE")
            fixed_tree_gate(args.approved_head)
            return base.launch_once(
                args.approved_head,
                args.preflight_path.resolve(),
                args.preflight_sha256,
                args.receipt.resolve(),
            )
        else:
            if args.proof is None:
                raise base.GateError("VALIDATION_PROOF_REQUIRED")
            receipt = load_execution_receipt()
            result = base.validate_snapshot_manifest(
                CAPTURE_ROOT / "manifest.json", expected_nonce=NONCE
            )
            manifest, manifest_record = base.load_canonical_json(
                CAPTURE_ROOT / "manifest.json"
            )
            if not (
                manifest.get("status") == "CAPTURE_COMPLETE_HOLD_5_OF_28"
                and manifest.get("reference_status") == "HOLD_5_OF_28"
                and manifest.get("seam", {}).get("trace_invocation_count") == 1
                and manifest.get("seam", {}).get("pbl_guard_invocation_count") == 0
                and manifest.get("provenance", {}).get("synthetic_reconstruction")
                is False
                and manifest.get("tuple", {}).get("producer_commit")
                == receipt["value"].get("approved_head")
            ):
                raise base.GateError("CAPTURE_MANIFEST_STATUS")
            result.update({
                "schema": "wrfgpu2-v0234-gpt-postfix-sp2-capture-validation-v1",
                "approved_head": args.approved_head,
                "fixed_tree": fixed_tree_gate(args.approved_head),
                "nonce": NONCE,
                "execution_receipt": receipt["record"],
                "execution_receipt_canonical_sha256": receipt["value"][
                    "canonical_payload_sha256"
                ],
                "capture_manifest": manifest_record,
                "output_tree": base.tree_inventory(CAPTURE_ROOT),
                "gpu_arm_count": 1,
                "retry_performed": False,
                "gpu_query_performed": False,
            })
            result["canonical_payload_sha256"] = base.canonical_without_self(result)
            base.atomic_json(args.proof.resolve(), result, replace=False)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (base.CapturePreempted, base.GateError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"REFUSE:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 130 if isinstance(exc, base.CapturePreempted) else 75


configure_base()


if __name__ == "__main__":
    raise SystemExit(main())
