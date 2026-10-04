#!/usr/bin/env python3
"""Capture one fixed MYNN adapter invocation from the fresh GPU seam state."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-19-v0234-gpt-single-authority-attribution"
NONCE = "a73aef522fbdca9c88ec1c50464db6e51819f47aa00cc9493a1fe2616cd2ebfe"
PREFIX = NONCE[:16]
AUTHORITY_ROOT = Path(f"<DATA_ROOT>/wrf_gpu2/v0234_gpt_postfix_sp2_{PREFIX}")
GPU_CAPTURE_ROOT = AUTHORITY_ROOT / f"capture/authentic-{PREFIX}"
CPU_PARENT = AUTHORITY_ROOT / "cpu-adapter"
CPU_NAMESPACE = CPU_PARENT / "single-authority-v1"
FIXED_COMPONENT_HEAD = "672f55c0e6ac5ede82c288cdcaed63fd71a5e9a6"
FIXED_COUPLER_SHA256 = "694e51f8e9df30d7c5abd3729c8eb756b6dbfd45c83bf774f95a23747b0b2283"

BASE_SCRIPT = REPO / "scripts/v0234_gpt_single_authority_capture.py"
# ``16927ecd`` is the accepted post-SP2 recorder revision that added the
# authentic WRF hybrid metrics consumed by the subsequent correctness chain.
# Pin its bytes rather than the superseded pre-metric wrapper revision.
BASE_SCRIPT_ACCEPTED_COMMIT = "16927ecdc215f57d0ec40e10dc9c0cfaf5c42d8e"
BASE_SHA256 = "3298f2961c2730b446870231d1fa080f16ee0c091f3aa27ac0744537521f80f9"
SPEC = importlib.util.spec_from_file_location("v0234_postfix_cpu_capture_base", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("CPU capture base import unavailable")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


class ProfileFailure(RuntimeError):
    pass


def canonical_without_self(value: dict[str, Any]) -> str:
    return base.canonical_without_self(value)


def load_canonical(path: Path, expected_sha: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or base.sha256_file(path) != expected_sha:
        raise ProfileFailure(f"PROOF_FILE:{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if canonical_without_self(value) != value.get("canonical_payload_sha256"):
        raise ProfileFailure(f"PROOF_CANONICAL:{path}")
    return value


def git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args], check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise ProfileFailure(f"GIT:{' '.join(args)}:{result.stderr}")
    return result.stdout.strip()


def fixed_tree_gate(approved_head: str) -> dict[str, Any]:
    if git_text("rev-parse", "HEAD") != approved_head or git_text("status", "--porcelain"):
        raise ProfileFailure("HEAD_OR_WORKTREE")
    if subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", FIXED_COMPONENT_HEAD, approved_head],
        check=False,
    ).returncode:
        raise ProfileFailure("FIXED_HEAD_NOT_ANCESTOR")
    if subprocess.run(
        ["git", "-C", str(REPO), "diff", "--quiet", f"{FIXED_COMPONENT_HEAD}..{approved_head}", "--", "src/gpuwrf"],
        check=False,
    ).returncode:
        raise ProfileFailure("PRODUCTION_SOURCE_DRIFT")
    coupler = REPO / "src/gpuwrf/coupling/physics_couplers.py"
    if base.sha256_file(coupler) != FIXED_COUPLER_SHA256:
        raise ProfileFailure("FIXED_COUPLER_DRIFT")
    return {
        "approved_head": approved_head,
        "accepted_component_head": FIXED_COMPONENT_HEAD,
        "production_source_unchanged": True,
        "coupler_sha256": FIXED_COUPLER_SHA256,
    }


def gpu_authority(validation_path: Path, validation_sha: str) -> dict[str, Any]:
    validation = load_canonical(validation_path, validation_sha)
    if not (
        validation.get("schema")
        == "wrfgpu2-v0234-gpt-postfix-sp2-capture-validation-v1"
        and validation.get("valid") is True
        and validation.get("nonce") == NONCE
        and validation.get("gpu_arm_count") == 1
        and validation.get("retry_performed") is False
        and validation.get("gpu_query_performed") is False
        and Path(validation.get("manifest", "")) == GPU_CAPTURE_ROOT / "manifest.json"
    ):
        raise ProfileFailure("GPU_VALIDATION_STATUS")
    manifest_sha = validation.get("manifest_file_sha256")
    if not isinstance(manifest_sha, str) or len(manifest_sha) != 64:
        raise ProfileFailure("GPU_MANIFEST_HASH")
    manifest = load_canonical(GPU_CAPTURE_ROOT / "manifest.json", manifest_sha)
    if not (
        manifest.get("nonce") == NONCE
        and manifest.get("status") == "CAPTURE_COMPLETE_HOLD_5_OF_28"
        and manifest.get("seam", {}).get("trace_invocation_count") == 1
        and manifest.get("seam", {}).get("pbl_guard_invocation_count") == 0
        and manifest.get("provenance", {}).get("synthetic_reconstruction") is False
    ):
        raise ProfileFailure("GPU_CAPTURE_MANIFEST_STATUS")
    return {
        "validation_path": str(validation_path),
        "validation_sha256": validation_sha,
        "validation_canonical_sha256": validation["canonical_payload_sha256"],
        "manifest_path": str(GPU_CAPTURE_ROOT / "manifest.json"),
        "manifest_sha256": manifest_sha,
        "manifest_canonical_sha256": manifest["canonical_payload_sha256"],
        "state_leaf_count": len(manifest.get("state_leaves", [])),
        "seam": manifest["seam"],
    }


def configure_base(authority: dict[str, Any]) -> None:
    base.SPRINT = SPRINT
    base.CAPTURE_ROOT = GPU_CAPTURE_ROOT
    base.CAPTURE_MANIFEST = GPU_CAPTURE_ROOT / "manifest.json"
    base.CAPTURE_MANIFEST_SHA256 = authority["manifest_sha256"]
    base.NAMESPACE_PARENT = CPU_PARENT


def preflight(
    approved_head: str,
    proof_output: Path,
    validation_path: Path,
    validation_sha: str,
) -> dict[str, Any]:
    authority = gpu_authority(validation_path, validation_sha)
    configure_base(authority)
    result = base.static_preflight(CPU_NAMESPACE, proof_output)
    result.update({
        "schema": "wrfgpu2-v0234-gpt-postfix-cpu-adapter-preflight-v1",
        "fixed_tree": fixed_tree_gate(approved_head),
        "gpu_capture_authority": authority,
        "profile": {
            "path": str(Path(__file__).resolve()),
            "sha256": base.sha256_file(Path(__file__).resolve()),
        },
        "base_harness": {"path": str(BASE_SCRIPT), "sha256": BASE_SHA256},
        "restart": False,
        "production_adapter_invocations_planned": 1,
        "passed": True,
    })
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    modes = value.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--capture", action="store_true")
    value.add_argument("--approved-head", required=True)
    value.add_argument("--gpu-validation", type=Path, required=True)
    value.add_argument("--gpu-validation-sha256", required=True)
    value.add_argument("--proof-output", type=Path, required=True)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = args.proof_output.resolve()
    validation = args.gpu_validation.resolve()
    try:
        if args.preflight:
            result = preflight(
                args.approved_head, output, validation, args.gpu_validation_sha256
            )
            base.atomic_json(output, result)
        else:
            authority = gpu_authority(validation, args.gpu_validation_sha256)
            configure_base(authority)
            fixed_tree_gate(args.approved_head)
            result = base.execute_capture(CPU_NAMESPACE, output)
        print(json.dumps({
            "passed": result.get("passed"),
            "schema": result.get("schema"),
            "namespace": str(CPU_NAMESPACE),
            "proof": str(output),
        }, sort_keys=True))
        return 0
    except (
        ProfileFailure, base.CaptureFailure, OSError, ValueError, TypeError,
        KeyError, json.JSONDecodeError, subprocess.SubprocessError,
    ) as exc:
        print(f"REFUSE:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
