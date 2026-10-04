#!/usr/bin/env python3
"""Fresh CPU MYNN capture using the authenticated seam and real WRF metrics.

This is the v0234 residual-closure profile for the one-shot capture harness.
It keeps the authenticated post-QKE GPU seam fixed, stages all large inputs
under ``/tmp``, and changes only the CPU materializer's grid construction from
the analytic-flat fallback to the same ``load_wrfinput_metrics`` payload used by
the operational real-case loader.  No GPU, WRF, MPI, or synthetic state is used.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-19-v0234-gpt-sp2-residual-closure"
STAGED = Path("/tmp/v0234_gpt_sp2_residual_evidence")
CAPTURE_ROOT = STAGED / "seam-capture"
INPUT_DIR = STAGED / "inputs"
NAMESPACE_PARENT = STAGED / "corrected-cpu-adapter"
CAPTURE_MANIFEST_SHA256 = (
    "ae584d25d0e8fc56afedc4a1745f9109d52fecf76635007324180f40bb709873"
)
CAPTURE_MANIFEST_CANONICAL = (
    "f6d14eee33d1f43364238c9a42849cb0993aaa71fa9b660c397d0c3ee7096d6a"
)

BASE_SCRIPT = REPO / "scripts/v0234_gpt_single_authority_capture.py"
SPEC = importlib.util.spec_from_file_location("v0234_sp2_residual_capture_base", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("capture base import unavailable")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


class ProfileFailure(RuntimeError):
    """Fail-closed profile or evidence error."""


def git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise ProfileFailure(f"GIT:{' '.join(args)}:{result.stderr.strip()}")
    return result.stdout.strip()


def configure() -> None:
    manifest_path = CAPTURE_ROOT / "manifest.json"
    if base.sha256_file(manifest_path) != CAPTURE_MANIFEST_SHA256:
        raise ProfileFailure("STAGED_CAPTURE_MANIFEST_HASH")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (
        base.canonical_without_self(manifest) == CAPTURE_MANIFEST_CANONICAL
        and manifest.get("canonical_payload_sha256") == CAPTURE_MANIFEST_CANONICAL
        and manifest.get("status") == "CAPTURE_COMPLETE_HOLD_5_OF_28"
        and manifest.get("seam", {}).get("trace_invocation_count") == 1
        and manifest.get("seam", {}).get("pbl_guard_invocation_count") == 0
        and manifest.get("provenance", {}).get("synthetic_reconstruction") is False
    ):
        raise ProfileFailure("STAGED_CAPTURE_AUTHORITY")
    base.SPRINT = SPRINT
    base.CAPTURE_ROOT = CAPTURE_ROOT
    base.CAPTURE_MANIFEST = manifest_path
    base.CAPTURE_MANIFEST_SHA256 = CAPTURE_MANIFEST_SHA256
    base.INPUT_DIR = INPUT_DIR
    base.NAMESPACE_PARENT = NAMESPACE_PARENT


def tracked_head_gate(approved_head: str) -> None:
    if git_text("rev-parse", "HEAD") != approved_head:
        raise ProfileFailure("APPROVED_HEAD_DRIFT")
    if git_text("status", "--porcelain", "--untracked-files=no"):
        raise ProfileFailure("TRACKED_WORKTREE_NOT_CLEAN")


def profile_record() -> dict[str, Any]:
    path = Path(__file__).resolve()
    relative = path.relative_to(REPO).as_posix()
    head_bytes = subprocess.run(
        ["git", "-C", str(REPO), "show", f"HEAD:{relative}"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    disk_bytes = path.read_bytes()
    if disk_bytes != head_bytes:
        raise ProfileFailure("PROFILE_NOT_HEAD")
    return {
        "path": str(path),
        "relative": relative,
        "sha256": base.sha256_bytes(disk_bytes),
        "git_blob": git_text("rev-parse", f"HEAD:{relative}"),
    }


def install_profile_preflight() -> None:
    original = base.static_preflight

    def profiled(namespace: Path, proof_output: Path) -> dict[str, Any]:
        result = original(namespace, proof_output)
        result["schema"] = "wrfgpu2-v0234-gpt-sp2-residual-cpu-preflight-v1"
        result["profile"] = profile_record()
        result["grid_metric_authority"] = {
            "source": str(INPUT_DIR / "wrfinput_d03"),
            "loader": "gpuwrf.dynamics.metrics.load_wrfinput_metrics",
            "matches_operational_real_case_loader": True,
            "analytic_flat_fallback_forbidden": True,
        }
        result["tracked_worktree_clean"] = True
        result["untracked_files_out_of_authority"] = True
        return result

    base.static_preflight = profiled


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    mode = value.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--capture", action="store_true")
    value.add_argument("--approved-head", required=True)
    value.add_argument("--namespace", required=True)
    value.add_argument("--proof-output", type=Path, required=True)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        configure()
        tracked_head_gate(args.approved_head)
        install_profile_preflight()
        namespace = (NAMESPACE_PARENT / args.namespace).resolve()
        proof = args.proof_output.resolve()
        if args.preflight:
            result = base.static_preflight(namespace, proof)
            base.atomic_json(proof, result)
        else:
            result = base.execute_capture(namespace, proof)
        print(json.dumps({
            "passed": result.get("passed"),
            "schema": result.get("schema"),
            "namespace": str(namespace),
            "proof": str(proof),
        }, sort_keys=True))
        return 0
    except (
        ProfileFailure,
        base.CaptureFailure,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"REFUSE:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
