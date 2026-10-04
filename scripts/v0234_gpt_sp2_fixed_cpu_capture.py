#!/usr/bin/env python3
"""Fresh one-shot CPU MYNN capture after the v0234 SP2 surface fixes.

The base harness loads the authenticated pre-PBL seam and invokes the production
MYNN adapter exactly once.  This profile replaces only the four State handles
sealed by ``surface-provenance-proof.json`` before that invocation, while keeping
the authentic atmosphere, heat/moisture fluxes, corrected WRF metrics, and every
other operand unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-19-v0234-gpt-sp2-residual-closure"
STAGED = Path("/tmp/v0234_gpt_sp2_residual_evidence")
CAPTURE_ROOT = STAGED / "seam-capture"
INPUT_DIR = STAGED / "inputs"
NAMESPACE_PARENT = STAGED / "surface-fixed-cpu-adapter"
SURFACE_PROOF = SPRINT / "surface-provenance-proof.json"
CORRECTION = STAGED / "surface-correction/source-authorized-v1.npz"

CAPTURE_MANIFEST_SHA256 = (
    "ae584d25d0e8fc56afedc4a1745f9109d52fecf76635007324180f40bb709873"
)
CAPTURE_MANIFEST_CANONICAL = (
    "f6d14eee33d1f43364238c9a42849cb0993aaa71fa9b660c397d0c3ee7096d6a"
)
SURFACE_PROOF_SHA256 = (
    "2b579b440043586fa33e6137eb10f58c71789967787ba34bce2e19cc79a5e979"
)
SURFACE_PROOF_CANONICAL = (
    "26f6efc05967abf7b350a7746579f187fea06a44da9cc7d7f47676af4a8b3819"
)
CORRECTION_SHA256 = (
    "fc738ea01d76210a786e01c8de4af1e01010b1d2f13280b7b12ca3029d565494"
)
CORRECTION_FIELDS = ("ustar", "tau_u", "tau_v", "rhosfc")
FIX_SOURCES = (
    "src/gpuwrf/io/land_state.py",
    "src/gpuwrf/physics/noah_mp.py",
    "src/gpuwrf/physics/noahmp_coupler.py",
    "src/gpuwrf/coupling/noahmp_surface_hook.py",
    "scripts/v0234_gpt_sp2_surface_provenance.py",
    "scripts/v0234_gpt_sp2_fixed_cpu_capture.py",
)

BASE_SCRIPT = REPO / "scripts/v0234_gpt_single_authority_capture.py"
SPEC = importlib.util.spec_from_file_location(
    "v0234_sp2_fixed_capture_base", BASE_SCRIPT,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("capture base import unavailable")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

CAPTURE_MODE = False
TRANSFORM_COUNT = {"value": 0}


class ProfileFailure(RuntimeError):
    """Fail-closed profile or correction-chain error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_without_self(value: Mapping[str, Any]) -> str:
    payload = {
        key: item for key, item in value.items()
        if key != "canonical_payload_sha256"
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args], check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise ProfileFailure(f"GIT:{' '.join(args)}:{result.stderr.strip()}")
    return result.stdout.strip()


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
        check=True, stdout=subprocess.PIPE,
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


def validate_correction_chain() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if (
        SURFACE_PROOF.is_symlink()
        or not SURFACE_PROOF.is_file()
        or sha256_file(SURFACE_PROOF) != SURFACE_PROOF_SHA256
    ):
        raise ProfileFailure("SURFACE_PROOF_HASH")
    proof = json.loads(SURFACE_PROOF.read_text(encoding="utf-8"))
    if not (
        canonical_without_self(proof) == SURFACE_PROOF_CANONICAL
        and proof.get("canonical_payload_sha256") == SURFACE_PROOF_CANONICAL
        and proof.get("passed") is True
        and proof.get("verdict")
        == "WRF_SEASONAL_SURFACE_AND_MYNN_DENSITY_FIX_PROVEN"
        and proof.get("production_pbl_adapter_invocations") == 0
        and proof.get("correction_contract", {}).get("changed_state_handles")
        == list(CORRECTION_FIELDS)
        and proof.get("correction_contract", {}).get("synthetic_reconstruction")
        is False
        and proof.get("correction_contract", {}).get("empirical_tuning") is False
        and proof.get("correction_artifact", {}).get("sha256")
        == CORRECTION_SHA256
    ):
        raise ProfileFailure("SURFACE_PROOF_AUTHORITY")
    if (
        CORRECTION.is_symlink()
        or not CORRECTION.is_file()
        or sha256_file(CORRECTION) != CORRECTION_SHA256
    ):
        raise ProfileFailure("CORRECTION_HASH")
    arrays: dict[str, np.ndarray] = {}
    with np.load(CORRECTION, allow_pickle=False) as archive:
        if set(archive.files) != set(CORRECTION_FIELDS):
            raise ProfileFailure(f"CORRECTION_SCHEMA:{archive.files}")
        for name in CORRECTION_FIELDS:
            value = np.asarray(archive[name])
            declared = proof["correction_artifact"]["arrays"][name]
            if not (
                value.shape == (base.NY, base.NX)
                and value.dtype.str == declared["dtype"]
                and base.sha256_bytes(np.ascontiguousarray(value).tobytes())
                == declared["logical_c_bitpayload_sha256"]
                and np.isfinite(value).all()
            ):
                raise ProfileFailure(f"CORRECTION_ARRAY:{name}")
            arrays[name] = np.array(value, copy=True)
    return proof, arrays


def install_state_transform(
    arrays: Mapping[str, np.ndarray], seam_manifest: Mapping[str, Any],
) -> None:
    import jax
    from gpuwrf.contracts.state import State

    old_records = {
        item["slot_name"]: item
        for item in seam_manifest["state_leaves"]
        if item.get("slot_name") in CORRECTION_FIELDS
    }
    if set(old_records) != set(CORRECTION_FIELDS):
        raise ProfileFailure("SEAM_CORRECTION_FIELDS")
    original = State.tree_unflatten.__func__

    def transformed(cls: Any, aux: Any, children: Any) -> Any:
        state = original(cls, aux, children)
        if TRANSFORM_COUNT["value"]:
            return state
        for name in CORRECTION_FIELDS:
            old = np.asarray(getattr(state, name))
            record = old_records[name]
            if not (
                list(old.shape) == record["shape"]
                and old.dtype.str == record["dtype"]
                and base.sha256_bytes(np.ascontiguousarray(old).tobytes())
                == record["logical_c_bitpayload_sha256"]
            ):
                raise ProfileFailure(f"TRANSFORM_TARGET:{name}")
        TRANSFORM_COUNT["value"] += 1
        return state.replace(**{
            name: jax.device_put(arrays[name]).astype(getattr(state, name).dtype)
            for name in CORRECTION_FIELDS
        })

    State.tree_unflatten = classmethod(transformed)


def configure() -> dict[str, Any]:
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
    return manifest


def install_profile_preflight(
    seam_manifest: Mapping[str, Any], proof: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    original = base.static_preflight

    def profiled(namespace: Path, proof_output: Path) -> dict[str, Any]:
        result = original(namespace, proof_output)
        result["schema"] = "wrfgpu2-v0234-gpt-sp2-surface-fixed-cpu-preflight-v1"
        result["profile"] = profile_record()
        result["grid_metric_authority"] = {
            "source": str(INPUT_DIR / "wrfinput_d03"),
            "loader": "gpuwrf.dynamics.metrics.load_wrfinput_metrics",
            "matches_operational_real_case_loader": True,
            "analytic_flat_fallback_forbidden": True,
        }
        result["source_authorized_surface_correction"] = {
            "proof_path": str(SURFACE_PROOF),
            "proof_sha256": SURFACE_PROOF_SHA256,
            "proof_canonical_payload_sha256": SURFACE_PROOF_CANONICAL,
            "proof_head": proof["git"]["head"],
            "artifact_path": str(CORRECTION),
            "artifact_sha256": CORRECTION_SHA256,
            "changed_state_handles": list(CORRECTION_FIELDS),
            "unchanged_state_handles": ["theta_flux", "qv_flux", "fltv"],
            "synthetic_reconstruction": False,
            "empirical_tuning": False,
            "model_sources": {
                relative: base.tracked_source_record(relative)
                for relative in FIX_SOURCES
            },
        }
        result["tracked_worktree_clean"] = True
        result["untracked_files_out_of_authority"] = True
        if CAPTURE_MODE:
            install_state_transform(arrays, seam_manifest)
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
    global CAPTURE_MODE
    args = parser().parse_args(argv)
    try:
        CAPTURE_MODE = bool(args.capture)
        seam_manifest = configure()
        tracked_head_gate(args.approved_head)
        proof, arrays = validate_correction_chain()
        install_profile_preflight(seam_manifest, proof, arrays)
        namespace = (NAMESPACE_PARENT / args.namespace).resolve()
        proof_output = args.proof_output.resolve()
        if args.preflight:
            result = base.static_preflight(namespace, proof_output)
            if TRANSFORM_COUNT["value"] != 0:
                raise ProfileFailure("PREFLIGHT_IMPORTED_BACKEND")
            base.atomic_json(proof_output, result)
        else:
            result = base.execute_capture(namespace, proof_output)
            if TRANSFORM_COUNT["value"] != 1:
                raise ProfileFailure(
                    f"SURFACE_TRANSFORM_COUNT:{TRANSFORM_COUNT['value']}"
                )
        print(json.dumps({
            "passed": result.get("passed"),
            "schema": result.get("schema"),
            "namespace": str(namespace),
            "proof": str(proof_output),
            "surface_transform_count": TRANSFORM_COUNT["value"],
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
