#!/usr/bin/env python3
"""One-shot CPU MYNN capture for the source-proven Noah-MP QAIR correction.

The base harness loads the authenticated pre-PBL seam and calls the production
MYNN adapter exactly once.  This profile first reapplies the already accepted
seasonal-surface correction, then adds the paired Noah-MP QAIR surface deltas
sealed by ``tsk-provenance-proof.json``.  Every other State leaf remains the
authenticated seam value.
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
SPRINT = REPO / ".agent/sprints/2026-07-20-v0234-gpt-land-tsk-provenance"
STAGED = Path("/tmp/v0234_gpt_sp2_residual_evidence")
SCRATCH = Path("/tmp/v0234_land_tsk_scratch")
CAPTURE_ROOT = STAGED / "seam-capture"
INPUT_DIR = STAGED / "inputs"
NAMESPACE_PARENT = SCRATCH / "qml-fixed-cpu-adapter"
PROVENANCE_PROOF = SPRINT / "tsk-provenance-proof.json"
DELTA = SCRATCH / "qml-source-delta-v1.npz"

CAPTURE_MANIFEST_SHA256 = (
    "ae584d25d0e8fc56afedc4a1745f9109d52fecf76635007324180f40bb709873"
)
CAPTURE_MANIFEST_CANONICAL = (
    "f6d14eee33d1f43364238c9a42849cb0993aaa71fa9b660c397d0c3ee7096d6a"
)
PROVENANCE_PROOF_SHA256 = (
    "140905f7b8998ab4b5a39fa78e2cfc1213e18baccc21479aabad052e2cda26c0"
)
PROVENANCE_PROOF_CANONICAL = (
    "c75c4b7ce65b64bd5fc5b2da431fce4de720fe7f6e02a407fe31ce8492cbc5a2"
)
PROVENANCE_HEAD = "592d25b2b76e367af14f24a43a1bf7a8724160db"
DELTA_SHA256 = "85dce69d3a9d6a33d699d852ad4779de53a322d96dcd738f7e9a128ae6d8a0a0"
DELTA_FIELDS = ("theta_flux", "qv_flux", "fltv", "t_skin")
FIX_SOURCES = (
    "src/gpuwrf/physics/noahmp_coupler.py",
    "src/gpuwrf/physics/noahmp/types.py",
    "src/gpuwrf/coupling/noahmp_surface_hook.py",
    "scripts/v0234_gpt_land_tsk_provenance.py",
    "scripts/v0234_gpt_land_tsk_cpu_capture.py",
    "scripts/v0234_gpt_sp2_fixed_cpu_capture.py",
)

SURFACE_PROFILE_PATH = REPO / "scripts/v0234_gpt_sp2_fixed_cpu_capture.py"
SPEC = importlib.util.spec_from_file_location(
    "v0234_land_tsk_prior_surface_profile", SURFACE_PROFILE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("prior surface profile import unavailable")
surface_profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(surface_profile)
base = surface_profile.base

CAPTURE_MODE = False
TRANSFORM_COUNT = {"value": 0}


class ProfileFailure(RuntimeError):
    """Fail-closed profile or correction-chain failure."""


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


def validate_provenance_chain() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if (
        PROVENANCE_PROOF.is_symlink()
        or not PROVENANCE_PROOF.is_file()
        or sha256_file(PROVENANCE_PROOF) != PROVENANCE_PROOF_SHA256
    ):
        raise ProfileFailure("PROVENANCE_PROOF_HASH")
    proof = json.loads(PROVENANCE_PROOF.read_text(encoding="utf-8"))
    contract = proof.get("delta_contract", {})
    if not (
        canonical_without_self(proof) == PROVENANCE_PROOF_CANONICAL
        and proof.get("canonical_payload_sha256") == PROVENANCE_PROOF_CANONICAL
        and proof.get("passed") is True
        and proof.get("verdict")
        == "WRF_QML_SPECIFIC_HUMIDITY_TSK_CANDIDATE_PROVEN"
        and proof.get("git", {}).get("head") == PROVENANCE_HEAD
        and proof.get("production_pbl_adapter_invocations") == 0
        and proof.get("wrf_or_mpi_executions") == 0
        and proof.get("gpu_actions") == 0
        and proof.get("delta_artifact", {}).get("sha256") == DELTA_SHA256
        and contract.get("fields") == list(DELTA_FIELDS)
        and contract.get("paired_ab_same_inputs_except_qair") is True
        and contract.get("water_delta_bitwise_zero") is True
        and contract.get("empirical_tuning") is False
        and contract.get("synthetic_wrf_reference") is False
        and contract.get("mask_selected_correction") is False
        and proof.get("gates", {}).get(
            "counterfactual_frozen_tsk_land_rms_and_max_strictly_improve"
        ) is True
    ):
        raise ProfileFailure("PROVENANCE_PROOF_AUTHORITY")
    if DELTA.is_symlink() or not DELTA.is_file() or sha256_file(DELTA) != DELTA_SHA256:
        raise ProfileFailure("DELTA_HASH")
    arrays: dict[str, np.ndarray] = {}
    with np.load(DELTA, allow_pickle=False) as archive:
        expected_names = {f"{name}_delta" for name in DELTA_FIELDS}
        if set(archive.files) != expected_names:
            raise ProfileFailure(f"DELTA_SCHEMA:{archive.files}")
        for field in DELTA_FIELDS:
            name = f"{field}_delta"
            value = np.asarray(archive[name])
            declared = proof["delta_artifact"]["arrays"][name]
            if not (
                value.shape == (base.NY, base.NX)
                and value.dtype.str == declared["dtype"]
                and base.sha256_bytes(np.ascontiguousarray(value).tobytes())
                == declared["logical_c_bitpayload_sha256"]
                and np.isfinite(value).all()
            ):
                raise ProfileFailure(f"DELTA_ARRAY:{name}")
            arrays[field] = np.array(value, copy=True)
    return proof, arrays


def install_state_transform(
    surface_arrays: Mapping[str, np.ndarray],
    deltas: Mapping[str, np.ndarray],
    seam_manifest: Mapping[str, Any],
) -> None:
    import jax
    import jax.numpy as jnp
    from gpuwrf.contracts.state import State

    absolute_fields = tuple(surface_profile.CORRECTION_FIELDS)
    target_fields = tuple(dict.fromkeys((*absolute_fields, *DELTA_FIELDS)))
    old_records = {
        item["slot_name"]: item
        for item in seam_manifest["state_leaves"]
        if item.get("slot_name") in target_fields
    }
    if set(old_records) != set(target_fields):
        raise ProfileFailure("SEAM_TRANSFORM_FIELDS")
    original = State.tree_unflatten.__func__

    def transformed(cls: Any, aux: Any, children: Any) -> Any:
        state = original(cls, aux, children)
        if TRANSFORM_COUNT["value"]:
            return state
        for name in target_fields:
            old = np.asarray(getattr(state, name))
            record = old_records[name]
            if not (
                list(old.shape) == record["shape"]
                and old.dtype.str == record["dtype"]
                and base.sha256_bytes(np.ascontiguousarray(old).tobytes())
                == record["logical_c_bitpayload_sha256"]
            ):
                raise ProfileFailure(f"TRANSFORM_TARGET:{name}")
        updates = {
            name: jax.device_put(surface_arrays[name]).astype(getattr(state, name).dtype)
            for name in absolute_fields
        }
        updates.update({
            name: (
                jnp.asarray(getattr(state, name))
                + jax.device_put(deltas[name]).astype(getattr(state, name).dtype)
            )
            for name in DELTA_FIELDS
        })
        TRANSFORM_COUNT["value"] += 1
        return state.replace(**updates)

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
    seam_manifest: Mapping[str, Any],
    surface_proof: Mapping[str, Any],
    surface_arrays: Mapping[str, np.ndarray],
    provenance_proof: Mapping[str, Any],
    deltas: Mapping[str, np.ndarray],
) -> None:
    original = base.static_preflight

    def profiled(namespace: Path, proof_output: Path) -> dict[str, Any]:
        result = original(namespace, proof_output)
        result["schema"] = "wrfgpu2-v0234-gpt-land-tsk-cpu-preflight-v1"
        result["profile"] = profile_record()
        result["grid_metric_authority"] = {
            "source": str(INPUT_DIR / "wrfinput_d03"),
            "loader": "gpuwrf.dynamics.metrics.load_wrfinput_metrics",
            "matches_operational_real_case_loader": True,
            "analytic_flat_fallback_forbidden": True,
        }
        result["prior_surface_correction"] = {
            "proof_sha256": surface_profile.SURFACE_PROOF_SHA256,
            "artifact_sha256": surface_profile.CORRECTION_SHA256,
            "changed_state_handles": list(surface_profile.CORRECTION_FIELDS),
            "source_authorized": True,
            "empirical_tuning": False,
            "proof_verdict": surface_proof["verdict"],
        }
        result["qml_tsk_correction"] = {
            "proof_path": str(PROVENANCE_PROOF),
            "proof_sha256": PROVENANCE_PROOF_SHA256,
            "proof_canonical_payload_sha256": PROVENANCE_PROOF_CANONICAL,
            "proof_head": provenance_proof["git"]["head"],
            "artifact_path": str(DELTA),
            "artifact_sha256": DELTA_SHA256,
            "changed_state_handles": list(DELTA_FIELDS),
            "application": provenance_proof["delta_contract"]["application"],
            "water_delta_bitwise_zero": True,
            "synthetic_wrf_reference": False,
            "empirical_tuning": False,
            "model_sources": {
                relative: base.tracked_source_record(relative)
                for relative in FIX_SOURCES
            },
        }
        result["tracked_worktree_clean"] = True
        result["untracked_files_out_of_authority"] = True
        if CAPTURE_MODE:
            install_state_transform(surface_arrays, deltas, seam_manifest)
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
        surface_proof, surface_arrays = surface_profile.validate_correction_chain()
        provenance_proof, deltas = validate_provenance_chain()
        install_profile_preflight(
            seam_manifest, surface_proof, surface_arrays, provenance_proof, deltas,
        )
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
                raise ProfileFailure(f"QML_TRANSFORM_COUNT:{TRANSFORM_COUNT['value']}")
        print(json.dumps({
            "passed": result.get("passed"),
            "schema": result.get("schema"),
            "namespace": str(namespace),
            "proof": str(proof_output),
            "qml_transform_count": TRANSFORM_COUNT["value"],
        }, sort_keys=True))
        return 0
    except (
        ProfileFailure,
        surface_profile.ProfileFailure,
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
