#!/usr/bin/env python3
"""One-shot CPU MYNN capture for the final accepted QML + top-buffer path."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-20-v0234-gpt-land-tsk-provenance"
SCRATCH = Path("/tmp/v0234_land_tsk_scratch")
NAMESPACE_PARENT = SCRATCH / "qml-rrtmg-final-accepted-fixed-cpu-adapter"
BUFFER_PROOF = SPRINT / "rrtmg-top-buffer-tsk-proof.json"
BUFFER_DELTA = SCRATCH / "rrtmg-buffer-surface-delta-v1.npz"
BUFFER_PROOF_SHA256 = "af78f7a3eba095c2a69b688eb8c3f441b57975a3b500c8802f2f92cc0312917a"
BUFFER_PROOF_CANONICAL = "ec2eeef215b9faaa356a5e858212993e4959f45dee6677986464371600499879"
BUFFER_DELTA_SHA256 = "f50ac95bb2e22d5e52a12f4fa4634140dd6df0a273a7183b58f167291cb98d0c"
DELTA_FIELDS = ("theta_flux", "qv_flux", "fltv", "t_skin")
FIX_SOURCES = (
    "src/gpuwrf/physics/noahmp_coupler.py",
    "src/gpuwrf/physics/noahmp/types.py",
    "src/gpuwrf/physics/rrtmg_lw.py",
    "src/gpuwrf/physics/rrtmg_sw.py",
    "src/gpuwrf/physics/wrf_clwrf_ghg.py",
    "src/gpuwrf/coupling/noahmp_surface_hook.py",
    "src/gpuwrf/coupling/physics_couplers.py",
    "scripts/v0234_gpt_land_tsk_provenance.py",
    "scripts/v0234_gpt_land_tsk_rrtmg_buffer_proof.py",
    "scripts/v0234_gpt_land_tsk_rrtmg_tsk_proof.py",
    "scripts/v0234_gpt_land_tsk_rrtmg_interfaces.py",
    "scripts/v0234_gpt_land_tsk_rrtmg_interface_real_wrf_proof.py",
    "scripts/v0234_gpt_land_tsk_rrtmg_sw_real_wrf_proof.py",
    "scripts/v0234_gpt_land_tsk_rrtmg_composed_tsk_proof.py",
    "scripts/v0234_gpt_land_tsk_rrtmg_co2_year.py",
    "scripts/v0234_gpt_land_tsk_rrtmg_cam_real_wrf.py",
    "scripts/v0234_gpt_land_tsk_cpu_capture.py",
    "scripts/v0234_gpt_land_tsk_rrtmg_cpu_capture.py",
)


QML_PROFILE_PATH = REPO / "scripts/v0234_gpt_land_tsk_cpu_capture.py"
SPEC = importlib.util.spec_from_file_location("v0234_qml_capture_profile", QML_PROFILE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("QML capture profile unavailable")
qml_profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qml_profile)
base = qml_profile.base
surface_profile = qml_profile.surface_profile

CAPTURE_MODE = False
TRANSFORM_COUNT = {"value": 0}


class CaptureProfileFailure(RuntimeError):
    """Fail-closed combined-profile authority error."""


def git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise CaptureProfileFailure(f"GIT:{' '.join(args)}:{result.stderr.strip()}")
    return result.stdout.strip()


def profile_record() -> dict[str, Any]:
    path = Path(__file__).resolve()
    relative = path.relative_to(REPO).as_posix()
    disk = path.read_bytes()
    head = subprocess.run(
        ["git", "-C", str(REPO), "show", f"HEAD:{relative}"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    if disk != head:
        raise CaptureProfileFailure("PROFILE_NOT_HEAD")
    return {
        "path": str(path),
        "relative": relative,
        "sha256": base.sha256_bytes(disk),
        "git_blob": git_text("rev-parse", f"HEAD:{relative}"),
    }


def validate_buffer_chain() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if base.sha256_file(BUFFER_PROOF) != BUFFER_PROOF_SHA256:
        raise CaptureProfileFailure("BUFFER_PROOF_HASH")
    proof = json.loads(BUFFER_PROOF.read_text(encoding="utf-8"))
    if not (
        qml_profile.canonical_without_self(proof) == BUFFER_PROOF_CANONICAL
        and proof.get("canonical_payload_sha256") == BUFFER_PROOF_CANONICAL
        and proof.get("passed") is True
        and proof.get("verdict")
        == "WRF_RRTMG_LW_TOP_BUFFER_D03_TSK_CANDIDATE_PROVEN"
        and proof.get("tsk_parity", {}).get("land_rms_and_max_strictly_improve") is True
        and proof.get("tsk_parity", {}).get("water_bitwise_invariant") is True
        and proof.get("delta_artifact", {}).get("sha256") == BUFFER_DELTA_SHA256
        and proof.get("delta_contract", {}).get("fields") == list(DELTA_FIELDS)
        and proof.get("delta_contract", {}).get("paired_same_inputs_except_lwdn") is True
        and proof.get("execution", {}).get("production_pbl_adapter_invocations") == 0
        and proof.get("execution", {}).get("wrf_or_mpi_executions") == 0
        and proof.get("execution", {}).get("gpu_actions") == 0
    ):
        raise CaptureProfileFailure("BUFFER_PROOF_AUTHORITY")
    if base.sha256_file(BUFFER_DELTA) != BUFFER_DELTA_SHA256:
        raise CaptureProfileFailure("BUFFER_DELTA_HASH")

    arrays: dict[str, np.ndarray] = {}
    with np.load(BUFFER_DELTA, allow_pickle=False) as archive:
        if set(archive.files) != {f"{name}_delta" for name in DELTA_FIELDS}:
            raise CaptureProfileFailure("BUFFER_DELTA_SCHEMA")
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
                raise CaptureProfileFailure(f"BUFFER_DELTA_ARRAY:{name}")
            arrays[field] = np.array(value, copy=True)
    return proof, arrays


def install_combined_state_transform(
    surface_arrays: Mapping[str, np.ndarray],
    qml_deltas: Mapping[str, np.ndarray],
    buffer_deltas: Mapping[str, np.ndarray],
    seam_manifest: Mapping[str, Any],
) -> None:
    import jax
    import jax.numpy as jnp
    from gpuwrf.contracts.state import State

    absolute_fields = tuple(surface_profile.CORRECTION_FIELDS)
    target_fields = tuple(dict.fromkeys((*absolute_fields, *DELTA_FIELDS)))
    records = {
        item["slot_name"]: item
        for item in seam_manifest["state_leaves"]
        if item.get("slot_name") in target_fields
    }
    if set(records) != set(target_fields):
        raise CaptureProfileFailure("SEAM_TRANSFORM_FIELDS")
    original = State.tree_unflatten.__func__

    def transformed(cls: Any, aux: Any, children: Any) -> Any:
        state = original(cls, aux, children)
        if TRANSFORM_COUNT["value"]:
            return state
        for name in target_fields:
            old = np.asarray(getattr(state, name))
            record = records[name]
            if not (
                list(old.shape) == record["shape"]
                and old.dtype.str == record["dtype"]
                and base.sha256_bytes(np.ascontiguousarray(old).tobytes())
                == record["logical_c_bitpayload_sha256"]
            ):
                raise CaptureProfileFailure(f"TRANSFORM_TARGET:{name}")
        updates = {
            name: jax.device_put(surface_arrays[name]).astype(getattr(state, name).dtype)
            for name in absolute_fields
        }
        updates.update(
            {
                name: (
                    jnp.asarray(getattr(state, name))
                    + jax.device_put(qml_deltas[name]).astype(getattr(state, name).dtype)
                    + jax.device_put(buffer_deltas[name]).astype(getattr(state, name).dtype)
                )
                for name in DELTA_FIELDS
            }
        )
        TRANSFORM_COUNT["value"] += 1
        return state.replace(**updates)

    State.tree_unflatten = classmethod(transformed)


def install_combined_preflight(
    seam_manifest: Mapping[str, Any],
    surface_arrays: Mapping[str, np.ndarray],
    qml_deltas: Mapping[str, np.ndarray],
    buffer_proof: Mapping[str, Any],
    buffer_deltas: Mapping[str, np.ndarray],
) -> None:
    original = base.static_preflight

    def profiled(namespace: Path, proof_output: Path) -> dict[str, Any]:
        result = original(namespace, proof_output)
        result["schema"] = (
            "wrfgpu2-v0234-gpt-land-tsk-qml-rrtmg-final-accepted-cpu-preflight-v1"
        )
        result["combined_profile"] = profile_record()
        result["rrtmg_lw_top_buffer_correction"] = {
            "proof_path": str(BUFFER_PROOF),
            "proof_sha256": BUFFER_PROOF_SHA256,
            "proof_canonical_payload_sha256": BUFFER_PROOF_CANONICAL,
            "proof_head": buffer_proof["git"]["head"],
            "artifact_path": str(BUFFER_DELTA),
            "artifact_sha256": BUFFER_DELTA_SHA256,
            "changed_state_handles": list(DELTA_FIELDS),
            "application": buffer_proof["delta_contract"]["application"],
            "water_delta_bitwise_zero": True,
            "source_authorized": True,
            "synthetic_wrf_reference": False,
            "empirical_tuning": False,
            "model_sources": {
                relative: base.tracked_source_record(relative) for relative in FIX_SOURCES
            },
        }
        result["correction_composition"] = {
            "order": [
                "authenticated seam",
                "seasonal absolute surface handles",
                "QML paired delta",
                "RRTMG-buffer paired delta",
            ],
            "same_four_incremental_handles": list(DELTA_FIELDS),
            "backend_offset_imported": False,
        }
        if CAPTURE_MODE:
            install_combined_state_transform(
                surface_arrays,
                qml_deltas,
                buffer_deltas,
                seam_manifest,
            )
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
        qml_profile.CAPTURE_MODE = False
        qml_profile.NAMESPACE_PARENT = NAMESPACE_PARENT
        seam_manifest = qml_profile.configure()
        qml_profile.tracked_head_gate(args.approved_head)
        surface_proof, surface_arrays = surface_profile.validate_correction_chain()
        qml_proof, qml_deltas = qml_profile.validate_provenance_chain()
        buffer_proof, buffer_deltas = validate_buffer_chain()
        qml_profile.install_profile_preflight(
            seam_manifest,
            surface_proof,
            surface_arrays,
            qml_proof,
            qml_deltas,
        )
        install_combined_preflight(
            seam_manifest,
            surface_arrays,
            qml_deltas,
            buffer_proof,
            buffer_deltas,
        )
        namespace = (NAMESPACE_PARENT / args.namespace).resolve()
        proof_output = args.proof_output.resolve()
        if args.preflight:
            result = base.static_preflight(namespace, proof_output)
            if TRANSFORM_COUNT["value"] != 0:
                raise CaptureProfileFailure("PREFLIGHT_IMPORTED_BACKEND")
            base.atomic_json(proof_output, result)
        else:
            result = base.execute_capture(namespace, proof_output)
            if TRANSFORM_COUNT["value"] != 1:
                raise CaptureProfileFailure(
                    f"COMBINED_TRANSFORM_COUNT:{TRANSFORM_COUNT['value']}"
                )
        print(
            json.dumps(
                {
                    "passed": result.get("passed"),
                    "schema": result.get("schema"),
                    "namespace": str(namespace),
                    "proof": str(proof_output),
                    "combined_transform_count": TRANSFORM_COUNT["value"],
                },
                sort_keys=True,
            )
        )
        return 0
    except (
        CaptureProfileFailure,
        qml_profile.ProfileFailure,
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
