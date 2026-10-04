#!/usr/bin/env python3
"""Fresh frozen-SP2 CPU capture for accepted QML + static-top + O3RAD."""

from __future__ import annotations

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
SCRATCH = Path("/tmp/v0234_land_tsk_scratch")
NAMESPACE_PARENT = SCRATCH / "qml-rrtmg-o3-fixed-cpu-adapter"
BASE_PROFILE = REPO / "scripts/v0234_gpt_land_tsk_rrtmg_cpu_capture.py"
O3_TSK_PROOF = SPRINT / "rrtmg-o3-tsk-proof.json"
O3_REAL_WRF_PROOF = SPRINT / "rrtmg-o3-real-wrf-proof.json"
O3_DELTA = SCRATCH / "rrtmg-o3-surface-delta-v3.npz"
O3_TSK_PROOF_SHA256 = "3505800030964c8706b631d9ed009a1194696854f4471cebe3798a1986eef6ac"
O3_TSK_PROOF_CANONICAL = "411413c2262710dbb5fb65fe0ba45de91882d4bbc326947d5e27310bc146b7bd"
O3_REAL_WRF_PROOF_SHA256 = "333d716259c1d012b7db6d7092717611d3c708113ddf247683161872d753e4d9"
O3_REAL_WRF_PROOF_CANONICAL = "62b84106b003c92bd5ded1eb0290912c1d12f50bbebf7e9ce7c2f15cb534d545"
O3_DELTA_SHA256 = "812e3a2fba5872b456efbb773aaa736c27dd78fbfa611f7dfb69c36a820a9de9"
DELTA_FIELDS = ("theta_flux", "qv_flux", "fltv", "t_skin")
MODEL_SOURCES = (
    "data/fixtures/wrf-cam-ozone-v1.json",
    "data/fixtures/wrf-cam-ozone-v1.npz",
    "src/gpuwrf/physics/wrf_cam_ozone.py",
    "src/gpuwrf/physics/rrtmg_lw.py",
    "src/gpuwrf/physics/rrtmg_sw.py",
    "src/gpuwrf/coupling/physics_couplers.py",
    "scripts/extract_wrf_cam_ozone.py",
    "scripts/v0234_gpt_land_tsk_rrtmg_o3_real_wrf.py",
    "scripts/v0234_gpt_land_tsk_rrtmg_o3_tsk_proof.py",
    "scripts/v0234_gpt_land_tsk_o3_cpu_capture.py",
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"capture profile unavailable: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


accepted = _load("v0234_o3_accepted_capture", BASE_PROFILE)
base = accepted.base
provenance = accepted.qml_profile


class O3CaptureFailure(RuntimeError):
    """Fail-closed O3 proof-chain or state-transform failure."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(record: Mapping[str, Any]) -> str:
    body = {
        key: value
        for key, value in record.items()
        if key != "canonical_payload_sha256"
    }
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _git(*args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args],
        check=False,
        text=not binary,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        error = result.stderr.decode() if binary else result.stderr
        raise O3CaptureFailure(f"GIT:{' '.join(args)}:{error.strip()}")
    return result.stdout if binary else result.stdout.strip()


def _profile_record() -> dict[str, Any]:
    path = Path(__file__).resolve()
    relative = path.relative_to(REPO).as_posix()
    disk = path.read_bytes()
    if disk != _git("show", f"HEAD:{relative}", binary=True):
        raise O3CaptureFailure("PROFILE_NOT_HEAD")
    return {
        "path": str(path),
        "relative": relative,
        "sha256": hashlib.sha256(disk).hexdigest(),
        "git_blob": _git("rev-parse", f"HEAD:{relative}"),
    }


def _validate_o3_chain() -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    if _sha256(O3_TSK_PROOF) != O3_TSK_PROOF_SHA256:
        raise O3CaptureFailure("O3_TSK_PROOF_HASH")
    tsk = json.loads(O3_TSK_PROOF.read_text(encoding="utf-8"))
    if not (
        _canonical(tsk) == O3_TSK_PROOF_CANONICAL
        and tsk.get("canonical_payload_sha256") == O3_TSK_PROOF_CANONICAL
        and tsk.get("passed") is True
        and tsk.get("verdict") == "WRF_O3RAD_STRICTLY_IMPROVES_D03_LAND_TSK"
        and tsk.get("gates", {}).get("production_composition_exact") is True
        and tsk.get("gates", {}).get("land_tsk_rms_and_max_strictly_improve")
        is True
        and tsk.get("gates", {}).get("water_bitwise_invariant") is True
        and tsk.get("gates", {}).get("shortwave_night_branch_exact") is True
        and tsk.get("delta_artifact", {}).get("sha256") == O3_DELTA_SHA256
        and tsk.get("delta_contract", {}).get("fields") == list(DELTA_FIELDS)
        and tsk.get("delta_contract", {}).get("water_delta_bitwise_zero") is True
        and tsk.get("delta_contract", {}).get("paired_same_inputs_except_lwdn")
        is True
    ):
        raise O3CaptureFailure("O3_TSK_PROOF_AUTHORITY")
    if _sha256(O3_REAL_WRF_PROOF) != O3_REAL_WRF_PROOF_SHA256:
        raise O3CaptureFailure("O3_REAL_WRF_PROOF_HASH")
    real = json.loads(O3_REAL_WRF_PROOF.read_text(encoding="utf-8"))
    if not (
        _canonical(real) == O3_REAL_WRF_PROOF_CANONICAL
        and real.get("canonical_payload_sha256") == O3_REAL_WRF_PROOF_CANONICAL
        and real.get("status") == "PASS"
        and real.get("is_self_compare") is False
        and real.get("gates", {}).get("real_wrf_glw_strictly_improves") is True
        and real.get("gates", {}).get("real_wrf_swdnb_strictly_improves") is True
        and real.get("gates", {}).get("omitted_leaves_byte_identical_to_9bfe1e30")
        is True
    ):
        raise O3CaptureFailure("O3_REAL_WRF_PROOF_AUTHORITY")
    if _sha256(O3_DELTA) != O3_DELTA_SHA256:
        raise O3CaptureFailure("O3_DELTA_HASH")
    arrays: dict[str, np.ndarray] = {}
    with np.load(O3_DELTA, allow_pickle=False) as archive:
        if set(archive.files) != {f"{name}_delta" for name in DELTA_FIELDS}:
            raise O3CaptureFailure("O3_DELTA_SCHEMA")
        for field in DELTA_FIELDS:
            name = f"{field}_delta"
            value = np.asarray(archive[name])
            declared = tsk["delta_artifact"]["arrays"][name]
            if not (
                value.shape == (base.NY, base.NX)
                and value.dtype.str == declared["dtype"]
                and base.sha256_bytes(np.ascontiguousarray(value).tobytes())
                == declared["logical_c_bitpayload_sha256"]
                and np.isfinite(value).all()
            ):
                raise O3CaptureFailure(f"O3_DELTA_ARRAY:{name}")
            arrays[field] = np.array(value, copy=True)
    return tsk, real, arrays


def _install_o3_transform(
    surface_arrays: Mapping[str, np.ndarray],
    qml_deltas: Mapping[str, np.ndarray],
    buffer_deltas: Mapping[str, np.ndarray],
    seam_manifest: Mapping[str, Any],
) -> None:
    import jax
    import jax.numpy as jnp
    from gpuwrf.contracts.state import State

    absolute_fields = tuple(accepted.surface_profile.CORRECTION_FIELDS)
    target_fields = tuple(dict.fromkeys((*absolute_fields, *DELTA_FIELDS)))
    records = {
        item["slot_name"]: item
        for item in seam_manifest["state_leaves"]
        if item.get("slot_name") in target_fields
    }
    if set(records) != set(target_fields):
        raise O3CaptureFailure("SEAM_TRANSFORM_FIELDS")
    original = State.tree_unflatten.__func__

    def transformed(cls: Any, aux: Any, children: Any) -> Any:
        state = original(cls, aux, children)
        if accepted.TRANSFORM_COUNT["value"]:
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
                raise O3CaptureFailure(f"TRANSFORM_TARGET:{name}")
        updates = {
            name: jax.device_put(surface_arrays[name]).astype(
                getattr(state, name).dtype
            )
            for name in absolute_fields
        }
        updates.update(
            {
                name: (
                    jnp.asarray(getattr(state, name))
                    + jax.device_put(qml_deltas[name]).astype(
                        getattr(state, name).dtype
                    )
                    + jax.device_put(buffer_deltas[name]).astype(
                        getattr(state, name).dtype
                    )
                    + jax.device_put(O3_ARRAYS[name]).astype(
                        getattr(state, name).dtype
                    )
                )
                for name in DELTA_FIELDS
            }
        )
        accepted.TRANSFORM_COUNT["value"] += 1
        return state.replace(**updates)

    State.tree_unflatten = classmethod(transformed)


O3_TSK, O3_REAL, O3_ARRAYS = _validate_o3_chain()
original_install_preflight = accepted.install_combined_preflight


def _install_o3_preflight(*args, **kwargs) -> None:
    original_install_preflight(*args, **kwargs)
    previous = base.static_preflight

    def profiled(namespace: Path, proof_output: Path) -> dict[str, Any]:
        result = previous(namespace, proof_output)
        result["schema"] = "wrfgpu2-v0234-qml-static-top-o3-cpu-preflight-v1"
        result["o3_capture_profile"] = _profile_record()
        result["wrf_o3rad_correction"] = {
            "tsk_proof_path": str(O3_TSK_PROOF),
            "tsk_proof_sha256": O3_TSK_PROOF_SHA256,
            "tsk_proof_canonical_payload_sha256": O3_TSK_PROOF_CANONICAL,
            "real_wrf_proof_path": str(O3_REAL_WRF_PROOF),
            "real_wrf_proof_sha256": O3_REAL_WRF_PROOF_SHA256,
            "real_wrf_proof_canonical_payload_sha256": O3_REAL_WRF_PROOF_CANONICAL,
            "delta_path": str(O3_DELTA),
            "delta_sha256": O3_DELTA_SHA256,
            "changed_state_handles": list(DELTA_FIELDS),
            "application": O3_TSK["delta_contract"]["application"],
            "source_authorized": True,
            "real_wrf_glw_strictly_improves": True,
            "real_wrf_swdnb_strictly_improves": True,
            "land_tsk_strictly_improves": True,
            "water_delta_bitwise_zero": True,
            "omitted_o3_leaves_byte_identical": True,
            "synthetic_wrf_reference": False,
            "empirical_tuning": False,
            "model_sources": {
                relative: base.tracked_source_record(relative)
                for relative in MODEL_SOURCES
            },
        }
        result["correction_composition"] = {
            "order": [
                "authenticated seam",
                "seasonal absolute surface handles",
                "QML paired delta",
                "RRTMG static-top paired delta",
                "WRF O3RAD paired delta",
            ],
            "same_four_incremental_handles": list(DELTA_FIELDS),
            "backend_offset_imported": False,
        }
        return result

    base.static_preflight = profiled


def main() -> int:
    try:
        accepted.NAMESPACE_PARENT = NAMESPACE_PARENT
        accepted.install_combined_state_transform = _install_o3_transform
        accepted.install_combined_preflight = _install_o3_preflight
        return int(accepted.main(sys.argv[1:]))
    except Exception as exc:  # noqa: BLE001 - profile must refuse closed
        print(f"REFUSE:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
