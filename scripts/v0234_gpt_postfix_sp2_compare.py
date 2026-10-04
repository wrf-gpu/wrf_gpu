#!/usr/bin/env python3
"""Build and compare a same-authority fixed-tree MYNN SP2 28-array bundle.

Preflight is backend-dark.  Materialization uses the one fresh GPU seam
capture plus exactly one recorded production CPU adapter invocation.  The
unchanged frozen exact comparator is dispatched offline against the authentic
WRF dump tree.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-19-v0234-gpt-single-authority-attribution"
NONCE = "a73aef522fbdca9c88ec1c50464db6e51819f47aa00cc9493a1fe2616cd2ebfe"
PREFIX = NONCE[:16]
AUTHORITY_ROOT = Path(f"<DATA_ROOT>/wrf_gpu2/v0234_gpt_postfix_sp2_{PREFIX}")
GPU_CAPTURE_ROOT = AUTHORITY_ROOT / f"capture/authentic-{PREFIX}"
CPU_NAMESPACE = AUTHORITY_ROOT / "cpu-adapter/single-authority-v1"
REFERENCE_ROOT = AUTHORITY_ROOT / "cpu-reference"
REFERENCE_ARCHIVE = REFERENCE_ROOT / "postfix-sp2-reference-28.npz"
REFERENCE_MANIFEST = REFERENCE_ROOT / "postfix-sp2-reference-manifest.json"
REFERENCE_VALIDATION = REFERENCE_ROOT / "postfix-sp2-reference-validation.json"
COMPARATOR_ROOT = AUTHORITY_ROOT / "comparator"

ACCEPTED_COMPONENT_HEAD = "672f55c0e6ac5ede82c288cdcaed63fd71a5e9a6"
FIXED_COUPLER_SHA256 = "694e51f8e9df30d7c5abd3729c8eb756b6dbfd45c83bf774f95a23747b0b2283"
PRE_FIX_RMS = {
    "rublten": 0.0002538443572720412,
    "rvblten": 0.00016647474452487508,
}
HISTORICAL_MIXED_AUTHORITY_RMS = {
    "rublten": 1.798167321463815e-05,
    "rvblten": 3.524225971155768e-05,
}

REFERENCE_SCRIPT = REPO / "scripts/v0234_pristine_pbl_reference.py"
REFERENCE_SCRIPT_SHA256 = "497fb47b4c686c7b9d7a50c6d46c041ea89c4318b55899a94a84da7fadca9457"
REF_SPEC = importlib.util.spec_from_file_location("v0234_postfix_reference_base", REFERENCE_SCRIPT)
if REF_SPEC is None or REF_SPEC.loader is None:
    raise RuntimeError("reference base import unavailable")
ref = importlib.util.module_from_spec(REF_SPEC)
REF_SPEC.loader.exec_module(ref)

COMPARATOR = Path(
    "<USER_HOME>/src/wrf_gpu2_wt/v0234-mynn-sp2-input-provenance/"
    "scripts/v0234_mynn_sp2_compare.py"
)
COMPARATOR_SHA256 = "0eb3000d28783c0f9a259830dc88ff566d544a3fe7c69e4064e71a4fab67881b"
PYTHON = Path("<USER_HOME>/miniconda3/bin/python3.13")
LEGACY_CONTRACT_ID = "2026-07-19-v0234-mynn-sp2-input-provenance-gpt"
FIXED_ROWS_SHA256 = "550178b158a849c5da55cee31b82e6c90ebee733620d141ac278c064764cf33a"

AUTHENTIC_28_MANIFEST = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_pristine_pbl_entry_closure_ac6712170cbe5084/"
    "cpu-reference/pbl-sp2-reference-manifest.json"
)
AUTHENTIC_28_MANIFEST_SHA256 = "7063326d9df10d2e78a7804d6d2a9d1bae666bbd7be4a62e51bf656670f81933"
AUTHENTIC_28_ARCHIVE = AUTHENTIC_28_MANIFEST.with_name("pbl-sp2-reference-28.npz")
AUTHENTIC_28_ARCHIVE_SHA256 = "a6416b7245d26f23f0df398dd6a3a926a1749cba2069dc3d0ea39c98bc3d2566"
AUTHENTIC_28_VALIDATION = AUTHENTIC_28_MANIFEST.with_name("pbl-sp2-reference-validation.json")
AUTHENTIC_28_VALIDATION_SHA256 = "acebb09e607df32e0e4d81d83930068ed5e9b7c0c1ddcc599154703fea03fbe3"

OLD_GPU_CAPTURE_ROOT = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_pristine_pbl_entry_closure_ac6712170cbe5084/"
    "capture/authentic-ac6712170cbe5084"
)
OLD_GPU_CAPTURE_MANIFEST_SHA256 = "295760b17cba2fde8e931188caab76fbf0564b7e9229784e8f739270f75309e6"
PRE_FIX_CPU_ARCHIVE = Path(
    "<DATA_ROOT>/wrf_gpu2/v0234_gpt_single_authority_attribution/"
    "cpu-production-adapter-authority-v1/single-authority-capture.npz"
)
PRE_FIX_CPU_ARCHIVE_SHA256 = "bc576333ba54d26db7272c8692b327b56f477e6183a6c02d596bf0e6685cd2fe"

SOURCE_SHA256 = {
    "src/gpuwrf/contracts/state.py": "f959da39d8957ef72f165c2e4fad98a03e8235a58379e5a2743f9a6da93af9f8",
    "src/gpuwrf/coupling/physics_couplers.py": FIXED_COUPLER_SHA256,
    "src/gpuwrf/io/gen2_accessor.py": "3f552e3b3552ee16de6770170881d52330e91a3b9a3960fb174c2e9dc568c898",
    "src/gpuwrf/physics/mynn_constants.py": "0172e9424a126e2bde760066be7ba3f483240b53889b834c1f6441a7ff058aee",
    "src/gpuwrf/physics/mynn_pbl.py": "e4778816aeeba28acabdee49043eeac2693ab54b218b6c221b684c89281a1434",
    "src/gpuwrf/physics/noahmp_coupler.py": "cd857efa18731d796fd4a60047eb833fae94588e3150d2973cbd8100e34ced76",
    "src/gpuwrf/physics/tridiagonal_solver.py": "1f141e6581c41daf3a720ecbedfbe557b72d4a0c84bf61a7d3b3a7d449da193f",
    "src/gpuwrf/runtime/operational_mode.py": "68efe76b9d1f91860e9a49a6e6c9ab9e573986dd11a47677fa161badb3a79282",
}

REQUIRED = ref.REQUIRED
RESIDUALS = ref.RESIDUALS
RESIDUAL_STAGE = ref.RESIDUAL_STAGE
EXPECTED_SHAPES = ref.EXPECTED_SHAPES


class PostfixFailure(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_without_self(value: Mapping[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "canonical_payload_sha256"}
    return hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise PostfixFailure(f"OUTPUT_NOT_FRESH:{path}")
    payload = dict(value)
    payload["canonical_payload_sha256"] = canonical_without_self(payload)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(
            payload, sort_keys=True, indent=2, allow_nan=False,
        ) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def require_file(path: Path, expected: str | None = None) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise PostfixFailure(f"NOT_REGULAR:{path}")
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise PostfixFailure(f"HASH:{path}:{actual}")
    return {"path": str(path), "sha256": actual, "size": path.stat().st_size}


def load_canonical(path: Path, expected: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    record = require_file(path, expected)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or canonical_without_self(value) != value.get("canonical_payload_sha256"):
        raise PostfixFailure(f"CANONICAL:{path}")
    return value, record


def git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args], check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise PostfixFailure(f"GIT:{' '.join(args)}:{result.stderr}")
    return result.stdout.strip()


def fixed_tree_gate(approved_head: str) -> dict[str, Any]:
    if git_text("rev-parse", "HEAD") != approved_head or git_text("status", "--porcelain"):
        raise PostfixFailure("HEAD_OR_WORKTREE")
    if subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", ACCEPTED_COMPONENT_HEAD, approved_head],
        check=False,
    ).returncode:
        raise PostfixFailure("FIXED_HEAD_ANCESTRY")
    if subprocess.run(
        ["git", "-C", str(REPO), "diff", "--quiet", f"{ACCEPTED_COMPONENT_HEAD}..{approved_head}", "--", "src/gpuwrf"],
        check=False,
    ).returncode:
        raise PostfixFailure("PRODUCTION_SOURCE_DRIFT")
    return {
        "approved_head": approved_head,
        "approved_tree": git_text("rev-parse", f"{approved_head}^{{tree}}"),
        "accepted_component_head": ACCEPTED_COMPONENT_HEAD,
        "production_source_unchanged": True,
        "sources": {name: require_file(REPO / name, digest) for name, digest in SOURCE_SHA256.items()},
    }


def configure_reference_base() -> None:
    ref.SPRINT = SPRINT
    ref.OUTPUT_AUTHORITY_ROOT = AUTHORITY_ROOT
    ref.CAPTURE_ROOT = GPU_CAPTURE_ROOT
    ref.REFERENCE_ROOT = REFERENCE_ROOT
    ref.REFERENCE_ARCHIVE = REFERENCE_ARCHIVE
    ref.REFERENCE_MANIFEST = REFERENCE_MANIFEST
    ref.REFERENCE_VALIDATION = REFERENCE_VALIDATION
    ref.NONCE = NONCE
    ref.SOURCE_SHA256 = SOURCE_SHA256


def validate_authentic_28() -> dict[str, Any]:
    require_file(AUTHENTIC_28_ARCHIVE, AUTHENTIC_28_ARCHIVE_SHA256)
    manifest, manifest_record = load_canonical(
        AUTHENTIC_28_MANIFEST, AUTHENTIC_28_MANIFEST_SHA256
    )
    validation, validation_record = load_canonical(
        AUTHENTIC_28_VALIDATION, AUTHENTIC_28_VALIDATION_SHA256
    )
    result = ref.validate_reference(AUTHENTIC_28_MANIFEST, AUTHENTIC_28_ARCHIVE)
    if not (
        manifest.get("reference_status") == "AUTHENTIC_28_OF_28"
        and validation.get("valid") is True
        and result.get("array_count") == 28
    ):
        raise PostfixFailure("AUTHENTIC_28_STATUS")
    return {
        "manifest": manifest_record,
        "archive": require_file(AUTHENTIC_28_ARCHIVE, AUTHENTIC_28_ARCHIVE_SHA256),
        "validation": validation_record,
        "array_count": 28,
        "status": "AUTHENTIC_28_OF_28",
    }


def preflight(approved_head: str) -> dict[str, Any]:
    ref.assert_backend_dark()
    configure_reference_base()
    resource = ref.resource_gate(cpu_backend=False)
    planned = (
        REFERENCE_ARCHIVE,
        REFERENCE_MANIFEST,
        REFERENCE_VALIDATION,
        COMPARATOR_ROOT / "run-proof-adapter.json",
        COMPARATOR_ROOT / "reference-adapter.json",
        COMPARATOR_ROOT / "scientific-comparison.json",
        COMPARATOR_ROOT / "comparison-terminal.json",
    )
    if any(path.exists() or path.is_symlink() for path in planned):
        raise PostfixFailure("PLANNED_OUTPUT_NOT_FRESH")
    return {
        "schema": "wrfgpu2-v0234-gpt-postfix-sp2-comparator-preflight-v1",
        "approved_head": approved_head,
        "fixed_tree": fixed_tree_gate(approved_head),
        "authentic_28": validate_authentic_28(),
        "wrf_dump": ref.dump_authority(deep=True),
        "exact_comparator": require_file(COMPARATOR, COMPARATOR_SHA256),
        "reference_base": require_file(REFERENCE_SCRIPT, REFERENCE_SCRIPT_SHA256),
        "pre_fix_rms": PRE_FIX_RMS,
        "historical_mixed_authority_rms": HISTORICAL_MIXED_AUTHORITY_RMS,
        "classification": {
            "improved_both": "both post-fix RMS values strictly below pre-fix",
            "mixed_or_regressed": "otherwise",
            "tolerance_changes": 0,
        },
        "backend_imported": False,
        "gpu_actions": 0,
        "wrf_or_mpi_executions": 0,
        "resource_gate": resource,
        "passed": True,
        "checked_utc": now(),
    }


def array_record(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject or not np.isfinite(array).all():
        raise PostfixFailure("ARRAY_NONFINITE")
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "byte_order": array.dtype.byteorder,
        "nbytes": int(array.nbytes),
        "logical_c_bitpayload_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def metrics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    lhs = np.asarray(left)
    rhs = np.asarray(right)
    if lhs.shape != rhs.shape:
        raise PostfixFailure(f"METRIC_SHAPE:{lhs.shape}:{rhs.shape}")
    delta = lhs.astype(np.float64) - rhs.astype(np.float64)
    if not np.isfinite(delta).all():
        raise PostfixFailure("METRIC_NONFINITE")
    return {
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "sse": float(np.sum(delta * delta)),
        "exact_fraction": float(np.mean(lhs == rhs)),
    }


def classify_postfix(postfix_rms: Mapping[str, Any]) -> str:
    observed: dict[str, float] = {}
    for field in ("rublten", "rvblten"):
        try:
            value = float(postfix_rms[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise PostfixFailure(f"POSTFIX_RMS:{field}") from exc
        if not math.isfinite(value) or value < 0.0:
            raise PostfixFailure(f"POSTFIX_RMS:{field}:{value}")
        observed[field] = value
    return (
        "POSTFIX_SP2_VALIDATED_IMPROVED_BOTH"
        if all(observed[field] < PRE_FIX_RMS[field] for field in observed)
        else "POSTFIX_SP2_VALIDATED_MIXED_OR_REGRESSED"
    )


def validate_cpu_capture(proof_path: Path, proof_sha: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    proof, proof_record = load_canonical(proof_path, proof_sha)
    if not (
        proof.get("schema") == "wrfgpu2-v0234-gpt-single-authority-capture-proof-v1"
        and proof.get("passed") is True
        and proof.get("namespace") == str(CPU_NAMESPACE)
        and proof.get("authority", {}).get("adapter_invocations") == 1
        and proof.get("authority", {}).get("backend") == "cpu"
        and proof.get("authority", {}).get("gpu_actions") == 0
    ):
        raise PostfixFailure("CPU_CAPTURE_PROOF_STATUS")
    consistency = proof.get("self_consistency", {})
    for component in ("u", "v"):
        item = consistency.get(component, {})
        if not (
            item.get("solve_x_vs_mean_output", {}).get("max_abs") == 0.0
            and item.get("solve_implied_vs_adapter_tendency", {}).get("max_abs")
            == 0.0
        ):
            raise PostfixFailure(f"CPU_CAPTURE_EXACT_CLOSURE:{component}")
    manifest_path = CPU_NAMESPACE / "manifest.json"
    manifest, manifest_record = load_canonical(
        manifest_path, proof.get("manifest", {}).get("sha256")
    )
    archive_path = CPU_NAMESPACE / "single-authority-capture.npz"
    require_file(archive_path, manifest.get("archive", {}).get("sha256"))
    source = manifest.get("imported_gpuwrf_sources", {}).get(
        "gpuwrf.coupling.physics_couplers", {}
    )
    if source.get("sha256") != FIXED_COUPLER_SHA256:
        raise PostfixFailure("CPU_CAPTURE_SOURCE_DRIFT")
    arrays: dict[str, np.ndarray] = {}
    declared = manifest.get("archive", {}).get("arrays", {})
    with np.load(archive_path, allow_pickle=False) as archive:
        if archive.files != manifest.get("archive", {}).get("array_order"):
            raise PostfixFailure("CPU_ARCHIVE_ORDER")
        for name in archive.files:
            value = np.ascontiguousarray(archive[name])
            if array_record(value) != {
                **declared[name], "byte_order": value.dtype.byteorder,
            }:
                # The historical capture schema omits byte_order.
                observed = array_record(value)
                observed.pop("byte_order")
                if observed != declared[name]:
                    raise PostfixFailure(f"CPU_ARCHIVE_ARRAY:{name}")
            arrays[name] = value
    return {
        "proof": proof_record,
        "proof_canonical_sha256": proof["canonical_payload_sha256"],
        "manifest": manifest_record,
        "manifest_canonical_sha256": manifest["canonical_payload_sha256"],
        "archive": require_file(archive_path, manifest["archive"]["sha256"]),
        "adapter_invocations": 1,
    }, arrays


def capture_state_comparison() -> dict[str, Any]:
    new_manifest, _ = load_canonical(GPU_CAPTURE_ROOT / "manifest.json")
    old_manifest, old_record = load_canonical(
        OLD_GPU_CAPTURE_ROOT / "manifest.json", OLD_GPU_CAPTURE_MANIFEST_SHA256
    )
    if [item.get("slot_name") for item in new_manifest.get("state_leaves", [])] != [
        item.get("slot_name") for item in old_manifest.get("state_leaves", [])
    ]:
        raise PostfixFailure("GPU_CAPTURE_SLOT_ORDER")
    records = []
    for new, old in zip(
        new_manifest["state_leaves"], old_manifest["state_leaves"], strict=True
    ):
        if new["kind"] != old["kind"]:
            raise PostfixFailure("GPU_CAPTURE_SLOT_KIND")
        same = new["kind"] == "none" or (
            new["logical_c_bitpayload_sha256"]
            == old["logical_c_bitpayload_sha256"]
        )
        records.append({"slot_name": new["slot_name"], "bit_exact": same})
    return {
        "old_manifest": old_record,
        "state_slot_count": len(records),
        "bit_exact_slot_count": sum(item["bit_exact"] for item in records),
        "all_state_slots_bit_exact": all(item["bit_exact"] for item in records),
        "differing_slots": [item["slot_name"] for item in records if not item["bit_exact"]],
    }


def validate_reference_bundle(
    manifest_path: Path = REFERENCE_MANIFEST,
    archive_path: Path = REFERENCE_ARCHIVE,
) -> dict[str, Any]:
    manifest, manifest_record = load_canonical(manifest_path)
    if not (
        manifest.get("schema") == "wrfgpu2-v0234-gpt-postfix-sp2-reference-v1"
        and manifest.get("reference_status") == "AUTHENTIC_POSTFIX_28_OF_28"
        and manifest.get("array_order") == list(REQUIRED)
        and manifest.get("capture_nonce") == NONCE
        and manifest.get("production_adapter_invocations") == 1
    ):
        raise PostfixFailure("POSTFIX_REFERENCE_STATUS")
    require_file(archive_path, manifest.get("archive_sha256"))
    with np.load(archive_path, allow_pickle=False) as archive:
        if tuple(archive.files) != REQUIRED:
            raise PostfixFailure("POSTFIX_REFERENCE_ORDER")
        for name in archive.files:
            value = np.ascontiguousarray(archive[name])
            if array_record(value) != manifest.get("arrays", {}).get(name):
                raise PostfixFailure(f"POSTFIX_REFERENCE_ARRAY:{name}")
            if tuple(value.shape) != EXPECTED_SHAPES[name] or value.dtype.str != "<f8":
                raise PostfixFailure(f"POSTFIX_REFERENCE_LAYOUT:{name}")
    return {
        "schema": "wrfgpu2-v0234-gpt-postfix-sp2-reference-validation-v1",
        "valid": True,
        "manifest": manifest_record,
        "manifest_canonical_sha256": manifest["canonical_payload_sha256"],
        "archive": require_file(archive_path, manifest["archive_sha256"]),
        "array_count": 28,
        "array_order": list(REQUIRED),
        "production_adapter_invocations": 1,
    }


def materialize(approved_head: str, cpu_proof: Path, cpu_proof_sha: str) -> dict[str, Any]:
    ref.assert_backend_dark()
    configure_reference_base()
    fixed = fixed_tree_gate(approved_head)
    resource = ref.resource_gate(cpu_backend=True)
    cpu_authority, cpu_arrays = validate_cpu_capture(cpu_proof, cpu_proof_sha)
    if any(REFERENCE_ROOT.iterdir()):
        raise PostfixFailure("REFERENCE_ROOT_NOT_FRESH")

    sys.path.insert(0, str(REPO / "src"))
    import jax
    import jax.numpy as jnp
    from gpuwrf.contracts.state import State
    from gpuwrf.coupling import physics_couplers as couplers
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.physics import mynn_pbl as mynn

    if jax.default_backend() != "cpu" or any(device.platform != "cpu" for device in jax.devices()):
        raise PostfixFailure("NON_CPU_BACKEND")
    if not bool(jax.config.jax_enable_x64):
        raise PostfixFailure("JAX_X64")
    state, gpu_capture = ref.load_capture_state(jax, State)
    grid = Gen2Run(ref.INPUT_DIR).grid("d03").as_grid_spec()
    diagnostic, initialized = ref._run_pbl(
        state, grid, jax, couplers, mynn, record=True
    )

    overlap = {
        "initialized_qke": metrics(
            diagnostic["mix_qke_initialized"], cpu_arrays["initialized_state_qke"]
        ),
        "mix_el": metrics(diagnostic["mix_el"], cpu_arrays["turbulence_el"]),
        "mix_dfm": metrics(diagnostic["mix_dfm"], cpu_arrays["turbulence_dfm"]),
        "rublten": metrics(diagnostic["rublten"], cpu_arrays["adapter_rublten"]),
        "rvblten": metrics(diagnostic["rvblten"], cpu_arrays["adapter_rvblten"]),
    }
    for component in ("u", "v"):
        for coefficient in ("a", "b", "c", "d", "x"):
            name = f"solve_{component}_{coefficient}"
            overlap[name] = metrics(diagnostic[name], cpu_arrays[name])
    # The production capture is the authority for every returned mixing,
    # coefficient, solve, output, and tendency array.  The independent lower
    # replay exists only to expose ``sm``, which ``_mym_turbulence`` computes
    # internally but does not return.  Different JAX execution boundaries can
    # differ by roundoff, so they must never replace the exact same-invocation
    # arrays or be promoted to an exact-identity gate.  The lifecycle boundary
    # itself remains exact: both paths must initialize the same QKE bitpayload.
    if overlap["initialized_qke"]["max_abs"] != 0.0:
        raise PostfixFailure("DIAGNOSTIC_REPLAY_LIFECYCLE_DRIFT")

    metas = [
        ref.parse_meta(ref.DUMP_ROOT / "mynnsp2" / f"rank{rank:04d}" / "meta.txt")
        for rank in range(6)
    ]
    wrf_surface = {
        name: ref.load_outer(metas, name, 2)
        for name in ("ust", "hfx", "qfx", "tsk", "ch")
    }
    wrf_mix = {
        "qke": ref.load_columns(metas, "mix_qke_initialized"),
        "el": ref.load_columns(metas, "mix_el_for_dfm"),
        "dfm": ref.load_columns(metas, "mix_dfm"),
    }
    wrf_system = {
        f"solve_{uv}_{name}": ref.load_columns(metas, f"solve_{uv}_{name}")
        for uv in ("u", "v") for name in ("a", "b", "c", "d", "x")
    }
    wrf_target = {
        name: ref.load_outer(metas, f"{name}_exit", 3)
        for name in ("rublten", "rvblten")
    }

    qv0 = np.asarray(state.qv)[0]
    cpm = 1004.5 * (1.0 + 0.84 * np.maximum(qv0, 0.0))
    surface_state = state.replace(
        ustar=jnp.asarray(wrf_surface["ust"], dtype=state.ustar.dtype),
        theta_flux=jnp.asarray(
            wrf_surface["hfx"] / (np.asarray(state.rhosfc) * cpm),
            dtype=state.theta_flux.dtype,
        ),
        qv_flux=jnp.asarray(
            wrf_surface["qfx"] / np.asarray(state.rhosfc),
            dtype=state.qv_flux.dtype,
        ),
        t_skin=jnp.asarray(wrf_surface["tsk"], dtype=state.t_skin.dtype),
    )
    fltv = (
        (1.0 + 0.61 * np.maximum(qv0, 0.0)) * np.asarray(surface_state.theta_flux)
        + 0.61 * np.asarray(state.theta)[0] * np.asarray(surface_state.qv_flux)
    )
    surface_state = surface_state.replace(
        fltv=jnp.asarray(fltv, dtype=state.fltv.dtype)
    )
    surface_run, _ = ref._run_pbl(surface_state, grid, jax, couplers, mynn)
    mix_initialized = state.replace(
        qke=jnp.asarray(wrf_mix["qke"], dtype=state.qke.dtype)
    )
    mixing_run, _ = ref._run_pbl(
        state,
        grid,
        jax,
        couplers,
        mynn,
        initialized_state=mix_initialized,
        dfm_substitution=wrf_mix["dfm"],
        el_substitution=wrf_mix["el"],
    )
    residuals = ref.ordered_surface_mixing_residuals(
        surface_run, mixing_run, wrf_target
    )
    u_before = np.asarray(couplers._u_mass(initialized))
    v_before = np.asarray(couplers._v_mass(initialized))
    residuals["residual_after_lower_bc_rublten"] = (
        (wrf_system["solve_u_x"] - u_before) / 6.0 - wrf_target["rublten"]
    )
    residuals["residual_after_lower_bc_rvblten"] = (
        (wrf_system["solve_v_x"] - v_before) / 6.0 - wrf_target["rvblten"]
    )

    values = {
        "surface_ust": np.asarray(state.ustar),
        "surface_hfx": np.asarray(state.theta_flux) * np.asarray(state.rhosfc) * cpm,
        "surface_qfx": np.asarray(state.qv_flux) * np.asarray(state.rhosfc),
        "surface_tsk": np.asarray(state.t_skin),
        "surface_ch": np.load(
            GPU_CAPTURE_ROOT / "auxiliary/surface_ch.npy", allow_pickle=False
        ),
        "mix_qke_initialized": cpu_arrays["initialized_state_qke"],
        "mix_el": cpu_arrays["turbulence_el"],
        "mix_sm": diagnostic["mix_sm"],
        "mix_dfm": cpu_arrays["turbulence_dfm"],
        **{
            f"solve_{uv}_{name}": cpu_arrays[f"solve_{uv}_{name}"]
            for uv in ("u", "v") for name in ("a", "b", "c", "d", "x")
        },
        "sp2_rublten": cpu_arrays["adapter_rublten"],
        "sp2_rvblten": cpu_arrays["adapter_rvblten"],
        "xland": np.asarray(state.xland),
        **residuals,
    }
    if tuple(values) != REQUIRED:
        raise PostfixFailure(f"REFERENCE_ORDER:{tuple(values)}")
    arrays = {
        name: np.ascontiguousarray(np.asarray(values[name], dtype=np.float64))
        for name in REQUIRED
    }
    for name, value in arrays.items():
        if tuple(value.shape) != EXPECTED_SHAPES[name] or not np.isfinite(value).all():
            raise PostfixFailure(f"REFERENCE_ARRAY:{name}:{value.shape}")

    temporary = REFERENCE_ROOT / f".postfix.tmp-{os.getpid()}.npz"
    np.savez(temporary, **arrays)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.link(temporary, REFERENCE_ARCHIVE)
    temporary.unlink()

    provenance = {}
    for name in REQUIRED:
        if name in RESIDUALS:
            provenance[name] = {
                "synthetic": False,
                "source": "source_ordered_authentic_wrf_substitution",
                "residual_stage": RESIDUAL_STAGE[name],
                "wrf_edges": ["sealed_substitution_operand", "sealed_wrf_target"],
            }
        elif name == "mix_sm":
            provenance[name] = {
                "synthetic": False,
                "source": "source_instrumented_diagnostic_replay_unreturned_sm_only",
                "wrf_edges": [],
                "production_replacement": False,
            }
        else:
            provenance[name] = {
                "synthetic": False,
                "source": "fresh_gpu_seam_and_same_authority_fixed_cpu_adapter",
                "wrf_edges": [],
            }

    with np.load(PRE_FIX_CPU_ARCHIVE, allow_pickle=False) as old_cpu:
        require_file(PRE_FIX_CPU_ARCHIVE, PRE_FIX_CPU_ARCHIVE_SHA256)
        fixed_vs_prefix = {
            field: metrics(
                arrays[f"sp2_{field}"], old_cpu[f"adapter_{field}"]
            )
            for field in ("rublten", "rvblten")
        }
    postfix_vs_wrf = {
        field: metrics(arrays[f"sp2_{field}"], wrf_target[field])
        for field in ("rublten", "rvblten")
    }
    improvement = {
        field: {
            "pre_fix_rms": PRE_FIX_RMS[field],
            "post_fix_rms": postfix_vs_wrf[field]["rms"],
            "improvement_factor": (
                PRE_FIX_RMS[field] / postfix_vs_wrf[field]["rms"]
                if postfix_vs_wrf[field]["rms"] else math.inf
            ),
            "strictly_improved": postfix_vs_wrf[field]["rms"] < PRE_FIX_RMS[field],
        }
        for field in ("rublten", "rvblten")
    }
    manifest = {
        "schema": "wrfgpu2-v0234-gpt-postfix-sp2-reference-v1",
        "capture_nonce": NONCE,
        "approved_head": approved_head,
        "fixed_tree": fixed,
        "gpu_capture": {
            "manifest": gpu_capture["manifest"],
            "manifest_canonical_sha256": gpu_capture["value"]["canonical_payload_sha256"],
        },
        "cpu_capture": cpu_authority,
        "production_adapter_invocations": 1,
        "diagnostic_lower_replays_in_materialization": 3,
        "diagnostic_replay_cross_execution_metrics": overlap,
        "diagnostic_replay_initialized_qke_bit_exact": True,
        "production_capture_exact_closure": True,
        "restart": False,
        "backend": "cpu",
        "gpu_actions": 0,
        "wrf_or_mpi_executions": 0,
        "wrf_dump_tree_sha256": ref.DUMP_TREE_SHA256,
        "source_sha256": SOURCE_SHA256,
        "archive_path": str(REFERENCE_ARCHIVE),
        "archive_sha256": sha256_file(REFERENCE_ARCHIVE),
        "array_order": list(REQUIRED),
        "arrays": {name: array_record(arrays[name]) for name in REQUIRED},
        "reference_status": "AUTHENTIC_POSTFIX_28_OF_28",
        "provenance": provenance,
        "same_seam_old_vs_new": capture_state_comparison(),
        "fixed_vs_pre_fix_cpu_outputs": fixed_vs_prefix,
        "postfix_vs_wrf": postfix_vs_wrf,
        "improvement_vs_frozen_authentic_pre_fix": improvement,
        "resource_gate": resource,
        "materialized_utc": now(),
    }
    atomic_json(REFERENCE_MANIFEST, manifest)
    validation = validate_reference_bundle()
    atomic_json(REFERENCE_VALIDATION, validation)
    return {
        "verdict": "AUTHENTIC_POSTFIX_28_OF_28",
        "reference_archive_sha256": sha256_file(REFERENCE_ARCHIVE),
        "reference_manifest_sha256": sha256_file(REFERENCE_MANIFEST),
        "reference_validation_sha256": sha256_file(REFERENCE_VALIDATION),
        "postfix_vs_wrf": postfix_vs_wrf,
        "improvement": improvement,
        "production_adapter_invocations": 1,
        "gpu_actions": 0,
    }


def dispatch_exact(
    approved_head: str,
    cpu_proof: Path,
    cpu_proof_sha: str,
    gpu_validation: Path,
    gpu_validation_sha: str,
) -> dict[str, Any]:
    fixed = fixed_tree_gate(approved_head)
    validation = validate_reference_bundle()
    authentic = validate_authentic_28()
    cpu_authority, _ = validate_cpu_capture(cpu_proof, cpu_proof_sha)
    gpu_value, gpu_record = load_canonical(gpu_validation, gpu_validation_sha)
    if gpu_value.get("nonce") != NONCE or gpu_value.get("gpu_arm_count") != 1:
        raise PostfixFailure("GPU_VALIDATION_LINK")

    outputs = {
        "run": COMPARATOR_ROOT / "run-proof-adapter.json",
        "reference": COMPARATOR_ROOT / "reference-adapter.json",
        "comparison": COMPARATOR_ROOT / "scientific-comparison.json",
        "terminal": COMPARATOR_ROOT / "comparison-terminal.json",
    }
    if any(path.exists() or path.is_symlink() for path in outputs.values()):
        raise PostfixFailure("COMPARATOR_OUTPUT_NOT_FRESH")
    with np.load(REFERENCE_ARCHIVE, allow_pickle=False) as arrays:
        declared = {
            name: {"shape": list(arrays[name].shape), "dtype": str(arrays[name].dtype)}
            for name in arrays.files
        }
    run_proof = {
        "schema": "wrfgpu2-mynn-sp2-run-proof-v1",
        "contract_id": LEGACY_CONTRACT_ID,
        "verdict": "WRF_DISCRIMINATOR_CAPTURED_COMPARISON_PENDING",
        "adapter_role": "fixed-tree same-authority post-fix exact comparison",
        "approved_head": approved_head,
        "capture_nonce": NONCE,
        "wrf_dump_tree_sha256": ref.DUMP_TREE_SHA256,
        "scientific_falsification_before_comparator": False,
        "wrf_or_mpi_execution_this_sprint": 0,
    }
    reference_adapter = {
        "schema": "wrfgpu2-mynn-sp2-gpu-reference-v1",
        "contract_id": LEGACY_CONTRACT_ID,
        "carry_sha256": ref.CARRY_SHA256,
        "fixed_rows_sha256": FIXED_ROWS_SHA256,
        "wrf_dump_tree_sha256": ref.DUMP_TREE_SHA256,
        "backend": "cpu",
        "gpu_actions": 0,
        "synthetic_values": False,
        "gpu_source_sha256": SOURCE_SHA256,
        "bundle_path": str(REFERENCE_ARCHIVE),
        "bundle_sha256": sha256_file(REFERENCE_ARCHIVE),
        "arrays": declared,
        "postfix_manifest_sha256": sha256_file(REFERENCE_MANIFEST),
        "capture_nonce": NONCE,
        "approved_head": approved_head,
    }
    atomic_json(outputs["run"], run_proof)
    atomic_json(outputs["reference"], reference_adapter)
    command = [
        str(PYTHON), str(COMPARATOR),
        "--run-proof", str(outputs["run"]),
        "--run-proof-sha256", sha256_file(outputs["run"]),
        "--dump-root", str(ref.DUMP_ROOT),
        "--reference-manifest", str(outputs["reference"]),
        "--reference-manifest-sha256", sha256_file(outputs["reference"]),
        "--output", str(outputs["comparison"]),
    ]
    result = subprocess.run(
        command, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={
            "HOME": os.environ.get("HOME", "<USER_HOME>"),
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "LANG": "C.UTF-8",
            "PYTHONPATH": str(REPO),
            **ref.THREAD_ENV,
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
        },
    )
    if result.returncode != 0:
        raise PostfixFailure(
            f"EXACT_COMPARATOR:{result.returncode}:{result.stderr}"
        )
    comparison, comparison_record = load_canonical(outputs["comparison"])
    manifest, manifest_record = load_canonical(REFERENCE_MANIFEST)
    measured = manifest["postfix_vs_wrf"]
    comparator_baseline = comparison.get("quantitative_attribution", {}).get("baseline", {})
    for field in ("rublten", "rvblten"):
        if comparator_baseline.get(field, {}).get("rms") != measured[field]["rms"]:
            raise PostfixFailure(f"EXACT_COMPARATOR_BASELINE:{field}")
    measured_rms = {
        field: measured[field]["rms"] for field in ("rublten", "rvblten")
    }
    improved = {
        field: measured_rms[field] < PRE_FIX_RMS[field]
        for field in measured_rms
    }
    terminal_verdict = classify_postfix(measured_rms)
    terminal = {
        "schema": "wrfgpu2-v0234-gpt-postfix-sp2-terminal-v1",
        "terminal_verdict": terminal_verdict,
        "passed": True,
        "approved_head": approved_head,
        "fixed_tree": fixed,
        "nonce": NONCE,
        "nonce_consumed_once": True,
        "gpu_arm_count": 1,
        "gpu_query_count": 0,
        "gpu_retry_count": 0,
        "wrf_or_mpi_executions": 0,
        "production_cpu_adapter_invocations": 1,
        "gpu_capture_validation": {
            **gpu_record,
            "canonical_payload_sha256": gpu_value["canonical_payload_sha256"],
        },
        "cpu_capture": cpu_authority,
        "postfix_reference": {
            "manifest": manifest_record,
            "manifest_canonical_sha256": manifest["canonical_payload_sha256"],
            "validation": validation,
        },
        "authentic_28_reference": authentic,
        "exact_comparator": {
            "source": require_file(COMPARATOR, COMPARATOR_SHA256),
            "command": command,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "comparison": comparison_record,
            "comparison_canonical_sha256": comparison["canonical_payload_sha256"],
            "scientific_verdict": comparison.get("verdict"),
        },
        "authentic_pre_fix_rms": PRE_FIX_RMS,
        "post_fix_rms": measured_rms,
        "improvement": manifest["improvement_vs_frozen_authentic_pre_fix"],
        "strictly_improved_both": all(improved.values()),
        "historical_mixed_authority_rms_context_only": HISTORICAL_MIXED_AUTHORITY_RMS,
        "tolerance_changes": 0,
        "generated_utc": now(),
    }
    atomic_json(outputs["terminal"], terminal)
    return {
        "terminal_verdict": terminal_verdict,
        "post_fix_rms": terminal["post_fix_rms"],
        "improvement": terminal["improvement"],
        "exact_comparator_verdict": comparison.get("verdict"),
        "comparison_sha256": comparison_record["sha256"],
        "terminal_sha256": sha256_file(outputs["terminal"]),
        "terminal_canonical_sha256": canonical_without_self(terminal),
        "passed": True,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    modes = value.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--materialize", action="store_true")
    modes.add_argument("--validate", action="store_true")
    modes.add_argument("--compare", action="store_true")
    value.add_argument("--approved-head")
    value.add_argument("--cpu-proof", type=Path)
    value.add_argument("--cpu-proof-sha256")
    value.add_argument("--gpu-validation", type=Path)
    value.add_argument("--gpu-validation-sha256")
    value.add_argument("--output", type=Path)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    configure_reference_base()
    try:
        if args.preflight:
            if not args.approved_head or args.output is None:
                raise PostfixFailure("PREFLIGHT_ARGS")
            result = preflight(args.approved_head)
            atomic_json(args.output.resolve(), result)
        elif args.materialize:
            if not all((args.approved_head, args.cpu_proof, args.cpu_proof_sha256)):
                raise PostfixFailure("MATERIALIZE_ARGS")
            result = materialize(
                args.approved_head,
                args.cpu_proof.resolve(),
                args.cpu_proof_sha256,
            )
            if args.output is not None:
                atomic_json(args.output.resolve(), result)
        elif args.validate:
            result = validate_reference_bundle()
            if args.output is not None:
                atomic_json(args.output.resolve(), result)
        else:
            if not all((
                args.approved_head,
                args.cpu_proof,
                args.cpu_proof_sha256,
                args.gpu_validation,
                args.gpu_validation_sha256,
            )):
                raise PostfixFailure("COMPARE_ARGS")
            result = dispatch_exact(
                args.approved_head,
                args.cpu_proof.resolve(),
                args.cpu_proof_sha256,
                args.gpu_validation.resolve(),
                args.gpu_validation_sha256,
            )
            if args.output is not None:
                atomic_json(args.output.resolve(), result)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (
        PostfixFailure, ref.ReferenceError, OSError, ValueError, TypeError,
        KeyError, json.JSONDecodeError, subprocess.SubprocessError,
    ) as exc:
        print(f"REFUSE:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
