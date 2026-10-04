#!/usr/bin/env python3
"""Repartition the source-authorized surface-fixed SP2 CPU capture.

This backend-dark wrapper binds the frozen five-category comparator to the
fresh one-invocation CPU archive, validates the complete surface-correction
chain, and proves that only the sealed UST/RHOSFC PBL handles changed at adapter
entry.  Comparator arithmetic is imported unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-19-v0234-gpt-sp2-residual-closure"
STAGED = Path("/tmp/v0234_gpt_sp2_residual_evidence")
CAPTURE_ROOT = STAGED / "surface-fixed-cpu-adapter/tskin-v1"
CAPTURE_ARCHIVE = CAPTURE_ROOT / "single-authority-capture.npz"
CAPTURE_MANIFEST = CAPTURE_ROOT / "manifest.json"
OLD_CAPTURE = STAGED / "corrected-cpu-adapter/metric-authority-v1/single-authority-capture.npz"
CORRECTION = STAGED / "surface-correction/source-authorized-v1.npz"
WRF_ROOT = STAGED / "wrf-dumps"
RAW_OUTPUT = STAGED / "partition-work/surface-fixed-v1-raw.json"

CAPTURE_PREFLIGHT = SPRINT / "tskin-capture-preflight.json"
CAPTURE_PROOF = SPRINT / "tskin-capture-proof.json"
SURFACE_PROOF = SPRINT / "surface-provenance-proof.json"
METRIC_PROVENANCE = SPRINT / "metric-provenance-proof.json"
CAPTURE_PROFILE = REPO / "scripts/v0234_gpt_sp2_fixed_cpu_capture.py"
FROZEN_DRIVER = REPO / "scripts/v0234_fable_sp2_postfix_partition.py"
FROZEN_COMPARATOR = REPO / "scripts/v0234_gpt_operand_attribution.py"
AUDIT_READER = REPO / "scripts/v0234_gpt_adversarial_audit.py"

EXPECTED = {
    "capture_archive": "880ac4d84258d3f8fccc7af3aab86ede89511a05c4be4d8aa9549958fc6fd94c",
    "capture_manifest": "9e503417fb1e0730bb8a9f32f9dff16d3b5a60e4c9ea4da92819bef2291685e2",
    "capture_manifest_canonical": "62f14834fca187855f61bae6721c0a680335d2cdfed6f0492e9d36cc20f055af",
    "capture_proof": "f1e6a568b6636461d7906936951f136c389327047ccbd04b32626d72d2c250f7",
    "capture_preflight": "e781c71199d524c2bc374a1e618918eedf9df07d93aeb08da191071b240dfc10",
    "surface_proof": "2b579b440043586fa33e6137eb10f58c71789967787ba34bce2e19cc79a5e979",
    "surface_proof_canonical": "26f6efc05967abf7b350a7746579f187fea06a44da9cc7d7f47676af4a8b3819",
    "correction": "fc738ea01d76210a786e01c8de4af1e01010b1d2f13280b7b12ca3029d565494",
    "old_capture": "609a539727f027f04ed2cdbd11e44a6830034a406fecd4d15f3206fcaa73b5ef",
    "metric_provenance": "1c0d017c05502e7793df9946ac36ffa0d7203487d2beded91527e57f9413e834",
    "capture_profile": "b29261c9098735dcabe31b1ed072398db34615aca97c2c218980fed6a498af29",
    "capture_head": "84ecd76cbad98f3bf4cf330cf6a8f36e78a544f8",
    "frozen_driver": "e7a2aebba9fb6460d3863a8a1e03ec9240621ea19a496bc429b23416c35d3b4c",
    "frozen_comparator": "247a1dca01b7aeb35af1adcfd3eb1dc459472c04275edc7f5b2cb3f380df3244",
    "audit_reader": "49bc7b9761e2da1e807000bbbc101d05039a585b8b00f4b31c3b404b66d79d2a",
    "wrf_tree": "88e94f6a7ded154bd2b51ba890a4efa17593fb908d5f48f070a508a4b2cb645b",
    "wrfinput": "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a",
}


class PartitionFailure(RuntimeError):
    """Fail-closed evidence, source, or comparator failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_without_self(value: Mapping[str, Any]) -> str:
    body = {
        key: item for key, item in value.items()
        if key != "canonical_payload_sha256"
    }
    return hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def checked_json(path: Path, expected_sha: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha:
        raise PartitionFailure(f"EVIDENCE_HASH:{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if canonical_without_self(value) != value.get("canonical_payload_sha256"):
        raise PartitionFailure(f"EVIDENCE_CANONICAL:{path}")
    return value


def import_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PartitionFailure(f"IMPORT:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_delta(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    lhs = np.asarray(left)
    rhs = np.asarray(right)
    if lhs.shape != rhs.shape or not np.isfinite(lhs).all() or not np.isfinite(rhs).all():
        raise PartitionFailure(f"DELTA_INPUT:{lhs.shape}:{rhs.shape}")
    delta = lhs.astype(np.float64) - rhs.astype(np.float64)
    return {
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "exact_fraction": float(np.mean(lhs == rhs)),
    }


def tracked_wrapper_record() -> dict[str, Any]:
    path = Path(__file__).resolve()
    relative = path.relative_to(REPO).as_posix()
    disk = path.read_bytes()
    head = subprocess.run(
        ["git", "-C", str(REPO), "show", f"HEAD:{relative}"],
        check=True, stdout=subprocess.PIPE,
    ).stdout
    if disk != head:
        raise PartitionFailure("WRAPPER_NOT_HEAD")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(disk).hexdigest(),
        "git_blob": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", f"HEAD:{relative}"],
            check=True, text=True, stdout=subprocess.PIPE,
        ).stdout.strip(),
    }


def validate_chain() -> dict[str, Any]:
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()
    if tracked:
        raise PartitionFailure(f"TRACKED_WORKTREE_NOT_CLEAN:{tracked}")
    for label, path in (
        ("capture_archive", CAPTURE_ARCHIVE),
        ("correction", CORRECTION),
        ("old_capture", OLD_CAPTURE),
        ("capture_profile", CAPTURE_PROFILE),
        ("frozen_driver", FROZEN_DRIVER),
        ("frozen_comparator", FROZEN_COMPARATOR),
        ("audit_reader", AUDIT_READER),
    ):
        if path.is_symlink() or not path.is_file():
            raise PartitionFailure(f"INPUT_NOT_REGULAR:{path}")
        actual = sha256_file(path)
        if actual != EXPECTED[label]:
            raise PartitionFailure(f"INPUT_HASH:{label}:{actual}")

    manifest = checked_json(CAPTURE_MANIFEST, EXPECTED["capture_manifest"])
    capture_proof = checked_json(CAPTURE_PROOF, EXPECTED["capture_proof"])
    preflight = checked_json(CAPTURE_PREFLIGHT, EXPECTED["capture_preflight"])
    surface = checked_json(SURFACE_PROOF, EXPECTED["surface_proof"])
    provenance = checked_json(METRIC_PROVENANCE, EXPECTED["metric_provenance"])
    authority = manifest.get("authority", {})
    correction_record = preflight.get("source_authorized_surface_correction", {})
    if not (
        manifest.get("canonical_payload_sha256")
        == EXPECTED["capture_manifest_canonical"]
        and manifest.get("passed") is True
        and manifest.get("status") == "SINGLE_AUTHORITY_CAPTURE_COMPLETE"
        and manifest.get("single_adapter_invocation") is True
        and manifest.get("single_namespace") is True
        and manifest.get("production_operands_and_tendencies_same_invocation") is True
        and authority.get("adapter_invocations") == 1
        and authority.get("gpu_actions") == 0
        and authority.get("wrf_or_mpi_executions") == 0
        and manifest.get("git", {}).get("head") == EXPECTED["capture_head"]
        and manifest.get("archive", {}).get("sha256") == EXPECTED["capture_archive"]
        and manifest.get("input_grid", {}).get("wrfinput_d03_sha256")
        == EXPECTED["wrfinput"]
        and capture_proof.get("passed") is True
        and capture_proof.get("verdict") == "SINGLE_AUTHORITY_CPU_CAPTURE_GREEN"
        and capture_proof.get("manifest", {}).get("sha256")
        == EXPECTED["capture_manifest"]
        and capture_proof.get("git", {}).get("head") == EXPECTED["capture_head"]
        and preflight.get("passed") is True
        and preflight.get("git", {}).get("head") == EXPECTED["capture_head"]
        and preflight.get("profile", {}).get("sha256") == EXPECTED["capture_profile"]
        and preflight.get("grid_metric_authority", {}).get(
            "analytic_flat_fallback_forbidden"
        ) is True
        and correction_record.get("proof_sha256") == EXPECTED["surface_proof"]
        and correction_record.get("artifact_sha256") == EXPECTED["correction"]
        and correction_record.get("changed_state_handles")
        == ["ustar", "tau_u", "tau_v", "rhosfc"]
        and correction_record.get("unchanged_state_handles")
        == ["theta_flux", "qv_flux", "fltv"]
        and correction_record.get("synthetic_reconstruction") is False
        and correction_record.get("empirical_tuning") is False
        and surface.get("passed") is True
        and surface.get("verdict")
        == "WRF_SEASONAL_SURFACE_AND_MYNN_DENSITY_FIX_PROVEN"
        and surface.get("canonical_payload_sha256")
        == EXPECTED["surface_proof_canonical"]
        and provenance.get("passed") is True
        and provenance.get("verdict")
        == "CPU_CAPTURE_ANALYTIC_FLAT_METRIC_DIVERGENCE_PROVEN"
    ):
        raise PartitionFailure("AUTHORITY_CHAIN")

    with np.load(CORRECTION, allow_pickle=False) as correction, np.load(
        CAPTURE_ARCHIVE, allow_pickle=False,
    ) as current, np.load(OLD_CAPTURE, allow_pickle=False) as previous:
        transformed = {
            name: exact_delta(current[f"entry_state_{name}"], correction[name])
            for name in ("ustar", "rhosfc")
        }
        unchanged = {
            name: exact_delta(
                current[f"entry_state_{name}"], previous[f"entry_state_{name}"]
            )
            for name in ("theta_flux", "qv_flux", "fltv")
        }
    if any(value["max_abs"] != 0.0 for value in (*transformed.values(), *unchanged.values())):
        raise PartitionFailure(f"ENTRY_TRANSFORM:{transformed}:{unchanged}")
    if "surface_terms_flux_t_skin" not in current.files:
        raise PartitionFailure("TSKIN_TRANSPORT_NOT_CAPTURED")
    return {
        "capture_manifest_canonical_sha256": manifest["canonical_payload_sha256"],
        "capture_proof_canonical_sha256": capture_proof["canonical_payload_sha256"],
        "capture_preflight_canonical_sha256": preflight["canonical_payload_sha256"],
        "surface_proof_canonical_sha256": surface["canonical_payload_sha256"],
        "metric_provenance_canonical_sha256": provenance["canonical_payload_sha256"],
        "entry_transform_exact_closure": transformed,
        "unchanged_heat_moisture_handle_exact_closure": unchanged,
        "t_skin_transport_capture": {
            "array": "surface_terms_flux_t_skin",
            "source": "State.t_skin -> SurfaceFluxes.t_skin -> DMP_mf ts",
            "default_fallback": "ts<=0 only for standalone callers without skin",
        },
        "wrapper": tracked_wrapper_record(),
    }


def configure_frozen_driver(driver: Any, comparator: Any) -> None:
    comparator.WRF_ROOT = WRF_ROOT

    def staged_reader() -> tuple[Any, dict[str, Any]]:
        reader_module = import_file("v0234_sp2_surface_fixed_reader", AUDIT_READER)
        return reader_module.IndependentWrfDump(WRF_ROOT), {
            "path": str(AUDIT_READER),
            "sha256": EXPECTED["audit_reader"],
            "staged_root": str(WRF_ROOT),
        }

    comparator.import_dump_reader = staged_reader
    driver.load_gpt_module = lambda: comparator
    driver.POSTFIX_ROOT = CAPTURE_ROOT
    driver.POSTFIX_ARCHIVE = CAPTURE_ARCHIVE
    driver.POSTFIX_MANIFEST = CAPTURE_MANIFEST
    driver.EXPECTED_POSTFIX_ARCHIVE = EXPECTED["capture_archive"]
    driver.EXPECTED_WRF_TREE = EXPECTED["wrf_tree"]


def run_frozen_driver(driver: Any) -> int:
    RAW_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if RAW_OUTPUT.exists() or RAW_OUTPUT.is_symlink():
        raise PartitionFailure(f"RAW_OUTPUT_NOT_FRESH:{RAW_OUTPUT}")
    original_argv = sys.argv
    try:
        sys.argv = [str(FROZEN_DRIVER), "--output", str(RAW_OUTPUT)]
        return int(driver.main())
    finally:
        sys.argv = original_argv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        if output.exists() or output.is_symlink():
            raise PartitionFailure(f"OUTPUT_NOT_FRESH:{output}")
        chain = validate_chain()
        driver = import_file("v0234_sp2_surface_fixed_driver", FROZEN_DRIVER)
        comparator = import_file("v0234_sp2_surface_fixed_comparator", FROZEN_COMPARATOR)
        configure_frozen_driver(driver, comparator)
        if run_frozen_driver(driver) != 0:
            raise PartitionFailure("FROZEN_DRIVER_FAILED")
        raw_sha = sha256_file(RAW_OUTPUT)
        raw = json.loads(RAW_OUTPUT.read_text(encoding="utf-8"))
        if canonical_without_self(raw) != raw.get("canonical_payload_sha256"):
            raise PartitionFailure("RAW_COMPARATOR_CANONICAL")

        proof = dict(raw)
        proof["schema"] = "wrfgpu2-v0234-gpt-sp2-surface-fixed-partition-v1"
        proof["passed"] = True
        proof["source_authorized_surface_fix"] = {
            "changed_adapter_entry_handles": ["ustar", "rhosfc"],
            "new_wrf_gate_handle": "t_skin",
            "gate_source": "module_bl_mynnedmf.F90:5892-5918",
            "surface_handles_proven_but_not_consumed_by_pbl": ["tau_u", "tau_v"],
            "unchanged_adapter_entry_handles": ["theta_flux", "qv_flux", "fltv"],
            "surface_proof_sha256": EXPECTED["surface_proof"],
            "correction_artifact_sha256": EXPECTED["correction"],
            "capture_archive_sha256": EXPECTED["capture_archive"],
            "capture_manifest_sha256": EXPECTED["capture_manifest"],
            **chain,
        }
        proof["frozen_comparator"] = {
            "driver_sha256": EXPECTED["frozen_driver"],
            "comparator_sha256": EXPECTED["frozen_comparator"],
            "audit_reader_sha256": EXPECTED["audit_reader"],
            "raw_output_path": str(RAW_OUTPUT),
            "raw_output_sha256": raw_sha,
            "raw_canonical_payload_sha256": raw["canonical_payload_sha256"],
            "arithmetic_reimplemented_by_wrapper": False,
        }
        comparator.atomic_json(output, proof)
        print(json.dumps({
            "passed": True,
            "output": str(output),
            "authentic_rms": {
                component: proof["components"][component][
                    "authentic_postfix_adapter_vs_pristine_wrf"
                ]["rms"]
                for component in ("u", "v")
            },
            "surface_span": proof["stage_differences_postfix"]["surface_span"],
            "combined_fractions": {
                name: value["signed_projection_fraction_of_authentic_error_sse"]
                for name, value in proof["combined_vector_shapley"].items()
            },
            "canonical_payload_sha256": canonical_without_self(proof),
        }, sort_keys=True, indent=2))
        return 0
    except Exception as exc:
        print(f"REFUSE:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
