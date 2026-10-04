#!/usr/bin/env python3
"""Repartition the corrected-metric SP2 capture with the frozen comparator.

This backend-dark wrapper binds the predecessor's already-reviewed five-way
NumPy comparator to the fresh CPU capture and the staged copy of the sealed
462-file pristine WRF dump.  It adds evidence-chain gates only; none of the
coefficient reconstruction, Shapley, or activity arithmetic is reimplemented.
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


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-19-v0234-gpt-sp2-residual-closure"
STAGED = Path("/tmp/v0234_gpt_sp2_residual_evidence")
CAPTURE_ROOT = STAGED / "corrected-cpu-adapter/metric-authority-v1"
CAPTURE_ARCHIVE = CAPTURE_ROOT / "single-authority-capture.npz"
CAPTURE_MANIFEST = CAPTURE_ROOT / "manifest.json"
WRF_ROOT = STAGED / "wrf-dumps"
RAW_OUTPUT = STAGED / "partition-work/metric-authority-v1-raw.json"

CAPTURE_PREFLIGHT = SPRINT / "metric-authority-capture-preflight.json"
CAPTURE_PROOF = SPRINT / "metric-authority-capture-proof.json"
METRIC_PROVENANCE = SPRINT / "metric-provenance-proof.json"
FROZEN_DRIVER = REPO / "scripts/v0234_fable_sp2_postfix_partition.py"
FROZEN_COMPARATOR = REPO / "scripts/v0234_gpt_operand_attribution.py"
AUDIT_READER = REPO / "scripts/v0234_gpt_adversarial_audit.py"

EXPECTED = {
    "capture_archive": "609a539727f027f04ed2cdbd11e44a6830034a406fecd4d15f3206fcaa73b5ef",
    "capture_manifest": "596da1e6af23389c44edd33f56b114f2bc904e4d9f6fdb5f05d9f71e75e07310",
    "capture_proof": "37b5db326a16a9c3e4e5900b7a7195c99f49413e345be2454162ae50d08a646e",
    "capture_preflight": "6ac39b6da39ef936d01aa19912c369fe24b3a840ce0cccc12ba44e4a05b34057",
    "metric_provenance": "1c0d017c05502e7793df9946ac36ffa0d7203487d2beded91527e57f9413e834",
    "frozen_driver": "e7a2aebba9fb6460d3863a8a1e03ec9240621ea19a496bc429b23416c35d3b4c",
    "frozen_comparator": "247a1dca01b7aeb35af1adcfd3eb1dc459472c04275edc7f5b2cb3f380df3244",
    "audit_reader": "49bc7b9761e2da1e807000bbbc101d05039a585b8b00f4b31c3b404b66d79d2a",
    "wrf_tree": "88e94f6a7ded154bd2b51ba890a4efa17593fb908d5f48f070a508a4b2cb645b",
    "wrfinput": "33ed2423c38be5d59b207d6619ef2386c810734fb8e5e84e46a096aef715300a",
}


class PartitionFailure(RuntimeError):
    """Fail-closed input, source, or comparator failure."""


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
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def checked_json(path: Path, expected_sha: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha:
        raise PartitionFailure(f"EVIDENCE_HASH:{path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not (
        canonical_without_self(value) == value.get("canonical_payload_sha256")
    ):
        raise PartitionFailure(f"EVIDENCE_CANONICAL:{path}")
    return value


def import_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PartitionFailure(f"IMPORT_SPEC:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_chain() -> dict[str, Any]:
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()
    if tracked:
        raise PartitionFailure(f"TRACKED_WORKTREE_NOT_CLEAN:{tracked}")

    for label, path in (
        ("capture_archive", CAPTURE_ARCHIVE),
        ("frozen_driver", FROZEN_DRIVER),
        ("frozen_comparator", FROZEN_COMPARATOR),
        ("audit_reader", AUDIT_READER),
    ):
        if path.is_symlink() or not path.is_file():
            raise PartitionFailure(f"SOURCE_NOT_REGULAR:{path}")
        actual = sha256_file(path)
        if actual != EXPECTED[label]:
            raise PartitionFailure(f"SOURCE_HASH:{label}:{actual}")

    manifest = checked_json(CAPTURE_MANIFEST, EXPECTED["capture_manifest"])
    capture_proof = checked_json(CAPTURE_PROOF, EXPECTED["capture_proof"])
    preflight = checked_json(CAPTURE_PREFLIGHT, EXPECTED["capture_preflight"])
    provenance = checked_json(METRIC_PROVENANCE, EXPECTED["metric_provenance"])
    authority = manifest.get("authority", {})
    if not (
        manifest.get("passed") is True
        and manifest.get("status") == "SINGLE_AUTHORITY_CAPTURE_COMPLETE"
        and manifest.get("single_adapter_invocation") is True
        and manifest.get("single_namespace") is True
        and manifest.get("production_operands_and_tendencies_same_invocation") is True
        and authority.get("adapter_invocations") == 1
        and authority.get("gpu_actions") == 0
        and authority.get("wrf_or_mpi_executions") == 0
        and manifest.get("archive", {}).get("sha256") == EXPECTED["capture_archive"]
        and manifest.get("input_grid", {}).get("wrfinput_d03_sha256") == EXPECTED["wrfinput"]
        and capture_proof.get("passed") is True
        and capture_proof.get("verdict") == "SINGLE_AUTHORITY_CPU_CAPTURE_GREEN"
        and capture_proof.get("manifest", {}).get("sha256") == EXPECTED["capture_manifest"]
        and preflight.get("passed") is True
        and preflight.get("grid_metric_authority", {}).get(
            "analytic_flat_fallback_forbidden"
        ) is True
        and provenance.get("passed") is True
        and provenance.get("verdict")
        == "CPU_CAPTURE_ANALYTIC_FLAT_METRIC_DIVERGENCE_PROVEN"
    ):
        raise PartitionFailure("AUTHORITY_CHAIN")
    return {
        "capture_manifest_canonical_sha256": manifest["canonical_payload_sha256"],
        "capture_proof_canonical_sha256": capture_proof["canonical_payload_sha256"],
        "capture_preflight_canonical_sha256": preflight["canonical_payload_sha256"],
        "metric_provenance_canonical_sha256": provenance["canonical_payload_sha256"],
    }


def install_staged_inputs(driver: Any, comparator: Any) -> None:
    comparator.WRF_ROOT = WRF_ROOT

    def staged_reader() -> tuple[Any, dict[str, Any]]:
        reader_module = import_file("v0234_sp2_corrected_reader", AUDIT_READER)
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
        driver = import_file("v0234_sp2_frozen_partition_driver", FROZEN_DRIVER)
        comparator = import_file("v0234_sp2_frozen_comparator", FROZEN_COMPARATOR)
        install_staged_inputs(driver, comparator)
        if run_frozen_driver(driver) != 0:
            raise PartitionFailure("FROZEN_DRIVER_FAILED")
        raw_sha = sha256_file(RAW_OUTPUT)
        raw = json.loads(RAW_OUTPUT.read_text(encoding="utf-8"))
        if canonical_without_self(raw) != raw.get("canonical_payload_sha256"):
            raise PartitionFailure("RAW_COMPARATOR_CANONICAL")

        proof = dict(raw)
        proof["schema"] = "wrfgpu2-v0234-gpt-sp2-corrected-partition-v1"
        proof["passed"] = True
        proof["metric_authority_correction"] = {
            "only_capture_change": (
                "analytic-flat GridSpec.metrics replaced by authenticated "
                "wrfinput_d03 metrics before the same one-shot adapter"
            ),
            "wrfinput_d03_sha256": EXPECTED["wrfinput"],
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
            "entry_rho_relative_rms": proof["stage_differences_postfix"][
                "pbl_entry"
            ]["rho"]["relative_rms"],
            "mass_flux_activity": proof["mass_flux_activity"]["s_aw"],
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
