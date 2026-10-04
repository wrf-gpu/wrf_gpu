#!/usr/bin/env python3
"""Admit and dispatch the frozen exact V0234 MYNN SP2 comparator.

This wrapper adds no scientific algebra.  It independently validates honest
28/28 reference authority, emits schema adapters bound to the current contract,
and invokes the exact prior comparator only by its frozen source SHA-256.
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

import numpy as np


REPO = Path(__file__).resolve().parent.parent
SPRINT = REPO / ".agent/sprints/2026-07-19-v0234-pristine-pbl-entry-closure-gpt"
AUTHORITY_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v0234_pristine_pbl_entry_closure_ac6712170cbe5084")
COMPARATOR_ROOT = AUTHORITY_ROOT / "comparator"
REFERENCE_MANIFEST = AUTHORITY_ROOT / "cpu-reference/pbl-sp2-reference-manifest.json"
REFERENCE_ARCHIVE = AUTHORITY_ROOT / "cpu-reference/pbl-sp2-reference-28.npz"
CONTRACT_COMMIT = "5bbfe0416979eb41dd72bf80a8539191740f8eea"
COORDINATION_COMMIT = "8231225573683dc56828caba2be3326508bdfd73"
LEGACY_CONTRACT_ID = "2026-07-19-v0234-mynn-sp2-input-provenance-gpt"
COMPARATOR = Path(
    "<USER_HOME>/src/wrf_gpu2_wt/v0234-mynn-sp2-input-provenance/"
    "scripts/v0234_mynn_sp2_compare.py"
)
COMPARATOR_SHA256 = "0eb3000d28783c0f9a259830dc88ff566d544a3fe7c69e4064e71a4fab67881b"
PYTHON = Path("<USER_HOME>/miniconda3/bin/python3.13")
CARRY_SHA256 = "224aa04ece14b31abfb602613e3e0b1d77968ecd0b59ce318775a0cfa6bef58d"
FIXED_ROWS_SHA256 = "550178b158a849c5da55cee31b82e6c90ebee733620d141ac278c064764cf33a"
DUMP_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v0234_mynn_sp2_d03_horizon55_gpt_fresh01/evidence-dumps-fresh-d03-runtime")

REFERENCE_SCRIPT = REPO / "scripts/v0234_pristine_pbl_reference.py"
SPEC = importlib.util.spec_from_file_location("v0234_reference_for_comparator", REFERENCE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("reference validator import unavailable")
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


class AdmissionError(RuntimeError):
    pass


def assert_backend_dark() -> None:
    forbidden = sorted(name for name in sys.modules if name == "jax" or name.startswith(("jax.", "gpuwrf.")))
    if forbidden:
        raise AdmissionError(f"BACKEND_IMPORTED:{forbidden}")


def canonical_without_self(value: dict[str, Any]) -> str:
    return reference.canonical({key: item for key, item in value.items() if key != "canonical_payload_sha256"})


def write_adapter(path: Path, value: dict[str, Any]) -> None:
    payload = dict(value)
    payload["canonical_payload_sha256"] = canonical_without_self(payload)
    reference.atomic_json(path, payload)


def comparator_authority() -> dict[str, Any]:
    return reference.require_file(COMPARATOR, COMPARATOR_SHA256)


def build_adapters(manifest_path: Path, archive_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    validation = reference.validate_reference(manifest_path, archive_path)
    manifest, _record = reference.load_canonical(manifest_path)
    if validation["array_count"] != 28 or manifest.get("comparator_dispatch_ready") is not True:
        raise AdmissionError("REFERENCE_BUNDLE_REQUIRED")
    reference.dump_authority(deep=True)
    with np.load(archive_path, allow_pickle=False) as arrays:
        declared = {
            name: {"shape": list(arrays[name].shape), "dtype": str(arrays[name].dtype)}
            for name in arrays.files
        }
    run_proof = {
        "schema": "wrfgpu2-mynn-sp2-run-proof-v1",
        "contract_id": LEGACY_CONTRACT_ID,
        "verdict": "WRF_DISCRIMINATOR_CAPTURED_COMPARISON_PENDING",
        "adapter_role": "current-contract admission of frozen output-neutral six-rank dump",
        "current_contract_commit": CONTRACT_COMMIT,
        "coordination_commit": COORDINATION_COMMIT,
        "dump_tree_sha256": reference.DUMP_TREE_SHA256,
        "dump_admission_sha256": reference.DUMP_ADMISSION_SHA256,
        "scientific_falsification_before_comparator": False,
        "wrf_or_mpi_execution_this_sprint": 0,
    }
    reference_adapter = {
        "schema": "wrfgpu2-mynn-sp2-gpu-reference-v1",
        "contract_id": LEGACY_CONTRACT_ID,
        "carry_sha256": CARRY_SHA256,
        "fixed_rows_sha256": FIXED_ROWS_SHA256,
        "wrf_dump_tree_sha256": reference.DUMP_TREE_SHA256,
        "backend": "cpu",
        "gpu_actions": 0,
        "synthetic_values": False,
        "gpu_source_sha256": reference.SOURCE_SHA256,
        "bundle_path": str(archive_path),
        "bundle_sha256": reference.sha256_file(archive_path),
        "arrays": declared,
        "current_reference_manifest_sha256": reference.sha256_file(manifest_path),
        "current_contract_commit": CONTRACT_COMMIT,
    }
    return run_proof, reference_adapter


def admission(manifest_path: Path, archive_path: Path, *, partial_expected: bool = False) -> dict[str, Any]:
    assert_backend_dark()
    comparator = comparator_authority()
    resource = reference.resource_gate(cpu_backend=False)
    if partial_expected:
        validation = reference.validate_reference(manifest_path, archive_path, allow_partial=True)
        return {
            "schema": "wrfgpu2-v0234-pristine-pbl-comparator-admission-v1",
            "verdict": "REFERENCE_BUNDLE_REQUIRED_HOLD_5_OF_28",
            "reference_validation": validation,
            "comparator": comparator,
            "dispatch_ready": False,
            "resource_gate": resource,
            "backend_imported": False,
            "gpu_action_or_query": False,
        }
    run_proof, adapter = build_adapters(manifest_path, archive_path)
    return {
        "schema": "wrfgpu2-v0234-pristine-pbl-comparator-admission-v1",
        "verdict": "EXACT_COMPARATOR_ADMITTED",
        "run_proof_adapter_canonical_sha256": canonical_without_self(run_proof),
        "reference_adapter_canonical_sha256": canonical_without_self(adapter),
        "comparator": comparator,
        "dispatch_ready": True,
        "resource_gate": resource,
        "backend_imported": False,
        "gpu_action_or_query": False,
    }


def dispatch(manifest_path: Path, archive_path: Path) -> dict[str, Any]:
    admitted = admission(manifest_path, archive_path)
    outputs = {
        "run": COMPARATOR_ROOT / "run-proof-adapter.json",
        "reference": COMPARATOR_ROOT / "reference-adapter.json",
        "comparison": COMPARATOR_ROOT / "scientific-comparison.json",
        "terminal": COMPARATOR_ROOT / "comparison-terminal.json",
    }
    if any(path.exists() or path.is_symlink() for path in outputs.values()):
        raise AdmissionError("COMPARATOR_OUTPUT_NOT_FRESH")
    run_proof, adapter = build_adapters(manifest_path, archive_path)
    write_adapter(outputs["run"], run_proof)
    write_adapter(outputs["reference"], adapter)
    command = [
        str(PYTHON), str(COMPARATOR),
        "--run-proof", str(outputs["run"]),
        "--run-proof-sha256", reference.sha256_file(outputs["run"]),
        "--dump-root", str(DUMP_ROOT),
        "--reference-manifest", str(outputs["reference"]),
        "--reference-manifest-sha256", reference.sha256_file(outputs["reference"]),
        "--output", str(outputs["comparison"]),
    ]
    result = subprocess.run(
        command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env={
            "HOME": os.environ.get("HOME", "<USER_HOME>"),
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "LANG": "C.UTF-8", "PYTHONPATH": str(REPO),
            **reference.THREAD_ENV, "CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu",
        },
    )
    if result.returncode != 0:
        raise AdmissionError(f"EXACT_COMPARATOR_FAILED:{result.returncode}:{result.stderr}")
    comparison, comparison_record = reference.load_canonical(outputs["comparison"])
    terminal = {
        "schema": "wrfgpu2-v0234-pristine-pbl-comparison-terminal-v1",
        "contract_commit": CONTRACT_COMMIT,
        "coordination_commit": COORDINATION_COMMIT,
        "verdict": comparison.get("verdict"),
        "exact_comparator": comparator_authority(),
        "comparison": comparison_record,
        "reference_manifest_sha256": reference.sha256_file(manifest_path),
        "reference_archive_sha256": reference.sha256_file(archive_path),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": command,
        "wrf_or_mpi_executions": 0,
        "gpu_actions": 0,
        "admission": admitted,
    }
    reference.atomic_json(outputs["terminal"], terminal)
    return {
        "verdict": comparison.get("verdict"),
        "comparison_sha256": comparison_record["sha256"],
        "terminal_sha256": reference.sha256_file(outputs["terminal"]),
        "gpu_actions": 0,
        "wrf_or_mpi_executions": 0,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    modes = value.add_mutually_exclusive_group(required=True)
    modes.add_argument("--admit", action="store_true")
    modes.add_argument("--dispatch", action="store_true")
    value.add_argument("--manifest", type=Path, default=REFERENCE_MANIFEST)
    value.add_argument("--archive", type=Path, default=REFERENCE_ARCHIVE)
    value.add_argument("--partial-expected", action="store_true")
    value.add_argument("--output", type=Path)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        manifest = args.manifest.resolve()
        archive = args.archive.resolve()
        result = (
            admission(manifest, archive, partial_expected=args.partial_expected)
            if args.admit else dispatch(manifest, archive)
        )
        if args.output is not None:
            reference.atomic_json(args.output.resolve(), result)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (AdmissionError, reference.ReferenceError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"REFUSE:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
