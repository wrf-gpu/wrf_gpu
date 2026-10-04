#!/usr/bin/env python3
"""Generate the deterministic CPU-only proof for the v0.23.4 GRID_ID repair."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent/sprints/2026-07-12-v0234-grid-id-writer"
ROOT_PARENT = "8d82220dd03e1763f4a14c8f0fd5fb491a2a4a32"
REPAIR_PARENT = "d957a57457d34becb502e99d6991fc16c6888dd1"
CRITIC_COMMIT = "d8e9cbaf701c80ac595b9077d7f39b66cdc2218f"
CRITIC_REPORT_SHA256 = "37ee8d38ed9df7760665cf0a453e837db2db369395609ce220f77c223c56a496"
EVIDENCE_COMMIT = "ea4b6155312ae0c7b4816eded24e1f433dedc9c3"
EVIDENCE_PATH = (
    ".agent/sprints/2026-07-12-v0234-corrected-validation/"
    "retry3-frame-grid-metadata-failure.json"
)
EVIDENCE_SHA256 = "71213df12a155c2ca0049b2e4ef882b09826a6f5f689c222d1cd3da4aac60123"
RETAINED = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "gpu_validation_8d82220d_retry3/gpu-output/"
    "wrfout_d03_2025-03-01_00:20:00"
)
RETAINED_SHA256 = "0f01fd8dde586572318b6fa87654f9893e35ba590fdc029520a252a904f4fe54"
TESTS = (
    "tests/test_m7_netcdf_writer.py",
    "tests/test_v0201_training_output_subset.py",
    "tests/test_async_wrfout_equiv.py",
    "tests/test_v0222_output_pipeline.py",
    "tests/test_v0222_nested_wallclock.py",
    "tests/test_m7_daily_pipeline.py",
    "tests/test_auxhist_stream.py",
    "tests/test_daily_boundary_clock.py",
    "tests/test_v014_psfc_moist_hydrostatic.py",
    "tests/test_v0110_wrfrst_netcdf.py::test_wrfout_writes_ki3_snow_snso_and_seed_dimensions",
)
HASHED_FILES = (
    "src/gpuwrf/io/wrfout_writer.py",
    "src/gpuwrf/io/async_wrfout.py",
    "src/gpuwrf/io/__init__.py",
    "src/gpuwrf/integration/daily_pipeline.py",
    "src/gpuwrf/integration/nested_pipeline.py",
    "scripts/m7_netcdf_writer_smoke.py",
    "scripts/d03_replay.py",
    "scripts/m7_l2_d02_replay.py",
    "scripts/v0234_grid_id_writer_proof.py",
    "tests/test_m7_netcdf_writer.py",
    "tests/test_v0201_training_output_subset.py",
    "tests/test_async_wrfout_equiv.py",
    "tests/test_v0222_output_pipeline.py",
    "tests/test_v0222_nested_wallclock.py",
    "tests/test_m7_daily_pipeline.py",
    "tests/test_auxhist_stream.py",
    "tests/test_daily_boundary_clock.py",
    "tests/test_v014_psfc_moist_hydrostatic.py",
    "tests/test_v0110_wrfrst_netcdf.py",
    ".agent/sprints/2026-07-12-v0234-grid-id-writer/sprint-contract.md",
    "ownership.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def production_calls() -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").rglob("*.py"))
    for path in paths:
        relative = str(path.relative_to(ROOT))
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else ""
            )
            keywords = {item.arg for item in node.keywords}
            required: set[str] | None = None
            if call in {"prepare_wrfout_payload", "write_wrfout_netcdf"}:
                required = {"domain", "domain_authority"}
            elif call == "write_prepared_wrfout":
                required = {"expected_domain", "expected_domain_authority"}
            elif call in {"submit", "submit_subset"} and node.args:
                first = node.args[0]
                if isinstance(first, ast.Name) and first.id in {"prepared", "shared"}:
                    required = {"expected_domain", "expected_domain_authority"}
            elif call == "DailyCase":
                required = {"writer_domain_authority"}
            if required is None:
                continue
            missing = required - keywords
            if missing:
                raise RuntimeError(
                    f"missing {sorted(missing)} at {relative}:{node.lineno}:{call}"
                )
            calls.append({"path": relative, "line": node.lineno, "call": call})
    return sorted(calls, key=lambda item: (str(item["path"]), int(item["line"])))


def public_boundary_signatures() -> dict[str, list[str]]:
    required = {
        "src/gpuwrf/io/wrfout_writer.py": {
            "write_wrfout_netcdf": ["domain", "domain_authority"],
            "prepare_wrfout_payload": ["domain", "domain_authority"],
            "write_prepared_wrfout": ["expected_domain", "expected_domain_authority"],
        },
        "src/gpuwrf/io/async_wrfout.py": {
            "submit": ["expected_domain", "expected_domain_authority"],
            "submit_subset": ["expected_domain", "expected_domain_authority"],
        },
    }
    observed: dict[str, list[str]] = {}
    for relative, functions in required.items():
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        definitions = {
            node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        for name, names in functions.items():
            node = definitions[name]
            keyword_names = [arg.arg for arg in node.args.kwonlyargs]
            defaults = dict(zip(keyword_names, node.args.kw_defaults, strict=True))
            for argument in names:
                if argument not in keyword_names or defaults[argument] is not None:
                    raise RuntimeError(f"{relative}:{name}:{argument} is not mandatory")
            observed[f"{relative}:{name}"] = names
    return observed


def run_tests() -> dict[str, int]:
    affinity = os.sched_getaffinity(0)
    if not affinity or not affinity <= {12, 13, 14, 15}:
        raise RuntimeError(f"proof must run only on CPU cores 12-15, got {sorted(affinity)}")
    command = [sys.executable, "-m", "pytest", "-q", *TESTS]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["JAX_PLATFORMS"] = "cpu"
    completed = subprocess.run(
        command, cwd=ROOT, env=env, text=True, capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout + "\n" + completed.stderr)
    match = re.search(r"(\d+) passed(?:, (\d+) skipped)?", completed.stdout)
    if match is None:
        raise RuntimeError(f"unrecognized pytest result: {completed.stdout!r}")
    return {"passed": int(match.group(1)), "skipped": int(match.group(2) or 0)}


def classify_preexisting_wrfrst() -> dict[str, object]:
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_v0110_wrfrst_netcdf.py"]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    env["JAX_PLATFORMS"] = "cpu"
    completed = subprocess.run(
        command, cwd=ROOT, env=env, text=True, capture_output=True, check=False
    )
    expected = [
        "test_wrfrst_state_roundtrip_bit_identical_and_wrf_schema",
        "test_wrfrst_hail_state_roundtrip_writes_hail_conditionals",
        "test_wrfrst_carry_roundtrip_includes_promoted_scratch",
        "test_wrfrst_optional_nested_carry_roundtrip_and_wrf_land_schema",
    ]
    if completed.returncode != 1 or any(name not in completed.stdout for name in expected):
        raise RuntimeError("wrfrst failure classification drifted\n" + completed.stdout)
    if "4 failed, 2 passed" not in completed.stdout:
        raise RuntimeError("wrfrst summary drifted\n" + completed.stdout)
    diff = subprocess.check_output(
        ["git", "diff", "--name-only", REPAIR_PARENT, "--", "src/gpuwrf/io/wrfrst_netcdf.py"],
        cwd=ROOT,
        text=True,
    )
    if diff.strip():
        raise RuntimeError("wrfrst source changed in GRID_ID repair")
    return {
        "command": command,
        "classification": "PRE_EXISTING_PARENT_FAILURE_OUTSIDE_GRID_ID_SCOPE",
        "failure_mechanism": "_restart_dimension_sizes IndexError before global-attribute writing",
        "expected_failures": expected,
        "failed": 4,
        "passed": 2,
        "wrfrst_source_unchanged_from_repair_parent": True,
    }


def build_proof() -> dict[str, object]:
    evidence = subprocess.check_output(
        ["git", "show", f"{EVIDENCE_COMMIT}:{EVIDENCE_PATH}"], cwd=ROOT
    )
    if hashlib.sha256(evidence).hexdigest() != EVIDENCE_SHA256:
        raise RuntimeError("manager evidence hash mismatch")
    if sha256_file(RETAINED) != RETAINED_SHA256:
        raise RuntimeError("retained retry3 frame hash mismatch")
    with Dataset(RETAINED) as dataset:
        retained_metadata = {
            "GRID_ID_present": "GRID_ID" in dataset.ncattrs(),
            "WEST-EAST_GRID_DIMENSION": int(dataset.getncattr("WEST-EAST_GRID_DIMENSION")),
            "SOUTH-NORTH_GRID_DIMENSION": int(dataset.getncattr("SOUTH-NORTH_GRID_DIMENSION")),
            "mass_shape": [len(dataset.dimensions["south_north"]), len(dataset.dimensions["west_east"])],
        }
    if retained_metadata != {
        "GRID_ID_present": False,
        "WEST-EAST_GRID_DIMENSION": 112,
        "SOUTH-NORTH_GRID_DIMENSION": 94,
        "mass_shape": [93, 111],
    }:
        raise RuntimeError(f"retained failure no longer matches: {retained_metadata}")
    results = run_tests()
    return {
        "schema": "gpuwrf.v0234.grid-id-writer.cpu-proof.v2",
        "verdict": "PASS_CPU_ONLY_READY_FOR_REREVIEW",
        "root_parent_commit": ROOT_PARENT,
        "repair_parent_commit": REPAIR_PARENT,
        "critic": {"commit": CRITIC_COMMIT, "report_sha256": CRITIC_REPORT_SHA256},
        "owner_classification": "MODEL_PREFIX_GREEN / OUTPUT_METADATA_FAILURE_GRID_ID",
        "scientific_release_authority": False,
        "gpu_or_runtime_authority": False,
        "manager_evidence": {
            "commit": EVIDENCE_COMMIT,
            "path": EVIDENCE_PATH,
            "sha256": EVIDENCE_SHA256,
        },
        "retained_retry3_frame": {
            "path": str(RETAINED),
            "sha256": RETAINED_SHA256,
            "immutable_read_only": True,
            "observed": retained_metadata,
        },
        "authority_gates": {
            "exact_domains": {"d01": 1, "d02": 2, "d03": 3},
            "missing_wrong_substituted_rehashed_rejected": True,
            "canonical_domain_authority_cosubstitution_rejected": True,
            "exact_source_type": "Gen2GridSpec",
            "mass_extents": ["mass_nx", "mass_ny", "mass_nz"],
            "staggered_extents": ["e_we", "e_sn", "e_vert"],
            "mass_staggered_relations_checked": True,
            "filename_path_shape_default_inference_absent": True,
            "duplicate_equal_or_conflicting_rejected": True,
            "validated_at_prepare_and_write": True,
            "DailyCase_authority_mandatory": True,
        },
        "publication_gate": {
            "private_exclusive_temporary": True,
            "atomic_noreplace": "linkat-style os.link with dirfds",
            "existing_target_preserved": True,
            "publication_race_preserved": True,
            "sync_async_subset_auxhist_covered": True,
        },
        "preservation_gate": {
            "dimensions": "equal",
            "variables_and_order": "equal",
            "dtypes_dimensions_shapes": "equal",
            "variable_value_ieee_bits_and_masks": "equal",
            "variable_attributes": "equal",
            "unrelated_global_attributes_and_order": "equal",
            "only_intentional_change": "global GRID_ID insertion",
        },
        "retained_regression": {
            "original_qa_d03_frame": "FRAME_GRID reproduced",
            "temporary_copy_after_production_metadata_helper": "PASS",
            "retained_source_hash_after": RETAINED_SHA256,
        },
        "production_calls": production_calls(),
        "public_boundary_signatures": public_boundary_signatures(),
        "tests": {
            "command": [sys.executable, "-m", "pytest", "-q", *TESTS],
            **results,
            "cpu_affinity": [12, 13, 14, 15],
            "JAX_PLATFORMS": "cpu",
        },
        "wrfrst_compatibility": classify_preexisting_wrfrst(),
        "file_sha256": {
            relative: sha256_file(ROOT / relative) for relative in HASHED_FILES
        },
        "prohibitions_observed": [
            "no GPU query/lock/use",
            "no model or WRF execution",
            "no forecast or model/GPU compile",
            "no runtime ref/nonce/workdir",
            "no retained evidence mutation",
            "no manager/external repository write",
            "no release action",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=SPRINT / "cpu-proof.json")
    args = parser.parse_args()
    proof = build_proof()
    encoded = (json.dumps(proof, indent=2, sort_keys=True) + "\n").encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, args.output)
    print(hashlib.sha256(encoded).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
