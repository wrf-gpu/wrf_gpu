from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _write_manifest(tmp_path: Path, variables: list[dict]) -> Path:
    manifest = {
        "fixture_id": "unit-compare-v1",
        "source": "analytic",
        "source_commit": "test",
        "wrf_version": None,
        "scenario": "unit test",
        "created_utc": "2026-05-19T00:00:00Z",
        "tier": 1,
        "precision_reference": "fp64",
        "generation_command": "pytest",
        "external_uri": None,
        "sample_slice_path": None,
        "git_commit": "test",
        "license_notes": "test data generated in pytest",
        "variables": variables,
        "files": [],
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return path


def _variable(name: str, shape: list[int], tolerance_abs: float = 1.0e-6, tolerance_rel: float = 1.0e-6) -> dict:
    return {
        "name": name,
        "units": "1",
        "shape": shape,
        "staggering": "mass",
        "dtype": "float64",
        "tolerance_abs": tolerance_abs,
        "tolerance_rel": tolerance_rel,
        "tolerance_rationale": "pytest synthetic tolerance",
        "tier_overrides": None,
    }


def _run_compare(manifest: Path, candidate: Path, reference: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "gpuwrf.validation.compare_fixture",
            "--manifest",
            str(manifest),
            "--candidate",
            str(candidate),
            "--reference",
            str(reference),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_identity_case_passes(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, [_variable("T", [2, 2])])
    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    np.savez(reference, T=np.ones((2, 2)))
    np.savez(candidate, T=np.ones((2, 2)))
    proc = _run_compare(manifest, candidate, reference)
    assert proc.returncode == 0, proc.stderr
    record = json.loads(proc.stdout)
    assert record["pass"] is True
    assert record["first_failure"] is None
    assert record["variables"][0]["max_abs_diff"] == 0.0


def test_single_variable_failure(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, [_variable("T", [2])])
    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    np.savez(reference, T=np.array([1.0, 1.0]))
    np.savez(candidate, T=np.array([1.0, 2.0]))
    proc = _run_compare(manifest, candidate, reference)
    assert proc.returncode == 1
    record = json.loads(proc.stdout)
    assert record["pass"] is False
    assert record["first_failure"] == "T"
    assert record["variables"][0]["violation_index"] == [1]


def test_shape_mismatch_returns_nonzero(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, [_variable("T", [2, 2])])
    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    np.savez(reference, T=np.ones((2, 2)))
    np.savez(candidate, T=np.ones((3, 2)))
    proc = _run_compare(manifest, candidate, reference)
    assert proc.returncode == 1
    record = json.loads(proc.stdout)
    assert record["variables"][0]["shape_ok"] is False


def test_multi_variable_one_pass_one_fail(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, [_variable("T", [2]), _variable("U", [2])])
    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    np.savez(reference, T=np.array([1.0, 1.0]), U=np.array([1.0, 1.0]))
    np.savez(candidate, T=np.array([1.0, 1.0]), U=np.array([1.0, 3.0]))
    proc = _run_compare(manifest, candidate, reference)
    assert proc.returncode == 1
    record = json.loads(proc.stdout)
    statuses = {variable["name"]: variable["pass"] for variable in record["variables"]}
    assert statuses == {"T": True, "U": False}
    assert record["first_failure"] == "U"
