from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

from gpuwrf.validation.compare_fixture import validate_manifest


ROOT = Path(__file__).resolve().parents[1]


def _variable(name: str = "T", shape: list[int] | None = None) -> dict:
    return {
        "name": name,
        "units": "K",
        "shape": shape or [2],
        "staggering": "mass",
        "dtype": "float64",
        "tolerance_abs": 1.0e-6,
        "tolerance_rel": 1.0e-6,
        "tolerance_rationale": "pytest synthetic tolerance",
        "tier_overrides": None,
    }


def _manifest(tmp_path: Path, variables: list[dict] | None = None) -> dict:
    return {
        "fixture_id": "edge-case-v1",
        "source": "analytic",
        "source_commit": "test",
        "wrf_version": None,
        "scenario": "edge case validation",
        "created_utc": "2026-05-19T00:00:00Z",
        "tier": 1,
        "precision_reference": "fp64",
        "generation_command": "pytest",
        "external_uri": None,
        "sample_slice_path": None,
        "git_commit": "test",
        "license_notes": "synthetic pytest fixture",
        "variables": variables or [_variable()],
        "files": [],
    }


def _write_manifest(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _run_compare(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return subprocess.run(
        [sys.executable, "-m", "gpuwrf.validation.compare_fixture", *args],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_wrf_derived_requires_non_empty_wrf_version(tmp_path: Path) -> None:
    data = _manifest(tmp_path)
    data["source"] = "wrf-derived"
    data["wrf_version"] = ""

    errors = validate_manifest(data, tmp_path / "manifest.yaml")

    assert any("$.wrf_version" in error and "non-empty" in error for error in errors)


def test_compare_uses_manifest_sample_slice_when_reference_omitted(tmp_path: Path) -> None:
    sample = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    np.savez(sample, T=np.array([1.0, 2.0]))
    np.savez(candidate, T=np.array([1.0, 2.0]))
    data = _manifest(tmp_path)
    data["sample_slice_path"] = str(sample.relative_to(ROOT)) if sample.is_relative_to(ROOT) else None
    if data["sample_slice_path"] is None:
        sample = ROOT / "tests" / "tmp_edge_reference.npz"
        candidate = ROOT / "tests" / "tmp_edge_candidate.npz"
        np.savez(sample, T=np.array([1.0, 2.0]))
        np.savez(candidate, T=np.array([1.0, 2.0]))
        data["sample_slice_path"] = str(sample.relative_to(ROOT))
    manifest = _write_manifest(tmp_path, data)

    try:
        proc = _run_compare(["--manifest", str(manifest), "--candidate", str(candidate)])
    finally:
        for path in (ROOT / "tests" / "tmp_edge_reference.npz", ROOT / "tests" / "tmp_edge_candidate.npz"):
            path.unlink(missing_ok=True)

    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["pass"] is True


def test_tier_override_changes_failure_to_pass(tmp_path: Path) -> None:
    variable = _variable()
    variable["tier_overrides"] = {"2": {"tolerance_abs": 0.2, "tolerance_rel": 0.0}}
    manifest = _write_manifest(tmp_path, _manifest(tmp_path, [variable]))
    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    np.savez(reference, T=np.array([1.0, 1.0]))
    np.savez(candidate, T=np.array([1.0, 1.1]))

    proc = _run_compare(
        [
            "--manifest",
            str(manifest),
            "--candidate",
            str(candidate),
            "--reference",
            str(reference),
            "--tier",
            "2",
        ]
    )

    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["variables"][0]["tolerance_abs"] == 0.2


def test_multi_variable_manifest_rejects_npy_candidate(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, _manifest(tmp_path, [_variable("T"), _variable("U")]))
    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npy"
    np.savez(reference, T=np.ones(2), U=np.ones(2))
    np.save(candidate, np.ones(2))

    proc = _run_compare(["--manifest", str(manifest), "--candidate", str(candidate), "--reference", str(reference)])

    assert proc.returncode == 2
    assert ".npy input is only valid for single-variable manifests" in proc.stderr


def test_missing_candidate_variable_reports_field_name(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, _manifest(tmp_path, [_variable("T"), _variable("U")]))
    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    np.savez(reference, T=np.ones(2), U=np.ones(2))
    np.savez(candidate, T=np.ones(2))

    proc = _run_compare(["--manifest", str(manifest), "--candidate", str(candidate), "--reference", str(reference)])

    assert proc.returncode == 2
    assert "candidate missing variable U" in proc.stderr
