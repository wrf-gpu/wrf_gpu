from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import yaml

from gpuwrf.fixtures.wrf_slice import SOURCE_WRFOUT, sha256_file
from gpuwrf.validation.compare_fixture import load_manifest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "fixtures/manifests/canary-wrf-d01-20260518T13-tslice-v1.yaml"
SAMPLE = ROOT / "fixtures/samples/canary-wrf-d01-20260518T13-tslice-v1.npz"


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return env


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        env=_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _compare(candidate: Path, reference: Path = SAMPLE) -> subprocess.CompletedProcess[str]:
    return _run(
        [
            sys.executable,
            "-m",
            "gpuwrf.validation.compare_fixture",
            "--manifest",
            str(MANIFEST),
            "--candidate",
            str(candidate),
            "--reference",
            str(reference),
        ]
    )


def test_source_wrfout_exists_readable_and_is_not_touched() -> None:
    before_stat = SOURCE_WRFOUT.stat()
    before_hash = sha256_file(SOURCE_WRFOUT)

    assert SOURCE_WRFOUT.is_file()
    assert os.access(SOURCE_WRFOUT, os.R_OK)

    after_stat = SOURCE_WRFOUT.stat()
    after_hash = sha256_file(SOURCE_WRFOUT)
    assert after_hash == before_hash
    assert after_stat.st_mode == before_stat.st_mode
    assert after_stat.st_size == before_stat.st_size
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns


def test_manifest_validates_and_has_wrf_version() -> None:
    proc = _run([sys.executable, "scripts/validate_fixture_manifest.py", str(MANIFEST)])

    assert proc.returncode == 0, proc.stderr
    data = load_manifest(MANIFEST)
    assert data["source"] == "wrf-derived"
    assert data["wrf_version"] == "4.7.1"
    assert str(SOURCE_WRFOUT) in data["source_commit"]
    assert "sha256=" in data["source_commit"]


def test_sample_file_loads_and_shapes_match_manifest() -> None:
    data = load_manifest(MANIFEST)
    with np.load(SAMPLE, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}

    for variable in data["variables"]:
        name = variable["name"]
        assert name in arrays
        assert list(arrays[name].shape) == variable["shape"]
        assert str(arrays[name].dtype) == variable["dtype"]


def test_sample_size_is_bounded() -> None:
    assert SAMPLE.stat().st_size <= 100_000


def test_full_external_file_exists_at_external_uri() -> None:
    data = load_manifest(MANIFEST)
    full = ROOT / data["external_uri"]

    assert full.is_file()
    with np.load(full, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    assert sorted(arrays) == sorted(variable["name"] for variable in data["variables"])
    assert all(array.dtype == np.float64 for array in arrays.values())


def test_cli_round_trip_identity_passes() -> None:
    proc = _compare(SAMPLE)

    assert proc.returncode == 0, proc.stderr
    record = json.loads(proc.stdout)
    assert record["pass"] is True
    assert record["first_failure"] is None


def test_cli_round_trip_perturbation_fails(tmp_path: Path) -> None:
    data = load_manifest(MANIFEST)
    target = next(variable for variable in data["variables"] if variable["name"] == "T")
    candidate = tmp_path / "canary-mutated.npz"
    shutil.copyfile(SAMPLE, candidate)
    with np.load(SAMPLE, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    arrays["T"] = arrays["T"].copy()
    arrays["T"].flat[0] += np.float32(10.0 * float(target["tolerance_abs"]))
    np.savez(candidate, **arrays)

    proc = _compare(candidate)

    assert proc.returncode == 1
    record = json.loads(proc.stdout)
    assert record["pass"] is False
    assert record["first_failure"] == "T"


def test_wrf_version_validator_parity_rejects_empty_value(tmp_path: Path) -> None:
    data = load_manifest(MANIFEST)
    data["wrf_version"] = ""
    manifest = tmp_path / "empty-wrf-version.yaml"
    manifest.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    proc = _run([sys.executable, "scripts/validate_fixture_manifest.py", str(manifest)])

    assert proc.returncode == 1
    assert "$.wrf_version: required non-empty string when source is wrf-derived" in proc.stderr
