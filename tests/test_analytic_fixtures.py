from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from gpuwrf.validation.compare_fixture import load_manifest


ROOT = Path(__file__).resolve().parents[1]
STENCIL_MANIFEST = ROOT / "fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml"
COLUMN_MANIFEST = ROOT / "fixtures/manifests/analytic-column-thermo-v1.yaml"
STENCIL_SAMPLE = ROOT / "fixtures/samples/analytic-stencil-3d-advdiff-v1.npz"
COLUMN_SAMPLE = ROOT / "fixtures/samples/analytic-column-thermo-v1.npz"


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


def _compare(manifest: Path, candidate: Path, reference: Path) -> subprocess.CompletedProcess[str]:
    return _run(
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
        ]
    )


def test_generator_determinism(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    proc1 = _run([sys.executable, "scripts/generate_analytic_fixtures.py", "--seed", "0", "--out", str(first / "samples"), "--manifest-out", str(first / "manifests")])
    proc2 = _run([sys.executable, "scripts/generate_analytic_fixtures.py", "--seed", "0", "--out", str(second / "samples"), "--manifest-out", str(second / "manifests")])

    assert proc1.returncode == 0, proc1.stderr
    assert proc2.returncode == 0, proc2.stderr
    for name in ("analytic-stencil-3d-advdiff-v1.npz", "analytic-column-thermo-v1.npz"):
        assert (first / "samples" / name).read_bytes() == (second / "samples" / name).read_bytes()


def test_manifests_validate() -> None:
    for manifest in (STENCIL_MANIFEST, COLUMN_MANIFEST):
        proc = _run([sys.executable, "scripts/validate_fixture_manifest.py", str(manifest)])
        assert proc.returncode == 0, proc.stderr
        data = load_manifest(manifest)
        assert data["source"] == "analytic"
        assert data["wrf_version"] is None
        assert "tolerance_abs" not in data
        assert data["sample_slice_path"].startswith("fixtures/samples/")


def test_cli_round_trip_identity_passes() -> None:
    for manifest, sample in ((STENCIL_MANIFEST, STENCIL_SAMPLE), (COLUMN_MANIFEST, COLUMN_SAMPLE)):
        proc = _compare(manifest, sample, sample)
        assert proc.returncode == 0, proc.stderr
        record = json.loads(proc.stdout)
        assert record["pass"] is True
        assert record["first_failure"] is None


def test_cli_round_trip_perturbation_fails(tmp_path: Path) -> None:
    for manifest, sample in ((STENCIL_MANIFEST, STENCIL_SAMPLE), (COLUMN_MANIFEST, COLUMN_SAMPLE)):
        data = load_manifest(manifest)
        target = data["variables"][0]
        variable_name = target["name"]
        candidate = tmp_path / f"{data['fixture_id']}-mutated.npz"
        shutil.copyfile(sample, candidate)
        with np.load(sample, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        arrays[variable_name] = arrays[variable_name].copy()
        arrays[variable_name].flat[0] += 10.0 * float(target["tolerance_abs"])
        np.savez(candidate, **arrays)

        proc = _compare(manifest, candidate, sample)

        assert proc.returncode == 1
        record = json.loads(proc.stdout)
        assert record["pass"] is False
        assert record["first_failure"] == variable_name
