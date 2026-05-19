from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from gpuwrf.validation.compare_fixture import load_manifest, validate_manifest


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "fixtures/manifests/fixture-manifest-template.yaml"


def _template_data() -> dict:
    with TEMPLATE.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_template_validates() -> None:
    assert load_manifest(TEMPLATE)["fixture_id"] == "analytic-template-smoke-v1"


def test_missing_required_field_rejected() -> None:
    data = _template_data()
    del data["source_commit"]
    assert any("$.source_commit" in error for error in validate_manifest(data, TEMPLATE))


def test_top_level_tolerance_rejected() -> None:
    data = _template_data()
    data["tolerance_abs"] = 1.0
    assert any("$.tolerance_abs: unknown field" in error for error in validate_manifest(data, TEMPLATE))


def test_malformed_tier_overrides_rejected() -> None:
    data = _template_data()
    data["variables"][0]["tier_overrides"] = {"5": {"tolerance_abs": 0.0, "tolerance_rel": 0.0}}
    assert any("tier override key" in error for error in validate_manifest(data, TEMPLATE))


def test_bad_checksum_rejected() -> None:
    data = _template_data()
    data["files"] = [{"path": "data/fixtures/example.bin", "checksum_sha256": "placeholder", "bytes": 1, "external": True}]
    assert any("$.files[0].checksum_sha256" in error for error in validate_manifest(data, TEMPLATE))


def test_oversized_sample_slice_rejected(tmp_path: Path) -> None:
    oversized = tmp_path / "oversized.npy"
    oversized.write_bytes(b"0" * 100_001)
    data = _template_data()
    data["sample_slice_path"] = str(oversized)
    assert any("$.sample_slice_path" in error and "100000" in error for error in validate_manifest(data, TEMPLATE))


def test_validator_script_reports_field_path(tmp_path: Path) -> None:
    data = _template_data()
    del data["variables"][0]["shape"]
    manifest = tmp_path / "broken.yaml"
    manifest.write_text(yaml.safe_dump(data), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "scripts/validate_fixture_manifest.py", str(manifest)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.returncode != 0
    assert "$.variables[0].shape" in proc.stderr
