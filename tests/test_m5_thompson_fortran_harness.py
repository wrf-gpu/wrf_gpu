from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def test_fortran_harness_binary_matches_manifest_after_fixture_generation():
    manifest = yaml.safe_load(Path("fixtures/manifests/analytic-thompson-column-v1.yaml").read_text(encoding="utf-8"))
    harness_entries = [entry for entry in manifest["files"] if entry["path"] == "data/scratch/wrf_thompson_harness"]
    assert harness_entries, "fixture manifest must record the external WRF harness binary"
    harness = Path(harness_entries[0]["path"])
    assert harness.exists()
    assert hashlib.sha256(harness.read_bytes()).hexdigest() == harness_entries[0]["checksum_sha256"]


def test_manifest_records_fortran_harness_source():
    manifest = yaml.safe_load(Path("fixtures/manifests/analytic-thompson-column-v1.yaml").read_text(encoding="utf-8"))
    assert manifest["source"] == "wrf-derived"
    assert "wrf-thompson-via-fortran-harness" in manifest["source_commit"]
    assert manifest["wrf_version"] == "v4.7.1"


def test_agent_success_notes_attempt3_harness_when_present():
    path = Path("artifacts/m5/agent_success.json")
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["sprint_attempts"] >= 3
    assert "Fortran harness" in payload["notes"]
