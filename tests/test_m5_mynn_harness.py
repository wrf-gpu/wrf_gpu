from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import yaml


def test_mynn_fixture_generation_records_harness_binary():
    if not Path("data/scratch/wrf_mynn_harness").exists():
        subprocess.run([sys.executable, "scripts/m5_generate_mynn_fixture.py"], check=True)
    manifest = yaml.safe_load(Path("fixtures/manifests/analytic-mynn-pbl-column-v1.yaml").read_text(encoding="utf-8"))
    entries = [entry for entry in manifest["files"] if entry["path"] == "data/scratch/wrf_mynn_harness"]
    assert entries
    harness = Path(entries[0]["path"])
    assert harness.exists()
    assert hashlib.sha256(harness.read_bytes()).hexdigest() == entries[0]["checksum_sha256"]
    assert "wrf-mynnedmf-object-linked-harness" in manifest["source_commit"]
    assert "module_bl_mynnedmf_o=present" in manifest["source_commit"]
