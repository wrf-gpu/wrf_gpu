from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import yaml


def test_rrtmg_fixture_generation_records_linked_harness_binary():
    if not Path("data/scratch/wrf_rrtmg_harness").exists():
        subprocess.run([sys.executable, "scripts/m5_generate_rrtmg_fixture.py"], check=True)
    manifest = yaml.safe_load(Path("fixtures/manifests/analytic-rrtmg-sw-column-v1.yaml").read_text(encoding="utf-8"))
    entries = [entry for entry in manifest["files"] if entry["path"] == "data/scratch/wrf_rrtmg_harness"]
    assert entries
    harness = Path(entries[0]["path"])
    assert harness.exists()
    assert hashlib.sha256(harness.read_bytes()).hexdigest() == entries[0]["checksum_sha256"]
    assert "module_ra_rrtmg_sw.F.o=present" in manifest["source_commit"]
    assert "module_ra_rrtmg_lw.F.o=present" in manifest["source_commit"]
    assert "full_rrtmg_driver_call=RRTMG_SWRAD+RRTMG_LWRAD" in manifest["source_commit"]
