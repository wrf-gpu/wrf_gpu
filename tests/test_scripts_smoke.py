from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_script(*args: str) -> dict:
    proc = subprocess.run([sys.executable, *args], cwd=ROOT, text=True, stdout=subprocess.PIPE)
    assert proc.returncode == 0, proc.stdout
    return json.loads(proc.stdout)


def test_gpu_sanity_check_is_graceful() -> None:
    data = run_script("scripts/gpu_sanity_check.py")
    assert data["ok"] is True
    assert "jax" in data["cuda_related_modules"]


def test_repo_status_snapshot_is_json() -> None:
    data = run_script("scripts/repo_status_snapshot.py")
    assert "ok" in data


def test_validate_adr_template() -> None:
    data = run_script("scripts/validate_adr.py", ".agent/decisions/ADR-0000-template.md")
    assert data["ok"] is True
