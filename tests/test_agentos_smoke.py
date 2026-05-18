from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_validate_agentos_passes() -> None:
    proc = subprocess.run([sys.executable, "scripts/validate_agentos.py"], cwd=ROOT, text=True, stdout=subprocess.PIPE)
    assert proc.returncode == 0, proc.stdout
    data = json.loads(proc.stdout)
    assert data["ok"] is True
    assert data["skills_checked"] == 13


def test_constitution_blocks_unreviewed_memory_changes() -> None:
    text = (ROOT / "PROJECT_CONSTITUTION.md").read_text(encoding="utf-8").lower()
    assert "physics correctness precedes speed claims" in text
    assert "patch, evidence, review" in text
