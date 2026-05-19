from __future__ import annotations

import json
from pathlib import Path

from gpuwrf.validation.tier1 import run_tier1


def test_tier1_runner_passes_and_writes_artifact(tmp_path: Path):
    out = tmp_path / "tier1.json"
    record = run_tier1(out)
    assert record["pass"] is True
    assert json.loads(out.read_text())["max_abs_err"] <= 1.0e-10


def test_tier1_artifact_passes_when_present():
    path = Path("artifacts/m4/tier1_advection_parity.json")
    if not path.exists():
        return
    record = json.loads(path.read_text())
    assert record["pass"] is True
    assert record["fixture_id"] == "analytic-stencil-3d-advdiff-v1"


def test_tier1_records_operator_mismatch_honestly():
    record = run_tier1()
    assert "dycore uses 5H/3V upwind" in record["operator"]
