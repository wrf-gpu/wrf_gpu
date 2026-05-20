from __future__ import annotations

import json
from pathlib import Path

from gpuwrf.validation.tier3 import convergence_record, run_tier3


def test_tier3_convergence_record_passes():
    record = convergence_record()
    assert record["pass"] is True
    assert record["observed_order"] >= record["expected_order"] - 0.5


def test_tier3_runner_writes_artifact(tmp_path: Path):
    out = tmp_path / "tier3.json"
    record = run_tier3(out)
    assert record["pass"] is True
    assert json.loads(out.read_text())["case"].startswith("1d-periodic")


def test_tier3_artifact_passes_when_present():
    path = Path("artifacts/m4/tier3_convergence.json")
    if not path.exists():
        return
    record = json.loads(path.read_text())
    assert record["pass"] is True
    assert len(record["errors_per_level"]) == 3
