from __future__ import annotations

import json
from pathlib import Path

from gpuwrf.io.proof_schemas import Forecast24h, SpacetimeBudget, validate_artifact


def test_forecast_24h_artifact_validates_when_present():
    path = Path("artifacts/m6/forecast_24h_summary.json")
    if not path.exists():
        return
    data = validate_artifact(path)
    Forecast24h.validate_dict(data)
    assert data["status"] == "PASS"
    assert data["host_device_transfer_bytes_post_init"] == 0
    assert Path(data["output_manifest"]).exists()
    manifest = json.loads(Path(data["output_manifest"]).read_text())
    assert {item["lead_hours"] for item in manifest["outputs"]} == {1.0, 6.0, 12.0, 18.0, 24.0}


def test_spacetime_budget_d02_validates_when_present():
    path = Path("artifacts/m6/spacetime_budget_d02.json")
    if not path.exists():
        return
    data = SpacetimeBudget.validate_file(path)
    assert data["host_device_transfer_bytes"] == 0
    assert data["temporary_bytes_per_step"] > 0
    assert data["debug_vs_stripped_hlo_diff_bytes"] == 0
