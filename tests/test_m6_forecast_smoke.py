from __future__ import annotations

import json
from pathlib import Path

from gpuwrf.coupling.driver import forecast_output_leads, steps_for_hours
from gpuwrf.io.proof_schemas import ForecastSmoke


def test_forecast_step_and_output_lead_helpers_are_arbitrary_length_safe():
    assert steps_for_hours(1.0, 60.0) == 60
    assert steps_for_hours(1.25, 60.0) == 75
    assert forecast_output_leads(1.0) == [1.0]
    assert forecast_output_leads(7.0) == [1.0, 6.0, 7.0]


def test_committed_1h_smoke_artifact_validates_when_present():
    path = Path("artifacts/m6/forecast_smoke_1h.json")
    if not path.exists():
        return
    data = ForecastSmoke.validate_file(path)
    assert data["status"] == "PASS"
    assert data["host_device_transfer_bytes_post_init"] == 0
    assert json.loads(path.read_text())["diagnostics"]["all_state_leaves_finite"] is True
