from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_transfer_audit_artifact_is_zero_post_init():
    audit_path = ROOT / "artifacts" / "m3" / "transfer_audit.json"
    assert audit_path.exists()
    audit = json.loads(audit_path.read_text())

    assert audit["host_to_device_bytes_post_init"] == 0
    assert audit["device_to_host_bytes_post_init"] == 0
    assert audit["iterations"] >= 1000


def test_spacetime_budget_artifact_has_required_m3_keys():
    budget_path = ROOT / "artifacts" / "m3" / "spacetime_budget.json"
    assert budget_path.exists()
    budget = json.loads(budget_path.read_text())

    for key in (
        "state_bytes",
        "tendency_bytes",
        "temporary_bytes_per_step",
        "total_persistent_bytes",
        "kernel_launches_per_step",
        "wall_time_per_step_us",
    ):
        assert key in budget
    assert budget["temporary_bytes_per_step"] == 0
    assert budget["kernel_launches_per_step"] <= 5
