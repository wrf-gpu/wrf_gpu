from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent" / "sprints" / "2026-05-25-m6b-honest-1h-canary-V3"
PINNED = [
    "20260521_18z_l3_24h_20260522T072630Z",
    "20260521_18z_l3_24h_20260522T133443Z",
    "20260509_18z_l3_24h_20260511T190519Z",
]


def _load(name: str):
    return json.loads((SPRINT / name).read_text(encoding="utf-8"))


def test_m6b_v3_uses_three_pinned_runs_and_correct_bounds():
    proof = _load("proof_1h_runs.json")

    assert proof["run_ids"] == PINNED
    assert proof["cpu_cores"] == "0-3"
    assert proof["sanitizer"] == "OFF"
    assert proof["bounds_policy"]["lower_30_levels_k"] == [200.0, 400.0]
    assert proof["bounds_policy"]["upper_14_levels_k"] == [250.0, 700.0]
    assert all(run["operational_entrypoint"] == "run_forecast_operational" for run in proof["runs"])
    assert all(run["namelist"]["acoustic_substeps"] == 10 for run in proof["runs"])


def test_m6b_v3_close_recommendation_matches_gates():
    proof = _load("proof_1h_runs.json")
    rmse = _load("proof_tier4_rmse.json")
    spatial = _load("proof_spatial_divergence.json")

    if proof["status"] == "PASS":
        assert proof["m6_close_recommendation"] == "CLOSE-M6"
        assert rmse["status"] == "PASS"
        assert rmse["fields"]["T2"]["mean_rmse"] <= 3.0
        assert rmse["fields"]["U10"]["mean_rmse"] <= 7.5
        assert rmse["fields"]["V10"]["mean_rmse"] <= 7.5
        assert spatial["status"] == "PASS"
    else:
        assert proof["m6_close_recommendation"] == "BLOCKER"
        assert proof["blocker"] in {
            "NONFINITE",
            "THETA_BOUNDS",
            "WIND_BOUNDS",
            "OPERATIONAL_SOURCE_AUDIT",
            "RMSE_ENVELOPE",
            "SPATIAL_DIVERGENCE",
        }


def test_m6b_v3_step_bounds_are_recorded_for_each_run():
    proof = _load("proof_1h_runs.json")

    for run in proof["runs"]:
        audit = run["bounds_audit"]
        assert audit["target_steps"] == 360
        assert audit["steps_checked"] >= 1
        for step in audit["per_step"]:
            assert isinstance(step["all_leaves_finite"], bool)
            assert isinstance(step["theta_lower_30_min_k"], float)
            assert isinstance(step["theta_lower_30_max_k"], float)
            assert isinstance(step["theta_upper_14_min_k"], float)
            assert isinstance(step["theta_upper_14_max_k"], float)
            assert isinstance(step["u_abs_max_m_s"], float)
            assert isinstance(step["v_abs_max_m_s"], float)
            assert isinstance(step["w_abs_max_m_s"], float)
