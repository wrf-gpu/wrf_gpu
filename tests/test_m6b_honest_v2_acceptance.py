from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent" / "sprints" / "2026-05-25-m6b-honest-1h-canary-RETRY"


def _load(name: str):
    return json.loads((SPRINT / name).read_text(encoding="utf-8"))


def test_m6b_retry_uses_pinned_runs_and_corrected_bounds():
    proof = _load("proof_1h_runs.json")

    assert proof["run_ids"] == [
        "20260509_18z_l3_24h_20260511T190519Z",
        "20260521_18z_l3_24h_20260522T072630Z",
        "20260523_18z_l3_24h_20260524T004313Z",
    ]
    assert proof["cpu_cores"] == "0-3"
    assert proof["sanitizer"] == "OFF"
    assert proof["bounds_policy"]["lower_30_levels_k"] == [200.0, 400.0]
    assert proof["bounds_policy"]["upper_14_levels_k"] == [250.0, 700.0]
    assert all(run["namelist"]["acoustic_substeps"] == 10 for run in proof["runs"])


def test_m6b_retry_close_recommendation_matches_gates():
    proof = _load("proof_1h_runs.json")
    rmse = _load("proof_tier4_rmse_v2.json")
    spatial = _load("proof_spatial_divergence_v2.json")

    if proof["status"] == "PASS":
        assert proof["m6_close_recommendation"] == "CLOSE-M6"
        assert proof["d2h_inheritance"]["status"] == "GO"
        assert rmse["status"] == "PASS"
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
            "D2H_INHERITANCE_MISSING",
        }


def test_m6b_retry_step_bounds_are_recorded_for_each_run():
    proof = _load("proof_1h_runs.json")

    for run in proof["runs"]:
        audit = run["bounds_audit"]
        assert audit["target_steps"] == 360
        assert audit["steps_checked"] >= 1
        for step in audit["per_step"]:
            assert isinstance(step["theta_lower_30_min_k"], float)
            assert isinstance(step["theta_lower_30_max_k"], float)
            assert isinstance(step["theta_upper_14_min_k"], float)
            assert isinstance(step["theta_upper_14_max_k"], float)
            assert isinstance(step["u_abs_max_m_s"], float)
            assert isinstance(step["v_abs_max_m_s"], float)
            assert isinstance(step["w_abs_max_m_s"], float)
