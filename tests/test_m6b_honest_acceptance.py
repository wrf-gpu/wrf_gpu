from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent" / "sprints" / "2026-05-25-m6b-honest-1h-canary"


def _load(name: str):
    return json.loads((SPRINT / name).read_text(encoding="utf-8"))


def test_m6b_honest_probe_uses_three_pinned_operational_runs():
    proof = _load("proof_operational_runs.json")

    assert len(proof["run_ids"]) >= 3
    assert proof["operational_mode"]["confirmed"] is True
    assert proof["operational_mode"]["not_validation_mode"] is True
    assert proof["operational_mode"]["sanitizer"] == "not_present_in_operational_path"
    assert all(run["operational_entrypoint"] == "run_forecast_operational" for run in proof["runs"])


def test_m6b_close_recommendation_matches_evidence():
    proof = _load("proof_operational_runs.json")
    rmse = _load("proof_tier4_rmse.json")

    if proof["status"] == "PASS":
        assert proof["m6_close_recommendation"] == "CLOSE-M6"
        assert rmse["status"] == "PASS"
        assert rmse["fields"]["T2"]["mean_rmse"] <= 3.0
        assert rmse["fields"]["U10"]["mean_rmse"] <= 7.5
        assert rmse["fields"]["V10"]["mean_rmse"] <= 7.5
    else:
        assert proof["m6_close_recommendation"] == "BLOCKER"
        assert proof["blocker"] in {"NONFINITE", "THETA_BOUNDS", "WIND_BOUNDS", "RMSE_ENVELOPE", "SPATIAL_DIVERGENCE"}
        assert rmse["status"] in {"NOT_RUN", "FAIL"}


def test_m6b_bounds_are_explicit_for_every_attempted_run():
    proof = _load("proof_operational_runs.json")

    for run in proof["runs"]:
        bounds = run["bounds"]
        assert run["all_leaves_finite"] is True
        assert isinstance(bounds["theta_min_k"], float)
        assert isinstance(bounds["theta_max_k"], float)
        assert isinstance(bounds["u_abs_max_m_s"], float)
        assert isinstance(bounds["v_abs_max_m_s"], float)
        assert isinstance(bounds["w_abs_max_m_s"], float)
