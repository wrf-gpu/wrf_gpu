from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent" / "sprints" / "2026-05-25-m6b-fix-carry-expansion"


def _load(name: str):
    return json.loads((SPRINT / name).read_text(encoding="utf-8"))


def test_m6b_carry_expansion_probe_passes_three_pinned_runs():
    proof = _load("proof_10s_probe.json")

    assert proof["status"] == "PASS"
    assert len(proof["run_ids"]) == 3
    assert proof["source_audit"]["status"] == "PASS"
    assert proof["source_audit"]["validation_mode_imports_absent"] is True
    assert proof["blocker"] is None


def test_m6b_carry_expansion_bounds_are_recorded_per_run():
    proof = _load("proof_10s_probe.json")

    for run in proof["runs"]:
        assert run["status"] == "PASS"
        assert run["all_leaves_finite"] is True
        bounds = run["bounds"]
        assert bounds["theta_bounded"] is True
        assert bounds["wind_bounded"] is True
        assert bounds["theta_levels_checked"] == [0, 30]
        assert 200.0 < bounds["theta_min_k"] < bounds["theta_max_k"] < 400.0
        assert bounds["u_abs_max_m_s"] <= 100.0
        assert bounds["v_abs_max_m_s"] <= 100.0
        assert bounds["w_abs_max_m_s"] <= 50.0


def test_m6b_promoted_operational_carry_is_documented():
    source = (ROOT / "src" / "gpuwrf" / "runtime" / "operational_state.py").read_text(encoding="utf-8")

    for field in ("t_2ave", "ww", "muave", "muts", "ph_tend", "_save"):
        assert field in source
    assert "M6b proof_bounds.json" in source
