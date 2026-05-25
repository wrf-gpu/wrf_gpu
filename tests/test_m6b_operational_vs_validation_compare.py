from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent" / "sprints" / "2026-05-25-m6b-operational-composition-bisection"


def _load(name: str):
    return json.loads((SPRINT / name).read_text(encoding="utf-8"))


def test_m6b_operational_vs_validation_step_proof_localizes_first_step():
    proof = _load("proof_bisection_step_level.json")

    assert proof["status"] == "DIVERGED"
    assert proof["first_divergence_step"] == 1
    assert proof["threshold"] == 1.0e-10
    assert proof["largest_bad_field_at_step_1"] is not None


def test_m6b_operational_vs_validation_substep_proof_names_defect():
    proof = _load("proof_bisection_substep_level.json")

    assert proof["status"] == "LOCALIZED"
    assert proof["localized_defect"] == "OPERATIONAL-COMPOSITION-DEFECT-LOCALIZED-AT-RK1-ACOUSTIC-LOOP-OMISSION"
    assert proof["first_divergence_rk_stage"] == 1
    assert proof["first_divergence_acoustic_substep"] == 1
    assert proof["not_a_fix"] is True
    assert "solve_em.F:1472-1475" in proof["wrf_source_citation"]["rk1_small_steps"]
    assert "module_small_step_em.F:1102-1108" in proof["wrf_source_citation"]["advance_mu_t_updates"]


def test_m6b_operational_comparator_is_diagnostic_only():
    source = (ROOT / "scripts" / "m6b_operational_vs_validation_compare.py").read_text(encoding="utf-8")

    assert "run_forecast_operational" in source
    assert "coupled_timestep_wrf" in source
    assert "sanitize" not in source.lower()
    assert "not_a_fix" in source
