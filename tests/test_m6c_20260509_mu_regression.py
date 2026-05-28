from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent" / "sprints" / "2026-05-26-m6c-20260509-mu-regression"


def _proof() -> dict:
    return json.loads((SPRINT / "proof_fix_validation.json").read_text(encoding="utf-8"))


def test_20260509_step2_bitwise_parity_is_pinned():
    proof = _proof()
    step2 = proof["checks"]["20260509_steps2"]["steps"]["2"]

    assert proof["status"] == "BLOCKED"
    assert proof["checks"]["20260509_steps2"]["status"] == "PASS"
    assert step2["status"] == "PASS"
    assert step2["mu_delta"] == 0.0
    assert step2["u_delta"] == 0.0
    assert step2["v_delta"] == 0.0
    assert step2["theta_delta"] == 0.0


def test_20260521_multistep_bitwise_invariant_is_preserved():
    proof = _proof()
    invariant = proof["checks"]["20260521_steps10_invariant"]

    assert invariant["status"] == "PASS"
    assert invariant["final_max_abs_delta"] == 0.0
    for step in ("2", "5", "10"):
        assert invariant["steps"][step]["status"] == "PASS"
        assert invariant["steps"][step]["mu_delta"] == 0.0
        assert invariant["steps"][step]["u_delta"] == 0.0
        assert invariant["steps"][step]["v_delta"] == 0.0
        assert invariant["steps"][step]["theta_delta"] == 0.0


def test_20260509_remaining_failure_is_step10_nonfinite_mu():
    proof = _proof()
    failed = proof["checks"]["20260509_steps10"]["steps"]["10"]
    localization = json.loads((SPRINT / "localization.json").read_text(encoding="utf-8"))

    assert proof["checks"]["20260509_steps10"]["status"] == "FAIL"
    assert failed["status"] == "FAIL"
    assert failed["largest_bad_field"] == "mu"
    assert failed["all_fields_finite"] is False
    assert localization["step_index"] == 10
    assert localization["both_candidate_initial_values_equal"] is True


def test_preservation_gates_are_recorded_green():
    proof = _proof()

    assert proof["checks"]["b6_preserved"]["passed"] is True
    assert proof["checks"]["b6_preserved"]["outcome"] == "SEVENTH-COUPLED-STEP-PARITY-ACHIEVED"
    assert proof["checks"]["tier4_20260509_guarded"]["bounds_status"] == "PASS"
    assert proof["checks"]["tier4_20260509_guarded"]["rmse_status"] == "PASS"
