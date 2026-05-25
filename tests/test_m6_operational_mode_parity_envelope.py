from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent" / "sprints" / "2026-05-25-m6-perf-design"


def test_operational_smoke_records_adr007_precision_boundary():
    proof = json.loads((SPRINT / "artifacts" / "proof_operational_smoke.json").read_text(encoding="utf-8"))

    assert proof["status"] == "PASS"
    assert proof["sanitizer"] == "not_present_in_operational_path"
    assert proof["precision"]["u"] == "float32"
    assert proof["precision"]["v"] == "float32"
    assert proof["precision"]["theta"] == "float32"
    assert proof["precision"]["w"] == "float64"
    assert proof["precision"]["p"] == "float64"
    assert proof["precision"]["ph"] == "float64"
    assert proof["precision"]["mu"] == "float64"


def test_tier4_acceptance_is_not_silently_claimed_without_golden_proof():
    status = json.loads((SPRINT / "artifacts" / "proof_acceptance_status.json").read_text(encoding="utf-8"))

    assert status["status"] == "BLOCKED"
    assert any("Tier-4 golden 1h RMSE was not run" in item for item in status["blocked_gates"])
    assert status["tier4_envelope_thresholds"] == {"T2_K": 3.0, "U10_m_s": 7.5, "V10_m_s": 7.5}


def test_solver_bakeoff_keeps_thomas_until_full_promotion_evidence_exists():
    proof = json.loads((SPRINT / "artifacts" / "proof_solver_bakeoff.json").read_text(encoding="utf-8"))

    assert proof["status"] == "PASS"
    assert proof["winner"] == "m6b2_lax_scan_thomas"
    assert proof["algorithms"]["pure_pcr"]["wall_ms_block_until_ready_real_canary_coefficients"] > 0
    assert proof["external_references"]["cusparse_gtsv"]["status"] == "not_run"
