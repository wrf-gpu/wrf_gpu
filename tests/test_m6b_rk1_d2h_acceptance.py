from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPRINT = ROOT / ".agent" / "sprints" / "2026-05-25-m6b-rk1-d2h-acceptance"
OPERATIONAL_MODE = ROOT / "src" / "gpuwrf" / "runtime" / "operational_mode.py"
HONEST_1H = ROOT / "scripts" / "m6b_canary_1h_honest_v2.py"


def _load(name: str) -> dict:
    return json.loads((SPRINT / name).read_text(encoding="utf-8"))


def test_operational_mode_lifts_localized_dynamic_d2h_emitters():
    source = OPERATIONAL_MODE.read_text(encoding="utf-8")

    assert "jax.lax.switch" not in source
    assert "jax.lax.cond" not in source
    assert "def _scan_forecast_segment" in source
    assert "run_radiation: bool" in source
    assert "advance_stage(carry, 1.0 / 3.0, 1)" in source
    for forbidden in (
        "device_get",
        "host_callback",
        "pure_callback",
        "io_callback",
        "sanitize_state",
        "snapshot(",
        "gpuwrf.dynamics.acoustic_loop",
        "gpuwrf.dynamics.coupled_step",
    ):
        assert forbidden not in source


def test_honest_1h_defaults_use_wrfout_rich_gen2_ids():
    source = HONEST_1H.read_text(encoding="utf-8")

    assert 'SPRINT = ROOT / ".agent" / "sprints" / "2026-05-25-m6b-rk1-d2h-acceptance"' in source
    assert "20260509_18z_l3_24h_20260511T190519Z" in source
    assert "20260521_18z_l3_24h_20260522T072630Z" in source
    assert "20260521_18z_l3_24h_20260522T133443Z" in source
    assert "20260523_18z_l3_24h_20260524T004313Z" not in source


def test_acceptance_proofs_record_rk1_d2h_and_m6b_gates_honestly():
    step1 = _load("proof_rk1_parity_step1.json")
    step10 = _load("proof_rk1_parity_step10.json")
    d2h = _load("proof_d2h_warmed_inter_kernel_zero.json")
    one_hour = _load("proof_m6b_1h_runs.json")
    tier4 = _load("proof_tier4_rmse.json")
    spatial = _load("proof_spatial_divergence.json")

    assert step1["requested_steps"] == 1
    assert step1["threshold"] == 1.0e-10
    assert "max_abs_delta" in step1
    assert step10["requested_steps"] == 10
    assert step10["threshold"] == 1.0e-8
    assert "max_abs_delta" in step10
    assert d2h["status"] == "PASS"
    assert d2h["d2h_inter_kernel"] == 0
    if step1["status"] == "PASS" and step10["status"] == "PASS":
        assert one_hour["m6_close_recommendation"] == "CLOSE-M6"
        assert tier4["status"] == "PASS"
        assert spatial["status"] == "PASS"
    else:
        assert one_hour["m6_close_recommendation"] == "BLOCKER"
        assert tier4["status"] in {"NOT_RUN", "FAIL"}
        assert spatial["status"] in {"NOT_RUN", "FAIL"}
