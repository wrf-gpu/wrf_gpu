from __future__ import annotations

from gpuwrf.validation.tier1_rrtmg import run_tier1_lw, run_tier1_sw


def test_rrtmg_sw_tier1_records_strict_fallback_result():
    record = run_tier1_sw()
    assert record["pass"] is False
    assert record["tolerances_met"] is False
    assert record["scenarios_tested"] == 3
    assert record["field_pass"]["heating_rate"] is True
    assert record["per_field_max_abs_err"]["flux_down"] > 1.0


def test_rrtmg_lw_tier1_records_strict_parity_result():
    record = run_tier1_lw()
    assert record["pass"] is True
    assert record["tolerances_met"] is True
    assert record["scenarios_tested"] == 3
    assert record["field_pass"]["heating_rate"] is True
    assert record["per_field_max_abs_err"]["flux_up"] <= 1.0
