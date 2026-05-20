from __future__ import annotations

from gpuwrf.validation.tier1_rrtmg import run_tier1_lw, run_tier1_sw


def test_rrtmg_sw_tier1_parity_passes():
    record = run_tier1_sw()
    assert record["pass"] is True
    assert record["tolerances_met"] is True
    assert record["scenarios_tested"] == 3


def test_rrtmg_lw_tier1_parity_passes():
    record = run_tier1_lw()
    assert record["pass"] is True
    assert record["tolerances_met"] is True
    assert record["scenarios_tested"] == 3
