from __future__ import annotations

from gpuwrf.validation.tier1_thompson import run_tier1


def test_m5_thompson_tier1_parity_passes():
    record = run_tier1()
    assert record["pass"] is True
    assert record["tolerances_met"] is True
    assert record["scenarios_tested"] >= 3
