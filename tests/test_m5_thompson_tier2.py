from __future__ import annotations

from gpuwrf.validation.tier2_thompson import invariant_record


def test_m5_thompson_tier2_invariants_pass():
    record = invariant_record(n_steps=10)
    assert record["pass"] is True
    assert record["positivity"]["violations"] == 0
    assert record["nan_inf"]["violations"] == 0
    assert record["water_budget"]["relative_residual"] <= 1.0e-8
