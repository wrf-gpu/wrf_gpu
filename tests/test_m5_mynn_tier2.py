from __future__ import annotations

from gpuwrf.validation.tier2_mynn import independent_budget_record, invariant_record


def test_mynn_tier2_invariants_pass():
    record = invariant_record(n_steps=10)
    assert record["pass"] is True
    assert record["positivity"]["violations"] == 0
    assert record["nan_inf"]["violations"] == 0
    assert record["momentum_budget"]["pass"] is True
    assert record["tke_budget"]["pass"] is True


def test_mynn_tier2_independent_wrf_budget_passes():
    record = independent_budget_record()
    assert record["pass"] is True
    assert record["oracle"].startswith("WRF module_bl_mynnedmf mynn_tendencies")
    for field in ("u", "v", "theta", "qv"):
        assert record["field_pass"][field] is True
        assert record["per_field_max_abs_residual"][field] <= record["tolerance_abs"]
