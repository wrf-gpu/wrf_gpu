"""Stochastic physics + two-way feedback / batched members are refused (critic FIX v50-stoch finding 4)."""
import copy

import pytest

from gpuwrf.io.namelist_check import UnsupportedSchemeError, validate_operational_namelist
from gpuwrf.io.stochastic_config import stochastic_nesting_refusals

TWO = {"stoch": {"skebs": 1}, "domains": {"max_dom": 2, "parent_id": [1, 1], "e_vert": [45, 45], "feedback": 1},
       "time_control": {"start_hour": [0, 0]}}


def test_helper_refuses_feedback_and_batches_only_when_active():
    names = ("d01", "d02")
    assert stochastic_nesting_refusals(TWO, names, feedback=False, batch_size=1) == []
    assert any("feedback" in e for e in stochastic_nesting_refusals(TWO, names, feedback=True, batch_size=1))
    assert any("B>1" in e for e in stochastic_nesting_refusals(TWO, names, feedback=False, batch_size=2))
    assert any("B>1" in e for e in stochastic_nesting_refusals(TWO, ("d01",), feedback=True, batch_size=2))
    assert stochastic_nesting_refusals(TWO, ("d01",), feedback=True, batch_size=1) == []   # no nest, no feedback
    off = copy.deepcopy(TWO); off["stoch"] = {"skebs": 0}
    assert stochastic_nesting_refusals(off, names, feedback=True, batch_size=4) == []


def test_namelist_admission_refuses_two_domain_skebs_with_feedback():
    with pytest.raises(UnsupportedSchemeError, match="two-way nest feedback"):
        validate_operational_namelist(copy.deepcopy(TWO))
    one_way = copy.deepcopy(TWO); one_way["domains"]["feedback"] = 0
    try:
        validate_operational_namelist(one_way)
    except UnsupportedSchemeError as exc:   # other catalog gaps are fine; this refusal must be absent
        assert "two-way nest feedback" not in str(exc)


def test_binding_report_refuses_feedback():
    from gpuwrf.io.namelist_binding import namelist_honour_report
    errors, _warnings = namelist_honour_report(copy.deepcopy(TWO), ("d01", "d02"))
    assert any("two-way nest feedback" in e for e in errors)
