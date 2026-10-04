"""The Step 1 mechanism object must stay honest about what it does NOT show.

The capture it mines is the WARM Stage 2 process. Its attractive-looking family
shares (codegen ~70%, autotune ~3%) say nothing about the cold >601.4 s compile,
because a warm process skips exactly the work H1 is about. The manager's
standing instruction is that the two compile numbers must not be equated, and
the easiest way to violate it is to quietly let a warm measurement stand in as a
cold one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "v025") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "v025"))

OBJECT = REPO / "proofs/v025/m0/step1_mechanism_evidence.json"

pytestmark = pytest.mark.skipif(not OBJECT.exists(), reason="mechanism object not built")


def _obj() -> dict:
    return json.loads(OBJECT.read_text())


# --------------------------------------------------------------------------- #
# scoping                                                                      #
# --------------------------------------------------------------------------- #
def test_the_attribution_declares_itself_warm_and_not_cold():
    scope = _obj()["exclusive_attribution"]["SCOPE"]
    assert "WARM" in scope
    assert "601.4" in scope
    assert "not be equated" in scope.lower() or "NOT evidence" in scope


def test_the_warm_shares_are_not_claimed_to_bear_on_h1():
    bearing = _obj()["exclusive_attribution"]["bearing_on_H1"]
    assert bearing.startswith("NONE")
    assert "neither supports nor" in bearing


def test_the_autotune_span_is_not_reported_as_an_autotune_duration():
    """222 s of span is 26 s of writes around a 196 s gap; the gap is unattributed."""
    timing = _obj()["autotune_cache_write_timing"]
    assert timing["status"] == "MEASURED"
    assert "NOT a measure of autotune duration" in timing["interpretation"]
    assert "UNATTRIBUTED" in timing["interpretation"]
    assert timing["largest_gap_with_no_writes_seconds"] > 0.5 * timing["span_seconds"], (
        "the gap dominates the span, which is the whole reason the span is not a duration"
    )


def test_the_gap_content_is_left_open_rather_than_guessed():
    interpretation = _obj()["autotune_cache_write_timing"]["interpretation"]
    assert "neither is claimed" in interpretation


# --------------------------------------------------------------------------- #
# the mechanism actually ran                                                   #
# --------------------------------------------------------------------------- #
def test_the_required_xla_flags_were_verified_present():
    preflight = _obj()["xla_flag_preflight"]
    assert preflight["status"] == "OK"
    assert preflight["flags"]["xla_gpu_autotune_level"] is True
    assert all(preflight["flags"].values())


def test_the_attribution_ran_on_the_real_capture_not_a_fixture():
    attribution = _obj()["exclusive_attribution"]
    assert attribution["ranges_parsed"] > 100_000
    assert attribution["source_nsys_rep_sha256"]


def test_the_attribution_meets_the_pre_registered_bar():
    attribution = _obj()["exclusive_attribution"]
    assert attribution["status"] == "OK"
    assert attribution["unknown_share"] <= attribution["max_unknown_share"]


def test_the_inclusive_contrast_is_recorded():
    """The reason exclusive time is used at all, kept as a number in the object."""
    attribution = _obj()["exclusive_attribution"]
    assert attribution["inclusive_inflation_factor"] > 1.5


def test_thunk_execution_is_excluded_from_the_compile_total():
    attribution = _obj()["exclusive_attribution"]
    assert "runtime" in attribution["non_compile_seconds"]
    assert "runtime" not in attribution["family_share_of_compile"]


def test_the_object_records_no_gpu_was_touched():
    assert "No GPU was requested" in _obj()["purpose"]


def test_the_h1_rule_is_carried_with_its_feasibility_argument():
    rule = _obj()["h1_rule"]
    assert rule["t_off_budget_seconds"] <= rule["t_on_lower_bound_seconds"]
    assert "never required to finish" in rule["why_feasible"]
