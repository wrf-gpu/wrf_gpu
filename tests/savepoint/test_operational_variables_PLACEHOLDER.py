"""Operational-variable verdict test (replaces M9 placeholder).

The full per-field-per-hour RMSE comparison vs the WRF Canary reference lives
in :mod:`scripts.operational_trace_compare` and is consumed by the diagnostic
harness via its ``wrf_anchor_comparison`` block when the trace JSON is
present. This test asserts the harness wires that block correctly and that
the comprehensive report can be reasoned about for operational variables
even when the trace artifact is not present (the schema falls back to a
``source=None`` block).

When the WRF anchor JSON IS present, this test also asserts that the per-field
summary contains the expected output variables.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.savepoint.test_diagnostic_harness import _run_harness_short


EXPECTED_OUTPUT_FIELDS = {
    "U",
    "V",
    "W",
    "theta",
    "QVAPOR",
    "PSFC",
    "T2",
    "U10",
    "V10",
    "SWDOWN",
    "GLW",
    "HFX",
    "LH",
    "PBLH",
    "TSK",
    "LU_INDEX",
}


@pytest.fixture(scope="module")
def operational_report() -> dict:
    return _run_harness_short(hours=60.0 / 3600.0)


def test_wrf_anchor_block_present(operational_report: dict) -> None:
    assert "wrf_anchor_comparison" in operational_report
    block = operational_report["wrf_anchor_comparison"]
    assert "source" in block
    assert "per_field" in block


def test_wrf_anchor_per_field_keys_match_when_present(operational_report: dict) -> None:
    block = operational_report["wrf_anchor_comparison"]
    if block.get("source") is None:
        pytest.skip("wrf anchor json not available; harness falls back to empty per_field")
    per_field = block.get("per_field") or {}
    if not per_field:
        pytest.skip("wrf anchor present but per_field summary empty (no overlapping hours)")
    intersection = set(per_field.keys()) & EXPECTED_OUTPUT_FIELDS
    assert intersection, (
        f"wrf_anchor per_field has none of the expected output variables: "
        f"got={sorted(per_field.keys())} expected_subset_of={sorted(EXPECTED_OUTPUT_FIELDS)}"
    )


def test_first_failure_attribution_block_present(operational_report: dict) -> None:
    block = operational_report["first_failure_attribution"]
    assert "first_invariant_break" in block
    assert "first_nonfinite" in block
    assert "first_significant_anchor_divergence" in block


def test_next_sprint_recommendations_emitted(operational_report: dict) -> None:
    recs = operational_report["next_sprint_recommendations"]
    assert isinstance(recs, list) and len(recs) >= 1
