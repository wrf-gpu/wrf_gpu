"""Gate-1 regression: physically-inactive scheme != MISSING operator.

The prior harness flagged a correctly-inactive scheme (e.g. microphysics on a
dry quiescent column, SW radiation at night) as ``MISSING``/``NOISY_ZERO`` —
the microphysics silent-failure FALSE POSITIVE recorded in
``proofs/diagnostic_harness/diagnostic_report_smoke_3step.json``.  The Gate-1
fix records a per-operator PHYSICAL OPPORTUNITY signal and only flags zero-delta
as MISSING when the operator actually had forcing to act on.

These tests exercise the host-side classifier directly with synthetic
accumulator arrays, so they are fast, deterministic, and run on CPU-only CI.
"""

from __future__ import annotations

import numpy as np

from gpuwrf.diagnostics.comprehensive_harness import (
    DIAGNOSTIC_FIELD_INDEX,
    DIAGNOSTIC_OPERATORS,
    _OP_INDEX,
    _operator_verdict,
)


_OP_COUNT = len(DIAGNOSTIC_OPERATORS)
_FIELD_COUNT = len(DIAGNOSTIC_FIELD_INDEX)


def _zero_deltas(steps: int = 3) -> tuple[np.ndarray, np.ndarray]:
    mean_abs = np.zeros((steps, _OP_COUNT, _FIELD_COUNT), dtype=np.float64)
    max_abs = np.zeros((steps, _OP_COUNT, _FIELD_COUNT), dtype=np.float64)
    return mean_abs, max_abs


def test_microphysics_zero_delta_no_opportunity_is_inactive_physical() -> None:
    """Dry quiescent column: zero microphysics delta + zero opportunity => INACTIVE_PHYSICAL."""

    mean_abs, max_abs = _zero_deltas()
    verdict, comment = _operator_verdict(
        mean_abs,
        max_abs,
        _OP_INDEX["microphysics_thompson"],
        operator_was_called=True,
        physical_opportunity_steps=0,
    )
    assert verdict == "INACTIVE_PHYSICAL", (verdict, comment)
    assert "correctly inactive" in comment


def test_microphysics_zero_delta_with_opportunity_is_missing() -> None:
    """Condensate present (opportunity > 0) but zero delta => genuine MISSING bug."""

    mean_abs, max_abs = _zero_deltas()
    verdict, comment = _operator_verdict(
        mean_abs,
        max_abs,
        _OP_INDEX["microphysics_thompson"],
        operator_was_called=True,
        physical_opportunity_steps=3,
    )
    assert verdict == "MISSING", (verdict, comment)
    assert "physical opportunity" in comment


def test_microphysics_active_when_delta_nonzero() -> None:
    """Non-zero in-scope delta => ACTIVE regardless of opportunity bookkeeping."""

    mean_abs, max_abs = _zero_deltas()
    op = _OP_INDEX["microphysics_thompson"]
    for fld in ("qv", "qc", "qr", "qi", "qs", "qg", "theta"):
        mean_abs[:, op, DIAGNOSTIC_FIELD_INDEX.index(fld)] = 1.0e-6
        max_abs[:, op, DIAGNOSTIC_FIELD_INDEX.index(fld)] = 1.0e-5
    verdict, _ = _operator_verdict(
        mean_abs, max_abs, op, operator_was_called=True, physical_opportunity_steps=3
    )
    assert verdict == "ACTIVE", verdict


def test_surface_layer_partial_zero_no_opportunity_is_inactive_physical() -> None:
    """Partial zero-delta (NOISY_ZERO candidate) with no opportunity => INACTIVE_PHYSICAL."""

    mean_abs, max_abs = _zero_deltas()
    op = _OP_INDEX["surface_layer"]
    # ustar moved but the flux fields did not — would be NOISY_ZERO if opportunity existed.
    mean_abs[:, op, DIAGNOSTIC_FIELD_INDEX.index("ustar")] = 1.0e-4
    verdict, comment = _operator_verdict(
        mean_abs, max_abs, op, operator_was_called=True, physical_opportunity_steps=0
    )
    assert verdict == "INACTIVE_PHYSICAL", (verdict, comment)


def test_passive_guard_unaffected_by_opportunity() -> None:
    """Passive guards (empty scope) keep PASSIVE_OK when they never fire."""

    mean_abs, max_abs = _zero_deltas()
    verdict, _ = _operator_verdict(
        mean_abs,
        max_abs,
        _OP_INDEX["dynamics_guards"],
        operator_was_called=True,
        physical_opportunity_steps=0,
    )
    assert verdict == "PASSIVE_OK", verdict


def test_uncalled_operator_is_inactive() -> None:
    mean_abs, max_abs = _zero_deltas()
    verdict, _ = _operator_verdict(
        mean_abs,
        max_abs,
        _OP_INDEX["rrtmg"],
        operator_was_called=False,
        physical_opportunity_steps=0,
    )
    assert verdict == "INACTIVE", verdict
