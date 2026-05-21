from __future__ import annotations

import numpy as np

from gpuwrf.physics.thompson_column import step_thompson_column
from gpuwrf.validation.tier1_thompson import load_fixture_state


def _candidate_and_reference():
    state, dt, expected = load_fixture_state()
    candidate = step_thompson_column(state, dt, debug=False)
    return candidate, expected


def _abs_err(candidate, expected, field: str, index: tuple[int, int]) -> float:
    return float(abs(np.asarray(getattr(candidate, field), dtype=np.float64)[index] - expected[field][index]))


def test_rain_evaporation_and_warm_graupel_melt_cell_matches_wrf_mass_oracle():
    candidate, expected = _candidate_and_reference()
    index = (2, 2)
    assert _abs_err(candidate, expected, "qv", index) <= 1.0e-9
    assert _abs_err(candidate, expected, "qr", index) <= 1.0e-9
    assert _abs_err(candidate, expected, "qg", index) <= 1.0e-12
    assert _abs_err(candidate, expected, "T", index) <= 1.0e-4


def test_deposition_nucleation_cell_matches_wrf_ice_number_oracle():
    candidate, expected = _candidate_and_reference()
    index = (1, 8)
    assert _abs_err(candidate, expected, "Ni", index) <= 1.0
    assert _abs_err(candidate, expected, "qi", index) <= 1.0e-8
    assert _abs_err(candidate, expected, "qc", index) <= 1.0e-8
    assert _abs_err(candidate, expected, "T", index) <= 1.0e-4


def test_tracked_number_balance_residual_is_bounded_by_carryforward_oracle():
    candidate, expected = _candidate_and_reference()
    nr_err = np.max(np.abs(np.asarray(candidate.Nr, dtype=np.float64) - expected["Nr"]))
    ni_err = np.max(np.abs(np.asarray(candidate.Ni, dtype=np.float64) - expected["Ni"]))
    assert nr_err <= 1.0e5
    assert ni_err <= 1.0e3
