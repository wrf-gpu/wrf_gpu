from __future__ import annotations

import json
from pathlib import Path

from gpuwrf.validation.tier2 import invariant_record, make_ideal_grid


def test_tier2_small_grid_invariants_pass():
    record = invariant_record(make_ideal_grid(4, 6, 6), n_steps=2, dt=0.25, n_acoustic=1)
    assert record["pass"] is True
    assert record["qv_positivity_violations"] == 0


def test_tier2_artifact_passes_when_present():
    path = Path("artifacts/m4/tier2_invariants.json")
    if not path.exists():
        return
    record = json.loads(path.read_text())
    assert record["pass"] is True
    assert record["mass_residual_relative"] <= 1.0e-10


def test_ideal_grid_has_expected_c_grid_shapes():
    grid = make_ideal_grid(5, 7, 9)
    assert grid.nz == 5
    assert grid.ny == 7
    assert grid.nx == 9
