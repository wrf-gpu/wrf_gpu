from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.m6b2_tridiag_solve_compare import SOURCE_WRFOUT, compare_tier, synthetic_dryrun


@pytest.mark.skipif(not SOURCE_WRFOUT.exists(), reason="M6B2 source wrfout is unavailable")
def test_m6b2_column_tridiag_savepoint_parity(tmp_path: Path):
    result = compare_tier("column", 1, tmp_path / "savepoints")

    assert result["passed"], result["outcome"]
    step = result["results"][0]
    assert step["fwd"]["fields"]["tri_fwd"]["passed"]
    assert step["back"]["fields"]["tri_solution"]["passed"]


@pytest.mark.skipif(not SOURCE_WRFOUT.exists(), reason="M6B2 source wrfout is unavailable")
def test_m6b2_synthetic_dryrun_catches_fwd_and_back_perturbations():
    result = synthetic_dryrun()

    assert result["passed"]
    assert result["clean_self_compare_passed"]
    assert result["fwd_and_back_perturbations_caught"]
