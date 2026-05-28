from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.m6b3_scratch_state_compare import COMPARE_FIELDS, SOURCE_WRFOUT, compare_tier, synthetic_dryrun


@pytest.mark.skipif(not SOURCE_WRFOUT.exists(), reason="M6B3 source wrfout is unavailable")
def test_m6b3_column_scratch_state_savepoint_parity(tmp_path: Path):
    result = compare_tier("column", 1, tmp_path / "savepoints")

    assert result["passed"], result["outcome"]
    step = result["results"][0]
    for boundary in step["boundaries"].values():
        assert boundary["passed"]
    observed = {
        name
        for boundary in step["boundaries"].values()
        for name in boundary["fields"]
    }
    assert set(COMPARE_FIELDS) == observed


@pytest.mark.skipif(not SOURCE_WRFOUT.exists(), reason="M6B3 source wrfout is unavailable")
def test_m6b3_synthetic_dryrun_catches_scratch_perturbations():
    result = synthetic_dryrun()

    assert result["passed"]
    assert result["clean_self_compare_passed"]
    assert result["scratch_perturbations_caught"]
