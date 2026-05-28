from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.m6b1_advance_mu_t_compare import SOURCE_WRFOUT, compare_tier, synthetic_dryrun


@pytest.mark.skipif(not SOURCE_WRFOUT.exists(), reason="M6B1 source wrfout is unavailable")
def test_m6b1_column_advance_mu_t_savepoint_parity(tmp_path: Path):
    result = compare_tier("column", 1, tmp_path / "savepoints")

    assert result["passed"], result["outcome"]
    step = result["results"][0]
    for name in ("mu", "mudf", "muts", "muave", "ww", "theta", "ph_tend"):
        assert step["fields"][name]["passed"], name


@pytest.mark.skipif(not SOURCE_WRFOUT.exists(), reason="M6B1 source wrfout is unavailable")
def test_m6b1_synthetic_dryrun_catches_mu_and_muts_perturbations():
    result = synthetic_dryrun()

    assert result["passed"]
    assert result["clean_self_compare_passed"]
    assert result["mu_and_muts_perturbations_caught"]
