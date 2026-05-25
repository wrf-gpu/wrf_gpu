from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.m6b4_acoustic_recurrence_compare import COMPARE_FIELDS, SOURCE_WRFOUT, compare_tier, synthetic_dryrun
from gpuwrf.validation.savepoint_schema import SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS, VALID_BOUNDARIES, VALID_OPERATORS


def test_m6b4_schema_v5_supports_acoustic_recurrence_boundaries() -> None:
    assert SCHEMA_VERSION == "m6b4-savepoint-v5"
    assert "m6b4-savepoint-v5" in SUPPORTED_SCHEMA_VERSIONS
    assert "acoustic_substep_complete" in VALID_BOUNDARIES
    assert "acoustic_loop_complete" in VALID_BOUNDARIES
    assert "acoustic_recurrence" in VALID_OPERATORS


@pytest.mark.skipif(not SOURCE_WRFOUT.exists(), reason="M6B4 source wrfout is unavailable")
def test_m6b4_column_acoustic_recurrence_parity_one_substep(tmp_path: Path) -> None:
    result = compare_tier("column", 1, tmp_path / "savepoints")

    assert result["passed"], result["outcome"]
    assert result["savepoint_count"] == 2
    step = result["results"][0]
    assert step["passed"]
    assert result["loop_result"]["passed"]
    assert set(COMPARE_FIELDS) == set(step["fields"])


@pytest.mark.skipif(not SOURCE_WRFOUT.exists(), reason="M6B4 source wrfout is unavailable")
def test_m6b4_synthetic_dryrun_catches_boundary_perturbations() -> None:
    result = synthetic_dryrun()

    assert result["passed"]
    assert result["clean_self_compare_passed"]
    assert result["boundary_field_perturbations_caught"]
