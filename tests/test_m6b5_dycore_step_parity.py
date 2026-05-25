from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.m6b5_dycore_step_compare import COMPARE_FIELDS, NAMELIST_DISABLED, SOURCE_WRFOUT, compare_tier, synthetic_dryrun
from gpuwrf.validation.savepoint_schema import SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS, VALID_BOUNDARIES, VALID_OPERATORS
from gpuwrf.validation.savepoint_schema import load_tolerance_ladder


def test_m6b5_schema_v6_supports_dycore_step_boundary() -> None:
    assert SCHEMA_VERSION == "m6b5-savepoint-v6"
    assert "m6b5-savepoint-v6" in SUPPORTED_SCHEMA_VERSIONS
    assert "dycore_step_complete" in VALID_BOUNDARIES
    assert "dycore_step" in VALID_OPERATORS


def test_m6b5_tolerance_ladder_has_per_timestep_entries() -> None:
    ladder = load_tolerance_ladder()
    assert ladder["schema_version"] == "m6b5-tolerance-ladder-v6"
    dycore = ladder["dycore_step_tolerances"]
    assert dycore["steps"] == 10
    assert dycore["rk_stages_per_step"] == 3
    assert dycore["acoustic_substeps_per_stage"] == 10
    assert set(COMPARE_FIELDS) == set(dycore["fields"])


def test_m6b5_namelist_contract_disables_physics_and_boundary() -> None:
    assert NAMELIST_DISABLED == {
        "mp_physics": 0,
        "bl_pbl_physics": 0,
        "ra_lw_physics": 0,
        "ra_sw_physics": 0,
        "cu_physics": 0,
        "sf_sfclay_physics": 0,
        "sf_surface_physics": 0,
        "specified": False,
    }


@pytest.mark.skipif(not SOURCE_WRFOUT.exists(), reason="M6B5 source wrfout is unavailable")
def test_m6b5_column_dycore_step_parity_one_step(tmp_path: Path) -> None:
    result = compare_tier("column", 1, tmp_path / "savepoints")

    assert result["passed"], result["outcome"]
    assert result["savepoint_count"] == 1
    step = result["results"][0]
    assert step["passed"]
    assert set(COMPARE_FIELDS) == set(step["fields"])


@pytest.mark.skipif(not SOURCE_WRFOUT.exists(), reason="M6B5 source wrfout is unavailable")
def test_m6b5_synthetic_dryrun_catches_boundary_perturbations() -> None:
    result = synthetic_dryrun()

    assert result["passed"]
    assert result["clean_self_compare_passed"]
    assert result["boundary_field_perturbations_caught"]
