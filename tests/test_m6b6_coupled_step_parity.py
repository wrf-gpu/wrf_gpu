from __future__ import annotations

from pathlib import Path
import sys

import jax
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The coupled-step parity tests below build a very large coupled dycore+physics
# XLA graph that exhausts the CPU XLA backend and SIGSEGVs the whole pytest
# process (the same memory limit that gates tests/savepoint/test_dycore_100_steps).
# They are GPU-targeted parity replays; run them on the JAX GPU backend. Skipping
# on CPU avoids a hard segfault that would crash the entire test run. (Coupled
# dycore parity is also covered by the idealized Straka/Skamarock close-gate.)
_CPU_SEGV_SKIP = pytest.mark.skipif(
    jax.default_backend() == "cpu",
    reason=(
        "coupled-step parity builds a large XLA graph that SIGSEGVs the CPU "
        "backend; GPU-targeted parity replay -- run on the GPU backend"
    ),
)

from scripts.m6b6_coupled_step_compare import COMPARE_FIELDS, NAMELIST_PHYSICS_BOUNDARY_ON, SOURCE_WRFBDY, SOURCE_WRFOUT, compare_tier, synthetic_dryrun
from gpuwrf.dynamics.coupled_step import PHYSICS_TENDENCY_FIELDS
from gpuwrf.validation.savepoint_schema import SCHEMA_VERSION, SUPPORTED_SCHEMA_VERSIONS, VALID_BOUNDARIES, VALID_OPERATORS
from gpuwrf.validation.savepoint_schema import load_tolerance_ladder


def test_m6b6_schema_v7_supports_coupled_step_boundary() -> None:
    assert SCHEMA_VERSION == "m6b6-savepoint-v7"
    assert "m6b6-savepoint-v7" in SUPPORTED_SCHEMA_VERSIONS
    assert "coupled_step_complete" in VALID_BOUNDARIES
    assert "coupled_step" in VALID_OPERATORS


def test_m6b6_tolerance_ladder_has_coupled_step_entries() -> None:
    ladder = load_tolerance_ladder()
    coupled = ladder["coupled_step_tolerances"]

    assert coupled["steps"] == 10
    assert coupled["rk_stages_per_step"] == 3
    assert coupled["acoustic_substeps_per_stage"] == 10
    assert coupled["physics"] == {
        "mp_physics": 8,
        "bl_pbl_physics": 5,
        "ra_lw_physics": 4,
        "ra_sw_physics": 4,
    }
    assert set(COMPARE_FIELDS) == set(coupled["fields"])
    assert set(PHYSICS_TENDENCY_FIELDS).issubset(coupled["fields"])


def test_m6b6_namelist_contract_enables_m5_physics_and_boundary() -> None:
    assert NAMELIST_PHYSICS_BOUNDARY_ON == {
        "mp_physics": 8,
        "bl_pbl_physics": 5,
        "ra_lw_physics": 4,
        "ra_sw_physics": 4,
        "cu_physics": 0,
        "sf_sfclay_physics": 0,
        "sf_surface_physics": 0,
        "specified": True,
    }


@_CPU_SEGV_SKIP
@pytest.mark.skipif(not SOURCE_WRFOUT.exists() or not SOURCE_WRFBDY.exists(), reason="M6B6 source wrfout/wrfbdy is unavailable")
def test_m6b6_column_coupled_step_parity_one_step(tmp_path: Path) -> None:
    result = compare_tier("column", 1, tmp_path / "savepoints")

    assert result["passed"], result["outcome"]
    assert result["savepoint_count"] == 1
    step = result["results"][0]
    assert step["passed"]
    assert set(COMPARE_FIELDS) == set(step["fields"])


@_CPU_SEGV_SKIP
@pytest.mark.skipif(not SOURCE_WRFOUT.exists() or not SOURCE_WRFBDY.exists(), reason="M6B6 source wrfout/wrfbdy is unavailable")
def test_m6b6_synthetic_dryrun_catches_coupled_perturbations() -> None:
    result = synthetic_dryrun()

    assert result["passed"]
    assert result["clean_self_compare_passed"]
    assert result["boundary_field_perturbations_caught"]
