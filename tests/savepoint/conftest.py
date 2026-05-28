from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable

import pytest


os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.validation.savepoint_io import read_savepoint  # noqa: E402
from scripts import m6b6_coupled_step_compare as m6b6  # noqa: E402


@dataclass(frozen=True)
class SavepointGroup:
    name: str
    operator: str
    wrf_routine: str
    status: str
    milestone_owner: str


SAVEPOINT_GROUPS: tuple[SavepointGroup, ...] = (
    SavepointGroup(
        name="dycore.coupled_step_complete",
        operator="gpuwrf.dynamics.coupled_step.coupled_timesteps_wrf",
        wrf_routine="WRF solve_em.F Runge_Kutta_loop plus small_steps acoustic loop",
        status="100-step column reference preserved by M8.B",
        milestone_owner="M8.B",
    ),
    SavepointGroup(
        name="dycore.lateral_boundary_replay",
        operator="gpuwrf.coupling.boundary_apply.apply_lateral_boundaries",
        wrf_routine="WRF solve_em.F specified lateral-boundary tendency application",
        status="placeholder until M9 writes operational reference states",
        milestone_owner="M9",
    ),
    SavepointGroup(
        name="physics.tendency_couplers",
        operator="Thompson, MYNN, RRTMG, and surface-layer adapters",
        wrf_routine="WRF module_mp_thompson.F, module_bl_mynn.F, module_ra_rrtmg_*.F, and surface driver routines",
        status="placeholder until M9 writes operational reference states",
        milestone_owner="M9",
    ),
    SavepointGroup(
        name="operational.surface_variables",
        operator="gpuwrf.runtime.operational_mode.run_forecast_operational",
        wrf_routine="WRF solve_em.F forecast step plus WRF diagnostic/output variable production",
        status="placeholder until M9 writes operational reference states",
        milestone_owner="M9",
    ),
)


@pytest.fixture(scope="session")
def savepoint_groups() -> tuple[SavepointGroup, ...]:
    return SAVEPOINT_GROUPS


@pytest.fixture(scope="session")
def wrf_fortran_reference_paths() -> dict[str, Path]:
    paths = {"wrfout": m6b6.SOURCE_WRFOUT, "wrfbdy": m6b6.SOURCE_WRFBDY}
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        pytest.skip(f"M6B6 source {', '.join(missing)} unavailable for savepoint reference loading")
    return paths


@pytest.fixture
def wrf_reference_root(tmp_path: Path) -> Path:
    return tmp_path / "wrf_fortran_reference"


@pytest.fixture(scope="session")
def m6b6_compare_fields() -> tuple[str, ...]:
    return tuple(m6b6.COMPARE_FIELDS)


@pytest.fixture(scope="session")
def m6b6_compare_tier() -> Callable[[str, int, Path], dict[str, object]]:
    return m6b6.compare_tier


@pytest.fixture
def wrf_fortran_reference_loader() -> Callable[[Path], dict[str, Any]]:
    def load(path: Path) -> dict[str, Any]:
        return read_savepoint(Path(path)).arrays

    return load


@pytest.fixture
def jax_state_under_test_loader() -> Callable[[str, int], list[dict[str, Any]]]:
    def load(tier: str, steps: int) -> list[dict[str, Any]]:
        snapshots, _context = m6b6._coupled_steps(tier, int(steps))
        return snapshots

    return load
