from __future__ import annotations

import ast
from pathlib import Path

from gpuwrf.dynamics.acoustic_loop import AcousticLoopState, acoustic_substep_wrf
from gpuwrf.dynamics.coupled_step import CoupledStepConfig, coupled_timestep_wrf
from gpuwrf.dynamics.core.acoustic import AcousticCoreState, acoustic_substep_core
from gpuwrf.dynamics.core.coupled import CoupledCoreConfig, coupled_timestep_core
from gpuwrf.dynamics.core.dycore import DycoreCoreConfig, dycore_timestep_core
from gpuwrf.dynamics.dycore_step import DycoreStepConfig, dycore_timestep_wrf


ROOT = Path(__file__).resolve().parents[1]
OPERATIONAL_MODE = ROOT / "src" / "gpuwrf" / "runtime" / "operational_mode.py"


def test_validation_wrappers_reexport_shared_core_identities():
    assert AcousticLoopState is AcousticCoreState
    assert acoustic_substep_wrf is acoustic_substep_core
    assert DycoreStepConfig is DycoreCoreConfig
    assert dycore_timestep_wrf is dycore_timestep_core
    assert CoupledStepConfig is CoupledCoreConfig
    assert coupled_timestep_wrf is coupled_timestep_core


def test_operational_imports_core_not_validation_wrappers():
    tree = ast.parse(OPERATIONAL_MODE.read_text(encoding="utf-8"))
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    assert "gpuwrf.dynamics.core.acoustic" in imported_modules
    assert "gpuwrf.dynamics.core.coupled" in imported_modules
    assert "gpuwrf.dynamics.validation_wrappers" not in imported_modules
    assert "gpuwrf.dynamics.acoustic_loop" not in imported_modules
    assert "gpuwrf.dynamics.dycore_step" not in imported_modules
    assert "gpuwrf.dynamics.coupled_step" not in imported_modules


def test_custom_operational_small_step_was_removed():
    source = OPERATIONAL_MODE.read_text(encoding="utf-8")

    assert "_wrf_small_step_acoustic" not in source
    assert "acoustic_substep_core" in source
    assert "coupled_timestep_core" in source
