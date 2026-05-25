from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPERATIONAL_MODE = ROOT / "src" / "gpuwrf" / "runtime" / "operational_mode.py"


def test_operational_rk1_dispatch_runs_one_acoustic_substep():
    source = OPERATIONAL_MODE.read_text(encoding="utf-8")

    assert "solve_em.F:1472-1475" in source
    assert "lambda value: advance_stage(value, 1.0 / 3.0, 1)" in source
    assert "lambda value: advance_stage(value, 1.0 / 3.0, False)" not in source


def test_operational_mode_does_not_import_validation_coupled_or_acoustic_loop():
    tree = ast.parse(OPERATIONAL_MODE.read_text(encoding="utf-8"))
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    assert "gpuwrf.dynamics.acoustic_loop" not in imported_modules
    assert "gpuwrf.dynamics.coupled_step" not in imported_modules
