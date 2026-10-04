"""scripts/wn3_score.py output_integrity: missing CPU variables, degenerate (constant) GPU fields, WRF globals (CPU)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("wn3_score", REPO / "scripts" / "wn3_score.py")
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)
GLOBALS = {name: 1 for name in score.REQUIRED_GLOBAL_ATTRS}


def _frame(path: Path, variables: dict[str, np.ndarray], attrs: dict) -> Path:
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("x", 8)
        for name, values in variables.items():
            ds.createVariable(name, "f4", ("x",))[:] = values
        for key, value in attrs.items():
            ds.setncattr(key, value)
    return path


def _pair(tmp_path: Path, gpu_vars: dict, gpu_attrs: dict = GLOBALS, frames: int = 3):
    ramp = np.arange(8, dtype="f4")
    cpu = [_frame(tmp_path / f"cpu_{i}.nc", {"T2": ramp + i, "SMOIS": ramp * 0.1, "W": ramp - 3}, GLOBALS)
           for i in range(frames)]
    gpu = [_frame(tmp_path / f"gpu_{i}.nc", gpu_vars, gpu_attrs) for i in range(frames)]
    return cpu, gpu


def test_complete_and_varying_gpu_output_passes(tmp_path):
    ramp = np.arange(8, dtype="f4")
    cpu, gpu = _pair(tmp_path, {"T2": ramp, "SMOIS": ramp * 0.1, "W": ramp})
    result = score.output_integrity(cpu, gpu)
    assert result["pass"] and not result["missing_variables_vs_cpu"] and not result["degenerate_fields"]


def test_cpu_variable_missing_from_gpu_fails(tmp_path):
    ramp = np.arange(8, dtype="f4")
    cpu, gpu = _pair(tmp_path, {"T2": ramp, "W": ramp})  # SMOIS absent, every remaining field fine
    result = score.output_integrity(cpu, gpu)
    assert not result["pass"]
    assert result["missing_variables_vs_cpu"] == {"SMOIS": "all frames"}
    assert not result["degenerate_fields"] and not result["global_attrs"]["missing_required"]


def test_documented_optional_variable_may_be_missing(tmp_path, monkeypatch):
    ramp = np.arange(8, dtype="f4")
    monkeypatch.setattr(score, "OPTIONAL_CPU_VARIABLES", {"SMOIS": "test-only optional stream"})
    cpu, gpu = _pair(tmp_path, {"T2": ramp, "W": ramp})
    result = score.output_integrity(cpu, gpu)
    assert result["pass"] and result["optional_missing_allowed"] == {"SMOIS": "test-only optional stream"}


def test_constant_gpu_field_and_missing_globals_fail(tmp_path):
    ramp = np.arange(8, dtype="f4")
    cpu, gpu = _pair(tmp_path, {"T2": ramp, "SMOIS": np.zeros(8, "f4"), "W": ramp}, gpu_attrs={})
    result = score.output_integrity(cpu, gpu)
    assert not result["pass"] and sorted(result["degenerate_fields"]) == ["SMOIS"]
    assert result["global_attrs"]["missing_required"] == list(score.REQUIRED_GLOBAL_ATTRS)
