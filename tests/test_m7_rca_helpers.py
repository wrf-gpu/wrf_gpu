from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


m7_rca_hour_by_hour = _load_script("m7_rca_hour_by_hour")
m7_rca_spatial_maps = _load_script("m7_rca_spatial_maps")


def test_field_stats_reports_mean_max_and_correlation() -> None:
    cpu = np.array([[1.0, 2.0], [3.0, 4.0]])
    gpu = cpu + 2.0

    stats = m7_rca_hour_by_hour.field_stats(gpu, cpu)

    assert stats["status"] == "OK"
    assert stats["mean_diff"] == pytest.approx(2.0)
    assert stats["max_abs_diff"] == pytest.approx(2.0)
    assert stats["correlation"] == pytest.approx(1.0)


def test_boundary_split_detects_boundary_concentration() -> None:
    diff = np.zeros((16, 18), dtype=np.float64)
    diff[:5, :] = 10.0

    split = m7_rca_spatial_maps.split_boundary_interior(diff, width=5)

    assert split["boundary_cell_count"] > 0
    assert split["interior_cell_count"] > 0
    assert split["concentration"] == "BOUNDARY_CONCENTRATED"


def test_spatial_classification_detects_uniform_bias() -> None:
    diff = np.ones((20, 20), dtype=np.float64) * -3.0
    split = m7_rca_spatial_maps.split_boundary_interior(diff, width=5)

    classification = m7_rca_spatial_maps.classify_spatial_pattern(diff, split)

    assert classification == "SPATIALLY_UNIFORM_BIAS"
