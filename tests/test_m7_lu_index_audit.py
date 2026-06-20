from __future__ import annotations

from pathlib import Path

from netCDF4 import Dataset
import numpy as np
import pytest

from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.io.land_state import load_prescribed_land_state


RUN_PATH = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z")
CPU_LEAD1 = RUN_PATH / "wrfout_d02_2026-05-21_19:00:00"


def _read_lu_index(path: Path) -> np.ndarray:
    with Dataset(path, "r") as dataset:
        values = dataset.variables["LU_INDEX"][:]
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    array = np.asarray(values)
    if array.shape[:1] == (1,):
        array = array[0]
    return np.asarray(array)


def _category_counts(values: np.ndarray) -> dict[int, int]:
    finite = np.asarray(values)[np.isfinite(values)]
    categories, counts = np.unique(finite.astype(np.int64), return_counts=True)
    return {int(category): int(count) for category, count in zip(categories, counts)}


def test_prescribed_land_state_lu_index_matches_gen2_reference_distribution():
    if not RUN_PATH.exists() or not CPU_LEAD1.exists():
        pytest.skip("20260521 Gen2 d02 fixture unavailable")

    run = Gen2Run(RUN_PATH)
    state = load_prescribed_land_state(run, "d02", 0)

    loaded = np.asarray(state.lu_index)
    wrfinput = _read_lu_index(run.wrfinput_file("d02"))
    cpu_lead1 = _read_lu_index(CPU_LEAD1)

    assert loaded.shape == (66, 159)
    assert np.array_equal(loaded, wrfinput)
    assert np.array_equal(loaded, cpu_lead1)
    assert _category_counts(loaded) == {5: 164, 9: 83, 10: 251, 13: 15, 16: 255, 17: 9726}
