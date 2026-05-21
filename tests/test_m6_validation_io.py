from __future__ import annotations

import numpy as np

from gpuwrf.io.gen2_accessor import DEFAULT_M6_GEN2_RUN_DIR, Gen2Run
from gpuwrf.io.validation import domain_mask, lead_time_slice, load_gen2_var, regrid, unit_convert


RUN_PATH = DEFAULT_M6_GEN2_RUN_DIR


class _Grid:
    def __init__(self, ny: int, nx: int) -> None:
        self.ny = ny
        self.nx = nx


def test_shared_loader_and_lead_time_slice():
    run = Gen2Run(RUN_PATH)

    assert lead_time_slice(run, 6) == 6
    field = load_gen2_var(run, "d02", "T", time=lead_time_slice(run, 0))
    assert tuple(field.shape) == (44, 66, 159)
    assert bool(np.isfinite(np.asarray(field)).all())


def test_regrid_unit_convert_and_masks():
    src = np.arange(9, dtype=np.float64).reshape(3, 3)
    out = regrid(src, _Grid(3, 3), _Grid(5, 5))
    assert tuple(out.shape) == (5, 5)
    assert np.isclose(np.asarray(out)[0, 0], 0.0)
    assert np.isclose(np.asarray(out)[-1, -1], 8.0)

    assert np.isclose(np.asarray(unit_convert(np.array([273.15]), "K", "C"))[0], 0.0)
    assert np.isclose(np.asarray(unit_convert(np.array([1.0]), "kg/kg", "g/kg"))[0], 1000.0)
    assert np.isclose(np.asarray(unit_convert(np.array([100000.0]), "Pa", "hPa"))[0], 1000.0)

    grid = Gen2Run(RUN_PATH).grid("d02")
    canary = domain_mask(grid, "canary")
    land = domain_mask(grid, "land")
    sea = domain_mask(grid, "sea")
    assert canary.shape == (66, 159)
    assert land.shape == sea.shape == canary.shape
    assert bool(np.all(land | sea))
    assert not bool(np.any(land & sea))
