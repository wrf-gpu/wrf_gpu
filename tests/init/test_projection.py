from __future__ import annotations

import os
from pathlib import Path

from netCDF4 import Dataset
import numpy as np
import pytest

from gpuwrf.init.projection import (
    LambertGrid,
    coriolis_from_lat,
    lambert_forward,
    lambert_inverse,
    lambert_map_factor,
    rotation_from_lon,
)


WPS_ROOT = Path(os.environ.get("GPUWRF_WPS_CASES_ROOT", "/mnt/data/canairy_meteo/runs/wps_cases"))
DOMAINS = ("d01", "d02", "d03")


def _first_case() -> Path:
    cases = [
        path
        for path in sorted(WPS_ROOT.glob("*"))
        if all((path / "l3" / f"geo_em.{domain}.nc").exists() for domain in DOMAINS)
    ]
    if not cases:
        pytest.skip(f"no WPS geo_em cases under {WPS_ROOT}")
    return cases[0]


def _max_abs(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(candidate, np.float64) - np.asarray(reference, np.float64))))


def _max_rel(candidate: np.ndarray, reference: np.ndarray) -> float:
    reference = np.asarray(reference, np.float64)
    diff = np.abs(np.asarray(candidate, np.float64) - reference)
    return float(np.max(diff / np.maximum(np.abs(reference), 1.0e-12)))


def test_lambert_forward_inverse_round_trip() -> None:
    lat = np.array([[25.9, 28.3, 30.6]], dtype=np.float64)
    lon = np.array([[-20.5, -16.4, -12.1]], dtype=np.float64)
    x, y = lambert_forward(lat, lon, truelat1=25.0, truelat2=30.0, stand_lon=-16.4)
    lat2, lon2 = lambert_inverse(x, y, truelat1=25.0, truelat2=30.0, stand_lon=-16.4)
    assert _max_abs(lat2, lat) < 1.0e-10
    assert _max_abs(lon2, lon) < 1.0e-10


def test_lambert_grid_reproduces_geo_em_coordinates_and_metrics() -> None:
    case = _first_case()
    for domain in DOMAINS:
        with Dataset(str(case / "l3" / f"geo_em.{domain}.nc")) as dataset:
            grid = LambertGrid.from_wps_dataset(dataset)
            derived = grid.derive_fields()
            for name in (
                "XLAT_M",
                "XLONG_M",
                "XLAT_U",
                "XLONG_U",
                "XLAT_V",
                "XLONG_V",
                "XLAT_C",
                "XLONG_C",
                "CLAT",
                "CLONG",
            ):
                assert _max_abs(derived[name], dataset.variables[name][0]) <= 1.0e-4, (domain, name)
            for name in (
                "MAPFAC_M",
                "MAPFAC_MX",
                "MAPFAC_MY",
                "MAPFAC_U",
                "MAPFAC_UX",
                "MAPFAC_UY",
                "MAPFAC_V",
                "MAPFAC_VX",
                "MAPFAC_VY",
            ):
                assert _max_rel(derived[name], dataset.variables[name][0]) <= 1.0e-5, (domain, name)
            for name in ("F", "E"):
                assert _max_abs(derived[name], dataset.variables[name][0]) <= 1.0e-9, (domain, name)
            for name in ("SINALPHA", "COSALPHA"):
                assert _max_abs(derived[name], dataset.variables[name][0]) <= 1.0e-6, (domain, name)


def test_scalar_metric_helpers_match_wps_formula() -> None:
    mapfac = lambert_map_factor(28.3, truelat1=25.0, truelat2=30.0)
    f, e = coriolis_from_lat(28.3)
    sina, cosa = rotation_from_lon(-16.0, truelat1=25.0, truelat2=30.0, stand_lon=-16.4)
    assert 0.999 <= mapfac <= 1.001
    assert 6.8e-5 <= f <= 7.1e-5
    assert 1.27e-4 <= e <= 1.29e-4
    assert -0.004 <= sina <= 0.0
    assert 0.999 <= cosa <= 1.0
