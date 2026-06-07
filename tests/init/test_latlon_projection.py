"""CPU proofs for the WRF cylindrical-equidistant lat/lon projection (map_proj=6).

These mirror WRF ``share/module_llxy.F`` (``llij_latlon``/``ijll_latlon`` and the
Cassini ``rotate_coords`` rotation). The proofs are pure-CPU and off-path for the
existing Lambert Canary/Switzerland cases.

Proof objects:
  1. forward∘inverse round-trip to round-off over a full grid (unrotated + rotated)
  2. known-coordinate check: a global lat/lon grid's corner lat/lons and cell
     spacing match the namelist parameters (lat1/lon1/latinc/loninc)
  3. map-factor sanity: msf_x = 1/cos(lat), msf_y = 1, both ≈ 1 near the equator
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwrf.init.projection import (
    LatLonGrid,
    latlon_forward,
    latlon_inverse,
    latlon_map_factor,
)


def _max_abs(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.max(np.abs(np.asarray(candidate, np.float64) - np.asarray(reference, np.float64)))
    )


# --- Proof 1a: round-trip, unrotated regular lat/lon grid --------------------
def test_latlon_round_trip_unrotated() -> None:
    lat1, lon1, latinc, loninc = -89.5, -179.5, 1.0, 1.0
    nx, ny = 360, 180
    # Sample a dense scatter of geographic points inside the grid.
    rng = np.random.default_rng(0)
    lat = rng.uniform(-89.0, 89.0, size=2000)
    lon = rng.uniform(-179.0, 179.0, size=2000)
    i, j = latlon_forward(lat, lon, lat1=lat1, lon1=lon1, latinc=latinc, loninc=loninc)
    lat2, lon2 = latlon_inverse(i, j, lat1=lat1, lon1=lon1, latinc=latinc, loninc=loninc)
    assert _max_abs(lat2, lat) < 1.0e-9
    # Longitude compared on the unit circle to be robust to ±180 wrap.
    dlon = (lon2 - lon + 180.0) % 360.0 - 180.0
    assert _max_abs(dlon, np.zeros_like(dlon)) < 1.0e-9
    # Forward indices must be consistent with the analytic linear formula.
    assert _max_abs(j, (lat - lat1) / latinc + 1.0) < 1.0e-9
    del nx, ny


# --- Proof 1b: round-trip over the full grid mesh (LatLonGrid) ----------------
def test_latlon_grid_index_round_trip() -> None:
    grid = LatLonGrid(lat1=-89.5, lon1=-179.5, latinc=1.0, loninc=1.0, nx=360, ny=180)
    lat, lon = grid.latlon("M")
    i, j = latlon_forward(
        lat,
        lon,
        lat1=grid.lat1,
        lon1=grid.lon1,
        latinc=grid.latinc,
        loninc=grid.loninc,
    )
    # Mass point k (0-based) must map back to 1-based index knowni/j + k.
    ii_expected = (grid.knowni + np.arange(grid.nx))[None, :] * np.ones((grid.ny, 1))
    jj_expected = (grid.knownj + np.arange(grid.ny))[:, None] * np.ones((1, grid.nx))
    # Longitude wrap: compare i modulo the full-circle span.
    span_i = 360.0 / grid.loninc
    di = (i - ii_expected + span_i / 2.0) % span_i - span_i / 2.0
    assert _max_abs(di, np.zeros_like(di)) < 1.0e-8
    assert _max_abs(j, jj_expected) < 1.0e-8


# --- Proof 1c: round-trip with a rotated pole (Cassini) -----------------------
def test_latlon_round_trip_rotated_cassini() -> None:
    # Rotated lat/lon: north pole moved to 37N, 25E (an arbitrary rotation).
    kw = dict(
        lat1=-30.0,
        lon1=-40.0,
        latinc=0.25,
        loninc=0.25,
        knowni=1.0,
        knownj=1.0,
        pole_lat=37.0,
        pole_lon=25.0,
        stand_lon=10.0,
    )
    rng = np.random.default_rng(1)
    # Sample computational-grid indices, go to geographic, and back.
    i = rng.uniform(1.0, 200.0, size=3000)
    j = rng.uniform(1.0, 200.0, size=3000)
    lat, lon = latlon_inverse(i, j, **kw)
    i2, j2 = latlon_forward(lat, lon, **kw)
    assert _max_abs(i2, i) < 1.0e-6
    assert _max_abs(j2, j) < 1.0e-6


def test_latlon_rotated_geographic_round_trip() -> None:
    kw = dict(
        lat1=-30.0,
        lon1=-40.0,
        latinc=0.25,
        loninc=0.25,
        pole_lat=37.0,
        pole_lon=25.0,
        stand_lon=10.0,
    )
    rng = np.random.default_rng(2)
    lat = rng.uniform(0.0, 60.0, size=3000)
    lon = rng.uniform(-30.0, 30.0, size=3000)
    i, j = latlon_forward(lat, lon, **kw)
    lat2, lon2 = latlon_inverse(i, j, **kw)
    assert _max_abs(lat2, lat) < 1.0e-6
    dlon = (lon2 - lon + 180.0) % 360.0 - 180.0
    assert _max_abs(dlon, np.zeros_like(dlon)) < 1.0e-6


# --- Proof 2: known-coordinate / namelist consistency -------------------------
def test_latlon_grid_corners_match_namelist() -> None:
    lat1, lon1, latinc, loninc = -89.5, -179.5, 1.0, 1.0
    nx, ny = 360, 180
    grid = LatLonGrid(lat1=lat1, lon1=lon1, latinc=latinc, loninc=loninc, nx=nx, ny=ny)
    lat, lon = grid.latlon("M")
    # SW mass corner == (lat1, lon1).
    assert abs(lat[0, 0] - lat1) < 1.0e-9
    assert abs(((lon[0, 0] - lon1 + 180.0) % 360.0) - 180.0) < 1.0e-9
    # NE mass corner == (lat1 + (ny-1)*latinc, lon1 + (nx-1)*loninc), lon wrapped.
    assert abs(lat[-1, -1] - (lat1 + (ny - 1) * latinc)) < 1.0e-9
    ne_lon_expected = (lon1 + (nx - 1) * loninc + 180.0) % 360.0 - 180.0
    assert abs(((lon[-1, -1] - ne_lon_expected + 180.0) % 360.0) - 180.0) < 1.0e-9
    # Uniform cell spacing in both directions.
    assert _max_abs(np.diff(lat[:, 0]), np.full(ny - 1, latinc)) < 1.0e-9
    dlon = (np.diff(lon[0, :]) + 180.0) % 360.0 - 180.0
    assert _max_abs(dlon, np.full(nx - 1, loninc)) < 1.0e-9


def test_latlon_grid_staggering_offsets() -> None:
    grid = LatLonGrid(lat1=0.0, lon1=0.0, latinc=1.0, loninc=1.0, nx=10, ny=8)
    lat_m, lon_m = grid.latlon("M")
    lat_u, lon_u = grid.latlon("U")
    lat_v, lon_v = grid.latlon("V")
    # U is +1 column (nx+1) and offset -0.5 cell in longitude vs mass.
    assert lon_u.shape == (8, 11)
    assert abs((lon_u[0, 0]) - (lon_m[0, 0] - 0.5 * grid.loninc)) < 1.0e-9
    # V is +1 row (ny+1) and offset -0.5 cell in latitude vs mass.
    assert lat_v.shape == (9, 10)
    assert abs((lat_v[0, 0]) - (lat_m[0, 0] - 0.5 * grid.latinc)) < 1.0e-9


# --- Proof 3: map-factor sanity ----------------------------------------------
def test_latlon_map_factor_equator_and_metric() -> None:
    # Near the equator, both map factors ≈ 1.
    msf_x0, msf_y0 = latlon_map_factor(0.0)
    assert abs(msf_x0 - 1.0) < 1.0e-12
    assert abs(msf_y0 - 1.0) < 1.0e-12
    msf_x_small, msf_y_small = latlon_map_factor(0.5)
    assert 0.99996 <= msf_x_small <= 1.00005
    assert msf_y_small == pytest.approx(1.0)
    # 1/cos metric: at 60N msf_x = 1/cos(60) = 2.
    msf_x60, _ = latlon_map_factor(60.0)
    assert msf_x60 == pytest.approx(2.0, rel=1e-12)
    # Vectorized + monotone increase toward the pole.
    lats = np.array([0.0, 30.0, 45.0, 60.0, 80.0], dtype=np.float64)
    msf_x, msf_y = latlon_map_factor(lats)
    assert np.all(np.diff(msf_x) > 0.0)
    assert np.allclose(msf_y, 1.0)
    assert np.allclose(msf_x, 1.0 / np.cos(np.deg2rad(lats)))


def test_latlon_grid_map_factor_fields_shapes_and_equator() -> None:
    grid = LatLonGrid(lat1=-2.0, lon1=0.0, latinc=1.0, loninc=1.0, nx=20, ny=5)
    fields = grid.derive_fields()
    assert fields["MAPFAC_MX"].shape == (5, 20)
    assert fields["MAPFAC_UX"].shape == (5, 21)
    assert fields["MAPFAC_VX"].shape == (6, 20)
    # msf_y is identically 1 for the cylindrical grid.
    assert np.allclose(fields["MAPFAC_MY"], 1.0)
    # Coriolis sign follows hemisphere.
    assert fields["F"][0, 0] < 0.0  # southern row at -2N
    assert fields["F"][-1, 0] > 0.0  # northern row at +2N
    # Unrotated grid has no wind rotation.
    assert np.allclose(fields["SINALPHA"], 0.0)
    assert np.allclose(fields["COSALPHA"], 1.0)


def test_latlon_grid_rejects_non_map_proj_6() -> None:
    with pytest.raises(ValueError, match="MAP_PROJ=6"):
        LatLonGrid(lat1=0.0, lon1=0.0, latinc=1.0, loninc=1.0, nx=4, ny=4, map_proj=1)
