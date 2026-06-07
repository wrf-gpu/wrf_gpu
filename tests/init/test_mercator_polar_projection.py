"""CPU proofs for the WRF Mercator (map_proj=3) and polar-stereographic
(map_proj=2) projections.

These mirror WRF ``share/module_llxy.F`` (``set_merc``/``llij_merc``/
``ijll_merc`` and ``set_ps``/``llij_ps``/``ijll_ps``). The proofs are pure-CPU
and additive — they do not touch the existing Lambert/lat-lon lanes.

Proof objects (per projection):
  1. forward∘inverse round-trip to round-off over a dense scatter + full grid
  2. known-coordinate check:
       Mercator  -> map factor == 1 exactly on the true latitude; the reference
                    point (lat1, lon1) maps back to (knowni, knownj)
       Polar     -> the pole (lat == hemi*90) maps to (polei, polej) and the
                    inverse of that point returns hemi*90; map factor == 1 on the
                    true latitude
  3. map-factor identity: msf(truelat1) == 1; isotropy (MAPFAC_*X == MAPFAC_*Y);
     monotone growth away from the true latitude
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gpuwrf.init.projection import (
    MercatorGrid,
    PolarGrid,
    mercator_forward,
    mercator_inverse,
    mercator_map_factor,
    polar_forward,
    polar_inverse,
    polar_map_factor,
    rotation_from_lon_ps,
)


def _max_abs(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.max(np.abs(np.asarray(candidate, np.float64) - np.asarray(reference, np.float64)))
    )


def _dlon_wrap(delta: np.ndarray) -> np.ndarray:
    return (np.asarray(delta, np.float64) + 180.0) % 360.0 - 180.0


# =============================================================================
# Mercator (map_proj=3)
# =============================================================================

MERC_KW = dict(truelat1=20.0, lat1=0.0, lon1=-30.0, dx_m=12000.0)


# --- Proof 1a: scalar/scatter round-trip ------------------------------------
def test_mercator_round_trip_scatter() -> None:
    rng = np.random.default_rng(10)
    # Mercator is unusable at the poles; sample a generous mid-latitude band.
    lat = rng.uniform(-70.0, 70.0, size=5000)
    lon = rng.uniform(-179.0, 179.0, size=5000)
    i, j = mercator_forward(lat, lon, **MERC_KW)
    lat2, lon2 = mercator_inverse(i, j, **MERC_KW)
    assert _max_abs(lat2, lat) < 1.0e-9
    assert _max_abs(_dlon_wrap(lon2 - lon), np.zeros_like(lon)) < 1.0e-9


def test_mercator_round_trip_scalar() -> None:
    i, j = mercator_forward(28.3, -16.4, **MERC_KW)
    assert isinstance(i, float) and isinstance(j, float)
    lat, lon = mercator_inverse(i, j, **MERC_KW)
    assert abs(lat - 28.3) < 1.0e-9
    assert abs(lon - (-16.4)) < 1.0e-9


# --- Proof 1b: full-grid round-trip via MercatorGrid -------------------------
def test_mercator_grid_round_trip() -> None:
    grid = MercatorGrid(
        truelat1=20.0, stand_lon=-30.0, lat1=10.0, lon1=-30.0,
        dx_m=10000.0, dy_m=10000.0, nx=80, ny=60, knowni=1.0, knownj=1.0,
    )
    lat, lon = grid.latlon("M")
    i, j = mercator_forward(
        lat, lon, truelat1=grid.truelat1, lat1=grid.lat1, lon1=grid.lon1,
        dx_m=grid.dx_m, knowni=grid.knowni, knownj=grid.knownj,
    )
    ii_expected = (grid.knowni + np.arange(grid.nx))[None, :] * np.ones((grid.ny, 1))
    jj_expected = (grid.knownj + np.arange(grid.ny))[:, None] * np.ones((1, grid.nx))
    assert _max_abs(i, ii_expected) < 1.0e-7
    assert _max_abs(j, jj_expected) < 1.0e-7


# --- Proof 2: known coordinate (reference point) -----------------------------
def test_mercator_reference_point_maps_to_known_ij() -> None:
    kw = dict(truelat1=20.0, lat1=12.5, lon1=-40.0, dx_m=9000.0, knowni=3.0, knownj=7.0)
    i, j = mercator_forward(12.5, -40.0, **kw)
    assert abs(i - 3.0) < 1.0e-9
    assert abs(j - 7.0) < 1.0e-9


# --- Proof 3: map factor == 1 on the true latitude; isotropy/monotone --------
def test_mercator_map_factor_true_latitude() -> None:
    # Identically 1 on the true latitude.
    assert abs(mercator_map_factor(20.0, truelat1=20.0) - 1.0) < 1.0e-14
    assert abs(mercator_map_factor(0.0, truelat1=0.0) - 1.0) < 1.0e-14
    # cos(truelat1)/cos(lat): at the equator with truelat1=20 -> cos(20).
    assert mercator_map_factor(0.0, truelat1=20.0) == pytest.approx(
        math.cos(math.radians(20.0)), rel=1e-13
    )
    # msf = cos(truelat1)/cos(lat) is monotone in |lat|: minimum at the equator
    # (cos(lat) maximal), growing without bound toward the poles. It crosses 1
    # exactly at +/- truelat1 and is symmetric in latitude.
    lats = np.array([0.0, 20.0, 40.0, 60.0, 75.0], dtype=np.float64)
    msf = mercator_map_factor(lats, truelat1=20.0)
    assert np.argmin(msf) == 0  # minimum at the equator
    assert np.all(np.diff(msf) > 0.0)  # strictly increasing with latitude
    # Symmetric about the equator.
    assert mercator_map_factor(-37.0, truelat1=20.0) == pytest.approx(
        mercator_map_factor(37.0, truelat1=20.0), rel=1e-14
    )


def test_mercator_grid_fields_isotropic_and_shapes() -> None:
    grid = MercatorGrid(
        truelat1=20.0, stand_lon=-30.0, lat1=10.0, lon1=-30.0,
        dx_m=10000.0, dy_m=10000.0, nx=30, ny=20,
    )
    fields = grid.derive_fields()
    assert fields["XLAT_M"].shape == (20, 30)
    assert fields["MAPFAC_U"].shape == (20, 31)
    assert fields["MAPFAC_V"].shape == (21, 30)
    # Isotropic map factor.
    assert np.allclose(fields["MAPFAC_MX"], fields["MAPFAC_MY"])
    assert np.allclose(fields["MAPFAC_M"], fields["MAPFAC_MX"])
    # North-aligned: no wind rotation.
    assert np.allclose(fields["SINALPHA"], 0.0)
    assert np.allclose(fields["COSALPHA"], 1.0)
    # Coriolis sign follows hemisphere (all rows here are positive-lat).
    assert np.all(fields["F"] > 0.0)


def test_mercator_grid_rejects_non_map_proj_3() -> None:
    with pytest.raises(ValueError, match="MAP_PROJ=3"):
        MercatorGrid(
            truelat1=20.0, stand_lon=0.0, lat1=0.0, lon1=0.0,
            dx_m=1000.0, dy_m=1000.0, nx=4, ny=4, map_proj=1,
        )


# =============================================================================
# Polar-stereographic (map_proj=2)
# =============================================================================

# Northern-hemisphere reference configuration (Arctic domain).
PS_KW = dict(truelat1=60.0, stand_lon=-90.0, lat1=45.0, lon1=-120.0, dx_m=25000.0)
# Southern-hemisphere configuration (Antarctic domain).
PS_KW_S = dict(truelat1=-71.0, stand_lon=0.0, lat1=-60.0, lon1=0.0, dx_m=30000.0)


# --- Proof 1a: scatter round-trip (NH) --------------------------------------
def test_polar_round_trip_scatter_nh() -> None:
    rng = np.random.default_rng(20)
    lat = rng.uniform(10.0, 89.0, size=5000)
    lon = rng.uniform(-179.0, 179.0, size=5000)
    i, j = polar_forward(lat, lon, **PS_KW)
    lat2, lon2 = polar_inverse(i, j, **PS_KW)
    assert _max_abs(lat2, lat) < 1.0e-7
    assert _max_abs(_dlon_wrap(lon2 - lon), np.zeros_like(lon)) < 1.0e-7


# --- Proof 1b: scatter round-trip (SH) --------------------------------------
def test_polar_round_trip_scatter_sh() -> None:
    rng = np.random.default_rng(21)
    lat = rng.uniform(-89.0, -10.0, size=5000)
    lon = rng.uniform(-179.0, 179.0, size=5000)
    i, j = polar_forward(lat, lon, **PS_KW_S)
    lat2, lon2 = polar_inverse(i, j, **PS_KW_S)
    assert _max_abs(lat2, lat) < 1.0e-7
    assert _max_abs(_dlon_wrap(lon2 - lon), np.zeros_like(lon)) < 1.0e-7


def test_polar_round_trip_scalar() -> None:
    i, j = polar_forward(70.0, -100.0, **PS_KW)
    assert isinstance(i, float) and isinstance(j, float)
    lat, lon = polar_inverse(i, j, **PS_KW)
    assert abs(lat - 70.0) < 1.0e-7
    assert abs(_dlon_wrap(np.array(lon - (-100.0)))) < 1.0e-7


# --- Proof 1c: full-grid round-trip via PolarGrid ----------------------------
def test_polar_grid_round_trip() -> None:
    grid = PolarGrid(
        truelat1=60.0, stand_lon=-90.0, lat1=45.0, lon1=-120.0,
        dx_m=25000.0, dy_m=25000.0, nx=70, ny=70, knowni=1.0, knownj=1.0,
    )
    lat, lon = grid.latlon("M")
    i, j = polar_forward(
        lat, lon, truelat1=grid.truelat1, stand_lon=grid.stand_lon,
        lat1=grid.lat1, lon1=grid.lon1, dx_m=grid.dx_m,
        knowni=grid.knowni, knownj=grid.knownj,
    )
    ii_expected = (grid.knowni + np.arange(grid.nx))[None, :] * np.ones((grid.ny, 1))
    jj_expected = (grid.knownj + np.arange(grid.ny))[:, None] * np.ones((1, grid.nx))
    assert _max_abs(i, ii_expected) < 1.0e-5
    assert _max_abs(j, jj_expected) < 1.0e-5


# --- Proof 2: pole handling --------------------------------------------------
def test_polar_pole_maps_to_pole_point_nh() -> None:
    # The geographic North Pole projects exactly to the precomputed pole point,
    # and the inverse of that grid point returns hemi*90 (WRF r2==0 branch).
    from gpuwrf.init.projection import _ps_constants

    _hemi, _rebydx, _scale_top, polei, polej = _ps_constants(
        PS_KW["truelat1"], PS_KW["stand_lon"], PS_KW["lat1"], PS_KW["lon1"],
        PS_KW["dx_m"], 1.0, 1.0,
    )
    i, j = polar_forward(90.0, 0.0, **PS_KW)  # longitude is irrelevant at the pole
    assert abs(i - polei) < 1.0e-6
    assert abs(j - polej) < 1.0e-6
    lat, lon = polar_inverse(polei, polej, **PS_KW)
    assert abs(lat - 90.0) < 1.0e-9
    assert abs(lon - (PS_KW["stand_lon"] + 90.0)) < 1.0e-9  # reflon at the pole


def test_polar_pole_maps_to_pole_point_sh() -> None:
    from gpuwrf.init.projection import _ps_constants

    _hemi, _rebydx, _scale_top, polei, polej = _ps_constants(
        PS_KW_S["truelat1"], PS_KW_S["stand_lon"], PS_KW_S["lat1"], PS_KW_S["lon1"],
        PS_KW_S["dx_m"], 1.0, 1.0,
    )
    i, j = polar_forward(-90.0, 17.0, **PS_KW_S)
    assert abs(i - polei) < 1.0e-6
    assert abs(j - polej) < 1.0e-6
    lat, _lon = polar_inverse(polei, polej, **PS_KW_S)
    assert abs(lat - (-90.0)) < 1.0e-9


def test_polar_reference_point_maps_to_known_ij() -> None:
    kw = dict(
        truelat1=60.0, stand_lon=-90.0, lat1=45.0, lon1=-120.0,
        dx_m=25000.0, knowni=4.0, knownj=9.0,
    )
    i, j = polar_forward(45.0, -120.0, **kw)
    assert abs(i - 4.0) < 1.0e-6
    assert abs(j - 9.0) < 1.0e-6


# --- Proof 3: map factor == 1 on the true latitude ---------------------------
def test_polar_map_factor_true_latitude() -> None:
    # Identically 1 on the true latitude (both hemispheres).
    assert abs(polar_map_factor(60.0, truelat1=60.0) - 1.0) < 1.0e-14
    assert abs(polar_map_factor(-71.0, truelat1=-71.0) - 1.0) < 1.0e-14
    # At the (NH) pole, msf = scale_top/(1+1) -> minimum; grows toward equator.
    lats = np.array([90.0, 75.0, 60.0, 45.0, 30.0], dtype=np.float64)
    msf = polar_map_factor(lats, truelat1=60.0)
    assert np.all(np.diff(msf) > 0.0)  # monotone increase from pole to equator
    # Known closed form at the pole.
    scale_top = 1.0 + math.sin(math.radians(60.0))
    assert polar_map_factor(90.0, truelat1=60.0) == pytest.approx(scale_top / 2.0, rel=1e-13)


def test_polar_grid_fields_isotropic_and_rotation() -> None:
    grid = PolarGrid(
        truelat1=60.0, stand_lon=-90.0, lat1=45.0, lon1=-120.0,
        dx_m=25000.0, dy_m=25000.0, nx=40, ny=40,
    )
    fields = grid.derive_fields()
    assert fields["XLAT_M"].shape == (40, 40)
    assert fields["MAPFAC_U"].shape == (40, 41)
    assert fields["MAPFAC_V"].shape == (41, 40)
    # Isotropic.
    assert np.allclose(fields["MAPFAC_MX"], fields["MAPFAC_MY"])
    assert np.allclose(fields["MAPFAC_M"], fields["MAPFAC_MX"])
    # sin^2 + cos^2 == 1 for the rotation fields.
    assert np.allclose(fields["SINALPHA"] ** 2 + fields["COSALPHA"] ** 2, 1.0)
    # On the standard longitude the rotation angle is zero (cos==1, sin==0).
    _lat, lon = grid.latlon("M")
    on_std = np.isclose(lon, grid.stand_lon, atol=1e-6)
    if np.any(on_std):
        assert np.allclose(fields["COSALPHA"][on_std], 1.0, atol=1e-9)
        assert np.allclose(fields["SINALPHA"][on_std], 0.0, atol=1e-9)
    # Northern-hemisphere domain: Coriolis positive everywhere.
    assert np.all(fields["F"] > 0.0)


def test_polar_rotation_helper_on_standard_longitude() -> None:
    sina, cosa = rotation_from_lon_ps(-90.0, truelat1=60.0, stand_lon=-90.0)
    assert abs(sina) < 1.0e-15
    assert abs(cosa - 1.0) < 1.0e-15
    # Hemisphere flips the sign of the rotation.
    s_n, _ = rotation_from_lon_ps(-60.0, truelat1=60.0, stand_lon=-90.0)
    s_s, _ = rotation_from_lon_ps(-60.0, truelat1=-60.0, stand_lon=-90.0)
    assert s_n == pytest.approx(-s_s)


def test_polar_grid_rejects_non_map_proj_2() -> None:
    with pytest.raises(ValueError, match="MAP_PROJ=2"):
        PolarGrid(
            truelat1=60.0, stand_lon=0.0, lat1=45.0, lon1=0.0,
            dx_m=1000.0, dy_m=1000.0, nx=4, ny=4, map_proj=3,
        )
