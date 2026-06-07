"""WRF/WPS map-projection helpers for native static-geog/real-init geometry.

Covers the Lambert-conformal projection (``map_proj=1``, :class:`LambertGrid`)
used by the Canary/Switzerland cases, and the cylindrical-equidistant lat/lon
projection (``map_proj=6`` / WPS PROJ_CASSINI, :class:`LatLonGrid`) used by the
global/rotated ARW configuration. Mercator/Polar (``map_proj=2``/``3``) forward
maps are not yet implemented; callers requesting them fail closed with a clear
error (tracked for a later version).

All transforms mirror WRF ``share/module_llxy.F`` conventions and run on CPU /
NumPy in fp64 (offline init-side geometry, never inside the GPU timestep loop).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping, Any

import numpy as np


WRF_EARTH_RADIUS_M = 6_370_000.0
WRF_GEOGRID_OMEGA_S = 7.292e-5

GridStagger = Literal["M", "U", "V", "CORNER"]


def _as_float_attr(attrs: Mapping[str, Any] | Any, name: str, default: float | None = None) -> float:
    value = _get_attr(attrs, name, default)
    if value is None:
        raise ValueError(f"missing required WPS projection attr {name!r}")
    return float(value)


def _as_int_attr(attrs: Mapping[str, Any] | Any, name: str, default: int | None = None) -> int:
    value = _get_attr(attrs, name, default)
    if value is None:
        raise ValueError(f"missing required WPS projection attr {name!r}")
    return int(value)


def _get_attr(attrs: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    if isinstance(attrs, Mapping):
        for key in (name, name.upper(), name.lower()):
            if key in attrs:
                return attrs[key]
        return default
    for key in (name, name.upper(), name.lower()):
        if hasattr(attrs, key):
            return getattr(attrs, key)
    return default


def normalize_longitude_deg(lon_deg: np.ndarray | float) -> np.ndarray | float:
    """Normalize longitude to WRF's conventional [-180, 180) degree interval."""

    normalized = (np.asarray(lon_deg, dtype=np.float64) + 180.0) % 360.0 - 180.0
    if np.isscalar(lon_deg):
        return float(normalized)
    return normalized


def lambert_cone(truelat1: float, truelat2: float) -> float:
    """Return the WRF secant/tangent Lambert cone factor."""

    phi1 = math.radians(float(truelat1))
    phi2 = math.radians(float(truelat2))
    if abs(phi1 - phi2) < 1.0e-12:
        return math.sin(phi1)
    return math.log(math.cos(phi1) / math.cos(phi2)) / math.log(
        math.tan(math.pi / 4.0 + phi2 / 2.0) / math.tan(math.pi / 4.0 + phi1 / 2.0)
    )


def _lambert_factor(truelat1: float, cone: float) -> float:
    phi1 = math.radians(float(truelat1))
    return math.cos(phi1) * math.tan(math.pi / 4.0 + phi1 / 2.0) ** cone / cone


def lambert_forward(
    lat_deg: np.ndarray | float,
    lon_deg: np.ndarray | float,
    *,
    truelat1: float,
    truelat2: float,
    stand_lon: float,
    ref_lat: float = 0.0,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Project latitude/longitude to WRF Lambert x/y meters.

    The false origin is the latitude ``ref_lat`` on ``stand_lon``. Geogrid uses
    this same formula internally; callers normally use it only to anchor the
    domain center before adding grid-index offsets.
    """

    scalar = np.isscalar(lat_deg) and np.isscalar(lon_deg)
    lat_rad = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon_rad = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    cone = lambert_cone(truelat1, truelat2)
    factor = _lambert_factor(truelat1, cone)
    rho = WRF_EARTH_RADIUS_M * factor / np.tan(math.pi / 4.0 + lat_rad / 2.0) ** cone
    rho0 = WRF_EARTH_RADIUS_M * factor / math.tan(
        math.pi / 4.0 + math.radians(ref_lat) / 2.0
    ) ** cone
    theta = cone * (lon_rad - math.radians(float(stand_lon)))
    x = rho * np.sin(theta)
    y = rho0 - rho * np.cos(theta)
    if scalar:
        return float(x), float(y)
    return x, y


def lambert_inverse(
    x_m: np.ndarray | float,
    y_m: np.ndarray | float,
    *,
    truelat1: float,
    truelat2: float,
    stand_lon: float,
    ref_lat: float = 0.0,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Invert :func:`lambert_forward` to latitude/longitude degrees."""

    scalar = np.isscalar(x_m) and np.isscalar(y_m)
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    cone = lambert_cone(truelat1, truelat2)
    factor = _lambert_factor(truelat1, cone)
    rho0 = WRF_EARTH_RADIUS_M * factor / math.tan(
        math.pi / 4.0 + math.radians(ref_lat) / 2.0
    ) ** cone
    y_from_pole = rho0 - y
    rho = np.sqrt(x * x + y_from_pole * y_from_pole)
    if cone < 0.0:
        rho = -rho
    theta = np.arctan2(x, y_from_pole)
    lat = 2.0 * np.arctan((WRF_EARTH_RADIUS_M * factor / rho) ** (1.0 / cone)) - math.pi / 2.0
    lon = math.radians(float(stand_lon)) + theta / cone
    lat_deg = np.rad2deg(lat)
    lon_deg = normalize_longitude_deg(np.rad2deg(lon))
    if scalar:
        return float(lat_deg), float(lon_deg)
    return lat_deg, lon_deg


def lambert_map_factor(
    lat_deg: np.ndarray | float,
    *,
    truelat1: float,
    truelat2: float,
) -> np.ndarray | float:
    """Return WRF Lambert map factor at latitude."""

    scalar = np.isscalar(lat_deg)
    lat_rad = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    cone = lambert_cone(truelat1, truelat2)
    factor = _lambert_factor(truelat1, cone)
    rho = WRF_EARTH_RADIUS_M * factor / np.tan(math.pi / 4.0 + lat_rad / 2.0) ** cone
    mapfac = cone * rho / (WRF_EARTH_RADIUS_M * np.cos(lat_rad))
    if scalar:
        return float(mapfac)
    return mapfac


# ---------------------------------------------------------------------------
# Cylindrical-equidistant lat/lon (WRF MAP_PROJ=6 / WPS PROJ_CASSINI) helpers
# ---------------------------------------------------------------------------
#
# This mirrors WRF ``share/module_llxy.F`` (``llij_latlon``/``ijll_latlon`` and
# the Cassini ``rotate_coords`` rotation used by ``llij_cassini``/``ijll_cassini``).
# WRF's ``map_proj=6`` is the regular (optionally rotated) cylindrical lat/lon
# grid used by the ARW global configuration. The grid is described by a known
# reference point ``(knowni, knownj)`` at geographic ``(lat1, lon1)`` with
# constant degree increments ``latinc``/``loninc`` along the grid axes, plus an
# optional rotation pole ``(pole_lat, pole_lon)`` and standard longitude
# ``stand_lon``. When the pole is the true North Pole (``pole_lat == 90``) the
# rotation is the identity and the projection reduces to a plain lat/lon grid.
#
# Indices here follow WRF's 1-based convention internally: ``knowni``/``knownj``
# default to 1.0 (the first mass point), and ``i``/``j`` returned by the forward
# transform are 1-based real grid indices, matching ``llij_latlon``.


def _rotate_coords(
    ilat: np.ndarray | float,
    ilon: np.ndarray | float,
    *,
    pole_lat: float,
    pole_lon: float,
    stand_lon: float,
    direction: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Port of WRF ``module_llxy.F`` ``rotate_coords`` (Skamarock 2007).

    ``direction < 0`` rotates geographic -> computational; ``direction >= 0``
    rotates computational -> geographic. All angles are degrees.
    """

    phi_np = math.radians(float(pole_lat))
    lam_np = math.radians(float(pole_lon))
    lam_0 = math.radians(float(stand_lon))
    rlat = np.deg2rad(np.asarray(ilat, dtype=np.float64))
    rlon = np.deg2rad(np.asarray(ilon, dtype=np.float64))

    if direction < 0:
        dlam = math.pi - lam_0
    else:
        dlam = lam_np

    sinphi = (
        math.cos(phi_np) * np.cos(rlat) * np.cos(rlon - dlam)
        + math.sin(phi_np) * np.sin(rlat)
    )
    # Clamp for round-off before arcsin (WRF relies on SQRT(1-sinphi^2) >= 0).
    sinphi = np.clip(sinphi, -1.0, 1.0)
    cosphi = np.sqrt(1.0 - sinphi * sinphi)
    coslam = (
        math.sin(phi_np) * np.cos(rlat) * np.cos(rlon - dlam)
        - math.cos(phi_np) * np.sin(rlat)
    )
    sinlam = np.cos(rlat) * np.sin(rlon - dlam)
    nonzero = cosphi != 0.0
    safe_cosphi = np.where(nonzero, cosphi, 1.0)
    coslam = np.where(nonzero, coslam / safe_cosphi, coslam)
    sinlam = np.where(nonzero, sinlam / safe_cosphi, sinlam)

    olat = np.rad2deg(np.arcsin(sinphi))
    olon = np.rad2deg(np.arctan2(sinlam, coslam) - dlam - lam_0 + lam_np)
    # WRF wraps olon to [-180, 180] with repeated +/-360 loops.
    olon = (olon + 180.0) % 360.0 - 180.0
    return np.asarray(olat, dtype=np.float64), np.asarray(olon, dtype=np.float64)


def latlon_forward(
    lat_deg: np.ndarray | float,
    lon_deg: np.ndarray | float,
    *,
    lat1: float,
    lon1: float,
    latinc: float,
    loninc: float,
    knowni: float = 1.0,
    knownj: float = 1.0,
    pole_lat: float = 90.0,
    pole_lon: float = 0.0,
    stand_lon: float = 0.0,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Geographic lat/lon -> WRF lat/lon grid (i, j); ``llij_cassini``+``llij_cyl``.

    Returns 1-based real grid indices. ``(lat1, lon1)`` is the geographic
    coordinate of the known reference point ``(knowni, knownj)``. When
    ``pole_lat == 90`` (no rotation) this reduces to ``llij_latlon``.
    """

    scalar = np.isscalar(lat_deg) and np.isscalar(lon_deg)
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)

    rotated = abs(float(pole_lat)) != 90.0
    if rotated:
        comp_lat, comp_lon = _rotate_coords(
            lat,
            lon,
            pole_lat=pole_lat,
            pole_lon=pole_lon,
            stand_lon=stand_lon,
            direction=-1,
        )
    else:
        comp_lat, comp_lon = lat, lon

    deltalat = comp_lat - float(lat1)
    deltalon = comp_lon - float(lon1)
    # WRF llij_cyl wraps the longitude delta into [0, 360).
    deltalon = np.where(deltalon < 0.0, deltalon + 360.0, deltalon)
    deltalon = np.where(deltalon > 360.0, deltalon - 360.0, deltalon)

    i = deltalon / float(loninc)
    j = deltalat / float(latinc)

    span_i = 360.0 / float(loninc)
    i = np.where(i <= 0.0, i + span_i, i)
    i = np.where(i > span_i, i - span_i, i)

    i = i + float(knowni)
    j = j + float(knownj)
    if scalar:
        return float(i), float(j)
    return np.asarray(i, dtype=np.float64), np.asarray(j, dtype=np.float64)


def latlon_inverse(
    i: np.ndarray | float,
    j: np.ndarray | float,
    *,
    lat1: float,
    lon1: float,
    latinc: float,
    loninc: float,
    knowni: float = 1.0,
    knownj: float = 1.0,
    pole_lat: float = 90.0,
    pole_lon: float = 0.0,
    stand_lon: float = 0.0,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """WRF lat/lon grid (i, j) -> geographic lat/lon; ``ijll_cyl``+``ijll_cassini``."""

    scalar = np.isscalar(i) and np.isscalar(j)
    i_work = np.asarray(i, dtype=np.float64) - float(knowni)
    j_work = np.asarray(j, dtype=np.float64) - float(knownj)

    comp_lat = float(lat1) + j_work * float(latinc)
    comp_lon = float(lon1) + i_work * float(loninc)

    rotated = abs(float(pole_lat)) != 90.0
    if rotated:
        lat, lon = _rotate_coords(
            comp_lat,
            comp_lon,
            pole_lat=pole_lat,
            pole_lon=pole_lon,
            stand_lon=stand_lon,
            direction=1,
        )
    else:
        lat = comp_lat
        lon = normalize_longitude_deg(comp_lon)

    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if scalar:
        return float(lat), float(lon)
    return lat, lon


def latlon_map_factor(
    comp_lat_deg: np.ndarray | float,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Return WRF cylindrical-equidistant map factors ``(msf_x, msf_y)``.

    For the regular (computational) lat/lon grid the meridional spacing is
    uniform (``msf_y = 1``) while the physical east-west spacing of a constant
    degree-of-longitude grid contracts by ``cos(lat)`` toward the poles, so the
    zonal map factor is ``msf_x = 1 / cos(comp_lat)`` (WPS geogrid convention for
    ``map_proj=6`` / the global ARW grid; near the equator both are ≈ 1). The
    latitude argument is the *computational* latitude (post-rotation), which for
    an unrotated grid equals the geographic latitude.
    """

    scalar = np.isscalar(comp_lat_deg)
    lat_rad = np.deg2rad(np.asarray(comp_lat_deg, dtype=np.float64))
    cos_lat = np.cos(lat_rad)
    # Guard the poles (cos -> 0); WRF sets the polar V map factor to 0 there.
    msf_x = np.where(np.abs(cos_lat) < 1.0e-12, 0.0, 1.0 / np.where(np.abs(cos_lat) < 1.0e-12, 1.0, cos_lat))
    msf_y = np.ones_like(lat_rad)
    if scalar:
        return float(msf_x), float(msf_y)
    return np.asarray(msf_x, dtype=np.float64), np.asarray(msf_y, dtype=np.float64)


def coriolis_from_lat(lat_deg: np.ndarray | float) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Return WRF geogrid F/E Coriolis terms from latitude."""

    scalar = np.isscalar(lat_deg)
    lat_rad = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    two_omega = 2.0 * WRF_GEOGRID_OMEGA_S
    f = two_omega * np.sin(lat_rad)
    e = two_omega * np.cos(lat_rad)
    if scalar:
        return float(f), float(e)
    return f, e


def rotation_from_lon(
    lon_deg: np.ndarray | float,
    *,
    truelat1: float,
    truelat2: float,
    stand_lon: float,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Return WRF mass-point SINALPHA/COSALPHA from Lambert longitude rotation."""

    scalar = np.isscalar(lon_deg)
    cone = lambert_cone(truelat1, truelat2)
    angle = -cone * np.deg2rad(np.asarray(lon_deg, dtype=np.float64) - float(stand_lon))
    sina = np.sin(angle)
    cosa = np.cos(angle)
    if scalar:
        return float(sina), float(cosa)
    return sina, cosa


@dataclass(frozen=True)
class LambertGrid:
    """WRF Lambert grid definition sufficient to derive static coordinate metrics."""

    truelat1: float
    truelat2: float
    stand_lon: float
    cen_lat: float
    cen_lon: float
    dx_m: float
    dy_m: float
    nx: int
    ny: int
    map_proj: int = 1
    moad_cen_lat: float | None = None
    pole_lat: float = 90.0
    pole_lon: float = 0.0
    grid_id: int = 1
    parent_id: int = 1
    parent_grid_ratio: int = 1
    i_parent_start: int = 1
    j_parent_start: int = 1

    def __post_init__(self) -> None:
        if self.map_proj != 1:
            raise ValueError(f"LambertGrid supports only MAP_PROJ=1; got {self.map_proj}")
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("LambertGrid nx/ny must be positive")

    @classmethod
    def from_wps_dataset(cls, dataset: Any) -> "LambertGrid":
        """Build a grid definition from a WPS ``geo_em``/``met_em`` Dataset."""

        return cls(
            map_proj=_as_int_attr(dataset, "MAP_PROJ"),
            truelat1=_as_float_attr(dataset, "TRUELAT1"),
            truelat2=_as_float_attr(dataset, "TRUELAT2"),
            stand_lon=_as_float_attr(dataset, "STAND_LON"),
            cen_lat=_as_float_attr(dataset, "CEN_LAT"),
            cen_lon=_as_float_attr(dataset, "CEN_LON"),
            moad_cen_lat=_as_float_attr(dataset, "MOAD_CEN_LAT"),
            pole_lat=_as_float_attr(dataset, "POLE_LAT", 90.0),
            pole_lon=_as_float_attr(dataset, "POLE_LON", 0.0),
            dx_m=_as_float_attr(dataset, "DX"),
            dy_m=_as_float_attr(dataset, "DY"),
            nx=int(len(dataset.dimensions["west_east"])),
            ny=int(len(dataset.dimensions["south_north"])),
            grid_id=_as_int_attr(dataset, "grid_id", 1),
            parent_id=_as_int_attr(dataset, "parent_id", 1),
            parent_grid_ratio=_as_int_attr(dataset, "parent_grid_ratio", 1),
            i_parent_start=_as_int_attr(dataset, "i_parent_start", 1),
            j_parent_start=_as_int_attr(dataset, "j_parent_start", 1),
        )

    def xy(self, stagger: GridStagger = "M") -> tuple[np.ndarray, np.ndarray]:
        """Return absolute Lambert x/y meters for a WRF mass/U/V/corner grid."""

        x_center, y_center = lambert_forward(
            self.cen_lat,
            self.cen_lon,
            truelat1=self.truelat1,
            truelat2=self.truelat2,
            stand_lon=self.stand_lon,
        )
        if stagger == "M":
            i = np.arange(self.nx, dtype=np.float64) - (self.nx - 1.0) / 2.0
            j = np.arange(self.ny, dtype=np.float64) - (self.ny - 1.0) / 2.0
        elif stagger == "U":
            i = np.arange(self.nx + 1, dtype=np.float64) - self.nx / 2.0
            j = np.arange(self.ny, dtype=np.float64) - (self.ny - 1.0) / 2.0
        elif stagger == "V":
            i = np.arange(self.nx, dtype=np.float64) - (self.nx - 1.0) / 2.0
            j = np.arange(self.ny + 1, dtype=np.float64) - self.ny / 2.0
        elif stagger == "CORNER":
            i = np.arange(self.nx + 1, dtype=np.float64) - self.nx / 2.0
            j = np.arange(self.ny + 1, dtype=np.float64) - self.ny / 2.0
        else:
            raise ValueError(f"unknown stagger {stagger!r}")
        return x_center + i[None, :] * self.dx_m, y_center + j[:, None] * self.dy_m

    def latlon(self, stagger: GridStagger = "M") -> tuple[np.ndarray, np.ndarray]:
        """Derive WRF latitude/longitude fields for a stagger."""

        x, y = self.xy(stagger)
        return lambert_inverse(
            x,
            y,
            truelat1=self.truelat1,
            truelat2=self.truelat2,
            stand_lon=self.stand_lon,
        )

    def map_factor(self, stagger: GridStagger = "M") -> np.ndarray:
        """Derive the isotropic WRF map factor for a stagger."""

        lat, _lon = self.latlon(stagger)
        return np.asarray(
            lambert_map_factor(lat, truelat1=self.truelat1, truelat2=self.truelat2),
            dtype=np.float64,
        )

    def coriolis(self) -> tuple[np.ndarray, np.ndarray]:
        """Derive mass-grid F/E Coriolis fields."""

        lat, _lon = self.latlon("M")
        f, e = coriolis_from_lat(lat)
        return np.asarray(f, dtype=np.float64), np.asarray(e, dtype=np.float64)

    def rotation(self) -> tuple[np.ndarray, np.ndarray]:
        """Derive mass-grid SINALPHA/COSALPHA fields."""

        _lat, lon = self.latlon("M")
        sina, cosa = rotation_from_lon(
            lon,
            truelat1=self.truelat1,
            truelat2=self.truelat2,
            stand_lon=self.stand_lon,
        )
        return np.asarray(sina, dtype=np.float64), np.asarray(cosa, dtype=np.float64)

    def derive_fields(self) -> dict[str, np.ndarray]:
        """Derive the coordinate/map-factor fields present in ``geo_em``."""

        xlat_m, xlong_m = self.latlon("M")
        xlat_u, xlong_u = self.latlon("U")
        xlat_v, xlong_v = self.latlon("V")
        xlat_c, xlong_c = self.latlon("CORNER")
        mapfac_m = self.map_factor("M")
        mapfac_u = self.map_factor("U")
        mapfac_v = self.map_factor("V")
        f, e = self.coriolis()
        sina, cosa = self.rotation()
        return {
            "XLAT_M": xlat_m,
            "XLONG_M": xlong_m,
            "XLAT_U": xlat_u,
            "XLONG_U": xlong_u,
            "XLAT_V": xlat_v,
            "XLONG_V": xlong_v,
            "XLAT_C": xlat_c,
            "XLONG_C": xlong_c,
            "CLAT": xlat_m,
            "CLONG": xlong_m,
            "MAPFAC_M": mapfac_m,
            "MAPFAC_MX": mapfac_m,
            "MAPFAC_MY": mapfac_m,
            "MAPFAC_U": mapfac_u,
            "MAPFAC_UX": mapfac_u,
            "MAPFAC_UY": mapfac_u,
            "MAPFAC_V": mapfac_v,
            "MAPFAC_VX": mapfac_v,
            "MAPFAC_VY": mapfac_v,
            "F": f,
            "E": e,
            "SINALPHA": sina,
            "COSALPHA": cosa,
        }


@dataclass(frozen=True)
class LatLonGrid:
    """WRF cylindrical-equidistant lat/lon grid (``map_proj=6`` / Cassini).

    The grid is regular in degrees: mass points are spaced ``loninc`` in
    longitude and ``latinc`` in latitude, with the SW mass point at geographic
    ``(lat1, lon1)``. ``pole_lat``/``pole_lon`` describe the rotation pole; when
    ``pole_lat == 90`` (true North Pole) the rotation is the identity and this is
    a plain global/regional lat/lon grid (e.g. AIFS-style cylindrical input).
    Rotated lat/lon (``pole_lat != 90``) is the WRF Cassini convention used for
    the global ARW dycore.

    C-grid staggering: mass (M) points anchor the grid; U points are offset
    +0.5 grid cell in i (longitude), V points +0.5 in j (latitude). The U/V
    longitude/latitude increments are reused unchanged.
    """

    lat1: float
    lon1: float
    latinc: float
    loninc: float
    nx: int
    ny: int
    map_proj: int = 6
    knowni: float = 1.0
    knownj: float = 1.0
    pole_lat: float = 90.0
    pole_lon: float = 0.0
    stand_lon: float = 0.0
    grid_id: int = 1
    parent_id: int = 1
    parent_grid_ratio: int = 1
    i_parent_start: int = 1
    j_parent_start: int = 1

    def __post_init__(self) -> None:
        if self.map_proj != 6:
            raise ValueError(f"LatLonGrid supports only MAP_PROJ=6; got {self.map_proj}")
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("LatLonGrid nx/ny must be positive")
        if self.latinc == 0.0 or self.loninc == 0.0:
            raise ValueError("LatLonGrid latinc/loninc must be non-zero")

    @property
    def is_rotated(self) -> bool:
        """True when the rotation pole differs from the geographic North Pole."""

        return abs(float(self.pole_lat)) != 90.0

    @classmethod
    def from_wps_dataset(cls, dataset: Any) -> "LatLonGrid":
        """Build a lat/lon grid from a WPS ``geo_em``/``met_em`` Dataset.

        WPS writes the increments as ``DX``/``DY`` *in degrees* for ``map_proj=6``
        and the SW corner as ``corner_lats``/``corner_lons`` / ``cen_lat`` etc.
        We read the increments from ``DX``/``DY`` and the SW reference from
        ``corner_lats``/``corner_lons`` (index 0 = unstaggered SW corner) when
        available, else fall back to ``cen_lat``/``cen_lon`` recentred.
        """

        map_proj = _as_int_attr(dataset, "MAP_PROJ")
        nx = int(len(dataset.dimensions["west_east"]))
        ny = int(len(dataset.dimensions["south_north"]))
        loninc = _as_float_attr(dataset, "DX")
        latinc = _as_float_attr(dataset, "DY")
        stand_lon = _as_float_attr(dataset, "STAND_LON", 0.0)
        pole_lat = _as_float_attr(dataset, "POLE_LAT", 90.0)
        pole_lon = _as_float_attr(dataset, "POLE_LON", 0.0)

        corner_lats = _get_attr(dataset, "corner_lats")
        corner_lons = _get_attr(dataset, "corner_lons")
        if corner_lats is not None and corner_lons is not None:
            lat1 = float(np.asarray(corner_lats)[0])
            lon1 = float(np.asarray(corner_lons)[0])
        else:
            cen_lat = _as_float_attr(dataset, "CEN_LAT")
            cen_lon = _as_float_attr(dataset, "CEN_LON")
            lat1 = cen_lat - (ny - 1) / 2.0 * latinc
            lon1 = cen_lon - (nx - 1) / 2.0 * loninc

        return cls(
            map_proj=map_proj,
            lat1=lat1,
            lon1=lon1,
            latinc=latinc,
            loninc=loninc,
            nx=nx,
            ny=ny,
            pole_lat=pole_lat,
            pole_lon=pole_lon,
            stand_lon=stand_lon,
            grid_id=_as_int_attr(dataset, "grid_id", 1),
            parent_id=_as_int_attr(dataset, "parent_id", 1),
            parent_grid_ratio=_as_int_attr(dataset, "parent_grid_ratio", 1),
            i_parent_start=_as_int_attr(dataset, "i_parent_start", 1),
            j_parent_start=_as_int_attr(dataset, "j_parent_start", 1),
        )

    def _index_grids(self, stagger: GridStagger) -> tuple[np.ndarray, np.ndarray]:
        """Return 1-based (i, j) index meshes for a stagger (knowni/j-based)."""

        if stagger == "M":
            i = np.arange(self.nx, dtype=np.float64)
            j = np.arange(self.ny, dtype=np.float64)
            ish, jsh = 0.0, 0.0
        elif stagger == "U":
            i = np.arange(self.nx + 1, dtype=np.float64)
            j = np.arange(self.ny, dtype=np.float64)
            ish, jsh = -0.5, 0.0
        elif stagger == "V":
            i = np.arange(self.nx, dtype=np.float64)
            j = np.arange(self.ny + 1, dtype=np.float64)
            ish, jsh = 0.0, -0.5
        elif stagger == "CORNER":
            i = np.arange(self.nx + 1, dtype=np.float64)
            j = np.arange(self.ny + 1, dtype=np.float64)
            ish, jsh = -0.5, -0.5
        else:
            raise ValueError(f"unknown stagger {stagger!r}")
        # Mass point k (0-based) sits at 1-based grid index knowni + k. Staggered
        # points are shifted -0.5 cell so the U/V faces straddle the mass cell.
        ii = (self.knowni + i + ish)[None, :] * np.ones((j.size, 1))
        jj = (self.knownj + j + jsh)[:, None] * np.ones((1, i.size))
        return ii, jj

    def latlon(self, stagger: GridStagger = "M") -> tuple[np.ndarray, np.ndarray]:
        """Derive WRF latitude/longitude fields for a stagger."""

        ii, jj = self._index_grids(stagger)
        return latlon_inverse(
            ii,
            jj,
            lat1=self.lat1,
            lon1=self.lon1,
            latinc=self.latinc,
            loninc=self.loninc,
            knowni=self.knowni,
            knownj=self.knownj,
            pole_lat=self.pole_lat,
            pole_lon=self.pole_lon,
            stand_lon=self.stand_lon,
        )

    def _comp_lat(self, stagger: GridStagger) -> np.ndarray:
        """Computational (post-rotation) latitude mesh for a stagger."""

        ii, jj = self._index_grids(stagger)
        # Computational latitude is the un-rotated linear grid latitude.
        return self.lat1 + (jj - self.knownj) * self.latinc

    def map_factor(self, stagger: GridStagger = "M") -> tuple[np.ndarray, np.ndarray]:
        """Derive WRF (msf_x, msf_y) map factors for a stagger."""

        comp_lat = self._comp_lat(stagger)
        msf_x, msf_y = latlon_map_factor(comp_lat)
        return np.asarray(msf_x, dtype=np.float64), np.asarray(msf_y, dtype=np.float64)

    def coriolis(self) -> tuple[np.ndarray, np.ndarray]:
        """Derive mass-grid F/E Coriolis fields from geographic latitude."""

        lat, _lon = self.latlon("M")
        f, e = coriolis_from_lat(lat)
        return np.asarray(f, dtype=np.float64), np.asarray(e, dtype=np.float64)

    def derive_fields(self) -> dict[str, np.ndarray]:
        """Derive the coordinate/map-factor fields present in ``geo_em``."""

        xlat_m, xlong_m = self.latlon("M")
        xlat_u, xlong_u = self.latlon("U")
        xlat_v, xlong_v = self.latlon("V")
        xlat_c, xlong_c = self.latlon("CORNER")
        mapfac_mx, mapfac_my = self.map_factor("M")
        mapfac_ux, mapfac_uy = self.map_factor("U")
        mapfac_vx, mapfac_vy = self.map_factor("V")
        f, e = self.coriolis()
        # Unrotated cylindrical grid is north-aligned: no wind rotation.
        sina = np.zeros_like(xlat_m)
        cosa = np.ones_like(xlat_m)
        return {
            "XLAT_M": xlat_m,
            "XLONG_M": xlong_m,
            "XLAT_U": xlat_u,
            "XLONG_U": xlong_u,
            "XLAT_V": xlat_v,
            "XLONG_V": xlong_v,
            "XLAT_C": xlat_c,
            "XLONG_C": xlong_c,
            "CLAT": xlat_m,
            "CLONG": xlong_m,
            "MAPFAC_M": mapfac_mx,
            "MAPFAC_MX": mapfac_mx,
            "MAPFAC_MY": mapfac_my,
            "MAPFAC_U": mapfac_ux,
            "MAPFAC_UX": mapfac_ux,
            "MAPFAC_UY": mapfac_uy,
            "MAPFAC_V": mapfac_vx,
            "MAPFAC_VX": mapfac_vx,
            "MAPFAC_VY": mapfac_vy,
            "F": f,
            "E": e,
            "SINALPHA": sina,
            "COSALPHA": cosa,
        }
