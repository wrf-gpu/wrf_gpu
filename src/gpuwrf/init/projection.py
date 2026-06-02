"""WRF/WPS Lambert projection helpers for v0.3.0 native static geog ingest."""

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
