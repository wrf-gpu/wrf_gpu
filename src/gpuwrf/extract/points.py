"""EXTRACT_SPEC V1 points: frozen point CSVs, nearest mass cell per nest, native profiles (host only).

Point CSV formats are accepted only with their exact frozen ALISIOS headers (no guessing):
* ``stations``  (POINTS_V1/V2.csv): id,lat,lon,height_m,network,name,kind,height_source,coordinate_source
* ``grid_bound`` (SEA_TRANSECTS_V1.csv): id,kind,nest,island,zone,y,x,lat,lon,hgt_m,dem_m,line_order
  (cells frozen on a reference geometry; ``hgt_m`` is a model HGT, never a station height).
Lists are never generated here: the sea/north-slope selection is an ALISIOS gate.

Nearest cell = WT1 ``teacher_extract_v2.mapped_points``: unit-sphere xyz of the f32 grid
(``np.deg2rad`` of f32 XLAT/XLONG) against the f64 point, Euclidean chord, distance
``2*6371000*arcsin(min(1, chord/2))``. A point is IN the nest only inside the mass-point hull
(fractional index from the local grid Jacobian in [0, n-1]); no extrapolation. Statuses per
(point, nest): OK, NO_COORDINATE, OUTSIDE_DOMAIN, BEYOND_MAX_DISTANCE, GEOMETRY_MISMATCH
(grid-bound row whose frozen cell is not this run's cell). Height: OK / HEIGHT_UNKNOWN /
NOT_A_STATION (grid-bound). Profiles use only OK points.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

EARTH_RADIUS_M = 6371000.0
DEFAULT_MAX_DISTANCE_CELLS = 1.5  # WT1: 1.5 km on the 1 km d03 grid; explicit per grid = 1.5 * dx
HULL_TOLERANCE_CELLS = 1e-6

FORMATS = {
    "stations": ("id", "lat", "lon", "height_m", "network", "name", "kind", "height_source",
                 "coordinate_source"),
    "grid_bound": ("id", "kind", "nest", "island", "zone", "y", "x", "lat", "lon", "hgt_m", "dem_m",
                   "line_order"),
}
POINT_STATUS = ("OK", "NO_COORDINATE", "OUTSIDE_DOMAIN", "BEYOND_MAX_DISTANCE", "GEOMETRY_MISMATCH")

# Native-level profile variables (time, point, level), float32.
PROFILE_FIELDS = ("z_msl_m", "z_agl_m", "theta_K", "T_K", "pressure_Pa", "QV_kg_kg", "QCLOUD_kg_kg",
                  "QICE_kg_kg", "QNCLOUD_per_kg", "CLDFRA", "U_grid_ms", "V_grid_ms", "U_ms", "V_ms")
PROFILE_RAW = ("T", "P", "PB", "PH", "PHB", "QVAPOR", "QCLOUD", "QICE", "QNCLOUD", "CLDFRA", "U", "V",
               "HGT", "COSALPHA", "SINALPHA")
PROFILE_REQUIRED = ("T", "P", "PB", "PH", "PHB", "QVAPOR", "QCLOUD", "CLDFRA", "U", "V", "HGT",
                    "COSALPHA", "SINALPHA")
PROFILE_UNITS = {
    "z_msl_m": "m MSL", "z_agl_m": "m AGL", "theta_K": "K", "T_K": "K", "pressure_Pa": "Pa",
    "QV_kg_kg": "kg kg-1 (dry air)", "QCLOUD_kg_kg": "kg kg-1 (dry air)", "QICE_kg_kg": "kg kg-1 (dry air)",
    "QNCLOUD_per_kg": "kg-1 (number per kg dry air)", "CLDFRA": "1",
    "U_grid_ms": "m s-1 (grid relative)", "V_grid_ms": "m s-1 (grid relative)",
    "U_ms": "m s-1 (earth relative)", "V_ms": "m s-1 (earth relative)",
}
PROFILE_DEFINITIONS = {
    "z_msl_m": "f4((z_w[k]+z_w[k+1])/2), z_w=(f8 PH+f8 PHB)/9.81 (WT1 frame_fields)",
    "z_agl_m": "z_msl_m - HGT (f4)",
    "theta_K": "f4(f8(T)+300) (WT1)",
    "T_K": "f4(theta8*(p8/100000)**(2/7)), p8 = f8 P + f8 PB (WT1 frame_fields)",
    "pressure_Pa": "f4(f8 P + f8 PB)",
    "QV_kg_kg": "QVAPOR as written", "QCLOUD_kg_kg": "QCLOUD as written", "QICE_kg_kg": "QICE as written",
    "QNCLOUD_per_kg": "QNCLOUD as written (only mp_physics=28; else NOT_AVAILABLE)",
    "CLDFRA": "CLDFRA as written",
    "U_grid_ms": "f4((f8 U[k,y,x]+f8 U[k,y,x+1])/2)", "V_grid_ms": "f4((f8 V[k,y,x]+f8 V[k,y+1,x])/2)",
    "U_ms": "f4(ug8*COSALPHA - vg8*SINALPHA) from the f8 destaggered winds (WT1 profile_table)",
    "V_ms": "f4(vg8*COSALPHA + ug8*SINALPHA) (WT1 profile_table)",
}


@dataclass(frozen=True)
class Point:
    id: str
    source: str
    row: int
    kind: str
    lat: float | None
    lon: float | None
    height_m: float | None
    height_status: str
    bound: tuple[str, int, int] | None = None
    extra: tuple[tuple[str, str], ...] = field(default=())


@dataclass(frozen=True)
class PointSource:
    path: str
    sha256: str
    format: str
    rows: int


def _float(text: str):
    text = text.strip()
    if text == "":
        return None
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"non-finite coordinate/height {text!r}")
    return value


def load_point_csvs(paths: Sequence[Path]) -> tuple[tuple[Point, ...], tuple[PointSource, ...]]:
    points: list[Point] = []
    sources: list[PointSource] = []
    seen: dict[str, str] = {}
    for path in paths:
        path = Path(path).resolve()
        data = path.read_bytes()
        reader = csv.reader(io.StringIO(data.decode("utf-8")))
        header = tuple(next(reader))
        fmt = next((k for k, h in FORMATS.items() if h == header), None)
        if fmt is None:
            raise ValueError(f"point CSV {path}: unknown header {header}; accepted: {FORMATS}")
        n = 0
        for i, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError(f"point CSV {path} line {i}: {len(row)} columns, expected {len(header)}")
            rec = dict(zip(header, row))
            pid = rec["id"].strip()
            if not pid:
                raise ValueError(f"point CSV {path} line {i}: empty id")
            if pid in seen:
                raise ValueError(f"duplicate point id {pid!r} ({seen[pid]} and {path.name})")
            seen[pid] = path.name
            lat, lon = _float(rec["lat"]), _float(rec["lon"])
            if (lat is None) != (lon is None):
                raise ValueError(f"point {pid}: only one of lat/lon given")
            if lat is not None and not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 360.0):
                raise ValueError(f"point {pid}: coordinate out of range")
            if fmt == "stations":
                height = _float(rec["height_m"])
                points.append(Point(pid, path.name, i, rec["kind"].strip(), lat, lon, height,
                                    "OK" if height is not None else "HEIGHT_UNKNOWN",
                                    extra=tuple((k, rec[k]) for k in ("network", "name", "height_source",
                                                                      "coordinate_source"))))
            else:
                nest = rec["nest"].strip()
                y, x = int(rec["y"]), int(rec["x"])
                points.append(Point(pid, path.name, i, rec["kind"].strip(), lat, lon, None, "NOT_A_STATION",
                                    bound=(nest, y, x),
                                    extra=tuple((k, rec[k]) for k in ("island", "zone", "hgt_m", "dem_m",
                                                                      "line_order"))))
            n += 1
        sources.append(PointSource(str(path), hashlib.sha256(data).hexdigest(), fmt, n))
    return tuple(points), tuple(sources)


def earth_xyz(lat, lon):
    """WT1 teacher_extract_v2.earth_xyz (dtype follows the input: f32 grid, f64 point)."""
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    return np.column_stack((np.cos(lat).ravel() * np.cos(lon).ravel(),
                            np.cos(lat).ravel() * np.sin(lon).ravel(), np.sin(lat).ravel()))


def _nearest(grid_xyz64: np.ndarray, p: np.ndarray) -> tuple[int, float]:
    d2 = ((grid_xyz64 - p) ** 2).sum(axis=1)
    ix = int(np.argmin(d2))
    chord = float(np.sqrt(d2[ix]))
    return ix, float(2 * EARTH_RADIUS_M * np.arcsin(min(1, chord / 2)))


def _fractional_index(xlat, xlong, j, i, lat, lon):
    """Fractional (j, i) of the point from the local grid Jacobian at cell (j, i), local tangent plane."""
    ny, nx = xlat.shape
    coslat = math.cos(math.radians(lat))
    k = math.pi / 180.0 * EARTH_RADIUS_M

    def en(jj, ii):
        return np.array([(float(xlong[jj, ii]) - lon) * coslat * k, (float(xlat[jj, ii]) - lat) * k])

    i0, i1 = max(i - 1, 0), min(i + 1, nx - 1)
    j0, j1 = max(j - 1, 0), min(j + 1, ny - 1)
    di = (en(j, i1) - en(j, i0)) / (i1 - i0)
    dj = (en(j1, i) - en(j0, i)) / (j1 - j0)
    jac = np.column_stack((di, dj))
    fi, fj = np.linalg.solve(jac, -en(j, i))
    return j + fj, i + fi


@dataclass(frozen=True)
class Mapping1:
    point: Point
    domain: str
    status: str
    y: int | None = None
    x: int | None = None
    distance_m: float | None = None
    cell_lat: float | None = None
    cell_lon: float | None = None
    hgt_m: float | None = None
    landmask: float | None = None
    frac_y: float | None = None
    frac_x: float | None = None
    note: str = ""

    def row(self) -> dict:
        p = self.point
        diff = (p.height_m - self.hgt_m) if (p.height_m is not None and self.hgt_m is not None) else None
        out = {"id": p.id, "source": p.source, "source_row": p.row, "kind": p.kind, "domain": self.domain,
               "status": self.status, "height_status": p.height_status,
               "lat": p.lat, "lon": p.lon, "height_m": p.height_m,
               "grid_y": self.y, "grid_x": self.x, "distance_m": self.distance_m,
               "cell_lat": self.cell_lat, "cell_lon": self.cell_lon, "model_HGT_m": self.hgt_m,
               "height_minus_model_HGT_m": diff, "LANDMASK": self.landmask,
               "frac_y": self.frac_y, "frac_x": self.frac_x,
               "bound_nest": p.bound[0] if p.bound else None, "bound_y": p.bound[1] if p.bound else None,
               "bound_x": p.bound[2] if p.bound else None, "note": self.note}
        out.update({f"src_{k}": v for k, v in p.extra})
        return out


def _bound_mismatch(p: Point, xlat, xlong, hgt, landmask, nearest) -> str:
    """Why a frozen (nest, y, x) row does not describe this run's cell ('' when it does).

    Checked: cell inside the grid; f32 XLAT/XLONG == frozen lat/lon; the cell is the nearest cell of
    the frozen lat/lon; f32 HGT == frozen hgt_m (when given); sea rows (kind sea*) have LANDMASK 0."""
    _, by, bx = p.bound
    ny, nx = xlat.shape
    if not (0 <= by < ny and 0 <= bx < nx):
        return f"frozen cell ({by},{bx}) outside this run's {ny}x{nx} grid"
    if xlat[by, bx] != np.float32(p.lat) or xlong[by, bx] != np.float32(p.lon):
        return f"frozen cell ({by},{bx}) lat/lon differ from this run's cell"
    if (by, bx) != tuple(nearest):
        return f"frozen cell ({by},{bx}) is not this run's nearest cell {tuple(nearest)} of the frozen lat/lon"
    extra = dict(p.extra)
    frozen_h = extra.get("hgt_m")
    if frozen_h not in (None, ""):
        if hgt[by, bx] != np.float32(float(frozen_h)):
            return f"frozen hgt_m {frozen_h} != this run's HGT {float(hgt[by, bx])!r} at ({by},{bx})"
    if str(p.kind).startswith("sea") and landmask[by, bx] != 0:
        return f"frozen sea cell ({by},{bx}) has LANDMASK {float(landmask[by, bx])} in this run"
    return ""


def map_points(points: Sequence[Point], domain: str, static: Mapping[str, np.ndarray],
               max_distance_m: float) -> tuple[Mapping1, ...]:
    xlat = np.asarray(static["XLAT"], np.float32)
    xlong = np.asarray(static["XLONG"], np.float32)
    hgt = np.asarray(static["HGT"], np.float32)
    landmask = np.asarray(static["LANDMASK"], np.float32)
    ny, nx = xlat.shape
    grid64 = earth_xyz(xlat, xlong).astype(np.float64)
    out = []
    for p in points:
        if p.lat is None:
            out.append(Mapping1(p, domain, "NO_COORDINATE"))
            continue
        ix, dist = _nearest(grid64, earth_xyz([p.lat], [p.lon])[0])
        j, i = np.unravel_index(ix, xlat.shape)
        j, i = int(j), int(i)
        fy, fx = _fractional_index(xlat, xlong, j, i, p.lat, p.lon)
        common = dict(y=j, x=i, distance_m=dist, cell_lat=float(xlat[j, i]), cell_lon=float(xlong[j, i]),
                      hgt_m=float(hgt[j, i]), landmask=float(landmask[j, i]), frac_y=float(fy), frac_x=float(fx))
        if p.bound is not None and p.bound[0] == domain:
            # A frozen grid-bound point is checked against its own nest BEFORE the hull test: a shifted
            # footprint must surface as GEOMETRY_MISMATCH, never as OUTSIDE_DOMAIN.
            reason = _bound_mismatch(p, xlat, xlong, hgt, landmask, (j, i))
            if reason:
                out.append(Mapping1(p, domain, "GEOMETRY_MISMATCH", **common, note=reason))
                continue
        tol = HULL_TOLERANCE_CELLS
        if not (-tol <= fy <= ny - 1 + tol and -tol <= fx <= nx - 1 + tol):
            out.append(Mapping1(p, domain, "OUTSIDE_DOMAIN", **common))
            continue
        if dist > max_distance_m:
            out.append(Mapping1(p, domain, "BEYOND_MAX_DISTANCE", **common))
            continue
        out.append(Mapping1(p, domain, "OK", **common))
    return tuple(out)


def profile_columns(raw: Mapping[str, np.ndarray], jj: np.ndarray, ii: np.ndarray, *,
                    qncloud_available: bool, qice_expected: bool | None = None) -> tuple[dict, dict]:
    """Native-level columns (point, level) at the given cells; WT1 frame_fields/profile_table order."""
    missing = [k for k in PROFILE_REQUIRED if k not in raw]
    status: dict[str, dict] = {}
    out: dict[str, np.ndarray] = {}
    if missing:
        for f in PROFILE_FIELDS:
            status[f] = {"status": "MISSING_INPUT", "missing": missing}
        return out, status

    def col(name):  # (nz, n) -> (n, nz)
        return np.asarray(raw[name])[:, jj, ii].T

    zw = (col("PH").astype("f8") + col("PHB").astype("f8")) / 9.81  # (n, nz+1)
    zmsl = ((zw[:, :-1] + zw[:, 1:]) / 2).astype("f4")
    out["z_msl_m"] = zmsl
    out["z_agl_m"] = zmsl - np.asarray(raw["HGT"], np.float32)[jj, ii][:, None]
    pressure = col("P").astype("f8") + col("PB").astype("f8")
    theta = col("T").astype("f8") + 300
    out["theta_K"] = theta.astype("f4")
    out["T_K"] = (theta * (pressure / 100000) ** (2 / 7)).astype("f4")
    out["pressure_Pa"] = pressure.astype("f4")
    out["QV_kg_kg"] = col("QVAPOR").astype("f4")
    out["QCLOUD_kg_kg"] = col("QCLOUD").astype("f4")
    out["CLDFRA"] = col("CLDFRA").astype("f4")
    u, v = np.asarray(raw["U"]), np.asarray(raw["V"])
    ug = (u[:, jj, ii].T.astype("f8") + u[:, jj, ii + 1].T.astype("f8")) / 2
    vg = (v[:, jj, ii].T.astype("f8") + v[:, jj + 1, ii].T.astype("f8")) / 2
    c = np.asarray(raw["COSALPHA"], np.float32)[jj, ii][:, None]
    s = np.asarray(raw["SINALPHA"], np.float32)[jj, ii][:, None]
    out["U_grid_ms"] = ug.astype("f4")
    out["V_grid_ms"] = vg.astype("f4")
    out["U_ms"] = (ug * c - vg * s).astype("f4")
    out["V_ms"] = (vg * c + ug * s).astype("f4")
    for f in out:
        status[f] = {"status": "OK"}
    if "QICE" in raw:
        out["QICE_kg_kg"] = col("QICE").astype("f4")
        status["QICE_kg_kg"] = {"status": "OK"}
    elif qice_expected:
        status["QICE_kg_kg"] = {"status": "MISSING", "reason": "the scheme carries QICE but the payload lacks it"}
    else:
        status["QICE_kg_kg"] = {"status": "NOT_AVAILABLE", "reason": "QICE not in this microphysics schema"}
    if qncloud_available and "QNCLOUD" in raw:
        out["QNCLOUD_per_kg"] = col("QNCLOUD").astype("f4")
        status["QNCLOUD_per_kg"] = {"status": "OK"}
    elif qncloud_available:
        status["QNCLOUD_per_kg"] = {"status": "MISSING", "reason": "mp_physics=28 but the payload lacks QNCLOUD"}
    else:
        status["QNCLOUD_per_kg"] = {"status": "NOT_AVAILABLE",
                                  "reason": "QNCLOUD only under mp_physics=28 (Addendum 1)"}
    return out, status
