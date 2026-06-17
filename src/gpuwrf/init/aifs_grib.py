"""AIFS GRIB2 reader for the v0.3.0 native metgrid-equivalent ingest.

The reader intentionally keys messages by the authoritative ``Vtable.AIFS_PURE``
GRIB2 triplets and level codes. ECMWF soil moisture decodes with shortName
``unknown``, so short-name matching is used only for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import struct
from typing import Iterable

import numpy as np

from gpuwrf.config.paths import aifs_vtable_path


# Env-overridable via GPUWRF_AIFS_VTABLE (config.paths.aifs_vtable_path); an
# explicit vtable_path= argument to the reader still takes precedence.
DEFAULT_AIFS_VTABLE = aifs_vtable_path()

ISOBARIC_LEVELS_PA = (
    100000.0,
    92500.0,
    85000.0,
    70000.0,
    60000.0,
    50000.0,
    40000.0,
    30000.0,
    25000.0,
    20000.0,
    15000.0,
    10000.0,
    5000.0,
)

WPS_SURFACE_LEVEL = 200100.0
WPS_MEAN_SEA_LEVEL = 201300.0
WPS_MISSING_INT = 2147483647

GRIB2_LEVEL_TYPE_BY_CODE = {
    1: "surface",
    100: "isobaricInhPa",
    101: "meanSea",
    103: "heightAboveGround",
    106: "depthBelowLandLayer",
}


@dataclass(frozen=True)
class VtableEntry:
    """One data row from ``Vtable.AIFS_PURE``."""

    line_number: int
    grib1_param: int
    grib1_level_type: int
    from_level: int | None
    to_level: int | None
    metgrid_name: str
    units: str
    description: str
    discipline: int
    category: int
    parameter: int
    grib2_level_code: int
    raw_line: str

    @property
    def schema_name(self) -> str:
        """Return the frozen-schema variable name for this field."""

        return "GHT" if self.metgrid_name == "HGT" else self.metgrid_name

    @property
    def level_type(self) -> str:
        return GRIB2_LEVEL_TYPE_BY_CODE.get(self.grib2_level_code, f"code:{self.grib2_level_code}")

    @property
    def soil_depth_mm(self) -> tuple[int, int] | None:
        if self.grib2_level_code != 106:
            return None
        if self.from_level is None or self.to_level is None:
            return None
        return (self.from_level, self.to_level)


@dataclass(frozen=True)
class AIFSGrid:
    """Regular lat-lon AIFS source grid metadata."""

    grid_type: str
    ni: int
    nj: int
    latitude_first: float
    longitude_first: float
    latitude_last: float
    longitude_last: float
    di: float
    dj: float
    j_scans_positively: bool
    i_scans_negatively: bool

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nj, self.ni)

    @property
    def latitudes(self) -> np.ndarray:
        step = self.dj if self.j_scans_positively else -self.dj
        return (self.latitude_first + step * np.arange(self.nj, dtype=np.float64)).astype(np.float64)

    @property
    def longitudes_0_360(self) -> np.ndarray:
        step = -self.di if self.i_scans_negatively else self.di
        return np.mod(self.longitude_first + step * np.arange(self.ni, dtype=np.float64), 360.0)

    @property
    def longitudes_180(self) -> np.ndarray:
        lon = self.longitudes_0_360
        return ((lon + 180.0) % 360.0) - 180.0

    def compatible_with(self, other: "AIFSGrid") -> bool:
        return (
            self.grid_type == other.grid_type
            and self.ni == other.ni
            and self.nj == other.nj
            and np.isclose(self.latitude_first, other.latitude_first)
            and np.isclose(self.longitude_first, other.longitude_first)
            and np.isclose(self.latitude_last, other.latitude_last)
            and np.isclose(self.longitude_last, other.longitude_last)
            and np.isclose(self.di, other.di)
            and np.isclose(self.dj, other.dj)
            and self.j_scans_positively == other.j_scans_positively
            and self.i_scans_negatively == other.i_scans_negatively
        )


@dataclass(frozen=True)
class AIFSMessage:
    """A decoded GRIB message matched to a Vtable entry."""

    entry: VtableEntry
    field_name: str
    wps_field_name: str
    short_name: str
    grib_name: str
    units: str
    level_type: str
    wps_level: float
    level_pa: float | None
    height_m: float | None
    soil_depth_mm: tuple[int, int] | None
    data: np.ndarray
    grib_keys: dict[str, int | float | str | None]

    @property
    def provenance(self) -> dict[str, object]:
        return {
            "field": self.field_name,
            "wps_field": self.wps_field_name,
            "short_name": self.short_name,
            "grib_name": self.grib_name,
            "units": self.units,
            "level_type": self.level_type,
            "wps_level": self.wps_level,
            "level_pa": self.level_pa,
            "height_m": self.height_m,
            "soil_depth_mm": self.soil_depth_mm,
            "vtable_line": self.entry.line_number,
            "vtable_raw_line": self.entry.raw_line,
            "grib_keys": dict(self.grib_keys),
        }


@dataclass(frozen=True)
class AIFSGribFile:
    """Decoded messages and metadata for one ``step_NNN.grib2`` file."""

    path: Path
    sha256: str
    valid_time: str
    grid: AIFSGrid
    vtable_path: Path
    messages: tuple[AIFSMessage, ...]

    def messages_for(self, field_name: str) -> tuple[AIFSMessage, ...]:
        return tuple(msg for msg in self.messages if msg.field_name == field_name)

    def get(
        self,
        field_name: str,
        *,
        level_pa: float | None = None,
        wps_level: float | None = None,
        soil_depth_mm: tuple[int, int] | None = None,
        height_m: float | None = None,
    ) -> AIFSMessage:
        matches = []
        for msg in self.messages_for(field_name):
            if level_pa is not None:
                if msg.level_pa is None or not np.isclose(msg.level_pa, level_pa):
                    continue
            if wps_level is not None:
                if not np.isclose(msg.wps_level, wps_level):
                    continue
            if soil_depth_mm is not None and msg.soil_depth_mm != soil_depth_mm:
                continue
            if height_m is not None:
                if msg.height_m is None or not np.isclose(msg.height_m, height_m):
                    continue
            matches.append(msg)
        if len(matches) != 1:
            filters = {
                "level_pa": level_pa,
                "wps_level": wps_level,
                "soil_depth_mm": soil_depth_mm,
                "height_m": height_m,
            }
            raise KeyError(f"expected one {field_name} message for {filters}, found {len(matches)}")
        return matches[0]

    def stack_isobaric(
        self,
        field_name: str,
        levels_pa: Iterable[float] = ISOBARIC_LEVELS_PA,
    ) -> np.ndarray:
        return np.stack([self.get(field_name, level_pa=level).data for level in levels_pa], axis=0)

    def levels_pa_for(self, field_name: str) -> tuple[float, ...]:
        levels = [msg.level_pa for msg in self.messages_for(field_name) if msg.level_pa is not None]
        return tuple(sorted(levels, reverse=True))


@dataclass(frozen=True)
class WpsIntermediateRecord:
    """One WPS intermediate-format record, array aligned as ``(south_north, west_east)``."""

    version: int
    hdate: str
    xfcst: float
    map_source: str
    field_name: str
    units: str
    description: str
    level: float
    nx: int
    ny: int
    iproj: int
    projection: dict[str, int | float | str]
    is_wind_grid_rel: bool
    data: np.ndarray

    @property
    def schema_name(self) -> str:
        return "GHT" if self.field_name == "HGT" else self.field_name


def parse_aifs_vtable(path: str | Path = DEFAULT_AIFS_VTABLE) -> tuple[VtableEntry, ...]:
    """Parse the authoritative AIFS Vtable into typed entries."""

    path = Path(path)
    entries: list[VtableEntry] = []
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        if "|" not in raw:
            continue
        parts = [part.strip() for part in raw.split("|")]
        if not parts or not _is_int(parts[0]):
            continue
        if len(parts) < 11:
            raise ValueError(f"malformed Vtable line {line_number}: {raw!r}")
        entries.append(
            VtableEntry(
                line_number=line_number,
                grib1_param=int(parts[0]),
                grib1_level_type=int(parts[1]),
                from_level=_parse_optional_int(parts[2]),
                to_level=_parse_optional_int(parts[3]),
                metgrid_name=parts[4],
                units=parts[5],
                description=parts[6],
                discipline=int(parts[7]),
                category=int(parts[8]),
                parameter=int(parts[9]),
                grib2_level_code=int(parts[10]),
                raw_line=raw.rstrip(),
            )
        )
    if not entries:
        raise ValueError(f"no Vtable data rows found in {path}")
    return tuple(entries)


def read_aifs_grib(
    path: str | Path,
    *,
    vtable_path: str | Path = DEFAULT_AIFS_VTABLE,
    require_all: bool = True,
) -> AIFSGribFile:
    """Decode one AIFS GRIB2 step using Vtable triplets and level metadata."""

    path = Path(path)
    vtable_path = Path(vtable_path)
    entries = parse_aifs_vtable(vtable_path)
    entry_list = list(entries)
    sha = sha256_file(path)
    messages: list[AIFSMessage] = []
    grid: AIFSGrid | None = None
    valid_time: str | None = None

    eccodes = _import_eccodes()
    with path.open("rb") as fh:
        while True:
            gid = eccodes.codes_grib_new_from_file(fh)
            if gid is None:
                break
            try:
                entry = _matching_entry(eccodes, gid, entry_list)
                if entry is None:
                    if require_all:
                        raise ValueError(f"unmatched GRIB message: {_message_debug(eccodes, gid)}")
                    continue
                msg_grid = _grid_from_gid(eccodes, gid)
                if grid is None:
                    grid = msg_grid
                elif not grid.compatible_with(msg_grid):
                    raise ValueError(f"inconsistent AIFS grids in {path}: {grid} vs {msg_grid}")
                if valid_time is None:
                    valid_time = _valid_time_from_gid(eccodes, gid)
                level_pa = _level_pa(eccodes, gid, entry)
                height_m = _height_m(eccodes, gid, entry)
                soil_depth = _soil_depth_mm(eccodes, gid) if entry.grib2_level_code == 106 else None
                wps_level = _wps_level(eccodes, gid, entry, level_pa)
                values = np.asarray(eccodes.codes_get_values(gid), dtype=np.float32).reshape(
                    (msg_grid.nj, msg_grid.ni)
                )
                messages.append(
                    AIFSMessage(
                        entry=entry,
                        field_name=entry.schema_name,
                        wps_field_name=entry.metgrid_name,
                        short_name=_codes_get(eccodes, gid, "shortName", default=""),
                        grib_name=_codes_get(eccodes, gid, "name", default=""),
                        units=_codes_get(eccodes, gid, "units", default=""),
                        level_type=entry.level_type,
                        wps_level=wps_level,
                        level_pa=level_pa,
                        height_m=height_m,
                        soil_depth_mm=soil_depth,
                        data=values,
                        grib_keys=_diagnostic_keys(eccodes, gid),
                    )
                )
            finally:
                eccodes.codes_release(gid)

    if grid is None or valid_time is None:
        raise ValueError(f"no decodable GRIB messages found in {path}")
    decoded = AIFSGribFile(
        path=path,
        sha256=sha,
        valid_time=valid_time,
        grid=grid,
        vtable_path=vtable_path,
        messages=tuple(messages),
    )
    if require_all:
        validate_aifs_grib(decoded, entries)
    return decoded


def validate_aifs_grib(decoded: AIFSGribFile, entries: Iterable[VtableEntry] | None = None) -> None:
    """Assert that every Vtable entry and required AIFS level is present."""

    entries = tuple(entries) if entries is not None else parse_aifs_vtable(decoded.vtable_path)
    for entry in entries:
        name = entry.schema_name
        if entry.grib2_level_code == 100:
            got = decoded.levels_pa_for(name)
            if tuple(ISOBARIC_LEVELS_PA) != got:
                raise ValueError(f"{name} levels {got} != expected {ISOBARIC_LEVELS_PA}")
        elif entry.grib2_level_code == 103:
            if entry.from_level is None:
                raise ValueError(f"heightAboveGround Vtable entry lacks From level: {entry.raw_line}")
            decoded.get(name, wps_level=WPS_SURFACE_LEVEL, height_m=float(entry.from_level))
        elif entry.grib2_level_code == 106:
            depth = entry.soil_depth_mm
            if depth is None:
                raise ValueError(f"soil Vtable entry lacks depth band: {entry.raw_line}")
            decoded.get(name, wps_level=WPS_SURFACE_LEVEL, soil_depth_mm=depth)
        elif entry.grib2_level_code == 101:
            decoded.get(name, wps_level=WPS_MEAN_SEA_LEVEL)
        else:
            decoded.get(name, wps_level=WPS_SURFACE_LEVEL)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_wps_intermediate(
    path: str | Path,
    *,
    fields: Iterable[str] | None = None,
) -> tuple[WpsIntermediateRecord, ...]:
    """Read WPS intermediate-format v5 records.

    Arrays are returned transposed from Fortran ``(nx, ny)`` storage to the Python
    source-grid convention ``(ny, nx)``, matching :func:`read_aifs_grib`.
    """

    wanted = {field.strip() for field in fields} if fields is not None else None
    records: list[WpsIntermediateRecord] = []
    with Path(path).open("rb") as fh:
        while True:
            version_record = _read_fortran_record(fh)
            if version_record is None:
                break
            version = struct.unpack(">i", version_record)[0]
            if version != 5:
                raise ValueError(f"expected WPS intermediate version 5, got {version}")
            header = _read_required_fortran_record(fh)
            off = 0
            hdate = _decode_ascii(header[off : off + 24])
            off += 24
            xfcst = struct.unpack(">f", header[off : off + 4])[0]
            off += 4
            map_source = _decode_ascii(header[off : off + 32])
            off += 32
            field_name = _decode_ascii(header[off : off + 9])
            off += 9
            units = _decode_ascii(header[off : off + 25])
            off += 25
            description = _decode_ascii(header[off : off + 46])
            off += 46
            level = struct.unpack(">f", header[off : off + 4])[0]
            off += 4
            nx = struct.unpack(">i", header[off : off + 4])[0]
            off += 4
            ny = struct.unpack(">i", header[off : off + 4])[0]
            off += 4
            iproj = struct.unpack(">i", header[off : off + 4])[0]

            projection_record = _read_required_fortran_record(fh)
            wind_record = _read_required_fortran_record(fh)
            slab_record = _read_required_fortran_record(fh)

            if wanted is not None and field_name not in wanted:
                continue
            projection = _parse_wps_projection(iproj, projection_record)
            wind_int = struct.unpack(">i", wind_record)[0]
            slab = np.frombuffer(slab_record, dtype=">f4").reshape((nx, ny), order="F")
            records.append(
                WpsIntermediateRecord(
                    version=version,
                    hdate=hdate,
                    xfcst=xfcst,
                    map_source=map_source,
                    field_name=field_name,
                    units=units,
                    description=description,
                    level=level,
                    nx=nx,
                    ny=ny,
                    iproj=iproj,
                    projection=projection,
                    is_wind_grid_rel=bool(wind_int),
                    data=slab.T.copy(),
                )
            )
    return tuple(records)


def _parse_optional_int(text: str) -> int | None:
    if text == "" or text == "*":
        return None
    return int(text)


def _is_int(text: str) -> bool:
    try:
        int(text)
    except ValueError:
        return False
    return True


def _import_eccodes():
    try:
        import eccodes  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without local eccodes
        raise RuntimeError(
            "eccodes is required to read AIFS GRIB2 files. Install python-eccodes/libeccodes."
        ) from exc
    return eccodes


def _codes_get(eccodes, gid, key: str, *, default=None):
    try:
        return eccodes.codes_get(gid, key)
    except Exception:
        return default


def _codes_get_long(eccodes, gid, key: str, *, default=None):
    try:
        return eccodes.codes_get_long(gid, key)
    except Exception:
        return default


def _matching_entry(eccodes, gid, entries: list[VtableEntry]) -> VtableEntry | None:
    discipline = int(_codes_get_long(eccodes, gid, "discipline"))
    category = int(_codes_get_long(eccodes, gid, "parameterCategory"))
    parameter = int(_codes_get_long(eccodes, gid, "parameterNumber"))
    level_code = int(_codes_get_long(eccodes, gid, "typeOfFirstFixedSurface"))
    candidates = [
        entry
        for entry in entries
        if entry.discipline == discipline
        and entry.category == category
        and entry.parameter == parameter
        and entry.grib2_level_code == level_code
    ]
    matches = [entry for entry in candidates if _entry_level_matches(eccodes, gid, entry)]
    if len(matches) > 1:
        raise ValueError(f"ambiguous Vtable match for {_message_debug(eccodes, gid)}")
    return matches[0] if matches else None


def _entry_level_matches(eccodes, gid, entry: VtableEntry) -> bool:
    if entry.grib2_level_code == 100:
        return True
    if entry.grib2_level_code == 103:
        value = _fixed_surface_value(eccodes, gid, first=True)
        return entry.from_level is not None and np.isclose(value, float(entry.from_level))
    if entry.grib2_level_code == 106:
        return _soil_depth_mm(eccodes, gid) == entry.soil_depth_mm
    return True


def _grid_from_gid(eccodes, gid) -> AIFSGrid:
    return AIFSGrid(
        grid_type=str(_codes_get(eccodes, gid, "gridType")),
        ni=int(_codes_get(eccodes, gid, "Ni")),
        nj=int(_codes_get(eccodes, gid, "Nj")),
        latitude_first=float(_codes_get(eccodes, gid, "latitudeOfFirstGridPointInDegrees")),
        longitude_first=float(_codes_get(eccodes, gid, "longitudeOfFirstGridPointInDegrees")),
        latitude_last=float(_codes_get(eccodes, gid, "latitudeOfLastGridPointInDegrees")),
        longitude_last=float(_codes_get(eccodes, gid, "longitudeOfLastGridPointInDegrees")),
        di=float(_codes_get(eccodes, gid, "iDirectionIncrementInDegrees")),
        dj=float(_codes_get(eccodes, gid, "jDirectionIncrementInDegrees")),
        j_scans_positively=bool(int(_codes_get(eccodes, gid, "jScansPositively"))),
        i_scans_negatively=bool(int(_codes_get(eccodes, gid, "iScansNegatively"))),
    )


def _fixed_surface_value(eccodes, gid, *, first: bool) -> float | None:
    which = "First" if first else "Second"
    scaled = _codes_get(eccodes, gid, f"scaledValueOf{which}FixedSurface", default=WPS_MISSING_INT)
    scale = _codes_get(eccodes, gid, f"scaleFactorOf{which}FixedSurface", default=WPS_MISSING_INT)
    if scaled in (None, WPS_MISSING_INT) or scale in (None, WPS_MISSING_INT):
        return None
    return float(scaled) * (10.0 ** (-float(scale)))


def _level_pa(eccodes, gid, entry: VtableEntry) -> float | None:
    if entry.grib2_level_code != 100:
        return None
    fixed = _fixed_surface_value(eccodes, gid, first=True)
    if fixed is not None:
        return float(fixed)
    return float(_codes_get(eccodes, gid, "level")) * 100.0


def _height_m(eccodes, gid, entry: VtableEntry) -> float | None:
    if entry.grib2_level_code != 103:
        return None
    fixed = _fixed_surface_value(eccodes, gid, first=True)
    return float(fixed) if fixed is not None else float(_codes_get(eccodes, gid, "level"))


def _soil_depth_mm(eccodes, gid) -> tuple[int, int]:
    first_cm = _fixed_surface_value(eccodes, gid, first=True)
    second_cm = _fixed_surface_value(eccodes, gid, first=False)
    if first_cm is None or second_cm is None:
        raise ValueError(f"missing GRIB soil depth surfaces: {_message_debug(eccodes, gid)}")
    return (int(round(first_cm * 100.0)), int(round(second_cm * 100.0)))


def _wps_level(eccodes, gid, entry: VtableEntry, level_pa: float | None) -> float:
    if entry.grib2_level_code == 100:
        if level_pa is None:
            raise ValueError(f"missing isobaric level: {_message_debug(eccodes, gid)}")
        return float(level_pa)
    if entry.grib2_level_code == 101:
        return WPS_MEAN_SEA_LEVEL
    return WPS_SURFACE_LEVEL


def _valid_time_from_gid(eccodes, gid) -> str:
    date = int(_codes_get(eccodes, gid, "validityDate", default=_codes_get(eccodes, gid, "dataDate")))
    time = int(_codes_get(eccodes, gid, "validityTime", default=_codes_get(eccodes, gid, "dataTime")))
    year = date // 10000
    month = (date // 100) % 100
    day = date % 100
    hour = time // 100
    minute = time % 100
    return f"{year:04d}-{month:02d}-{day:02d}_{hour:02d}:{minute:02d}:00"


def _diagnostic_keys(eccodes, gid) -> dict[str, int | float | str | None]:
    keys = (
        "discipline",
        "parameterCategory",
        "parameterNumber",
        "typeOfFirstFixedSurface",
        "typeOfSecondFixedSurface",
        "level",
        "scaledValueOfFirstFixedSurface",
        "scaleFactorOfFirstFixedSurface",
        "scaledValueOfSecondFixedSurface",
        "scaleFactorOfSecondFixedSurface",
        "typeOfLevel",
        "shortName",
    )
    return {key: _codes_get(eccodes, gid, key) for key in keys}


def _message_debug(eccodes, gid) -> str:
    keys = _diagnostic_keys(eccodes, gid)
    return ", ".join(f"{key}={value!r}" for key, value in keys.items())


def _read_fortran_record(fh) -> bytes | None:
    raw_len = fh.read(4)
    if raw_len == b"":
        return None
    if len(raw_len) != 4:
        raise ValueError("truncated Fortran record marker")
    nbytes = struct.unpack(">i", raw_len)[0]
    payload = fh.read(nbytes)
    if len(payload) != nbytes:
        raise ValueError("truncated Fortran record payload")
    raw_end = fh.read(4)
    if len(raw_end) != 4:
        raise ValueError("truncated Fortran record trailer")
    end_nbytes = struct.unpack(">i", raw_end)[0]
    if end_nbytes != nbytes:
        raise ValueError(f"Fortran record marker mismatch: {nbytes} != {end_nbytes}")
    return payload


def _read_required_fortran_record(fh) -> bytes:
    record = _read_fortran_record(fh)
    if record is None:
        raise ValueError("unexpected EOF in WPS intermediate file")
    return record


def _decode_ascii(raw: bytes) -> str:
    return raw.decode("ascii").strip()


def _parse_wps_projection(iproj: int, record: bytes) -> dict[str, int | float | str]:
    off = 0
    startloc = _decode_ascii(record[off : off + 8])
    off += 8

    def take_float() -> float:
        nonlocal off
        value = struct.unpack(">f", record[off : off + 4])[0]
        off += 4
        return float(value)

    if iproj in (0, 4):
        return {
            "iproj": iproj,
            "startloc": startloc,
            "startlat": take_float(),
            "startlon": take_float(),
            "deltalat": take_float(),
            "deltalon": take_float(),
            "earth_radius_km": take_float(),
        }
    if iproj == 1:
        return {
            "iproj": iproj,
            "startloc": startloc,
            "startlat": take_float(),
            "startlon": take_float(),
            "dx_km": take_float(),
            "dy_km": take_float(),
            "truelat1": take_float(),
            "earth_radius_km": take_float(),
        }
    if iproj == 3:
        return {
            "iproj": iproj,
            "startloc": startloc,
            "startlat": take_float(),
            "startlon": take_float(),
            "dx_km": take_float(),
            "dy_km": take_float(),
            "xlonc": take_float(),
            "truelat1": take_float(),
            "truelat2": take_float(),
            "earth_radius_km": take_float(),
        }
    if iproj == 5:
        return {
            "iproj": iproj,
            "startloc": startloc,
            "startlat": take_float(),
            "startlon": take_float(),
            "dx_km": take_float(),
            "dy_km": take_float(),
            "xlonc": take_float(),
            "truelat1": take_float(),
            "earth_radius_km": take_float(),
        }
    if iproj == 6:
        return {
            "iproj": iproj,
            "startloc": startloc,
            "startlat": take_float(),
            "startlon": take_float(),
            "dx_km": take_float(),
            "dy_km": take_float(),
            "centerlat": take_float(),
            "centerlon": take_float(),
            "earth_radius_km": take_float(),
        }
    raise ValueError(f"unsupported WPS projection code {iproj}")
