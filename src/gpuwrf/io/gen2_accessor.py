"""Read-only accessors for the pinned Canairy Gen2 WRF backfill run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
from gpuwrf.io.netcdf_lock import Dataset

try:  # JAX is present in project test environments but keep import-time behavior narrow.
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover - exercised only in non-JAX utility contexts.
    jax = None
    jnp = None

from gpuwrf.config import paths
from gpuwrf.contracts.grid import BCMetadata, GridSpec, Projection, TerrainProvenance, VerticalCoord


# Path indirection (see gpuwrf.config.paths): the read-only Gen2 corpus root and
# its default L3 run dir are environment-overridable so a clean clone is runnable
# without <DATA_ROOT>. On the user's workstation set GPUWRF_CANAIRY_ROOT=<DATA_ROOT>/canairy_meteo
# to keep the original layout (and write-protection) working unchanged.
GEN2_READ_ONLY_ROOT = paths.canairy_root()
DEFAULT_M6_GEN2_RUN_DIR = (
    paths.wrf_l3_root() / "20260521_18z_l3_24h_20260522T133443Z"
)
DEFAULT_M6_BOUNDARY_REPLAY = Path("data/fixtures/m6/d02_boundary_replay_v2.zarr")
MAP_PROJ_NAMES = {1: "lambert", 2: "polar", 3: "mercator"}
DOMAIN_RE = re.compile(r"^d(?P<num>\d{2})$")
WRFOUT_TIME_RE = re.compile(r"wrfout_d\d{2}_(?P<stamp>\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})$")


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _reject_gen2_write_target(path: Path) -> None:
    if _is_under(path, GEN2_READ_ONLY_ROOT):
        raise PermissionError(f"refusing to write inside read-only Gen2 data domain: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _namelist_tokens(text: str) -> list[str]:
    """Tokenize delimiters outside strings; Fortran doubles embedded quotes."""

    tokens: list[str] = []
    pos = 0
    while pos < len(text):
        char = text[pos]
        if char.isspace():
            pos += 1
        elif char == "!":
            end = text.find("\n", pos)
            pos = len(text) if end < 0 else end + 1
        elif char in {"'", '"'}:
            start, quote = pos, char
            pos += 1
            while pos < len(text):
                if text[pos] == quote:
                    if pos + 1 < len(text) and text[pos + 1] == quote:
                        pos += 2
                        continue
                    pos += 1
                    tokens.append(text[start:pos])
                    break
                pos += 1
            else:
                raise ValueError("unterminated namelist character value")
        elif char in "&/=,*():":
            tokens.append(char)
            pos += 1
        else:
            start = pos
            while pos < len(text) and not text[pos].isspace() and text[pos] not in "&/=,*():!'\"":
                pos += 1
            tokens.append(text[start:pos])
    return tokens


def _parse_scalar(text: str) -> Any:
    raw = text.strip()
    if len(raw) >= 2 and raw[0] in {"'", '"'} and raw[-1] == raw[0]:
        return raw[1:-1].replace(raw[0] * 2, raw[0])
    lowered = raw.lower()
    if lowered in {".true.", "true", ".t.", "t"}:
        return True
    if lowered in {".false.", "false", ".f.", "f"}:
        return False
    normalized = raw.replace("D", "E").replace("d", "e")
    try:
        if not any(char in normalized for char in ".eE"):
            return int(normalized)
        return float(normalized)
    except ValueError as exc:
        raise ValueError(f"invalid namelist value {raw!r}; character values must be quoted") from exc


def parse_namelist(path: Path) -> dict[str, dict[str, Any]]:
    """Read WRF scalar/one-dimensional namelists with list-directed semantics.

    Null slots are ``None``: consumers apply their Registry/default value.
    Reassigning a variable leaves previously assigned null slots untouched.
    Repeat counts apply to every scalar type, including nulls and characters.
    Unsupported syntax or malformed values raise instead of becoming strings.
    """

    tokens = _namelist_tokens(path.read_text(encoding="utf-8"))
    groups: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    pos = 0
    identifier = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
    def assignment(at: int) -> bool:
        return at + 1 < len(tokens) and bool(identifier.fullmatch(tokens[at])) and tokens[at + 1] == "="

    while pos < len(tokens):
        token = tokens[pos]
        if token == "&":
            pos += 1
            if pos >= len(tokens) or not identifier.fullmatch(tokens[pos]):
                raise ValueError(f"{path}: invalid namelist group")
            name = tokens[pos].lower()
            if name == "end":
                current = None
            else:
                if current is not None:
                    raise ValueError(f"{path}: previous namelist group was not closed")
                current = groups.setdefault(name, {})
            pos += 1
            continue
        if token == "/":
            if current is None:
                raise ValueError(f"{path}: unexpected group terminator")
            current = None
            pos += 1
            continue
        if current is None or not assignment(pos):
            raise ValueError(f"{path}: expected namelist assignment, got {token!r}")
        key = token.lower()
        pos += 2
        values: list[Any] = []
        expect_value = True
        while pos < len(tokens) and tokens[pos] not in {"/", "&"} and not assignment(pos):
            token = tokens[pos]
            if token == ",":
                if expect_value:
                    values.append(None)
                expect_value = True
                pos += 1
                continue
            count = 1
            if pos + 1 < len(tokens) and tokens[pos + 1] == "*":
                if not token.isdecimal() or int(token) < 1:
                    raise ValueError(f"{path}: invalid repeat count {token!r}")
                count = int(token)
                pos += 2
                if pos >= len(tokens) or tokens[pos] in {",", "/", "&"} or assignment(pos):
                    values.extend([None] * count)
                    expect_value = False
                    continue
                token = tokens[pos]
            values.extend([_parse_scalar(token)] * count)
            pos += 1
            expect_value = False
        prior = current.get(key, [])
        merged = list(prior) if isinstance(prior, list) else [prior]
        if len(merged) < len(values):
            merged.extend([None] * (len(values) - len(merged)))
        for index, value in enumerate(values):
            if value is not None:
                merged[index] = value
        if merged:
            current[key] = merged[0] if len(merged) == 1 else merged
    if current is not None:
        raise ValueError(f"{path}: unterminated namelist group")
    return groups


def _domain_number(domain: str) -> int:
    match = DOMAIN_RE.match(domain)
    if match is None:
        raise ValueError(f"domain must look like d01, d02, ...; got {domain!r}")
    return int(match.group("num"))


def _domain_id(number: int) -> str:
    return f"d{int(number):02d}"


def _netcdf_attrs(dataset: Dataset, names: Iterable[str]) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for name in names:
        if hasattr(dataset, name):
            value = getattr(dataset, name)
            if hasattr(value, "item"):
                value = value.item()
            attrs[name] = value
    return attrs


@dataclass(frozen=True)
class Gen2GridSpec:
    """GridSpec-compatible metadata for one Gen2 WRF domain."""

    id: str
    dx_m: float
    dy_m: float
    e_we: int
    e_sn: int
    e_vert: int
    mass_nx: int
    mass_ny: int
    mass_nz: int
    grid_proj: str
    map_proj_id: int
    cen_lat: float
    cen_lon: float
    truelat1: float
    truelat2: float
    stand_lon: float
    parent_id: int
    parent_grid_ratio: int
    i_parent_start: int
    j_parent_start: int
    znu: tuple[float, ...]
    znw: tuple[float, ...]
    top_pressure_pa: float
    source_wrfout: str
    source_namelist: str

    @property
    def nx(self) -> int:
        return self.mass_nx

    @property
    def ny(self) -> int:
        return self.mass_ny

    @property
    def nz(self) -> int:
        return self.mass_nz

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["znu"] = list(self.znu)
        data["znw"] = list(self.znw)
        return data

    def static_field(self, name: str) -> np.ndarray:
        """Load a static WRF field from the domain's first history file."""

        with Dataset(self.source_wrfout, "r") as dataset:
            if name not in dataset.variables:
                raise KeyError(f"{name!r} is not present in {self.source_wrfout}")
            variable = dataset.variables[name]
            data = variable[0] if "Time" in variable.dimensions else variable[:]
            return np.asarray(np.ma.filled(data, np.nan))

    def as_grid_spec(self) -> GridSpec:
        """Convert to the project GridSpec contract for downstream model code."""

        if jnp is None:  # pragma: no cover - only if JAX is not importable.
            raise RuntimeError("JAX is required to convert Gen2GridSpec to GridSpec")
        projection = Projection(
            self.grid_proj, self.cen_lat, self.cen_lon, self.dx_m, self.dy_m, self.mass_nx, self.mass_ny
        )
        terrain_height_np = self.static_field("HGT").astype(np.float64)
        terrain = TerrainProvenance(
            source_path=self.source_wrfout,
            sha256="gen2-manifest-sha256",
            shape=(self.mass_ny, self.mass_nx),
            units="m",
            projection_transform="native-wrf-lambert",
            max_elevation_m=float(np.nanmax(terrain_height_np)),
            coastline_sanity_check_passed=True,
        )
        eta_levels = jnp.asarray(self.znw, dtype=jnp.float64)
        vertical = VerticalCoord("hybrid_eta", self.mass_nz, self.top_pressure_pa, eta_levels)
        bc = BCMetadata(
            source="AIFS",
            fields=("U", "V", "T", "QVAPOR", "PH"),
            update_cadence_h=1,
            interpolation="linear",
            restart_compatible=True,
        )
        terrain_height = jnp.asarray(terrain_height_np, dtype=jnp.float64)
        return GridSpec(projection, terrain, vertical, bc, eta_levels, terrain_height)


class LazyNetCDFArray:
    """Lazy WRF variable handle that becomes device-resident once materialized."""

    def __init__(self, run: "Gen2Run", path: Path, variable: str, time_index: int | None) -> None:
        self._run = run
        self.path = path
        self.variable = variable
        self.time_index = time_index

    def materialize(self):
        """Return a cached JAX device array for this variable slice."""

        return self._run._load_device_array(self.path, self.variable, self.time_index)

    def __jax_array__(self):
        return self.materialize()

    def __array__(self, dtype=None):
        array = np.asarray(self.materialize())
        if dtype is not None:
            array = array.astype(dtype)
        return array

    @property
    def shape(self) -> tuple[int, ...]:
        with Dataset(self.path, "r") as dataset:
            variable = dataset.variables[self.variable]
            shape = tuple(int(size) for size in variable.shape)
            if self.time_index is not None and variable.dimensions and variable.dimensions[0] == "Time":
                return shape[1:]
            return shape

    @property
    def dtype(self) -> np.dtype:
        with Dataset(self.path, "r") as dataset:
            return np.dtype(dataset.variables[self.variable].dtype)

    def __repr__(self) -> str:
        return (
            f"LazyNetCDFArray(path={self.path.name!r}, variable={self.variable!r}, "
            f"time_index={self.time_index!r}, shape={self.shape!r})"
        )


class Gen2Run:
    """Read-only adapter for a Gen2 WRF run directory."""

    def __init__(self, run_dir: str | Path) -> None:
        self.path = Path(run_dir).expanduser().resolve()
        if not self.path.is_dir():
            raise FileNotFoundError(f"Gen2 run directory not found: {self.path}")
        self.run_id = self.path.name
        self._namelist: dict[str, dict[str, Any]] | None = None
        self._domains: list[str] | None = None
        self._grid_cache: dict[str, Gen2GridSpec] = {}
        self._variable_cache: dict[str, list[str]] = {}
        self._device_cache: dict[tuple[str, str, int | None], Any] = {}
        self._input_handles: dict[Path, Dataset] = {}

    @contextmanager
    def input_read_scope(self, domain: str):
        """Keep one read-only input handle during init; always close on exit.

        Device-array caching and lazy materialization retain their usual behavior.
        Nested scopes reuse the outer handle, which remains the outer owner's.
        """
        path = self.wrfinput_file(domain)
        if path in self._input_handles:
            yield self
            return
        with Dataset(path, "r") as dataset:
            self._input_handles[path] = dataset
            try:
                yield self
            finally:
                del self._input_handles[path]

    @contextmanager
    def _read_dataset(self, path: Path):
        dataset = self._input_handles.get(path)
        if dataset is not None:
            yield dataset
        else:
            with Dataset(path, "r") as opened:
                yield opened

    @property
    def namelist(self) -> dict[str, dict[str, Any]]:
        if self._namelist is None:
            self._namelist = parse_namelist(self.path / "namelist.input")
        return self._namelist

    @property
    def domains(self) -> list[str]:
        if self._domains is None:
            max_dom = self.namelist.get("domains", {}).get("max_dom")
            if isinstance(max_dom, int):
                numbers = range(1, max_dom + 1)
            else:
                numbers = sorted(
                    {
                        int(match.group("num"))
                        for file_path in self.path.glob("wrfout_d0*_*")
                        if (match := re.search(r"wrfout_d(?P<num>\d{2})_", file_path.name))
                    }
                )
            self._domains = [_domain_id(number) for number in numbers]
        return list(self._domains)

    def history_files(self, domain: str) -> list[Path]:
        _domain_number(domain)
        files = sorted(self.path.glob(f"wrfout_{domain}_*"))
        if not files:
            wrfinput = self.wrfinput_file(domain)
            if wrfinput.exists():
                return [wrfinput]
            raise FileNotFoundError(f"no wrfout files for {domain} in {self.path}")
        return files

    def wrfinput_file(self, domain: str) -> Path:
        _domain_number(domain)
        path = self.path / f"wrfinput_{domain}"
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    def wrfinput_variables(self, domain: str) -> list[str]:
        """Return variables present in the domain's `wrfinput` file."""

        with self._read_dataset(self.wrfinput_file(domain)) as dataset:
            return list(dataset.variables.keys())

    def time_axis(self, domain: str) -> list[datetime]:
        times: list[datetime] = []
        for path in self.history_files(domain):
            match = WRFOUT_TIME_RE.match(path.name)
            if match is None:
                continue
            times.append(datetime.strptime(match.group("stamp"), "%Y-%m-%d_%H:%M:%S").replace(tzinfo=timezone.utc))
        return times

    def grid(self, domain: str) -> Gen2GridSpec:
        if domain in self._grid_cache:
            return self._grid_cache[domain]
        index = _domain_number(domain) - 1
        first = self.history_files(domain)[0]
        domains_nml = self.namelist.get("domains", {})
        with Dataset(first, "r") as dataset:
            dims = dataset.dimensions
            attrs = _netcdf_attrs(
                dataset,
                ("DX", "DY", "MAP_PROJ", "CEN_LAT", "CEN_LON", "TRUELAT1", "TRUELAT2", "STAND_LON"),
            )
            map_proj_id = int(attrs.get("MAP_PROJ", 1))
            znu = tuple(float(value) for value in np.asarray(dataset.variables["ZNU"][0]))
            znw = tuple(float(value) for value in np.asarray(dataset.variables["ZNW"][0]))
            mass_nx = int(len(dims["west_east"]))
            mass_ny = int(len(dims["south_north"]))
            mass_nz = int(len(dims["bottom_top"]))
        grid = Gen2GridSpec(
            id=domain,
            dx_m=float(self._nml_list_value(domains_nml, "dx", index, attrs.get("DX"))),
            dy_m=float(self._nml_list_value(domains_nml, "dy", index, attrs.get("DY"))),
            e_we=int(self._nml_list_value(domains_nml, "e_we", index, mass_nx + 1)),
            e_sn=int(self._nml_list_value(domains_nml, "e_sn", index, mass_ny + 1)),
            e_vert=int(self._nml_list_value(domains_nml, "e_vert", index, mass_nz + 1)),
            mass_nx=mass_nx,
            mass_ny=mass_ny,
            mass_nz=mass_nz,
            grid_proj=MAP_PROJ_NAMES.get(map_proj_id, f"wrf_map_proj_{map_proj_id}"),
            map_proj_id=map_proj_id,
            cen_lat=float(attrs.get("CEN_LAT")),
            cen_lon=float(attrs.get("CEN_LON")),
            truelat1=float(attrs.get("TRUELAT1")),
            truelat2=float(attrs.get("TRUELAT2")),
            stand_lon=float(attrs.get("STAND_LON")),
            parent_id=int(self._nml_list_value(domains_nml, "parent_id", index, 1)),
            parent_grid_ratio=int(self._nml_list_value(domains_nml, "parent_grid_ratio", index, 1)),
            i_parent_start=int(self._nml_list_value(domains_nml, "i_parent_start", index, 1)),
            j_parent_start=int(self._nml_list_value(domains_nml, "j_parent_start", index, 1)),
            znu=znu,
            znw=znw,
            top_pressure_pa=float(self._nml_list_value(domains_nml, "p_top_requested", 0, 5000.0)),
            source_wrfout=str(first),
            source_namelist=str(self.path / "namelist.input"),
        )
        self._grid_cache[domain] = grid
        return grid

    def variables(self, domain: str) -> list[str]:
        if domain not in self._variable_cache:
            first = self.history_files(domain)[0]
            with Dataset(first, "r") as dataset:
                self._variable_cache[domain] = list(dataset.variables.keys())
        return list(self._variable_cache[domain])

    def load(self, domain: str, var: str, time: int | str | datetime | None = None, lazy: bool = True):
        path = self._file_for_time(domain, time)
        with self._read_dataset(path) as dataset:
            if var not in dataset.variables:
                raise KeyError(f"{var!r} not present in {path}")
            time_index = 0 if "Time" in dataset.variables[var].dimensions else None
        handle = LazyNetCDFArray(self, path, var, time_index)
        return handle if lazy else handle.materialize()

    def load_wrfinput(self, domain: str, var: str, lazy: bool = True):
        """Load one variable from `wrfinput_<domain>` through the shared cache."""

        path = self.wrfinput_file(domain)
        with self._read_dataset(path) as dataset:
            if var not in dataset.variables:
                raise KeyError(f"{var!r} not present in {path}")
            time_index = 0 if "Time" in dataset.variables[var].dimensions else None
        handle = LazyNetCDFArray(self, path, var, time_index)
        return handle if lazy else handle.materialize()

    def build_manifest(self, *, include_sha256: bool = True) -> dict[str, Any]:
        domains = [self.grid(domain).to_dict() for domain in self.domains]
        file_patterns = ("wrfout_d0*_*", "wrfinput_d0*", "wrfbdy_*", "namelist.input", "namelist.output")
        seen: set[Path] = set()
        files = []
        for pattern in file_patterns:
            for path in sorted(self.path.glob(pattern)):
                if path in seen or not path.is_file():
                    continue
                seen.add(path)
                stat = path.stat()
                entry = {
                    "path": path.name,
                    "size_bytes": int(stat.st_size),
                    "mtime": float(stat.st_mtime),
                    "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                }
                if include_sha256:
                    entry["sha256"] = _sha256(path)
                files.append(entry)
        return {
            "run_id": self.run_id,
            "path": str(self.path),
            "no_write_audit": True,
            "domains": domains,
            "files": files,
            "variable_inventory": {domain: self.variables(domain) for domain in self.domains},
            "source_citations": [
                ".agent/references/cpu-wrf-baseline.md",
                str(self.path / "namelist.input"),
                "/home/user/src/canairy_meteo/Gen2/wrf-gpu.md",
            ],
        }

    def write_manifest(self, output_path: str | Path, *, include_sha256: bool = True) -> dict[str, Any]:
        target = Path(output_path)
        _reject_gen2_write_target(target)
        manifest = self.build_manifest(include_sha256=include_sha256)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest

    @staticmethod
    def _nml_list_value(group: dict[str, Any], key: str, index: int, default: Any) -> Any:
        value = group.get(key, default)
        if isinstance(value, list):
            if index >= len(value):
                return default
            value = value[index]
        return default if value is None else value

    def _file_for_time(self, domain: str, time: int | str | datetime | None) -> Path:
        files = self.history_files(domain)
        if time is None:
            return files[0]
        if isinstance(time, int):
            return files[time]
        if isinstance(time, datetime):
            stamp = time.strftime("%Y-%m-%d_%H:%M:%S")
        else:
            stamp = str(time).replace("T", "_")
        for path in files:
            if stamp in path.name:
                return path
        raise FileNotFoundError(f"no {domain} wrfout history file for time {time!r}")

    def _read_variable(self, path: Path, variable: str, time_index: int | None) -> np.ndarray:
        with self._read_dataset(path) as dataset:
            netcdf_var = dataset.variables[variable]
            data = netcdf_var[time_index] if time_index is not None else netcdf_var[:]
            return np.asarray(np.ma.filled(data, np.nan))

    def _load_device_array(self, path: Path, variable: str, time_index: int | None):
        key = (str(path), variable, time_index)
        if key not in self._device_cache:
            host = self._read_variable(path, variable, time_index)
            self._device_cache[key] = jax.device_put(host) if jax is not None else host
        return self._device_cache[key]


__all__ = [
    "DEFAULT_M6_BOUNDARY_REPLAY",
    "DEFAULT_M6_GEN2_RUN_DIR",
    "GEN2_READ_ONLY_ROOT",
    "Gen2GridSpec",
    "Gen2Run",
    "LazyNetCDFArray",
    "parse_namelist",
]
