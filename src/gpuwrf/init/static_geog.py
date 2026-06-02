"""Static ``geo_em`` loader for the v0.3.0 metgrid-equivalent artifact."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from netCDF4 import Dataset
import numpy as np

from gpuwrf.init.metgrid_schema import (
    METGRID_SCHEMA_VERSION,
    MetgridFieldSpec,
    MetgridProjection,
    metem_field_specs,
)
from gpuwrf.init.projection import LambertGrid


STATIC_GEOG_GROUPS = frozenset(("coord", "mapfac", "geog2d", "geog3d"))
DIM_ALIASES = {
    "land_cat": "z-dimension0021",
    "soil_cat": "z-dimension0016",
    "month": "z-dimension0012",
}


@dataclass(frozen=True)
class StaticGeogData:
    """Partial metgrid-equivalent static-geography payload for one domain."""

    domain: str
    projection: MetgridProjection
    lambert_grid: LambertGrid
    arrays: dict[str, np.ndarray]
    provenance: dict[str, str] = field(default_factory=dict)
    schema_version: str = METGRID_SCHEMA_VERSION

    def validate(self) -> None:
        specs = static_geog_specs()
        spec_index = {spec.name: spec for spec in specs}
        for spec in specs:
            if spec.name not in self.arrays:
                raise ValueError(f"missing static geo_em field {spec.name!r}")
        for name, arr in self.arrays.items():
            if name not in spec_index:
                raise ValueError(f"unexpected static geo_em field {name!r}")
            expected = expected_shape(spec_index[name], self.projection)
            if tuple(arr.shape) != expected:
                raise ValueError(f"{name} shape {arr.shape} != {expected}")
        if self.domain not in ("d01", "d02", "d03", "d04", "d05"):
            raise ValueError(f"unexpected WRF domain {self.domain!r}")


def static_geog_specs() -> tuple[MetgridFieldSpec, ...]:
    """Return the frozen schema fields owned by S2 static geog."""

    return tuple(
        spec
        for spec in metem_field_specs()
        if spec.source == "geo_em" and spec.group in STATIC_GEOG_GROUPS
    )


def static_geog_field_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in static_geog_specs())


def expected_shape(spec: MetgridFieldSpec, projection: MetgridProjection) -> tuple[int, ...]:
    sizes = {
        "south_north": projection.ny,
        "west_east": projection.nx,
        "south_north_stag": projection.ny + 1,
        "west_east_stag": projection.nx + 1,
        "z-dimension0021": projection.num_land_cat,
        "z-dimension0016": projection.num_soil_cat,
        "z-dimension0012": 12,
    }
    try:
        return tuple(sizes[dim] for dim in spec.dims)
    except KeyError as exc:
        raise ValueError(f"static geog field {spec.name!r} has unsupported dim {exc.args[0]!r}") from exc


def metgrid_projection_from_dataset(dataset: Any) -> MetgridProjection:
    """Extract frozen-schema projection metadata from a WPS NetCDF dataset."""

    return MetgridProjection(
        map_proj=_int_attr(dataset, "MAP_PROJ"),
        truelat1=_float_attr(dataset, "TRUELAT1"),
        truelat2=_float_attr(dataset, "TRUELAT2"),
        stand_lon=_float_attr(dataset, "STAND_LON"),
        moad_cen_lat=_float_attr(dataset, "MOAD_CEN_LAT"),
        pole_lat=_float_attr(dataset, "POLE_LAT", 90.0),
        pole_lon=_float_attr(dataset, "POLE_LON", 0.0),
        dx_m=_float_attr(dataset, "DX"),
        dy_m=_float_attr(dataset, "DY"),
        nx=int(len(dataset.dimensions["west_east"])),
        ny=int(len(dataset.dimensions["south_north"])),
        grid_id=_int_attr(dataset, "grid_id", 1),
        parent_id=_int_attr(dataset, "parent_id", 1),
        parent_grid_ratio=_int_attr(dataset, "parent_grid_ratio", 1),
        i_parent_start=_int_attr(dataset, "i_parent_start", 1),
        j_parent_start=_int_attr(dataset, "j_parent_start", 1),
        mminlu=str(_attr(dataset, "MMINLU", "MODIFIED_IGBP_MODIS_NOAH")),
        num_land_cat=_int_attr(dataset, "NUM_LAND_CAT", 21),
        num_soil_cat=_int_attr(dataset, "NUM_SOIL_CAT", 16),
        iswater=_int_attr(dataset, "ISWATER", 17),
        islake=_int_attr(dataset, "ISLAKE", 21),
        isice=_int_attr(dataset, "ISICE", 15),
        isurban=_int_attr(dataset, "ISURBAN", 13),
        isoilwater=_int_attr(dataset, "ISOILWATER", 14),
    )


def load_static_geog(path: str | Path) -> StaticGeogData:
    """Load all S2-owned static fields from one ``geo_em.d0N.nc`` file."""

    geo_path = Path(path)
    with Dataset(str(geo_path)) as dataset:
        projection = metgrid_projection_from_dataset(dataset)
        lambert_grid = LambertGrid.from_wps_dataset(dataset)
        arrays = {
            spec.name: _read_schema_field(dataset, spec, projection)
            for spec in static_geog_specs()
        }
        data = StaticGeogData(
            domain=_domain_from_path(geo_path),
            projection=projection,
            lambert_grid=lambert_grid,
            arrays=arrays,
            provenance={
                "source": str(geo_path),
                "title": str(_attr(dataset, "TITLE", "")),
                "schema_version": METGRID_SCHEMA_VERSION,
            },
        )
    data.validate()
    return data


def _read_schema_field(
    dataset: Dataset,
    spec: MetgridFieldSpec,
    projection: MetgridProjection,
) -> np.ndarray:
    if spec.name not in dataset.variables:
        raise KeyError(f"{spec.name!r} not found in {getattr(dataset, 'filepath', lambda: '<dataset>')()}")
    variable = dataset.variables[spec.name]
    dims = tuple(variable.dimensions)
    if not dims or dims[0] != "Time":
        raise ValueError(f"{spec.name} expected leading Time dim, got {dims}")
    source_dims = tuple(DIM_ALIASES.get(dim, dim) for dim in dims[1:])
    if source_dims != spec.dims:
        raise ValueError(f"{spec.name} dims {source_dims} != schema dims {spec.dims}")
    array = np.asarray(variable[0], dtype=np.float32)
    expected = expected_shape(spec, projection)
    if tuple(array.shape) != expected:
        raise ValueError(f"{spec.name} shape {array.shape} != {expected}")
    return array


def _domain_from_path(path: Path) -> str:
    name = path.name
    for domain in ("d01", "d02", "d03", "d04", "d05"):
        if f".{domain}." in name or name.endswith(f".{domain}.nc"):
            return domain
    raise ValueError(f"cannot infer WRF domain from {path}")


def _attr(dataset: Any, name: str, default: Any = None) -> Any:
    for key in (name, name.upper(), name.lower()):
        if hasattr(dataset, key):
            return getattr(dataset, key)
    return default


def _float_attr(dataset: Any, name: str, default: float | None = None) -> float:
    value = _attr(dataset, name, default)
    if value is None:
        raise ValueError(f"missing required attr {name!r}")
    return float(value)


def _int_attr(dataset: Any, name: str, default: int | None = None) -> int:
    value = _attr(dataset, name, default)
    if value is None:
        raise ValueError(f"missing required attr {name!r}")
    return int(value)
