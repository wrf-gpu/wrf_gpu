"""Serialize a :class:`MetEmArtifact` to met_em-format NetCDF (v0.3.0 S3).

Writes a file structurally identical to a real WPS ``met_em.d0N.*.nc`` (metgrid
V4.6.0): the ``Time``/``DateStrLen`` axes, the staggered dimension set, the
``Times`` char array, per-variable attrs (``FieldType=104``, ``MemoryOrder``,
``stagger``, ``units``, ``description``, ``sr_x/sr_y``), and the full global-attr
+ ``FLAG_*`` block that ``real.exe`` reads. fp32 storage like met_em.

The exact attribute set is taken from the live oracle (recon §2 + the d01 met_em
inspection): all data vars carry ``FieldType=104``, ``MemoryOrder='XYZ'`` (3D) or
``'XY '`` (2D, trailing space), ``stagger`` per the schema, ``sr_x=sr_y=1``.
``units``/``description`` for the derived PRES/ST/SM/SOIL_LAYERS are empty in
met_em; we mirror that (configurable via ``MET_EM_VAR_ATTRS``).
"""

from __future__ import annotations

import numpy as np

try:
    import netCDF4
except Exception as exc:  # pragma: no cover
    netCDF4 = None
    _NETCDF_IMPORT_ERROR = exc

from gpuwrf.init.metgrid_schema import (
    ISOBARIC_LEVELS_PA,
    NUM_METGRID_LEVELS,
    MetEmArtifact,
    metem_field_specs,
)

# met_em FieldType for all real data vars.
FIELD_TYPE = 104

# Per-variable (units, description) overrides matching the live met_em oracle,
# where metgrid emits empty strings for the derived fields. For all other fields
# the schema units/description are used.
MET_EM_VAR_ATTRS: dict[str, tuple[str, str]] = {
    "PRES": ("", ""),
    "ST": ("", ""),
    "SM": ("", ""),
    "SOIL_LAYERS": ("", ""),
}


def _memory_order(ndim: int) -> str:
    # met_em: 3D -> 'XYZ' ; 2D -> 'XY ' (trailing space, len 3)
    return "XYZ" if ndim == 3 else "XY "


def _dim_names_with_time(dims: tuple[str, ...]) -> tuple[str, ...]:
    return ("Time",) + tuple(dims)


def write_met_em(
    artifact: MetEmArtifact,
    path: str,
    *,
    require_optional: bool = False,
) -> str:
    """Write ``artifact`` to ``path`` as a met_em-format NetCDF. Validates the
    artifact against the frozen schema first. Returns ``path``."""

    if netCDF4 is None:  # pragma: no cover
        raise RuntimeError(f"netCDF4 unavailable: {_NETCDF_IMPORT_ERROR}")

    artifact.validate(require_optional=require_optional)
    specs = {s.name: s for s in metem_field_specs()}
    proj = artifact.projection
    nx, ny = proj.nx, proj.ny

    ds = netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC")
    try:
        # --- dimensions (order mirrors met_em) ---
        ds.createDimension("Time", None)  # UNLIMITED
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("west_east", nx)
        ds.createDimension("south_north", ny)
        ds.createDimension("num_metgrid_levels", NUM_METGRID_LEVELS)
        ds.createDimension("num_st_layers", artifact._dim_size("num_st_layers"))
        ds.createDimension("num_sm_layers", artifact._dim_size("num_sm_layers"))
        ds.createDimension("south_north_stag", ny + 1)
        ds.createDimension("west_east_stag", nx + 1)
        ds.createDimension("z-dimension0012", 12)
        ds.createDimension("z-dimension0016", proj.num_soil_cat)
        ds.createDimension("z-dimension0021", proj.num_land_cat)

        # --- Times char variable ---
        tvar = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        tstr = artifact.valid_time.ljust(19)[:19]
        tvar[0, :] = np.array(list(tstr), dtype="S1")

        # --- data variables (write all present, schema order) ---
        for name, spec in specs.items():
            if name not in artifact.arrays:
                continue
            arr = np.asarray(artifact.arrays[name], dtype=np.float32)
            dim_names = _dim_names_with_time(spec.dims)
            v = ds.createVariable(name, "f4", dim_names)
            v[0, ...] = arr
            units, desc = MET_EM_VAR_ATTRS.get(name, (spec.units, spec.description))
            v.setncattr("FieldType", np.int32(FIELD_TYPE))
            v.setncattr("MemoryOrder", _memory_order(arr.ndim))
            v.setncattr("units", units)
            v.setncattr("description", desc)
            v.setncattr("stagger", spec.stagger)
            v.setncattr("sr_x", np.int32(1))
            v.setncattr("sr_y", np.int32(1))

        _write_global_attrs(ds, artifact)
    finally:
        ds.close()
    return path


def _write_global_attrs(ds, artifact: MetEmArtifact) -> None:
    proj = artifact.projection
    nx, ny = proj.nx, proj.ny

    ds.setncattr("TITLE", "OUTPUT FROM GPUWRF METGRID-EQUIVALENT V0.3.0")
    ds.setncattr("SIMULATION_START_DATE", artifact.valid_time)
    ds.setncattr("WEST-EAST_GRID_DIMENSION", np.int32(nx + 1))
    ds.setncattr("SOUTH-NORTH_GRID_DIMENSION", np.int32(ny + 1))
    ds.setncattr("BOTTOM-TOP_GRID_DIMENSION", np.int32(NUM_METGRID_LEVELS))
    ds.setncattr("WEST-EAST_PATCH_START_UNSTAG", np.int32(1))
    ds.setncattr("WEST-EAST_PATCH_END_UNSTAG", np.int32(nx))
    ds.setncattr("WEST-EAST_PATCH_START_STAG", np.int32(1))
    ds.setncattr("WEST-EAST_PATCH_END_STAG", np.int32(nx + 1))
    ds.setncattr("SOUTH-NORTH_PATCH_START_UNSTAG", np.int32(1))
    ds.setncattr("SOUTH-NORTH_PATCH_END_UNSTAG", np.int32(ny))
    ds.setncattr("SOUTH-NORTH_PATCH_START_STAG", np.int32(1))
    ds.setncattr("SOUTH-NORTH_PATCH_END_STAG", np.int32(ny + 1))
    ds.setncattr("GRIDTYPE", "C")
    ds.setncattr("DX", np.float32(proj.dx_m))
    ds.setncattr("DY", np.float32(proj.dy_m))
    ds.setncattr("DYN_OPT", np.int32(2))
    ds.setncattr("CEN_LAT", np.float32(proj.moad_cen_lat))
    ds.setncattr("CEN_LON", np.float32(proj.stand_lon))
    ds.setncattr("TRUELAT1", np.float32(proj.truelat1))
    ds.setncattr("TRUELAT2", np.float32(proj.truelat2))
    ds.setncattr("MOAD_CEN_LAT", np.float32(proj.moad_cen_lat))
    ds.setncattr("STAND_LON", np.float32(proj.stand_lon))
    ds.setncattr("POLE_LAT", np.float32(proj.pole_lat))
    ds.setncattr("POLE_LON", np.float32(proj.pole_lon))
    ds.setncattr("MAP_PROJ", np.int32(proj.map_proj))
    ds.setncattr("MMINLU", proj.mminlu)
    ds.setncattr("NUM_LAND_CAT", np.int32(proj.num_land_cat))
    ds.setncattr("ISWATER", np.int32(proj.iswater))
    ds.setncattr("ISLAKE", np.int32(proj.islake))
    ds.setncattr("ISICE", np.int32(proj.isice))
    ds.setncattr("ISURBAN", np.int32(proj.isurban))
    ds.setncattr("ISOILWATER", np.int32(proj.isoilwater))
    ds.setncattr("grid_id", np.int32(proj.grid_id))
    ds.setncattr("parent_id", np.int32(proj.parent_id))
    ds.setncattr("i_parent_start", np.int32(proj.i_parent_start))
    ds.setncattr("j_parent_start", np.int32(proj.j_parent_start))
    ds.setncattr("i_parent_end", np.int32(nx + 1))
    ds.setncattr("j_parent_end", np.int32(ny + 1))
    ds.setncattr("parent_grid_ratio", np.int32(proj.parent_grid_ratio))
    ds.setncattr("sr_x", np.int32(1))
    ds.setncattr("sr_y", np.int32(1))
    ds.setncattr("NUM_METGRID_SOIL_LEVELS", np.int32(artifact._dim_size("num_st_layers")))

    # FLAG_* block (real.exe reads these). Set 1 for fields present; the recon
    # confirms the full set is 1 for the Canary case.
    flags = {
        "FLAG_METGRID": 1,
        "FLAG_EXCLUDED_MIDDLE": 0,
        "FLAG_SOIL_LAYERS": 1,
        "FLAG_PSFC": 1,
        "FLAG_SM000010": 1,
        "FLAG_SM010040": 1,
        "FLAG_ST000010": 1,
        "FLAG_ST010040": 1,
        "FLAG_SLP": 1 if "PMSL" in artifact.arrays else 0,
        "FLAG_SH": 1,
        "FLAG_SOILHGT": 1 if "SOILHGT" in artifact.arrays else 0,
        "FLAG_MF_XY": 1,
        "FLAG_LAI12M": 1 if "LAI12M" in artifact.arrays else 0,
    }
    for k, val in flags.items():
        ds.setncattr(k, np.int32(val))
