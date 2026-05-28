"""Minimal WRF-compatible NetCDF wrfout writer for M7 output handoff."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import cos, pi
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
from netCDF4 import Dataset

from gpuwrf.physics.surface_constants import CP_D, KARMAN, P0_PA, R_D_OVER_CP, XLV
from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics

P0_THETA_OFFSET_K = 300.0
CP_AIR_J_KG_K = CP_D
LV_J_KG = XLV
DATE_STR_LEN = 19
DEFAULT_SOIL_LAYERS = 4


DOWNSTREAM_CRITICAL_VARIABLES: tuple[str, ...] = (
    "Times",
    "XTIME",
    "XLAT",
    "XLONG",
    "HGT",
    "LANDMASK",
    "LU_INDEX",
    "U10",
    "V10",
    "T2",
    "Q2",
    "PSFC",
    "RAINC",
    "RAINNC",
    "RAINSH",
    "SWDOWN",
    "GLW",
    "PBLH",
    "UST",
    "HFX",
    "LH",
    "TSK",
    "CLDFRA",
    "QCLOUD",
    "QICE",
    "QRAIN",
)

MINIMUM_WRFOUT_VARIABLES: tuple[str, ...] = (
    *DOWNSTREAM_CRITICAL_VARIABLES,
    "XLAT_U",
    "XLONG_U",
    "XLAT_V",
    "XLONG_V",
    "U",
    "V",
    "W",
    "T",
    "QVAPOR",
    "P",
    "PB",
    "PH",
    "PHB",
    "MU",
    "MUB",
)


@dataclass(frozen=True)
class WrfoutVariableSpec:
    """WRF variable schema metadata for the v0 minimum output subset."""

    name: str
    dimensions: tuple[str, ...]
    memory_order: str
    description: str
    units: str
    stagger: str = ""
    coordinates: str | None = None
    dtype: str = "f4"


def _spec(
    name: str,
    dimensions: tuple[str, ...],
    memory_order: str,
    description: str,
    units: str,
    *,
    stagger: str = "",
    coordinates: str | None = None,
    dtype: str = "f4",
) -> WrfoutVariableSpec:
    return WrfoutVariableSpec(
        name=name,
        dimensions=dimensions,
        memory_order=memory_order,
        description=description,
        units=units,
        stagger=stagger,
        coordinates=coordinates,
        dtype=dtype,
    )


XY = ("Time", "south_north", "west_east")
XYZ = ("Time", "bottom_top", "south_north", "west_east")
U_XYZ = ("Time", "bottom_top", "south_north", "west_east_stag")
V_XYZ = ("Time", "bottom_top", "south_north_stag", "west_east")
W_XYZ = ("Time", "bottom_top_stag", "south_north", "west_east")
Z_XYZ = ("Time", "bottom_top_stag", "south_north", "west_east")
U_XY = ("Time", "south_north", "west_east_stag")
V_XY = ("Time", "south_north_stag", "west_east")

WRFOUT_VARIABLE_SPECS: dict[str, WrfoutVariableSpec] = {
    "XTIME": _spec(
        "XTIME",
        ("Time",),
        "0  ",
        "minutes since simulation start",
        "minutes since simulation start",
    ),
    "XLAT": _spec("XLAT", XY, "XY ", "LATITUDE, SOUTH IS NEGATIVE", "degree_north", coordinates="XLONG XLAT"),
    "XLONG": _spec("XLONG", XY, "XY ", "LONGITUDE, WEST IS NEGATIVE", "degree_east", coordinates="XLONG XLAT"),
    "XLAT_U": _spec(
        "XLAT_U",
        U_XY,
        "XY ",
        "LATITUDE, SOUTH IS NEGATIVE",
        "degree_north",
        stagger="X",
        coordinates="XLONG_U XLAT_U",
    ),
    "XLONG_U": _spec(
        "XLONG_U",
        U_XY,
        "XY ",
        "LONGITUDE, WEST IS NEGATIVE",
        "degree_east",
        stagger="X",
        coordinates="XLONG_U XLAT_U",
    ),
    "XLAT_V": _spec(
        "XLAT_V",
        V_XY,
        "XY ",
        "LATITUDE, SOUTH IS NEGATIVE",
        "degree_north",
        stagger="Y",
        coordinates="XLONG_V XLAT_V",
    ),
    "XLONG_V": _spec(
        "XLONG_V",
        V_XY,
        "XY ",
        "LONGITUDE, WEST IS NEGATIVE",
        "degree_east",
        stagger="Y",
        coordinates="XLONG_V XLAT_V",
    ),
    "HGT": _spec("HGT", XY, "XY ", "Terrain Height", "m", coordinates="XLONG XLAT XTIME"),
    "LANDMASK": _spec(
        "LANDMASK",
        XY,
        "XY ",
        "LAND MASK (1 FOR LAND, 0 FOR WATER)",
        "",
        coordinates="XLONG XLAT XTIME",
    ),
    "LU_INDEX": _spec("LU_INDEX", XY, "XY ", "LAND USE CATEGORY", "", coordinates="XLONG XLAT XTIME"),
    "U10": _spec("U10", XY, "XY ", "U at 10 M", "m s-1", coordinates="XLONG XLAT XTIME"),
    "V10": _spec("V10", XY, "XY ", "V at 10 M", "m s-1", coordinates="XLONG XLAT XTIME"),
    "T2": _spec("T2", XY, "XY ", "TEMP at 2 M", "K", coordinates="XLONG XLAT XTIME"),
    "Q2": _spec("Q2", XY, "XY ", "QV at 2 M", "kg kg-1", coordinates="XLONG XLAT XTIME"),
    "PSFC": _spec("PSFC", XY, "XY ", "SFC PRESSURE", "Pa", coordinates="XLONG XLAT XTIME"),
    "RAINC": _spec(
        "RAINC",
        XY,
        "XY ",
        "ACCUMULATED TOTAL CUMULUS PRECIPITATION",
        "mm",
        coordinates="XLONG XLAT XTIME",
    ),
    "RAINNC": _spec(
        "RAINNC",
        XY,
        "XY ",
        "ACCUMULATED TOTAL GRID SCALE PRECIPITATION",
        "mm",
        coordinates="XLONG XLAT XTIME",
    ),
    "RAINSH": _spec(
        "RAINSH",
        XY,
        "XY ",
        "ACCUMULATED SHALLOW CUMULUS PRECIPITATION",
        "mm",
        coordinates="XLONG XLAT XTIME",
    ),
    "SWDOWN": _spec(
        "SWDOWN",
        XY,
        "XY ",
        "DOWNWARD SHORT WAVE FLUX AT GROUND SURFACE",
        "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "GLW": _spec(
        "GLW",
        XY,
        "XY ",
        "DOWNWARD LONG WAVE FLUX AT GROUND SURFACE",
        "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "PBLH": _spec("PBLH", XY, "XY ", "PBL HEIGHT", "m", coordinates="XLONG XLAT XTIME"),
    "UST": _spec("UST", XY, "XY ", "U* IN SIMILARITY THEORY", "m s-1", coordinates="XLONG XLAT XTIME"),
    "HFX": _spec(
        "HFX",
        XY,
        "XY ",
        "UPWARD HEAT FLUX AT THE SURFACE",
        "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "LH": _spec(
        "LH",
        XY,
        "XY ",
        "LATENT HEAT FLUX AT THE SURFACE",
        "W m-2",
        coordinates="XLONG XLAT XTIME",
    ),
    "TSK": _spec(
        "TSK",
        XY,
        "XY ",
        "SURFACE SKIN TEMPERATURE",
        "K",
        coordinates="XLONG XLAT XTIME",
    ),
    "CLDFRA": _spec("CLDFRA", XYZ, "XYZ", "CLOUD FRACTION", "", coordinates="XLONG XLAT XTIME"),
    "QCLOUD": _spec(
        "QCLOUD",
        XYZ,
        "XYZ",
        "Cloud water mixing ratio",
        "kg kg-1",
        coordinates="XLONG XLAT XTIME",
    ),
    "QICE": _spec("QICE", XYZ, "XYZ", "Ice mixing ratio", "kg kg-1", coordinates="XLONG XLAT XTIME"),
    "QRAIN": _spec(
        "QRAIN",
        XYZ,
        "XYZ",
        "Rain water mixing ratio",
        "kg kg-1",
        coordinates="XLONG XLAT XTIME",
    ),
    "U": _spec(
        "U",
        U_XYZ,
        "XYZ",
        "x-wind component",
        "m s-1",
        stagger="X",
        coordinates="XLONG_U XLAT_U XTIME",
    ),
    "V": _spec(
        "V",
        V_XYZ,
        "XYZ",
        "y-wind component",
        "m s-1",
        stagger="Y",
        coordinates="XLONG_V XLAT_V XTIME",
    ),
    "W": _spec(
        "W",
        W_XYZ,
        "XYZ",
        "z-wind component",
        "m s-1",
        stagger="Z",
        coordinates="XLONG XLAT XTIME",
    ),
    "T": _spec(
        "T",
        XYZ,
        "XYZ",
        "perturbation potential temperature theta-t0",
        "K",
        coordinates="XLONG XLAT XTIME",
    ),
    "QVAPOR": _spec(
        "QVAPOR",
        XYZ,
        "XYZ",
        "Water vapor mixing ratio",
        "kg kg-1",
        coordinates="XLONG XLAT XTIME",
    ),
    "P": _spec("P", XYZ, "XYZ", "perturbation pressure", "Pa", coordinates="XLONG XLAT XTIME"),
    "PB": _spec("PB", XYZ, "XYZ", "BASE STATE PRESSURE", "Pa", coordinates="XLONG XLAT XTIME"),
    "PH": _spec(
        "PH",
        Z_XYZ,
        "XYZ",
        "perturbation geopotential",
        "m2 s-2",
        stagger="Z",
        coordinates="XLONG XLAT XTIME",
    ),
    "PHB": _spec(
        "PHB",
        Z_XYZ,
        "XYZ",
        "base-state geopotential",
        "m2 s-2",
        stagger="Z",
        coordinates="XLONG XLAT XTIME",
    ),
    "MU": _spec(
        "MU",
        XY,
        "XY ",
        "perturbation dry air mass in column",
        "Pa",
        coordinates="XLONG XLAT XTIME",
    ),
    "MUB": _spec(
        "MUB",
        XY,
        "XY ",
        "base state dry air mass in column",
        "Pa",
        coordinates="XLONG XLAT XTIME",
    ),
}


def write_wrfout_netcdf(
    state: Any,
    grid: Any,
    namelist: Mapping[str, Any] | Any | None,
    path: str | Path,
    *,
    valid_time: datetime | date | str,
    lead_hours: float,
    run_start: datetime | date | str,
) -> Path:
    """Write one WRF-style ``wrfout`` NetCDF file for the M7 minimum variable set.

    The function accepts plain Python/numpy objects as well as the project
    ``State``/``GridSpec`` objects. Device arrays, if passed after an operational
    run, are converted only at this output boundary.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    run_start_dt = _coerce_datetime(run_start)
    valid_dt = _coerce_datetime(valid_time)
    nx, ny, nz = _grid_extent(grid)
    dimensions = _dimension_sizes(nx=nx, ny=ny, nz=nz, namelist=namelist)
    fields = _build_output_fields(state, grid, namelist, dimensions)

    with Dataset(target, "w", format="NETCDF4") as dataset:
        _create_dimensions(dataset, dimensions)
        _write_global_attrs(dataset, grid, namelist, dimensions, run_start_dt, valid_dt)
        _write_times(dataset, valid_dt)
        _write_xtime(dataset, run_start_dt, lead_hours)
        for name in MINIMUM_WRFOUT_VARIABLES:
            if name in {"Times", "XTIME"}:
                continue
            spec = WRFOUT_VARIABLE_SPECS[name]
            _write_float_variable(dataset, spec, fields[name], dimensions)
    return target


def _dimension_sizes(*, nx: int, ny: int, nz: int, namelist: Mapping[str, Any] | Any | None) -> dict[str, int | None]:
    soil_layers = int(_lookup(namelist, "soil_layers_stag", DEFAULT_SOIL_LAYERS))
    return {
        "Time": None,
        "DateStrLen": DATE_STR_LEN,
        "west_east": int(nx),
        "west_east_stag": int(nx) + 1,
        "south_north": int(ny),
        "south_north_stag": int(ny) + 1,
        "bottom_top": int(nz),
        "bottom_top_stag": int(nz) + 1,
        "soil_layers_stag": soil_layers,
    }


def _create_dimensions(dataset: Dataset, dimensions: Mapping[str, int | None]) -> None:
    for name in (
        "Time",
        "DateStrLen",
        "west_east",
        "west_east_stag",
        "south_north",
        "south_north_stag",
        "bottom_top",
        "bottom_top_stag",
        "soil_layers_stag",
    ):
        dataset.createDimension(name, dimensions[name])


def _write_times(dataset: Dataset, valid_time: datetime) -> None:
    times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
    encoded = _wrf_time_string(valid_time).encode("ascii")
    times[0, :] = np.frombuffer(encoded, dtype="S1")


def _write_xtime(dataset: Dataset, run_start: datetime, lead_hours: float) -> None:
    spec = WRFOUT_VARIABLE_SPECS["XTIME"]
    xtime = dataset.createVariable("XTIME", spec.dtype, spec.dimensions)
    xtime[:] = np.asarray([float(lead_hours) * 60.0], dtype=np.float32)
    _set_variable_attrs(xtime, spec)
    time_label = run_start.strftime("%Y-%m-%d %H:%M:%S")
    xtime.description = f"minutes since {time_label}"
    xtime.units = f"minutes since {time_label}"


def _write_float_variable(
    dataset: Dataset,
    spec: WrfoutVariableSpec,
    data: np.ndarray,
    dimensions: Mapping[str, int | None],
) -> None:
    expected_shape = _shape_for_dimensions(spec.dimensions, dimensions)
    array = _coerce_array(spec.name, data, expected_shape)
    variable = dataset.createVariable(spec.name, spec.dtype, spec.dimensions)
    _set_variable_attrs(variable, spec)
    variable[0, ...] = array


def _set_variable_attrs(variable: Any, spec: WrfoutVariableSpec) -> None:
    variable.FieldType = np.int32(104)
    variable.MemoryOrder = spec.memory_order
    variable.description = spec.description
    variable.units = spec.units
    variable.stagger = spec.stagger
    if spec.coordinates is not None:
        variable.coordinates = spec.coordinates


def _write_global_attrs(
    dataset: Dataset,
    grid: Any,
    namelist: Mapping[str, Any] | Any | None,
    dimensions: Mapping[str, int | None],
    run_start: datetime,
    valid_time: datetime,
) -> None:
    projection = _lookup(grid, "projection")
    lat_0 = float(_lookup(projection, "lat_0", _lookup(namelist, "cen_lat", 0.0)))
    lon_0 = float(_lookup(projection, "lon_0", _lookup(namelist, "cen_lon", 0.0)))
    dx_m = float(_lookup(projection, "dx_m", _lookup(namelist, "dx", 0.0)))
    dy_m = float(_lookup(projection, "dy_m", _lookup(namelist, "dy", dx_m)))
    kind = str(_lookup(projection, "kind", "lambert")).lower()
    map_proj = {"lambert": 1, "polar": 2, "mercator": 3}.get(kind, 1)

    attrs: dict[str, Any] = {
        "TITLE": str(_lookup(namelist, "title", " OUTPUT FROM GPUWRF WRF-COMPATIBLE NETCDF WRITER")),
        "START_DATE": _wrf_time_string(run_start),
        "SIMULATION_START_DATE": _wrf_time_string(run_start),
        "WEST-EAST_GRID_DIMENSION": np.int32(int(dimensions["west_east_stag"])),
        "SOUTH-NORTH_GRID_DIMENSION": np.int32(int(dimensions["south_north_stag"])),
        "BOTTOM-TOP_GRID_DIMENSION": np.int32(int(dimensions["bottom_top_stag"])),
        "DX": np.float32(dx_m),
        "DY": np.float32(dy_m),
        "GRIDTYPE": "C",
        "MAP_PROJ": np.int32(map_proj),
        "CEN_LAT": np.float32(_lookup(namelist, "cen_lat", lat_0)),
        "CEN_LON": np.float32(_lookup(namelist, "cen_lon", lon_0)),
        "TRUELAT1": np.float32(_lookup(namelist, "truelat1", lat_0)),
        "TRUELAT2": np.float32(_lookup(namelist, "truelat2", lat_0)),
        "MOAD_CEN_LAT": np.float32(_lookup(namelist, "moad_cen_lat", lat_0)),
        "STAND_LON": np.float32(_lookup(namelist, "stand_lon", lon_0)),
        "GMT": np.float32(run_start.hour + run_start.minute / 60.0),
        "JULYR": np.int32(valid_time.year),
        "JULDAY": np.int32(valid_time.timetuple().tm_yday),
        "ISWATER": np.int32(_lookup(namelist, "iswater", 17)),
        "ISLAKE": np.int32(_lookup(namelist, "islake", 21)),
        "ISICE": np.int32(_lookup(namelist, "isice", 15)),
        "ISURBAN": np.int32(_lookup(namelist, "isurban", 13)),
        "ISOILWATER": np.int32(_lookup(namelist, "isoilwater", 14)),
        "Conventions": "WRF-ARW",
        "history": "Created by gpuwrf.io.wrfout_writer.write_wrfout_netcdf",
    }
    for name, value in attrs.items():
        dataset.setncattr(name, value)


def _build_output_fields(
    state: Any,
    grid: Any,
    namelist: Mapping[str, Any] | Any | None,
    dimensions: Mapping[str, int | None],
) -> dict[str, np.ndarray]:
    shape_xy = _shape_for_dimensions(XY, dimensions)
    shape_xyz = _shape_for_dimensions(XYZ, dimensions)
    shape_u = _shape_for_dimensions(U_XYZ, dimensions)
    shape_v = _shape_for_dimensions(V_XYZ, dimensions)
    shape_w = _shape_for_dimensions(W_XYZ, dimensions)
    shape_u_xy = _shape_for_dimensions(U_XY, dimensions)
    shape_v_xy = _shape_for_dimensions(V_XY, dimensions)
    shape_z = _shape_for_dimensions(Z_XYZ, dimensions)

    u = _field_array(state, ("U", "u"), shape_u)
    v = _field_array(state, ("V", "v"), shape_v)
    w = _field_array(state, ("W", "w"), shape_w)
    theta = _field_array(state, ("theta", "THETA"), shape_xyz, default=300.0)
    qv = _field_array(state, ("QVAPOR", "qv", "qvapor"), shape_xyz)
    qc = _field_array(state, ("QCLOUD", "qc", "qcloud"), shape_xyz)
    qi = _field_array(state, ("QICE", "qi", "qice"), shape_xyz)
    qr = _field_array(state, ("QRAIN", "qr", "qrain"), shape_xyz)

    p_pert, p_base = _perturbation_base_pair(
        state,
        total_names=("p_total", "p", "P_total"),
        perturbation_names=("p_perturbation", "P"),
        base_names=("pb", "p_base", "PB"),
        shape=shape_xyz,
    )
    ph_pert, ph_base = _perturbation_base_pair(
        state,
        total_names=("ph_total", "ph", "PH_total"),
        perturbation_names=("ph_perturbation", "PH"),
        base_names=("phb", "ph_base", "PHB"),
        shape=shape_z,
    )
    mu_pert, mu_base = _perturbation_base_pair(
        state,
        total_names=("mu_total", "mu", "MU_total"),
        perturbation_names=("mu_perturbation", "MU"),
        base_names=("mub", "mu_base", "MUB"),
        shape=shape_xy,
    )

    xlat, xlong = _latlon_fields(state, grid, namelist, shape_xy)
    xlat_u, xlong_u = _latlon_fields(state, grid, namelist, shape_u_xy, suffix="_u")
    xlat_v, xlong_v = _latlon_fields(state, grid, namelist, shape_v_xy, suffix="_v")

    terrain = _grid_or_state_array(state, grid, ("HGT", "hgt", "terrain_height"), shape_xy)
    landmask = _landmask(state, shape_xy)
    lu_index = _field_array(state, ("LU_INDEX", "lu_index", "ivgtyp"), shape_xy, default=np.where(landmask > 0.5, 2.0, 17.0))
    hfx = _optional_field_array(state, ("HFX", "hfx"), shape_xy)
    lh = _optional_field_array(state, ("LH", "lh"), shape_xy)
    if hfx is None or lh is None:
        surface_fluxes = _surface_flux_fallbacks(
            state=state,
            grid=grid,
            theta=theta,
            qv=qv,
            p_total=p_pert + p_base,
            ph_total=ph_pert + ph_base,
            u=u,
            v=v,
            landmask=landmask,
            shape_xy=shape_xy,
        )
        if hfx is None:
            hfx = surface_fluxes["HFX"]
        if lh is None:
            lh = surface_fluxes["LH"]

    fields = {
        "XLAT": xlat,
        "XLONG": xlong,
        "XLAT_U": xlat_u,
        "XLONG_U": xlong_u,
        "XLAT_V": xlat_v,
        "XLONG_V": xlong_v,
        "HGT": terrain,
        "LANDMASK": landmask,
        "LU_INDEX": lu_index,
        "U": u,
        "V": v,
        "W": w,
        "T": theta - P0_THETA_OFFSET_K,
        "QVAPOR": qv,
        "P": p_pert,
        "PB": p_base,
        "PH": ph_pert,
        "PHB": ph_base,
        "MU": mu_pert,
        "MUB": mu_base,
        "QCLOUD": qc,
        "QICE": qi,
        "QRAIN": qr,
        "CLDFRA": _field_array(
            state,
            ("CLDFRA", "cldfra", "cloud_fraction"),
            shape_xyz,
            default=np.where((qc + qi + qr) > 1.0e-8, 1.0, 0.0),
        ),
        "U10": _field_array(state, ("U10", "u10"), shape_xy, default=_unstagger_x(u)[0]),
        "V10": _field_array(state, ("V10", "v10"), shape_xy, default=_unstagger_y(v)[0]),
        "T2": _field_array(state, ("T2", "t2"), shape_xy, default=theta[0]),
        "Q2": _field_array(state, ("Q2", "q2"), shape_xy, default=qv[0]),
        "PSFC": _field_array(state, ("PSFC", "psfc"), shape_xy, default=p_pert[0] + p_base[0]),
        "RAINC": _field_array(state, ("RAINC", "rainc"), shape_xy),
        "RAINNC": _field_array(state, ("RAINNC", "rainnc", "rain_acc"), shape_xy),
        "RAINSH": _field_array(state, ("RAINSH", "rainsh"), shape_xy),
        "SWDOWN": _field_array(state, ("SWDOWN", "swdown"), shape_xy),
        "GLW": _field_array(state, ("GLW", "glw"), shape_xy),
        "PBLH": _field_array(state, ("PBLH", "pblh"), shape_xy),
        "UST": _field_array(state, ("UST", "ustar"), shape_xy),
        "HFX": hfx,
        "LH": lh,
        "TSK": _field_array(state, ("TSK", "tsk", "t_skin"), shape_xy, default=theta[0]),
    }
    return {name: np.asarray(value, dtype=np.float32) for name, value in fields.items()}


def _surface_flux_fallbacks(
    *,
    state: Any,
    grid: Any,
    theta: np.ndarray,
    qv: np.ndarray,
    p_total: np.ndarray,
    ph_total: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    landmask: np.ndarray,
    shape_xy: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Recompute sfclay-style HFX/LH when direct WRF flux diagnostics are absent."""

    u_mass = _unstagger_x(u)
    v_mass = _unstagger_y(v)
    dz = np.maximum((ph_total[1:, :, :] - ph_total[:-1, :, :]) / 9.80665, 1.0)
    t_air0 = theta[0] * (np.maximum(p_total[0], 1.0) / P0_PA) ** R_D_OVER_CP
    t_skin = _field_array(state, ("TSK", "tsk", "t_skin"), shape_xy, default=t_air0)
    xland = _field_array(state, ("xland", "XLAND"), shape_xy, default=np.where(landmask > 0.5, 1.0, 2.0))
    soil_moisture = _optional_field_array(state, ("soil_moisture", "SMOIS"), shape_xy)
    mavail = _optional_field_array(state, ("mavail", "MAVAIL"), shape_xy)
    if mavail is None:
        mavail = soil_moisture if soil_moisture is not None else np.ones(shape_xy, dtype=np.float32)
    column_state = SimpleNamespace(
        u=np.moveaxis(u_mass, 0, -1),
        v=np.moveaxis(v_mass, 0, -1),
        theta=np.moveaxis(theta, 0, -1),
        qv=np.moveaxis(qv, 0, -1),
        p=np.moveaxis(p_total, 0, -1),
        dz=np.moveaxis(dz, 0, -1),
        t_skin=t_skin,
        soil_moisture=soil_moisture,
        xland=xland,
        lakemask=_field_array(state, ("lakemask", "LAKEMASK"), shape_xy),
        mavail=mavail,
        roughness_m=_optional_field_array(state, ("roughness_m", "ZNT"), shape_xy),
        ustar=_field_array(state, ("UST", "ustar"), shape_xy),
        dx_m=float(_lookup(_lookup(grid, "projection"), "dx_m", _lookup(grid, "dx", 3000.0))),
    )
    diag = surface_layer_with_diagnostics(column_state)
    ustar = np.asarray(diag.fluxes.ustar, dtype=np.float64)
    rhosfc = np.asarray(diag.fluxes.rhosfc, dtype=np.float64)
    fh = np.asarray(diag.fh, dtype=np.float64)
    qv_flux = np.asarray(diag.fluxes.qv_flux, dtype=np.float64)

    theta_air = theta[0].astype(np.float64)
    theta_surface = np.asarray(t_skin, dtype=np.float64) * (P0_PA / np.maximum(p_total[0], 1.0)) ** R_D_OVER_CP
    cpm = CP_AIR_J_KG_K * (1.0 + 0.8 * np.maximum(qv[0].astype(np.float64), 0.0))
    aerodynamic_resistance = fh / np.maximum(KARMAN * ustar, 1.0e-12)
    hfx = rhosfc * cpm * (theta_surface - theta_air) / np.maximum(aerodynamic_resistance, 1.0e-12)
    lh = qv_flux * rhosfc * LV_J_KG
    return {"HFX": hfx.astype(np.float32), "LH": lh.astype(np.float32)}


def _perturbation_base_pair(
    state: Any,
    *,
    total_names: tuple[str, ...],
    perturbation_names: tuple[str, ...],
    base_names: tuple[str, ...],
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    total = _optional_field_array(state, total_names, shape)
    perturbation = _optional_field_array(state, perturbation_names, shape)
    base = _optional_field_array(state, base_names, shape)
    zeros = np.zeros(shape, dtype=np.float32)

    if total is None and perturbation is None and base is None:
        return zeros, zeros
    if total is None and perturbation is not None and base is not None:
        total = perturbation + base
    if perturbation is None and total is not None and base is not None:
        perturbation = total - base
    if base is None and total is not None and perturbation is not None:
        base = total - perturbation
    if perturbation is None and total is not None:
        perturbation = total
    if base is None:
        base = zeros
    return perturbation.astype(np.float32), base.astype(np.float32)


def _field_array(
    obj: Any,
    names: tuple[str, ...],
    shape: tuple[int, ...],
    *,
    default: Any = 0.0,
) -> np.ndarray:
    value = _optional_field_array(obj, names, shape)
    if value is None:
        value = default
    return _coerce_array(names[0], value, shape)


def _optional_field_array(obj: Any, names: tuple[str, ...], shape: tuple[int, ...]) -> np.ndarray | None:
    for name in names:
        value = _lookup(obj, name, None)
        if value is not None:
            return _coerce_array(name, value, shape)
    return None


def _grid_or_state_array(
    state: Any,
    grid: Any,
    names: tuple[str, ...],
    shape: tuple[int, ...],
    *,
    default: Any = 0.0,
) -> np.ndarray:
    value = _optional_field_array(state, names, shape)
    if value is None:
        value = _optional_field_array(grid, names, shape)
    if value is None:
        value = default
    return _coerce_array(names[0], value, shape)


def _landmask(state: Any, shape: tuple[int, ...]) -> np.ndarray:
    explicit = _optional_field_array(state, ("LANDMASK", "landmask"), shape)
    if explicit is not None:
        return explicit
    xland = _optional_field_array(state, ("xland", "XLAND"), shape)
    if xland is None:
        return np.ones(shape, dtype=np.float32)
    return np.where(xland <= 1.5, 1.0, 0.0).astype(np.float32)


def _latlon_fields(
    state: Any,
    grid: Any,
    namelist: Mapping[str, Any] | Any | None,
    shape: tuple[int, int],
    *,
    suffix: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    lat_names = ("XLAT" + suffix.upper(), "xlat" + suffix, "lat" + suffix)
    lon_names = ("XLONG" + suffix.upper(), "xlong" + suffix, "lon" + suffix)
    lat = _optional_field_array(state, lat_names, shape)
    lon = _optional_field_array(state, lon_names, shape)
    if lat is not None and lon is not None:
        return lat, lon
    projection = _lookup(grid, "projection")
    lat_0 = float(_lookup(projection, "lat_0", _lookup(namelist, "cen_lat", 0.0)))
    lon_0 = float(_lookup(projection, "lon_0", _lookup(namelist, "cen_lon", 0.0)))
    dx_m = float(_lookup(projection, "dx_m", _lookup(namelist, "dx", 3000.0)))
    dy_m = float(_lookup(projection, "dy_m", _lookup(namelist, "dy", dx_m)))
    ny, nx = shape
    y = np.arange(ny, dtype=np.float64) - (ny - 1) / 2.0
    x = np.arange(nx, dtype=np.float64) - (nx - 1) / 2.0
    lat_step = dy_m / 111_320.0
    lon_step = dx_m / max(111_320.0 * cos(lat_0 * pi / 180.0), 1.0)
    lat_grid = lat_0 + y[:, None] * lat_step
    lon_grid = lon_0 + x[None, :] * lon_step
    return (
        np.broadcast_to(lat_grid, shape).astype(np.float32),
        np.broadcast_to(lon_grid, shape).astype(np.float32),
    )


def _unstagger_x(u: np.ndarray) -> np.ndarray:
    return 0.5 * (u[..., :-1] + u[..., 1:])


def _unstagger_y(v: np.ndarray) -> np.ndarray:
    return 0.5 * (v[..., :-1, :] + v[..., 1:, :])


def _shape_for_dimensions(dimensions: tuple[str, ...], sizes: Mapping[str, int | None]) -> tuple[int, ...]:
    return tuple(int(sizes[name]) for name in dimensions if name != "Time")


def _coerce_array(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape == (1, *shape):
        array = array[0]
    if array.shape == shape:
        return array.astype(np.float32, copy=False)
    if array.shape == ():
        return np.full(shape, float(array), dtype=np.float32)
    try:
        return np.broadcast_to(array, shape).astype(np.float32)
    except ValueError as exc:
        raise ValueError(f"{name} shape {array.shape} cannot be written to WRF shape {shape}") from exc


def _grid_extent(grid: Any) -> tuple[int, int, int]:
    nx = _lookup(grid, "nx", None)
    ny = _lookup(grid, "ny", None)
    nz = _lookup(grid, "nz", None)
    projection = _lookup(grid, "projection")
    vertical = _lookup(grid, "vertical")
    if nx is None:
        nx = _lookup(projection, "nx", None)
    if ny is None:
        ny = _lookup(projection, "ny", None)
    if nz is None:
        nz = _lookup(vertical, "nz", None)
    missing = [name for name, value in {"nx": nx, "ny": ny, "nz": nz}.items() if value is None]
    if missing:
        raise ValueError(f"grid is missing {', '.join(missing)}")
    return int(nx), int(ny), int(nz)


def _lookup(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _coerce_datetime(value: datetime | date | str) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    text = str(value).strip().replace("Z", "")
    for fmt in ("%Y-%m-%d_%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return datetime.fromisoformat(text).replace(tzinfo=None)


def _wrf_time_string(value: datetime) -> str:
    return value.strftime("%Y-%m-%d_%H:%M:%S")
