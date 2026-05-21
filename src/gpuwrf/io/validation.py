"""Shared validation I/O helpers for M6 and later validation sprints."""

from __future__ import annotations

import re
from typing import Any, Literal

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover
    jax = None
    jnp = None

from gpuwrf.io.gen2_accessor import Gen2Run


Region = Literal["canary", "land", "sea"]


def load_gen2_var(run: Gen2Run, domain: str, var: str, time: int | str | None):
    """Load one Gen2 variable as a device array through the shared accessor."""

    return run.load(domain, var, time=time, lazy=False)


def _as_numpy(field: Any) -> np.ndarray:
    return np.asarray(field)


def _grid_shape(grid: Any, fallback: tuple[int, int]) -> tuple[int, int]:
    ny = getattr(grid, "ny", None)
    nx = getattr(grid, "nx", None)
    if ny is not None and nx is not None:
        return int(ny), int(nx)
    projection = getattr(grid, "projection", None)
    if projection is not None:
        return int(projection.ny), int(projection.nx)
    return fallback


def _linear_axis(src_n: int, dst_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dst_n <= 1:
        coords = np.zeros((dst_n,), dtype=np.float64)
    else:
        coords = np.linspace(0.0, float(src_n - 1), dst_n, dtype=np.float64)
    lower = np.floor(coords).astype(np.int64)
    lower = np.clip(lower, 0, max(src_n - 2, 0))
    upper = np.clip(lower + 1, 0, src_n - 1)
    weight = coords - lower
    return lower, upper, weight


def _bilinear_2d(field: np.ndarray, dst_shape: tuple[int, int]) -> np.ndarray:
    src_y, src_x = field.shape
    dst_y, dst_x = dst_shape
    y0, y1, wy = _linear_axis(src_y, dst_y)
    x0, x1, wx = _linear_axis(src_x, dst_x)
    f00 = field[y0[:, None], x0[None, :]]
    f10 = field[y0[:, None], x1[None, :]]
    f01 = field[y1[:, None], x0[None, :]]
    f11 = field[y1[:, None], x1[None, :]]
    wx2 = wx[None, :]
    wy2 = wy[:, None]
    return f00 * (1.0 - wx2) * (1.0 - wy2) + f10 * wx2 * (1.0 - wy2) + f01 * (1.0 - wx2) * wy2 + f11 * wx2 * wy2


def regrid(src_field: Any, src_grid: Any, dst_grid: Any, method: str = "bilinear"):
    """Regrid a 2-D or 3-D field using the shared M6 validation path.

    This helper covers the regular-grid cases used by M6 validators. The d02
    boundary replay uses the stricter WRF Lambert/curvilinear helper in
    `boundary_replay.py` because lateral forcing needs stagger-specific
    coordinate handling.
    """

    if method != "bilinear":
        raise ValueError(f"unsupported regrid method {method!r}; only 'bilinear' is frozen for M6")
    src = _as_numpy(src_field)
    dst_shape = _grid_shape(dst_grid, src.shape[-2:])
    if src.ndim == 2:
        out = _bilinear_2d(src, dst_shape)
    elif src.ndim == 3:
        out = np.stack([_bilinear_2d(level, dst_shape) for level in src], axis=0)
    else:
        raise ValueError(f"regrid expects 2-D or 3-D fields; got shape {src.shape}")
    return jax.device_put(out) if jax is not None else out


def domain_mask(grid: Any, region: str = "canary") -> np.ndarray:
    """Return a boolean mask for the requested Canary validation region."""

    shape = _grid_shape(grid, (1, 1))
    if region == "canary":
        return np.ones(shape, dtype=bool)

    landmask = None
    if hasattr(grid, "static_field"):
        try:
            landmask = np.asarray(grid.static_field("LANDMASK"))
        except KeyError:
            landmask = None
    if landmask is not None:
        land = landmask > 0.5
        if region == "land":
            return land
        if region == "sea":
            return ~land

    terrain = None
    if hasattr(grid, "static_field"):
        try:
            terrain = np.asarray(grid.static_field("HGT"))
        except KeyError:
            terrain = None
    elif hasattr(grid, "terrain_height"):
        terrain = np.asarray(grid.terrain_height)

    if region == "land":
        return np.ones(shape, dtype=bool) if terrain is None else terrain > 0.0
    if region == "sea":
        return np.zeros(shape, dtype=bool) if terrain is None else terrain <= 0.0

    match = re.fullmatch(r"elevation_band_(?P<band>\d+)", region)
    if match is not None:
        if terrain is None:
            raise ValueError("elevation-band masks require HGT/terrain_height metadata")
        band = int(match.group("band"))
        low = 500.0 * band
        high = low + 500.0
        return (terrain >= low) & (terrain < high)

    raise ValueError(f"unknown domain mask region {region!r}")


def lead_time_slice(run: Gen2Run, lead_hours: int | float) -> int:
    """Return the history-file index closest to a requested lead hour."""

    history_interval = run.namelist.get("time_control", {}).get("history_interval", 60)
    if isinstance(history_interval, list):
        minutes = float(history_interval[0])
    else:
        minutes = float(history_interval)
    hours_per_output = minutes / 60.0
    if hours_per_output <= 0.0:
        raise ValueError("history_interval must be positive")
    return int(round(float(lead_hours) / hours_per_output))


def unit_convert(field: Any, from_unit: str, to_unit: str):
    """Convert common WRF validation units."""

    src = from_unit.strip().lower().replace("degc", "c").replace("°c", "c")
    dst = to_unit.strip().lower().replace("degc", "c").replace("°c", "c")
    data = _as_numpy(field)
    if src == dst:
        out = data
    elif src in {"k", "kelvin"} and dst in {"c", "celsius"}:
        out = data - 273.15
    elif src in {"c", "celsius"} and dst in {"k", "kelvin"}:
        out = data + 273.15
    elif src in {"kg/kg", "kg kg-1", "kg kg^-1"} and dst in {"g/kg", "g kg-1", "g kg^-1"}:
        out = data * 1000.0
    elif src in {"g/kg", "g kg-1", "g kg^-1"} and dst in {"kg/kg", "kg kg-1", "kg kg^-1"}:
        out = data / 1000.0
    elif src == "pa" and dst == "hpa":
        out = data / 100.0
    elif src == "hpa" and dst == "pa":
        out = data * 100.0
    else:
        raise ValueError(f"unsupported unit conversion: {from_unit!r} -> {to_unit!r}")
    return jax.device_put(out) if jax is not None else out


__all__ = ["domain_mask", "lead_time_slice", "load_gen2_var", "regrid", "unit_convert"]
