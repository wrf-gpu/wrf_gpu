"""Forecast-vs-observation validation scaffold for M7 surface scores."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from gpuwrf.config import paths
from gpuwrf.io.data_inventory import parse_wrfout_valid_time
from gpuwrf.io.gen2_wrfout_loader import normalize_valid_time


# Path indirection (see gpuwrf.config.paths): GPUWRF_AEMET_ROOT / GPUWRF_CANAIRY_ROOT
# override; default is checkout-relative. Only used by the optional --score path.
DEFAULT_AEMET_ROOT = paths.aemet_root()
DEFAULT_SCORE_VARIABLES = ("T2", "U10", "V10", "WIND10")
DEFAULT_FSS_THRESHOLD_MM = 1.0
DEFAULT_FSS_WINDOW_CELLS = 9


@dataclass(frozen=True)
class ScoreReport:
    """Station-score aggregate with JSON-friendly serialization."""

    status: str
    variables: tuple[str, ...]
    scores: Mapping[str, Mapping[str, Any]]
    joined_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "M7StationScoreReport",
            "schema_version": 1,
            "status": self.status,
            "variables": list(self.variables),
            "joined_rows": int(self.joined_rows),
            "scores": {key: dict(value) for key, value in self.scores.items()},
        }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isfinite(value):
            return float(value)
        return None
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if pd.isna(value):
        return None
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _paths_from_root(root_or_paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(root_or_paths, (str, Path)):
        root = Path(root_or_paths)
        if root.is_dir():
            return sorted(root.glob("*.parquet"))
        return [root]
    return sorted(Path(path) for path in root_or_paths)


def _canonical_station_metadata(df: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    if "latitude" in df.columns and "lat" not in df.columns:
        rename["latitude"] = "lat"
    if "longitude" in df.columns and "lon" not in df.columns:
        rename["longitude"] = "lon"
    if "elevation_m" in df.columns and "elev_m" not in df.columns:
        rename["elevation_m"] = "elev_m"
    normalized = df.rename(columns=rename).copy()
    required = {"station_id", "lat", "lon"}
    missing = sorted(required - set(normalized.columns))
    if missing:
        raise ValueError(f"station metadata missing required columns: {missing}")
    keep = [column for column in ("station_id", "lat", "lon", "elev_m") if column in normalized.columns]
    normalized = normalized[keep].dropna(subset=["station_id", "lat", "lon"])
    normalized["station_id"] = normalized["station_id"].astype(str)
    return normalized.drop_duplicates(subset=["station_id"], keep="first").reset_index(drop=True)


def _time_series(df: pd.DataFrame) -> pd.Series:
    if "time" in df.columns:
        source = df["time"]
    elif "ts_utc" in df.columns:
        source = df["ts_utc"]
    else:
        raise ValueError("table must contain 'time' or 'ts_utc'")
    return pd.to_datetime(source, utc=True).dt.floor("s")


def inventory_aemet_observations(root_or_paths: str | Path | Iterable[str | Path] = DEFAULT_AEMET_ROOT) -> dict[str, Any]:
    """Enumerate AEMET station parquet files and summarize usable variables."""

    files = _paths_from_root(root_or_paths)
    file_records: list[dict[str, Any]] = []
    all_columns: dict[str, set[str]] = {}
    station_ids: set[str] = set()
    variable_non_null = {
        "T2": 0,
        "U10": 0,
        "V10": 0,
        "WIND10": 0,
        "PRECIP": 0,
        "RH": 0,
        "PRESSURE": 0,
    }
    min_time: pd.Timestamp | None = None
    max_time: pd.Timestamp | None = None
    min_lat = min_lon = np.inf
    max_lat = max_lon = -np.inf

    for path in files:
        df = pd.read_parquet(path)
        dtypes = {column: str(dtype) for column, dtype in df.dtypes.items()}
        for column, dtype in dtypes.items():
            all_columns.setdefault(column, set()).add(dtype)
        time = _time_series(df)
        if not time.empty:
            local_min = time.min()
            local_max = time.max()
            min_time = local_min if min_time is None else min(min_time, local_min)
            max_time = local_max if max_time is None else max(max_time, local_max)
        if "station_id" in df.columns:
            station_ids.update(df["station_id"].dropna().astype(str).unique())
        station_meta = _canonical_station_metadata(df) if {"station_id"}.issubset(df.columns) else pd.DataFrame()
        if not station_meta.empty:
            min_lat = min(min_lat, float(station_meta["lat"].min()))
            max_lat = max(max_lat, float(station_meta["lat"].max()))
            min_lon = min(min_lon, float(station_meta["lon"].min()))
            max_lon = max(max_lon, float(station_meta["lon"].max()))
        if "temp_c" in df.columns:
            variable_non_null["T2"] += int(df["temp_c"].notna().sum())
        if "wind_speed_mps" in df.columns and "wind_dir_deg" in df.columns:
            wind_components = df["wind_speed_mps"].notna() & df["wind_dir_deg"].notna()
            variable_non_null["U10"] += int(wind_components.sum())
            variable_non_null["V10"] += int(wind_components.sum())
            variable_non_null["WIND10"] += int(df["wind_speed_mps"].notna().sum())
        if "precip_mm" in df.columns:
            variable_non_null["PRECIP"] += int(df["precip_mm"].notna().sum())
        if "rh_pct" in df.columns:
            variable_non_null["RH"] += int(df["rh_pct"].notna().sum())
        if "pressure_hpa" in df.columns:
            variable_non_null["PRESSURE"] += int(df["pressure_hpa"].notna().sum())
        file_records.append(
            {
                "path": str(path),
                "file": path.name,
                "row_count": int(len(df)),
                "station_count": int(df["station_id"].nunique()) if "station_id" in df.columns else 0,
                "columns": dtypes,
                "time_start": time.min().isoformat() if not time.empty else None,
                "time_end": time.max().isoformat() if not time.empty else None,
            }
        )

    variables_present = {name: count for name, count in variable_non_null.items() if count > 0}
    bbox = None
    if np.isfinite([min_lat, min_lon, max_lat, max_lon]).all():
        bbox = {"lat_min": min_lat, "lat_max": max_lat, "lon_min": min_lon, "lon_max": max_lon}
    return {
        "schema": "M7AemetObservationInventory",
        "schema_version": 1,
        "source": "AEMET station parquet",
        "file_count": int(len(files)),
        "station_count": int(len(station_ids)),
        "schema_columns": {column: sorted(dtypes) for column, dtypes in sorted(all_columns.items())},
        "variables_present": variables_present,
        "temporal_coverage": {
            "start": min_time.isoformat() if min_time is not None else None,
            "end": max_time.isoformat() if max_time is not None else None,
        },
        "spatial_bbox": bbox,
        "files": file_records,
    }


def _aemet_to_canonical(df: pd.DataFrame, variables: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        columns = ["station_id", "time", "lat", "lon", "elev_m", *variables]
        return pd.DataFrame(columns=columns)
    out = df.rename(columns={"latitude": "lat", "longitude": "lon", "elevation_m": "elev_m"}).copy()
    out["station_id"] = out["station_id"].astype(str)
    out["time"] = _time_series(out)
    speed = pd.to_numeric(out["wind_speed_mps"], errors="coerce") if "wind_speed_mps" in out.columns else None
    direction = pd.to_numeric(out["wind_dir_deg"], errors="coerce") if "wind_dir_deg" in out.columns else None
    direction_rad = np.deg2rad(direction.astype(float)) if direction is not None else None
    for variable in variables:
        if variable == "T2" and "temp_c" in out.columns:
            out[variable] = pd.to_numeric(out["temp_c"], errors="coerce") + 273.15
        elif variable == "WIND10" and speed is not None:
            out[variable] = speed
        elif variable == "U10" and speed is not None and direction_rad is not None:
            out[variable] = -speed.astype(float) * np.sin(direction_rad)
        elif variable == "V10" and speed is not None and direction_rad is not None:
            out[variable] = -speed.astype(float) * np.cos(direction_rad)
        elif variable == "PRECIP" and "precip_mm" in out.columns:
            out[variable] = pd.to_numeric(out["precip_mm"], errors="coerce")
    keep = ["station_id", "time", "lat", "lon", "elev_m", *[variable for variable in variables if variable in out.columns]]
    keep = [column for column in keep if column in out.columns]
    return out[keep]


def load_aemet_observations(
    root_or_paths: str | Path | Iterable[str | Path] = DEFAULT_AEMET_ROOT,
    *,
    variables: Sequence[str] = DEFAULT_SCORE_VARIABLES,
    start_time: str | datetime | np.datetime64 | None = None,
    end_time: str | datetime | np.datetime64 | None = None,
) -> pd.DataFrame:
    """Load AEMET rows normalized to forecast variable names and units."""

    files = _paths_from_root(root_or_paths)
    frames: list[pd.DataFrame] = []
    start = pd.Timestamp(normalize_valid_time(start_time)) if start_time is not None else None
    end = pd.Timestamp(normalize_valid_time(end_time)) if end_time is not None else None
    for path in files:
        df = pd.read_parquet(path)
        normalized = _aemet_to_canonical(df, variables)
        if start is not None:
            normalized = normalized[normalized["time"] >= start]
        if end is not None:
            normalized = normalized[normalized["time"] <= end]
        if not normalized.empty:
            frames.append(normalized)
    if not frames:
        return pd.DataFrame(columns=["station_id", "time", "lat", "lon", "elev_m", *variables])
    return pd.concat(frames, ignore_index=True)


def _decode_wrf_time(dataset: Dataset, path: Path) -> datetime:
    try:
        return parse_wrfout_valid_time(path)
    except ValueError:
        if "Times" not in dataset.variables:
            raise
        raw = np.asarray(dataset.variables["Times"][0])
        if raw.dtype.kind == "S":
            stamp = b"".join(raw.tolist()).decode("ascii", errors="replace").strip()
        else:
            stamp = "".join(raw.astype(str).tolist()).strip()
        return normalize_valid_time(stamp)


def _read_2d(dataset: Dataset, name: str) -> np.ndarray:
    if name not in dataset.variables:
        raise KeyError(f"{name!r} not present in {dataset.filepath()}")
    variable = dataset.variables[name]
    data = np.asarray(np.ma.filled(variable[:], np.nan), dtype=np.float64)
    if variable.dimensions and variable.dimensions[0] == "Time":
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"{name!r} must be 2D after Time squeeze, got shape {data.shape}")
    return data


def _read_forecast_field(dataset: Dataset, variable: str) -> np.ndarray:
    if variable == "WIND10":
        return np.hypot(_read_2d(dataset, "U10"), _read_2d(dataset, "V10"))
    if variable == "PRECIP":
        rain = np.zeros_like(_read_2d(dataset, "RAINNC"))
        for name in ("RAINNC", "RAINC"):
            if name in dataset.variables:
                rain = rain + _read_2d(dataset, name)
        return rain
    return _read_2d(dataset, variable)


def _bilinear_inverse(lon: np.ndarray, lat: np.ndarray, station_lon: float, station_lat: float) -> tuple[float, float, float]:
    p00 = np.asarray([lon[0, 0], lat[0, 0]], dtype=np.float64)
    p10 = np.asarray([lon[0, 1], lat[0, 1]], dtype=np.float64)
    p01 = np.asarray([lon[1, 0], lat[1, 0]], dtype=np.float64)
    p11 = np.asarray([lon[1, 1], lat[1, 1]], dtype=np.float64)
    target = np.asarray([station_lon, station_lat], dtype=np.float64)
    linear = np.column_stack((p10 - p00, p01 - p00))
    try:
        tx, ty = np.linalg.solve(linear, target - p00)
    except np.linalg.LinAlgError:
        tx, ty = 0.5, 0.5
    cross = p00 - p10 - p01 + p11
    for _ in range(8):
        point = p00 + tx * (p10 - p00) + ty * (p01 - p00) + tx * ty * cross
        residual = point - target
        jac = np.column_stack((p10 - p00 + ty * cross, p01 - p00 + tx * cross))
        try:
            step = np.linalg.solve(jac, residual)
        except np.linalg.LinAlgError:
            break
        tx -= float(step[0])
        ty -= float(step[1])
        if float(np.linalg.norm(step)) < 1.0e-10:
            break
    point = p00 + tx * (p10 - p00) + ty * (p01 - p00) + tx * ty * cross
    return float(tx), float(ty), float(np.linalg.norm(point - target))


def _cell_candidates(nearest_j: int, nearest_i: int, ny: int, nx: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for j in (nearest_j - 1, nearest_j):
        for i in (nearest_i - 1, nearest_i):
            if 0 <= j < ny - 1 and 0 <= i < nx - 1:
                candidates.append((j, i))
    return candidates


def _station_cell(lats: np.ndarray, lons: np.ndarray, station_lat: float, station_lon: float) -> tuple[int, int, float, float, bool]:
    distance2 = (lats - station_lat) ** 2 + (lons - station_lon) ** 2
    nearest_j, nearest_i = np.unravel_index(int(np.nanargmin(distance2)), lats.shape)
    best: tuple[float, int, int, float, float, bool] | None = None
    for j, i in _cell_candidates(nearest_j, nearest_i, *lats.shape):
        cell_lon = lons[j : j + 2, i : i + 2]
        cell_lat = lats[j : j + 2, i : i + 2]
        tx, ty, residual = _bilinear_inverse(cell_lon, cell_lat, station_lon, station_lat)
        inside = -1.0e-6 <= tx <= 1.0 + 1.0e-6 and -1.0e-6 <= ty <= 1.0 + 1.0e-6
        clipped_tx = float(np.clip(tx, 0.0, 1.0))
        clipped_ty = float(np.clip(ty, 0.0, 1.0))
        priority = residual + (0.0 if inside else 1.0e6)
        candidate = (priority, j, i, clipped_tx, clipped_ty, inside)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        return int(nearest_j), int(nearest_i), 0.0, 0.0, False
    _, j, i, tx, ty, inside = best
    return j, i, tx, ty, inside


def _bilinear_value(field: np.ndarray, j: int, i: int, tx: float, ty: float) -> float:
    values = field[j : j + 2, i : i + 2]
    if values.shape != (2, 2) or not np.isfinite(values).all():
        return float("nan")
    return float(
        (1.0 - tx) * (1.0 - ty) * values[0, 0]
        + tx * (1.0 - ty) * values[0, 1]
        + (1.0 - tx) * ty * values[1, 0]
        + tx * ty * values[1, 1]
    )


def interpolate_to_stations(
    wrfout_path: str | Path,
    station_metadata: pd.DataFrame,
    *,
    variables: Sequence[str],
    valid_time: str | datetime | np.datetime64 | None,
) -> pd.DataFrame:
    """Interpolate 2D WRF fields to station locations with bilinear weights."""

    stations = _canonical_station_metadata(station_metadata)
    output_columns = ["station_id", "time", "lat", "lon", "elev_m", "inside_domain", *variables]
    if stations.empty:
        return pd.DataFrame(columns=output_columns)

    source = Path(wrfout_path)
    with Dataset(source, "r") as dataset:
        lats = _read_2d(dataset, "XLAT")
        lons = _read_2d(dataset, "XLONG")
        fields = {variable: _read_forecast_field(dataset, variable) for variable in variables}
        file_time = _decode_wrf_time(dataset, source)
    timestamp = pd.Timestamp(normalize_valid_time(valid_time) if valid_time is not None else file_time)
    records: list[dict[str, Any]] = []
    for row in stations.itertuples(index=False):
        station_id = str(getattr(row, "station_id"))
        lat = float(getattr(row, "lat"))
        lon = float(getattr(row, "lon"))
        j, i, tx, ty, inside = _station_cell(lats, lons, lat, lon)
        record: dict[str, Any] = {
            "station_id": station_id,
            "time": timestamp,
            "lat": lat,
            "lon": lon,
            "inside_domain": bool(inside),
        }
        if "elev_m" in stations.columns:
            record["elev_m"] = float(getattr(row, "elev_m")) if pd.notna(getattr(row, "elev_m")) else np.nan
        for variable, field in fields.items():
            record[variable] = _bilinear_value(field, j, i, tx, ty) if inside else np.nan
        records.append(record)
    return pd.DataFrame.from_records(records, columns=output_columns)


def _prepare_score_table(df: pd.DataFrame, variables: Sequence[str], suffix: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["station_id", "time", *[f"{variable}{suffix}" for variable in variables]])
    table = df.copy()
    table["station_id"] = table["station_id"].astype(str)
    table["time"] = _time_series(table)
    keep = ["station_id", "time", *[variable for variable in variables if variable in table.columns]]
    table = table[keep]
    return table.rename(columns={variable: f"{variable}{suffix}" for variable in variables if variable in table.columns})


def compute_station_scores(
    forecast_at_stations: pd.DataFrame,
    observations_at_stations: pd.DataFrame,
    *,
    variables: Sequence[str],
) -> ScoreReport:
    """Compute per-variable BIAS, RMSE, MAE, and sample count after time join."""

    variable_order = tuple(variables)
    forecast = _prepare_score_table(forecast_at_stations, variable_order, "_forecast")
    observations = _prepare_score_table(observations_at_stations, variable_order, "_obs")
    joined = forecast.merge(observations, on=["station_id", "time"], how="inner")
    scores: dict[str, dict[str, Any]] = {}
    any_samples = False
    for variable in variable_order:
        f_col = f"{variable}_forecast"
        o_col = f"{variable}_obs"
        if f_col not in joined.columns or o_col not in joined.columns:
            scores[variable] = {
                "status": "MISSING_VARIABLE",
                "sample_count": 0,
                "bias": None,
                "rmse": None,
                "mae": None,
            }
            continue
        pair = joined[[f_col, o_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if pair.empty:
            scores[variable] = {
                "status": "NO_VALID_PAIRS",
                "sample_count": 0,
                "bias": None,
                "rmse": None,
                "mae": None,
            }
            continue
        diff = pair[f_col].to_numpy(dtype=np.float64) - pair[o_col].to_numpy(dtype=np.float64)
        any_samples = True
        scores[variable] = {
            "status": "OK",
            "sample_count": int(diff.size),
            "bias": float(np.mean(diff)),
            "rmse": float(np.sqrt(np.mean(diff * diff))),
            "mae": float(np.mean(np.abs(diff))),
        }
    status = "OK" if any_samples else ("NO_OVERLAP" if joined.empty else "NO_VALID_PAIRS")
    return ScoreReport(status=status, variables=variable_order, scores=scores, joined_rows=int(len(joined)))


def _window_sum(mask: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    radius = window_size // 2
    padded = np.pad(mask.astype(np.float64), ((radius, radius), (radius, radius)), mode="constant", constant_values=0.0)
    integral = np.pad(np.cumsum(np.cumsum(padded, axis=0), axis=1), ((1, 0), (1, 0)), mode="constant")
    y0 = np.arange(mask.shape[0])
    x0 = np.arange(mask.shape[1])
    y1 = y0 + window_size
    x1 = x0 + window_size
    return integral[y1[:, None], x1[None, :]] - integral[y0[:, None], x1[None, :]] - integral[y1[:, None], x0[None, :]] + integral[y0[:, None], x0[None, :]]


def compute_fractions_skill_score(
    forecast_precip_mm: np.ndarray,
    observed_precip_mm: np.ndarray,
    *,
    threshold_mm: float = DEFAULT_FSS_THRESHOLD_MM,
    window_size: int = DEFAULT_FSS_WINDOW_CELLS,
) -> dict[str, Any]:
    """Compute FSS for precipitation exceedance at one neighbourhood scale."""

    forecast = np.asarray(forecast_precip_mm, dtype=np.float64)
    observed = np.asarray(observed_precip_mm, dtype=np.float64)
    if forecast.shape != observed.shape:
        raise ValueError(f"forecast and observed grids must have the same shape, got {forecast.shape} and {observed.shape}")
    finite_forecast = np.isfinite(forecast)
    finite_observed = np.isfinite(observed)
    forecast_mask = (forecast >= threshold_mm) & finite_forecast
    observed_mask = (observed >= threshold_mm) & finite_observed
    denominator_cells = float(window_size * window_size)
    forecast_fraction = _window_sum(forecast_mask, window_size) / denominator_cells
    observed_fraction = _window_sum(observed_mask, window_size) / denominator_cells
    numerator = float(np.mean((forecast_fraction - observed_fraction) ** 2))
    denominator = float(np.mean(forecast_fraction**2 + observed_fraction**2))
    if denominator == 0.0:
        fss = 1.0 if numerator == 0.0 else 0.0
    else:
        fss = 1.0 - numerator / denominator
    return {
        "schema": "M7PrecipFSS",
        "schema_version": 1,
        "threshold_mm": float(threshold_mm),
        "window_size_cells": int(window_size),
        "forecast_grid_shape": [int(value) for value in forecast.shape],
        "forecast_exceedance_cells": int(forecast_mask.sum()),
        "observed_exceedance_cells": int(observed_mask.sum()),
        "observed_finite_cells": int(finite_observed.sum()),
        "fss": float(np.clip(fss, 0.0, 1.0)),
        "status": "OK",
    }


def read_wrf_precip_delta(start_wrfout_path: str | Path, end_wrfout_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return accumulated WRF precipitation delta and its lat/lon grid."""

    def total(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with Dataset(path, "r") as dataset:
            lats = _read_2d(dataset, "XLAT")
            lons = _read_2d(dataset, "XLONG")
            rain = np.zeros_like(_read_2d(dataset, "RAINNC"))
            for name in ("RAINNC", "RAINC"):
                if name in dataset.variables:
                    rain = rain + _read_2d(dataset, name)
            return rain, lats, lons

    start_total, lats, lons = total(start_wrfout_path)
    end_total, end_lats, end_lons = total(end_wrfout_path)
    if start_total.shape != end_total.shape or lats.shape != end_lats.shape or lons.shape != end_lons.shape:
        raise ValueError("start and end wrfout grids do not match")
    return np.maximum(end_total - start_total, 0.0), lats, lons


def station_precip_to_grid(station_precip: pd.DataFrame, lats: np.ndarray, lons: np.ndarray, *, variable: str = "PRECIP") -> np.ndarray:
    """Project station precipitation to nearest model cells for sparse FSS input."""

    stations = _canonical_station_metadata(station_precip)
    values = station_precip[["station_id", variable]].copy() if variable in station_precip.columns else pd.DataFrame(columns=["station_id", variable])
    values["station_id"] = values["station_id"].astype(str)
    merged = stations.merge(values.groupby("station_id", as_index=False)[variable].mean(), on="station_id", how="left")
    grid = np.full(lats.shape, np.nan, dtype=np.float64)
    for row in merged.dropna(subset=[variable]).itertuples(index=False):
        distance2 = (lats - float(getattr(row, "lat"))) ** 2 + (lons - float(getattr(row, "lon"))) ** 2
        j, i = np.unravel_index(int(np.nanargmin(distance2)), lats.shape)
        current = grid[j, i]
        value = float(getattr(row, variable))
        grid[j, i] = value if not np.isfinite(current) else 0.5 * (current + value)
    return grid


def compute_precip_fss_for_wrfouts(
    start_wrfout_path: str | Path,
    end_wrfout_path: str | Path,
    station_precip: pd.DataFrame,
    *,
    threshold_mm: float = DEFAULT_FSS_THRESHOLD_MM,
    window_size: int = DEFAULT_FSS_WINDOW_CELLS,
) -> dict[str, Any]:
    """Compute station-projected precipitation FSS for one WRF accumulation window."""

    forecast, lats, lons = read_wrf_precip_delta(start_wrfout_path, end_wrfout_path)
    observed = station_precip_to_grid(station_precip, lats, lons, variable="PRECIP")
    result = compute_fractions_skill_score(forecast, observed, threshold_mm=threshold_mm, window_size=window_size)
    coverage_fraction = result["observed_finite_cells"] / float(np.prod(forecast.shape))
    result.update(
        {
            "forecast_source_start": str(start_wrfout_path),
            "forecast_source_end": str(end_wrfout_path),
            "observation_source": "AEMET station precipitation projected to nearest WRF grid cells",
            "observed_grid_coverage_fraction": float(coverage_fraction),
            "reliability": "station_sparse" if coverage_fraction < 0.01 else "station_grid",
            "caveat": (
                "FSS is finite but station-projected precipitation is sparse; use as scaffold sanity only."
                if coverage_fraction < 0.01
                else "FSS uses station-projected observations."
            ),
        }
    )
    return result


__all__ = [
    "DEFAULT_AEMET_ROOT",
    "DEFAULT_FSS_THRESHOLD_MM",
    "DEFAULT_FSS_WINDOW_CELLS",
    "DEFAULT_SCORE_VARIABLES",
    "ScoreReport",
    "compute_fractions_skill_score",
    "compute_precip_fss_for_wrfouts",
    "compute_station_scores",
    "interpolate_to_stations",
    "inventory_aemet_observations",
    "load_aemet_observations",
    "read_wrf_precip_delta",
    "station_precip_to_grid",
    "write_json",
]
