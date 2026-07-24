"""WRF CLWRF run-date greenhouse-gas interpolation for RRTMG.

This is a host-side transcription of ``module_ra_clWRF_support.F``.  WRF's
default ``GHG_INPUT=1`` reads ``run/CAMtr_volume_mixing_ratio`` once, assigns
each annual record to mid-June, and interpolates each gas independently to the
current fractional Julian day.  The returned scalars are static RRTMG column
metadata; no file access or interpolation enters a JAX timestep loop.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

from gpuwrf.config.paths import wrf_run_path


class CLWRFGreenhouseGases(NamedTuple):
    """Run-date trace-gas volume mixing ratios consumed by WRF RRTMG."""

    co2_vmr: float
    n2o_vmr: float
    ch4_vmr: float
    cfc11_vmr: float
    cfc12_vmr: float


class _CLWRFTable(NamedTuple):
    years: np.ndarray
    values: np.ndarray


_GAS_SCALES = (1.0e-6, 1.0e-9, 1.0e-9, 1.0e-12, 1.0e-12)


def _coerce_datetime_utc(value) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, date):
        result = datetime(value.year, value.month, value.day)
    else:
        text = str(value).strip().replace("Z", "+00:00")
        result = datetime.fromisoformat(text)
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _is_leap(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or year % 400 == 0


def _days_in_year(year: int) -> np.float32:
    return np.float32(366.0 if _is_leap(year) else 365.0)


def _mid_june_julian(year: int) -> np.float32:
    """Replay CLWRF's default-REAL ``mondata=6`` date expression."""

    february = 29 if _is_leap(year) else 28
    monday = (0, 31, february, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    return np.float32(
        np.float32(sum(monday[:6]))
        + np.float32(monday[6]) / np.float32(2.0)
        - np.float32(0.5)
    )


@lru_cache(maxsize=8)
def _load_table(resolved_path: str) -> _CLWRFTable:
    path = Path(resolved_path)
    lines = path.read_text(encoding="ascii", errors="strict").splitlines()
    if len(lines) < 4:
        raise ValueError(f"CLWRF gas table has too few rows: {path}")
    years: list[int] = []
    rows: list[tuple[float, ...]] = []
    for line in lines[2:]:
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 6:
            raise ValueError(f"invalid CLWRF gas-table row: {line!r}")
        years.append(int(fields[0]))
        rows.append(tuple(float(value) for value in fields[1:]))
    year_array = np.asarray(years, dtype=np.int32)
    value_array = np.asarray(rows, dtype=np.float64)
    if (
        year_array.ndim != 1
        or value_array.shape != (year_array.size, 5)
        or year_array.size < 2
        or np.any(np.diff(year_array) <= 0)
        or not np.isfinite(value_array).all()
    ):
        raise ValueError(f"invalid CLWRF gas table: {path}")
    year_array.setflags(write=False)
    value_array.setflags(write=False)
    return _CLWRFTable(year_array, value_array)


def _timeline_value(year: int, julian: np.float32, origin_year: int) -> np.float32:
    result = np.float32(julian)
    for current_year in range(origin_year, year):
        result = np.float32(result + _days_in_year(current_year))
    return result


def _co2_valid_bracket(
    years: np.ndarray,
    co2_values: np.ndarray,
    target_year: int,
    target_julian: np.float32,
) -> tuple[int, int]:
    """Return WRF ``valid_years``' CO2-selected interpolation bracket.

    ``read_CAMgases`` calls ``valid_years`` once with ``co2r`` and then reuses
    those two record indices for every gas.  The SSP245 table is complete and
    positive, so the source's sparse-record branches reduce to the nearest
    CO2 dates around the run date (or the first/last two positive records for
    extrapolation).  Selecting the bracket once here is important: choosing a
    separate bracket per gas would not be a faithful transcription if a future
    table contained a missing non-CO2 entry.
    """

    dates = [
        (int(year), float(_mid_june_julian(int(year)))) for year in years
    ]
    target = (int(target_year), float(target_julian))
    valid = [index for index, value in enumerate(co2_values) if value > 0.0]
    if len(valid) < 2:
        raise ValueError("CLWRF interpolation requires two positive gas records")
    lower = [index for index in valid if dates[index] <= target]
    upper = [index for index in valid if dates[index] > target]
    if not lower:
        return valid[0], valid[1]
    if not upper:
        return valid[-2], valid[-1]
    return lower[-1], upper[0]


def _interpolate_one(
    table: _CLWRFTable,
    gas_index: int,
    target_year: int,
    target_julian: np.float32,
    lower: int,
    upper: int,
) -> float:
    lower_year = int(table.years[lower])
    upper_year = int(table.years[upper])
    origin_year = min(target_year, lower_year)
    lower_time = _timeline_value(
        lower_year, _mid_june_julian(lower_year), origin_year
    )
    upper_time = _timeline_value(
        upper_year, _mid_june_julian(upper_year), origin_year
    )
    target_time = _timeline_value(target_year, target_julian, origin_year)
    delta_time = np.float32(upper_time - lower_time)
    if delta_time == np.float32(0.0):
        raise ValueError("CLWRF gas records have an identical source date")
    fact1 = np.float32((upper_time - target_time) / delta_time)
    fact2 = np.float32((target_time - lower_time) / delta_time)
    interpolated = (
        np.float64(table.values[lower, gas_index]) * np.float64(fact1)
        + np.float64(table.values[upper, gas_index]) * np.float64(fact2)
    )
    if interpolated < 0.0:
        raise ValueError("CLWRF interpolation produced a negative gas value")
    if gas_index == 0:
        interpolated = max(interpolated, np.float64(270.0))
    elif gas_index == 1:
        interpolated = max(interpolated, np.float64(270.0))
    elif gas_index == 2:
        interpolated = max(interpolated, np.float64(700.0))
    # WRF source literals 1.e-6/1.e-9/1.e-12 are default REAL and are rounded
    # before the mixed-kind multiplication with REAL(r8) ``interp_gas``.
    scale = np.float64(np.float32(_GAS_SCALES[gas_index]))
    return float(interpolated * scale)


def clwrf_ssp245_gases_for_time(
    time_utc,
    *,
    table_path: str | Path | None = None,
) -> CLWRFGreenhouseGases:
    """Return exact WRF CLWRF SSP245 gases for one host-side run date.

    ``table_path`` defaults to the WRF runtime link
    ``$GPUWRF_WRF_ROOT/run/CAMtr_volume_mixing_ratio``.  It is explicit in
    regressions so the transcription can be tested without global environment
    state.
    """

    value = _coerce_datetime_utc(time_utc)
    julian = np.float32(
        np.float32(value.timetuple().tm_yday)
        + np.float32(
            value.hour * 3600
            + value.minute * 60
            + value.second
            + value.microsecond / 1_000_000.0
        )
        / np.float32(86400.0)
    )
    path = (
        Path(table_path)
        if table_path is not None
        else wrf_run_path("CAMtr_volume_mixing_ratio")
    )
    table = _load_table(str(path.expanduser().resolve(strict=True)))
    lower, upper = _co2_valid_bracket(
        table.years,
        table.values[:, 0],
        value.year,
        julian,
    )
    gases = tuple(
        _interpolate_one(
            table,
            index,
            value.year,
            julian,
            lower,
            upper,
        )
        for index in range(5)
    )
    return CLWRFGreenhouseGases(*gases)
