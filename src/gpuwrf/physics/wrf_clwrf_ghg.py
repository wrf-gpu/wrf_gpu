"""WRF CLWRF run-date greenhouse-gas interpolation for RRTMG.

This is a host-side transcription of ``module_ra_clWRF_support.F``.  WRF's
default ``GHG_INPUT=1`` reads ``run/CAMtr_volume_mixing_ratio`` once, assigns
each annual record to mid-June, and interpolates each gas independently to the
current fractional Julian day at every radiation call.  All dates use WRF's
zero-based ``grid%julian`` axis (0.0 at Jan 1 00Z): the REAL record date
(165.5, leap 166.5) selects the bracket and its INTEGER truncation is the
interpolation knot (v0.25 S3 calendar amendment; verified bitwise against the
compiled WRF routine).

``clwrf_ssp245_gases_for_time`` evaluates one host-side date.  Compiled
programs use :func:`clwrf_gas_clock` (host: reads the table, selects brackets)
plus :func:`clwrf_gases_at_lead` (traceable: evaluates the same interpolation at
``init + lead``), so no file access or date constant enters a JAX program.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp
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


def _interpolation_knot(year: int) -> np.float32:
    """Record date used by ``interpolate_CAMgases``: WRF passes ``juldata`` into
    INTEGER ``njulm``/``njulp`` (truncation), so 165.5 -> 165 (166.5 -> 166)."""

    return np.float32(int(_mid_june_julian(year)))


def _wrf_julian(value: datetime) -> np.float32:
    """WRF ``grid%julian`` (0.0 at Jan 1 00Z) for a UTC time.

    ``ESMF_Time.F90`` forms ``dayOfYear_r8 = seconds/86400 + 1`` and
    ``module_domain.F`` stores ``REAL(dayOfYear_r8) - 1.0``; RRTMG passes it to
    ``read_CAMgases`` unchanged."""

    seconds = (value - datetime(value.year, 1, 1, tzinfo=timezone.utc)).total_seconds()
    day_of_year_r8 = np.float64(seconds) / np.float64(86400.0) + np.float64(1.0)
    return np.float32(np.float32(day_of_year_r8) - np.float32(1.0))


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
        lower_year, _interpolation_knot(lower_year), origin_year
    )
    upper_time = _timeline_value(
        upper_year, _interpolation_knot(upper_year), origin_year
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
    julian = _wrf_julian(value)
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


# --------------------------------------------------------------------------- #
# Traced valid-time evaluation (v0.25 S3).
#
# A compiled forecast step must take its gases from OPERANDS: with a date-blind
# jit/AOT key, any gas value baked into the program at trace time is silently
# reused for every later date that hits the same executable.
# --------------------------------------------------------------------------- #
_GAS_CLOCK_YEARS = 2
# ``read_CAMgases`` floors CO2/N2O/CH4; the CFCs are not floored.
_GAS_FLOORS = (270.0, 270.0, 700.0, -np.inf, -np.inf)
# Default-REAL source literals, rounded before the REAL(r8) product (host rule).
_GAS_SCALES_R8 = tuple(float(np.float32(scale)) for scale in _GAS_SCALES)


class CLWRFGasClock(NamedTuple):
    """Per-run CLWRF anchor that lets a traced program evaluate its own gases.

    Built once on the host by :func:`clwrf_gas_clock` and passed into compiled
    programs as ordinary array leaves.  It covers every valid time in the init
    year and the following year: year slot ``k`` stores both CO2-selected
    brackets (before / from that year's mid-June record) as
    :func:`clwrf_ssp245_gases_for_time` selects them.
    """

    base_seconds: Any  # f64[]: init time, seconds after Jan 1 00Z of the init year
    year_start_seconds: Any  # f64[K+1]: Jan 1 00Z of init year + k
    knot_julian: Any  # f32[K]: mid-June record date of year k
    lower_time: Any  # f32[K, 2]: bracket record dates on CLWRF's min-year axis
    upper_time: Any  # f32[K, 2]
    target_offset: Any  # f32[K, 2]: days CLWRF adds to the model julian on that axis
    lower_gas: Any  # f64[K, 2, 5]: bracket table records, file units
    upper_gas: Any  # f64[K, 2, 5]


def clwrf_gas_clock(
    time_utc,
    *,
    table_path: str | Path | None = None,
) -> CLWRFGasClock:
    """Build the host-side :class:`CLWRFGasClock` for a run initialised at ``time_utc``.

    Only interpolating brackets are accepted, so the traced blend never
    extrapolates past the table (the host function's negative-value error
    cannot arise in a compiled program)."""

    value = _coerce_datetime_utc(time_utc)
    path = (
        Path(table_path)
        if table_path is not None
        else wrf_run_path("CAMtr_volume_mixing_ratio")
    )
    table = _load_table(str(path.expanduser().resolve(strict=True)))
    year0 = value.year
    days = [int(_days_in_year(year0 + slot)) for slot in range(_GAS_CLOCK_YEARS)]
    record_dates = [(int(year), float(_mid_june_julian(int(year)))) for year in table.years]
    shape = (_GAS_CLOCK_YEARS, 2)
    knot_julian = np.empty(_GAS_CLOCK_YEARS, dtype=np.float32)
    lower_time = np.empty(shape, dtype=np.float32)
    upper_time = np.empty(shape, dtype=np.float32)
    target_offset = np.empty(shape, dtype=np.float32)
    lower_gas = np.empty(shape + (5,), dtype=np.float64)
    upper_gas = np.empty(shape + (5,), dtype=np.float64)
    for slot in range(_GAS_CLOCK_YEARS):
        year = year0 + slot
        knot_julian[slot] = _mid_june_julian(year)
        # Julian 0.0 (Jan 1 00Z) precedes the record; the REAL record date itself
        # already selects the later bracket (``_co2_valid_bracket`` uses <=).
        for side, probe in enumerate((np.float32(0.0), knot_julian[slot])):
            lower, upper = _co2_valid_bracket(
                table.years, table.values[:, 0], year, probe
            )
            if not record_dates[lower] <= (year, float(probe)) < record_dates[upper]:
                raise ValueError(
                    f"CLWRF gas clock: {year} needs extrapolation beyond the gas table"
                )
            origin = min(year, int(table.years[lower]))
            lower_time[slot, side] = _timeline_value(
                int(table.years[lower]), _interpolation_knot(int(table.years[lower])), origin
            )
            upper_time[slot, side] = _timeline_value(
                int(table.years[upper]), _interpolation_knot(int(table.years[upper])), origin
            )
            target_offset[slot, side] = _timeline_value(year, np.float32(0.0), origin)
            lower_gas[slot, side] = table.values[lower]
            upper_gas[slot, side] = table.values[upper]
    if (lower_gas < 0.0).any() or (upper_gas < 0.0).any():
        raise ValueError("CLWRF gas clock brackets a missing (negative) gas record")
    second_of_day = (
        value.hour * 3600
        + value.minute * 60
        + value.second
        + value.microsecond / 1_000_000.0
    )
    return CLWRFGasClock(
        base_seconds=np.float64((value.timetuple().tm_yday - 1) * 86400 + second_of_day),
        year_start_seconds=np.cumsum([0, *days]).astype(np.float64) * 86400.0,
        knot_julian=knot_julian,
        lower_time=lower_time,
        upper_time=upper_time,
        target_offset=target_offset,
        lower_gas=lower_gas,
        upper_gas=upper_gas,
    )


def _weak_float64(value):
    """Cast to a weakly typed float64, the promotion class of a Python float.

    The traced gases replace host ``float`` scalars and must promote exactly
    like them (e.g. stay fp32 against fp32 column amounts).  JAX has no public
    weak-type cast; the S3 regression pins this internal one."""

    from jax._src.lax import lax as lax_internal  # noqa: PLC0415

    return lax_internal._convert_element_type(
        value, np.dtype(np.float64), weak_type=True
    )


def clwrf_gases_at_lead(clock: CLWRFGasClock, lead_seconds) -> CLWRFGreenhouseGases:
    """Traceable CLWRF gases at valid time ``init + lead_seconds``.

    ``clock`` leaves and ``lead_seconds`` may be tracers; no date or traced value
    becomes a Python number, so a compiled program is date-independent and reads
    its valid-time gases from its operands.  The fractional Julian day and the
    record weights use WRF's default-REAL (fp32) arithmetic like the host
    transcription; XLA may round a few of these steps differently (FMA, divide
    by a constant), a sub-1e-9 relative gas difference.  Valid times outside the
    clock's two-year window give NaN gases, which radiation finite guards reject.
    """

    f32, f64 = jnp.float32, jnp.float64
    starts = jnp.asarray(clock.year_start_seconds, dtype=f64)
    seconds = jnp.asarray(clock.base_seconds, dtype=f64) + jnp.asarray(
        lead_seconds, dtype=f64
    )
    slot = jnp.sum(seconds >= starts[1:-1]).astype(jnp.int32)
    # WRF grid%julian: REAL(seconds/86400 + 1) - 1.0, zero-based (``_wrf_julian``).
    julian = (1.0 + (seconds - starts[slot]) / 86400.0).astype(f32) - jnp.float32(1.0)
    side = (julian >= jnp.asarray(clock.knot_julian, dtype=f32)[slot]).astype(jnp.int32)
    lower_time = jnp.asarray(clock.lower_time, dtype=f32)[slot, side]
    upper_time = jnp.asarray(clock.upper_time, dtype=f32)[slot, side]
    target = julian + jnp.asarray(clock.target_offset, dtype=f32)[slot, side]
    delta = upper_time - lower_time
    fact1 = ((upper_time - target) / delta).astype(f64)
    fact2 = ((target - lower_time) / delta).astype(f64)
    blend = (
        jnp.asarray(clock.lower_gas, dtype=f64)[slot, side] * fact1
        + jnp.asarray(clock.upper_gas, dtype=f64)[slot, side] * fact2
    )
    gases = jnp.maximum(blend, jnp.asarray(_GAS_FLOORS, dtype=f64)) * jnp.asarray(
        _GAS_SCALES_R8, dtype=f64
    )
    in_window = (seconds >= starts[0]) & (seconds < starts[-1])
    gases = jnp.where(in_window, gases, jnp.nan)
    return CLWRFGreenhouseGases(*(_weak_float64(gases[index]) for index in range(5)))
