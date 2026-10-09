"""MO20 critic probes (copied into the private snapshot's tests/v025/b_noahmp; removed after): exact year boundary and a
leap->common YEARLEN wrap through the REAL step seam, and the output-time step-back, against the host WRF _wrf_julian."""
from datetime import datetime, timedelta, timezone

import pytest

from gpuwrf.physics.wrf_clwrf_ghg import _wrf_julian
from gpuwrf.runtime import operational_mode as op
from test_noahmp_julian_advance import _clock_base, _step_clock


@pytest.mark.parametrize("start,dt,step", [
    (datetime(2026, 12, 31, 18, tzinfo=timezone.utc), 18.0, 1201),   # step START exactly 2027-01-01 00:00:00
    (datetime(2028, 12, 31, 18, tzinfo=timezone.utc), 18.0, 2001),   # leap 2028 -> 2029: YEARLEN 366 -> 365
    (datetime(2028, 12, 31, 18, tzinfo=timezone.utc), 18.0, 1200),   # last step of 2028: still 366, julian 365.99...
])
def test_mo20_year_edges(monkeypatch, start, dt, step):
    monkeypatch.setenv("GPUWRF_NOAHMP_JULIAN_ADVANCE", "1")
    clock = _step_clock(monkeypatch, start, dt, step)
    now = start + timedelta(seconds=(step - 1) * dt)
    assert float(clock.julian) == float(_wrf_julian(now)), (now, float(clock.julian), float(_wrf_julian(now)))
    leap = now.year % 4 == 0 and (now.year % 100 != 0 or now.year % 400 == 0)
    assert float(clock.yearlen) == (366.0 if leap else 365.0)


def test_mo20_output_time_steps_back_one_dt(monkeypatch):
    monkeypatch.setenv("GPUWRF_NOAHMP_JULIAN_ADVANCE", "1")
    start = datetime(2026, 5, 3, 0, tzinfo=timezone.utc)
    base = _clock_base(start)
    nml = type("N", (), {"dt_s": 6.0, "noahmp_julian": -1.0, "noahmp_yearlen": -1.0})()
    lead = 3 * 3600.0
    clock = op._noahmp_clock(nml, base, lead, output_time=True)
    assert float(clock.julian) == float(_wrf_julian(start + timedelta(seconds=lead - 6.0)))
    assert float(clock.julian) != float(_wrf_julian(start + timedelta(seconds=lead)))
