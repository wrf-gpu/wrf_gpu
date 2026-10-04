"""Regression tests for the ``dt`` / forecast-hour non-divisible boundary.

The v0.25 M0 CPU baseline question is a 6.8% gap. ``dt=54`` does not divide
3600, so a "one hour" WRF run actually advances 3618 s and normalising it by
3600 inflates the rate by 0.5%. These tests pin that the parser (a) derives the
actual advance from stamps, (b) reports both normalisations, (c) refuses to
normalise a series that contradicts the caller's dt, and (d) exposes the
66/67-step sawtooth instead of hiding it inside an hourly sum.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "v025"))

from wrf_timing import (  # noqa: E402
    WRF_TIME_FMT,
    DomainSeries,
    TimingConsistencyError,
    block_profile,
    hourly_profile,
    parse_rsl,
    reduce_series,
    window_geometry,
)

DT = 54.0
T0 = datetime(2026, 7, 26, 0, 0, 0)


def make_series(n_steps: int, elapsed: float | list[float] = 1.0, dt: float = DT) -> DomainSeries:
    """A synthetic contiguous domain-1 series stamped exactly like WRF does."""
    values = [elapsed] * n_steps if isinstance(elapsed, (int, float)) else list(elapsed)
    assert len(values) == n_steps
    times = [
        (T0 + timedelta(seconds=(k + 1) * dt)).strftime(WRF_TIME_FMT) for k in range(n_steps)
    ]
    return DomainSeries(domain=1, times=times, elapsed=values)


def render_rsl(series: DomainSeries) -> str:
    return "".join(
        f"Timing for main: time {t} on domain   {series.domain}:   {e:.5f} elapsed seconds\n"
        for t, e in zip(series.times, series.elapsed)
    )


# --- the boundary itself -------------------------------------------------


def test_one_forecast_hour_at_dt54_is_67_steps_and_3618_seconds():
    """The load-bearing fact: 3600/54 is not an integer."""
    assert 3600 % DT != 0
    series = make_series(67)
    geom = window_geometry(series, dt_seconds=DT, scheduled_window_seconds=3600.0)
    assert geom["steps"] == 67
    assert geom["actual_model_advance_seconds"] == pytest.approx(3618.0)
    assert geom["end_time"] == "2026-07-26_01:00:18"
    assert geom["dt_divides_hour"] is False
    assert geom["window_overshoot_seconds"] == pytest.approx(18.0)
    assert geom["window_overshoot_percent"] == pytest.approx(0.5, abs=0.01)


def test_both_normalisations_reported_and_differ_by_the_overshoot():
    """Scheduled-window and actual-advance rates must both be visible."""
    series = make_series(67, elapsed=2.0)  # 134 s of work
    red = reduce_series(series, dt_seconds=DT, scheduled_window_seconds=3600.0)

    actual = red["step_sum_per_fc_hour"]
    scheduled = red["step_sum_per_fc_hour_scheduled_window"]

    assert red["step_sum_seconds"] == pytest.approx(134.0)
    assert actual == pytest.approx(134.0 / (3618.0 / 3600.0))
    assert scheduled == pytest.approx(134.0)
    # The scheduled convention is the flattering one by exactly the overshoot.
    assert scheduled / actual == pytest.approx(3618.0 / 3600.0)
    assert scheduled > actual


def test_canonical_rate_uses_actual_advance_not_the_scheduled_window():
    """Never choose the favourable convention: canonical == actual advance."""
    series = make_series(67, elapsed=2.0)
    with_sched = reduce_series(series, dt_seconds=DT, scheduled_window_seconds=3600.0)
    without_sched = reduce_series(series, dt_seconds=DT)
    assert with_sched["step_sum_per_fc_hour"] == pytest.approx(
        without_sched["step_sum_per_fc_hour"]
    )


def test_divisible_dt_has_zero_overshoot():
    """A dt that divides 3600 must show no correction at all."""
    series = make_series(60, elapsed=1.0, dt=60.0)
    geom = window_geometry(series, dt_seconds=60.0, scheduled_window_seconds=3600.0)
    assert geom["dt_divides_hour"] is True
    assert geom["actual_model_advance_seconds"] == pytest.approx(3600.0)
    assert geom["window_overshoot_percent"] == pytest.approx(0.0)


# --- fail-closed validation ---------------------------------------------


def test_wrong_dt_is_rejected_not_normalised():
    """n*dt must match the stamped span, or refuse to produce a rate."""
    series = make_series(67)  # stamped at dt=54
    with pytest.raises(TimingConsistencyError, match="Refusing to normalise|not a single contiguous"):
        window_geometry(series, dt_seconds=18.0)


def test_non_contiguous_series_is_rejected():
    """A gap means two interleaved runs; a rate over it is meaningless."""
    series = make_series(10)
    series.times[5] = (T0 + timedelta(seconds=99 * DT)).strftime(WRF_TIME_FMT)
    with pytest.raises(TimingConsistencyError, match="not a single contiguous run"):
        window_geometry(series, dt_seconds=DT)


def test_scheduled_window_off_by_more_than_one_step_is_rejected():
    """Describing a 2-hour log as a 1-hour window must fail, not rescale."""
    series = make_series(134)  # ~2 forecast hours
    with pytest.raises(TimingConsistencyError, match="not within one step"):
        window_geometry(series, dt_seconds=DT, scheduled_window_seconds=3600.0)


def test_start_time_drift_is_rejected():
    series = make_series(67)
    with pytest.raises(TimingConsistencyError, match="drift"):
        window_geometry(
            series,
            dt_seconds=DT,
            start_time=datetime(2026, 7, 26, 6, 0, 0),
        )


def test_empty_series_is_rejected():
    with pytest.raises(TimingConsistencyError, match="empty step series"):
        window_geometry(DomainSeries(domain=1), dt_seconds=DT)


# --- the sawtooth --------------------------------------------------------


def test_hourly_profile_exposes_the_66_67_sawtooth():
    """Hours must carry their step count; equal-cost steps must not look unequal."""
    series = make_series(200, elapsed=1.0)  # exactly 3 forecast hours
    profile = hourly_profile(series, dt_seconds=DT)
    # Hour 0 holds only the 66 steps that END at or before 3600 s; step 67 ends
    # at 3618 s and is credited to hour 1.
    assert [h["steps"] for h in profile] == [66, 67, 67]
    # Identical per-step cost, yet the hourly sums differ by 1.5% purely from
    # step count -- which is exactly why the count is reported alongside.
    assert [h["sum_seconds"] for h in profile] == [66.0, 67.0, 67.0]
    assert all(h["mean_per_step"] == pytest.approx(1.0) for h in profile)


def test_two_legitimate_one_hour_conventions_differ_by_one_step():
    """The trap: "1 forecast hour" names two different windows at dt=54.

    A ``run_hours=1`` WRF job must reach the 01:00 boundary, so it takes 67
    steps and advances 3618 s. The *first whole hour of model time* holds only
    the 66 steps ending at or before 3600 s. The two differ by 1.5% in work and
    0.5% in window -- both real, neither wrong, and mixing them silently is how
    a 6.8% baseline gap gets manufactured.
    """
    run_of_one_hour = make_series(67, elapsed=1.0)
    geom = window_geometry(run_of_one_hour, dt_seconds=DT, scheduled_window_seconds=3600.0)
    assert geom["steps"] == 67
    assert geom["actual_model_advance_seconds"] == pytest.approx(3618.0)

    first_whole_hour = hourly_profile(make_series(200, elapsed=1.0), dt_seconds=DT)[0]
    assert first_whole_hour["steps"] == 66

    # Same per-step cost, ~1.5% different total, from convention alone.
    assert run_of_one_hour.elapsed[0] == pytest.approx(first_whole_hour["mean_per_step"])
    assert 67 / 66 == pytest.approx(1.01515, abs=1e-5)


def test_block_profile_is_exact_and_bias_free():
    """200 steps == 3 forecast hours exactly, so blocks are comparable."""
    series = make_series(400, elapsed=1.0)
    blocks = block_profile(series, dt_seconds=DT)
    assert blocks["steps_per_block"] == 200
    assert blocks["hours_per_block"] == pytest.approx(3.0)
    assert len(blocks["blocks"]) == 2
    rates = [b["per_fc_hour"] for b in blocks["blocks"]]
    assert rates[0] == pytest.approx(rates[1])
    assert blocks["per_fc_hour_spread"]["cv"] == pytest.approx(0.0)


def test_162_hour_production_window_is_exactly_divisible():
    """The retained production run's full window has no overshoot at all."""
    series = make_series(10800, elapsed=1.0)
    geom = window_geometry(series, dt_seconds=DT, scheduled_window_seconds=162 * 3600.0)
    assert geom["actual_model_advance_seconds"] == pytest.approx(162 * 3600.0)
    assert geom["window_overshoot_seconds"] == pytest.approx(0.0)


# --- warm-up exclusion ---------------------------------------------------


def test_warmup_exclusion_rescales_onto_the_full_window():
    """Dropping the table-build step must not also drop a step's worth of window."""
    values = [100.0] + [2.0] * 66  # step 1 carries a one-time cost
    series = make_series(67, elapsed=values)
    red = reduce_series(series, dt_seconds=DT, warmup_steps=1)
    assert red["warmup_seconds"] == pytest.approx(100.0)
    # 66 warm steps at 2.0 rescaled onto all 67 steps.
    assert red["step_sum_warm_seconds"] == pytest.approx(2.0 * 67)
    assert red["step_sum_seconds"] == pytest.approx(100.0 + 132.0)


# --- parser round-trip ---------------------------------------------------


def test_parse_rsl_round_trip(tmp_path):
    series = make_series(5, elapsed=[1.0, 2.0, 3.0, 4.0, 5.0])
    path = tmp_path / "rsl.out.0000"
    path.write_text(render_rsl(series))
    parsed, io = parse_rsl(path)
    assert list(parsed) == [1]
    assert parsed[1].elapsed == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert parsed[1].times == series.times
    assert io == []
