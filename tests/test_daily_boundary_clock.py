"""v0.14 Switzerland d01 LBC clock fix: daily-pipeline boundary windowing.

Root cause under test: ``_run_forecast_sequence`` calls the operational
forecast entry once per hour, and that entry restarts its step clock at 1 on
every call. With the full-time-axis ``*_bdy`` leaves in the State, the in-scan
``interpolate_boundary_leaf`` walk re-forced the lateral boundary toward leaf
level 1 each hour, pinning the spec zone at the hour-1 value for the whole run
(Switzerland d01 72h: GPU boundary ring == CPU truth h01 bit-exact at every
lead through h72; PSFC bias +2380 Pa by h72).

The fix re-anchors the leaves before every segment to a 2-level window holding
the exact boundary values at the segment's GLOBAL start / start+cadence, so the
restarted in-call walk reproduces the true global-time forcing.

Pure CPU; the forecast is a stub that records the boundary window it receives.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gpuwrf.integration.daily_pipeline import (
    DailyCase,
    DailyPipelineConfig,
    _boundary_leaf_value_at,
    _capture_boundary_leaves,
    _rewindow_boundary_leaves,
    _run_forecast_sequence,
)

from test_auxhist_stream import _synthetic_case, _synthetic_state


# --------------------------------------------------------------------------- #
# Host interpolation mirrors interpolate_boundary_leaf exactly.                #
# --------------------------------------------------------------------------- #
def test_host_leaf_value_matches_device_interpolation() -> None:
    from gpuwrf.coupling.boundary_apply import interpolate_boundary_leaf

    rng = np.random.default_rng(20260611)
    leaf = rng.normal(size=(7, 4, 5, 3, 11)).astype(np.float64)
    for cadence in (3600.0, 10800.0, 21600.0):
        for t in (0.0, 600.0, 3600.0, 5400.0, 3 * cadence, 6.5 * cadence, 50 * cadence):
            host = _boundary_leaf_value_at(leaf, t, cadence)
            device = np.asarray(interpolate_boundary_leaf(leaf, t, cadence))
            np.testing.assert_allclose(host, device, rtol=0.0, atol=0.0)


def test_integer_record_times_return_exact_levels() -> None:
    leaf = np.arange(5, dtype=np.float64)[:, None, None, None, None] * np.ones(
        (5, 4, 2, 3, 6), dtype=np.float64
    )
    for k in range(5):
        value = _boundary_leaf_value_at(leaf, k * 3600.0, 3600.0)
        assert np.array_equal(value, leaf[k])
    # Clamped frozen past the last record (existing semantics preserved).
    assert np.array_equal(_boundary_leaf_value_at(leaf, 99 * 3600.0, 3600.0), leaf[-1])


# --------------------------------------------------------------------------- #
# Windowing: replay (hourly records) and native-init (6-hourly records).       #
# --------------------------------------------------------------------------- #
def _ramp_leaves(ntimes: int) -> dict[str, np.ndarray]:
    ramp = np.arange(ntimes, dtype=np.float64)[:, None, None, None, None]
    return {
        "u_bdy": ramp * np.ones((ntimes, 4, 5, 3, 8), dtype=np.float64),
        "mu_bdy": 100.0 * ramp * np.ones((ntimes, 4, 5, 1, 8), dtype=np.float64),
    }


class _ReplaceableState(SimpleNamespace):
    def replace(self, **kwargs):
        new = copy.copy(self)
        for key, value in kwargs.items():
            setattr(new, key, value)
        return new


def test_rewindow_walks_hourly_records_one_level_per_hour() -> None:
    leaves = _ramp_leaves(73)
    state = _ReplaceableState()
    for hour in (1, 2, 24, 72):
        t0 = (hour - 1) * 3600.0
        windowed = _rewindow_boundary_leaves(
            state, leaves, segment_start_s=t0, record_cadence_s=3600.0, window_s=3600.0
        )
        assert windowed.u_bdy.shape[0] == 2
        # Hour h must be forced from record h-1 toward record h -- the value the
        # broken loop only ever produced for h == 1.
        assert float(windowed.u_bdy[0].mean()) == float(hour - 1)
        assert float(windowed.u_bdy[1].mean()) == float(hour)
        assert float(windowed.mu_bdy[1].mean()) == 100.0 * float(hour)


def test_rewindow_native_init_interval_consumes_records_at_true_cadence() -> None:
    # 12 records at 21600 s (6-hourly wrfbdy) + synthesized terminal -> 13 levels.
    leaves = _ramp_leaves(13)
    state = _ReplaceableState()
    # Hour 9 spans [8h, 9h] -> record index 8/6 = 1.333.. to 1.5.
    windowed = _rewindow_boundary_leaves(
        state, leaves, segment_start_s=8 * 3600.0, record_cadence_s=21600.0, window_s=3600.0
    )
    np.testing.assert_allclose(float(windowed.u_bdy[0].mean()), 8.0 / 6.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(float(windowed.u_bdy[1].mean()), 9.0 / 6.0, rtol=0, atol=1e-12)


def test_capture_skips_cases_without_lateral_boundary() -> None:
    # Synthetic dict-namelist case (no run_boundary attr) -> windowing disabled.
    state = _ReplaceableState(u_bdy=np.zeros((3, 4, 5, 2, 6)))
    assert _capture_boundary_leaves(state, {"title": "x"}) == {}
    # run_boundary False -> disabled.
    namelist = SimpleNamespace(run_boundary=False)
    assert _capture_boundary_leaves(state, namelist) == {}
    # No replace method -> disabled.
    plain = SimpleNamespace(u_bdy=np.zeros((3, 4, 5, 2, 6)))
    assert _capture_boundary_leaves(plain, SimpleNamespace(run_boundary=True)) == {}
    # Active boundary + pytree-like state -> captured.
    active = _capture_boundary_leaves(state, SimpleNamespace(run_boundary=True))
    assert set(active) == {"u_bdy"}


# --------------------------------------------------------------------------- #
# Loop-level regression: the hourly sequence must advance the boundary clock.  #
# --------------------------------------------------------------------------- #
class _AttrDict(dict):
    """Dict namelist (what the wrfout writer expects) carrying attributes."""


def _boundary_case(run_dir: Path, hours: int) -> DailyCase:
    base = _synthetic_case(run_dir)
    ntimes = hours + 1
    state = _ReplaceableState(**vars(_synthetic_state()))
    for name, leaf in _ramp_leaves(ntimes).items():
        setattr(state, name, leaf)
    namelist = _AttrDict(base.namelist)
    namelist.run_boundary = True
    namelist.boundary_config = SimpleNamespace(update_cadence_s=3600.0)
    return DailyCase(
        state=state,
        grid=base.grid,
        namelist=namelist,
        run_start=datetime(2026, 5, 21, 18, tzinfo=timezone.utc),
        metadata={"run_id": "lbc-clock-proof", "run_dir": str(run_dir), "source": "synthetic"},
    )


def test_forecast_sequence_advances_boundary_window_every_hour(tmp_path: Path) -> None:
    hours = 4
    run_dir = tmp_path / "run_lbc"
    run_dir.mkdir(parents=True, exist_ok=True)
    received: list[np.ndarray] = []

    def _recording_forecast_fn(state, namelist, fhours):
        del namelist, fhours
        received.append(np.asarray(state.u_bdy))
        return state

    config = DailyPipelineConfig(
        run_id="lbc-clock-proof",
        hours=hours,
        output_dir=tmp_path / "out_lbc",
        proof_dir=tmp_path / "proof_lbc",
        score=False,
        refresh_land_state_hourly=False,
    )
    result = _run_forecast_sequence(
        config,
        output_dir=config.output_dir,
        forecast_fn=_recording_forecast_fn,
        case_builder=lambda cfg: (_boundary_case(run_dir, hours), run_dir),
    )
    assert result.status == "PASS"
    assert len(received) == hours
    for hour, window in enumerate(received, start=1):
        # The pre-fix loop handed the FULL (hours+1)-level leaf to every call,
        # whose restarted in-call walk froze the boundary at level 1. The fixed
        # loop must hand a 2-level window anchored at the hour's global start.
        assert window.shape[0] == 2
        assert float(window[0].mean()) == float(hour - 1)
        assert float(window[1].mean()) == float(hour)
