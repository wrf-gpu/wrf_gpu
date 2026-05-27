from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from gpuwrf.validation.forecast_vs_obs import (
    compute_fractions_skill_score,
    compute_station_scores,
    interpolate_to_stations,
)


def _write_synthetic_wrfout(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("DateStrLen", 19)
        dataset.createDimension("south_north", 3)
        dataset.createDimension("west_east", 3)
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0, :] = np.asarray(list("2026-05-25_00:00:00"), dtype="S1")

        lat = np.repeat(np.arange(3, dtype=np.float64)[:, None], 3, axis=1)
        lon = np.repeat(np.arange(3, dtype=np.float64)[None, :], 3, axis=0)
        t2 = lon + 10.0 * lat
        for name, values in {
            "XLAT": lat,
            "XLONG": lon,
            "T2": t2,
            "U10": np.full((3, 3), 3.0),
            "V10": np.full((3, 3), 4.0),
            "RAINNC": np.zeros((3, 3)),
            "RAINC": np.zeros((3, 3)),
        }.items():
            variable = dataset.createVariable(name, "f8", ("Time", "south_north", "west_east"))
            variable[0, :, :] = values


def test_interpolate_to_stations_bilinear_known_answer(tmp_path: Path):
    wrfout = tmp_path / "wrfout_d02_2026-05-25_00:00:00"
    _write_synthetic_wrfout(wrfout)
    stations = pd.DataFrame(
        {
            "station_id": ["A"],
            "lat": [0.25],
            "lon": [0.25],
            "elev_m": [10.0],
        }
    )

    result = interpolate_to_stations(
        wrfout,
        stations,
        variables=("T2", "WIND10"),
        valid_time=datetime(2026, 5, 25, tzinfo=timezone.utc),
    )

    assert len(result) == 1
    assert bool(result.loc[0, "inside_domain"]) is True
    assert np.isclose(result.loc[0, "T2"], 2.75)
    assert np.isclose(result.loc[0, "WIND10"], 5.0)


def test_compute_station_scores_known_bias_rmse_mae():
    times = pd.to_datetime(["2026-05-25T00:00:00Z", "2026-05-25T01:00:00Z"])
    forecast = pd.DataFrame({"station_id": ["A", "A"], "time": times, "T2": [2.0, 4.0]})
    observations = pd.DataFrame({"station_id": ["A", "A"], "time": times, "T2": [1.0, 2.0]})

    report = compute_station_scores(forecast, observations, variables=("T2",))

    scores = report.to_dict()["scores"]["T2"]
    assert report.status == "OK"
    assert scores["sample_count"] == 2
    assert scores["bias"] == 1.5
    assert scores["rmse"] == np.sqrt(2.5)
    assert scores["mae"] == 1.5


def test_fractions_skill_score_identical_pattern_is_one():
    precip = np.zeros((7, 7), dtype=np.float64)
    precip[2:5, 2:5] = 3.0

    result = compute_fractions_skill_score(precip, precip, threshold_mm=1.0, window_size=3)

    assert result["status"] == "OK"
    assert result["fss"] == 1.0


def test_fractions_skill_score_shifted_pattern_is_finite():
    forecast = np.zeros((7, 7), dtype=np.float64)
    observed = np.zeros((7, 7), dtype=np.float64)
    forecast[2:5, 2:5] = 3.0
    observed[3:6, 3:6] = 3.0

    result = compute_fractions_skill_score(forecast, observed, threshold_mm=1.0, window_size=3)

    assert 0.0 <= result["fss"] < 1.0


def test_interpolate_to_stations_handles_missing_stations(tmp_path: Path):
    wrfout = tmp_path / "wrfout_d02_2026-05-25_00:00:00"
    _write_synthetic_wrfout(wrfout)

    result = interpolate_to_stations(
        wrfout,
        pd.DataFrame(columns=["station_id", "lat", "lon", "elev_m"]),
        variables=("T2",),
        valid_time="2026-05-25T00:00:00Z",
    )

    assert result.empty
    assert list(result.columns) == ["station_id", "time", "lat", "lon", "elev_m", "inside_domain", "T2"]


def test_compute_station_scores_all_nan_obs_has_no_valid_pairs():
    time = pd.to_datetime(["2026-05-25T00:00:00Z"])
    forecast = pd.DataFrame({"station_id": ["A"], "time": time, "T2": [280.0]})
    observations = pd.DataFrame({"station_id": ["A"], "time": time, "T2": [np.nan]})

    report = compute_station_scores(forecast, observations, variables=("T2",))

    assert report.status == "NO_VALID_PAIRS"
    assert report.scores["T2"]["sample_count"] == 0


def test_compute_station_scores_no_temporal_overlap():
    forecast = pd.DataFrame({"station_id": ["A"], "time": pd.to_datetime(["2026-05-25T00:00:00Z"]), "T2": [280.0]})
    observations = pd.DataFrame({"station_id": ["A"], "time": pd.to_datetime(["2026-05-26T00:00:00Z"]), "T2": [279.0]})

    report = compute_station_scores(forecast, observations, variables=("T2",))

    assert report.status == "NO_OVERLAP"
    assert report.joined_rows == 0
    assert report.scores["T2"]["sample_count"] == 0
