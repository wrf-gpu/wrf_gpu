from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

from gpuwrf.validation.data_quality import compute_rmse_against_gen2


def _write_wrfout(path: Path, valid_time: datetime, *, offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("DateStrLen", 19)
        dataset.createDimension("south_north", 3)
        dataset.createDimension("west_east", 4)
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        stamp = valid_time.strftime("%Y-%m-%d_%H:%M:%S")
        times[0, :] = np.asarray(list(stamp), dtype="S1")
        base = np.arange(12, dtype=np.float32).reshape(3, 4) + np.float32(offset)
        for field in ("U10", "V10", "T2", "Q2", "PSFC", "RAINNC"):
            variable = dataset.createVariable(field, "f4", ("Time", "south_north", "west_east"))
            variable[0, :, :] = base + np.float32(len(field))


def _make_run(tmp_path: Path) -> tuple[Path, datetime]:
    start = datetime(2026, 5, 20, 18, tzinfo=timezone.utc)
    run = tmp_path / "20260520_18z_l3_24h_20260521T045847Z"
    run.mkdir()
    for lead in range(2):
        valid = start + timedelta(hours=lead)
        _write_wrfout(run / f"wrfout_d02_{valid:%Y-%m-%d_%H:%M:%S}", valid, offset=float(lead))
    return run, start


def _state(offset: float = 0.0) -> dict[str, np.ndarray]:
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    return {
        "U10": base + 3.0 + np.float32(offset),
        "V10": base + 3.0 + np.float32(offset),
        "T2": base + 2.0 + np.float32(offset),
    }


def test_rmse_adapter_returns_zero_for_identical_file_truth(tmp_path: Path):
    run, start = _make_run(tmp_path)
    truth_file = run / f"wrfout_d02_{start:%Y-%m-%d_%H:%M:%S}"

    result = compute_rmse_against_gen2(_state(), truth_file, start.isoformat())

    assert result["U10"]["rmse"] == 0.0
    assert result["T2"]["error_map"].shape == (3, 4)


def test_rmse_adapter_reports_per_cell_error_map(tmp_path: Path):
    run, start = _make_run(tmp_path)
    truth_file = run / f"wrfout_d02_{start:%Y-%m-%d_%H:%M:%S}"

    result = compute_rmse_against_gen2(_state(offset=2.0), truth_file, start.isoformat(), fields=("U10",))

    assert np.isclose(result["U10"]["rmse"], 2.0)
    assert np.allclose(np.asarray(result["U10"]["error_map"]), 2.0)


def test_rmse_adapter_accepts_run_directory_and_valid_time(tmp_path: Path):
    run, start = _make_run(tmp_path)
    valid = start + timedelta(hours=1)

    result = compute_rmse_against_gen2(_state(), run, valid.isoformat(), fields=("T2",))

    assert np.isclose(result["T2"]["rmse"], 1.0)
    assert result["T2"]["valid_time_utc"] == "2026-05-20T19:00:00+00:00"


def test_rmse_adapter_rejects_missing_forecast_field(tmp_path: Path):
    run, start = _make_run(tmp_path)
    truth_file = run / f"wrfout_d02_{start:%Y-%m-%d_%H:%M:%S}"

    with pytest.raises(KeyError):
        compute_rmse_against_gen2({"U10": np.zeros((3, 4), dtype=np.float32)}, truth_file, start.isoformat(), fields=("V10",))


def test_rmse_adapter_rejects_shape_mismatch(tmp_path: Path):
    run, start = _make_run(tmp_path)
    truth_file = run / f"wrfout_d02_{start:%Y-%m-%d_%H:%M:%S}"

    with pytest.raises(ValueError, match="shape mismatch"):
        compute_rmse_against_gen2({"U10": np.zeros((2, 2), dtype=np.float32)}, truth_file, start.isoformat(), fields=("U10",))


def test_rmse_adapter_rejects_file_valid_time_mismatch(tmp_path: Path):
    run, start = _make_run(tmp_path)
    truth_file = run / f"wrfout_d02_{start:%Y-%m-%d_%H:%M:%S}"

    with pytest.raises(ValueError, match="does not match"):
        compute_rmse_against_gen2(_state(), truth_file, (start + timedelta(hours=1)).isoformat(), fields=("U10",))
