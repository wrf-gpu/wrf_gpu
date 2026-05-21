from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

from gpuwrf.io.gen2_wrfout_loader import Gen2WrfoutLoader, normalize_valid_time, read_wrfout_file


SURFACE_FIELDS = ("U10", "V10", "T2", "Q2", "PSFC", "RAINNC")
BOUNDARY_FIELDS = ("U", "V", "T", "QVAPOR", "PH")


def _write_wrfout(path: Path, valid_time: datetime, *, base: float = 0.0, omit: set[str] | None = None) -> None:
    omit = omit or set()
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("DateStrLen", 19)
        dataset.createDimension("south_north", 3)
        dataset.createDimension("west_east", 4)
        dataset.createDimension("bottom_top", 2)
        dataset.DX = 3000.0
        dataset.DY = 3000.0
        dataset.MAP_PROJ = 1
        dataset.CEN_LAT = 28.0
        dataset.CEN_LON = -16.0
        dataset.TRUELAT1 = 28.0
        dataset.TRUELAT2 = 28.0
        dataset.STAND_LON = -16.0
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        stamp = valid_time.strftime("%Y-%m-%d_%H:%M:%S")
        times[0, :] = np.asarray(list(stamp), dtype="S1")
        surface = np.arange(12, dtype=np.float32).reshape(3, 4) + np.float32(base)
        for field in SURFACE_FIELDS:
            if field in omit:
                continue
            variable = dataset.createVariable(field, "f4", ("Time", "south_north", "west_east"))
            variable[0, :, :] = surface + np.float32(len(field))
        volume = np.arange(24, dtype=np.float32).reshape(2, 3, 4) + np.float32(base)
        for field in BOUNDARY_FIELDS:
            if field in omit:
                continue
            variable = dataset.createVariable(field, "f4", ("Time", "bottom_top", "south_north", "west_east"))
            variable[0, :, :, :] = volume + np.float32(len(field))


def _make_run(tmp_path: Path, *, hours: int = 2, omit: set[str] | None = None) -> tuple[Path, datetime]:
    start = datetime(2026, 5, 20, 18, tzinfo=timezone.utc)
    run = tmp_path / "20260520_18z_l3_24h_20260521T045847Z"
    run.mkdir()
    (run / "wrfbdy_d01").write_text("", encoding="utf-8")
    for lead in range(hours + 1):
        valid = start + timedelta(hours=lead)
        _write_wrfout(run / f"wrfout_d02_{valid:%Y-%m-%d_%H:%M:%S}", valid, base=float(lead), omit=omit)
    return run, start


def test_loader_selects_valid_time_and_round_trips_field_values(tmp_path: Path):
    run, start = _make_run(tmp_path)

    payload = Gen2WrfoutLoader(run, start + timedelta(hours=1)).load(fields=("T2",))

    assert payload["valid_time_utc"] == "2026-05-20T19:00:00+00:00"
    assert payload["fields"]["T2"].shape == (3, 4)
    assert np.isclose(payload["fields"]["T2"][0, 0], 3.0)


def test_loader_time_axis_is_filename_ordered(tmp_path: Path):
    run, start = _make_run(tmp_path, hours=3)

    times = Gen2WrfoutLoader(run).time_axis

    assert times == [start + timedelta(hours=lead) for lead in range(4)]


def test_loader_can_return_jax_arrays_at_consumer_boundary(tmp_path: Path):
    run, _ = _make_run(tmp_path)

    payload = Gen2WrfoutLoader(run).load(fields=("U10",), as_jax=True)

    assert payload["fields"]["U10"].shape == (3, 4)
    assert type(payload["fields"]["U10"]).__module__.startswith("jax")


def test_missing_field_raises_key_error(tmp_path: Path):
    run, _ = _make_run(tmp_path, omit={"Q2"})

    with pytest.raises(KeyError, match="Q2"):
        Gen2WrfoutLoader(run).load(fields=("Q2",))


def test_iter_chunks_loads_one_file_at_a_time(tmp_path: Path):
    run, _ = _make_run(tmp_path, hours=2)

    chunks = list(Gen2WrfoutLoader(run).iter_chunks(fields=("RAINNC",)))

    assert len(chunks) == 3
    assert [chunk["valid_time_utc"] for chunk in chunks] == [
        "2026-05-20T18:00:00+00:00",
        "2026-05-20T19:00:00+00:00",
        "2026-05-20T20:00:00+00:00",
    ]


def test_read_wrfout_file_uses_times_variable_when_name_is_not_parseable(tmp_path: Path):
    valid = datetime(2026, 5, 20, 18, tzinfo=timezone.utc)
    path = tmp_path / "synthetic_wrfout.nc"
    _write_wrfout(path, valid)

    payload = read_wrfout_file(path, fields=("PSFC",))

    assert payload["valid_time"][0] == np.datetime64("2026-05-20T18:00:00")
    assert normalize_valid_time(payload["valid_time"][0]) == valid
