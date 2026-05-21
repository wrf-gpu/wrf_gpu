from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
import zarr

from gpuwrf.io.data_inventory import build_gen2_d02_inventory
from gpuwrf.validation.data_quality import (
    audit_run_quality,
    build_quality_audit,
    compare_boundary_replay_to_wrfout,
    validate_quality_audit,
)


SURFACE_FIELDS = ("U10", "V10", "T2", "Q2", "PSFC", "RAINNC")
BOUNDARY_FIELDS = ("U", "V", "T", "QVAPOR", "PH")


def _write_wrfout(
    path: Path,
    valid_time: datetime,
    *,
    value: float = 1.0,
    omit: set[str] | None = None,
    nan_field: str | None = None,
    spike_field: str | None = None,
) -> None:
    omit = omit or set()
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("DateStrLen", 19)
        dataset.createDimension("south_north", 10)
        dataset.createDimension("west_east", 10)
        dataset.createDimension("bottom_top", 2)
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        stamp = valid_time.strftime("%Y-%m-%d_%H:%M:%S")
        times[0, :] = np.asarray(list(stamp), dtype="S1")
        for field in SURFACE_FIELDS:
            if field in omit:
                continue
            data = np.full((10, 10), value + len(field), dtype=np.float32)
            if field == nan_field:
                data[0, 0] = np.nan
            if field == spike_field:
                data[0, 0] = 1000.0
            variable = dataset.createVariable(field, "f4", ("Time", "south_north", "west_east"))
            variable[0, :, :] = data
        base = np.arange(200, dtype=np.float32).reshape(2, 10, 10)
        for field in BOUNDARY_FIELDS:
            if field in omit:
                continue
            variable = dataset.createVariable(field, "f4", ("Time", "bottom_top", "south_north", "west_east"))
            variable[0, :, :, :] = base + np.float32(len(field))


def _make_run(
    tmp_path: Path,
    *,
    run_id: str = "20260520_18z_l3_24h_20260521T045847Z",
    hours: int = 24,
    omit: set[str] | None = None,
    nan_field: str | None = None,
    spike_field: str | None = None,
) -> tuple[Path, datetime]:
    start = datetime(2026, 5, 20, 18, tzinfo=timezone.utc)
    run = tmp_path / run_id
    run.mkdir()
    (run / "wrfbdy_d01").write_text("", encoding="utf-8")
    for lead in range(hours + 1):
        valid = start + timedelta(hours=lead)
        _write_wrfout(
            run / f"wrfout_d02_{valid:%Y-%m-%d_%H:%M:%S}",
            valid,
            value=float(lead),
            omit=omit,
            nan_field=nan_field if lead == 0 else None,
            spike_field=spike_field if lead == 0 else None,
        )
    return run, start


def _make_replay(path: Path, wrfout_path: Path, valid_time: datetime, *, delta: float = 0.0) -> None:
    with Dataset(wrfout_path, "r") as dataset:
        fields = {field: np.asarray(dataset.variables[field][0]) for field in BOUNDARY_FIELDS}
    root = zarr.open_group(str(path), mode="w")
    root.attrs["times_utc"] = [valid_time.isoformat()]
    for field, data in fields.items():
        group = root.create_group(field)
        group.create_array("W", data=np.asarray([data[:, :, 0] + delta], dtype=np.float32), overwrite=True)
        group.create_array("E", data=np.asarray([data[:, :, -1] + delta], dtype=np.float32), overwrite=True)
        group.create_array("S", data=np.asarray([data[:, 0, :] + delta], dtype=np.float32), overwrite=True)
        group.create_array("N", data=np.asarray([data[:, -1, :] + delta], dtype=np.float32), overwrite=True)


def test_quality_audit_green_run_has_histograms(tmp_path: Path):
    _make_run(tmp_path)
    inventory = build_gen2_d02_inventory(tmp_path)

    audit = build_quality_audit(inventory)

    validate_quality_audit(audit)
    assert audit["status_counts"]["GREEN"] == 1
    u10 = audit["runs"][0]["fields"]["U10"]
    assert u10["nan_count"] == 0
    assert len(u10["histogram"]["counts"]) == 20


def test_nan_injection_marks_run_fail(tmp_path: Path):
    _make_run(tmp_path, nan_field="T2")
    run_record = build_gen2_d02_inventory(tmp_path)["runs"][0]

    result = audit_run_quality(run_record)

    assert result["status"] == "FAIL"
    assert result["fields"]["T2"]["nan_count"] == 1


def test_missing_field_marks_run_fail(tmp_path: Path):
    _make_run(tmp_path, omit={"Q2"})
    run_record = build_gen2_d02_inventory(tmp_path)["runs"][0]

    result = audit_run_quality(run_record)

    assert result["status"] == "FAIL"
    assert result["missing_fields"]


def test_partial_run_is_reported_without_sampling(tmp_path: Path):
    _make_run(tmp_path, hours=2)
    run_record = build_gen2_d02_inventory(tmp_path)["runs"][0]

    result = audit_run_quality(run_record)

    assert result["status"] == "PARTIAL"
    assert result["sampled"] is False
    assert result["missing_time_step_count"] == 22


def test_spike_detector_marks_partial(tmp_path: Path):
    _make_run(tmp_path, spike_field="U10")
    run_record = build_gen2_d02_inventory(tmp_path)["runs"][0]

    result = audit_run_quality(run_record)

    assert result["status"] == "PARTIAL"
    assert result["fields"]["U10"]["spike_flag"] is True


def test_boundary_replay_cross_check_passes_when_strips_match(tmp_path: Path):
    run, start = _make_run(tmp_path)
    replay = tmp_path / "d02_boundary_replay.zarr"
    _make_replay(replay, run / f"wrfout_d02_{start:%Y-%m-%d_%H:%M:%S}", start)

    result = compare_boundary_replay_to_wrfout(replay, run, valid_time=start.isoformat())

    assert result["status"] == "GREEN"
    assert result["variables"]["U"]["aggregate_rel_mae_max"] == 0.0


def test_boundary_replay_cross_check_fails_above_one_percent(tmp_path: Path):
    run, start = _make_run(tmp_path)
    replay = tmp_path / "d02_boundary_replay_bad.zarr"
    _make_replay(replay, run / f"wrfout_d02_{start:%Y-%m-%d_%H:%M:%S}", start, delta=10.0)

    result = compare_boundary_replay_to_wrfout(replay, run, valid_time=start.isoformat())

    assert result["status"] == "FAIL"
    assert result["failures"]
