from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from netCDF4 import Dataset

from gpuwrf.integration.d02_replay import (
    _interp_parent_horizontal,
    _nested_axis_coords,
    _pack_nested_parent_history_3d,
)


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "m7_l2_d02_replay.py"
_SCRIPT_SPEC = importlib.util.spec_from_file_location("m7_l2_d02_replay", SCRIPT_PATH)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
m7_l2_d02_replay = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(m7_l2_d02_replay)
build_l2_inventory = m7_l2_d02_replay.build_l2_inventory
inventory_l2_run = m7_l2_d02_replay.inventory_l2_run
select_l2_run = m7_l2_d02_replay.select_l2_run


def _write_minimal_wrfout(path: Path, valid_time: datetime, *, domain_shape: tuple[int, int, int]) -> None:
    nz, ny, nx = domain_shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("DateStrLen", 19)
        dataset.createDimension("bottom_top", nz)
        dataset.createDimension("bottom_top_stag", nz + 1)
        dataset.createDimension("south_north", ny)
        dataset.createDimension("south_north_stag", ny + 1)
        dataset.createDimension("west_east", nx)
        dataset.createDimension("west_east_stag", nx + 1)
        dataset.DX = 3000.0
        dataset.DY = 3000.0
        dataset.MAP_PROJ = 1
        dataset.CEN_LAT = 28.0
        dataset.CEN_LON = -16.0
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0, :] = np.asarray(list(valid_time.strftime("%Y-%m-%d_%H:%M:%S")), dtype="S1")


def _make_run(root: Path, run_id: str, *, hours: int, missing_d02_final: bool = False) -> Path:
    run = root / run_id
    start = datetime(2026, 5, 1, 18, tzinfo=timezone.utc)
    for hour in range(hours + 1):
        valid = start + timedelta(hours=hour)
        stamp = valid.strftime("%Y-%m-%d_%H:%M:%S")
        _write_minimal_wrfout(run / f"wrfout_d01_{stamp}", valid, domain_shape=(2, 3, 4))
        if not (missing_d02_final and hour == hours):
            _write_minimal_wrfout(run / f"wrfout_d02_{stamp}", valid, domain_shape=(2, 6, 9))
    return run


def test_nested_axis_coords_use_parent_start_and_ratio() -> None:
    child = SimpleNamespace(parent_grid_ratio=3, i_parent_start=24, j_parent_start=20)

    y, x = _nested_axis_coords(child, y_len=3, x_len=4)

    np.testing.assert_allclose(y, [19.0, 19.0 + 1.0 / 3.0, 19.0 + 2.0 / 3.0])
    np.testing.assert_allclose(x, [23.0, 23.0 + 1.0 / 3.0, 23.0 + 2.0 / 3.0, 24.0])


def test_parent_horizontal_interpolation_is_bilinear_for_3d_fields() -> None:
    y, x = np.indices((5, 6), dtype=np.float64)
    parent = (10.0 * y + x)[None, :, :].astype(np.float32)

    child = _interp_parent_horizontal(parent, np.asarray([1.0, 1.5]), np.asarray([2.0, 2.5, 3.0]))

    assert child.shape == (1, 2, 3)
    np.testing.assert_allclose(child[0], [[12.0, 12.5, 13.0], [17.0, 17.5, 18.0]])


def test_pack_nested_parent_history_returns_child_sides_with_padding() -> None:
    class FakeRun:
        def load(self, domain, var, time, lazy=True):
            del domain, var, lazy
            y, x = np.indices((5, 6), dtype=np.float32)
            return (10.0 * y + x + np.float32(time))[None, :, :]

    child = SimpleNamespace(parent_grid_ratio=2, i_parent_start=1, j_parent_start=1)

    packed = _pack_nested_parent_history_3d(
        FakeRun(),
        child,
        "d01",
        "U",
        ntimes=2,
        child_shape=(1, 2, 3),
        max_side=4,
        dtype=np.float32,
    )

    assert packed.shape == (2, 4, 1, 4)
    np.testing.assert_allclose(packed[0, 0, 0, :2], [0.0, 5.0])
    np.testing.assert_allclose(packed[0, 2, 0, :3], [0.0, 0.5, 1.0])
    assert packed[0, 0, 0, 2] == 0.0
    np.testing.assert_allclose(packed[1, 2, 0, :3], [1.0, 1.5, 2.0])


def test_l2_inventory_marks_complete_runs_and_selects_latest_full(tmp_path: Path) -> None:
    older = _make_run(tmp_path, "20260501_18z_l2_2h_20260502T000000Z", hours=2)
    _make_run(tmp_path, "20260502_18z_l2_2h_20260503T000000Z", hours=2, missing_d02_final=True)

    older_record = inventory_l2_run(older, requested_hours=2)
    inventory = build_l2_inventory(tmp_path, requested_hours=2)

    assert older_record["d01_file_count"] == 3
    assert older_record["d02_file_count"] == 3
    assert older_record["complete_full"] is True
    assert inventory["complete_full_run_count"] == 1
    assert select_l2_run(inventory, requested_hours=2) == older.name
