from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from gpuwrf.validation.tier4_rmse_harness import run_tier4_rmse_harness


def _write_wrfout(path: Path, *, hour: int, offset: float) -> None:
    netCDF4 = __import__("netCDF4")

    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("south_north", 2)
        dataset.createDimension("west_east", 3)
        for index, variable in enumerate(("U10", "V10", "T2")):
            out = dataset.createVariable(variable, "f8", ("Time", "south_north", "west_east"))
            out[0, :, :] = np.full((2, 3), offset + hour + index, dtype=np.float64)


def _make_member(root: Path, cycle: str, created: str, *, offset: float) -> None:
    run_dir = root / f"{cycle}_l3_24h_{created}"
    start = datetime.strptime(cycle, "%Y%m%d_%Hz")
    for hour in range(25):
        stamp = (start + timedelta(hours=hour)).strftime("%Y-%m-%d_%H:%M:%S")
        _write_wrfout(run_dir / f"wrfout_d02_{stamp}", hour=hour, offset=offset)


def test_non_operational_mode_passes_probationary_at_n5(tmp_path: Path) -> None:
    for index in range(5):
        cycle = f"2026050{index + 1}_18z"
        _make_member(tmp_path, cycle, f"2026050{index + 2}T010000Z", offset=float(index))

    payload = run_tier4_rmse_harness(
        roots=(tmp_path,),
        non_operational=True,
        ending_cycle="20260505_18z",
        heldout_cycle=None,
        variables=("U10",),
        leads_h=(1,),
        pinned_grid_yx=(2, 3),
    )

    assert payload["status"] == "PASS_PROBATIONARY"
    assert payload["corpus_size_class"] == "bounded"
    assert payload["M7_close_class"] == "probationary"
    assert payload["required_member_count"] == 5
    assert payload["member_count"] == 5
    assert payload["member_split"]["selected_count"] == 5
    assert payload["rmse_records"]
    assert all(record["finite"] for record in payload["rmse_records"])


def test_operational_default_still_blocks_small_corpus(tmp_path: Path) -> None:
    for index in range(5):
        cycle = f"2026050{index + 1}_18z"
        _make_member(tmp_path, cycle, f"2026050{index + 2}T010000Z", offset=float(index))

    payload = run_tier4_rmse_harness(
        roots=(tmp_path,),
        non_operational=False,
        ending_cycle="20260505_18z",
        heldout_cycle=None,
        variables=("U10",),
        leads_h=(1,),
        pinned_grid_yx=(2, 3),
    )

    assert payload["status"] == "BLOCKED_CORPUS"
    assert payload["corpus_size_class"] == "standard"
    assert payload["required_member_count"] == 10
    assert payload["needed_members"] == 5
