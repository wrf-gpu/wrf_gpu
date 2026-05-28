from __future__ import annotations

from pathlib import Path

import zarr

from gpuwrf.io.boundary_replay import BOUNDARY_VARIABLES, SIDES, TOLERANCES, extract_d02_boundary
from gpuwrf.io.gen2_accessor import DEFAULT_M6_BOUNDARY_REPLAY, DEFAULT_M6_GEN2_RUN_DIR, Gen2Run
from gpuwrf.paths import reference_path


RUN_PATH = reference_path("runs", "wrf_l3", "20260519_18z_l3_24h_20260520T025228Z")
FIXTURE = Path("data/fixtures/m6/d02_boundary_replay_v1.zarr")
FIXTURE_V2 = DEFAULT_M6_BOUNDARY_REPLAY


def _ensure_fixture() -> None:
    if not FIXTURE.exists():
        extract_d02_boundary(Gen2Run(RUN_PATH), str(FIXTURE))


def test_d02_boundary_replay_zarr_schema_and_shapes():
    _ensure_fixture()
    root = zarr.open_group(str(FIXTURE), mode="r")

    assert root.attrs["schema"] == "d02_boundary_replay_v1"
    assert root.attrs["source_parent_domain"] == "d01"
    assert root.attrs["target_domain"] == "d02"
    assert len(root.attrs["times_utc"]) == 25
    for var in BOUNDARY_VARIABLES:
        for side in SIDES:
            array = root[var][side]
            assert array.shape[0] == 25
            assert array.shape[1] in (44, 45)
            assert array.shape[2] > 0


def test_d02_boundary_round_trip_tolerances_are_documented_and_pass():
    _ensure_fixture()
    root = zarr.open_group(str(FIXTURE), mode="r")
    validation = root.attrs["validation"]

    for var in BOUNDARY_VARIABLES:
        aggregate = validation[var]["aggregate"]
        assert aggregate["passed"] is True
        assert validation[var]["tolerance"] == TOLERANCES[var]


def test_d02_boundary_replay_v2_fixture_is_re_pinned_when_present():
    if not FIXTURE_V2.exists():
        return
    root = zarr.open_group(str(FIXTURE_V2), mode="r")

    assert root.attrs["schema"] == "d02_boundary_replay_v2"
    assert root.attrs["run_id"] == DEFAULT_M6_GEN2_RUN_DIR.name
    assert len(root.attrs["times_utc"]) == 25
