from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gpuwrf.io.proof_schemas import Tier4ProbtestTolerances
from gpuwrf.validation.tier4_probtest import (
    PROTOTYPE_LABEL,
    derive_stratified_tolerance_records,
    select_historical_members,
    write_json,
)


def test_stratified_tolerance_uses_rms_member_std_without_cap():
    samples = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 2.0], [5.0, 8.0]],
            [[3.0, 2.0], [7.0, 12.0]],
        ],
        dtype=np.float64,
    )
    masks = {
        "land": np.asarray([[True, False], [True, False]]),
        "sea": np.asarray([[False, True], [False, True]]),
    }

    records = derive_stratified_tolerance_records(samples, masks, tolerance_factor=1.96)

    land_std = np.asarray([1.0, 2.0])
    expected_land_sigma = float(np.sqrt(np.mean(land_std * land_std)))
    assert np.isclose(records["land"]["sigma_rms_member_std"], expected_land_sigma)
    assert np.isclose(records["land"]["tolerance"], 1.96 * expected_land_sigma)
    assert records["sea"]["tolerance"] > 0.0


def test_member_selection_excludes_heldout_and_uses_latest_duplicate(tmp_path: Path):
    names = [
        "20260508_18z_l3_24h_20260509T010000Z",
        "20260509_18z_l3_24h_20260510T010000Z",
        "20260509_18z_l3_24h_20260510T020000Z",
        "20260510_18z_l3_24h_20260511T010000Z",
        "20260511_18z_l3_24h_20260512T010000Z",
    ]
    for name in names:
        (tmp_path / name).mkdir()

    selected = select_historical_members(
        tmp_path,
        ending_cycle="20260511_18z",
        count=3,
        heldout_cycle="20260510_18z",
    )

    assert [path.name for path in selected] == [
        "20260508_18z_l3_24h_20260509T010000Z",
        "20260509_18z_l3_24h_20260510T020000Z",
        "20260511_18z_l3_24h_20260512T010000Z",
    ]


def test_tier4_schema_requires_prototype_freeze_fields(tmp_path: Path):
    payload = {
        "artifact_type": "tier4_probtest_tolerances",
        "run_id": "unit",
        "status": "PASS",
        "prototype_label": PROTOTYPE_LABEL,
        "domain": "d02",
        "sample_size": 10,
        "variables": ["U10"],
        "leads_h": [6],
        "strata": ["land", "sea", "elevation_band_0"],
        "member_manifest": "artifacts/m6/tier4/ensemble_member_manifest.json",
        "tolerances": {"U10": {"6h": {"land": {"tolerance": 1.0}}}},
        "freeze_time_utc": "2026-05-21T00:00:00+00:00",
        "tolerance_factor": 1.96,
        "method": {"candidate_peek_policy": "frozen before candidate"},
        "heldout_policy": {"heldout_cycle": "20260519_18z"},
        "artifact_paths": [],
    }
    path = tmp_path / "probtest_tolerances.json"
    write_json(path, payload)

    assert Tier4ProbtestTolerances.validate_file(path)["prototype_label"] == PROTOTYPE_LABEL

    bad = json.loads(path.read_text(encoding="utf-8"))
    del bad["prototype_label"]
    try:
        Tier4ProbtestTolerances.validate_dict(bad)
    except ValueError as exc:
        assert "prototype_label" in str(exc)
    else:
        raise AssertionError("schema accepted a Tier-4 artifact without prototype_label")
