"""CPU tests for scoring policy and false-pass rejection; no physics claim."""
import json
from pathlib import Path

import numpy as np
import pytest

from gpuwrf.diagnostics.census_transparency import compare_array, compare_wrfout, verify_receipts


BOUNDS = json.loads(Path(__file__).with_name("transparency_bounds.json").read_text())


def test_receipt_identity_rejects_different_source_work_or_flags():
    off = {"git_head": "commit", "src_tree": "tree", "rc": 0,
           "argv": ["runner", "--input-dir", "/case", "--hours", "3"],
           "env": {"GPUWRF_CENSUS": "0", "GPUWRF_DYN_FP32": "0"}}
    on = {**off, "env": {**off["env"], "GPUWRF_CENSUS": "1"}}
    assert verify_receipts(off, on, 3)["src_tree"] == "tree"
    for changed in ({**on, "src_tree": "other"}, {**on, "rc": 1},
                    {**on, "argv": ["runner", "--input-dir", "/other", "--hours", "3"]},
                    {**on, "env": {**on["env"], "GPUWRF_DYN_FP32": "1"}}):
        with pytest.raises(ValueError):
            verify_receipts(off, changed, 3)


def test_d6_fraction_and_maximum_gate():
    for field in BOUNDS["required_output_fields"]:
        spec = BOUNDS["fields"][field]
        assert spec["max_abs"] == spec["rmse"] == spec["d6_rmse"] * 0.001
    off = np.zeros(100)
    on = off.copy()
    on[0] = 0.003
    result = compare_array("state.theta", off, on, BOUNDS)
    assert result["rmse"] < result["limits"]["rmse"]
    assert not result["pass"]  # Small pooled RMS cannot hide a maximum breach.
    on[0] = 0.001
    assert compare_array("state.theta", off, on, BOUNDS)["pass"]


def test_shape_dtype_nonfinite_and_exact_fields_fail_closed():
    off = np.ones(2)
    assert not compare_array("T", off, off[:1], BOUNDS)["pass"]
    assert not compare_array("T", off, off.astype(np.float32), BOUNDS)["pass"]
    assert not compare_array("T", off, np.array([1.0, np.nan]), BOUNDS)["pass"]
    assert not compare_array("base_state.pb", off, off + 1e-10, BOUNDS)["pass"]
    assert not compare_array("NCA", np.array([1]), np.array([2]), BOUNDS)["pass"]


def _files(directory, hours=1):
    from netCDF4 import Dataset
    directory.mkdir()
    for domain in ("d01", "d02"):
        for hour in range(hours + 1):
            path = directory / f"wrfout_{domain}_2026-07-26_{hour:02d}:00:00"
            with Dataset(path, "w") as ds:
                ds.createDimension("n", 2)
                for name in BOUNDS["required_output_fields"]:
                    ds.createVariable(name, "f8", ("n",))[:] = 0.0


def test_same_work_outputs_pass_and_tiny_maximum_breach_fails(tmp_path):
    from netCDF4 import Dataset
    off, on = tmp_path / "off", tmp_path / "on"
    _files(off)
    _files(on)
    assert compare_wrfout(off, on, BOUNDS, 1)["pass"]
    with Dataset(on / "wrfout_d02_2026-07-26_01:00:00", "a") as ds:
        ds["U10"][:] = 0.003
    assert not compare_wrfout(off, on, BOUNDS, 1)["pass"]


@pytest.mark.parametrize("missing", ["domain", "hour", "field"])
def test_missing_coverage_cannot_pass(tmp_path, missing):
    from netCDF4 import Dataset
    off, on = tmp_path / "off", tmp_path / "on"
    _files(off)
    _files(on)
    if missing == "domain":
        for path in on.glob("wrfout_d02_*"):
            path.unlink()
    elif missing == "hour":
        (on / "wrfout_d01_2026-07-26_01:00:00").unlink()
    else:
        with Dataset(on / "wrfout_d01_2026-07-26_01:00:00", "a") as ds:
            ds.renameVariable("T2", "unexpected")
    with pytest.raises(ValueError, match="missing|mismatched|inventory"):
        compare_wrfout(off, on, BOUNDS, 1)
