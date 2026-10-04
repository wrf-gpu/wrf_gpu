"""MYNN's computed EL prefix is unchanged in WRF's staggered history array."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.wrfout_writer import prepare_wrfout_payload, write_prepared_wrfout
from test_m7_netcdf_writer import writer_authority


WN3 = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run")
D5 = Path("<DATA_ROOT>/alisios/runs/20260725_18z/wn2_wrf/alisios_operational_d01_121x71_d02_268x118_oldgrid_v1/attempt_001/cpu_case")
CASES = [(WN3, "d01"), (WN3, "d02"), (WN3, "d03"), (D5, "d01"), (D5, "d02")]


def _write_el(tmp_path, value, grid, domain):
    authority = writer_authority(grid, domain)
    path = tmp_path / "history.nc"
    prepared = prepare_wrfout_payload(
        SimpleNamespace(el_pbl=value), grid, {"dx": 9000.0, "dy": 9000.0}, path,
        domain=domain, domain_authority=authority,
        valid_time="2026-02-28T01:00:00Z", lead_hours=1,
        run_start="2026-02-28T00:00:00Z", variable_subset=("EL_PBL",),
        full_variable_set=True,
    )
    write_prepared_wrfout(prepared, expected_domain=domain, expected_domain_authority=authority)
    return path


@pytest.mark.parametrize("root,domain", CASES, ids=["WN3-d01", "WN3-d02", "WN3-d03", "D5-d01", "D5-d02"])
@pytest.mark.parametrize("frame_index", [0, 1, 6, 24])
def test_mass_el_history_matches_original_cpu_header_and_values(tmp_path, root, domain, frame_index):
    if not root.is_dir():
        pytest.skip("original CPU-WRF reference not mounted")
    files = sorted(root.glob(f"wrfout_{domain}_*"))
    assert files, (root, domain)
    source = files[min(frame_index, len(files) - 1)]
    with Dataset(source) as cpu:
        variable = cpu["EL_PBL"]
        dimensions, dtype = variable.dimensions, variable.dtype
        truth = np.asarray(variable[0])
        assert dimensions == ("Time", "bottom_top_stag", "south_north", "west_east")
        assert not np.any(np.ma.getmaskarray(variable[0, -1]))
    # Independently measured WRF convention, including the sign bit of zero.
    assert truth[-1].tobytes() == np.zeros_like(truth[-1]).tobytes(), source
    nz1, ny, nx = truth.shape
    grid = SimpleNamespace(nx=nx, ny=ny, nz=nz1 - 1)
    mass = truth[:-1].copy()
    before = mass.tobytes()
    output = _write_el(tmp_path, mass, grid, domain)
    assert mass.tobytes() == before, "writer changed the live MYNN diagnostic"
    with Dataset(output) as gpu:
        variable = gpu["EL_PBL"]
        assert variable.dimensions == dimensions and variable.dtype == dtype
        result = np.asarray(variable[0])
        assert result.shape == truth.shape
        assert result[:-1].tobytes() == before, "computed EL levels were shifted or interpolated"
        assert result[-1].tobytes() == truth[-1].tobytes(), "unused top differs from CPU-WRF"
        assert result[0].tobytes() == truth[0].tobytes(), "bottom EL was changed"


def test_already_staggered_el_is_preserved(tmp_path):
    grid = SimpleNamespace(nx=3, ny=2, nz=4)
    value = np.arange(30, dtype=np.float32).reshape(5, 2, 3) + np.float32(0.125)
    output = _write_el(tmp_path, value, grid, "d01")
    with Dataset(output) as ds:
        assert np.asarray(ds["EL_PBL"][0]).tobytes() == value.tobytes()


def test_wrong_el_shape_remains_an_error(tmp_path):
    grid = SimpleNamespace(nx=3, ny=2, nz=4)
    with pytest.raises(ValueError, match="EL_PBL"):
        _write_el(tmp_path, np.ones((3, 2, 3), dtype=np.float32), grid, "d01")
