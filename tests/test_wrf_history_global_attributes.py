"""CPU-WRF inventory gates for real WN3 history metadata (no model execution)."""
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

from netCDF4 import Dataset
import numpy as np
import pytest

from gpuwrf.io.gen2_accessor import parse_namelist
from gpuwrf.io.wrf_history_metadata import wrf_history_global_attributes
from gpuwrf.io.wrfout_writer import _write_global_attrs, prepare_wrfout_payload, write_prepared_wrfout
from test_m7_netcdf_writer import synthetic_case, writer_authority

FIXTURE = json.loads((Path(__file__).parent/'fixtures/wrf_history_attributes_wn3_0227.json').read_text())
CASE = Path('<USER_HOME>/wrf_gpu2_lanes/wn3/cases/20260227_18z_a1')


@pytest.mark.parametrize('domain', ['d01', 'd02', 'd03'])
def test_all_cpu_wrf_history_attributes_are_supplied(domain):
    expected = FIXTURE[domain]['attributes']
    assert len(expected) == 155
    attrs = wrf_history_global_attributes(CASE/f'wrfinput_{domain}',
        parse_namelist(CASE/'namelist.input'), domain, expected['DT'])
    assert set(expected) <= set(attrs)
    for name, value in expected.items():
        if name in {'TITLE', 'NTASKS_X', 'NTASKS_Y', 'NTASKS_TOTAL'}:
            continue  # GPU provenance and one process/case are intentional.
        np.testing.assert_array_equal(attrs[name], value, err_msg=name)
    assert attrs['NTASKS_TOTAL'] == attrs['NTASKS_X'] == attrs['NTASKS_Y'] == 1


def test_metadata_reaches_actual_host_payload_and_writer(tmp_path):
    state, grid, namelist = synthetic_case()
    start = datetime(2026, 2, 27, 18, tzinfo=timezone.utc)
    attrs = wrf_history_global_attributes(CASE/'wrfinput_d02',
        parse_namelist(CASE/'namelist.input'), 'd02', 18.0)
    attrs['GRID_ID'] = np.int32(999)  # The writer's authenticated identity wins.
    authority = writer_authority(grid, 'd02')
    prepared = prepare_wrfout_payload(state, grid, namelist, tmp_path/'history.nc',
        domain='d02', domain_authority=authority, valid_time=start,
        lead_hours=0.0, run_start=start, source_global_attrs=attrs)
    write_prepared_wrfout(prepared, expected_domain='d02', expected_domain_authority=authority)
    with Dataset(prepared.target) as ds:
        assert set(FIXTURE['d02']['attributes']) <= set(ds.ncattrs())
        for name in ('PARENT_ID','I_PARENT_START','J_PARENT_START','PARENT_GRID_RATIO',
                     'BUCKET_MM','BUCKET_J','DT','MMINLU','NUM_LAND_CAT'):
            np.testing.assert_array_equal(ds.getncattr(name), attrs[name], err_msg=name)
        assert ds.GRID_ID == 2
        assert ds.getncattr('WEST-EAST_GRID_DIMENSION') == prepared.dimensions['west_east_stag']
        assert 'GPUWRF' in ds.TITLE


def test_julian_header_keeps_simulation_start_across_midnight(tmp_path):
    _, grid, namelist = synthetic_case()
    with Dataset(tmp_path/'header.nc', 'w') as ds:
        _write_global_attrs(ds, grid, namelist,
            {'west_east_stag': 3, 'south_north_stag': 3, 'bottom_top_stag': 4},
            datetime(2026,2,27,18,tzinfo=timezone.utc),
            datetime(2026,2,28,1,tzinfo=timezone.utc))
        assert ds.JULDAY == 58 and ds.JULYR == 2026
