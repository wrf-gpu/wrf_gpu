"""History transcribes radiation/PBL leaves and preserves the WRF t0 convention."""
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from netCDF4 import Dataset
import numpy as np

from gpuwrf.io.wrfout_writer import prepare_wrfout_payload, write_prepared_wrfout
from test_m7_netcdf_writer import synthetic_case, writer_authority
from test_initial_history_wrf_t0 import INPUT, _writer


def test_held_cloud_fraction_and_bl_fields_reach_history(tmp_path):
    state, grid, namelist = synthetic_case()
    volume = (grid.nz, grid.ny, grid.nx)
    diag = {name: np.full(volume, value, np.float32) for name,value in
            [('CLDFRA',.37),('QC_BL',2e-5),('CLDFRA_BL',.23),('DTAUX3D',.11),('DTAUY3D',-.09)]}
    diag.update({name: np.full((grid.ny,grid.nx),value,np.float32) for name,value in
        [('SWDNBC',201.),('SWUPBC',31.),('SWDNTC',702.),('SWUPTC',112.),
         ('LWDNBC',310.),('LWUPBC',370.),('LWDNTC',0.),('LWUPTC',231.)]})
    start = datetime(2026,2,27,18,tzinfo=timezone.utc)
    authority = writer_authority(grid, 'd01')
    prepared = prepare_wrfout_payload(state,grid,namelist,tmp_path/'history.nc',
        domain='d01',domain_authority=authority,valid_time=start,
        lead_hours=1.,run_start=start,diagnostics=diag,full_variable_set=True)
    write_prepared_wrfout(prepared,expected_domain='d01',expected_domain_authority=authority)
    with Dataset(prepared.target) as ds:
        for name,value in diag.items():
            np.testing.assert_array_equal(ds.variables[name][0],value,err_msg=name)


def test_t0_qke_history_keeps_input_value_without_changing_model_seed(tmp_path):
    def no_solve(*args,**kwargs):
        raise AssertionError('t0 must not re-solve physics')
    writer,state = _writer(tmp_path,no_solve)
    state.qke = np.full_like(state.theta, .75)
    model_qke = np.asarray(state.qke).copy()
    result = writer('d01',0,state)
    with Dataset(result['wrfout']) as ds:
        assert np.all(ds.variables['QKE'][0] == 0)
        assert np.all(ds.variables['CLDFRA'][0] == 0)
        for name in ('LH','GLW','PBLH'):
            assert np.all(ds.variables[name][0] == INPUT.get(name,0.))
    np.testing.assert_array_equal(state.qke,model_qke)


def test_actual_output_callback_transcribes_held_radiation_and_current_bl(tmp_path):
    writer,state = _writer(tmp_path,lambda *a,**k: {})
    writer._full_variable_set = True
    shape = state.theta.shape
    state.qc_bl = np.full(shape,2e-5,np.float32)
    state.cldfra_bl = np.full(shape,.23,np.float32)
    state.dtaux3d = np.full(shape,.11,np.float32)
    state.dtauy3d = np.full(shape,-.09,np.float32)
    expected = {'CLDFRA':np.full(shape,.37,np.float32),
                'QC_BL':state.qc_bl,'CLDFRA_BL':state.cldfra_bl,
                'DTAUX3D':state.dtaux3d,'DTAUY3D':state.dtauy3d}
    rad = SimpleNamespace(cloud_fraction=expected['CLDFRA'])
    for name,attr,value in (
        ('SWDNBC','sw_clear_sfc_down',201.),('SWUPBC','sw_clear_sfc_up',31.),
        ('SWDNTC','sw_clear_toa_down',702.),('SWUPTC','sw_clear_toa_up',112.),
        ('LWDNBC','lw_clear_sfc_down',310.),('LWUPBC','lw_clear_sfc_up',370.),
        ('LWDNTC','lw_clear_toa_down',0.),('LWUPTC','lw_clear_toa_up',231.),
    ):
        expected[name] = np.full(shape[1:],value,np.float32)
        setattr(rad,attr,expected[name])
    result = writer('d01',200,SimpleNamespace(state=state,radiation_diagnostics=rad))
    with Dataset(result['wrfout']) as ds:
        for name,value in expected.items():
            np.testing.assert_array_equal(ds.variables[name][0],value,err_msg=name)
