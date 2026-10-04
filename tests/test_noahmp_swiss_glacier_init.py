"""Original CPU-WRF lead-zero savepoint is the real NOAHMP_INIT oracle."""
from pathlib import Path
import jax
import numpy as np
import pytest
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.noahmp_land_init import build_noahmp_land_state

SOURCE = Path(__file__).resolve().parents[1]/'examples/switzerland_d01'
CPU = Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu/wrfout_d01_2023-01-15_00:00:00')


def test_real_swiss_glacier_and_porosity_initialization_match_pristine_wrf():
    if not CPU.exists():
        pytest.skip('original Swiss CPU-WRF NOAHMP_INIT savepoint not mounted')
    land, static, metadata = build_noahmp_land_state(SOURCE, 'd01')
    land, static = jax.device_get((land, static))
    ice = np.asarray(static.ivgtyp) == int(static.parameters.isice)
    assert int(ice.sum()) == 22  # All real glacier columns in the shipped case.
    with Dataset(CPU) as cpu:
        for name, attr in (('SMOIS','smois'),('TSLB','tslb'),('SH2O','sh2o')):
            reference = np.asarray(cpu[name][0])
            value = np.asarray(getattr(land,attr),dtype=reference.dtype)
            np.testing.assert_array_equal(value[:,ice],reference[:,ice],err_msg=name)
        # Porosity limiting is also an in-place WRF initialization operation;
        # retain all non-glacier columns and both temperature dimensions exactly.
        for name,attr in (('SMOIS','smois'),('TSLB','tslb')):
            reference = np.asarray(cpu[name][0])
            np.testing.assert_array_equal(np.asarray(getattr(land,attr),dtype=reference.dtype),reference,err_msg=name)
        np.testing.assert_array_equal(np.asarray(land.sneqv,dtype=np.float32)[ice],cpu['SNOW'][0][ice])
        np.testing.assert_allclose(np.asarray(land.snowh)[ice],cpu['SNOWH'][0][ice],atol=1e-7,rtol=1e-7)
        np.testing.assert_array_equal(np.asarray(land.isnow)[ice],cpu['ISNOW'][0][ice])
    assert np.all(np.asarray(land.smois)[:,ice] == 1)
    assert np.all(np.asarray(land.sh2o)[:,ice] == 0)
    assert np.all(np.asarray(land.tslb)[:,ice] <= 263.15)


def test_positive_xice_uses_normal_soil_branch_on_a_real_swiss_column(tmp_path):
    import shutil
    if not CPU.exists():
        pytest.skip('original Swiss CPU-WRF NOAHMP_INIT savepoint not mounted')
    (tmp_path/'namelist.input').symlink_to(SOURCE/'namelist.input')
    target = tmp_path/'wrfinput_d01'
    shutil.copyfile(SOURCE/'wrfinput_d01',target)
    with Dataset(target,'r+') as ds, Dataset(CPU) as cpu:
        warm = np.all(np.asarray(ds['TSLB'][0]) >= 273.149,axis=0)
        warm &= np.asarray(ds['IVGTYP'][0]) != 15
        y,x = np.argwhere(warm)[0]
        expected = {name:np.asarray(cpu[name][0])[:,y,x] for name in ('SMOIS','TSLB','SH2O')}
        # Preserve the real soil/temperature inputs. XICE>0 excludes the glacier
        # branch even with the ice vegetation category (WRF INIT:2073).
        ds['IVGTYP'][0,y,x] = 15
        ds['SEAICE'][0,y,x] = .5
    land, _, _ = build_noahmp_land_state(tmp_path,'d01')
    for name,attr in (('SMOIS','smois'),('TSLB','tslb'),('SH2O','sh2o')):
        actual = np.asarray(getattr(land,attr),dtype=expected[name].dtype)[:,y,x]
        np.testing.assert_array_equal(actual,expected[name],err_msg=name)
    assert np.all(np.asarray(land.sh2o)[:,y,x] > 0)
