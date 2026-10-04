"""Held radiation CLDFRA producer, WRF t0 seed and E78 restart coverage."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np
import pytest

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.coupling import physics_couplers as C
from gpuwrf.io.restart import read_restart, write_restart
from gpuwrf.io.wrfrst_netcdf import read_wrfrst_carry, write_wrfrst_carry
from gpuwrf.runtime import operational_mode as O
from gpuwrf.runtime.operational_state import initial_operational_carry
from gpuwrf.runtime.restart_store import RestartStore


def diagnostic(cf):
    xy=jnp.zeros(cf.shape[1:],jnp.float64)
    return C.RRTMGRadiationDiagnostics(*(xy if i!=10 else xy.astype(jnp.int32) for i in range(15)),
        *(xy+i for i in range(8)),cloud_fraction=cf)


@pytest.mark.parametrize('entry',['diagnostics','tendency'])
def test_actual_radiation_entries_publish_the_cloud_input(monkeypatch,entry):
    cf=jnp.arange(24,dtype=jnp.float32).reshape(2,3,4)/24
    xy=jnp.ones((2,3),jnp.float32)
    columns=SimpleNamespace(cloud_fraction=cf)
    geometry=SimpleNamespace(coszen=xy)
    sw=SimpleNamespace(heating_rate=jnp.zeros_like(cf),surface_down=xy,surface_down_topographic=xy,
        surface_up=xy,surface_up_topographic=xy,topographic_correction_factor=xy,toa_down=xy,toa_up=xy)
    lw=SimpleNamespace(heating_rate=jnp.zeros_like(cf),surface_down=xy,surface_up=xy,toa_down=xy,toa_up=xy)
    sw.clear_flux_down=jnp.arange(30,dtype=jnp.float32).reshape(2,3,5)
    sw.clear_flux_up=sw.clear_flux_down+100
    lw.clear_flux_down=sw.clear_flux_down+200
    lw.clear_flux_up=sw.clear_flux_down+300
    monkeypatch.setattr(C,'_rrtmg_real_enabled',lambda:False)
    monkeypatch.setattr(C,'_rrtmg_column_inputs',lambda *a,**k:(columns,columns,xy,xy,geometry,None))
    monkeypatch.setattr(C,'solve_rrtmg_sw_column',lambda *a,**k:sw)
    monkeypatch.setattr(C,'solve_rrtmg_lw_column',lambda *a,**k:lw)
    state=SimpleNamespace(theta=jnp.full((4,2,3),300.),p=jnp.full((4,2,3),90000.),t_skin=xy)
    if entry=='diagnostics':
        result=C.rrtmg_radiation_diagnostics(state,None,with_clear_sky=True)
    else:
        _,result=C.rrtmg_theta_tendency(state,None,_with_diagnostics=True,with_clear_sky=True)
    # Non-square dimensions and distinct cells detect any transpose mistake.
    expected=np.asarray(cf).transpose(2,0,1)
    assert np.array_equal(np.asarray(result.cloud_fraction),expected)
    assert result.cloud_fraction.dtype==cf.dtype
    assert np.array_equal(result.sw_clear_sfc_down,sw.clear_flux_down[...,0])
    assert np.array_equal(result.sw_clear_toa_up,sw.clear_flux_up[...,-1])
    assert np.array_equal(result.lw_clear_sfc_up,lw.clear_flux_up[...,0])
    assert all(getattr(result,f'{kind}_clear_{level}_{direction}') is not None
               for kind in ('sw','lw') for level in ('sfc','toa') for direction in ('down','up'))


def test_noah_init_cloud_fraction_matches_wrf_t0_and_keeps_fluxes(monkeypatch):
    state=SimpleNamespace(qv=jnp.ones((4,2,3),jnp.float32),t_skin=jnp.ones((2,3),jnp.float32))
    rad=diagnostic(jnp.full_like(state.qv,.7))._replace(swnorm=jnp.full((2,3),100.),glw=jnp.full((2,3),300.),coszen=jnp.full((2,3),.5))
    calls=[]
    monkeypatch.setattr(O,'rrtmg_radiation_diagnostics',lambda *a,**k:calls.append(k) or rad)
    nl=SimpleNamespace(grid=None,time_utc=datetime(2026,2,27),radiation_static=None,topo_shading=0,slope_rad=0,
                       topo_shadow_length_m=25000.,ra_sw_physics=4,ra_lw_physics=4)
    forcing,result=O.noahmp_initial_rad(state,nl,_with_diagnostics=True)
    assert result.cloud_fraction.shape==state.qv.shape
    assert not np.asarray(result.cloud_fraction).any()
    assert [float(np.asarray(v)[0,0]) for v in forcing]==[100.,300.,.5]
    assert calls[0]['with_clear_sky'] is True


def fixture_carry():
    grid=GridSpec.canary_3km_template()
    state=State(**{name:jnp.full(shape,1.,DEFAULT_DTYPES.dtype_for(name)) for name,shape in _state_field_shapes(grid).items()})
    cf=jnp.arange(state.qv.size,dtype=jnp.float32).reshape(state.qv.shape)/state.qv.size
    carry=initial_operational_carry(state).replace(radiation_diagnostics=diagnostic(cf))
    nl=O.OperationalNamelist(grid=grid,tendencies=None,metrics=grid.metrics,dt_s=54.,acoustic_substeps=4,
        bl_pbl_physics=0,cu_physics=0,ra_sw_physics=4,ra_lw_physics=4)
    return grid,nl,carry


def test_generic_initial_carry_has_a_typed_zero_cloud_fraction():
    _,nl,carry=fixture_carry()
    seed=O._initial_carry_for_run(carry.state,nl)
    assert seed.radiation_diagnostics.cloud_fraction.shape==carry.state.qv.shape
    assert not np.asarray(seed.radiation_diagnostics.cloud_fraction).any()
    assert all(getattr(seed.radiation_diagnostics,f'{kind}_clear_{level}_{direction}') is not None
               for kind in ('sw','lw') for level in ('sfc','toa') for direction in ('down','up'))


def test_driver_refreshes_cloud_and_clear_fluxes_once_then_holds(monkeypatch):
    _,nl,carry=fixture_carry()
    fresh=diagnostic(jnp.full_like(carry.state.qv,.8))._replace(sw_clear_sfc_down=jnp.full(carry.state.t_skin.shape,123.))
    calls=[]
    def radiation(*a,**kw):
        calls.append(kw)
        return jnp.zeros_like(carry.rthraten),fresh
    monkeypatch.setattr(O,'rrtmg_theta_tendency',radiation)
    refreshed=O._refresh_rrtmg_driver(carry,nl,54.,True,None)
    held=O._refresh_rrtmg_driver(refreshed,nl,108.,False,None)
    assert len(calls)==1 and calls[0]['with_clear_sky'] is True
    assert held.radiation_diagnostics.cloud_fraction is refreshed.radiation_diagnostics.cloud_fraction
    assert np.asarray(held.radiation_diagnostics.sw_clear_sfc_down).min()==123.
    # Traced cadence must have identical cloud/clear leaf shapes in both arms.
    fn=jax.jit(lambda run:O._refresh_rrtmg_driver(carry,nl,54.,run,None))
    assert np.asarray(fn(jnp.bool_(True)).radiation_diagnostics.cloud_fraction).min()==np.float32(.8)


@pytest.mark.parametrize('format',['pickle','netcdf','store'])
def test_held_cloud_fraction_roundtrips_exactly(format,tmp_path):
    grid,nl,carry=fixture_carry()
    if format=='pickle':
        p=tmp_path/'carry.pkl';write_restart(carry,nl,grid,11,p);restored=read_restart(p)[0]
    elif format=='netcdf':
        p=tmp_path/'carry.nc';write_wrfrst_carry(carry,grid,{},p,valid_time='2026-02-27_18:09:54',run_start='2026-02-27_18:00:00',step_index=11)
        restored=read_wrfrst_carry(p)[0]
    else:
        store=RestartStore(tmp_path/'store',10**8,2,0)
        p,_=store.save({'d01':carry},{'d01':11},{'d01':54.},{})
        restored=store.read(p)[0]['carries']['d01']
    left=np.asarray(carry.radiation_diagnostics.cloud_fraction);right=np.asarray(restored.radiation_diagnostics.cloud_fraction)
    assert (left.dtype,left.shape,left.tobytes())==(right.dtype,right.shape,right.tobytes())


@pytest.mark.parametrize('format',['pickle','netcdf','store'])
def test_pre_cloud_schema_is_rejected_without_silent_none(format,tmp_path):
    grid,nl,carry=fixture_carry()
    if format=='pickle':
        p=tmp_path/'carry.pkl';write_restart(carry,nl,grid,11,p)
        data=pickle.loads(p.read_bytes());data['carry'].pop('radiation_diagnostics_schema_version')
        data['carry']['extra_fields']['radiation_diagnostics']=data['carry']['extra_fields']['radiation_diagnostics']._replace(cloud_fraction=None)
        p.write_bytes(pickle.dumps(data))
        read=lambda:read_restart(p)
    elif format=='netcdf':
        p=tmp_path/'carry.nc';write_wrfrst_carry(carry,grid,{},p,valid_time='2026-02-27_18:09:54',run_start='2026-02-27_18:00:00',step_index=11)
        with netCDF4.Dataset(p,'r+') as nc:nc.delncattr('GPUWRF_RADIATION_DIAGNOSTICS_SCHEMA_VERSION')
        read=lambda:read_wrfrst_carry(p)
    else:
        store=RestartStore(tmp_path/'store',10**8,2,0);p,_=store.save({'d01':carry},{'d01':11},{'d01':54.},{})
        data=pickle.loads((p/'snapshot.pkl').read_bytes())
        data['carries']['d01'].pop('radiation_diagnostics_schema_version')
        data['carries']['d01']['extra_fields']['radiation_diagnostics']=data['carries']['d01']['extra_fields']['radiation_diagnostics']._replace(cloud_fraction=None)
        blob=pickle.dumps(data);(p/'snapshot.pkl').write_bytes(blob)
        manifest=json.loads((p/'manifest.json').read_text());manifest.update(payload_bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest())
        (p/'manifest.json').write_text(json.dumps(manifest))
        read=lambda:store.read(p,device=False)
    with pytest.raises(ValueError,match='radiation diagnostics schema.*E78'):read()
