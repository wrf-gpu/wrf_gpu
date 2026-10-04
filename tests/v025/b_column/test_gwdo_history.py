"""GWDO diagnostic routing: producer -> State -> RK -> history/restart.

These are seam tests, not a new physics fidelity claim. Distinct profile values
make dropped leaves, wrong orientation, and substituted momentum sources fail.
"""
import pickle
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from netCDF4 import Dataset

from gpuwrf.contracts import state as state_module
from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import State
from gpuwrf.coupling import physics_couplers as pc
from gpuwrf.physics.gwd_gwdo import GWDOTendency
from gpuwrf.runtime import operational_mode as op
from gpuwrf.runtime.operational_state import initial_operational_carry
from gpuwrf.io import restart
from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry, read_wrfrst_carry


@pytest.fixture
def state_grid(monkeypatch):
    monkeypatch.setattr(state_module, '_gpu_device', lambda: jax.devices('cpu')[0])
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid).replace(theta=jnp.full((grid.nz,grid.ny,grid.nx),300.,jnp.float32),
        p=jnp.full((grid.nz,grid.ny,grid.nx),90000.,jnp.float64),
        ph=jnp.broadcast_to(jnp.arange(grid.nz+1)[:,None,None]*981.,(grid.nz+1,grid.ny,grid.nx)),
        mu=jnp.full((grid.ny,grid.nx),80000.,jnp.float64))
    return state,grid


def profiles(shape):
    x=jnp.arange(np.prod(shape),dtype=jnp.float32).reshape(shape)/1024+0.125
    return x,-x-0.25


def legacy_producer(monkeypatch, shape):
    monkeypatch.setenv('GPUWRF_GWDO_NATIVE_REAL','0')
    x,y=profiles(shape)
    sx,sy=profiles(shape[1:])
    col=lambda value: jnp.moveaxis(value,0,-1).reshape((-1,shape[0]))
    zero=jnp.zeros((shape[1]*shape[2],shape[0]))
    # DTAU deliberately differs from momentum: catches using the wrong field.
    out=GWDOTendency(zero,zero,col(x),col(y),sx.reshape(-1),sy.reshape(-1))
    calls=[]
    monkeypatch.setattr(pc,'gwdo_columns',lambda *a: calls.append(a) or out)
    return x,y,calls


def test_legacy_diagnostic_fields_and_orientation(state_grid,monkeypatch):
    state,grid=state_grid
    x,y,calls=legacy_producer(monkeypatch,state.theta.shape)
    ru,rv,dx,dy,sx,sy=pc.gwdo_tendencies(state,54.,object(),None,return_diagnostics=True)
    np.testing.assert_array_equal(dx,x); np.testing.assert_array_equal(dy,y)
    ex,ey=profiles(state.theta.shape[1:])
    np.testing.assert_array_equal(sx,ex); np.testing.assert_array_equal(sy,ey)
    assert np.all(np.asarray(ru)==0) and np.all(np.asarray(rv)==0)
    assert len(calls)==1


def test_native_diagnostics_use_single_kernel(state_grid,monkeypatch):
    from gpuwrf.kernels import phys_gwdo_column as kernel
    state,grid=state_grid; x,y=profiles(state.theta.shape); calls=[]
    sx,sy=profiles(state.theta.shape[1:])
    monkeypatch.setenv('GPUWRF_GWDO_NATIVE_REAL','1')
    def native(*a,**kw):
        assert kw.get('return_surface_stress') is True
        calls.append(a)
        return x,y,sx,sy
    monkeypatch.setattr(kernel,'gwdo_tendencies_from_state',native)
    out=pc.gwdo_tendencies(state,54.,object(),grid,return_diagnostics=True)
    assert out[0] is out[2] and out[1] is out[3] and len(calls)==1
    assert out[4] is sx and out[5] is sy


@pytest.mark.parametrize('force_fp64',[False,True])
def test_initial_carry_seeds_only_active_gwdo(state_grid,force_fp64):
    state,grid=state_grid
    nl=op.OperationalNamelist(grid=grid,tendencies=None,metrics=grid.metrics,dt_s=54,
        acoustic_substeps=4,run_physics=True,mp_physics=0,cu_physics=0,
        ra_sw_physics=0,ra_lw_physics=0,bl_pbl_physics=0,sf_sfclay_physics=0,
        gwd_opt=1,gwdo_statics=object(),force_fp64=force_fp64)
    active=op._initial_carry_for_run(state,nl)
    for name in ('dtaux3d','dtauy3d','dusfcg','dvsfcg'):
        value=getattr(active.state,name)
        assert value is not None
        shape=state.theta.shape[1:] if name in ('dusfcg','dvsfcg') else state.theta.shape
        assert value.shape==shape and value.dtype==jnp.float32
        assert np.all(np.asarray(value)==0)
    inactive=op._initial_carry_for_run(state,replace(nl,gwd_opt=0))
    assert all(getattr(inactive.state,name) is None for name in ('dtaux3d','dtauy3d','dusfcg','dvsfcg'))


def test_source_leaf_gwdo_diagnostics_survive_rk(state_grid,monkeypatch):
    state,grid=state_grid; x,y=profiles(state.theta.shape)
    sx,sy=profiles(state.theta.shape[1:])
    zero=jnp.zeros_like(state.theta)
    nl=op.OperationalNamelist(grid=grid,tendencies=None,metrics=grid.metrics,dt_s=54,
        acoustic_substeps=4,run_physics=True,mp_physics=0,cu_physics=0,
        ra_sw_physics=0,ra_lw_physics=0,bl_pbl_physics=5,sf_sfclay_physics=0,
        gwd_opt=1,gwdo_statics=object(),rad_rk_tendf=1,force_fp64=False)
    monkeypatch.setattr(op,'surface_adapter',lambda s,*a,**k:s)
    monkeypatch.setattr(op,'mynn_adapter_with_source_leaves',lambda s,*a,**k:
        SimpleNamespace(state=s,rthblten=zero,rqvblten=zero,rublten=zero,rvblten=zero))
    seen=[]
    def producer(*args,**kwargs):
        assert kwargs.get('return_diagnostics') is True
        seen.append(args[0])
        return zero,zero,x,y,sx,sy
    monkeypatch.setattr(op,'gwdo_tendencies',producer)
    carry=op._initial_carry_for_run(state,nl)
    forcing=op._physics_step_forcing(carry,nl,0.,run_radiation=False)
    assert len(seen)==1
    np.testing.assert_array_equal(forcing.state.dtaux3d,x)
    np.testing.assert_array_equal(forcing.state.dtauy3d,y)
    np.testing.assert_array_equal(forcing.state.dusfcg,sx)
    np.testing.assert_array_equal(forcing.state.dvsfcg,sy)
    after=op._apply_physics_non_dry_updates(carry.state,carry.state,forcing.state)
    np.testing.assert_array_equal(after.dtaux3d,x); np.testing.assert_array_equal(after.dtauy3d,y)
    np.testing.assert_array_equal(after.dusfcg,sx); np.testing.assert_array_equal(after.dvsfcg,sy)
    held=op._physics_step_forcing(carry.replace(state=after),replace(nl,run_physics=False),54.,run_radiation=False)
    np.testing.assert_array_equal(held.state.dtaux3d,x)
    np.testing.assert_array_equal(held.state.dusfcg,sx); np.testing.assert_array_equal(held.state.dvsfcg,sy)


def test_adapter_rk_actual_callback_transcribes_gwdo(tmp_path,monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from test_initial_history_wrf_t0 import _writer
    writer,stub=_writer(tmp_path,lambda *a,**k:{})
    grid=writer.bundles['d01'].grid
    monkeypatch.setattr(state_module,'_gpu_device',lambda:jax.devices('cpu')[0])
    state=State.zeros(grid).replace(theta=jnp.asarray(stub.theta),p=jnp.asarray(stub.p_total),
        ph=jnp.asarray(stub.ph_total),mu=jnp.asarray(stub.mu_total),qv=jnp.asarray(stub.qv))
    x,y,calls=legacy_producer(monkeypatch,state.theta.shape)
    produced=pc.gwdo_adapter(state,18.,object(),None)
    after=op._apply_physics_non_dry_updates(state,state,produced)
    writer._full_variable_set=True
    result=writer('d01',200,SimpleNamespace(state=after,radiation_diagnostics=None))
    with Dataset(result['wrfout']) as ds:
        np.testing.assert_array_equal(ds.variables['DTAUX3D'][0],x)
        np.testing.assert_array_equal(ds.variables['DTAUY3D'][0],y)
        sx,sy=profiles(state.theta.shape[1:])
        np.testing.assert_array_equal(ds.variables['DUSFCG'][0],sx)
        np.testing.assert_array_equal(ds.variables['DVSFCG'][0],sy)
    assert len(calls)==1


def test_gwdo_restart_roundtrip_and_old_schema_refused(state_grid,tmp_path):
    state,grid=state_grid; x,y=profiles(state.theta.shape)
    sx,sy=profiles(state.theta.shape[1:])
    state=state.replace(dtaux3d=x,dtauy3d=y,dusfcg=sx,dvsfcg=sy)
    carry=initial_operational_carry(state); path=tmp_path/'checkpoint.pkl'
    restart.write_restart(carry,{},grid,1,path)
    restored=restart.read_restart(path)[0]
    np.testing.assert_array_equal(restored.state.dtaux3d,x)
    np.testing.assert_array_equal(restored.state.dtauy3d,y)
    np.testing.assert_array_equal(restored.state.dusfcg,sx)
    np.testing.assert_array_equal(restored.state.dvsfcg,sy)
    with path.open('rb') as f: payload=pickle.load(f)
    payload['carry']['gwdo_diagnostics_schema_version']=1
    with path.open('wb') as f: pickle.dump(payload,f)
    with pytest.raises(ValueError,match='GWDO diagnostics schema'):
        restart.read_restart(path)


def test_gwdo_wrfrst_roundtrip_and_old_schema_refused(state_grid,tmp_path):
    state,grid=state_grid; x,y=profiles(state.theta.shape)
    sx,sy=profiles(state.theta.shape[1:])
    carry=initial_operational_carry(state.replace(dtaux3d=x,dtauy3d=y,dusfcg=sx,dvsfcg=sy))
    path=tmp_path/'wrfrst.nc'
    write_wrfrst_carry(carry,grid,{},path,valid_time='2026-07-25_18:00:54',run_start='2026-07-25_18:00:00',step_index=1)
    restored=read_wrfrst_carry(path)[0]
    np.testing.assert_array_equal(restored.state.dtaux3d,x)
    np.testing.assert_array_equal(restored.state.dtauy3d,y)
    np.testing.assert_array_equal(restored.state.dusfcg,sx)
    np.testing.assert_array_equal(restored.state.dvsfcg,sy)
    with Dataset(path,'a') as ds: ds.GPUWRF_GWDO_DIAGNOSTICS_SCHEMA_VERSION=1
    with pytest.raises(ValueError,match='GWDO diagnostics schema'):
        read_wrfrst_carry(path)
