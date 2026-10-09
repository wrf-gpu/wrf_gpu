"""BP109: independent pi_phy in MYNN's operational surface boundary."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import surface_layer as SL, noahmp_coupler as NC
from gpuwrf.coupling import physics_couplers as C, noahmp_surface_hook as H
from gpuwrf.physics.fp32 import surface_layer_real as REAL

FLAG = 'GPUWRF_MYNN_PHY_EXNER'
HERE = Path(__file__).parent
DATA = json.loads((HERE/'fixtures/mynn_surface_exner_wrf.json').read_text())


def finite(values):
    for name, x in values.items():
        assert np.isfinite(np.asarray(x)).all(), name+' nonfinite'


def inputs():
    rows=DATA['records']; names=['hfx','qfx','rho','qv','tsk','p_hyd','exner']
    arrays={n:np.asarray([r[n] for r in rows],np.float32) for n in names}
    ref=np.asarray([r['reference_bits'] for r in rows],np.int32).view(np.float32)
    finite(dict(arrays,reference=ref));return arrays,ref


@pytest.fixture(autouse=True)
def cache():
    jax.clear_caches();yield;jax.clear_caches()


def test_190_original_warm_fluxes(monkeypatch):
    a,ref=inputs();monkeypatch.setenv(FLAG,'1')
    call=lambda:SL.mynn_driver_surface_fluxes(*[jnp.asarray(a[n]) for n in ['hfx','qfx','rho','qv','tsk','p_hyd']],exner=jnp.asarray(a['exner']))
    with jax.enable_x64(False):
        out=jax.jit(call)()
    got=np.stack([np.asarray(x) for x in out],axis=1);finite(dict(candidate=got,reference=ref))
    ulp=np.abs(got.astype(float)-ref.astype(float))/np.maximum(np.spacing(abs(ref)).astype(float),np.finfo(np.float32).tiny)
    assert np.max(ulp)<=DATA['native_tolerance_ulps'], float(np.max(ulp))


@pytest.mark.parametrize('dtype',[jnp.float32,jnp.float64])
def test_flag_off_ignores_supplied_pi_exactly(monkeypatch,dtype):
    a,_=inputs();monkeypatch.setenv(FLAG,'0')
    values=[jnp.asarray(a[n],dtype) for n in ['hfx','qfx','rho','qv','tsk','p_hyd']]
    before=SL.mynn_driver_surface_fluxes(*values)
    after=SL.mynn_driver_surface_fluxes(*values,exner=jnp.asarray(a['exner'],dtype))
    for i,(old,new) in enumerate(zip(before,after)):
        finite({f'old{i}':old,f'new{i}':new});np.testing.assert_array_equal(new,old)


def scene():
    spec=importlib.util.spec_from_file_location('bp109_noah_scene',HERE.parents[1]/'test_noahmp_coupler.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    assert mod.HAVE_TABLES,'pristine tables required';return mod._build()


@pytest.mark.parametrize('producer',['surface','noah'])
def test_actual_surface_producers_pass_independent_pi(monkeypatch,producer):
    state,land,static,rad,clock=scene();pi=jnp.full_like(state.p,.997)
    state=state._replace(exner=pi) if hasattr(state,'_replace') and 'exner' in state._fields else SimpleNamespace(**state._asdict(),exner=pi) if hasattr(state,'_asdict') else SimpleNamespace(**vars(state),exner=pi)
    monkeypatch.setenv(FLAG,'1');monkeypatch.setenv('GPUWRF_MYNN_FLTV_WRF','1')
    module=SL if producer=='surface' else NC;fn=module.mynn_driver_surface_fluxes;seen=[]
    def capture(*args,**kw):
        finite(dict(kw));assert 'exner' in kw
        np.testing.assert_array_equal(np.asarray(kw['exner']),np.asarray(pi)[...,0]);seen.append(1)
        return fn(*args,**kw)
    monkeypatch.setattr(module,'mynn_driver_surface_fluxes',capture)
    if producer=='surface':SL.surface_layer_with_diagnostics(state)
    else:
        # Logical producer-only control; numeric fidelity is the 190 original
        # column primitive above. Stop at the real blended-flux call.
        class Captured(Exception):pass
        def stop(*args,**kw):capture(*args,**kw);raise Captured
        monkeypatch.setattr(module,'mynn_driver_surface_fluxes',stop)
        with pytest.raises(Captured):NC.noahmp_surface_adapter(state,land,static,radiation=rad,clock=clock,dt=90.)
    assert seen


@pytest.mark.parametrize('producer',['surface_view','noah_view'])
def test_actual_views_derive_pi_from_dynamic_pressure(monkeypatch,producer):
    # Actual builders with controlled geometry callbacks isolate pressure
    # authority; all meteorological arrays are from the real warm fixture.
    a,_=inputs();n=4;p=jnp.asarray(a['p_hyd'][:n])[None,None,:]
    # Recover nonhyd pressure from the saved independent Exner, for a
    # distinctive actual value at this logical builder boundary.
    pdyn=(jnp.asarray(a['exner'][:n])**jnp.float32(3.5))*jnp.float32(1e5)
    pdyn=pdyn[None,None,:];pdyn=jnp.concatenate([pdyn,pdyn*.95],axis=0)
    theta=jnp.full_like(pdyn,300);qv=jnp.full_like(pdyn,.008)
    st=SimpleNamespace(p=pdyn,theta=theta,qv=qv,qc=jnp.zeros_like(qv),u=jnp.zeros((2,1,n+1)),v=jnp.zeros((2,2,n)),t_skin=jnp.full((1,n),295.),soil_moisture=jnp.ones((1,n))*.3,xland=jnp.ones((1,n))*2,lakemask=jnp.zeros((1,n)),mavail=jnp.ones((1,n)),roughness_m=jnp.ones((1,n))*.001,ustar=jnp.ones((1,n))*.2)
    module=C if producer=='surface_view' else H
    monkeypatch.setenv(FLAG,'1')
    monkeypatch.setattr(module,'_wrf_hydrostatic_pressure_from_state',lambda *args,**kw:(jnp.concatenate([p,p*.95],axis=0),p[0]))
    monkeypatch.setattr(module,'_wrf_phy_prep_rho_from_state',lambda *args,**kw:jnp.ones_like(pdyn))
    monkeypatch.setattr(module,'_surface_dz_from_state',lambda *args,**kw:jnp.ones((1,n,2))*60)
    monkeypatch.setattr(module,'_mynn_exner_from_pressure',lambda pressure:pressure/100000.)
    if producer=='noah_view':monkeypatch.setattr(module,'_mynn_column_uses_wrf_phy_prep',lambda grid:True)
    grid=SimpleNamespace(metrics=object(),projection=SimpleNamespace(dx_m=9000.))
    view=module._surface_column_view(st,grid) if producer=='surface_view' else module._build_column_view(st,grid)
    assert view.exner is not None
    expected=jnp.moveaxis(jnp.asarray(pdyn,view.exner.dtype)/100000.,0,-1)
    finite(dict(candidate=view.exner,reference=expected))
    np.testing.assert_array_equal(np.asarray(view.exner),np.asarray(expected))


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
@pytest.mark.parametrize('field',['hfx','qfx','rho','qv','tsk','p_hyd','exner','reference'])
def test_original_gate_rejects_nonfinite(monkeypatch,bad,field):
    a,ref=inputs();a['reference']=ref;a[field]=a[field].copy();a[field].flat[0]=bad
    with pytest.raises(AssertionError,match='nonfinite'):finite(a)
