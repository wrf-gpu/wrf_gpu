"""KF coupling lifetime and deferred consumption against WRF REAL literals."""
import importlib.util
import json
import os
import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling import kf_rk as K

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def finite(*arrays):
    assert all(np.isfinite(np.asarray(a)).all() for a in arrays)

@pytest.fixture(scope='module')
def original(tmp_path_factory):
    spec=importlib.util.spec_from_file_location('kf_rk_literal',HERE/'kf_rk_literal.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,module.build(tmp_path_factory.mktemp('pristine_kf_rk'))

def test_default_key_and_live_trace_split(monkeypatch):
    from gpuwrf.runtime.aot_cheap_key import global_trace_env_hash
    monkeypatch.delenv('GPUWRF_KF_TEND_RK_WRF',raising=False)
    assert not K.kf_tend_rk_enabled()
    off=global_trace_env_hash();monkeypatch.setenv('GPUWRF_KF_TEND_RK_WRF','1')
    assert K.kf_tend_rk_enabled() and global_trace_env_hash()!=off

def test_actual_kf_prepare_does_not_euler_or_clear_early(monkeypatch):
    from gpuwrf.runtime import operational_mode as op
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.diagnostics import census
    spec=importlib.util.spec_from_file_location('rk_state_fixture',ROOT/'tests/test_rrtm_lw_operational_wiring.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    grid=fixture._grid(ny=3,nx=4,nz=8);state=fixture._state(grid)
    rates=tuple(jnp.full_like(state.theta,(i+1)*1e-5) for i in range(6))+(jnp.full_like(state.t_skin,.002),)
    carry=initial_operational_carry(state).replace(cumulus_carry=(jnp.zeros_like(state.theta),jnp.full_like(state.t_skin,54.)),
        cumulus_tendencies=rates,census=census.initial_census())
    nml=SimpleNamespace(dt_s=54.,cumulus_cadence_steps=6,cudt_minutes=5.,grid=grid)
    monkeypatch.setattr(op,'kf_adapter',lambda st,dt,w,n,**kw:(kw['held_tendencies'],w,n))
    monkeypatch.setenv('GPUWRF_KF_TEND_RK_WRF','1')
    got,cy=jax.jit(lambda st,c:op._kf_cadence_step(st,c,nml,jnp.asarray(2)))(state,carry)
    for name in ('theta','qv','qc','qr','qi','qs','rainc_acc'):
        np.testing.assert_array_equal(getattr(got,name),getattr(state,name))
    np.testing.assert_array_equal(cy.cumulus_carry[1],carry.cumulus_carry[1])
    for a,b in zip(cy.cumulus_tendencies,rates):np.testing.assert_array_equal(a,b)

def test_cu_is_added_before_one_conversion_and_mask():
    from gpuwrf.runtime import operational_mode as op
    shape=(3,5,6)
    rr=np.float32(461.6)/np.float32(287.)
    theta=jnp.full(shape,300.,jnp.float64);qv=jnp.full(shape,.01,jnp.float64)
    mu=jnp.full((5,6),80000.,jnp.float64);metrics=SimpleNamespace(c1h=jnp.asarray([1.,.6,.2]),c2h=jnp.asarray([0.,300.,5000.]))
    ra=jnp.full(shape,-1e-4);bl=jnp.full(shape,2e-4);qb=jnp.full(shape,1e-6)
    cu0=jnp.full(shape,3e-4);qc0=jnp.full(shape,-2e-6)
    mass=metrics.c1h[:,None,None]*mu[None]+metrics.c2h[:,None,None]
    cu=(K.kf_add_a2a(mass*cu0,specified_or_nested=True),K.kf_add_a2a(mass*qc0,specified_or_nested=True))
    dry=op._source_leaf_dry_tendencies(mu,ra,bl,qb,None,None,qv,theta,metrics,jnp.float64,kf_tendf=cu)
    # Independent WRF-literal order; this algebra test is regression only.
    expected=(1.+(461.6/287.)*qv)*((mass*ra+mass*bl)+cu[0])+(461.6/287.)*theta/(1.+(461.6/287.)*qv)*(mass*qb+cu[1])
    finite(dry.t_tendf,expected)
    np.testing.assert_allclose(dry.t_tendf,expected,rtol=3e-15,atol=1e-12)

def test_actual_driver_inlet_keeps_time_n_before_pbl(monkeypatch):
    from gpuwrf.runtime import operational_mode as op
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.diagnostics import census
    spec=importlib.util.spec_from_file_location('inlet_fixture',ROOT/'tests/test_rrtm_lw_operational_wiring.py')
    f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    grid=f._grid(ny=4,nx=5,nz=8);state=f._state(grid)
    nml=dataclasses.replace(f._namelist(grid,ra_sw_physics=0,ra_lw_physics=0),
        run_physics=True,use_noahmp=False,mp_physics=0,cu_physics=1,bl_pbl_physics=5,
        sf_sfclay_physics=0,rad_rk_tendf=1)
    held=tuple(jnp.zeros_like(state.theta) for _ in range(6))+(jnp.zeros_like(state.t_skin),)
    carry=initial_operational_carry(state).replace(cumulus_carry=(jnp.zeros_like(state.theta),jnp.full_like(state.t_skin,-100.)),
        cumulus_tendencies=held,census=census.initial_census())
    def pbl(st,*args,**kwargs):
        zero=jnp.zeros_like(st.theta)
        return SimpleNamespace(state=st.replace(theta=st.theta+10.,qv=st.qv+.003),
            rthblten=zero,rqvblten=zero,rublten=zero,rvblten=zero)
    captured=[]
    class ReachedDriver(Exception):pass
    def driver(st,*args,**kwargs):
        captured.append(st)
        raise ReachedDriver
    monkeypatch.setattr(op,'mynn_adapter_with_source_leaves',pbl)
    monkeypatch.setattr(op,'kf_adapter',driver)
    monkeypatch.setenv('GPUWRF_KF_TEND_RK_WRF','1')
    # Stop at the REAL driver callback, before the shared scalar dependency.
    with pytest.raises(ReachedDriver):
        op._physics_step_forcing(carry,nml,jnp.asarray(0.),run_radiation=False)
    assert captured
    np.testing.assert_array_equal(captured[0].qv,state.qv)
    np.testing.assert_array_equal(captured[0].theta,state.theta)

@pytest.mark.parametrize('joint',[False,True])
def test_actual_prod_rk_dispatch_d1_and_composed_keys(monkeypatch,joint):
    from gpuwrf.runtime import operational_mode as op
    from gpuwrf.runtime.domain_tree import DomainTree
    sys.path.insert(0,str(ROOT/'tests/v025/b_core'))
    from prod_inputs import prod_domains
    hierarchy,bundles,_,_,_,carries=prod_domains()
    tree=DomainTree.from_domains(hierarchy,bundles,feedback_enabled=False)
    nml=tree.domains['d01'].namelist;carry=carries['d01']
    events=[];records={}
    def watch(owner,name,event):
        original=getattr(owner,name)
        def wrapped(*args,**kw):
            result=original(*args,**kw)
            events.append(event);records.setdefault(event,[]).append((args,kw,result))
            return result
        monkeypatch.setattr(owner,name,wrapped)
    watch(K,'couple_kf_rates','kf_couple')
    watch(K,'finish_kf_rates','kf_finish')
    watch(op,'_source_leaf_dry_tendencies','theta_source')
    watch(op,'_rk_update_scalar_pd','pd')
    watch(op,'_apply_moisture_large_step','scalar_consume')
    watch(op,'_rk_scan_step','rk_sources')
    watch(op,'_kf_cadence_step','driver')
    watch(op,'mynn_adapter_with_source_leaves','pbl_driver')
    monkeypatch.setenv('GPUWRF_KF_TEND_RK_WRF','1')
    monkeypatch.setenv('GPUWRF_PHYS_TEND_RK_WRF',str(int(joint)))
    shapes=jax.tree.map(lambda v:jax.ShapeDtypeStruct(v.shape,v.dtype),carry)
    clock=op.build_clock_base(nml)
    jax.clear_caches()
    try:
        got=jax.eval_shape(lambda c:op._advance_chunk_fori(c,nml,jnp.asarray(1,jnp.int32),clock,n_steps=1,
            cadence=int(nml.radiation_cadence_steps)),shapes)
    finally:jax.clear_caches()
    assert events.count('kf_couple')==1 and events.count('kf_finish')==1
    assert events.count('scalar_consume')==3 and events.count('pd')==1
    assert events.index('kf_finish')>max(i for i,e in enumerate(events) if e=='scalar_consume')
    assert records['theta_source'][0][1]['kf_tendf'] is not None
    assert set(K.KF_MOIST_SPECIES)<=set(records['rk_sources'][0][1]['physics_tendencies'].moist_tendf)
    driver=records['driver'][0][0][0];pbl_entry=records['pbl_driver'][0][0][0]
    assert driver.theta is pbl_entry.theta and driver.qv is pbl_entry.qv
    assert [str(x) for x in jax.tree.leaves(got)]==[str(x) for x in jax.tree.leaves(shapes)]

@pytest.mark.parametrize('nca',[54.,108.])
@pytest.mark.parametrize('joint',[False,True])
def test_actual_boundary_last_nca_consume_then_finish_and_restart(monkeypatch,tmp_path,nca,joint):
    from gpuwrf.runtime import operational_mode as op
    from gpuwrf.runtime.operational_state import initial_operational_carry
    from gpuwrf.diagnostics import census
    from gpuwrf.io.restart import write_restart,read_restart
    from gpuwrf.io.wrfrst_netcdf import write_wrfrst_carry,read_wrfrst_carry
    from gpuwrf.runtime.restart_store import RestartStore
    spec=importlib.util.spec_from_file_location('last_step_fixture',ROOT/'tests/test_rrtm_lw_operational_wiring.py')
    f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
    grid=f._grid(ny=5,nx=6,nz=8);state=f._state(grid)
    nml=dataclasses.replace(f._namelist(grid,ra_sw_physics=0,ra_lw_physics=0),
        run_physics=True,use_noahmp=False,mp_physics=0,cu_physics=1,bl_pbl_physics=0,
        sf_sfclay_physics=0,rad_rk_tendf=1,run_boundary=False,dt_s=54.,cumulus_cadence_steps=6)
    rates=tuple(jnp.full_like(state.theta,(i+1)*1e-8) for i in range(6))+(jnp.full_like(state.t_skin,.002),)
    carry=initial_operational_carry(state).replace(cumulus_carry=(jnp.zeros_like(state.theta),jnp.full_like(state.t_skin,nca)),
        cumulus_tendencies=rates,rthraten=jnp.full_like(state.theta,3e-4),census=census.initial_census())
    monkeypatch.setenv('GPUWRF_KF_TEND_RK_WRF','1')
    monkeypatch.setenv('GPUWRF_PHYS_TEND_RK_WRF',str(int(joint)))
    monkeypatch.setattr(op,'kf_adapter',lambda st,dt,w,n,**kw:(kw['held_tendencies'],w,n))
    final_mu=state.mu_total+100.
    consumed=[]
    def rk(c,nml,**kw):
        source=kw['physics_tendencies'].moist_tendf
        assert set(K.KF_MOIST_SPECIES)<=set(source)
        np.testing.assert_array_equal(c.cumulus_carry[1],nca)
        assert np.asarray(c.cumulus_tendencies[0]).any()
        origin=c.state
        for dt in (18.,27.,54.):
            stage=origin.replace(mu_total=final_mu)
            stage=op._apply_moisture_large_step(stage,origin,
                q_tendencies=tuple(source[n] for n in K.KF_MOIST_SPECIES),dt_rk=dt,
                metrics=nml.metrics,species=K.KF_MOIST_SPECIES)
            consumed.append(dt)
        return c.replace(state=stage)
    monkeypatch.setattr(op,'_rk_scan_step',rk)
    got=op._physics_boundary_step(carry,nml,jnp.asarray(2),run_radiation=False)
    assert consumed==[18.,27.,54.]
    np.testing.assert_array_equal(got.cumulus_carry[1],nca-54.)
    np.testing.assert_allclose(got.state.rainc_acc,np.asarray(state.rainc_acc)+54.*.002,rtol=1e-15,atol=1e-15)
    mass0=grid.metrics.c1h[:,None,None]*state.mu_total[None]+grid.metrics.c2h[:,None,None]
    massf=grid.metrics.c1h[:,None,None]*final_mu[None]+grid.metrics.c2h[:,None,None]
    for rate,old in zip(got.cumulus_tendencies[:6],rates[:6]):
        expected=np.zeros_like(old) if nca<=54. else mass0*old/massf
        np.testing.assert_allclose(rate,expected,rtol=2e-15,atol=1e-15)
    expected_rad=mass0*carry.rthraten/massf if joint else carry.rthraten
    np.testing.assert_allclose(got.rthraten,expected_rad,rtol=2e-15,atol=1e-15)
    for name,rate in zip(K.KF_MOIST_SPECIES,rates[1:6]):
        expected=(mass0*getattr(state,name)+54.*mass0*rate)/massf
        # The public endpoint enforces the configured carry dtype after RK.
        expected=np.asarray(expected,dtype=np.asarray(getattr(got.state,name)).dtype)
        np.testing.assert_array_equal(getattr(got.state,name),expected)
    def exact(a,b):
        aa,ta=jax.tree.flatten(a);bb,tb=jax.tree.flatten(b);assert ta==tb
        for x,y in zip(aa,bb,strict=True):
            xx,yy=np.asarray(x),np.asarray(y);finite(xx,yy)
            assert xx.dtype==yy.dtype and xx.shape==yy.shape and xx.tobytes()==yy.tobytes()
    write_restart(got,nml,grid,2,tmp_path/'kf.pkl')
    exact(got,read_restart(tmp_path/'kf.pkl')[0])
    write_wrfrst_carry(got,grid,{},tmp_path/'kf.nc',valid_time='2026-02-28_00:01:48',run_start='2026-02-28_00:00:00',step_index=2)
    exact(got,read_wrfrst_carry(tmp_path/'kf.nc')[0])
    store=RestartStore(tmp_path/'store',10**8,2,0)
    generation,_=store.save({'d01':got},{'d01':2},{'d01':54.},{'KF_TEND_RK_WRF':True})
    exact(got,store.read(generation)[0]['carries']['d01'])

@pytest.mark.parametrize('nca',[27.,54.,81.,108.,-24.])
def test_two_steps_changing_mass_last_consumption_and_hold(original,nca):
    module,fn=original
    # Positive/negative KF profiles exercise each moist field, no self-compare.
    shape=(4,2,3);x=np.arange(24,dtype=np.float32).reshape(shape)+1
    rates=tuple(np.float32((i+1)*1e-8)*x for i in range(6))+(np.full((2,3),.001,np.float32),)
    c1=np.asarray([1.,.7,.3,0.],np.float32);c2=np.asarray([0.,100.,200.,5000.],np.float32)
    metrics=SimpleNamespace(c1h=jnp.asarray(c1),c2h=jnp.asarray(c2))
    mu=np.asarray([[80000.,80500.,81000.],[79000.,79500.,82000.]],np.float32)
    nc=np.full((2,3),nca,np.float32)
    for index in range(2):
        new_mu=mu+np.asarray([[12.,-23.,41.],[-17.,36.,-8.]],np.float32)
        expected_coupled,expected_held,expected_nca,expected_rain=module.run(fn,rates,c1,c2,mu,new_mu,nc,54.)
        coupled=jax.jit(lambda rr,m:K.couple_kf_rates(rr,m,metrics))(rates,mu)
        held,nc_out,rain=jax.jit(lambda rr,n,m:K.finish_kf_rates(rr,n,m,metrics,54.))(coupled,nc,new_mu)
        finite(*coupled,*held,nc_out,rain,*expected_coupled,*expected_held,expected_nca,expected_rain)
        for a,b in zip(coupled[:6],expected_coupled):np.testing.assert_allclose(a,b,rtol=3e-7,atol=1e-10)
        for a,b in zip(held,expected_held):np.testing.assert_allclose(a,b,rtol=3e-7,atol=1e-12)
        np.testing.assert_array_equal(nc_out,expected_nca)
        np.testing.assert_allclose(rain,expected_rain,rtol=2e-7,atol=0)
        # Last active rates were consumed (coupled above) before their clear.
        if index==0 and 0<nca<81:assert np.any(np.asarray(coupled[0])!=0) and not np.any(np.asarray(held[0]))
        rates=tuple(np.asarray(a) for a in held);nc=np.asarray(nc_out);mu=new_mu

def test_add_a2a_ring_and_scalar_map_ownership():
    a=jnp.arange(2*5*6,dtype=jnp.float32).reshape(2,5,6)+1
    out=np.asarray(K.kf_add_a2a(a,specified_or_nested=True))
    assert not out[:,0,:].any() and not out[:,-1,:].any()
    assert not out[:,:,0].any() and not out[:,:,-1].any()
    np.testing.assert_array_equal(out[:,1:-1,1:-1],np.asarray(a)[:,1:-1,1:-1])
    periodic=K.kf_add_a2a(a,specified_or_nested=True,periodic_x=True)
    np.testing.assert_array_equal(np.asarray(periodic)[:,1:-1],np.asarray(a)[:,1:-1])
    scalar=K.kf_scalar_tendencies((a,)*6+(jnp.ones((5,6)),),specified_or_nested=True)
    assert tuple(scalar)==K.KF_MOIST_SPECIES
    for value in scalar.values():np.testing.assert_array_equal(value,out)

def test_original_p0227_real_carry_mass_sequence(original,tmp_path):
    folder=os.environ.get('GPUWRF_KF_RK_REAL_CASE')
    if folder is None:pytest.skip('requires original P0227 restart operands')
    from netCDF4 import Dataset
    module,fn=original
    folder=Path(folder)
    stamps=['2026-02-28_12:00:00','2026-03-01_00:00:00','2026-03-01_12:00:00']
    masses=[];receipts=[]
    for stamp in stamps:
        path=folder/('wrfrst_d01_'+stamp)
        with Dataset(path) as ds:
            ds.set_auto_mask(False)
            masses.append(np.asarray(ds['MU_2'][0],np.float32)+np.asarray(ds['MUB'][0],np.float32))
            if stamp==stamps[0]:
                names=('RTHCUTEN','RQVCUTEN','RQCCUTEN','RQRCUTEN','RQICUTEN','RQSCUTEN','PRATEC')
                rates=tuple(np.asarray(ds[n][0],np.float32) for n in names)
                nc=np.asarray(ds['NCA'][0],np.float32)
                c1=np.asarray(ds['C1H'][0],np.float32);c2=np.asarray(ds['C2H'][0],np.float32)
                dt=float(ds.DT)
            receipts.append(dict(path=str(path),mtime_ns=path.stat().st_mtime_ns,bytes=path.stat().st_size))
    metrics=SimpleNamespace(c1h=jnp.asarray(c1),c2h=jnp.asarray(c2))
    assert np.any(rates[0]!=0) and np.any(nc>0)
    errors=[]
    for mu,muts in zip(masses[:-1],masses[1:]):
        cc,hh,nn,rr=module.run(fn,rates,c1,c2,mu,muts,nc,dt)
        coupled=jax.jit(lambda r,m:K.couple_kf_rates(r,m,metrics))(rates,mu)
        held,nout,rain=jax.jit(lambda r,n,m:K.finish_kf_rates(r,n,m,metrics,dt))(coupled,nc,muts)
        finite(*coupled,*held,nout,rain,*cc,*hh,nn,rr)
        for a,b in zip(coupled[:6],cc):np.testing.assert_allclose(a,b,rtol=4e-7,atol=1e-9)
        for a,b in zip(held,hh):np.testing.assert_allclose(a,b,rtol=4e-7,atol=1e-12)
        np.testing.assert_array_equal(nout,nn)
        np.testing.assert_allclose(rain,rr,rtol=2e-7,atol=0)
        errors.append(max(float(np.max(np.abs(np.asarray(a)-b))) for a,b in zip(coupled[:6],cc)))
        rates=tuple(np.asarray(a) for a in held);nc=np.asarray(nout)
    (tmp_path/'REAL_RECEIPT.json').write_text(json.dumps(dict(primary=receipts,max_coupling_abs=errors,
        geometry=[44,70,120],scope='controlled two-step KF lifetime using real source rates/NCA and three real mass fields; NOT a coupled forecast'),indent=2)+'\n')

@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_finite_guard_rejects_input_and_truth(bad):
    with pytest.raises(AssertionError):finite(np.asarray([bad]),np.asarray([1.]))
    with pytest.raises(AssertionError):finite(np.asarray([1.]),np.asarray([bad]))
