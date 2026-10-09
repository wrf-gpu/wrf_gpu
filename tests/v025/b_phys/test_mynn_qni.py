"""Original REAL4 QNI scalar solve and passive plume transport, real Swiss ice."""
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import mynn_pbl as P, mynn_edmf as E
from gpuwrf.kernels import phys_mynn_plume as L, phys_mynn_columns as N
from gpuwrf.kernels.phys_mynn_ni import mix_ice_number

DATA=json.loads((Path(__file__).parent/'fixtures/mynn_qni_wrf.json').read_text())
FLOORS=json.loads((Path(__file__).parent/'fixtures/mynn_qni_floors_wrf.json').read_text())


@pytest.fixture(autouse=True)
def clear_flag_traces():
    jax.clear_caches()
    yield
    jax.clear_caches()


def finite(*groups):
    for role,fields in groups:
        for n,x in fields.items():assert np.isfinite(x).all(),f'{role} {n} nonfinite'


def scalar_gate(candidate,reference,inputs):
    finite(('candidate',{'ni':candidate}),('reference',{'ni':reference}),('input',inputs))
    # Integer exponent decoding avoids host FTZ corrupting np.spacing on
    # subnormals. Manager 07:43Z / E117: retain every value, with the explicit
    # IEEE REAL32 minimum-normal floor as the precision allowance.
    exponent=(np.asarray(reference,np.float32).view(np.uint32)>>23)&255
    spacing=np.ldexp(np.ones(reference.shape,np.float64),np.maximum(exponent.astype(np.int32),1)-127-23)
    bound=np.maximum(8*spacing,np.finfo(np.float32).tiny)
    assert np.all(np.abs(candidate.astype(float)-reference.astype(float))<=bound), 'original QNI solve >8 ULP/REAL32 tiny'


def mf_gate(candidate,reference,inputs):
    finite(('candidate',{'s_awqni':candidate}),('reference',{'s_awqni':reference}),('input',inputs))
    for i in range(len(reference)):
        scale=float(abs(reference[i]).max());bound=scale*.05 if scale else 1e-12
        assert abs(candidate[i]-reference[i]).max()<=bound, 'original passive QNI flux differs'


def state_from_row(row,dtype=jnp.float32):
    v=np.asarray(row['input']['values'],np.float32)
    names=('u','v','w','theta','qv','tke','p','rho','dz','qc','qi','qs','qsq')
    vals={n:jnp.asarray(v[:,:,i],dtype) for i,n in enumerate(names)};z=jnp.zeros_like(vals['qv'])
    return P.MynnPBLColumnState(**vals,km=z,kh=z,el=z,ni=jnp.asarray(row['input']['ni'],dtype),exner=jnp.asarray(row['input']['exner'],dtype))


@pytest.mark.parametrize('row',DATA['rows'],ids=lambda r:r['stem'])
@pytest.mark.parametrize('dtype',[jnp.float32,jnp.float64],ids=['native','retained'])
def test_pristine_scalar_number_solve(monkeypatch,row,dtype):
    monkeypatch.setenv('GPUWRF_MYNN_TRIDIAG_REAL','1')
    n={k:np.asarray(v,np.float32) for k,v in row['scalar'].items()};s=state_from_row(row,dtype)
    s=s.replace(rho=jnp.asarray(n['rho'],dtype),dz=jnp.asarray(n['dz'],dtype))
    finite(('input',n))
    def run(q,ib,ia,s,kh,aw,awn):return mix_ice_number(q,ib,ia,s,{'dfh':kh},row['dt'],{'s_aw':aw,'s_awqni':awn})
    with jax.enable_x64(dtype==jnp.float64):
        out=jax.jit(run)(jnp.asarray(n['qni'],dtype),jnp.asarray(n['sqi'],dtype),jnp.asarray(n['sqi2'],dtype),s,jnp.asarray(n['dfh'],dtype),jnp.asarray(n['s_aw1'],dtype),jnp.asarray(n['s_awqni1'],dtype))
    scalar_gate(np.asarray(out),n['after'],n)
    np.testing.assert_array_equal(np.asarray(out)[:,-1],jnp.asarray(n['after'],dtype)[:,-1])
    assert np.any(n['after']!=n['qni']), 'actual scalar witness inactive'


def mf_args(row,dtype):
    m={k:np.asarray(v,np.float32) for k,v in row['mf'].items()};finite(('input',m))
    args=[jnp.asarray(m[k],dtype) for k in ('qt1','qv1','qc1','u1','v1','w1','th1','thl1','thv1','tk1','qke1','p1','ex1','rho1','dz1','zw1')]
    vals=m['scalar'];sc={k:jnp.asarray(vals[:,i],dtype) for i,k in enumerate(('pblh','flt','fltv','flq','flqv','ts','psig_shcu','xland','dx','dt','ust'))}
    zw=m['zw1'];dz=m['dz1'];cloud=(np.maximum(m['qc1'],m['qc_bl1'])>1e-5)&(m['cldfra_bl1']>=.5)&((zw[:,:-1]+.5*dz)<=np.asarray(sc['pblh'])[:,None]+500);cloud[:,-1]=False
    first=cloud.argmax(axis=-1);cb=np.where(cloud.any(axis=-1),zw[np.arange(len(zw)),first],9000.).astype(np.float32)
    kw={k:sc[k] for k in ('ust','flt','fltv','flq','flqv','pblh','ts','xland','psig_shcu')};kw.update(dx=row['dx'],dt=row['dt'],cloud_base=jnp.asarray(cb,dtype),qni=jnp.asarray(m['qni1'],dtype))
    return args,kw,m


@pytest.mark.parametrize('row',DATA['rows'],ids=lambda r:r['stem'])
@pytest.mark.parametrize('entry',['native','retained'])
def test_pristine_passive_qni_plume_flux(monkeypatch,row,entry):
    monkeypatch.setenv('GPUWRF_MYNN_PLUME_SUMS','1');monkeypatch.setenv('GPUWRF_EDMF_FUSED_PLUME','0')
    dtype=jnp.float32 if entry=='native' else jnp.float64;args,kw,inputs=mf_args(row,dtype)
    if entry=='native':fn=lambda *a:L.dmp_mf_columns_native(*a,**kw,interpret=True)
    else:fn=lambda *a:E.dmp_mf_columns(*a,**kw)
    with jax.enable_x64(entry=='retained'):out=jax.jit(fn)(*args)
    finite(('candidate',{n:np.asarray(v) for n,v in out.items()}))
    mf_gate(np.asarray(out['s_awqni']),np.asarray(row['scalar']['s_awqni1'],np.float32),inputs)


def test_real_mf_witness_is_nonzero():
    assert sum(np.count_nonzero(np.asarray(r['scalar']['s_awqni1'])) for r in DATA['rows'])>=40


@pytest.mark.parametrize('case',FLOORS['cases'],ids=lambda c:c['kind'])
def test_original_number_floor(monkeypatch,case):
    monkeypatch.setenv('GPUWRF_MYNN_TRIDIAG_REAL','1')
    row=next(r for r in DATA['rows'] if r['stem']=='d02_h12')
    n={k:np.asarray(v,np.float32) for k,v in case['operands'].items()}
    s=state_from_row(row).replace(rho=jnp.asarray(n['rho']),dz=jnp.asarray(n['dz']))
    out=mix_ice_number(jnp.asarray(n['qni']),jnp.asarray(n['sqi']),jnp.asarray(n['sqi2']),s,
        {'dfh':jnp.asarray(n['dfh'])},case['dt'],
        {'s_aw':jnp.asarray(n['s_aw1']),'s_awqni':jnp.asarray(n['s_awqni1'])})
    scalar_gate(np.asarray(out),np.asarray(case['reference'],np.float32),n)
    expected=0. if case['kind']=='negative_preclip' else np.float32(1e-6)
    np.testing.assert_array_equal(np.asarray(out)[:,-1],expected)


def test_active_mass_flux_requires_actual_number_flux():
    row=DATA['rows'][0];s=state_from_row(row)
    with pytest.raises(KeyError,match='s_awqni'):
        mix_ice_number(s.ni,s.qi,s.qi,s,{'dfh':jnp.zeros_like(s.qv)},row['dt'],
            {'s_aw':jnp.zeros(s.qv.shape[:-1]+(45,),s.qv.dtype)})


@pytest.mark.parametrize('entry',['native','retained'])
def test_whole_entry_persists_number_result(monkeypatch,entry):
    # Wiring regression: actual complete Swiss column, with the unrelated
    # expensive turbulence computations replaced by controlled finite results.
    row=DATA['rows'][0];s=state_from_row(row);z=jnp.zeros_like(s.qv)
    monkeypatch.setattr(P,'_MYNN_SGS_CLOUD',False)
    monkeypatch.setattr(P,'_mym_turbulence',lambda *a,**kw:{'el':s.el,'pblh':z[...,0]})
    monkeypatch.setattr(P,'_mym_predict_qke',lambda *a,**kw:(2*s.tke,z,z,z))
    monkeypatch.setattr(P,'_retrieve_exchange_coeffs',lambda *a,**kw:(z,z))
    changed=s.ni+jnp.asarray(7.,s.ni.dtype)
    monkeypatch.setattr(P,'_apply_mean_tendencies_with_clouds',lambda *a,**kw:
        (s.u,s.v,s.theta,s.qv,s.qc,s.qi,changed))
    fn=N._advance_native if entry=='native' else P._step_mynn_pbl_impl_with_pblh
    out,_=fn(s,row['dt'],False,edmf=False)
    finite(('candidate',{'ni':np.asarray(out.ni)}))
    np.testing.assert_array_equal(out.ni,changed)


@pytest.mark.parametrize('gate', [scalar_gate,mf_gate],ids=['scalar','plume'])
@pytest.mark.parametrize('role',['candidate','reference','input'])
@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_gate_rejects_nonfinite(gate,role,bad):
    a=np.ones((2,44),np.float32);b=a.copy();inputs={'rho':a.copy()}
    (a if role=='candidate' else b if role=='reference' else inputs['rho']).flat[0]=bad
    with pytest.raises(AssertionError,match='nonfinite'):gate(a,b,inputs)


INPUT_FIELDS=[('scalar',n) for n in DATA['rows'][0]['scalar']]+[('mf',n) for n in DATA['rows'][0]['mf']]
@pytest.mark.parametrize('kind,field',INPUT_FIELDS)
@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_gate_checks_every_original_operand(kind,field,bad):
    row=DATA['rows'][0];inputs={n:np.asarray(v,np.float32).copy() for n,v in row[kind].items()}
    inputs[field].flat[0]=bad;a=np.ones((2,44),np.float32)
    with pytest.raises(AssertionError,match='input .* nonfinite'):
        (scalar_gate if kind=='scalar' else mf_gate)(a,a,inputs)


def test_optional_number_survives_tree_and_cold_none():
    s=state_from_row(DATA['rows'][0])
    got=jax.tree.map(lambda x:x,s)
    np.testing.assert_array_equal(got.ni,s.ni)
    assert got==s and hash(got)==hash(s)
    none=s.replace(ni=None);assert P._clip_state(none).ni is None


@pytest.mark.parametrize('entry',['native','retained'])
def test_both_plume_suppliers_forward_raw_number(monkeypatch,entry):
    from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
    row=DATA['rows'][1];s=state_from_row(row);ff=np.asarray(row['input']['flux'],np.float32);B=len(ff)
    f=SurfaceFluxes(ustar=jnp.asarray(ff[:,0]),theta_flux=jnp.asarray(ff[:,1]),qv_flux=jnp.asarray(ff[:,2]),fltv=jnp.asarray(ff[:,3]),xland=jnp.asarray(ff[:,4]),t_skin=jnp.asarray(ff[:,5]),rhosfc=jnp.asarray(ff[:,6]),tau_u=jnp.zeros(B,jnp.float32),tau_v=jnp.zeros(B,jnp.float32))
    class Stop(Exception):pass
    def capture(*args,**kw):
        assert 'qni' in kw,'raw ice-number missing from DMP call'
        np.testing.assert_array_equal(np.asarray(kw['qni']),np.asarray(s.ni))
        raise Stop()
    monkeypatch.setattr(L,'dmp_mf_columns_native',capture);monkeypatch.setattr(E,'dmp_mf_columns',capture)
    with pytest.raises(Stop):(N._native_edmf_arrays if entry=='native' else P._edmf_arrays_from_state)(s,f,f.fltv,jnp.asarray(row['mf']['scalar'])[:,0],row['dt'],row['dx'])


def test_cloudmix_off_holds_optional_number(monkeypatch):
    s=state_from_row(DATA['rows'][0]);monkeypatch.setenv('GPUWRF_MYNN_CLOUDMIX','0')
    monkeypatch.setattr(P,'_apply_mean_tendencies_legacy',lambda *args,**kw:(s.u,s.v,s.theta,s.qv))
    result=P._apply_mean_tendencies_with_clouds(s,{},18.,None,None,None)
    assert len(result)==7
    np.testing.assert_array_equal(result[6],s.ni)


def test_number_solve_uses_ice_before_moisture_check(monkeypatch):
    from gpuwrf.kernels import phys_mynn_cloudmix as C,phys_mynn_ni as NI
    s=state_from_row(DATA['rows'][0]);B=s.u.shape[0];zero=jnp.zeros_like(s.u[...,0]);interfaces=jnp.zeros(s.u.shape[:-1]+(45,),s.u.dtype)
    from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
    f=SurfaceFluxes(ustar=zero,theta_flux=zero,qv_flux=zero,fltv=zero,t_skin=zero,rhosfc=jnp.ones_like(zero),xland=jnp.ones_like(zero),tau_u=zero,tau_v=zero)
    monkeypatch.setattr(P,'_apply_mean_tendencies_legacy',lambda *args,**kw:(s.u,s.v,s.theta,s.qv))
    monkeypatch.setattr(C,'_scalar',lambda x,*args,**kw:x)
    events=[]
    def number(ni,ib,ia,*args,**kw):events.append('number');np.testing.assert_array_equal(ia,ib);return ni
    def repair(qv,qc,qi,thl,*args):events.append('repair');return qv,qc,jnp.zeros_like(qi),thl
    monkeypatch.setattr(NI,'mix_ice_number',number);monkeypatch.setattr(C,'moisture_check',repair)
    C.apply_mean_cloudmix(s,{'dfh':jnp.zeros_like(s.qv)},18.,f,jnp.ones_like(zero),jnp.ones_like(zero),None)
    assert events==['number','repair'], 'WRF number floor must precede moisture_check'


def test_actual_coupler_number_supplier_and_writeback(monkeypatch):
    from types import SimpleNamespace
    from gpuwrf.coupling import physics_couplers as C
    row=DATA['rows'][0];column=state_from_row(row);B,K=column.qv.shape
    field=lambda a:jnp.asarray(a,jnp.float32).T.reshape(K,B,1)
    state=SimpleNamespace(theta=field(column.theta),p=field(column.p),qv=field(column.qv),
        qc=field(column.qc),qi=field(column.qi),qs=field(column.qs),qsq=field(column.qsq),
        qke=field(2*column.tke),Ni=field(column.ni),u=field(column.u),v=field(column.v),
        el_pbl=field(column.el),maxmf=jnp.zeros((B,1),jnp.float32),
        maxwidth=jnp.zeros((B,1),jnp.float32),ztop_plume=jnp.zeros((B,1),jnp.float32))
    monkeypatch.setenv('GPUWRF_MYNN_FP32_COLUMNS','1');monkeypatch.setenv('GPUWRF_MYNN_PHY_EXNER','0')
    monkeypatch.setattr(C,'_mynn_column_uses_wrf_phy_prep',lambda grid:False)
    monkeypatch.setattr(C,'_rho_from_state',lambda *args,**kw:field(column.rho))
    monkeypatch.setattr(C,'_column_dz_from_state',lambda *args,**kw:column.dz)
    for method,name in [('_u_mass','u'),('_v_mass','v'),('_w_mass','w')]:
        monkeypatch.setattr(C,method,lambda *args,field_name=name,**kw:field(getattr(column,field_name)))
    monkeypatch.setenv('GPUWRF_MYNN_QNI_MIXING','0');off=C._mynn_column_from_state(state,None)
    assert off.ni is None
    monkeypatch.setenv('GPUWRF_MYNN_QNI_MIXING','1');on=C._mynn_column_from_state(state,None)
    assert on.ni is not None,'actual State.Ni supplier absent'
    np.testing.assert_array_equal(np.asarray(on.ni).reshape(B,K),column.ni)
    # Complete real Ni values, with unrelated geometry mocked: wiring only.
    monkeypatch.setattr(C,'_add_a2c_u_increment',lambda a,delta:a)
    monkeypatch.setattr(C,'_add_a2c_v_increment',lambda a,delta:a)
    monkeypatch.setattr(C,'_output_dtype',lambda *args,**kw:jnp.float32)
    state.replace=lambda **kw:kw
    monkeypatch.setenv('GPUWRF_MYNN_CLOUDMIX','1')
    changed=on.replace(ni=on.ni+jnp.asarray(7.,on.ni.dtype))
    result=C._state_from_mynn_output(state,changed,theta_output_is_dry=False)
    assert 'Ni' in result,'actual MYNN number writeback absent'
    np.testing.assert_array_equal(result['Ni'],field(changed.ni))
    monkeypatch.setenv('GPUWRF_MYNN_CLOUDMIX','0')
    assert 'Ni' not in C._state_from_mynn_output(state,changed,theta_output_is_dry=False)


def test_real_number_change_survives_operational_rk(monkeypatch):
    from types import SimpleNamespace
    from gpuwrf.runtime import operational_mode as O
    assert 'Ni' in O._PHYSICS_NON_DRY_INCREMENT_FIELDS
    row=DATA['rows'][0];ni=jnp.asarray(row['input']['ni'],jnp.float32)
    class View:
        def __init__(self,a):self.Ni=a
        def __getattr__(self,n):return None
        def replace(self,**kw):return kw
    got=O._apply_physics_non_dry_updates(View(ni),View(ni),View(ni+7))
    assert 'Ni' in got
    np.testing.assert_array_equal(got['Ni'],ni+7)
