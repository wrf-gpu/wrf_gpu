"""BP96: independent WRF pi_phy, not pressure-derived p_hyd Exner."""
import ctypes
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import mynn_pbl as P
from gpuwrf.kernels import phys_mynn_columns as N
from gpuwrf.coupling import physics_couplers as C
from test_mynn_dheat import batch,FIXTURE as HEAT

HERE=Path(__file__).parent
FLAG='GPUWRF_MYNN_PHY_EXNER'


def finite_close(actual,expected,**kw):
    for side,fields in [('candidate',actual),('reference',expected)]:
        for name,value in fields.items():assert np.isfinite(np.asarray(value)).all(),f'{side} {name} nonfinite'
    assert actual.keys()==expected.keys()
    for name in actual:np.testing.assert_allclose(actual[name],expected[name],err_msg=name,**kw)


def real_columns(h=1):
    row=json.loads((HERE/f'fixtures/mynn_phy_exner_h{h}.json').read_text())
    a={n:jnp.asarray(row[n],jnp.float32) for n in ['p_hyd','p_dyn','pi_phy','theta','qv','qc','qi','qs']}
    for name,value in a.items():assert np.isfinite(value).all(),name
    z=jnp.zeros_like(a['qv'])
    # Unused wind/turbulence/geometry auxiliaries are neutral for this
    # thermodynamic component probe, not a whole-MYNN fidelity fixture.
    state=P.MynnPBLColumnState(z,z,z,a['theta'],a['qv'],z,a['p_hyd'],jnp.ones_like(z),
                              jnp.ones_like(z)*50,z,z,z,qc=a['qc'],qi=a['qi'],qs=a['qs'],exner=a['pi_phy'])
    return row,a,state


@pytest.fixture(scope='module')
def original_pi(tmp_path_factory):
    source=HERE.parents[2]/'data/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F'
    lines=source.read_text().splitlines()
    body='\n'.join(next(s for s in lines if term in s) for term in
                   ['p_phy(i,k,j) = p(i,k,j) + pb(i,k,j)','pi_phy(i,k,j) = (p_phy(i,k,j)/p1000mb)**rcp'])
    out=tmp_path_factory.mktemp('phy_pi')
    (out/'pi.f90').write_text('''subroutine bp_pi(n,p,pb,p_phy,pi_phy) bind(C)
use iso_c_binding
implicit none
integer(c_int),value::n
integer::i,k,j
real(c_float),intent(in)::p(n,1,1),pb(n,1,1)
real(c_float),intent(out)::p_phy(n,1,1),pi_phy(n,1,1)
real(c_float),parameter::p1000mb=100000.,rcp=2./7.
k=1;j=1
do i=1,n
'''+body+'''
enddo
end subroutine
''')
    subprocess.run(['<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran','-O2','-fPIC','-shared','-ffree-line-length-none',str(out/'pi.f90'),'-o',str(out/'pi.so')],check=True,capture_output=True)
    lib=ctypes.CDLL(str(out/'pi.so'));ptr=np.ctypeslib.ndpointer(np.float32,flags='C_CONTIGUOUS');lib.bp_pi.argtypes=[ctypes.c_int]+[ptr]*4
    def oracle(row):
        p,pb=[np.ascontiguousarray(row[n],np.float32).reshape(-1) for n in ['p_prime','p_base']]
        po=np.zeros_like(p);pi=np.zeros_like(p);lib.bp_pi(len(p),p,pb,po,pi)
        shape=np.asarray(row['p_dyn']).shape
        return po.reshape(shape),pi.reshape(shape)
    return oracle


@pytest.mark.parametrize('h',[1,6])
def test_supplied_pi_matches_literal_original_phy_prep(h,original_pi):
    row,a,state=real_columns(h);p,pi=original_pi(row)
    finite_close({'p':p,'pi':pi},{'p':np.asarray(a['p_dyn']),'pi':np.asarray(a['pi_phy'])},rtol=0,atol=0)
    finite_close({'pi':np.asarray(P._column_exner(state))},{'pi':pi},rtol=0,atol=0)
    assert np.max(np.abs(np.asarray(P._exner_from_pressure(state.p))-pi))>1e-6
    np.testing.assert_array_equal(np.asarray(state.p),np.asarray(a['p_hyd']))


@pytest.mark.parametrize('h',[1,6])
def test_optional_leaf_survives_clip_replace_batch_and_tree(h):
    _,_,state=real_columns(h)
    np.testing.assert_array_equal(P._clip_state(state).exner,state.exner)
    out=jax.tree.map(lambda x:x,state);assert out==state and hash(out)==hash(state)
    absent=state.replace(exner=None);assert P._column_exner(absent).shape==state.p.shape
    np.testing.assert_array_equal(P._column_exner(absent),P._exner_from_pressure(state.p))


def test_real_coupler_keeps_hydro_p_and_supplies_dynamic_pi(monkeypatch,original_pi):
    from types import SimpleNamespace
    row,a,_=real_columns();B,K=a['p_dyn'].shape
    field=lambda name:jnp.transpose(a[name]).reshape(K,B,1)
    theta=field('theta')*(1+C.WRF_RV_OVER_RD*field('qv'))
    state=SimpleNamespace(p=field('p_dyn'),theta=theta,qv=field('qv'),qc=field('qc'),qi=field('qi'),qs=field('qs'),
                          qke=jnp.zeros_like(theta),qsq=jnp.zeros_like(theta))
    monkeypatch.setenv('GPUWRF_MYNN_FP32_COLUMNS','1')
    monkeypatch.setattr(C,'_mynn_column_uses_wrf_phy_prep',lambda grid:True)
    monkeypatch.setattr(C,'_wrf_hydrostatic_pressure_from_state',lambda *args,**kw:(field('p_hyd'),jnp.zeros((B,1),jnp.float32)))
    monkeypatch.setattr(C,'_wrf_phy_prep_rho_from_state',lambda *args,**kw:jnp.ones_like(theta))
    monkeypatch.setattr(C,'_surface_dz_from_state',lambda *args,**kw:C._to_columns(jnp.ones_like(theta)*50))
    for name in ['_u_mass','_v_mass','_w_mass']:
        monkeypatch.setattr(C,name,lambda *args,**kw:jnp.zeros_like(theta))
    grid=SimpleNamespace(metrics=object())
    monkeypatch.setenv(FLAG,'1');out=C._mynn_column_from_state(state,grid)
    assert out.exner is not None,'active MYNN pi_phy was not forwarded'
    finite_close({'pi':np.asarray(out.exner).reshape(B,K)}, {'pi':original_pi(row)[1]},rtol=2e-7,atol=0)
    finite_close({'p':np.asarray(out.p).reshape(B,K)}, {'p':np.asarray(a['p_hyd'])},rtol=0,atol=0)
    monkeypatch.setenv(FLAG,'0');off=C._mynn_column_from_state(state,grid);assert off.exner is None
    np.testing.assert_array_equal(off.p,out.p)


def test_both_actual_whole_callers_forward_independent_pi(monkeypatch):
    row=json.loads(HEAT.read_text())['rows'][0];state,flux=batch(row)
    # Wiring regression: distinct supplied pi is observable before the solver.
    pi=P._exner_from_pressure(state.p)+jnp.asarray(.001,state.p.dtype)
    state=state.replace(exner=pi)
    monkeypatch.setattr(P,'_MYNN_SGS_CLOUD',True)
    class Captured(Exception):pass
    def spy(**kw):
        finite_close({'pi':np.asarray(kw['exner'])},{'pi':np.asarray(pi)},rtol=0,atol=0)
        np.testing.assert_array_equal(kw['p'],state.p)
        raise Captured()
    monkeypatch.setattr(P,'mym_condensation_cloudpdf2',spy)
    for entry in [P._step_mynn_pbl_impl,N._advance_native]:
        with pytest.raises(Captured):entry(state,54.,False,surface=flux,edmf=False,dx=9000.)


def test_both_plume_suppliers_use_dynamic_pi_for_skin_theta(monkeypatch):
    from gpuwrf.physics import mynn_edmf as E
    from gpuwrf.kernels import phys_mynn_plume as MF
    state,flux=batch(json.loads(HEAT.read_text())['rows'][0])
    pi=P._exner_from_pressure(state.p)*jnp.asarray(.9999,state.p.dtype)
    state=state.replace(exner=pi)
    seen=[]
    class Captured(Exception):pass
    def spy(*args,**kw):
        finite_close({'p':np.asarray(args[11]),'pi':np.asarray(args[12])},
                     {'p':np.asarray(state.p),'pi':np.asarray(pi)},rtol=0,atol=0)
        finite_close({'ts':np.asarray(kw['ts'])},{'ts':np.asarray(flux.t_skin/pi[...,0])},rtol=0,atol=0)
        seen.append(True);raise Captured()
    monkeypatch.setattr(E,'dmp_mf_columns',spy)
    monkeypatch.setattr(MF,'dmp_mf_columns_native',spy)
    pblh=jnp.ones_like(flux.ustar)*500
    for entry in [P._edmf_arrays_from_state,N._native_edmf_arrays]:
        with pytest.raises(Captured):entry(state,flux,flux.fltv,pblh,54.,9000.)
    assert len(seen)==2


@pytest.mark.parametrize('side',['candidate','reference'])
@pytest.mark.parametrize('field',['p','pi'])
@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_numeric_gate_rejects_nonfinite_before_compare(side,field,bad):
    a={n:np.ones(4) for n in ['p','pi']};b={n:v.copy() for n,v in a.items()}
    (a if side=='candidate' else b)[field][-1]=bad
    with pytest.raises(AssertionError,match=f'{side} {field} nonfinite'):finite_close(a,b,rtol=0,atol=0)
