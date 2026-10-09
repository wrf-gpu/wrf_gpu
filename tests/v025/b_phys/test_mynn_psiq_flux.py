"""BP95: the distinct WRF MYNN-SL flux-loop moisture resistances."""
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import surface_layer as SL
from gpuwrf.physics.fp32 import surface_layer_real as REAL

HERE = Path(__file__).parent
assert Path(SL.__file__).resolve().is_relative_to(HERE.parents[2]/'src'), 'test/model tree mismatch'
spec = importlib.util.spec_from_file_location('bp95_real_surface_inputs', HERE/'test_sfclay_snow_andreas.py')
scene = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scene)
pristine = scene.pristine


def cases():
    swiss = json.loads((HERE/'fixtures/sfclay_snow_swiss.json').read_text())
    canary = json.loads((HERE/'fixtures/sfclay_snow_canary.json').read_text())
    return [('swiss',swiss['columns'],swiss['dx_m'])]+[(n,a['columns'],a['dx_m']) for n,a in canary['arms'].items()]


def require_finite_fields(actual, expected):
    for side, values in [('candidate',actual),('reference',expected)]:
        for name,value in values.items():
            assert np.isfinite(np.asarray(value)).all(), f'{side} {name} nonfinite'


def finite_close(actual, expected, **kw):
    require_finite_fields(actual,expected)
    assert actual.keys()==expected.keys()
    for name in actual:
        np.testing.assert_allclose(actual[name],expected[name],err_msg=name,**kw)


def capture(state,depth):
    seen={}
    def profile(frame,event,arg):
        if event=='return' and frame.f_code is SL._surface_layer_impl.__code__:
            seen.update(frame.f_locals)
    prior=sys.getprofile()
    sys.setprofile(profile)
    try:
        diag=SL.surface_layer_with_diagnostics(state,snowh=depth)
    finally:
        sys.setprofile(prior)
    assert seen, 'actual product surface call not captured'
    for name in ['za','znt','z_q','psih','psih2','rhox','mavail','ustar','qsfcmr','qx','psiq_flux','psiq2_flux']:
        assert np.isfinite(np.asarray(seen[name])).all(), name
    for leaf in jax.tree.leaves(diag):
        assert np.isfinite(np.asarray(leaf)).all()
    return diag,seen


@pytest.fixture(scope='module')
def flux_literal(tmp_path_factory):
    source=HERE.parents[2]/'data/wrf_pristine/WRF/phys/module_sf_mynn.F'
    lines=source.read_text().splitlines()
    start=next(i for i,s in enumerate(lines) if 'PSIQ=MAX(LOG((ZA(I)+z_q(i))/z_q(I))' in s)
    psi=lines[start:start+2]
    flqc=next(s for s in lines if 'FLQC(I)=RHO1D(I)*MAVAIL(I)*UST(I)*KARMAN/PSIQ' in s)
    qfx=[next(s for s in lines if 'QFX(I)=FLQC(I)*(QSFCMR(I)-QV1D(I))' in s),
          next(s for s in lines if 'QFX(I)=MAX(QFX(I),-0.02)' in s)]
    q2start=next(i for i,s in enumerate(lines) if 'Q2(I)=QSFCMR(I)' in s.replace(' ',''))
    q2=lines[q2start:q2start+3]
    body='\n'.join(psi+[flqc]+qfx+q2)
    out=tmp_path_factory.mktemp('psiq_literal')
    (out/'flux.f90').write_text('''subroutine bp_flux(n,za,z_q,psih,psih2,rho1d,mavail,ust,qs fcmr,qv1d,karman,out) bind(C)
use iso_c_binding
implicit none
integer(c_int),value::n
integer::i
real(c_float),intent(in)::za(n),z_q(n),psih(n),psih2(n),rho1d(n),mavail(n),ust(n),qsfcmr(n),qv1d(n)
real(c_float),value::karman
real(c_float),intent(out)::out(n,4)
real(c_float)::psiq,psiq2,flqc(n),qfx(n),q2(n)
do i=1,n
'''.replace('qs fcmr','qsfcmr')+body+'''
out(i,1)=psiq
out(i,2)=psiq2
out(i,3)=qfx(i)
out(i,4)=q2(i)
enddo
end subroutine
''')
    compiler=os.environ.get('FC','<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran')
    subprocess.run([compiler,'-O2','-shared','-fPIC','-ffree-line-length-none',str(out/'flux.f90'),'-o',str(out/'flux.so')],check=True,capture_output=True)
    lib=ctypes.CDLL(str(out/'flux.so'))
    ptr=np.ctypeslib.ndpointer(np.float32,flags='F_CONTIGUOUS')
    lib.bp_flux.argtypes=[ctypes.c_int]+[ptr]*9+[ctypes.c_float,ptr]
    def oracle(values):
        names=['za','z_q','psih','psih2','rhox','mavail','ustar','qsfcmr','qx']
        arrays=[np.asfortranarray(np.asarray(values[n],np.float32).reshape(-1)) for n in names]
        result=np.zeros((len(arrays[0]),4),np.float32,order='F')
        lib.bp_flux(len(arrays[0]),*arrays,.4,result)
        return {n:result[:,i] for i,n in enumerate(['psiq','psiq2','qfx','q2'])}
    return oracle


@pytest.mark.parametrize('native',[False,True])
def test_actual_flux_and_q2_use_literal_pristine_forms(flux_literal,monkeypatch,native):
    monkeypatch.setattr(REAL,'_NATIVE_REAL',native)
    monkeypatch.setattr(SL,'_MYNN_PSIQ_FLUX_WRF',True)
    for _,rows,dx in cases():
        state=scene.view(rows)._replace(dx_m=dx)
        depth=jnp.asarray([r['snowh'] for r in rows]).reshape(len(rows),1)
        diag,v=capture(state,depth)
        got=dict(psiq=np.asarray(v['psiq_flux']).reshape(-1),psiq2=np.asarray(v['psiq2_flux']).reshape(-1),
                 qfx=np.asarray(diag.fluxes.qv_flux*v['rhox']).reshape(-1),q2=np.asarray(diag.q2).reshape(-1))
        finite_close(got,flux_literal(v),rtol=8e-7,atol=2e-9)


def test_complete_native_surface_columns_pass_original_real4_wrf(pristine,monkeypatch):
    monkeypatch.setattr(REAL,'_NATIVE_REAL',True)
    monkeypatch.setattr(SL,'_MYNN_PSIQ_FLUX_WRF',True)
    for _,rows,dx in cases():
        state=scene.view(rows)._replace(dx_m=dx)
        depth=jnp.asarray([r['snowh'] for r in rows]).reshape(len(rows),1)
        diag,_=capture(state,depth)
        actual=scene.diagnostics_arrays(diag)
        expected=scene.oracle(pristine[0],rows,dx)
        require_finite_fields(actual,expected)
        scene.assert_surface_parity(actual,expected)


@pytest.mark.parametrize('native',[False,True])
def test_heat_wind_and_stability_unchanged(monkeypatch,native):
    monkeypatch.setattr(REAL,'_NATIVE_REAL',native)
    unchanged=('ustar','theta_flux','tau_u','tau_v','rhosfc')
    for _,rows,dx in cases():
        state=scene.view(rows)._replace(dx_m=dx)
        depth=jnp.asarray([r['snowh'] for r in rows]).reshape(len(rows),1)
        monkeypatch.setattr(SL,'_MYNN_PSIQ_FLUX_WRF',False)
        old,_=capture(state,depth)
        monkeypatch.setattr(SL,'_MYNN_PSIQ_FLUX_WRF',True)
        new,_=capture(state,depth)
        for name in ('hfx','u10','v10','th2','t2','qsfc','mol','rmol','zol','regime','psim','psih','br','znt','chs2','cqs2'):
            np.testing.assert_array_equal(np.asarray(getattr(new,name)),np.asarray(getattr(old,name)),err_msg=name)
        for name in unchanged:
            np.testing.assert_array_equal(np.asarray(getattr(new.fluxes,name)),np.asarray(getattr(old.fluxes,name)),err_msg=name)


@pytest.mark.parametrize('side',['candidate','reference'])
@pytest.mark.parametrize('field',list(dict.fromkeys(['psiq','psiq2','qfx','q2']+scene.OUT_COLS)))
@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_numeric_gate_rejects_each_nonfinite_field(side,field,bad):
    a={n:np.zeros(3) for n in list(dict.fromkeys(['psiq','psiq2','qfx','q2']+scene.OUT_COLS))}
    b={n:v.copy() for n,v in a.items()}
    (a if side=='candidate' else b)[field][-1]=bad
    with pytest.raises(AssertionError,match=f'{side} {field} nonfinite'):
        finite_close(a,b,rtol=0,atol=0)
