"""Pristine SCALE_AWARE and both real MYNN callers; BP87 component gates."""
from pathlib import Path
import ctypes,json,subprocess
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.physics import mynn_pbl as P, mynn_edmf as E
from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
from gpuwrf.kernels import phys_mynn_columns as N,phys_mynn_plume as K


def rows():
    return json.loads((Path(__file__).parent/'fixtures/mynn_scale_aware_columns.json').read_text())['rows']


def assert_factor_matches_pristine(got,reference):
    assert np.isfinite(got).all() and np.isfinite(reference).all(), 'nonfinite factor'
    np.testing.assert_allclose(got,reference,rtol=3e-7,atol=1e-7)


def assert_factor_forwarded(got,expected):
    assert got is not None, 'missing factor'
    assert np.isfinite(got).all() and np.isfinite(expected).all(), 'nonfinite factor'
    np.testing.assert_array_equal(got,expected)


def test_scale_aware_matches_original_real4_objects(monkeypatch,tmp_path):
    source=Path(__file__).resolve().parents[3]/'data/wrf_pristine/WRF/phys'
    wrapper='''subroutine bp87(n,dx,h,b,s) bind(C)
use iso_c_binding
use module_bl_mynnedmf, only: SCALE_AWARE
integer(c_int),value :: n
real(c_float),intent(in) :: dx(n),h(n)
real(c_float),intent(out) :: b(n),s(n)
integer :: i
do i=1,n
call SCALE_AWARE(dx(i),h(i),b(i),s(i))
enddo
end subroutine
'''
    (tmp_path/'oracle.f90').write_text(wrapper)
    subprocess.run(['timeout','60','<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran','-O2','-fPIC','-shared',
       '-I'+str(source),'oracle.f90',str(source/'module_bl_mynnedmf.o'),str(source/'module_bl_mynnedmf_common.o'),'-o','oracle.so'],cwd=tmp_path,check=True,capture_output=True)
    data=rows();dx=np.array([r['dx'] for r in data],np.float32);h=np.array([r['pblh'] for r in data],np.float32)
    b=np.empty_like(h);s=np.empty_like(h);lib=ctypes.CDLL(str(tmp_path/'oracle.so'))
    ptr=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');lib.bp87.argtypes=[ctypes.c_int]+[ptr]*4
    lib.bp87(len(h),dx,h,b,s)
    monkeypatch.setenv('GPUWRF_MYNN_SCALE_AWARE','1')
    got=np.asarray(jax.jit(P._edmf_scale_aware_psig)(jnp.asarray(dx),jnp.asarray(h)))
    assert_factor_matches_pristine(got,s)
    assert float(s.min())<.98 and np.all(got<1),'real fine-grid taper is inactive'
    monkeypatch.setenv('GPUWRF_MYNN_SCALE_AWARE','0')
    assert P._edmf_scale_aware_psig(dx,h) is None


def test_both_mynn_suppliers_forward_taper_and_preserve_off(monkeypatch):
    f=json.loads((Path(__file__).parent/'fixtures/mynn_sgs_drizzle_wrf.json').read_text())
    vals=np.array(f['input_values'],np.float32);names=f['input_fields']
    kw={n:jnp.asarray(vals[:,:,i]) for i,n in enumerate(names)};z=jnp.zeros_like(kw['qv'])
    state=P.MynnPBLColumnState(**kw,km=z,kh=z,el=z)
    count=state.qv.shape[0];sur=jnp.ones(count,jnp.float32)
    flux=SurfaceFluxes(sur,sur,sur,sur,sur,sur,sur,sur,sur)
    h=jnp.array([r['pblh'] for r in rows() if r['domain']=='d03' and r['lead_h']==30],jnp.float32)
    observed=[]
    def spy(*args,**kwargs):observed.append(kwargs.get('psig_shcu'));return None
    monkeypatch.setattr(E,'dmp_mf_columns',spy);monkeypatch.setattr(K,'dmp_mf_columns_native',spy)
    for value in ['0','1']:
        monkeypatch.setenv('GPUWRF_MYNN_SCALE_AWARE',value)
        for fn in [P._edmf_arrays_from_state,N._native_edmf_arrays]:
            fn(state,flux,sur,h,6.,1000.)
            if value=='0':assert observed[-1] is None
            else:assert_factor_forwarded(observed[-1],np.asarray(P._edmf_scale_aware_psig(1000.,h)))
    assert len(observed)==4


@pytest.mark.parametrize('checker',[assert_factor_matches_pristine,assert_factor_forwarded])
@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
@pytest.mark.parametrize('side',['candidate','reference','both'])
def test_actual_factor_checks_refuse_nonfinite(checker,bad,side):
    got=np.array([.95,.99],np.float32);reference=got.copy()
    if side in ['candidate','both']:got[0]=bad
    if side in ['reference','both']:reference[0]=bad
    with pytest.raises(AssertionError,match='nonfinite'):
        checker(got,reference)
