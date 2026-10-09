"""Original REAL4 cloud-deck selection and fresh-condensation caller wiring."""
from pathlib import Path
import ctypes
import hashlib
import json
import re
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import mynn_pbl as P, mynn_edmf as E
from gpuwrf.kernels import phys_mynn_columns as N
from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes

FLAG="GPUWRF_MYNN_PLUME_CLOUD_BASE"
FIXTURE=Path(__file__).parent/"fixtures/mynn_plume_cloud_base_wrf.json"


def finite_equal(candidate,reference):
    for side,fields in (("candidate",candidate),("reference",reference)):
        for name,value in fields.items():
            assert np.isfinite(value).all(),f"{side} {name} nonfinite"
    assert candidate.keys()==reference.keys()
    for name in candidate:np.testing.assert_array_equal(candidate[name],reference[name],err_msg=name)


def batch(row):
    v=np.asarray(row["input"],np.float32);f=np.asarray(row["flux"],np.float32)
    names=("u","v","w","theta","qv","tke","p","rho","dz","qc","qi","qs","qsq")
    values={n:jnp.asarray(v[:,:,i]) for i,n in enumerate(names)};z=jnp.zeros_like(values["qv"])
    state=P.MynnPBLColumnState(**values,km=z,kh=z,el=z)
    flux=SurfaceFluxes(ustar=jnp.asarray(f[:,0]),theta_flux=jnp.asarray(f[:,1]),qv_flux=jnp.asarray(f[:,2]),fltv=jnp.asarray(f[:,3]),xland=jnp.asarray(f[:,4]),t_skin=jnp.asarray(f[:,5]),rhosfc=jnp.asarray(f[:,6]),tau_u=jnp.zeros(len(v),jnp.float32),tau_v=jnp.zeros(len(v),jnp.float32))
    return state,flux,jnp.asarray(row["pre_qc_bl"],jnp.float32),jnp.asarray(row["pre_cldfra_bl"],jnp.float32)


@pytest.fixture(scope="session")
def pristine_cloud_base(tmp_path_factory):
    source=Path(__file__).parents[3]/"data/wrf_pristine/WRF/phys/module_bl_mynnedmf.F"
    text=source.read_text();start=text.index(" cloud_base  = 9000.0",text.index("SUBROUTINE DMP_mf"))
    end=text.index(" enddo",start)+len(" enddo");body=text[start:end]
    assert "qc_sgs = max(qc1(k), qc_bl1(k))" in body
    assert "cldfra_bl1(k) .ge. 0.5" in body and "cloud_base = zw1(k)" in body
    wrapper='''subroutine cb(n,zw1,dz1,w1,qc1,qc_bl1,cldfra_bl1,pblh,cloud_base) bind(C)
use iso_c_binding
implicit none
integer(c_int),value::n
real(c_float),intent(in)::zw1(n+1),dz1(n),w1(n),qc1(n),qc_bl1(n),cldfra_bl1(n),pblh
real(c_float),intent(out)::cloud_base
real(c_float)::maxw,zagl,wpbl,qc_sgs
real(c_float),parameter::half=.5,zero=0.
integer::k,k50,kte
kte=n;maxw=zero;k50=1
'''+body+"\nend subroutine\n"
    root=tmp_path_factory.mktemp("bp89_cloud_base");(root/"cb.f90").write_text(wrapper)
    subprocess.run(["timeout","60","<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran","-O2","-ffp-contract=off","-fPIC","-shared","cb.f90","-o","cb.so"],cwd=root,check=True,capture_output=True)
    (root/"provenance.json").write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),body_sha256=hashlib.sha256(body.encode()).hexdigest(),kind=4,scope="literal original DMP_mf first-cloud loop, unchanged equations")))
    lib=ctypes.CDLL(str(root/"cb.so"));ptr=np.ctypeslib.ndpointer(dtype=np.float32,flags="C_CONTIGUOUS")
    lib.cb.argtypes=[ctypes.c_int]+[ptr]*8;lib.cb.restype=None
    def oracle(state,qc_bl,cf,pblh):
        sqc=np.asarray(P._specific_moisture_components(state)[1],np.float32)
        dz=np.asarray(state.dz,np.float32);w=np.asarray(state.w,np.float32)
        zw=np.concatenate((np.zeros((len(dz),1),np.float32),np.cumsum(dz,axis=-1)),axis=-1)
        for name,value in {"sqc":sqc,"dz":dz,"w":w,"qc_bl":qc_bl,"cldfra_bl":cf,"pblh":pblh,"zw":zw}.items():assert np.isfinite(value).all(),name
        result=[]
        for i in range(len(dz)):
            arrays=[np.ascontiguousarray(x,np.float32) for x in [zw[i],dz[i],w[i],sqc[i],qc_bl[i],cf[i],np.array([pblh[i]])]];out=np.zeros(1,np.float32)
            lib.cb(dz.shape[-1],*arrays,out);result.append(out[0])
        return np.asarray(result,np.float32)
    return oracle


def test_real_cloud_deck_height_matches_original_loop(monkeypatch,pristine_cloud_base):
    monkeypatch.setenv(FLAG,"1")
    for row in json.loads(FIXTURE.read_text())["rows"]:
        state,flux,qc_bl,cf=batch(row);pblh=jnp.asarray(row["pblh_wrf"],jnp.float32)
        got=P._edmf_cloud_base_kwargs(state,pblh,qc_bl,cf)["cloud_base"]
        expected=pristine_cloud_base(state,np.asarray(qc_bl),np.asarray(cf),np.asarray(pblh))
        finite_equal({"cloud_base":np.asarray(got)},{"cloud_base":expected})
        assert np.any(expected<9000.)
        monkeypatch.setenv(FLAG,"0")
        assert P._edmf_cloud_base_kwargs(state,pblh,qc_bl,cf)=={}
        assert P._edmf_fresh_cloud_kwargs(qc_bl,cf)=={}
        monkeypatch.setenv(FLAG,"1")


def test_both_actual_mynn_calls_forward_fresh_clouds(monkeypatch):
    monkeypatch.setenv(FLAG,"1");monkeypatch.setattr(P,"_MYNN_SGS_CLOUD",True)
    row=json.loads(FIXTURE.read_text())["rows"][1];state,flux,qc_bl,cf=batch(row)
    assert np.any(np.asarray(cf)!=np.asarray(state.cldfra_bl))
    monkeypatch.setattr(P,"mym_condensation_cloudpdf2",lambda **kw:(qc_bl,jnp.zeros_like(qc_bl),cf))
    seen=[]
    class Captured(Exception):pass
    def spy(s,f,t,h,dt,dx,**kw):
        assert "qc_bl" in kw and "cldfra_bl" in kw,"fresh condensation diagnostics missing from DMP call"
        finite_equal({"qc_bl":np.asarray(kw["qc_bl"]),"cldfra_bl":np.asarray(kw["cldfra_bl"])},{"qc_bl":np.asarray(qc_bl),"cldfra_bl":np.asarray(cf)})
        seen.append(kw);raise Captured()
    monkeypatch.setattr(P,"_edmf_arrays_from_state",spy);monkeypatch.setattr(N,"_native_edmf_arrays",spy)
    for entry in (P._step_mynn_pbl_impl,N._advance_native):
        with pytest.raises(Captured):entry(state,6.,False,surface=flux,edmf=True,dx=1000.)
    assert len(seen)==2


def test_both_suppliers_deliver_current_cloud_base_to_dmp(monkeypatch):
    monkeypatch.setenv(FLAG,"1")
    from gpuwrf.kernels import phys_mynn_plume as L
    row=json.loads(FIXTURE.read_text())["rows"][1];state,flux,qc_bl,cf=batch(row)
    pblh=P._get_pblh(state,2*state.tke,flux.xland)
    expected=P._edmf_cloud_base_kwargs(state,pblh,qc_bl,cf)["cloud_base"]
    class Captured(Exception):pass
    seen=[]
    def spy(*a,**kw):
        assert "cloud_base" in kw,"cloud base missing from DMP setup"
        finite_equal({"cloud_base":np.asarray(kw["cloud_base"])},{"cloud_base":np.asarray(expected)})
        seen.append(kw);raise Captured()
    monkeypatch.setattr(E,"dmp_mf_columns",spy);monkeypatch.setattr(L,"dmp_mf_columns_native",spy)
    for supplier in (P._edmf_arrays_from_state,N._native_edmf_arrays):
        with pytest.raises(Captured):supplier(state,flux,flux.fltv,pblh,6.,1000.,qc_bl=qc_bl,cldfra_bl=cf)
    assert len(seen)==2


@pytest.mark.parametrize("field",["cloud_base","qc_bl","cldfra_bl","sqc","dz","w","pblh","zw"])
@pytest.mark.parametrize("bad",[np.nan,np.inf,-np.inf])
@pytest.mark.parametrize("side",["candidate","reference","both"])
def test_numeric_gate_rejects_nonfinite(field,bad,side):
    a={field:np.ones(3)};b={field:np.ones(3)}
    if side in {"candidate","both"}:a[field][1]=bad
    if side in {"reference","both"}:b[field][1]=bad
    with pytest.raises(AssertionError,match="nonfinite"):finite_equal(a,b)
