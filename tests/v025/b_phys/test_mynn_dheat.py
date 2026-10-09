"""WRF post-predict dissipative heating and THL RHS on real ES02 columns."""
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

from gpuwrf.physics import mynn_pbl as P
from gpuwrf.kernels import phys_mynn_columns as N
from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes

FLAG="GPUWRF_MYNN_DHEAT"
FIXTURE=Path(__file__).parent/"fixtures/mynn_dheat_wrf.json"


def finite_close(got,truth,**kw):
    for side,fields in (("candidate",got),("reference",truth)):
        for name,x in fields.items():assert np.isfinite(x).all(),f"{side} {name} nonfinite"
    assert got.keys()==truth.keys()
    for name in got:np.testing.assert_allclose(got[name],truth[name],err_msg=name,**kw)


def batch(row):
    v=np.asarray(row["input"],np.float32);f=np.asarray(row["flux"],np.float32)
    names=("u","v","w","theta","qv","tke","p","rho","dz","qc","qi","qs","qsq")
    values={n:jnp.asarray(v[:,:,i]) for i,n in enumerate(names)};z=jnp.zeros_like(values["qv"])
    state=P.MynnPBLColumnState(**values,km=z,kh=z,el=z)
    flux=SurfaceFluxes(ustar=jnp.asarray(f[:,0]),theta_flux=jnp.asarray(f[:,1]),qv_flux=jnp.asarray(f[:,2]),fltv=jnp.asarray(f[:,3]),xland=jnp.asarray(f[:,4]),t_skin=jnp.asarray(f[:,5]),rhosfc=jnp.asarray(f[:,6]),tau_u=jnp.zeros(len(v),jnp.float32),tau_v=jnp.zeros(len(v),jnp.float32))
    return state,flux


@pytest.fixture(scope="session")
def pristine_rate(tmp_path_factory):
    source=Path(__file__).parents[3]/"data/wrf_pristine/WRF/phys/module_bl_mynnedmf.F"
    text=source.read_text();assert re.search(r"integer, parameter :: dheat_opt = 1",text)
    body=re.search(r"    if \(dheat_opt > 0\) then.*?\n    endif",text,re.S).group(0)
    assert "qke1(k)**1.5" in body and "diss_heat1(kte) = 0." in body
    wrapper='''subroutine heat(n,qke1,el1,p1,diss_heat1) bind(C)
use iso_c_binding
use module_bl_mynnedmf_common,only:cp
use module_bl_mynnedmf,only:b1
implicit none
integer(c_int),value::n
real(c_float),intent(in)::qke1(n),el1(n),p1(n)
real(c_float),intent(out)::diss_heat1(n)
integer,parameter::dheat_opt=1,kts=1
integer::k,kte
real(c_float),parameter::half=.5,one=1.
kte=n
'''+body+"\nend subroutine\n"
    root=tmp_path_factory.mktemp("bp92_heat");(root/"heat.f90").write_text(wrapper)
    phys=source.parent
    subprocess.run(["timeout","60","<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran","-O2","-ftree-vectorize","-funroll-loops","-w","-ffree-form","-ffree-line-length-none","-fPIC","-shared","-I"+str(phys),"heat.f90",str(phys/"module_bl_mynnedmf.o"),str(phys/"module_bl_mynnedmf_common.o"),"-o","heat.so"],cwd=root,check=True,capture_output=True)
    (root/"provenance.json").write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),body_sha256=hashlib.sha256(body.encode()).hexdigest(),kind=4,scope="literal WRF DHEAT1 post-predict source, original common constants")))
    lib=ctypes.CDLL(str(root/"heat.so"));ptr=np.ctypeslib.ndpointer(dtype=np.float32,flags="C_CONTIGUOUS")
    lib.heat.argtypes=[ctypes.c_int]+[ptr]*4;lib.heat.restype=None
    def oracle(qke,el,p):
        assert all(np.isfinite(a).all() for a in [qke,el,p])
        out=np.empty_like(qke,dtype=np.float32)
        for i in range(len(out)):
            args=[np.ascontiguousarray(a[i],np.float32) for a in [qke,el,p]]
            lib.heat(qke.shape[-1],*args,out[i])
        return out
    return oracle


def test_postpredict_rate_matches_original_real4(monkeypatch,pristine_rate):
    monkeypatch.setenv(FLAG,"1")
    for row in json.loads(FIXTURE.read_text())["rows"]:
        state,_=batch(row);q=np.asarray(row["qke_new"],np.float32);el=np.asarray(row["el"],np.float32)
        original=pristine_rate(q,el,np.asarray(state.p))
        got=np.asarray(jax.jit(lambda q,e,p:P._mynn_dheat_kwargs(q,e,p)["diss_heat"])(jnp.asarray(q),jnp.asarray(el),state.p))
        finite_close({"heat":got},{"heat":original},rtol=6e-7,atol=2e-12)
        finite_close({"original":original},{"original":np.asarray(row["diss_heat"],np.float32)},rtol=0,atol=0)
        assert got.dtype==np.float32 and np.all(got[:,-1]==0.) and np.any(got[:,:-1]>0.)
    monkeypatch.setenv(FLAG,"0")
    assert P._mynn_dheat_kwargs(jnp.asarray(q),jnp.asarray(el),state.p)=={}


def test_both_actual_calls_use_new_qke_not_dissipation_diagnostic(monkeypatch):
    monkeypatch.setenv(FLAG,"1");monkeypatch.setattr(P,"_MYNN_SGS_CLOUD",False)
    row=json.loads(FIXTURE.read_text())["rows"][0];state,flux=batch(row)
    q=jnp.asarray(row["qke_new"],jnp.float32);el=jnp.asarray(row["el"],jnp.float32);zero=jnp.zeros_like(q)
    expected=P._mynn_dheat_kwargs(q,el,state.p)["diss_heat"]
    assert np.any(np.asarray(q)!=np.asarray(2*state.tke))
    monkeypatch.setattr(P,"_mym_turbulence",lambda *a,**kw:{"el":el,"pblh":jnp.ones(len(q),jnp.float32)})
    monkeypatch.setattr(P,"_mym_predict_qke",lambda *a,**kw:(q,zero,jnp.full_like(q,1000.),zero))
    seen=[]
    class Captured(Exception):pass
    def spy(*a,**kw):
        assert "diss_heat" in kw,"post-predict DHEAT missing from mean solver"
        finite_close({"heat":np.asarray(kw["diss_heat"])},{"heat":np.asarray(expected)},rtol=0,atol=0)
        seen.append(kw);raise Captured()
    monkeypatch.setattr(P,"_apply_mean_tendencies_with_clouds",spy)
    for entry in (P._step_mynn_pbl_impl,N._advance_native):
        with pytest.raises(Captured):entry(state,54.,False,surface=flux,edmf=False,dx=9000.)
    assert len(seen)==2


@pytest.mark.parametrize("mf",[False,True])
def test_heat_enters_thl_rhs_after_flux_terms_and_top_is_unheated(monkeypatch,mf):
    monkeypatch.setenv(FLAG,"1")
    row=json.loads(FIXTURE.read_text())["rows"][1];state,flux=batch(row)
    q=jnp.asarray(row["qke_new"],jnp.float32);el=jnp.asarray(row["el"],jnp.float32)
    rate=P._mynn_dheat_kwargs(q,el,state.p)["diss_heat"]
    zeros=jnp.zeros_like(q);interfaces=jnp.zeros(q.shape[:-1]+(q.shape[-1]+1,),jnp.float32)
    # Same real operands, isolate the RHS before the unchanged tridiagonal.
    captured=[]
    def solve(a,b,c,d):captured.append(np.asarray(d));return d
    monkeypatch.setattr(P,"_solve_tridiagonal",solve)
    thl=P._liquid_potential_temperature(state)
    bottom=54./state.dz[:,0]*flux.rhosfc*flux.theta_flux/state.rho[:,0]
    if mf:
        awx=jnp.concatenate((jnp.zeros_like(interfaces[:,:1]),jnp.full_like(interfaces[:,1:],.03125)),axis=-1)
        call=lambda **kw:P._diffusion_solve_with_mf(thl,zeros,state,54.,bottom,interfaces,awx,**kw)
    else:
        call=lambda **kw:P._diffusion_solve_with_surface(thl,zeros,state,54.,bottom,**kw)
    call();old=captured[-1]
    call(column_rhs=54.*rate);got=captured[-1]
    expected=old.copy();expected[:,:-1]=expected[:,:-1]+np.asarray(54.*rate[:,:-1])
    finite_close({"rhs":got},{"rhs":expected},rtol=0,atol=0)
    np.testing.assert_array_equal(got[:,-1],old[:,-1])


@pytest.mark.parametrize("field",["qke","el","pressure","heat","rhs"])
@pytest.mark.parametrize("bad",[np.nan,np.inf,-np.inf])
@pytest.mark.parametrize("side",["candidate","reference","both"])
def test_numeric_gate_rejects_nonfinite(field,bad,side):
    a={field:np.ones(3)};b={field:np.ones(3)}
    if side in {"candidate","both"}:a[field][1]=bad
    if side in {"reference","both"}:b[field][1]=bad
    with pytest.raises(AssertionError,match="nonfinite"):finite_close(a,b,rtol=1e-6,atol=0.)


@pytest.mark.parametrize("mf_enabled",[False,True])
def test_actual_mean_solver_receives_heat_and_changes_only_theta(monkeypatch,mf_enabled):
    monkeypatch.setenv(FLAG,"1")
    row=json.loads(FIXTURE.read_text())["rows"][1];state,flux=batch(row)
    q=jnp.asarray(row["qke_new"],jnp.float32);el=jnp.asarray(row["el"],jnp.float32)
    rate=P._mynn_dheat_kwargs(q,el,state.p)["diss_heat"]
    zeros=jnp.zeros_like(q);surface_zero=jnp.zeros_like(flux.ustar)
    flux=flux._replace(ustar=surface_zero,theta_flux=surface_zero,qv_flux=surface_zero)
    turb={"dfm":zeros,"dfh":zeros}
    interfaces=jnp.zeros(q.shape[:-1]+(q.shape[-1]+1,),jnp.float32)
    mf=None if not mf_enabled else {n:interfaces for n in ["s_aw","s_awu","s_awv","s_awthl","s_awqv","s_awqc"]}
    # Real clear columns; zero turbulent transport isolates the original
    # source term. Both the ordinary and cloudmix mean callers must forward it.
    wind=jnp.maximum(jnp.sqrt(state.u[:,0]**2+state.v[:,0]**2),.2)
    for cloudmix in ["0","1"]:
        monkeypatch.setenv("GPUWRF_MYNN_CLOUDMIX",cloudmix)
        baseline=P._apply_mean_tendencies_with_clouds(state,turb,54.,flux,wind,flux.rhosfc,mf=mf)
        heated=P._apply_mean_tendencies_with_clouds(state,turb,54.,flux,wind,flux.rhosfc,mf=mf,diss_heat=rate)
        expected=np.asarray(baseline[2])+np.asarray(54.*rate)
        finite_close({"theta":np.asarray(heated[2])},{"theta":expected},rtol=0,atol=0)
        for index in [0,1,3,4,5]:
            finite_close({"leaf":np.asarray(heated[index])},{"leaf":np.asarray(baseline[index])},rtol=0,atol=0)
        assert np.any(np.asarray(heated[2])[:,:4]!=np.asarray(baseline[2])[:,:4])
