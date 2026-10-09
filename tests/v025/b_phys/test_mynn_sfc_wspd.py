"""Real daytime WRF caller seam and original REAL4 MYNN surface boundary."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import ctypes
import hashlib
import json
import re
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import State, _state_field_shapes
from gpuwrf.coupling import physics_couplers as C
from gpuwrf.coupling import noahmp_surface_hook as H
from gpuwrf.physics import mynn_pbl as P, surface_layer as S, noahmp_coupler as N

FLAG = "GPUWRF_MYNN_SFC_WSPD"
FIXTURE = Path(__file__).parent / "fixtures/mynn_sfc_wspd_day_wrf.json"


class View(SimpleNamespace):
    def replace(self, **kw):
        return View(**(vars(self) | kw))


def finite_close(got, expected, **kw):
    for side, fields in (("candidate", got), ("reference", expected)):
        for name, value in fields.items():
            assert np.isfinite(np.asarray(value)).all(), f"{side} {name} is nonfinite"
    assert got.keys() == expected.keys()
    for name in got:
        np.testing.assert_allclose(got[name], expected[name], **kw, err_msg=name)


def real_inputs():
    fixture = json.loads(FIXTURE.read_text())
    records = fixture["columns"]
    def arr(name):
        return jnp.asarray([[r[name] for r in records]], jnp.float32)
    def field(name, default=0.):
        return jnp.asarray([[r["values"].get(name, default) for r in records]], jnp.float32)
    view = View(u=arr("u")[..., None], v=arr("v")[..., None],
        theta=arr("theta")[..., None], qv=arr("qv")[..., None],
        p=arr("p")[..., None], dz=arr("dz")[..., None],
        t_air=arr("temp")[..., None], rho=arr("rho")[..., None],
        psfc=arr("psfc"), t_skin=field("TSK"), soil_moisture=arr("soil_moisture"),
        xland=field("XLAND", 1.), lakemask=field("LAKEMASK"),
        mavail=jnp.ones_like(arr("u")), roughness_m=field("ZNT", .01),
        ustar=field("UST", .1), mol=field("MOL"), hfx=field("HFX"),
        qfx=field("QFX"), qsfc=arr("qsfc_seed"), pblh=field("PBLH", 1000.), dx_m=1000.)
    for name, value in vars(view).items():
        assert np.isfinite(value).all(), name
    return fixture, records, view, arr, field


def real_state(view):
    base = GridSpec.canary_3km_template()
    eta = jnp.linspace(1., 0., 4, dtype=jnp.float32)
    grid = replace(base, projection=replace(base.projection, nx=96, ny=1),
        vertical=replace(base.vertical, nz=3, eta_levels=eta), eta_levels=eta,
        terrain=replace(base.terrain, shape=(1, 96)), terrain_height=jnp.zeros((1, 96),jnp.float32), metrics=None)
    state = State(**{n:jnp.zeros(shape, jnp.float32) for n,shape in _state_field_shapes(grid).items()})
    fields = {n:getattr(view,n) for n in ["t_skin","soil_moisture","xland","lakemask","mavail","roughness_m","ustar","mol","hfx","qfx","qsfc","pblh"]}
    return state.replace(**fields), grid


@pytest.fixture(scope="session")
def pristine_bc(tmp_path_factory):
    source = Path(__file__).parents[3] / "data/wrf_pristine/WRF/phys/module_bl_mynnedmf.F"
    text = source.read_text()
    bodies = []
    for wind in ("u", "v"):
        start = text.index(f"d(k)={wind}(k)  + dtz(k)*")
        before = text.rfind("b(k)=1.", 0, start)
        end = text.index(f"+ sub_{wind}(k)*delt + det_{wind}(k)*delt", start)
        end += len(f"+ sub_{wind}(k)*delt + det_{wind}(k)*delt")
        bodies.append(text[before:end])
    pre = '''subroutine bp90_bc(n,usts,wspds,rhosfcs,rhos,dzs,us,vs,uo,vo) bind(C)
use iso_c_binding
implicit none
integer(c_int), value :: n
real(c_float),intent(in)::usts(n),wspds(n),rhosfcs(n),rhos(n),dzs(n),us(n),vs(n)
real(c_float),intent(out)::uo(n),vo(n)
real(c_float)::ust,wspd,rhosfc,delt,uoce,voce,onoff
real(c_float)::b(2),c(2),d(2),dtz(2),rhoinv(2),kmdz(3),u(2),v(2)
real(c_float)::s_aw1(3),sd_aw1(3),s_awu1(3),sd_awu1(3),s_awv1(3),sd_awv1(3)
real(c_float)::sub_u(2),det_u(2),sub_v(2),det_v(2)
integer::i,k
k=1;delt=6.;uoce=0.;voce=0.;onoff=0.
kmdz=0.;s_aw1=0.;sd_aw1=0.;s_awu1=0.;sd_awu1=0.;s_awv1=0.;sd_awv1=0.
sub_u=0.;det_u=0.;sub_v=0.;det_v=0.
do i=1,n
 ust=usts(i);wspd=wspds(i);rhosfc=rhosfcs(i)
 dtz=delt/dzs(i);rhoinv=1./rhos(i);u=us(i);v=vs(i)
'''
    code = pre+bodies[0]+"\nuo(i)=d(k)/b(k)\n"+bodies[1]+"\nvo(i)=d(k)/b(k)\nenddo\nend subroutine\n"
    root = tmp_path_factory.mktemp("bp90_original_bc")
    (root/"bc.f90").write_text(code)
    subprocess.run(["timeout","60","<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran", "-O2", "-ffp-contract=off", "-fPIC", "-shared", "bc.f90", "-o", "bc.so"], cwd=root, check=True, capture_output=True)
    (root/"provenance.json").write_text(json.dumps({"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(), "literal_bodies_sha256":[hashlib.sha256(b.encode()).hexdigest() for b in bodies], "scope":"original b/c/d bottom-row expressions; zero diffusion/MF/ocean velocity isolates momentum surface BC; REAL4"}))
    lib=ctypes.CDLL(str(root/"bc.so")); ptr=np.ctypeslib.ndpointer(dtype=np.float32,flags="C_CONTIGUOUS")
    lib.bp90_bc.argtypes=[ctypes.c_int]+[ptr]*9;lib.bp90_bc.restype=None
    def oracle(*args):
        arrays=[np.ascontiguousarray(a,np.float32).reshape(-1) for a in args]
        out=[np.empty_like(arrays[0]) for _ in range(2)]
        lib.bp90_bc(len(arrays[0]),*arrays,*out)
        return out
    return oracle


@pytest.mark.parametrize("native_surface", [False, True])
def test_real_sfclay_noah_mynn_handoff_and_pristine_bc(monkeypatch, pristine_bc, native_surface):
    monkeypatch.setenv(FLAG,"1")
    monkeypatch.setenv("GPUWRF_MYNN_FP32_COLUMNS","1")
    from gpuwrf.physics.fp32 import surface_layer_real
    monkeypatch.setattr(surface_layer_real,"_NATIVE_REAL",native_surface)
    fixture, records, view, arr, field=real_inputs()
    diag=S.surface_layer_with_diagnostics(view,first_timestep=False,snowh=arr("snowh"))
    finite_close({"wspd":diag.fluxes.wspd},{"wspd":fixture["baseline"]["wrfwind"]},rtol=2e-7,atol=0.)
    # Noah changes heat/moisture fluxes after sfclay. The WRF caller retains
    # the already computed WSPD, not a recomputation using those new fluxes.
    nm=View(hfx=field("HFX"),qfx=field("QFX"),lh=field("LH"),znt=field("ZNT"),tsk=field("TSK"),qsfc=arr("qsfc_seed"))
    land=View(ch=jnp.zeros_like(arr("u")),cm=jnp.zeros_like(arr("u")),snowh=arr("snowh"))
    forcing=View(qair=arr("qv")/(1+arr("qv")),psfc=arr("psfc"),sfctmp=arr("temp"))
    monkeypatch.setattr(N,"surface_layer_with_diagnostics",lambda *a,**kw:diag)
    monkeypatch.setattr(N,"noah_mp_step",lambda state,*a,**kw:(state,nm))
    wb,landout,blended=N.noahmp_surface_adapter(view,land,None,forcing=forcing)
    assert blended.wspd is diag.fluxes.wspd
    state,grid=real_state(view)
    assert state.sfc_wspd.shape==state.xland.shape and state.sfc_wspd.dtype==jnp.float32
    # Without Noah-MP the standalone surface adapter writes the same handle.
    monkeypatch.setattr(C,"_surface_column_view",lambda *a:view)
    monkeypatch.setattr(C,"surface_layer_with_diagnostics",lambda *a,**kw:diag)
    bare=C.surface_adapter(state,6.,grid=grid)
    finite_close({"wspd":bare.sfc_wspd},{"wspd":np.asarray(diag.fluxes.wspd,dtype=np.float32)},rtol=0,atol=0)
    monkeypatch.setattr(H,"_build_column_view",lambda *a:view)
    monkeypatch.setattr(H,"noahmp_surface_adapter",lambda *a,**kw:(wb,landout,blended))
    state,_=H.noahmp_surface_step(state,land,None,dt=6.,grid=grid)
    recipient=C._surface_fluxes_from_state(state)
    finite_close({"wspd":recipient.wspd},{"wspd":np.asarray(diag.fluxes.wspd,dtype=np.float32)},rtol=0,atol=0)
    column=View(u=view.u,v=view.v)
    flux,wind,_,rho_sfc=P._surface_terms(column,recipient)
    finite_close({"wind":wind},{"wind":fixture["baseline"]["wrfwind"]},rtol=2e-7,atol=0)
    actual_drag=np.asarray(rho_sfc)*np.asarray(flux.ustar)**2/np.asarray(wind)
    wrf_drag=np.asarray(rho_sfc)*np.asarray(flux.ustar)**2/np.asarray(fixture["baseline"]["wrfwind"])
    finite_close({"drag":actual_drag},{"drag":wrf_drag},rtol=2e-7,atol=0)
    # Actual implicit BC, with zero exchange/MF to isolate the WRF bottom row.
    def profile(x): return jnp.repeat(x,2,axis=-1).reshape(96,2)
    zero=jnp.zeros((96,2),jnp.float32)
    col=P.MynnPBLColumnState(profile(view.u),profile(view.v),zero,profile(view.theta),profile(view.qv),zero,
        profile(view.p),profile(view.rho),profile(view.dz),zero,zero,zero)
    fb=jax.tree.map(lambda a: None if a is None else jnp.broadcast_to(a,(1,96)).reshape(96),flux)
    _,windb,_,rb=P._surface_terms(col,fb)
    expected=pristine_bc(fb.ustar,windb,rb,col.rho[:,0],col.dz[:,0],col.u[:,0],col.v[:,0])
    got=P._apply_mean_tendencies_legacy(col,{"dfm":zero,"dfh":zero},6.,fb,windb,rb)
    finite_close({"u":np.asarray(got[0])[:,0],"v":np.asarray(got[1])[:,0]},dict(zip(["u","v"],expected)),rtol=3e-7,atol=2e-7)
    # The extra handle is NOT an explicit stress RHS in pristine MYNN.
    changed=fb._replace(tau_u=fb.tau_u+1000.,tau_v=fb.tau_v-1000.)
    again=P._apply_mean_tendencies_legacy(col,{"dfm":zero,"dfh":zero},6.,changed,windb,rb)
    finite_close({"u":again[0],"v":again[1]},{"u":got[0],"v":got[1]},rtol=0,atol=0)
    # Stop each actual implementation after its surface read to check that
    # both retained and native paths consume the same WRF denominator.
    from gpuwrf.kernels import phys_mynn_columns as native
    seen=[];read=P._surface_terms
    class Captured(Exception): pass
    def tap(*a,**kw):
        values=read(*a,**kw);seen.append(values[1]);raise Captured()
    monkeypatch.setattr(P,"_surface_terms",tap)
    for entry in (P._step_mynn_pbl_impl,native._advance_native):
        with pytest.raises(Captured):entry(col,6.,False,surface=fb)
    assert len(seen)==2
    for observed in seen:finite_close({"wind":observed},{"wind":windb},rtol=0,atol=0)


def test_off_keeps_original_surface_terms_and_key_isolated(monkeypatch):
    fixture,records,view,arr,field=real_inputs()
    monkeypatch.setenv(FLAG,"0")
    diag=S.surface_layer_with_diagnostics(view,first_timestep=False,snowh=arr("snowh"))
    assert diag.fluxes.wspd is None
    original=lambda c:jnp.maximum(jnp.sqrt(c.u[...,0]*c.u[...,0]+c.v[...,0]*c.v[...,0]),.2)
    got=lambda c:P._surface_terms(c,diag.fluxes)[1]
    finite_close({"wind":got(view)},{"wind":original(view)},rtol=0,atol=0)
    # View is not a pytree; pass the two real operand arrays explicitly.
    actual=jax.jit(lambda u,v:got(View(u=u,v=v))).lower(view.u,view.v).compiler_ir("hlo").as_hlo_text()
    legacy=jax.jit(lambda u,v:original(View(u=u,v=v))).lower(view.u,view.v).compiler_ir("hlo").as_hlo_text()
    normalize=lambda t:re.sub(r"^HloModule [^,]+", "HloModule canonical", t)
    assert normalize(actual)==normalize(legacy)
    state,_=real_state(view)
    assert state.sfc_wspd is None
    from gpuwrf.runtime.aot_cheap_key import global_trace_env_hash
    off=global_trace_env_hash();monkeypatch.setenv(FLAG,"1")
    assert global_trace_env_hash()!=off
    with pytest.raises(ValueError,match="WSPD is missing"):
        P._surface_terms(view,diag.fluxes)


@pytest.mark.parametrize("field",["wspd","wind","drag","u","v"])
@pytest.mark.parametrize("bad",[np.nan,np.inf,-np.inf])
@pytest.mark.parametrize("side",["candidate","reference","both"])
def test_numeric_gate_rejects_nonfinite(field,bad,side):
    got={field:np.ones(96)};expected={field:np.ones(96)}
    if side in {"candidate","both"}:got[field][7]=bad
    if side in {"reference","both"}:expected[field][7]=bad
    with pytest.raises(AssertionError,match="nonfinite"):
        finite_close(got,expected,rtol=1e-7,atol=0.)
