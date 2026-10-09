"""Original REAL4 mym_length CASE1 with current-step plume enhancement."""
from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import mynn_pbl as P
from gpuwrf.kernels import phys_mynn_columns as N
from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
from gpuwrf.runtime import aot_cheap_key as K

FIXTURE = Path(__file__).parent / "fixtures/mynn_elb_mf_wrf.json"
PACKET = json.loads(FIXTURE.read_text())


@pytest.fixture(autouse=True)
def clear_flag_traces():
    jax.clear_caches()
    yield
    jax.clear_caches()


def batch(row, dtype=jnp.float32):
    v = np.asarray(row["values"], np.float32)
    names = ("u", "v", "w", "theta", "qv", "tke", "p", "rho", "dz", "qc", "qi", "qs", "qsq")
    vals = {n: jnp.asarray(v[:, :, i], dtype) for i, n in enumerate(names)}
    z = jnp.zeros_like(vals["qv"])
    s = P.MynnPBLColumnState(**vals, km=z, kh=z, el=z,
        qc_bl=jnp.asarray(row["qc_bl"], dtype), qi_bl=jnp.asarray(row["qi_bl"], dtype),
        cldfra_bl=jnp.asarray(row["cldfra_bl"], dtype))
    f = np.asarray(row["flux"], np.float32)
    flux = SurfaceFluxes(ustar=jnp.asarray(f[:, 0], dtype), theta_flux=jnp.asarray(f[:, 1], dtype),
        qv_flux=jnp.asarray(f[:, 2], dtype), fltv=jnp.asarray(f[:, 3], dtype), xland=jnp.asarray(f[:, 4], dtype),
        t_skin=jnp.asarray(f[:, 5], dtype), rhosfc=jnp.asarray(f[:, 6], dtype),
        tau_u=jnp.zeros(len(f), dtype), tau_v=jnp.zeros(len(f), dtype))
    return s, flux


def gate(candidate, reference, inputs):
    for role, fields in (("candidate", candidate), ("reference", reference), ("input", inputs)):
        for name, value in fields.items():
            assert np.isfinite(value).all(), f"{role} {name} nonfinite"
    assert candidate.keys() == reference.keys()
    assert np.max(np.abs(candidate["qkw"] - reference["qkw"])) <= 2e-6
    for i in range(len(candidate["el"])):
        scale = max(float(np.abs(reference["el"][i]).max()), 1e-30)
        assert np.abs(candidate["el"][i] - reference["el"][i]).max() / scale <= .05, "original MF mixing length differs"


@pytest.mark.parametrize("row", PACKET["rows"], ids=lambda x: f"tau{x['tau']}")
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64], ids=["native", "retained"])
def test_pristine_mym_length_with_active_plumes(monkeypatch, row, dtype):
    monkeypatch.setattr(P, "_MYNN_ELB_MF", True)
    s, f = batch(row, dtype)
    mf = {"s_aw": jnp.asarray(row["s_aw"], dtype)}
    kw = P._mynn_length_mf_kwargs(s, mf)
    np.testing.assert_allclose(np.asarray(kw["qkw_mf"]), row["qkw_mf"], rtol=1e-6, atol=1e-8)
    def call(s, qke, dtv, fltv, ustar, xland, pblh, psig, rmol, qmf):
        return P._mym_length_option1_with(s, qke, dtv, fltv, ustar, 1000., xland,
            pblh=pblh, psig_bl=psig, rmol=rmol, qkw_mf=qmf)
    with jax.enable_x64(dtype == jnp.float64):
        out = jax.jit(call)(s, 2*s.tke, jnp.asarray(row["dtv"], dtype), f.fltv, f.ustar, f.xland,
            jnp.asarray(row["pblh"], dtype), jnp.asarray(row["psig_bl"], dtype),
            jnp.asarray(row["rmol"], dtype), kw["qkw_mf"])
    ref = np.asarray(row["WRF_MF"], np.float32)
    gate({"qkw": np.asarray(out[0]), "el": np.asarray(out[1])},
         {"qkw": ref[:, :, 0], "el": ref[:, :, 1]}, {n: np.asarray(v) for n, v in row.items() if n != "tau"})
    assert np.max(np.abs(ref[:, :, 1] - np.asarray(row["WRF_noMF"])[:, :, 1])) > 20, "actual MF witness inactive"


def test_volume_flux_uses_previous_interface_and_off_is_absent(monkeypatch):
    row = PACKET["rows"][0];s, _ = batch(row)
    mf = {"s_aw": jnp.asarray(row["s_aw"])}
    monkeypatch.setattr(P, "_MYNN_ELB_MF", False)
    assert P._mynn_length_mf_kwargs(s, mf) == {}
    monkeypatch.setattr(P, "_MYNN_ELB_MF", True)
    assert P._mynn_length_mf_kwargs(s, None) == {}
    q = np.asarray(P._mynn_length_mf_kwargs(s, mf)["qkw_mf"])
    assert np.all(q[:, 0] == 0)
    np.testing.assert_allclose(q, row["qkw_mf"], rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("entry", ["native", "retained"])
@pytest.mark.parametrize("sgs", [False, True])
def test_both_whole_callers_supply_current_mf(monkeypatch, entry, sgs):
    monkeypatch.setattr(P, "_MYNN_ELB_MF", True)
    monkeypatch.setattr(P, "_MYNN_SGS_CLOUD", sgs)
    row=PACKET["rows"][0];s,f=batch(row)
    witness={"s_aw":jnp.asarray(row["s_aw"],s.rho.dtype),"edmf_a":jnp.asarray(row["edmf_a"],s.rho.dtype),
             "edmf_qc":jnp.zeros_like(s.qv),"edmf_qt":s.qv}
    calls=[]
    def mf_call(*args,**kw):calls.append(1);return witness
    monkeypatch.setattr(N,"_native_edmf_arrays",mf_call)
    monkeypatch.setattr(P,"_edmf_arrays_from_state",mf_call)
    expected=np.asarray(row["qkw_mf"],np.float32)
    class Stop(Exception):pass
    def capture(*args,**kw):
        assert len(calls)==1,"current-step plume must be computed before turbulence"
        assert "qkw_mf" in kw,"MF mixing-length supplier missing"
        np.testing.assert_allclose(np.asarray(kw["qkw_mf"]),expected,rtol=1e-6,atol=1e-8)
        raise Stop()
    monkeypatch.setattr(P,"_mym_turbulence",capture)
    with jax.enable_x64(False),pytest.raises(Stop):
        (N._advance_native if entry=="native" else P._step_mynn_pbl_impl)(s,6.,False,surface=f,edmf=True,dx=1000.)


def test_resolved_flag_key(monkeypatch):
    assert ("gpuwrf.physics.mynn_pbl","_MYNN_ELB_MF") in K.IMPORT_TIME_ENV_CONSTANTS
    monkeypatch.setattr(P,"_MYNN_ELB_MF",False);a=K.module_const_env_hash()
    monkeypatch.setenv("GPUWRF_MYNN_ELB_MF","1");assert K.module_const_env_hash()==a
    monkeypatch.setattr(P,"_MYNN_ELB_MF",True);assert K.module_const_env_hash()!=a


INPUT_FIELDS=tuple(k for k in PACKET["rows"][0] if k!="tau")
@pytest.mark.parametrize("field",INPUT_FIELDS)
@pytest.mark.parametrize("bad",[np.nan,np.inf,-np.inf])
def test_gate_rejects_bad_inputs(field,bad):
    row=PACKET["rows"][0];inp={n:np.asarray(v).copy() for n,v in row.items() if n!="tau"}
    ref=np.asarray(row["WRF_MF"]);good={"qkw":ref[:,:,0],"el":ref[:,:,1]}
    inp[field].flat[0]=bad
    with pytest.raises(AssertionError,match="nonfinite"):gate(good,good,inp)


@pytest.mark.parametrize("field",["qkw","el"])
@pytest.mark.parametrize("bad",[np.nan,np.inf,-np.inf])
@pytest.mark.parametrize("role",["candidate","reference"])
def test_gate_rejects_bad_outputs(field,bad,role):
    ref=np.asarray(PACKET["rows"][0]["WRF_MF"]);a={"qkw":ref[:,:,0].copy(),"el":ref[:,:,1].copy()};b={n:v.copy() for n,v in a.items()}
    (a if role=="candidate" else b)[field].flat[0]=bad
    with pytest.raises(AssertionError,match="nonfinite"):gate(a,b,{})
