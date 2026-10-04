"""Rain01-grounded runtime wiring regression; stubs are NOT a WRF field oracle."""
from dataclasses import replace
import json
from pathlib import Path
import importlib.util
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.runtime import operational_mode as op
from gpuwrf.io.lower_boundary import LowerBoundary

CASES=json.loads(Path(__file__).with_name("B35_RAIN01.json").read_text())["cases"]

def fixture():
 p=Path(__file__).with_name("test_microphysics_order.py")
 spec=importlib.util.spec_from_file_location("b35_order_fixture",p)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 return m._fixture()

@pytest.mark.parametrize("domain",["d02","d03"])
@pytest.mark.parametrize("order",[False,True])
@pytest.mark.parametrize("aux4",[False,True])
def test_rain01_increment_enters_exactly_once(monkeypatch,domain,order,aux4):
 monkeypatch.setenv("GPUWRF_MICROPHYSICS_WRF_ORDER",str(int(order)))
 observed=CASES[domain];carry,nml=fixture()
 entry=observed["entry"];phys=observed["physics"]
 s=carry.state.replace(_cast=False,qr=jnp.full_like(carry.state.qr,entry["qr"],dtype=jnp.float64),Nr=jnp.full_like(carry.state.Nr,entry["Nr"],dtype=jnp.float64),qc=jnp.full_like(carry.state.qc,entry["qc"],dtype=jnp.float64),xland=jnp.full_like(carry.state.xland,2.))
 carry=carry.replace(state=s)
 shape=s.t_skin.shape
 boundary=LowerBoundary(jnp.asarray([0],jnp.int32),jnp.full((1,)+shape,299.),jnp.zeros((1,)+shape),jnp.full((1,)+shape,50.),jnp.zeros((1,)+shape)) if aux4 else None
 nml=replace(nml,lower_boundary=boundary,force_fp64=True)
 events=[]
 delta_qr=phys["qr"]-entry["qr"];delta_nr=phys["Nr"]-entry["Nr"];delta_qc=phys["qc"]-entry["qc"]
 def mp(state,dt):
  events.append("mp")
  return state.replace(qr=state.qr+delta_qr,Nr=state.Nr+delta_nr,qc=state.qc+delta_qc)
 def rk(carry,nml,**kw):
  events.append("rk")
  # The transport contribution is isolated. RK must see pre-MP water/number.
  np.testing.assert_array_equal(np.asarray(carry.state.qr),np.asarray(s.qr))
  np.testing.assert_array_equal(np.asarray(carry.state.Nr),np.asarray(s.Nr))
  return carry
 monkeypatch.setattr(op,"thompson_adapter",mp)
 monkeypatch.setattr(op,"_rk_scan_step",rk)
 monkeypatch.setattr(op,"surface_adapter",lambda state,*a,**kw:state)
 after,_=op._physics_boundary_step_with_limiter_diagnostics(carry,nml,jnp.int32(1),run_radiation=False)
 np.testing.assert_allclose(np.asarray(after.state.qr),phys["qr"],rtol=0,atol=1e-16)
 np.testing.assert_allclose(np.asarray(after.state.Nr),phys["Nr"],rtol=1e-14,atol=0)
 np.testing.assert_allclose(np.asarray(after.state.qc),phys["qc"],rtol=0,atol=1e-16)
 assert events==(["rk","mp"] if order else ["mp","rk"])
 if aux4:np.testing.assert_array_equal(np.asarray(after.state.t_skin),299.)
