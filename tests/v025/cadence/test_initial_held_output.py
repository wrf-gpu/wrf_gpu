"""Initialization/hold wiring regression; sentinels are not WRF-fidelity evidence."""
import dataclasses
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gpuwrf.runtime import operational_mode as op
from gpuwrf.integration import nested_pipeline as npl
from gpuwrf.coupling.physics_couplers import RRTMGRadiationDiagnostics
from gpuwrf.diagnostics import census as cs

spec=importlib.util.spec_from_file_location('cadence_fixture',Path(__file__).resolve().parents[2]/'test_rrtm_lw_operational_wiring.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)

@pytest.mark.parametrize('counted',[False,True])
def test_init_writer_and_first_hold_share_one_full_solve(monkeypatch,counted):
 grid=fixture._grid(ny=2,nx=2,nz=8);state=fixture._state(grid)
 nml=dataclasses.replace(fixture._namelist(grid),use_noahmp=True,noahmp_land=None,radiation_cadence_steps=33)
 values=[jnp.full((2,2),float(i+1),jnp.float64) for i in range(15)];values[10]=jnp.zeros((2,2),jnp.int32)
 # Full production diagnostics carry CLDFRA on the mass grid. Initialization
 # seeds it at zero; the stub must return the same concrete scan structure.
 rad=RRTMGRadiationDiagnostics(*values,cloud_fraction=jnp.zeros_like(state.qv));calls=[]
 def init_solve(*args,**kwargs):calls.append(1);return rad
 monkeypatch.setattr(op,'rrtmg_radiation_diagnostics',init_solve)
 carry=op._initial_carry_for_run(state,nml)
 forcing,full=op.noahmp_initial_rad(carry.state,nml,_with_diagnostics=True)
 carry=carry.replace(noahmp_land=object(),noahmp_rad=forcing,radiation_diagnostics=full)
 assert len(calls)==1
 counters=cs.initial_census() if counted else None
 run=jax.jit(lambda pred:op._refresh_noahmp_rad(state,nml,jnp.array(54.),pred,
     carry.noahmp_rad,held_diagnostics=carry.radiation_diagnostics,census=counters))
 held=run(jnp.asarray(False))
 if counted:held,_=held
 for actual,expected in zip(jax.tree.leaves(held[1]),jax.tree.leaves(rad),strict=True):
  np.testing.assert_array_equal(actual,expected)
 before_writer_calls=len(calls)  # JAX traced the unexecuted refresh branch once.
 def forbidden(*args,**kwargs):raise AssertionError('output radiation recomputation')
 monkeypatch.setattr(op,'rrtmg_radiation_diagnostics',forbidden)
 z=jnp.zeros((2,2),jnp.float64)
 monkeypatch.setattr(op,'surface_layer_diagnostics',lambda *a,**k:SimpleNamespace(hfx=z,lh=z,t2=z,pblh=z,u10=z,v10=z))
 monkeypatch.setattr(op,'overlay_noahmp_land_diagnostics',lambda *a,**k:(z,z,z,z))
 monkeypatch.setattr(op,'_lane_noahmp_static',lambda *a:None)
 for lead in [0.,54.]:
  out=npl._noahmp_surface_diagnostics_for_output(state,nml,datetime(2026,7,26,tzinfo=timezone.utc),
      lead_seconds=lead,noahmp_land=carry.noahmp_land,noahmp_rad=held[0],radiation_diagnostics=held[1])
  np.testing.assert_array_equal(out['LWDNB'],out['GLW'])
  np.testing.assert_array_equal(out['LWUPT'],rad.lw_toa_up)
  np.testing.assert_array_equal(out['LWUPB'],rad.glw_up)
 assert len(calls)==before_writer_calls
