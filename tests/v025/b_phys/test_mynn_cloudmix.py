"""Caller-floor and unified flag regressions for the WRF fidelity component."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import jax.numpy as jnp
import pytest
from gpuwrf.kernels.phys_mynn_cloudmix import cloudmix_enabled,pressure_mass_weights,moisture_check
from gpuwrf.physics import mynn_pbl as P


def test_source_surface_mass_floor_activates_conservative_repair():
    fixture=json.loads((Path(__file__).parent/'fixtures/moisture_surface_floor.json').read_text())
    values=np.asarray(fixture['values'],np.float32)
    dp,ex,v,c,i,t=[jnp.asarray(values[:,k][None,:]) for k in range(6)]
    state=SimpleNamespace(rho=dp/jnp.float32(9.81),dz=jnp.ones_like(dp))
    weights=pressure_mass_weights(state)
    assert float(weights[0,0]) >= .5*float(weights[0,1])
    actual=moisture_check(v,c,i,t,weights,ex)
    data=np.column_stack([np.asarray(value)[0] for value in actual])
    delta=np.abs(data-np.asarray(fixture['expected']))
    assert delta[:,:3].max() < fixture['species_abs_bound']
    assert delta[:,3].max() < fixture['theta_abs_bound']


@pytest.mark.parametrize('value,enabled',[(None,False),('0',False),('false',False),('no',False),('off',False),('',False),('1',True),('true',True),('yes',True),('on',True),('1 ',True),(' TRUE ',True)])
def test_one_cloudmix_predicate_controls_pblh_and_dispatch(monkeypatch,value,enabled):
    if value is None:monkeypatch.delenv('GPUWRF_MYNN_CLOUDMIX',raising=False)
    else:monkeypatch.setenv('GPUWRF_MYNN_CLOUDMIX',value)
    assert cloudmix_enabled() is enabled
    # A real-like threshold crossing pins the PBLH caller to that same predicate.
    dz=jnp.full((1,6),50.,jnp.float64)
    theta=jnp.array([[290.,290.,290.6,291.1,291.4,292.]],jnp.float64)
    zero=jnp.zeros_like(theta)
    state=P.MynnPBLColumnState(zero,zero,zero,theta,zero,zero,jnp.full_like(theta,90000.),jnp.ones_like(theta),dz,zero,zero,zero)
    land=P._get_pblh(state,zero,jnp.array([1.]))
    water=P._get_pblh(state,zero,jnp.array([2.]))
    assert (float(land[0])>float(water[0])) is enabled
