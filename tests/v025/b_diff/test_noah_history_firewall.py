"""Noah land-history exports leave the energy step through an optimization barrier (b-diff BD74).

The barrier is a fusion firewall only (A22/A23 class): values pass through bitwise; it is active
with the NOAHMP native-REAL + iteration-barrier release pair unless GPUWRF_NOAHMP_HISTORY_BARRIER=0.
"""
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.noahmp import energy


def _values():
    rng = np.random.default_rng(1)
    return {"PSN": jnp.asarray(rng.standard_normal((7, 9)), jnp.float32),
            "CHLEAF": jnp.asarray(rng.standard_normal((7, 9)), jnp.float32)}


@pytest.mark.parametrize("optout,expect", [(None, True), ("0", False)])
def test_history_firewall_dispatch_and_values(monkeypatch, optout, expect):
    monkeypatch.setenv("GPUWRF_NOAHMP_NATIVE_REAL", "1")
    monkeypatch.setenv("GPUWRF_NOAHMP_ITERATION_BARRIER", "1")
    if optout is None:
        monkeypatch.delenv("GPUWRF_NOAHMP_HISTORY_BARRIER", raising=False)
    else:
        monkeypatch.setenv("GPUWRF_NOAHMP_HISTORY_BARRIER", optout)
    values = _values()
    jaxpr = str(jax.make_jaxpr(lambda v: energy._history_firewall(v))(values))  # fresh lambda (E140)
    assert ("optimization_barrier" in jaxpr) is expect
    out = jax.jit(lambda v: energy._history_firewall(v))(values)
    assert all(np.asarray(out[k]).tobytes() == np.asarray(values[k]).tobytes() for k in values)


def test_history_firewall_off_without_release_pair(monkeypatch):
    monkeypatch.delenv("GPUWRF_NOAHMP_ITERATION_BARRIER", raising=False)
    jaxpr = str(jax.make_jaxpr(lambda v: energy._history_firewall(v))(_values()))
    assert "optimization_barrier" not in jaxpr


def test_energy_canopy_routes_history_through_firewall():
    source = inspect.getsource(energy.noahmp_energy_canopy)
    assert "history=(_history_firewall({" in source and "}) if history else None)" in source


def test_driver_routes_water_snow_history_through_same_firewall():
    # NF02 (b-thompson): RUNSF/RUNSB (noahmp_water_hydro) and SNOM_INCREMENT (snow ponding) join the
    # carried land history in noah_mp_step, outside the energy-canopy dict -> same firewall object.
    from gpuwrf.physics.noahmp import noahmp_driver

    source = inspect.getsource(noahmp_driver.noah_mp_step)
    assert "history=(_history_firewall({" in source and "}) if history else None)" in source
    assert '"RUNSF": runoff[0]' in source and '"SNOM_INCREMENT"' in source
    assert noahmp_driver._history_firewall is energy._history_firewall
