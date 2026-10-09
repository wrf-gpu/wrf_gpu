"""EXR critic (b-core): with pi_phy supplied, NO MYNN consumer may fall back to the pressure-derived Exner.

Poison ``_exner_from_pressure`` (the only pressure->Exner helper reachable inside the step) with NaN: both whole entries
(retained + native), SGS cloud + EDMF on, cloudmix off/on, must return finite outputs bitwise equal to the unpoisoned run.
Deletion-sensitive for every consumer that would revert to ``_exner_from_pressure(state.p)`` (thl, thlv, mean tendencies,
cloudmix, condensation, _column_exner); the inline supplier Exner stays covered by the plume-supplier spy test.
"""
import contextlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import mynn_pbl as P
from gpuwrf.kernels import phys_mynn_columns as N
from test_mynn_dheat import batch, FIXTURE as HEAT


def _leaves(out):
    return [np.asarray(x) for x in jax.tree.leaves(out)]


@pytest.mark.parametrize('cloudmix', ['0', '1'])
@pytest.mark.parametrize('entry', ['retained', 'native'])
def test_supplied_pi_reaches_every_consumer(entry, cloudmix, monkeypatch):
    fn = {'retained': P._step_mynn_pbl_impl, 'native': N._advance_native}[entry]
    # the retained (legacy) EDMF scan is f64-only (x64 on), as in the product's force_fp64 path
    ctx = jax.enable_x64(True) if entry == 'retained' else contextlib.nullcontext()
    with ctx:
        _check(fn, entry, cloudmix, monkeypatch)


def _check(fn, entry, cloudmix, monkeypatch):
    state, flux = batch(json.loads(HEAT.read_text())['rows'][0])
    if entry == 'retained':
        state, flux = (jax.tree.map(lambda x: jnp.asarray(x, jnp.float64), t) for t in (state, flux))
    pi = P._exner_from_pressure(state.p) * jnp.asarray(.9999, state.p.dtype)
    state = state.replace(exner=pi)
    monkeypatch.setattr(P, '_MYNN_SGS_CLOUD', True)
    monkeypatch.setenv('GPUWRF_MYNN_CLOUDMIX', cloudmix)
    run = lambda: _leaves(fn(state, 54., False, surface=flux, edmf=True, dx=9000.))  # noqa: E731
    jax.clear_caches()
    ref = run()
    for i, x in enumerate(ref):
        assert np.isfinite(x).all(), f'reference leaf {i} nonfinite'
    monkeypatch.setattr(P, '_exner_from_pressure', lambda p: jnp.full_like(p, jnp.nan))
    jax.clear_caches()
    got = run()
    for i, x in enumerate(got):
        assert np.isfinite(x).all(), f'leaf {i} nonfinite: a consumer used the pressure-derived Exner'
    assert len(got) == len(ref)
    for i, (a, b) in enumerate(zip(got, ref)):
        np.testing.assert_array_equal(a, b, err_msg=f'leaf {i}')
    # control: the supplied pi is live in this configuration (a different pi changes the step)
    jax.clear_caches()
    alt = _leaves(fn(state.replace(exner=pi * jnp.asarray(1.001, pi.dtype)), 54., False, surface=flux, edmf=True, dx=9000.))
    assert any(not np.array_equal(a, b) for a, b in zip(alt, ref))
