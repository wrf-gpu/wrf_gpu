"""v0.15 S1 host-removal knob tests (mynn_edmf condensation / level-scan).

Pins the three properties the sprint relies on:
  1. DEFAULT-INERT: with no env set, `_condensation_edmf` keeps the v0.14
     fori_loop lowering (a `while` in StableHLO) at niter=50, and the plume
     level scan keeps unroll=1 -- the production graph stays Tier-S
     bit-identical to v0.14 (verified end-to-end by ab_s1_base.json 0/168).
  2. The unrolled lowering at EQUAL niter is value-identical to the fori
     lowering (loop peeling does not reassociate). GPU bitwise is additionally
     proven by proofs/perf/v015/cond_niter_oracle.json.
  3. niter is an explicit override; env supplies the default only.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from gpuwrf.physics import mynn_edmf as me

jax.config.update("jax_enable_x64", True)


def _inputs():
    rng = np.random.default_rng(7)
    qt = jnp.asarray(rng.uniform(0.0, 0.02, (64,)))
    thl = jnp.asarray(rng.uniform(260.0, 320.0, (64,)))
    p = jnp.asarray(rng.uniform(2.0e4, 1.0e5, (64,)))
    z = jnp.full((64,), 500.0)
    return qt, thl, p, z


def test_default_env_keeps_fori_lowering(monkeypatch):
    monkeypatch.delenv("GPUWRF_MYNN_COND_NITER", raising=False)
    monkeypatch.delenv("GPUWRF_MYNN_COND_UNROLL", raising=False)
    monkeypatch.delenv("GPUWRF_MYNN_EDMF_LEVEL_UNROLL", raising=False)
    assert me._cond_niter() == 50
    assert me._cond_unroll() is False
    assert me._edmf_level_unroll() == 1
    qt, thl, p, z = _inputs()
    txt = jax.jit(me._condensation_edmf).lower(qt, thl, p, z).as_text()
    assert "stablehlo.while" in txt  # exact v0.14 lowering class


def test_unroll_env_removes_while(monkeypatch):
    monkeypatch.setenv("GPUWRF_MYNN_COND_UNROLL", "1")
    monkeypatch.setenv("GPUWRF_MYNN_COND_NITER", "16")
    jax.clear_caches()  # env knobs are trace-time: drop the cached default trace
    qt, thl, p, z = _inputs()
    txt = jax.jit(me._condensation_edmf).lower(qt, thl, p, z).as_text()
    assert "stablehlo.while" not in txt


@pytest.mark.parametrize("niter", [8, 16, 50])
def test_unrolled_matches_fori_at_equal_niter(monkeypatch, niter):
    qt, thl, p, z = _inputs()
    monkeypatch.setenv("GPUWRF_MYNN_COND_UNROLL", "0")
    thv_f, qc_f = jax.jit(lambda *a: me._condensation_edmf(*a, niter=niter))(qt, thl, p, z)
    monkeypatch.setenv("GPUWRF_MYNN_COND_UNROLL", "1")
    thv_u, qc_u = jax.jit(lambda *a: me._condensation_edmf(*a, niter=niter))(qt, thl, p, z)
    assert (np.asarray(thv_f) == np.asarray(thv_u)).all()
    assert (np.asarray(qc_f) == np.asarray(qc_u)).all()


def test_niter_cap_residual_is_below_wrf_exit_threshold(monkeypatch):
    """qc(16) vs qc(50) stays under WRF's OWN convergence acceptance.

    WRF exits this loop when |QC-QCold| < diff with diff = 1.e-6 kg/kg
    (module_bl_mynnedmf.F:6815, "usually converges in < 8 iterations"), i.e.
    any state within 1e-6 of the fixed point is a WRF-accepted answer. The
    16-iteration residual measured here (~1.6e-7 worst case on this sweep,
    GPU envelope sweep in proofs/perf/v015/cond_niter_oracle.json) is below
    that threshold: niter=16 is WRF-faithful by WRF's own criterion.
    """
    monkeypatch.setenv("GPUWRF_MYNN_COND_UNROLL", "0")
    jax.clear_caches()
    qt, thl, p, z = _inputs()
    _, qc50 = jax.jit(lambda *a: me._condensation_edmf(*a, niter=50))(qt, thl, p, z)
    _, qc16 = jax.jit(lambda *a: me._condensation_edmf(*a, niter=16))(qt, thl, p, z)
    assert float(jnp.max(jnp.abs(qc16 - qc50))) < 1.0e-6  # WRF diff threshold
