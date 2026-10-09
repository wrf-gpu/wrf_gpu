"""GPUWRF_ACOUSTIC_NO_MU_FLOOR (b-core ledger row 7, default off): drop the non-WRF dry-mass floor (CPU).

The native mass kernel (dyn_acoustic_fp32, specified/nested domains) scales the mu tendency when mu would fall below
max(1, .5*mut) (its scale 0 for a nonfinite tendency masks nothing: NaN*0 = NaN); WRF advance_mu_t
(module_small_step_em.F) has no floor. Release-case census
(native_mass_guard_events) is 0 on WN3 0227 3-nest strict 3 h, the W4 0227/0502/0614/0118 sweeps and PROD LW7/LW9, so the
floor never changed a value there. Truth: the UNCHANGED pristine advance_mu_t (binary32 ctypes, pristine_oracle.py) on
the REAL PROD d01 stage fixture: (1) where the floor is idle ON == OFF bitwise; (2) with an adversarial mu tendency the
floor engages: ON == pristine at Tier-P, OFF is clamped away from it (deletion-sensitive); (3) the floor never masked a
nonfinite tendency (its scale 0 times NaN stays NaN): both arms propagate it like WRF -- no hidden repair is removed.
"""
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.kernels import dyn_acoustic_fp32 as dyn

PHASE1 = Path("<USER_HOME>/wrf_gpu2_lanes/b-core/phase1")
BLOCK = (slice(30, 34), slice(50, 54))


def _finite_equal(actual, desired):
    actual, desired = np.asarray(actual), np.asarray(desired)
    assert np.isfinite(actual).all(), "nonfinite candidate"
    assert np.isfinite(desired).all(), "nonfinite reference"
    np.testing.assert_array_equal(actual, desired)


@pytest.fixture(scope="module")
def d01():
    libs = {n: PHASE1 / f"pristine_corrected/{n}/oracle.so" for n in ("reference", "fast", "off")}
    if not (PHASE1 / "fixtures/d01.npz").exists() or not all(p.exists() for p in libs.values()):
        pytest.skip("PROD d01 stage fixture / pristine acoustic oracle unavailable")
    sys.path.insert(0, str(Path(__file__).parent))
    from bench_acoustic import fixture
    state, cfg, _, _ = fixture(PHASE1 / "fixtures", "d01")
    assert cfg.specified or cfg.nested   # the floor only exists on specified/nested domains
    return state, replace(cfg, w_damping=0), libs


def _tier_p_rms(got, want, fast, off):
    """(rms(got - want), limit) on BLOCK; E200: every operand -- candidate AND the three pristine mu arrays that form the
    reference and its fast/off envelope -- must be finite BEFORE any envelope/limit arithmetic."""
    for role, a in (("candidate", got), ("reference", want), ("fast envelope", fast), ("off envelope", off)):
        assert np.isfinite(np.asarray(a)).all(), f"nonfinite {role}"
    env = np.abs(np.asarray(fast) - np.asarray(off))
    limit = 2e-5 * max(1.0, float(np.sqrt(np.mean(want[BLOCK] ** 2)))) + 2.0 * float(np.sqrt(np.mean(env[BLOCK] ** 2)))
    return float(np.sqrt(np.mean((np.asarray(got) - want)[BLOCK] ** 2))), limit


def _substep(monkeypatch, no_floor, state, cfg):
    monkeypatch.setattr(dyn, "_NO_MU_FLOOR", no_floor)
    co = jax.jit(lambda s: dyn.calc_coef_fp32(s, cfg, interpret=True))(state)   # fresh traces per arm
    out, events = jax.jit(lambda s: dyn.acoustic_substep_fp32(
        s, coefficients=co, cfg=cfg, interpret=True, return_guard_events=True))(state)
    return out, np.asarray(events)


def _adversarial(state, cfg, value=None):
    mut = np.asarray(state.mut, np.float32)
    mu_tend = np.asarray(state.mu_tend, np.float32).copy()
    mu_tend[BLOCK] = np.float32(-0.8) * mut[BLOCK] / np.float32(cfg.dt) if value is None else value
    return state.replace(mu_tend=jnp.asarray(mu_tend))


def test_bit_neutral_where_the_floor_is_idle(d01, monkeypatch):
    state, cfg, _ = d01
    off, ev_off = _substep(monkeypatch, False, state, cfg)
    on, ev_on = _substep(monkeypatch, True, state, cfg)
    assert int(ev_off.sum()) == 0 and int(ev_on.sum()) == 0
    for name in dyn.EVOLVING_FIELDS:
        a, b = getattr(on, name), getattr(off, name)
        if a is not None:
            _finite_equal(np.asarray(a), np.asarray(b))


def test_adversarial_mu_tendency_matches_pristine_only_without_floor(d01, monkeypatch):
    from pristine_oracle import Oracle
    state, cfg, libs = d01
    state = _adversarial(state, cfg)
    ref = {}
    for name, lib in libs.items():
        o = Oracle(lib, state, cfg)
        for phase in range(1, 6):
            o.run(phase)
        ref[name] = np.asarray(o.field("mu"), np.float64)
    off, ev_off = _substep(monkeypatch, False, state, cfg)
    on, ev_on = _substep(monkeypatch, True, state, cfg)
    assert int(ev_off[BLOCK].sum()) == 16 and int(ev_on.sum()) == 0     # the floor engaged only in the OFF arm
    ny, nx = state.mut.shape
    want, fast, off_ref = (ref[n][:ny, :nx] for n in ("reference", "fast", "off"))
    got_on = np.asarray(on.mu_work, np.float64); got_off = np.asarray(off.mu_work, np.float64)
    rms_on, limit = _tier_p_rms(got_on, want, fast, off_ref)
    rms_off, _ = _tier_p_rms(got_off, want, fast, off_ref)
    assert rms_on <= limit                 # no floor == WRF
    assert rms_off > 100.0 * limit         # the floor departs from WRF
    outside = np.ones(want.shape, bool); outside[BLOCK] = False
    _finite_equal(got_on[outside], got_off[outside])


def test_nonfinite_tendency_propagates_in_both_arms(d01, monkeypatch):
    state, cfg, _ = d01
    state = _adversarial(state, cfg, value=np.float32(np.nan))
    off, _ = _substep(monkeypatch, False, state, cfg)
    on, _ = _substep(monkeypatch, True, state, cfg)
    assert not np.isfinite(np.asarray(off.mu_work)[BLOCK]).any()      # scale 0 * NaN = NaN: the floor never masked
    assert not np.isfinite(np.asarray(on.mu_work)[BLOCK]).any()       # WRF advance_mu_t propagates it too


def test_flag_is_cheap_keyed_and_default_off():
    import os
    from gpuwrf.runtime.aot_cheap_key import IMPORT_TIME_ENV_CONSTANTS
    assert ("gpuwrf.kernels.dyn_acoustic_fp32", "_NO_MU_FLOOR") in IMPORT_TIME_ENV_CONSTANTS
    if os.environ.get("GPUWRF_ACOUSTIC_NO_MU_FLOOR", "0") != "1":
        assert dyn._NO_MU_FLOOR is False


@pytest.mark.parametrize("side", ["candidate", "reference", "both"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_numeric_gate_rejects_nonfinite(side, bad):
    actual, desired = np.ones((2, 3)), np.ones((2, 3))
    if side in ("candidate", "both"):
        actual[0, 0] = bad
    if side in ("reference", "both"):
        desired[0, 0] = bad
    with pytest.raises(AssertionError, match="nonfinite"):
        _finite_equal(actual, desired)


@pytest.mark.parametrize("role", ["candidate", "reference", "fast envelope", "off envelope"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_envelope_gate_rejects_nonfinite(role, bad):
    arrays = {r: np.ones((40, 60)) for r in ("candidate", "reference", "fast envelope", "off envelope")}
    arrays[role][BLOCK][0, 0] = bad          # one nonfinite cell inside the scored block
    with pytest.raises(AssertionError, match=f"nonfinite {role}"):
        _tier_p_rms(arrays["candidate"], arrays["reference"], arrays["fast envelope"], arrays["off envelope"])
    arrays[role][BLOCK][0, 0] = 1.0
    assert _tier_p_rms(arrays["candidate"], arrays["reference"], arrays["fast envelope"], arrays["off envelope"])[0] == 0.0
