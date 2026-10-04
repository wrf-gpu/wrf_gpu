"""Pristine-WRF ice-process oracle for the native-REAL Thompson fixes (FINDINGS B52, TH08).

Fixture: one pristine ``mp_gt_driver`` step (mp8, dt 54 s) on adversarial columns built by
proofs/thompson/ice_process_oracle (module_mp_thompson.F, gfortran, REAL inputs as in WRF):
A* ice-subsaturated ice + snow where one sublimation term alone exceeds rate_max (:2656/:2690),
B* nucleation-active ice + snow where the joint deposition limiter binds (pri_inu, :2862-2868),
C* supercooled rain + little ice where the ice limiter binds (prg_rci formed once, :2733),
D a control where no limiter binds. Gates were registered before the tests were written; each fix
has a mutant that removes exactly that fix and must fail its gate (E39/E102).
"""
import contextlib
import inspect
import os
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.kernels import phys_thompson_full as full
from gpuwrf.physics import thompson_column as tc

FIXTURE = Path(__file__).with_name("fixtures") / "ice_process_columns.npz"
C24 = dict(GPUWRF_THOMPSON_NATIVE_REAL="1", GPUWRF_THOMPSON_COLUMN_SED="1", GPUWRF_THOMPSON_SED_FP32="1",
           GPUWRF_THOMPSON_COLUMN_LAYOUT="1", GPUWRF_THOMPSON_FULL_COLUMN="1",
           GPUWRF_THOMPSON_FULL_COLUMN_EARLY_EXIT="1", GPUWRF_THOMPSON_SED_PREP_FUSED="0")
MUTANTS = {  # name -> (fixed source line(s), the same lines with the fix removed)
    "no_sublimation_clamp": ("sublimation_floor = ((state.qv - qvsi) * state.rho / float(dt) * 0.999).astype(jnp.float64)",
                             "sublimation_floor = None"),
    "no_inu_ratio": ("inu_mass = pri_inu * deposition_ratio * float(dt) / state.rho  # WRF scales pri_inu too (:2868)",
                     "pass"),
    "prg_rci_rescaled": ("    if not _native_real_enabled():\n        prg_rci = pri_rci + prr_rci\n",
                         "    prg_rci = pri_rci + prr_rci\n"),
}
FIELDS = ("qv", "qi", "qs", "th")


@contextlib.contextmanager
def _native_real_env():
    saved = {key: os.environ.get(key) for key in C24}
    os.environ.update(C24)
    jax.clear_caches()
    try:
        yield
    finally:
        for key, value in saved.items():
            os.environ.pop(key, None) if value is None else os.environ.__setitem__(key, value)
        jax.clear_caches()


def _mutant(name):
    fixed, removed = MUTANTS[name]
    src = textwrap.dedent(inspect.getsource(tc._ice_sources_with_process_flags))
    assert src.count(fixed) == 1, name
    namespace = dict(vars(tc))
    exec(compile(src.replace(fixed, removed), tc.__file__, "exec"), namespace)
    return namespace["_ice_sources_with_process_flags"]


def _state(z):
    f = lambda name: jnp.asarray(z[f"in_{name}"], jnp.float32)  # noqa: E731
    T, p, qv = f("th") * f("pii"), f("p"), f("qv")
    zero = jnp.zeros_like(qv)
    return tc.ThompsonColumnState(qv=qv, qc=f("qc"), qr=f("qr"), qi=f("qi"), qs=f("qs"), qg=f("qg"), Ni=f("ni"),
                                  Nr=f("nr"), Ns=zero, Ng=zero, T=T, p=p,
                                  rho=tc.density_from_pressure_temperature(p, T, qv), dz=f("dz8w"), w=f("w"))


def _run(z, mutant=None):
    original = tc._ice_sources_with_process_flags
    with _native_real_env():
        try:
            if mutant:
                tc._ice_sources_with_process_flags = _mutant(mutant)
            out, _ppt = full.full_column(_state(z), float(z["dt"]), interpret=True)
        finally:
            tc._ice_sources_with_process_flags = original
    g = lambda a: np.asarray(a, np.float64)  # noqa: E731
    return {"qv": g(out.qv), "qi": g(out.qi), "qs": g(out.qs), "th": g(out.T) / np.asarray(z["in_pii"], np.float64)}


def _errors(port, z):
    """Max |port - WRF| per column/field, relative to the column's WRF field maximum."""
    result = {}
    for i, name in enumerate(z["names"]):
        for field in FIELDS:
            wrf = np.asarray(z[f"wrf_{field}"][i], np.float64)
            result[str(name), field] = float(np.abs(port[field][i] - wrf).max() / max(np.abs(wrf).max(), 1e-9))
    return result


@pytest.fixture(scope="module")
def oracle():
    z = dict(np.load(FIXTURE))
    return z, _run(z)


def test_fixed_port_matches_pristine_wrf_on_ice_process_columns(oracle):
    z, port = oracle
    err = _errors(port, z)
    for col in ("A1_subl_snowdom", "A2_subl_cold", "B1_inu_ssati", "B2_inu_watersat", "D_control"):
        assert err[col, "qv"] <= 1e-6 and err[col, "th"] <= 1e-6, (col, err[col, "qv"], err[col, "th"])
    for col in ("A1_subl_snowdom", "A2_subl_cold"):
        assert err[col, "qi"] <= 1e-5 and err[col, "qs"] <= 1e-5, (col, err[col, "qi"], err[col, "qs"])


def test_sublimation_clamp_is_load_bearing(oracle):
    z, _port = oracle
    err = _errors(_run(z, "no_sublimation_clamp"), z)
    assert err["A1_subl_snowdom", "qi"] > 1e-3 and err["A2_subl_cold", "qi"] > 1e-3, err


def test_inu_deposition_ratio_is_load_bearing(oracle):
    z, port = oracle
    fixed, mutant = _errors(port, z), _errors(_run(z, "no_inu_ratio"), z)
    assert mutant["B1_inu_ssati", "qv"] > 1e-5, mutant["B1_inu_ssati", "qv"]
    assert mutant["B2_inu_watersat", "qv"] >= 5 * fixed["B2_inu_watersat", "qv"], (mutant, fixed)


def test_fixes_are_inert_where_no_limiter_binds(oracle):
    z, port = oracle
    i = list(z["names"]).index("D_control")
    for name in MUTANTS:
        other = _run(z, name)
        assert all(np.array_equal(port[f][i], other[f][i]) for f in FIELDS), name


def test_prg_rci_is_formed_once_before_the_limiters():
    """WRF :2733: graupel gains the UNSCALED pri_rci + prr_rci even when the ice limiter (:2888-2897)
    scales pri_rci; ice loss is unchanged. The column oracle cannot resolve this (the effect is bounded
    by the existing ice), so it is checked on the C1 inputs, where the ice limiter binds."""
    z = dict(np.load(FIXTURE))
    i = list(z["names"]).index("C1_rci_graupel")
    one = {key: (value[i:i + 1] if key.startswith("in_") else value) for key, value in z.items()}
    dt = float(z["dt"])
    results = {}
    with _native_real_env():
        state = tc._real_state(tc._warm_rain_collection(_state(one), dt))
        cold = tc._cold_collection_rates(state, dt, tc.COLD_COLLECTION_TABLES)
        for name, fn in (("fixed", tc._ice_sources_with_process_flags), ("mutant", _mutant("prg_rci_rescaled"))):
            out = fn(state, dt, cold_collection_rates=cold)[0]
            results[name] = {f: np.asarray(getattr(out, f), np.float64) for f in ("qi", "qg", "qr")}
    fixed, mutant = results["fixed"], results["mutant"]
    gain = fixed["qg"] - mutant["qg"]
    assert np.array_equal(fixed["qi"], mutant["qi"]) and np.array_equal(fixed["qr"], mutant["qr"])
    assert gain.max() > 0.0 and np.all(gain >= 0.0)
    assert np.all(gain <= np.asarray(one["in_qi"], np.float64) + 1e-12)
