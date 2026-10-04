"""Pristine-WRF graupel oracle for the native-REAL Thompson mp8 graupel fall law (B52b, TH10).

Fixture: one pristine ``mp_gt_driver`` step (mp8, dt 54 s) built by proofs/thompson/ice_process_oracle
(``oracle.py build graupel``). WRF mp8 calls thompson_init without ng, so av_g/bv_g(idx_bg1) are
av_g_old 442 / bv_g_old 0.89 (:459-464) for every graupel term; sedimentation uses cgg(6)*ogg3 (:3758)
and re-diagnoses ng from the post-process rg (:3287-3300); rain-snow/rain-graupel rates use the entry
rr/nr (idx_r1 :2279), not the number after rain break-up. Columns: C* rain freezing onto graupel,
E1 sublimation, F1 melt, G1 riming, H1 fall only. The frozen mp8 oracles hold no graupel at all.
Each fix has a mutant that restores exactly the old port behaviour and must fail its gate (E39/E102).
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

FIXTURE = Path(__file__).with_name("fixtures") / "graupel_process_columns.npz"
C24 = dict(GPUWRF_THOMPSON_NATIVE_REAL="1", GPUWRF_THOMPSON_COLUMN_SED="1", GPUWRF_THOMPSON_SED_FP32="1",
           GPUWRF_THOMPSON_COLUMN_LAYOUT="1", GPUWRF_THOMPSON_FULL_COLUMN="1",
           GPUWRF_THOMPSON_FULL_COLUMN_EARLY_EXIT="1", GPUWRF_THOMPSON_SED_PREP_FUSED="0")
SED_MUTANTS = {  # name -> (fixed _fall_speeds snippet, the old port behaviour)
    "legacy_sed_moment": ("rhof * av_g * cgg63 * ilamg ** bv_g", "rhof * av_g * 6.0 * ORG3 * ilamg ** bv_g"),
    "carried_ng": ("None if native else state.Ng", "state.Ng"),
}
BODY_MUTANTS = {"cold_rates_post_warm": ("state if cold_entry is None else cold_entry", "state")}
LEGACY_CONSTANTS = (tc.AV_G_MP8, tc.BV_G_MP8, tc.CGG6_OVER_CGG3, tc.T1_QG_QC, tc.CGE9, tc.T2_SUBL_QG, tc.T2_MELT_QG, tc.CGE11)
FIELDS = ("qv", "qc", "qr", "qg", "th")


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


def _state(z):
    f = lambda name: jnp.asarray(z[f"in_{name}"], jnp.float32)  # noqa: E731
    T, p, qv = f("th") * f("pii"), f("p"), f("qv")
    zero = jnp.zeros_like(qv)
    return tc.ThompsonColumnState(qv=qv, qc=f("qc"), qr=f("qr"), qi=f("qi"), qs=f("qs"), qg=f("qg"), Ni=f("ni"),
                                  Nr=f("nr"), Ns=zero, Ng=zero, T=T, p=p,
                                  rho=tc.density_from_pressure_temperature(p, T, qv), dz=f("dz8w"), w=f("w"))


def _patched(mutant):
    """{tc attribute: replacement} restoring one old port behaviour."""
    if mutant is None:
        return {}
    if mutant == "legacy_graupel_constants":
        return {"_graupel_constants": lambda: LEGACY_CONSTANTS}
    target = "_thompson_source_sink_body" if mutant in BODY_MUTANTS else "_fall_speeds"
    fixed, old = {**SED_MUTANTS, **BODY_MUTANTS}[mutant]
    src = textwrap.dedent(inspect.getsource(getattr(tc, target)))
    assert src.count(fixed) == 1, mutant
    namespace = dict(vars(tc))
    exec(compile(src.replace(fixed, old), tc.__file__, "exec"), namespace)
    return {target: namespace[target]}


def _run(z, mutant=None, interpret=True):
    patch = _patched(mutant)
    original = {name: getattr(tc, name) for name in patch}
    with _native_real_env():
        try:
            for name, value in patch.items():
                setattr(tc, name, value)
            out, _ppt = full.full_column(_state(z), float(z["dt"]), interpret=interpret)
        finally:
            for name, value in original.items():
                setattr(tc, name, value)
    g = lambda a: np.asarray(a, np.float64)  # noqa: E731
    return {"qv": g(out.qv), "qc": g(out.qc), "qr": g(out.qr), "qg": g(out.qg),
            "th": g(out.T) / np.asarray(z["in_pii"], np.float64)}


def _errors(port, z):
    """Max |port - WRF| per column/field, relative to the column's WRF field maximum."""
    return {(str(name), f): float(np.abs(port[f][i] - np.asarray(z[f"wrf_{f}"][i], np.float64)).max()
                                  / max(np.abs(np.asarray(z[f"wrf_{f}"][i], np.float64)).max(), 1e-9))
            for i, name in enumerate(z["names"]) for f in FIELDS}


@pytest.fixture(scope="module")
def oracle():
    z = dict(np.load(FIXTURE))
    return z, _errors(_run(z), z)


def test_fixed_port_matches_pristine_wrf_graupel(oracle):
    _z, err = oracle
    for col in ("E1_graupel_subl", "H1_graupel_fall"):
        assert max(err[col, f] for f in ("qg", "qv", "th")) <= 1e-6, (col, err)
    assert max(err["G1_graupel_rime", f] for f in ("qg", "qc")) <= 2e-4, err
    assert err["F1_graupel_melt", "qr"] <= 5e-4 and err["F1_graupel_melt", "qg"] <= 3e-3, err
    for col in ("C1_rci_graupel", "C2_rci_graupel_cold"):
        assert err[col, "qg"] <= 2e-3, (col, err[col, "qg"])


def test_mp8_graupel_constants_are_load_bearing(oracle):
    z, _err = oracle
    err = _errors(_run(z, "legacy_graupel_constants"), z)
    assert err["E1_graupel_subl", "qg"] > 1e-3 and err["H1_graupel_fall", "qg"] > 1e-3, err  # sublimation, fall
    assert err["F1_graupel_melt", "qr"] > 1e-2, err  # melt (t2_qg_me)
    assert err["G1_graupel_rime", "qg"] > 1e-3 and err["G1_graupel_rime", "qc"] > 1e-3, err  # riming (t1_qg_qc, vtg)


def test_sedimentation_moment_is_load_bearing(oracle):
    z, _err = oracle
    err = _errors(_run(z, "legacy_sed_moment"), z)
    assert all(err[str(name), "qg"] > 0.1 for name in z["names"]), err


def test_rediagnosed_graupel_number_is_load_bearing(oracle):
    z, fixed = oracle
    err = _errors(_run(z, "carried_ng"), z)
    for col in ("C1_rci_graupel", "C2_rci_graupel_cold"):
        assert err[col, "qg"] > 0.2 and err[col, "qg"] >= 5 * fixed[col, "qg"], (col, err[col, "qg"], fixed[col, "qg"])


def test_entry_state_cold_collection_is_load_bearing(oracle):
    z, fixed = oracle
    err = _errors(_run(z, "cold_rates_post_warm"), z)
    for col in ("C1_rci_graupel", "C2_rci_graupel_cold"):
        assert err[col, "qg"] > 1e-2 and err[col, "qg"] >= 10 * fixed[col, "qg"], (col, err[col, "qg"], fixed[col, "qg"])
