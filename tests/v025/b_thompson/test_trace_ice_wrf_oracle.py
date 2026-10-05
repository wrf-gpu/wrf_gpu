"""Pristine-WRF trace-hydrometeor oracle for the Thompson full column (b-noahmp NF13, alisios §3.3 trace QICE/QSNOW).

Fixture: one pristine ``mp_gt_driver`` step (mp8, dt 54 s) on adversarial columns built by
proofs/thompson/ice_process_oracle (``oracle.py build trace``; module_mp_thompson.F, gfortran, REAL inputs as in WRF):
T1/T2 sub-R1 (5e-13) traces with ssati < 0 everywhere (``no_micro`` early return, :1646-2020) and with ssati > 0
(full path): WRF's entry branches zero them (qc1d(k) = 0.0 ...) and mp_gt_driver writes them back;
T3 ice-supersaturated but hydrometeor-free; T4/T5 nucleation from vapour alone (ssati >= 0.25; ssatw > eps, T < 253.15);
T6/T7 trace ice/snow above 0 C; T8/T9 trace ice sublimating/depositing; T10 ice just above R1; T11 trace cloud water;
T12/T13 cloud ice above 0 C (liquid-saturated / subsaturated).
WRF keeps cloud-ice deposition/sublimation (pri_ide/prs_ide) and ice->snow autoconversion (prs_iau/pni_iau) inside
the ``if (temp(k).lt.T_0)`` cold block (:2554-2779): above 0 C cloud ice only melts instantly into cloud water at the
end of the step (:3942-3953), so the melt survives the step. The port sublimated/autoconverted warm cloud ice (T6:
qc 0 and qv +9.3e-10 vs WRF qc 1e-9, qv unchanged) -> fixed (cold-block gate); each fix/behaviour has a mutant (E39).
Gates (same scale rule as the TH08 ice oracle; set after the pre-fix T6 measurement): zero/non-zero pattern of every species EXACT per cell; per column
max |port - WRF| / max(|WRF|, floor) <= 1e-3 (floors 1e-9 kg/kg, 1 /kg); theta <= 1e-4 K (WRF's t1d/pii round trip
moves theta by 1 ulp = 3.1e-5 K).
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

FIXTURE = Path(__file__).with_name("fixtures") / "trace_process_columns.npz"
C24 = dict(GPUWRF_THOMPSON_NATIVE_REAL="1", GPUWRF_THOMPSON_COLUMN_SED="1", GPUWRF_THOMPSON_SED_FP32="1",
           GPUWRF_THOMPSON_COLUMN_LAYOUT="1", GPUWRF_THOMPSON_FULL_COLUMN="1",
           GPUWRF_THOMPSON_FULL_COLUMN_EARLY_EXIT="1", GPUWRF_THOMPSON_SED_PREP_FUSED="0")
SPECIES = ("qc", "qr", "qi", "qs", "qg", "ni", "nr")
FLOOR = {"ni": 1.0, "nr": 1.0}
TOL, TOL_TH = 1e-3, 1e-4
MUTANTS = {  # name -> (function, fixed source, the same source with the behaviour removed)
    "warm_ice_deposition": ("_ice_sources_with_process_flags",
                            "pri_ide_raw = jnp.where(active_ice & cold_block, pri_ide_raw, 0.0)",
                            "pri_ide_raw = jnp.where(active_ice, pri_ide_raw, 0.0)"),
    "warm_ice_autoconversion": ("_ice_sources_with_process_flags",
                                "prs_iau_mass = jnp.where(active_ice & cold_block, prs_iau_mass, 0.0)",
                                "prs_iau_mass = jnp.where(active_ice, prs_iau_mass, 0.0)"),
    "no_subR1_zeroing": ("_finish", "qi = jnp.where(state.qi <= R1, 0.0, jnp.maximum(state.qi, 0.0))",
                         "qi = jnp.maximum(state.qi, 0.0)"),
}


@contextlib.contextmanager
def _c24_env():
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


def _run(z, mutant=None):
    patched = None
    if mutant:
        fn, fixed, removed = MUTANTS[mutant]
        src = textwrap.dedent(inspect.getsource(getattr(tc, fn)))
        assert src.count(fixed) == 1, mutant
        namespace = dict(vars(tc))
        exec(compile(src.replace(fixed, removed), tc.__file__, "exec"), namespace)
        patched = (fn, getattr(tc, fn), namespace[fn])
    with _c24_env():
        try:
            if patched:
                setattr(tc, patched[0], patched[2])
            out, _ppt = full.full_column(_state(z), float(z["dt"]), interpret=True)
        finally:
            if patched:
                setattr(tc, patched[0], patched[1])
    g = lambda a: np.asarray(a, np.float64)  # noqa: E731
    return {"qv": g(out.qv), "qc": g(out.qc), "qr": g(out.qr), "qi": g(out.qi), "qs": g(out.qs), "qg": g(out.qg),
            "ni": g(out.Ni), "nr": g(out.Nr), "th": g(out.T) / np.asarray(z["in_pii"], np.float64)}


def _failures(port, z):
    bad = []
    for i, name in enumerate(z["names"]):
        for v in ("qv", *SPECIES, "th"):
            wrf = np.asarray(z[f"wrf_{v}"][i], np.float64)
            d = np.abs(port[v][i] - wrf)
            if v == "th":
                if d.max() > TOL_TH:
                    bad.append((str(name), v, float(d.max())))
                continue
            if v in SPECIES and ((wrf > 0) != (port[v][i] > 0)).any():
                bad.append((str(name), v, "zero pattern"))
            rel = d.max() / max(np.abs(wrf).max(), FLOOR.get(v, 1e-9))
            if rel > TOL:
                bad.append((str(name), v, float(rel)))
    return bad


@pytest.fixture(scope="module")
def z():
    return dict(np.load(FIXTURE))


def test_fixture_exercises_the_trace_paths(z):
    names = list(z["names"])
    i = {n: names.index(n) for n in names}
    for n in ("T1_subR1_no_micro", "T2_subR1_icesat"):   # WRF zeroes every sub-R1 species
        assert z["in_qi"][i[n]].max() == np.float32(5e-13) and not any(z[f"wrf_{v}"][i[n]].any() for v in SPECIES)
    for n in ("T6_ice_trace_warm", "T12_ice_warm_satw", "T13_ice_warm_subsat"):   # warm cloud ice melts, vapour untouched
        assert not z["wrf_qi"][i[n]].any() and z["wrf_qc"][i[n]].max() > 0
        assert np.array_equal(z["wrf_qv"][i[n]], z["in_qv"][i[n]]) and not z["wrf_qs"][i[n]].any()
    for n in ("T4_nucleate_ssati", "T5_nucleate_watersat"):
        assert z["wrf_qi"][i[n]].max() > 0 and not z["in_qi"][i[n]].any()


def test_full_column_matches_pristine_wrf_on_trace_columns(z):
    assert _failures(_run(z), z) == []


@pytest.mark.parametrize("mutant", sorted(MUTANTS))
def test_each_trace_behaviour_is_deletion_sensitive(z, mutant):
    assert _failures(_run(z, mutant), z), mutant
