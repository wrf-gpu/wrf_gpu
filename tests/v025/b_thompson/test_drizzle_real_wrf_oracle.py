"""Pristine-WRF warm-rain oracle on REAL drizzle columns for the Thompson full column (LL01, v0.3.3).

The frozen Thompson oracles (gate2/l3a/l3b, TH08 ice/graupel, NF13 trace, exit columns) hold no real drizzle column
(E154/E102: only synthetic 1-3-level qc bumps with qr = 0). Fixture (proofs/thompson/drizzle_oracle/build_fixture.py):
real WN3 0227 CPU-WRF history columns (d03 1 km dt 6 s, d02 3 km dt 18 s; tau 18-36; trade-wind stratocumulus and
orographic cloud, qc 0.1-1 g/kg, drizzle/onset/rain; land > 500 m, land <= 500 m, sea) and ONE pristine mp_gt_driver
step (module_mp_thompson.F, mp8, caller-faithful inputs: dry th, bottom-face w, dz8w with g = 9.81).
Gate on the one-step INCREMENTS (the processes, not the state): per column and field
max|d_port - d_WRF| <= TOL * max|d_WRF| + 2 ulp(max|WRF field|) (+ R1 = 1e-12 kg/kg for qc/qr: the worst
real level is trace rain 2.4e-11 evaporating, off by 4.4e-13), surface rain alike; every warm-rain process has a
deletion mutant (E39): Berry-Reinhardt autoconversion, accretion, the historic R2-clamped Nr source, the ventilation
term of rain evaporation, cloud-water sedimentation. Non-finite values fail the gate (NaN/+-Inf regression).
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

FIXTURE = Path(__file__).with_name("fixtures") / "drizzle_real_columns.npz"
C24 = dict(GPUWRF_THOMPSON_NATIVE_REAL="1", GPUWRF_THOMPSON_COLUMN_SED="1", GPUWRF_THOMPSON_SED_FP32="1",
           GPUWRF_THOMPSON_COLUMN_LAYOUT="1", GPUWRF_THOMPSON_FULL_COLUMN="1",
           GPUWRF_THOMPSON_FULL_COLUMN_EARLY_EXIT="1", GPUWRF_THOMPSON_SED_PREP_FUSED="0")
FIELDS = ("qv", "qc", "qr", "nr", "th")
TOL = {"qv": 1e-2, "qc": 1e-2, "qr": 1e-2, "nr": 1e-2, "th": 1e-2, "rain": 1e-3}
R1_FLOOR = {"qc": 1e-12, "qr": 1e-12}   # WRF's hydrometeor existence threshold R1 (trace-rain evaporation, E100)
MUTANTS = {  # name -> (function, fixed source, the same source with the process removed)
    "no_autoconversion": ("_warm_rain_collection",
                          "jnp.minimum(rc / float(dt), zeta / tau), 0.0)",
                          "jnp.minimum(rc / float(dt), 0.0 * zeta / tau), 0.0)"),
    "no_accretion": ("_warm_rain_collection",
                     "prr_rcw = jnp.where(active_rain & (mvd_r > D0R) & (mvd_c > D0C), prr_rcw_raw, 0.0)",
                     "prr_rcw = jnp.where(active_rain & (mvd_r > D0R) & (mvd_c > D0C), 0.0 * prr_rcw_raw, 0.0)"),
    "nr_source_r2_clamp": ("_warm_rain_collection",
                           "pnr_wau = prr_wau / (AM_R * NU_C_MP8 * 10.0 * D0R**3)",
                           "pnr_wau = prr_wau / jnp.maximum(AM_R * NU_C_MP8 * 10.0 * D0R**3, 1.0e-6)"),
    "no_evaporation_ventilation": ("_rain_evaporation",
                                   "+ T2_QR_EV * vsc2 * rhof2 * _dpow(lamr + 0.5 * FV_R, -CRE11))",
                                   ")"),
    # The full column replaces _sed_cloud_water by its register kernel; both read _cloud_water_fall_speed.
    "no_cloud_sedimentation": ("_cloud_water_fall_speed", "return jnp.where(active, vtc, 0.0)",
                               "return jnp.where(active, 0.0 * vtc, 0.0)"),
    "cloud_sed_in_updraft": ("_cloud_water_fall_speed", "active = (rc > R1) & (state.w < 1.0e-1)",
                             "active = rc > R1"),
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


def _run(z, idx, mutant=None):
    patched = None
    if mutant:
        fn, fixed, removed = MUTANTS[mutant]
        src = textwrap.dedent(inspect.getsource(getattr(tc, fn)))
        assert src.count(fixed) == 1, mutant
        namespace = dict(vars(tc))
        exec(compile(src.replace(fixed, removed), tc.__file__, "exec"), namespace)
        patched = (fn, getattr(tc, fn), namespace[fn])
    f = lambda name: jnp.asarray(z[f"in_{name}"][idx], jnp.float32)  # noqa: E731
    T, p, qv = f("th") * f("pii"), f("p"), f("qv")
    zero = jnp.zeros_like(qv)
    state = tc.ThompsonColumnState(qv=qv, qc=f("qc"), qr=f("qr"), qi=f("qi"), qs=f("qs"), qg=f("qg"), Ni=f("ni"),
                                   Nr=f("nr"), Ns=zero, Ng=zero, T=T, p=p,
                                   rho=tc.density_from_pressure_temperature(p, T, qv), dz=f("dz8w"), w=f("w"))
    with _c24_env():
        try:
            if patched:
                setattr(tc, patched[0], patched[2])
            out, ppt = full.full_column(state, float(z["dt"][idx[0]]), interpret=True)
        finally:
            if patched:
                setattr(tc, patched[0], patched[1])
    g = lambda a: np.asarray(a, np.float64)  # noqa: E731
    return {"qv": g(out.qv), "qc": g(out.qc), "qr": g(out.qr), "nr": g(out.Nr),
            "th": g(np.asarray(out.T, np.float32) / np.asarray(z["in_pii"][idx], np.float32)), "rain": g(ppt["rain"])}


def _excess(port, z, idx):
    """Per (column, field): max|d_port - d_WRF| beyond the 2-ulp floor, in units of max|d_WRF| (gate: <= TOL).

    A non-finite input, WRF reference or port value is a hard failure (inf) BEFORE any max/floor: Python's
    max(0.0, nan) is 0.0, so an all-NaN candidate would otherwise pass (review-writer 21:53Z).
    """
    res = {}
    for j, i in enumerate(idx):
        name = str(z["names"][i])
        for v in FIELDS:
            x, w = (np.asarray(z[f"{s}_{v}"][i], np.float64) for s in ("in", "wrf"))
            p = np.asarray(port[v][j], np.float64)
            if not (np.isfinite(x).all() and np.isfinite(w).all() and np.isfinite(p).all()):
                res[(name, v)] = np.inf
                continue
            dw, dp = w - x, p - x
            floor = 2.0 * float(np.spacing(np.float32(np.abs(w).max()))) + R1_FLOOR.get(v, 0.0)
            res[(name, v)] = max(0.0, np.abs(dp - dw).max() - floor) / max(np.abs(dw).max(), 1e-30)
        w, p = float(z["wrf_rainncv"][i]), float(port["rain"][j])
        if not (np.isfinite(w) and np.isfinite(p)):
            res[(name, "rain")] = np.inf
            continue
        res[(name, "rain")] = max(0.0, abs(p - w) - 2.0 * float(np.spacing(np.float32(w)))) / max(w, 1e-30)
    return res


def _failures(z, mutant=None, dts=None):
    bad = []
    for dt in sorted(set(z["dt"].tolist())) if dts is None else dts:
        idx = np.flatnonzero(z["dt"] == dt)
        for (name, v), e in _excess(_run(z, idx, mutant), z, idx).items():
            if e > TOL[v]:
                bad.append((name, v, e))
    return bad


@pytest.fixture(scope="module")
def z():
    return dict(np.load(FIXTURE))


def test_fixture_is_real_drizzle(z):
    """E154 census: warm cloud 0.1-1 g/kg with drizzle-scale rain, both domains, and every process active in WRF."""
    T = z["in_th"].astype(np.float64) * z["in_pii"]
    warm_cloud = (z["in_qc"] >= 1e-4) & (z["in_qc"] <= 1e-3) & (T > 273.15)
    drizzle = warm_cloud & (z["in_qr"] > 1e-8) & (z["in_qr"] <= 1e-5)
    assert warm_cloud.any(-1).sum() >= 0.8 * len(z["names"]) and drizzle.sum() >= 20
    assert set(z["dt"].tolist()) == {6.0, 18.0}
    assert (z["wrf_rainncv"] > 1e-6).sum() >= 10          # surface drizzle/rain in WRF
    assert ((z["wrf_qv"] - z["in_qv"]) > 1e-9).any(-1).sum() >= 10   # rain/cloud evaporation in WRF


def test_full_column_matches_pristine_wrf_on_real_drizzle(z):
    assert _failures(z) == []


@pytest.mark.parametrize("mutant", sorted(MUTANTS))
def test_each_warm_rain_process_is_deletion_sensitive(z, mutant):
    assert _failures(z, mutant, dts=[18.0]), mutant


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
@pytest.mark.parametrize("field", (*FIELDS, "rain"))
def test_nonfinite_candidate_fails_its_field(z, field, bad):
    """Gate function alone (no kernel run): WRF's own values pass with zero excess; one non-finite value in
    any gated candidate field fails exactly that (column, field)."""
    idx = np.flatnonzero(z["dt"] == 18.0)
    port = {v: np.array(z[f"wrf_{v}"][idx], np.float64) for v in FIELDS}
    port["rain"] = np.array(z["wrf_rainncv"][idx], np.float64)
    assert max(_excess(port, z, idx).values()) == 0.0
    j = len(idx) // 2
    if field == "rain":
        port["rain"][j] = bad
    else:
        port[field][j, 7] = bad
    ex = _excess(port, z, idx)
    name = str(z["names"][idx[j]])
    assert ex[(name, field)] == np.inf and ex[(name, field)] > TOL[field]
    assert all(e == 0.0 for key, e in ex.items() if key != (name, field))
