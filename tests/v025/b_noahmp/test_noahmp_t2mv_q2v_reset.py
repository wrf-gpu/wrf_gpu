"""Noah-MP vegetated-tile T2MV/Q2V on non-vegetated columns = WRF's ENERGY reset (0), not NaN.

WRF ENERGY zeroes T2MV/Q2V (module_sf_noahmplsm.F:2047-2048) and calls VEGE_FLUX only where
VEG .AND. FVEG>0 (:2237). The port evaluates the vegetated tile on every column and selects, so
the raw tile value leaked into NoahMPFluxes.t2mv/q2v: NaN on 8351/8400 PROD d01 cells (b-core
BC57r, b-thompson NF03). Oracle: the pristine NOAHMP_SFLX offline-driver savepoints
(proofs/noahmp/noahmp_offline_driver.F90): warm11 (5 bare/urban/snow FVEG=0 + 6 vegetated real
Canary d03 columns) and frozen4 (1 bare + 3 vegetated WN3 0227 d03 06Z columns).
Tolerances as the existing parity gates: T2MV 0.05 K (proofs/v090 TOL_K), Q2V 1e-5 kg/kg (B47).
Mutant: the raw tile value (the old producer) must fail on the bare columns (E39).
"""
import importlib.util
import inspect
import json
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
FROZEN = Path(__file__).with_name("fixtures") / "savepoints_energy_frozen.json"
TOL_T2MV, TOL_Q2V = 0.05, 1.0e-5
FIX = ('t2mv = jnp.where(use_veg, vf["t2mv"], 0.0)', 'q2v = jnp.where(use_veg, vf["q2v"], 0.0)')
OLD = ('t2mv = vf["t2mv"]', 'q2v = vf["q2v"]')


def _harness():
    path = ROOT / "proofs" / "v090" / "noahmp_t2mb_parity.py"
    spec = importlib.util.spec_from_file_location("noahmp_t2mb_parity_t2mv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _columns(corpus):
    path = ROOT / "proofs" / "noahmp" / "savepoints_energy.json" if corpus == "warm11" else FROZEN
    return json.loads(path.read_text())["columns"]


def _port(h, col, canopy):
    from gpuwrf.physics.noahmp.energy_radiation import radiation_twostream
    eg = h.eg
    ls, forcing, static, phen = eg.build_state(col), eg.build_forcing(col), eg.build_static(col), eg.build_phen(col)
    energy_p, rad_p = eg.build_params(col["vegtyp"], col["isltyp"])
    f = col["forcing"]
    rad, extras = radiation_twostream(ls, forcing, static, phen, rad_p, eg.DT)
    _ls, ef, _et = canopy(
        ls, forcing, static, rad, eg.DT, phen=phen, params=energy_p, rad_extras=extras,
        o2air=eg._f(0.209 * f["sfcprs"]), co2air=eg._f(395.0e-6 * f["sfcprs"]), foln=eg._f(1.0),
        isurban=int(eg._P.isurban))
    return ef, float(np.asarray(phen.fveg).reshape(-1)[0])


def _mutant_canopy():
    """noahmp_energy_canopy with the raw vegetated-tile T2MV/Q2V (the producer before this fix)."""
    from gpuwrf.physics.noahmp import energy
    src = textwrap.dedent(inspect.getsource(energy.noahmp_energy_canopy))
    for fixed, old in zip(FIX, OLD):
        assert src.count(fixed) == 1, fixed
        src = src.replace(fixed, old)
    namespace = dict(vars(energy))
    exec(compile(src, energy.__file__, "exec"), namespace)
    return namespace["noahmp_energy_canopy"]


def _errors(h, cols, canopy):
    rows = []
    for col in cols:
        ef, fveg = _port(h, col, canopy)
        t2d = col["wrf"]["t2diag"]
        g = lambda a: float(np.asarray(a).reshape(-1)[0])  # noqa: E731
        finite = all(np.isfinite(np.asarray(x)).all() for x in ef if x is not None and hasattr(x, "dtype")
                     and np.issubdtype(np.asarray(x).dtype, np.floating))
        rows.append(dict(name=col["name"], bare=fveg == 0.0, finite=finite,
                         t2mv=abs(g(ef.t2mv) - t2d["t2mv"]), q2v=abs(g(ef.q2v) - t2d["q2v"])))
    return rows


@pytest.mark.parametrize("native_real", ["0", "1"])
@pytest.mark.parametrize("corpus", ["warm11", "frozen4"])
def test_t2mv_q2v_match_pristine_wrf_on_bare_and_vegetated_columns(monkeypatch, native_real, corpus):
    from gpuwrf.physics.noahmp.energy import noahmp_energy_canopy
    monkeypatch.setenv("GPUWRF_NOAHMP_NATIVE_REAL", native_real)
    cols = _columns(corpus)
    rows = _errors(_harness(), cols, noahmp_energy_canopy)
    assert len(rows) == {"warm11": 11, "frozen4": 4}[corpus]
    assert any(r["bare"] for r in rows) and any(not r["bare"] for r in rows), rows
    bad = [r for r in rows if not (r["finite"] and r["t2mv"] <= TOL_T2MV and r["q2v"] <= TOL_Q2V)]
    assert not bad, bad


@pytest.mark.parametrize("native_real", ["0", "1"])
def test_raw_tile_value_fails_on_bare_columns(monkeypatch, native_real):
    monkeypatch.setenv("GPUWRF_NOAHMP_NATIVE_REAL", native_real)
    rows = _errors(_harness(), _columns("warm11"), _mutant_canopy())
    bare = [r for r in rows if r["bare"]]
    assert bare and all(not (r["finite"] and r["t2mv"] <= TOL_T2MV and r["q2v"] <= TOL_Q2V) for r in bare), bare
    assert all(r["t2mv"] <= TOL_T2MV and r["q2v"] <= TOL_Q2V for r in rows if not r["bare"]), rows
