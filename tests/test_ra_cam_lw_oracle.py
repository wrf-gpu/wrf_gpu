"""CAM longwave (ra_lw_physics = 3) JAX port vs the pristine-WRF true-caller oracle CAM01.

Oracle: ``proofs/cam_rad`` -- unchanged pristine V4.7.1 ``camradinit`` + ``camrad`` (libwrflib.a) with the
radiation_driver wiring on real WN3 0227 columns (+ augmented multi-region cloud columns); committed compact fixture
``data/fixtures/cam01-compact-v1.npz`` (40 columns), full 210-column fixture via ``CAM01_FIXTURE``.

Gates: r8 intermediates relative error <= 1e-12 (floor 1e-12 x field max; heating = flux divergence: 1e-11 relative +
1e-12 x field max absolute); WRF REAL outputs within 4 float32 ulp
(floor 1e-9 x field max); held REAL absorptivity arrays bit-identical.  Two mutants must fail: the module-CO2 quirk
(radtpl/radems/radabs read WRF's never-updated MODULE co2vmr) and the held-absorptivity path (doabsems=.false.).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import gpuwrf  # noqa: F401  (jax x64)
import jax
import jax.numpy as jnp

from gpuwrf.physics import ra_cam as R
from gpuwrf.physics import ra_cam_common as C
from gpuwrf.physics import ra_cam_lw as LW

FIXTURE = Path(os.environ.get("CAM01_FIXTURE", Path(__file__).resolve().parents[1] / "data" / "fixtures" / "cam01-compact-v1.npz"))


@pytest.fixture(scope="module")
def fx():
    if not FIXTURE.exists():
        pytest.skip(f"CAM01 fixture missing: {FIXTURE}")
    return dict(np.load(FIXTURE))


@pytest.fixture(scope="module")
def tables():
    try:
        return C.load_cam_abs_tables()
    except (FileNotFoundError, OSError) as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"WRF run/CAM_ABS_DATA unavailable: {exc}")


def _rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    floor = 1e-12 * max(np.abs(b).max(), 1e-300)
    return float((np.abs(a - b) / np.maximum(np.abs(b), floor)).max())


def _close(a, b, rtol=1e-11, afrac=1e-12):
    """|a - b| <= rtol*|b| + afrac*max|b| (flux divergences: absolute floor at the flux precision)."""

    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return bool(np.all(np.abs(a - b) <= rtol * np.abs(b) + afrac * np.abs(b).max()))


def _real_ok(a, b, ulps=4):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    tol = np.maximum(ulps * np.spacing(np.abs(b)), 1e-9 * np.abs(b).max())
    return bool(np.all(np.abs(a.astype(np.float64) - b) <= tol)), int((a != b).sum())


def test_oracle_census_covers_active_regimes(fx):
    """E154: the oracle exercises day+night, liquid+ice clouds, multi-region overlap, clear columns."""

    cld = fx["in_cldfra"].max(1)
    assert (fx["coszen"] > 0).sum() >= 10 and (fx["coszen"] <= 0).sum() >= 10
    assert (cld > 0).sum() >= 20 and (cld <= 0).sum() >= 3
    assert (fx["r8l_qliq"].max(1) > 1e-6).sum() >= 10 and (fx["r8l_qice"].max(1) > 1e-7).sum() >= 5
    assert fx["r8l_nmxrgn"].max() >= 5 and (fx["r8l_nmxrgn"] >= 2).sum() >= 8
    assert np.unique(fx["in_julian"].astype(np.int64)).size >= 2


def test_setup_matches_camrad_preprocessing(fx):
    cols = C.camrad_prepare(fx["in_t"], fx["in_p"], fx["in_p8w"], fx["in_qv"], fx["in_qc"], fx["in_qi"], fx["in_qs"],
                            fx["in_cldfra"], fx["in_xland"], fx["in_xice"], fx["in_snow"], fx["in_emiss"], fx["in_tsk"],
                            fx["in_xlat"], fx["coszen"], fx["in_albedo"])
    for name, key in (("q1", "q1"), ("qliq", "qliq"), ("qice", "qice"), ("cld", "cld"), ("pmid", "pmid"),
                      ("pint", "pint"), ("t", "t"), ("lwups", "lwups"), ("coszrs", "coszrs")):
        np.testing.assert_array_equal(np.asarray(getattr(cols, name)), fx["r8l_" + key], err_msg=name)
    opt = C.param_cldoptics_calc(cols)
    for name in ("cicewp", "cliqwp", "rel", "rei", "pmxrgn"):
        np.testing.assert_array_equal(np.asarray(getattr(opt, name)), fx["r8l_" + name], err_msg=name)
    np.testing.assert_array_equal(np.asarray(opt.nmxrgn), fx["r8l_nmxrgn"])
    assert _rel(opt.emis, fx["r8l_emis"]) <= 1e-12
    ozm, pin = C.cam_ozmixm(fx["in_xlat"])
    np.testing.assert_array_equal(np.asarray(pin), fx["pin"])
    o3 = C.radozn(cols.pmid, pin, C.oznint(ozm, fx["in_julian"]))
    assert _rel(o3, fx["r8l_o3vmr"]) <= 1e-12
    gas = [jnp.asarray(fx["r8l_" + k])[:, None] for k in ("n2ovmr", "ch4vmr", "f11vmr", "f12vmr")]
    for name, arr in zip(("n2o", "ch4", "cfc11", "cfc12"), C.trcmix_clwrf(cols.pmid, cols.clat, *gas)):
        assert _rel(arr, fx["r8l_" + name]) <= 1e-12, name


def _pieces(fx, tables):
    J = lambda k: jnp.asarray(fx[k])

    def run():
        tp = LW.radtpl(J("r8l_t"), J("r8l_lwups"), J("r8l_q1"), J("lwi_pnm"), jnp.log(J("r8l_pmid")),
                       jnp.log(J("r8l_pint")))
        plol, plos = LW.radoz2(J("r8l_o3vmr"), J("lwi_pnm"))
        tr = LW.trcpth(J("r8l_t"), J("lwi_pnm"), J("r8l_cfc11"), J("r8l_cfc12"), J("r8l_n2o"), J("r8l_ch4"),
                       J("r8l_q1"), J("r8l_co2mmr"))
        return tp, plol, plos, tr, LW.radems(tp, J("lwi_pnm"), plol, plos, tr, tables)

    return jax.jit(run)()


def test_radtpl_radoz2_trcpth_radems_parity(fx, tables):
    tp, plol, plos, tr, em = _pieces(fx, tables)
    worst = {}
    for group, names in ((tp, LW.RadtplOut._fields), (tr, LW.TrcpthOut._fields), (em, LW.RademsOut._fields)):
        for n in names:
            worst[n] = _rel(getattr(group, n), fx["lwi_" + n])
    worst["plol"] = _rel(plol, fx["lwi_plol"])
    worst["plos"] = _rel(plos, fx["lwi_plos"])
    bad = {k: v for k, v in worst.items() if v > 1e-12}
    assert not bad, bad


def test_mutant_module_co2_quirk_is_detected(fx, tables, monkeypatch):
    """Deleting WRF's module-co2vmr quirk (using the CLWRF CO2 in radtpl/radems) must break parity."""

    monkeypatch.setattr(LW, "CONST", C.CONST._replace(co2vmr_module=float(fx["r8l_co2vmr"][0])))
    tp, _plol, _plos, _tr, em = _pieces(fx, tables)
    assert _rel(tp.plco2, fx["lwi_plco2"]) > 1e-3
    assert _rel(em.emstot, fx["lwi_emstot"]) > 1e-6


def test_radclwmx_fluxes_parity(fx):
    J = lambda k: jnp.asarray(fx[k])

    def run():
        tp = LW.radtpl(J("r8l_t"), J("r8l_lwups"), J("r8l_q1"), J("lwi_pnm"), jnp.log(J("r8l_pmid")),
                       jnp.log(J("r8l_pint")))
        return LW.radclwmx_fluxes(J("lwi_abstot"), J("lwi_absnxt"), J("lwi_emstot"), tp, J("r8l_lwups"), J("lwi_pbr"),
                                  J("lwi_pnm"), J("r8l_cld"), J("r8l_emis"), J("r8l_pmxrgn"),
                                  J("r8l_nmxrgn").astype(jnp.int32))

    out = jax.jit(run)()
    k3 = C.lit(1e-3)
    assert _close(out.qrl, fx["r8l_qrl"])
    assert _close(out.qrlcs, fx["r8l_qrlcs"])
    for n in ("flup", "flupc", "fldn", "fldnc", "flwds", "flns", "flnt"):
        assert _rel(getattr(out, n) * k3, fx["r8l_" + n]) <= 1e-12, n
    assert _rel(out.flut * k3, fx["r8l_olrtoa"]) <= 1e-12


def _camrad_lw(fx, tables, arm, doabsems, held=None):
    gases = R.CamGases(*(jnp.asarray(fx["r8l_" + k]) for k in ("co2vmr", "n2ovmr", "ch4vmr", "f11vmr", "f12vmr")))
    ozm, pin = C.cam_ozmixm(fx["in_xlat"])
    t, qv, tsk = (fx["in_t"], fx["in_qv"], fx["in_tsk"]) if arm == "l" else (fx["h_t"], fx["h_qv"], fx["h_tsk"])

    def run(held_):
        return R.camrad_lw(t_phy=t, p_phy=fx["in_p"], p8w=fx["in_p8w"], pi_phy=fx["in_pi"], qv=qv, qc=fx["in_qc"],
                           qi=fx["in_qi"], qs=fx["in_qs"], cldfra=fx["in_cldfra"], xland=fx["in_xland"],
                           xice=fx["in_xice"], snow=fx["in_snow"], emiss=fx["in_emiss"], tsk=tsk, xlat=fx["in_xlat"],
                           coszen=fx["coszen"], albedo=fx["in_albedo"], julian=fx["in_julian"], gases=gases,
                           ozmixm=ozm, pin=pin, tables=tables, doabsems=doabsems, held=held_)

    return jax.jit(run)(held)


_LW_OUT = ("rthratenlw", "rthratenlwc", "glw", "olr", "lwcf", "lwupt", "lwuptc", "lwdnt", "lwdntc", "lwupb", "lwupbc",
           "lwdnb", "lwdnbc")


@pytest.fixture(scope="module")
def lw_fresh(fx, tables):
    return _camrad_lw(fx, tables, "l", True)


def test_camrad_lw_matches_wrf_real_outputs(fx, lw_fresh):
    bad = {n: _real_ok(getattr(lw_fresh, n), fx["l_" + n]) for n in _LW_OUT}
    bad = {n: v for n, v in bad.items() if not v[0]}
    assert not bad, bad
    np.testing.assert_array_equal(np.asarray(lw_fresh.cemiss), fx["l_cemiss"])
    np.testing.assert_array_equal(np.asarray(lw_fresh.held.abstot), fx["l_abstot3"])
    np.testing.assert_array_equal(np.asarray(lw_fresh.held.absnxt), fx["l_absnxt3"])
    np.testing.assert_array_equal(np.asarray(lw_fresh.held.emstot), fx["l_emstot3"])


def test_camrad_lw_held_absorptivities_match_wrf(fx, tables, lw_fresh):
    """doabsems = .false.: WRF reuses the held REAL abstot/absnxt/emstot of the last abs/ems call."""

    out = _camrad_lw(fx, tables, "h", False, lw_fresh.held)
    bad = {n: _real_ok(getattr(out, n), fx["h_" + n]) for n in _LW_OUT}
    bad = {n: v for n, v in bad.items() if not v[0]}
    assert not bad, bad


def test_mutant_held_path_is_detected(fx, tables):
    """Recomputing absorptivities instead of using the held ones (deleting the doabsems semantics) must fail."""

    out = _camrad_lw(fx, tables, "h", True)
    assert not _real_ok(out.rthratenlw, fx["h_rthratenlw"])[0]
    assert not _real_ok(out.glw, fx["h_glw"])[0]
