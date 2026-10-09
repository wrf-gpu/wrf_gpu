"""fid-q2 RE01: the true-caller pristine RRTMG oracle for GPUWRF_RRTMG_MP_RE (v0.3.3) — oracle self-checks.

The frozen tier-1 RRTMG fixtures came from a synthetic harness with has_reqc = 0-style constant radii (10/30/75 um)
and cldovrlp = 1; PROD/WN3 run pristine WRF with has_reqc/i/s = 1 (Thompson radii) and cldovrlp = 2 (Registry
default; CPU-WRF history attribute CLDOVRLP = 2).  These tests pin the oracle itself: it is the registered pristine
output, its clear columns cannot see either wiring, and both caller misses are detectable at the frozen bounds.
The port-side producer/consumer gates (b-column) use the same reader: ``rrtmg_mp_re_fixture``.
"""
from pathlib import Path

import numpy as np
import pytest

from rrtmg_mp_re_fixture import (ARMS, FIXTURE, FLUX_BOUND, arm_outputs, bound_ratio, columns_outside, load)

CPU = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run")
CATEGORIES = {"clear": 27, "ice": 18, "liq_land": 30, "liq_thick": 30, "liq_thin_ocean": 27, "mixed": 18,
              "sgs_only": 18, "snow": 18}
RE_QC_BG, RE_QI_BG, RE_QS_BG = np.float32(2.49e-6), np.float32(4.99e-6), np.float32(9.99e-6)


@pytest.fixture(scope="module")
def fx():
    return load(FIXTURE)


def test_fixture_is_the_registered_pristine_output(fx):
    ncol, nz = fx["in_t"].shape
    assert (ncol, nz) == (186, 44)
    assert {c: int((fx["category"] == c).sum()) for c in CATEGORIES} == CATEGORIES
    for arm in ARMS:
        assert fx[f"{arm}_lwdn"].shape == (ncol, nz + 2) and fx[f"{arm}_hsw"].shape == (ncol, nz)
    floats = [v for v in fx.values() if v.dtype.kind == "f"]
    assert all(np.isfinite(v).all() for v in floats)
    assert float(fx["in_p_top"]) == 5000.0 and np.all(fx["in_p_hyd_w"][:, -1] == 5000.0)


def test_clear_columns_cannot_see_the_radius_or_overlap_wiring(fx):
    clear = fx["category"] == "clear"
    assert not (fx["in_cldfra"][clear] > 0).any()
    for arm in ARMS[1:]:
        for key, value in arm_outputs(fx, arm).items():
            np.testing.assert_array_equal(value[clear], fx[f"A_{key}"][clear], err_msg=f"{arm} {key}")


def test_both_caller_misses_are_detectable_at_frozen_bounds(fx):
    cats = fx["category"]
    cloudy = cats != "clear"
    outside = {arm: columns_outside(fx, arm_outputs(fx, arm)) for arm in ARMS}
    assert not outside["A"].any()
    assert not (outside["B"] | outside["C"] | outside["D"] | outside["E"])[~cloudy].any()
    # radii alone (C), has_req=0 (B), overlap alone (E), the port today (D) -- measured 100 / 101 / 48 / 117 of 159
    # (scalar fluxes + surface..model-top interfaces + heating)
    assert outside["C"].sum() >= 80 and outside["B"].sum() >= 80 and outside["E"].sum() >= 35 and outside["D"].sum() >= 100
    for cat in CATEGORIES:
        if cat != "clear":
            assert outside["C"][cats == cat].any(), cat  # every cloudy regime exercises the radius wiring
    # the marine thin-cloud regime (LL01 F3): constant 10 um transmits more sunlight than Thompson's radii
    thin_day = (cats == "liq_thin_ocean") & (fx["col_coszen"] > 0.1)
    assert (fx["C_swdnb"] - fx["A_swdnb"])[thin_day].mean() > 5.0


def test_pristine_radii_follow_the_mp_gt_driver_contract(fx):
    rc, ri, rs = fx["col_re_cloud"], fx["col_re_ice"], fx["col_re_snow"]
    assert rc.min() >= RE_QC_BG and rc.max() <= np.float32(50e-6)
    assert ri.min() >= RE_QI_BG and ri.max() <= np.float32(125e-6)
    assert rs.min() >= RE_QS_BG and rs.max() <= np.float32(999e-6)
    # calc_effectRad: rho = 0.622 p / (R t (qv + 0.622)), R = 287.04; liquid radius only where rc = qc*rho > R1 = 1e-12
    rho = 0.622 * fx["in_p"] / (287.04 * fx["in_t"] * (fx["in_qv"] + 0.622))
    wet, dry = fx["in_qc"] * rho > 1e-11, fx["in_qc"] * rho < 1e-13
    assert wet.sum() > 300 and (rc[wet] >= np.float32(2.51e-6)).all()
    assert (rc[dry] == RE_QC_BG).all()
    # radii are a real spread, not the port's constants
    assert np.ptp(rc[wet]) > 5e-6 and np.ptp(ri[fx["in_qi"] * rho > 1e-10]) > 50e-6


def test_wrf_truth_arm_reproduces_cpu_wrf_radiation_history_best(fx):
    """Independent evidence (statistical): CPU-WRF's own history (the tau-30 min call, COSZEN-verified) is closest to A."""

    netCDF4 = pytest.importorskip("netCDF4")
    if not CPU.is_dir():
        pytest.skip("original WN3 0227 CPU-WRF history not mounted")
    names = ("SWDNB", "SWDNT", "SWUPT", "COSZEN")
    hist = {n: [] for n in names}
    for dom, (j, i), stamp in zip(fx["domain"], fx["ji"], _stamps(fx)):
        with netCDF4.Dataset(CPU / f"wrfout_{dom}_{stamp}") as ds:
            for n in names:
                hist[n].append(float(ds[n][0, j, i]))
    hist = {n: np.asarray(v) for n, v in hist.items()}
    day = (fx["category"] != "clear") & (hist["COSZEN"] > 0.15) & (fx["col_coszen"] > 0.15)
    assert day.sum() >= 60
    h_trans = hist["SWDNB"] / np.maximum(hist["SWDNT"], 1e-3)
    h_alb = hist["SWUPT"] / np.maximum(hist["SWDNT"], 1e-3)
    toa = {arm: np.maximum(fx[f"{arm}_swdnt"], 1e-3) for arm in ARMS}
    rms = {arm: (np.sqrt(np.mean((fx[f"{arm}_swdnb"] / toa[arm] - h_trans)[day] ** 2)),
                 np.sqrt(np.mean((fx[f"{arm}_swupt"] / toa[arm] - h_alb)[day] ** 2))) for arm in ARMS}
    for k in (0, 1):
        assert rms["A"][k] <= 0.9 * min(rms[arm][k] for arm in ARMS[1:]), rms


def _stamps(fx):
    hours = fx["in_tau"].astype(int)
    base = np.datetime64("2026-02-28T00:00")
    return [str(base + np.timedelta64(int(h), "h")).replace("T", "_") + ":00" for h in hours]


def test_bound_helper_is_the_tier1_rule():
    ref = np.array([100.0, -20.0])
    assert np.allclose(bound_ratio(ref, ref + np.array([6.0, 2.0]), FLUX_BOUND), [1.0, 1.0])


def test_non_finite_candidates_are_outside_not_silently_inside(fx):
    """review-writer RE01 blocker: NaN/Inf must never pass (NaN ratios do not compare > 1)."""

    truth = arm_outputs(fx, "A")
    assert not columns_outside(fx, truth).any()
    all_nan = {key: np.full_like(value, np.nan) for key, value in truth.items()}
    assert columns_outside(fx, all_nan).all()
    for bad in (np.nan, np.inf, -np.inf):
        for key in ("glw", "lwdn", "hsw"):
            one = {k: v.copy() for k, v in truth.items()}
            one[key][7] = bad  # one column, every level of a profile key
            flagged = columns_outside(fx, one)
            assert flagged[7] and flagged.sum() == 1, (bad, key)
    with pytest.raises(ValueError):
        columns_outside(fx, {"glw": truth["glw"][:-1]})
