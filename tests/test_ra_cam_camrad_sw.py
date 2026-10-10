"""CAM shortwave camrad call (ra_sw_physics = 3) end to end vs WRF's REAL outputs (oracle CAM01, arm S).

The radctl-level SW parity (r8 fluxes, aerosol, RH) is tests/test_ra_cam_sw_oracle.py; this module checks the
``camrad`` wrapper (preprocessing, cloud optics, ozone, unit scaling, REAL conversion, SWDOWN) on the committed
compact fixture (40 columns, day + night), same 4-ulp REAL gate as the LW test, plus WRF's mxaerl.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import gpuwrf  # noqa: F401
import jax
import jax.numpy as jnp

from gpuwrf.physics import ra_cam as R
from gpuwrf.physics import ra_cam_common as C

FIXTURE = Path(os.environ.get("CAM01_FIXTURE", Path(__file__).resolve().parents[1] / "data" / "fixtures"
                              / "cam01-compact-v1.npz"))
_SW_OUT = ("rthratensw", "rthratenswc", "gsw", "swcf", "coszr", "swddir", "swddif", "swupt", "swuptc", "swdnt",
           "swdntc", "swupb", "swupbc", "swdnb", "swdnbc")


def _real_ok(a, b, ulps=4):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    tol = np.maximum(ulps * np.spacing(np.abs(b)), 1e-9 * max(np.abs(b).max(), 1e-30))
    return bool(np.all(np.abs(a.astype(np.float64) - b) <= tol)), int((a != b).sum())


@pytest.fixture(scope="module")
def fx():
    if not FIXTURE.exists():
        pytest.skip(f"CAM01 fixture missing: {FIXTURE}")
    return dict(np.load(FIXTURE))


@pytest.fixture(scope="module")
def sw_out(fx):
    try:
        from gpuwrf.physics.ra_cam_sw import load_cam_aeropt_tables

        tables = load_cam_aeropt_tables()
    except (FileNotFoundError, OSError) as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"WRF run/CAM_AEROPT_DATA unavailable: {exc}")
    names = ("in_t", "in_p", "in_p8w", "in_pi", "in_qv", "in_qc", "in_qi", "in_qs", "in_cldfra", "in_xland",
             "in_xice", "in_snow", "in_emiss", "in_tsk", "in_xlat", "coszen", "in_albedo", "in_julian", "solcon",
             "r8s_co2vmr")

    def run(d):  # operands are traced jit arguments, as in the operational coupler (no constant folding)
        ozm, pin = C.cam_ozmixm(d["in_xlat"])
        return R.camrad_sw(t_phy=d["in_t"], p_phy=d["in_p"], p8w=d["in_p8w"], pi_phy=d["in_pi"], qv=d["in_qv"],
                           qc=d["in_qc"], qi=d["in_qi"], qs=d["in_qs"], cldfra=d["in_cldfra"], xland=d["in_xland"],
                           xice=d["in_xice"], snow=d["in_snow"], emiss=d["in_emiss"], tsk=d["in_tsk"],
                           xlat=d["in_xlat"], coszen=d["coszen"], albedo=d["in_albedo"], julian=d["in_julian"],
                           solcon=d["solcon"], co2vmr=d["r8s_co2vmr"], ozmixm=ozm, pin=pin,
                           mxaerl=int(fx["mxaerl"]), aeropt_tables=tables)

    return jax.jit(run)({k: jnp.asarray(fx[k]) for k in names})


def test_camrad_sw_matches_wrf_real_outputs(fx, sw_out):
    assert (fx["coszen"] > 0).sum() >= 10 and (fx["coszen"] <= 0).sum() >= 10
    bad = {n: _real_ok(getattr(sw_out, n), fx["s_" + n]) for n in _SW_OUT}
    bad = {n: v for n, v in bad.items() if not v[0]}
    assert not bad, bad
    day = fx["coszen"] > 0
    assert _real_ok(np.asarray(sw_out.swddni)[day], fx["s_swddni"][day])[0]
    assert np.asarray(sw_out.gsw)[~day].max() == 0.0


def test_swdown_is_wrf_radiation_driver_expression(fx, sw_out):
    """module_radiation_driver.F:3002: SWDOWN = GSW/(1 - ALBEDO) in REAL."""

    one_minus_alb = np.float32(1.0) - fx["in_albedo"].astype(np.float32)
    np.testing.assert_array_equal(np.asarray(sw_out.swdown), np.asarray(sw_out.gsw) / one_minus_alb)
    assert _real_ok(sw_out.swdown, fx["s_gsw"].astype(np.float32) / one_minus_alb)[0]
