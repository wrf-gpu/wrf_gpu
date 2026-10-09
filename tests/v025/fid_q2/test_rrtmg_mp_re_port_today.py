"""fid-q2 RE01: the port's RRTMG today == pristine arm D (constant 10/30/75 um radii + random McICA overlap).

Pins the oracle's model of the current port on the real 0227 operands and the size of the miss vs WRF truth (arm A:
Thompson radii, has_reqc/i/s = 1, cldovrlp = 2).  When GPUWRF_RRTMG_MP_RE (+ maximum-random McICA) land, the owner
adds the flag-ON arm: no column outside the frozen bounds vs A (and MP_RE alone == arm E); this OFF arm must keep
holding with the flags explicitly OFF.  CPU legacy column solvers (tests/conftest: CPU pytest = legacy unless set).
"""
from datetime import datetime

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics.rrtmg_lw import RRTMGLWColumnState, solve_rrtmg_lw_column
from gpuwrf.physics.rrtmg_sw import RRTMGSWColumnState, solve_rrtmg_sw_column
from gpuwrf.physics.wrf_clwrf_ghg import clwrf_ssp245_gases_for_time

from rrtmg_mp_re_fixture import FIXTURE, columns_outside, load

RRSW_SCON = 1368.22  # module_ra_rrtmg_sw.F rrsw_scon: solvar = solcon / rrsw_scon


def _port_today(fx):
    ncol, nz = fx["in_t"].shape
    out = {"glw": np.zeros(ncol), "olr": np.zeros(ncol), "swdnb": np.zeros(ncol), "swupt": np.zeros(ncol),
           "lwup": np.zeros((ncol, nz + 2)), "lwdn": np.zeros((ncol, nz + 2)), "swup": np.zeros((ncol, nz + 2)),
           "swdn": np.zeros((ncol, nz + 2)), "hlw": np.zeros((ncol, nz)), "hsw": np.zeros((ncol, nz))}
    hours = fx["in_tau"].astype(int)
    width = max(int((hours == h).sum()) for h in np.unique(hours))
    for h in np.unique(hours):
        idx = np.flatnonzero(hours == h)
        pad = np.resize(idx, width)  # one shape for every valid time: one compile per solver
        a = lambda key, cols=pad: jnp.asarray(np.asarray(fx[key][cols], np.float64))
        ghg = clwrf_ssp245_gases_for_time(datetime(2026, 2, 28) + np.timedelta64(int(h), "h").astype(object))
        common = dict(T=a("in_t"), p=a("in_p_hyd"), qv=a("in_qv"), qc=a("in_qc_rad"), qi=a("in_qi_rad"), qs=a("in_qs"),
                      qg=a("in_qg"), cloud_fraction=a("in_cldfra"), dz=a("in_dz8w")[:, :nz], rho=a("in_rho"),
                      pressure_interfaces=a("in_p_hyd_w"), temperature_interfaces=a("in_t8w"), ozone_vmr=a("col_o3"),
                      co2_vmr=ghg.co2_vmr, n2o_vmr=ghg.n2o_vmr, ch4_vmr=ghg.ch4_vmr)
        lw = solve_rrtmg_lw_column(RRTMGLWColumnState(
            **common, surface_temperature=a("in_tsk"), surface_emissivity=a("in_emiss"),
            top_pressure_pa=float(fx["in_p_top"]), cfc11_vmr=ghg.cfc11_vmr, cfc12_vmr=ghg.cfc12_vmr))
        sw = solve_rrtmg_sw_column(RRTMGSWColumnState(
            **common, surface_albedo=a("in_albedo"), coszen=a("col_coszen"),
            solar_source_scale=jnp.asarray(np.asarray(fx["col_solcon"][pad], np.float64) / RRSW_SCON)))
        n = idx.size
        pi = fx["in_pi"][idx].astype(np.float64)
        for key, value in (("glw", lw.surface_down), ("olr", lw.toa_up), ("lwup", lw.flux_up), ("lwdn", lw.flux_down),
                           ("swdnb", sw.surface_down), ("swupt", sw.toa_up), ("swup", sw.flux_up), ("swdn", sw.flux_down)):
            out[key][idx] = np.asarray(value)[:n]
        # port heating_rate is dT/dt; WRF RTHRATENLW/SW = dT/dt / pi_phy (module_ra_rrtmg_lw.F TTEN1D/PI3D)
        out["hlw"][idx] = np.asarray(lw.heating_rate)[:n] / pi
        out["hsw"][idx] = np.asarray(sw.heating_rate)[:n] / pi
    return out


@pytest.fixture(scope="module")
def port_and_fixture():
    fx = load(FIXTURE)
    return _port_today(fx), fx


def test_port_today_is_pristine_arm_d_at_frozen_bounds(port_and_fixture):
    port, fx = port_and_fixture
    outside = columns_outside(fx, port, reference_arm="D")
    assert not outside.any(), np.flatnonzero(outside)


def test_port_today_misses_wrf_truth_on_cloudy_columns(port_and_fixture):
    port, fx = port_and_fixture
    clear = fx["category"] == "clear"
    outside = columns_outside(fx, port, reference_arm="A")
    assert not outside[clear].any()
    assert outside.sum() >= 100  # measured 119 / 159 cloudy columns (scalars + interfaces + heating)
