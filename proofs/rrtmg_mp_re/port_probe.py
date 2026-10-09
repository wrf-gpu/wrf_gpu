"""RE01 port probe (CPU, legacy column solvers): today's port RRTMG on the oracle operands vs arms A/B/C.

Expectation if arm C models the port: port ~= C at frozen bounds, port != A on the radius-sensitive cloudy columns.
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.rrtmg_lw import RRTMGLWColumnState, solve_rrtmg_lw_column
from gpuwrf.physics.rrtmg_sw import RRTMGSWColumnState, solve_rrtmg_sw_column
from gpuwrf.physics.wrf_clwrf_ghg import clwrf_ssp245_gases_for_time

RRSW_SCON = 1368.22
fx = dict(np.load(sys.argv[1]))
meta = json.loads(Path(sys.argv[2]).read_text())["columns"]
ncol, nz = fx["in_t"].shape
f64 = lambda a: jnp.asarray(np.asarray(a, np.float64))
out = {k: np.zeros((ncol,) + s) for k, s in (("lw_sfc_dn", ()), ("lw_toa_up", ()), ("sw_sfc_dn", ()), ("sw_toa_up", ()),
                                             ("lw_heat", (nz,)), ("sw_heat", (nz,)))}
start = datetime(2026, 2, 28)
for stamp in sorted({m["stamp"] for m in meta}):
    idx = np.asarray([i for i, m in enumerate(meta) if m["stamp"] == stamp])
    when = datetime.strptime(stamp, "%Y-%m-%d_%H:%M:%S")
    ghg = clwrf_ssp245_gases_for_time(when)
    common = dict(T=f64(fx["in_t"][idx]), p=f64(fx["in_p_hyd"][idx]), qv=f64(fx["in_qv"][idx]), qc=f64(fx["in_qc_rad"][idx]),
                  qi=f64(fx["in_qi_rad"][idx]), qs=f64(fx["in_qs"][idx]), qg=f64(fx["in_qg"][idx]),
                  cloud_fraction=f64(fx["in_cldfra"][idx]), dz=f64(fx["in_dz8w"][idx, :nz]), rho=f64(fx["in_rho"][idx]))
    lw = RRTMGLWColumnState(**common, surface_temperature=f64(fx["in_tsk"][idx]), surface_emissivity=f64(fx["in_emiss"][idx]),
                            top_pressure_pa=float(fx["in_p_top"]), pressure_interfaces=f64(fx["in_p_hyd_w"][idx]),
                            temperature_interfaces=f64(fx["in_t8w"][idx]), co2_vmr=ghg.co2_vmr, n2o_vmr=ghg.n2o_vmr,
                            ch4_vmr=ghg.ch4_vmr, cfc11_vmr=ghg.cfc11_vmr, cfc12_vmr=ghg.cfc12_vmr,
                            ozone_vmr=f64(fx["col_o3"][idx]))
    sw = RRTMGSWColumnState(**common, surface_albedo=f64(fx["in_albedo"][idx]), coszen=f64(fx["col_coszen"][idx]),
                            solar_source_scale=f64(fx["col_solcon"][idx] / RRSW_SCON),
                            pressure_interfaces=f64(fx["in_p_hyd_w"][idx]), temperature_interfaces=f64(fx["in_t8w"][idx]),
                            co2_vmr=ghg.co2_vmr, n2o_vmr=ghg.n2o_vmr, ch4_vmr=ghg.ch4_vmr, ozone_vmr=f64(fx["col_o3"][idx]))
    rl = solve_rrtmg_lw_column(lw)
    rs = solve_rrtmg_sw_column(sw)
    out["lw_sfc_dn"][idx] = np.asarray(rl.surface_down); out["lw_toa_up"][idx] = np.asarray(rl.toa_up)
    out["sw_sfc_dn"][idx] = np.asarray(rs.surface_down); out["sw_toa_up"][idx] = np.asarray(rs.toa_up)
    out["lw_heat"][idx] = np.asarray(rl.heating_rate); out["sw_heat"][idx] = np.asarray(rs.heating_rate)
    print(stamp, len(idx), "flux shapes", np.asarray(rl.flux_up).shape, np.asarray(rs.flux_up).shape, flush=True)

cat = fx["category"]
day = fx["col_coszen"] > 0
res = {}
for arm in ("A", "B", "C", "D", "E"):
    r = {}
    for name, port, ref, mask in (("GLW", out["lw_sfc_dn"], fx[f"{arm}_lwdnb"], np.ones(ncol, bool)),
                                  ("OLR", out["lw_toa_up"], fx[f"{arm}_lwupt"], np.ones(ncol, bool)),
                                  ("SWDOWN", out["sw_sfc_dn"], fx[f"{arm}_swdnb"], day),
                                  ("SWUPT", out["sw_toa_up"], fx[f"{arm}_swupt"], day)):
        d = (port - ref)[mask]
        bound = np.abs(d) / (1.0 + 0.05 * np.abs(ref[mask]))
        r[name] = {"max_abs": float(np.abs(d).max()), "mean": float(d.mean()), "n_outside": int((bound > 1).sum()),
                   "n": int(mask.sum()), "clear_max_abs": float(np.abs((port - ref)[mask & (cat == "clear")]).max())}
    for name, port, ref in (("hLW", out["lw_heat"], fx[f"{arm}_hlw"]), ("hSW", out["sw_heat"], fx[f"{arm}_hsw"]),
                            ("hLW_x_pi", out["lw_heat"], fx[f"{arm}_hlw"] * fx["in_pi"]),
                            ("hSW_x_pi", out["sw_heat"], fx[f"{arm}_hsw"] * fx["in_pi"])):
        d = port - ref
        r[name] = {"max_abs_K_d": float(np.abs(d).max() * 86400),
                   "cols_outside": int(((np.abs(d) / (1e-4 + 0.05 * np.abs(ref))) > 1).any(1).sum())}
    res[arm] = r
Path(sys.argv[3]).write_text(json.dumps(res, indent=1))
for arm, r in res.items():
    print(arm, json.dumps(r))
