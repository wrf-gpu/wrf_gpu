"""Real RE01 operands through the product producer and LW/SW operators.

CPU legacy probe; native CUDA lowering is a separate gate. No tolerance fitting.
"""
from datetime import datetime, timedelta
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "fid_q2"))
from rrtmg_mp_re_fixture import load, columns_outside, bound_ratio, FLUX_BOUND, HEATING_BOUND
from gpuwrf.physics.rrtmg_mp_re import calc_thompson_effective_radii
from gpuwrf.physics.rrtmg_lw import RRTMGLWColumnState, solve_rrtmg_lw_column
from gpuwrf.physics.rrtmg_sw import RRTMGSWColumnState, solve_rrtmg_sw_column
from gpuwrf.physics.wrf_clwrf_ghg import clwrf_ssp245_gases_for_time


def run(fx, *, radii=True):
    ncol, nz = fx["in_t"].shape
    out = {k: np.zeros((ncol,) + shape) for k, shape in (
        ("glw", ()), ("olr", ()), ("swdnb", ()), ("swupt", ()),
        ("lwup", (nz + 2,)), ("lwdn", (nz + 2,)), ("swup", (nz + 2,)),
        ("swdn", (nz + 2,)), ("hlw", (nz,)), ("hsw", (nz,)))}
    hours = fx["in_tau"].astype(int)
    width = max(int((hours == h).sum()) for h in np.unique(hours))
    for hour in np.unique(hours):
        idx = np.flatnonzero(hours == hour)
        pad = np.resize(idx, width)
        a = lambda key: jnp.asarray(fx[key][pad], jnp.float64)
        ghg = clwrf_ssp245_gases_for_time(datetime(2026, 2, 28) + timedelta(hours=int(hour)))
        common = dict(T=a("in_t"), p=a("in_p_hyd"), qv=a("in_qv"), qc=a("in_qc_rad"),
                      qi=a("in_qi_rad"), qs=a("in_qs"), qg=a("in_qg"), cloud_fraction=a("in_cldfra"),
                      dz=a("in_dz8w")[:, :nz], rho=a("in_rho"), pressure_interfaces=a("in_p_hyd_w"),
                      temperature_interfaces=a("in_t8w"), ozone_vmr=a("col_o3"),
                      co2_vmr=ghg.co2_vmr, n2o_vmr=ghg.n2o_vmr, ch4_vmr=ghg.ch4_vmr)
        if radii:
            # MP uses physical QC and nonhydrostatic p_phy, not the radiation view.
            radius = calc_thompson_effective_radii(*(a("in_" + name) for name in ("t", "p", "qv", "qc", "qi", "ni", "qs")))
            common.update(zip(("re_cloud", "re_ice", "re_snow"), radius))
            common["xland"] = a("in_xland")
        lw = solve_rrtmg_lw_column(RRTMGLWColumnState(
            **common, surface_temperature=a("in_tsk"), surface_emissivity=a("in_emiss"),
            top_pressure_pa=float(fx["in_p_top"]), cfc11_vmr=ghg.cfc11_vmr, cfc12_vmr=ghg.cfc12_vmr))
        sw = solve_rrtmg_sw_column(RRTMGSWColumnState(
            **common, surface_albedo=a("in_albedo"), coszen=a("col_coszen"),
            solar_source_scale=a("col_solcon") / 1368.22))
        for key, value in (("glw", lw.surface_down), ("olr", lw.toa_up), ("swdnb", sw.surface_down),
                           ("swupt", sw.toa_up), ("lwup", lw.flux_up), ("lwdn", lw.flux_down),
                           ("swup", sw.flux_up), ("swdn", sw.flux_down)):
            out[key][idx] = np.asarray(value)[:len(idx)]
        out["hlw"][idx] = np.asarray(lw.heating_rate)[:len(idx)] / fx["in_pi"][idx]
        out["hsw"][idx] = np.asarray(sw.heating_rate)[:len(idx)] / fx["in_pi"][idx]
        print(f"tau={hour}, columns={len(idx)}", flush=True)
    return out


if __name__ == "__main__":
    import json
    arm, path = sys.argv[1:]
    fx = load()
    out = run(fx, radii=arm in ("A", "E"))
    outside = columns_outside(fx, out, reference_arm=arm)
    summary = {"arm": arm, "columns": len(outside), "outside": int(outside.sum()),
               "bad_indices": np.flatnonzero(outside).tolist(), "max_abs": {}, "max_bound_ratio": {}}
    for key, value in out.items():
        ref = fx[f"{arm}_{key}"]
        if key in ("lwup", "lwdn", "swup", "swdn"):
            ref, value = ref[:, :-1], value[:, :-1]
        summary["max_abs"][key] = float(np.max(np.abs(value - ref)))
        summary["max_bound_ratio"][key] = float(np.max(bound_ratio(ref, value, HEATING_BOUND if key in ("hlw", "hsw") else FLUX_BOUND)))
    np.savez_compressed(str(path) + ".npz", **out)
    Path(str(path) + ".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    assert not outside.any(), summary
