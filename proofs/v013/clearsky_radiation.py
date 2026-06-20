"""v0.13.0 CLEAR-SKY RADIATION FLUX PROOF (RRTMG SW+LW ...C vars).

Validates the new clear-sky (cloud-free) radiative-transfer pass added to the
JAX RRTMG SW/LW column solvers -- WRF's SECOND clear-sky stream
(`module_ra_rrtmg_sw.F` `pbbcd/pbbcu` via `vrtqdr_sw(zrefc,...)`, :8710-8743;
`module_ra_rrtmg_lw.F` `totdclfl/totuclfl`, :3417-3489).  These feed the 8 WRF
clear-sky wrfout vars:

  SW:  SWUPTC/SWDNTC (TOA up/down), SWUPBC/SWDNBC (surface up/down)
  LW:  LWUPTC/LWDNTC (TOA up/down), LWUPBC/LWDNBC (surface up/down)

Three independent checks (CPU, fp64; NEVER cores 4-31):

(A) ORACLE (NOT self-compare): feed the EXACT pristine-CPU-WRF column profiles
    (T/QVAPOR/P/clouds/TSK/ALBEDO/EMISS) from a real corpus wrfout into the JAX
    clear-sky pass and compare the JAX clear-sky fluxes to that SAME wrfout's
    genuine CPU-WRF RRTMG `...C` vars.  Because the JAX init/ozone/McICA seed
    differ from the operational CPU-WRF run, the ALL-SKY JAX-vs-WRF residual is
    measured on the identical profiles as the CALIBRATION FLOOR: the clear-sky
    residual must be of the SAME order as the all-sky residual (the clear-sky
    pass introduces no NEW error beyond the shared RRTMG state mismatch).  This
    is a Fortran-RRTMG-vs-JAX-RRTMG comparison, not a JAX-vs-JAX self-compare.

(B) ALL-SKY BYTE-UNCHANGED: solve(with_clear_sky=False) vs solve(..=True) must
    produce bit-identical main flux_down/flux_up (the clear-sky pass is purely
    additive; it must not perturb the operational outputs).

(C) PHYSICAL INVARIANTS + SCHEMA: WRF clear-sky physics must hold
    (SWDNTC==SWDNT exactly; clear-sky surface SW-down >= all-sky; clear-sky
    surface LW-down <= all-sky; clear-sky OLR >= all-sky OLR), and the 8 `...C`
    vars must be present in the wrfout writer schema with WRF-matching attrs.

Run (CPU only):
    GPUWRF_JAX_CACHE=0 JAX_PLATFORMS=cpu taskset -c 0-3 \
        python proofs/v013/clearsky_radiation.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

import netCDF4 as nc  # noqa: E402

from gpuwrf.physics.rrtmg_sw import RRTMGSWColumnState, solve_rrtmg_sw_column  # noqa: E402
from gpuwrf.physics.rrtmg_lw import RRTMGLWColumnState, solve_rrtmg_lw_column  # noqa: E402
from gpuwrf.io.wrfout_writer import WRFOUT_VARIABLE_SPECS  # noqa: E402

RDIR = "<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z"
DOM = "d03"
# A daytime time slice (sun up) so the SW clear-sky vars are meaningful.
WRFOUT = "wrfout_d03_2026-05-22_12:00:00"

GRAVITY = 9.81
RD = 287.0
P00 = 100000.0
CP = 1004.5
RCP = RD / CP
RRSW_SCON = 1368.22

CLEAR_SKY_VARS = [
    "SWUPTC", "SWDNTC", "SWUPBC", "SWDNBC",
    "LWUPTC", "LWDNTC", "LWUPBC", "LWDNBC",
]

# Absolute guardrail independent of the all-sky calibration floor. This keeps
# the oracle from passing solely because the shared all-sky JAX-vs-WRF mismatch
# is large on a variable.
ABSOLUTE_RMSE_CEILING_WM2 = 10.0


def to_columns(arr3d):
    """(nz,ny,nx) -> (ny*nx, nz) bottom-to-top mass-level columns (GPU layout)."""
    nz, ny, nx = arr3d.shape
    return np.moveaxis(arr3d, 0, -1).reshape(ny * nx, nz)


def load_wrf(path):
    ds = nc.Dataset(path)
    g = lambda k: np.squeeze(np.asarray(ds.variables[k][:]))
    Tpert = g("T")
    P = g("P"); PB = g("PB")
    QV = g("QVAPOR")
    QC = g("QCLOUD"); QI = g("QICE"); QS = g("QSNOW")
    QG = g("QGRAUP") if "QGRAUP" in ds.variables else np.zeros_like(QV)
    CLDFRA = g("CLDFRA")
    PH = g("PH"); PHB = g("PHB")
    COSZEN = g("COSZEN")
    ALBEDO = g("ALBEDO")
    TSK = g("TSK")
    EMISS = g("EMISS") if "EMISS" in ds.variables else np.full_like(TSK, 0.98)
    out = dict(
        wrf={v: g(v) for v in CLEAR_SKY_VARS
             + ["SWDNB", "SWUPB", "LWDNB", "LWUPB", "SWDNT", "SWUPT", "LWDNT", "LWUPT", "SWDOWN", "GLW", "OLR"]},
    )
    p_full = P + PB
    theta = Tpert + 300.0
    T = theta * (p_full / P00) ** RCP
    z_w = (PH + PHB) / GRAVITY
    dz = z_w[1:, :, :] - z_w[:-1, :, :]
    rho = p_full / (RD * T * (1.0 + 0.61 * QV))
    ds.close()
    out.update(dict(T=T, p=p_full, qv=QV, qc=QC, qi=QI, qs=QS, qg=QG, cldfra=CLDFRA,
                    dz=dz, rho=rho, coszen=COSZEN, albedo=ALBEDO, tsk=TSK, emiss=EMISS))
    return out


def pair_metrics(jax_arr, wrf_arr, mask):
    d = jax_arr[mask] - wrf_arr[mask]
    w = wrf_arr[mask]
    return {
        "n": int(mask.sum()),
        "wrf_mean": float(np.mean(w)),
        "jax_mean": float(np.mean(jax_arr[mask])),
        "bias_Wm2": float(np.mean(d)),
        "rmse_Wm2": float(np.sqrt(np.mean(d * d))),
        "mae_Wm2": float(np.mean(np.abs(d))),
        "max_abs_Wm2": float(np.max(np.abs(d))),
        "bias_pct": float(100.0 * np.mean(d) / max(abs(np.mean(w)), 1e-6)),
    }


def run():
    path = os.path.join(RDIR, WRFOUT)
    w = load_wrf(path)
    ny, nx = w["coszen"].shape
    ncol = ny * nx

    T = to_columns(w["T"]).astype(np.float64)
    p = to_columns(w["p"]).astype(np.float64)
    qv = to_columns(w["qv"]).astype(np.float64)
    qc = to_columns(w["qc"]).astype(np.float64)
    qi = to_columns(w["qi"]).astype(np.float64)
    qs = to_columns(w["qs"]).astype(np.float64)
    qg = to_columns(w["qg"]).astype(np.float64)
    cldfra = to_columns(w["cldfra"]).astype(np.float64)
    dz = to_columns(w["dz"]).astype(np.float64)
    rho = to_columns(w["rho"]).astype(np.float64)
    coszen = w["coszen"].reshape(-1).astype(np.float64)
    albedo = w["albedo"].reshape(-1).astype(np.float64)
    tsk = w["tsk"].reshape(-1).astype(np.float64)
    emiss = w["emiss"].reshape(-1).astype(np.float64)

    # SW TOA normalization: anchor the JAX solar source to WRF's SWDNTC so the
    # JAX TOA-incoming flux matches WRF's (COSZEN cancels; the clear-sky and
    # all-sky TOA-down are identical in WRF, so SWDNTC == SWDNT exactly).
    wrf_swdntc = w["wrf"]["SWDNTC"].reshape(-1).astype(np.float64)
    solar_source_scale = wrf_swdntc / np.maximum(coszen * RRSW_SCON, 1e-6)

    batch = 1024

    def sw_batched(clear):
        outs = []
        for s in range(0, ncol, batch):
            e = min(s + batch, ncol)
            st = RRTMGSWColumnState(
                jnp.asarray(T[s:e]), jnp.asarray(p[s:e]), jnp.asarray(qv[s:e]),
                jnp.asarray(qc[s:e]), jnp.asarray(qi[s:e]), jnp.asarray(qs[s:e]), jnp.asarray(qg[s:e]),
                jnp.asarray(cldfra[s:e]), jnp.asarray(albedo[s:e]), jnp.asarray(coszen[s:e]),
                jnp.asarray(dz[s:e]), jnp.asarray(rho[s:e]),
                solar_source_scale=jnp.asarray(solar_source_scale[s:e]),
            )
            r = solve_rrtmg_sw_column(st, debug=False, with_clear_sky=clear)
            if clear:
                outs.append((
                    np.asarray(r.flux_down[..., 0]), np.asarray(r.flux_up[..., 0]),
                    np.asarray(r.flux_down[..., -1]), np.asarray(r.flux_up[..., -1]),
                    np.asarray(r.clear_flux_down[..., 0]), np.asarray(r.clear_flux_up[..., 0]),
                    np.asarray(r.clear_flux_down[..., -1]), np.asarray(r.clear_flux_up[..., -1]),
                ))
            else:
                outs.append((np.asarray(r.flux_down[..., 0]), np.asarray(r.flux_up[..., 0]),
                             np.asarray(r.flux_down[..., -1]), np.asarray(r.flux_up[..., -1])))
        return tuple(np.concatenate(parts, axis=0) for parts in zip(*outs, strict=True))

    def lw_batched(clear):
        outs = []
        for s in range(0, ncol, batch):
            e = min(s + batch, ncol)
            st = RRTMGLWColumnState(
                jnp.asarray(T[s:e]), jnp.asarray(p[s:e]), jnp.asarray(qv[s:e]),
                jnp.asarray(qc[s:e]), jnp.asarray(qi[s:e]), jnp.asarray(qs[s:e]), jnp.asarray(qg[s:e]),
                jnp.asarray(cldfra[s:e]), jnp.asarray(tsk[s:e]), jnp.asarray(emiss[s:e]),
                jnp.asarray(dz[s:e]), jnp.asarray(rho[s:e]),
            )
            r = solve_rrtmg_lw_column(st, debug=False, with_clear_sky=clear)
            if clear:
                outs.append((
                    np.asarray(r.flux_down[..., 0]), np.asarray(r.flux_up[..., 0]),
                    np.asarray(r.flux_down[..., -1]), np.asarray(r.flux_up[..., -1]),
                    np.asarray(r.clear_flux_down[..., 0]), np.asarray(r.clear_flux_up[..., 0]),
                    np.asarray(r.clear_flux_down[..., -1]), np.asarray(r.clear_flux_up[..., -1]),
                ))
            else:
                outs.append((np.asarray(r.flux_down[..., 0]), np.asarray(r.flux_up[..., 0]),
                             np.asarray(r.flux_down[..., -1]), np.asarray(r.flux_up[..., -1])))
        return tuple(np.concatenate(parts, axis=0) for parts in zip(*outs, strict=True))

    # --- run all-sky-only (for byte-unchanged) and clear-sky-enabled passes ---
    sw_base = sw_batched(clear=False)
    sw_cs = sw_batched(clear=True)
    lw_base = lw_batched(clear=False)
    lw_cs = lw_batched(clear=True)

    # ---- (B) ALL-SKY BYTE-UNCHANGED ----
    byte_unchanged = {
        "SW_surface_down": bool(np.array_equal(sw_base[0], sw_cs[0])),
        "SW_surface_up": bool(np.array_equal(sw_base[1], sw_cs[1])),
        "SW_toa_down": bool(np.array_equal(sw_base[2], sw_cs[2])),
        "SW_toa_up": bool(np.array_equal(sw_base[3], sw_cs[3])),
        "LW_surface_down": bool(np.array_equal(lw_base[0], lw_cs[0])),
        "LW_surface_up": bool(np.array_equal(lw_base[1], lw_cs[1])),
        "LW_toa_down": bool(np.array_equal(lw_base[2], lw_cs[2])),
        "LW_toa_up": bool(np.array_equal(lw_base[3], lw_cs[3])),
    }
    byte_unchanged_pass = all(byte_unchanged.values())

    # Unpack clear-sky-enabled outputs.
    sw_sfc_dn, sw_sfc_up, sw_toa_dn, sw_toa_up, sw_csfc_dn, sw_csfc_up, sw_ctoa_dn, sw_ctoa_up = sw_cs
    lw_sfc_dn, lw_sfc_up, lw_toa_dn, lw_toa_up, lw_csfc_dn, lw_csfc_up, lw_ctoa_dn, lw_ctoa_up = lw_cs

    jax_clear = {
        "SWUPTC": sw_ctoa_up, "SWDNTC": sw_ctoa_dn, "SWUPBC": sw_csfc_up, "SWDNBC": sw_csfc_dn,
        "LWUPTC": lw_ctoa_up, "LWDNTC": lw_ctoa_dn, "LWUPBC": lw_csfc_up, "LWDNBC": lw_csfc_dn,
    }
    jax_allsky = {
        "SWUPT": sw_toa_up, "SWDNT": sw_toa_dn, "SWUPB": sw_sfc_up, "SWDNB": sw_sfc_dn,
        "LWUPT": lw_toa_up, "LWDNT": lw_toa_dn, "LWUPB": lw_sfc_up, "LWDNB": lw_sfc_dn,
    }
    allsky_of_clear = {
        "SWUPTC": "SWUPT", "SWDNTC": "SWDNT", "SWUPBC": "SWUPB", "SWDNBC": "SWDNB",
        "LWUPTC": "LWUPT", "LWDNTC": "LWDNT", "LWUPBC": "LWUPB", "LWDNBC": "LWDNB",
    }

    # Daylit land/ocean mask for SW; all columns for LW.
    daylit = coszen > 0.05

    # ---- (A) ORACLE vs pristine-CPU-WRF clear-sky + all-sky calibration floor ----
    oracle = {}
    for v in CLEAR_SKY_VARS:
        wrf_clear = w["wrf"][v].reshape(-1).astype(np.float64)
        wrf_all = w["wrf"][allsky_of_clear[v]].reshape(-1).astype(np.float64)
        mask = daylit if v.startswith("SW") else np.ones(ncol, dtype=bool)
        oracle[v] = {
            "clear_jax_vs_wrf": pair_metrics(jax_clear[v], wrf_clear, mask),
            "allsky_jax_vs_wrf_FLOOR": pair_metrics(jax_allsky[allsky_of_clear[v]], wrf_all, mask),
        }

    # PASS rule per var: the clear-sky JAX-vs-WRF RMSE must not exceed the all-sky
    # JAX-vs-WRF RMSE (the shared RRTMG state-mismatch floor) by more than a small
    # margin -> the clear-sky pass introduces no NEW systematic error.  margin =
    # max(1.15x floor, floor + 5 W/m2) to absorb the slightly different
    # cloud-sensitivity of the clear vs all-sky stream. A separate absolute RMSE
    # ceiling prevents a high all-sky floor from hiding a bad clear-sky pass.
    oracle_pass = {}
    oracle_absolute_pass = {}
    for v in CLEAR_SKY_VARS:
        clear_rmse = oracle[v]["clear_jax_vs_wrf"]["rmse_Wm2"]
        floor_rmse = oracle[v]["allsky_jax_vs_wrf_FLOOR"]["rmse_Wm2"]
        tol = max(1.15 * floor_rmse, floor_rmse + 5.0)
        floor_pass = clear_rmse <= tol
        absolute_pass = clear_rmse <= ABSOLUTE_RMSE_CEILING_WM2
        oracle_pass[v] = bool(floor_pass and absolute_pass)
        oracle_absolute_pass[v] = bool(absolute_pass)
        oracle[v]["tol_Wm2"] = float(tol)
        oracle[v]["absolute_rmse_ceiling_Wm2"] = float(ABSOLUTE_RMSE_CEILING_WM2)
        oracle[v]["floor_relative_pass"] = bool(floor_pass)
        oracle[v]["absolute_rmse_pass"] = bool(absolute_pass)
        oracle[v]["pass"] = oracle_pass[v]
    oracle_all_pass = all(oracle_pass.values())
    oracle_absolute_all_pass = all(oracle_absolute_pass.values())

    # ---- (C) PHYSICAL INVARIANTS ----
    invariants = {
        # WRF: clear-sky and all-sky TOA-down SW are identical (no cloud above TOA).
        "SWDNTC_eq_SWDNT_max_abs": float(np.max(np.abs(sw_ctoa_dn - sw_toa_dn))),
        # Clouds reduce surface SW-down -> clear-sky >= all-sky.
        "SW_clear_sfc_down_ge_allsky_frac": float(np.mean(sw_csfc_dn[daylit] >= sw_sfc_dn[daylit] - 1e-3)),
        # Clouds reflect SW to space -> clear-sky TOA-up <= all-sky.
        "SW_clear_toa_up_le_allsky_frac": float(np.mean(sw_ctoa_up[daylit] <= sw_toa_up[daylit] + 1e-3)),
        # Clouds add downward LW at surface -> clear-sky <= all-sky.
        "LW_clear_sfc_down_le_allsky_frac": float(np.mean(lw_csfc_dn <= lw_sfc_dn + 1e-3)),
        # Clouds trap LW -> clear-sky OLR (TOA-up) >= all-sky OLR.
        "LW_clear_olr_ge_allsky_frac": float(np.mean(lw_ctoa_up >= lw_toa_up - 1e-3)),
    }
    invariants_pass = (
        invariants["SWDNTC_eq_SWDNT_max_abs"] < 1e-9
        and invariants["SW_clear_sfc_down_ge_allsky_frac"] > 0.999
        and invariants["SW_clear_toa_up_le_allsky_frac"] > 0.999
        and invariants["LW_clear_sfc_down_le_allsky_frac"] > 0.999
        and invariants["LW_clear_olr_ge_allsky_frac"] > 0.999
    )

    # ---- (C) SCHEMA: 8 ...C vars in writer schema, attrs match WRF reference ----
    schema = {}
    ref = nc.Dataset(path)
    for v in CLEAR_SKY_VARS:
        in_schema = v in WRFOUT_VARIABLE_SPECS
        entry = {"in_schema": in_schema}
        if in_schema and v in ref.variables:
            spec = WRFOUT_VARIABLE_SPECS[v]
            rv = ref.variables[v]
            entry.update({
                "units_match": getattr(spec, "units", None) == getattr(rv, "units", None),
                "desc_match": getattr(spec, "description", None) == getattr(rv, "description", None),
                "memoryorder_match": getattr(spec, "memory_order", getattr(spec, "MemoryOrder", None)) == getattr(rv, "MemoryOrder", None),
            })
        schema[v] = entry
    ref.close()
    schema_pass = all(
        e.get("in_schema") and e.get("units_match", True) and e.get("desc_match", True)
        for e in schema.values()
    )

    overall = byte_unchanged_pass and oracle_all_pass and invariants_pass and schema_pass

    report = {
        "proof": "v0.13.0 clear-sky radiation flux (RRTMG SW+LW ...C vars)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "oracle_kind": "pristine-CPU-WRF RRTMG clear-sky wrfout (...C) vs JAX clear-sky on IDENTICAL profiles; all-sky JAX-vs-WRF = calibration floor (NOT self-compare)",
        "source_wrfout": path,
        "domain_shape": [int(ny), int(nx)],
        "n_columns": int(ncol),
        "n_daylit": int(daylit.sum()),
        "platform": jax.default_backend(),
        "fp64": bool(jax.config.read("jax_enable_x64")),
        "vars_added": CLEAR_SKY_VARS,
        "A_oracle": oracle,
        "A_oracle_pass": oracle_all_pass,
        "A_oracle_absolute_rmse_ceiling_Wm2": ABSOLUTE_RMSE_CEILING_WM2,
        "A_oracle_absolute_rmse_pass": oracle_absolute_all_pass,
        "B_allsky_byte_unchanged": byte_unchanged,
        "B_allsky_byte_unchanged_pass": byte_unchanged_pass,
        "C_invariants": invariants,
        "C_invariants_pass": invariants_pass,
        "C_schema": schema,
        "C_schema_pass": schema_pass,
        "OVERALL_PASS": bool(overall),
    }
    return report


if __name__ == "__main__":
    rep = run()
    out_path = os.path.join(os.path.dirname(__file__), "clearsky_radiation.json")
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({k: rep[k] for k in (
        "A_oracle_pass", "B_allsky_byte_unchanged_pass", "C_invariants_pass",
        "C_schema_pass", "OVERALL_PASS")}, indent=2))
    print("wrote", out_path)
    sys.exit(0 if rep["OVERALL_PASS"] else 1)
