#!/usr/bin/env python3
"""Independent verification of the Canary L2 d02 QVAPOR-bias attribution.

Re-derives, from the raw v0.15 GPU wrfout vs the CPU truth wrfout, the three
load-bearing claims in proofs/v015/all_green_fixes.md "Miss 2":

  (A) surface-layer flux law is EXONERATED  (h1 LH/QFX/HFX match CPU closely)
  (B) the QVAPOR error is a column-CONSERVING vertical redistribution
      (moister+colder below the trade inversion, drier+warmer above; the
       column-integrated qv difference is tiny relative to the per-level RMSE)
  (C) the error is OCEAN-localized and the resolved QCLOUD over the marine deck
      is sub-radiative (so the SGS-cloud chain is inert here).

No GPU. Pure xarray/numpy over interior+ocean columns. Writes a JSON proof
object the manager can attach to the v0.15 close.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

CPU = Path("/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/"
           "20260501_18z_l2_72h_20260519T173026Z")
GPU = Path("/mnt/data/wrf_gpu_validation/v015_canary_d02_72h_allgreen/"
           "gpu_output/l2_d02_20260501_18z_l2_72h_20260519T173026Z")
FRAME = 5  # drop the outer nest-relaxation ring like the atlas interior pool


def _open(path):
    return xr.open_dataset(path, decode_times=False)


def _interior(a):
    """Trim the outer FRAME cells on the two horizontal dims (last two)."""
    return a[..., FRAME:-FRAME, FRAME:-FRAME]


def _field(ds, name):
    return np.asarray(ds[name].values[0], dtype=np.float64)  # drop Time axis


def lead_files(stamp):
    cpu = CPU / f"wrfout_d02_{stamp}"
    gpu = GPU / f"wrfout_d02_{stamp}"
    return (cpu if cpu.exists() else None, gpu if gpu.exists() else None)


def surface_flux_check(stamps):
    """Claim A: per-lead surface-flux bias over OCEAN interior columns."""
    rows = []
    for stamp in stamps:
        cpuf, gpuf = lead_files(stamp)
        if cpuf is None or gpuf is None:
            continue
        c, g = _open(cpuf), _open(gpuf)
        xland = _interior(_field(c, "XLAND"))
        ocean = xland > 1.5
        out = {"stamp": stamp, "n_ocean": int(ocean.sum())}
        # GPU wrfout exposes LH (latent heat = moisture-flux observable) + HFX;
        # QFX is not written by the GPU writer, LH is the equivalent handle.
        for v in ("LH", "HFX"):
            cv = _interior(_field(c, v))[ocean]
            gv = _interior(_field(g, v))[ocean]
            cmean = float(np.mean(cv))
            gmean = float(np.mean(gv))
            out[f"{v}_cpu_mean"] = cmean
            out[f"{v}_gpu_mean"] = gmean
            out[f"{v}_mean_bias"] = gmean - cmean
            out[f"{v}_pct_bias"] = (100.0 * (gmean - cmean) / cmean
                                    if abs(cmean) > 1e-9 else float("nan"))
            out[f"{v}_rmse"] = float(np.sqrt(np.mean((gv - cv) ** 2)))
        c.close(); g.close()
        rows.append(out)
    return rows


def vertical_redistribution_check(stamp):
    """Claim B: vertical qv/T bias profile + column-conservation over ocean."""
    cpuf, gpuf = lead_files(stamp)
    c, g = _open(cpuf), _open(gpuf)
    xland = _interior(_field(c, "XLAND"))
    ocean = xland > 1.5  # (ny,nx)
    qc_cpu = _interior(_field(c, "QVAPOR"))   # (nz,ny,nx)
    qc_gpu = _interior(_field(g, "QVAPOR"))
    t_cpu = _interior(_field(c, "T"))         # perturbation theta
    t_gpu = _interior(_field(g, "T"))
    nz = qc_cpu.shape[0]
    prof = []
    for k in range(nz):
        qd = (qc_gpu[k] - qc_cpu[k])[ocean]
        td = (t_gpu[k] - t_cpu[k])[ocean]
        prof.append({
            "level": k,
            "qv_bias_g_per_kg": float(np.mean(qd) * 1000.0),
            "qv_rmse_g_per_kg": float(np.sqrt(np.mean(qd ** 2)) * 1000.0),
            "T_bias_K": float(np.mean(td)),
        })
    omask = np.broadcast_to(ocean, qc_cpu.shape)
    qv_col_cpu = qc_cpu.mean(axis=0)[ocean]
    qv_col_gpu = qc_gpu.mean(axis=0)[ocean]
    col_mean_bias = float(np.mean(qv_col_gpu - qv_col_cpu))
    pooled_level_rmse = float(np.sqrt(np.mean((qc_gpu - qc_cpu)[omask] ** 2)))
    c.close(); g.close()
    return {
        "stamp": stamp,
        "nz": nz,
        "ocean_columns": int(ocean.sum()),
        "column_mean_qv_bias_kg_per_kg": col_mean_bias,
        "pooled_per_level_qv_rmse_kg_per_kg": pooled_level_rmse,
        "redistribution_ratio_colmean_over_rmse": (
            abs(col_mean_bias) / pooled_level_rmse if pooled_level_rmse else float("nan")),
        "profile": prof,
    }


def ocean_localization_check(stamp):
    """Claim C: ocean-vs-land split of the per-level qv RMSE + QCLOUD level."""
    cpuf, gpuf = lead_files(stamp)
    c, g = _open(cpuf), _open(gpuf)
    xland = _interior(_field(c, "XLAND"))
    ocean = xland > 1.5
    land = ~ocean
    qc_cpu = _interior(_field(c, "QVAPOR"))
    qc_gpu = _interior(_field(g, "QVAPOR"))
    qcloud_cpu = _interior(_field(c, "QCLOUD"))
    d = qc_gpu - qc_cpu
    om = np.broadcast_to(ocean, d.shape)
    lm = np.broadcast_to(land, d.shape)
    qom = np.broadcast_to(ocean, qcloud_cpu.shape)
    out = {
        "stamp": stamp,
        "ocean_fraction": float(ocean.mean()),
        "ocean_pooled_qv_rmse": float(np.sqrt(np.mean(d[om] ** 2))),
        "land_pooled_qv_rmse": float(np.sqrt(np.mean(d[lm] ** 2))) if lm.any() else None,
        "qcloud_ocean_mean": float(qcloud_cpu[qom].mean()),
        "qcloud_ocean_max": float(qcloud_cpu[qom].max()),
        "qcloud_ocean_cells_gt_1e6": int((qcloud_cpu[qom] > 1e-6).sum()),
    }
    c.close(); g.close()
    return out


def main():
    stamps_flux = [
        "2026-05-01_19:00:00",  # h1
        "2026-05-02_00:00:00",  # h6
        "2026-05-02_06:00:00",  # h12
        "2026-05-02_18:00:00",  # h24
        "2026-05-03_18:00:00",  # h48
    ]
    result = {
        "purpose": "independent verification of Canary QVAPOR attribution (all_green_fixes Miss 2)",
        "cpu_truth": str(CPU),
        "gpu_run": str(GPU),
        "frame_dropped": FRAME,
        "A_surface_flux_check": surface_flux_check(stamps_flux),
        "B_vertical_redistribution_h48": vertical_redistribution_check("2026-05-03_18:00:00"),
        "B_vertical_redistribution_h24": vertical_redistribution_check("2026-05-02_18:00:00"),
        "C_ocean_localization_h48": ocean_localization_check("2026-05-03_18:00:00"),
    }
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/qvapor_attribution.json")
    out_path.write_text(json.dumps(result, indent=2))
    print("=== A) surface flux (ocean interior) ===")
    for r in result["A_surface_flux_check"]:
        print(f"  {r['stamp']}  LH cpu {r['LH_cpu_mean']:6.1f} gpu {r['LH_gpu_mean']:6.1f} "
              f"bias {r['LH_mean_bias']:+.2f} W/m2 ({r['LH_pct_bias']:+.1f}%)  |  "
              f"HFX cpu {r['HFX_cpu_mean']:6.1f} gpu {r['HFX_gpu_mean']:6.1f} "
              f"bias {r['HFX_mean_bias']:+.2f} ({r['HFX_pct_bias']:+.1f}%)")
    print("=== B) vertical redistribution h48 (ocean) ===")
    b = result["B_vertical_redistribution_h48"]
    print(f"  column-mean qv bias = {b['column_mean_qv_bias_kg_per_kg']:+.3e} kg/kg")
    print(f"  pooled per-level qv RMSE = {b['pooled_per_level_qv_rmse_kg_per_kg']:.3e} kg/kg")
    print(f"  ratio colmean/rmse = {b['redistribution_ratio_colmean_over_rmse']:.3f}  (<<1 => redistribution)")
    print("  level qv_bias[g/kg]  T_bias[K]")
    for p in b["profile"]:
        print(f"   k{p['level']:2d}  {p['qv_bias_g_per_kg']:+6.2f}    {p['T_bias_K']:+5.2f}")
    print("=== C) ocean localization h48 ===")
    c = result["C_ocean_localization_h48"]
    print(f"  ocean pooled qv RMSE = {c['ocean_pooled_qv_rmse']:.3e}  land = {c['land_pooled_qv_rmse']}")
    print(f"  QCLOUD ocean mean = {c['qcloud_ocean_mean']:.2e}  max = {c['qcloud_ocean_max']:.2e}  "
          f"cells>1e-6 = {c['qcloud_ocean_cells_gt_1e6']}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
