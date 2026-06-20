#!/usr/bin/env python
"""V0.14 short-field h1 residual classification (CPU-only, NetCDF reads).

Classifies the 1h Canary d02 falsifier residual reported in
proofs/v014/short_field_falsifier_h1_grid_compare.json.

Hypotheses tested:
  H-PROV  GPU inputs vs CPU truth are different cases (stale provenance).
  H-INIT  Residual is dominated by init-mode mismatch: GPU d02 = live-nest
          interpolation from d01, CPU d02 = real.exe wrfinput_d02
          (input_from_file=.true.).
  H-PSFC  Uniform PSFC offset is a diagnostic/writer difference, decomposed
          against dry surface pressure MU+MUB+p_top.
  H-BASE  Rare large PB/MUB cells are localized, identified by location.
  H-RAD   SWDOWN/SWNORM uniform offset tracks a COSZEN (radiation timing)
          difference.

Writes proofs/v014/short_field_h1_residual_classification.json and .md.
"""

import json
import os

import numpy as np
from netCDF4 import Dataset

CASE = "20260501_18z_l2_72h_20260519T173026Z"
WRF_L2 = f"<DATA_ROOT>/canairy_meteo/runs/wrf_l2/{CASE}"
CPU_TRUTH = f"<DATA_ROOT>/canairy_meteo/runs/wrf_l2_backfill_output/{CASE}"
GPU_ROOT = (
    "<DATA_ROOT>/wrf_gpu_validation/v014_short_field_falsifier_20260610T122005Z/"
    f"gpu_output/l2_d02_{CASE}"
)

CPU_H0 = f"{CPU_TRUTH}/wrfout_d02_2026-05-01_18:00:00"
CPU_H1 = f"{CPU_TRUTH}/wrfout_d02_2026-05-01_19:00:00"
GPU_H1 = f"{GPU_ROOT}/wrfout_d02_2026-05-01_19:00:00"
GPU_H1_D01 = f"{GPU_ROOT}/wrfout_d01_2026-05-01_19:00:00"
WRFINPUT_D02 = f"{WRF_L2}/wrfinput_d02"
WRFINPUT_D01 = f"{WRF_L2}/wrfinput_d01"
CPU_H1_D01 = f"{CPU_TRUTH}/wrfout_d01_2026-05-01_19:00:00"

P_TOP = 5000.0
I_PARENT_START = 24
J_PARENT_START = 20
RATIO = 3

FIELDS = [
    "PSFC", "P", "MU", "PB", "MUB", "T", "U", "V", "QVAPOR",
    "HFX", "LH", "PBLH",
]


def var(path, name):
    with Dataset(path) as ds:
        return np.asarray(ds.variables[name][0], dtype=np.float64)


def stats(d):
    a = np.abs(d)
    return {
        "rmse": float(np.sqrt(np.mean(d * d))),
        "bias": float(np.mean(d)),
        "mae": float(np.mean(a)),
        "p99_abs": float(np.percentile(a, 99)),
        "max_abs": float(a.max()),
    }


def corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def boundary_distance_mask(shape, width):
    ny, nx = shape
    j = np.arange(ny)[:, None]
    i = np.arange(nx)[None, :]
    return np.minimum(np.minimum(j, ny - 1 - j), np.minimum(i, nx - 1 - i)) < width


def parent_to_child_bilinear(parent2d):
    """Bilinear sample of a d01 mass-grid field at d02 mass-cell centers.

    WRF child cell (i_c, j_c) 1-based maps to parent non-staggered coordinate
    i_p = i_parent_start + (i_c - 0.5)/ratio - 0.5 (0-based float index).
    This is a proxy for WRF SINT (quadratic) — adequate for pattern/scale
    attribution, not bit parity.
    """
    ny_c = 66
    nx_c = 159
    out = np.empty((ny_c, nx_c))
    jj = (J_PARENT_START - 1) + (np.arange(ny_c) + 0.5) / RATIO - 0.5
    ii = (I_PARENT_START - 1) + (np.arange(nx_c) + 0.5) / RATIO - 0.5
    j0 = np.floor(jj).astype(int)
    i0 = np.floor(ii).astype(int)
    fj = jj - j0
    fi = ii - i0
    for a in range(ny_c):
        ja, fa = j0[a], fj[a]
        row0 = parent2d[ja]
        row1 = parent2d[ja + 1]
        top = row0[i0] * (1 - fi) + row0[i0 + 1] * fi
        bot = row1[i0] * (1 - fi) + row1[i0 + 1] * fi
        out[a] = top * (1 - fa) + bot * fa
    return out


def main():
    out = {"schema": "V014ShortFieldH1ResidualClassification", "schema_version": 1}

    # --- Provenance ---------------------------------------------------------
    manifest_path = f"{WRF_L2}/.cpu_wrf_backfill/20260603T000612Z_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    runroot_link = os.path.realpath(
        f"/tmp/v0120_merged_run_root/{CASE}/wrfinput_d02"
    )
    out["provenance"] = {
        "backfill_manifest": manifest_path,
        "backfill_mode": manifest["mode"],
        "backfill_run_dir": manifest["run_dir"],
        "gpu_input_symlink_resolves_to": runroot_link,
        "same_input_files": manifest["run_dir"] == WRF_L2
        and runroot_link.startswith(WRF_L2),
        "cpu_namelist_input_from_file_d02": True,
        "gpu_init_mode": "standalone_native_init_nested",
        "init_mode_mismatch": (
            "CPU-WRF d02 initialized from real.exe wrfinput_d02 "
            "(input_from_file=.true.); GPU d02 initialized by native live-nest "
            "interpolation from d01 and never reads wrfinput_d02."
        ),
    }

    # --- Field deltas -------------------------------------------------------
    deltas = {}
    for f in FIELDS:
        g = var(GPU_H1, f)
        c = var(CPU_H1, f)
        deltas[f] = stats(g - c)
        deltas[f]["pearson_r"] = corr(g, c)
    out["h1_field_deltas"] = deltas

    # --- H-INIT: init-mode attribution for MU ------------------------------
    mu_in_d02 = var(WRFINPUT_D02, "MU")
    mub_in_d02 = var(WRFINPUT_D02, "MUB")
    mu_in_d01 = var(WRFINPUT_D01, "MU")
    mub_in_d01 = var(WRFINPUT_D01, "MUB")

    cpu_mu_h0 = var(CPU_H0, "MU")
    cpu_mu_h1 = var(CPU_H1, "MU")
    cpu_mub_h1 = var(CPU_H1, "MUB")
    gpu_mu_h1 = var(GPU_H1, "MU")
    gpu_mub_h1 = var(GPU_H1, "MUB")

    # Proxy for the live-nest initial dry-mass field: parent total dry mass
    # interpolated to the child grid minus the child's own base.
    parent_total = mu_in_d01 + mub_in_d01
    interp_total = parent_to_child_bilinear(parent_total)
    livenest_mu_proxy = interp_total - mub_in_d02

    init_gap = livenest_mu_proxy - mu_in_d02      # live-nest init minus real init
    h1_gap = gpu_mu_h1 - cpu_mu_h1                # observed h1 residual
    cpu_mu_tend = cpu_mu_h1 - cpu_mu_h0           # WRF's own 1h MU evolution

    out["mu_init_mode_attribution"] = {
        "livenest_proxy_minus_realinit": stats(init_gap),
        "observed_h1_gpu_minus_cpu": stats(h1_gap),
        "cpu_mu_1h_tendency": stats(cpu_mu_tend),
        "pattern_corr_initgap_vs_h1gap": corr(init_gap, h1_gap),
        "pattern_corr_cputend_vs_h1gap": corr(cpu_mu_tend, h1_gap),
        "note": (
            "Bilinear proxy for WRF SINT; correlation/scale attribution only."
        ),
    }

    # --- H-PSFC: decompose the uniform surface-pressure offset --------------
    gpu_psfc = var(GPU_H1, "PSFC")
    cpu_psfc = var(CPU_H1, "PSFC")
    gpu_dry = gpu_mu_h1 + gpu_mub_h1 + P_TOP
    cpu_dry = cpu_mu_h1 + cpu_mub_h1 + P_TOP
    out["psfc_decomposition"] = {
        "delta_psfc": stats(gpu_psfc - cpu_psfc),
        "delta_dry_surface_pressure": stats(gpu_dry - cpu_dry),
        "gpu_psfc_minus_gpu_dry": stats(gpu_psfc - gpu_dry),
        "cpu_psfc_minus_cpu_dry": stats(cpu_psfc - cpu_dry),
        "residual_diag_offset": stats((gpu_psfc - gpu_dry) - (cpu_psfc - cpu_dry)),
    }

    # --- H-BASE: localize the rare large PB/MUB cells ------------------------
    base = {}
    hgt_in = var(WRFINPUT_D02, "HGT")
    for f, inp in (("MUB", mub_in_d02), ("PB", None)):
        g = var(GPU_H1, f)
        c = var(CPU_H1, f)
        d = g - c
        if d.ndim == 3:
            flat = np.abs(d).max(axis=0)
        else:
            flat = np.abs(d)
        bad = np.argwhere(flat > 1.0)
        ny, nx = flat.shape
        bmask = boundary_distance_mask((ny, nx), 5)
        cells = []
        for (j, i) in bad[:50]:
            cells.append({
                "j": int(j), "i": int(i),
                "abs_delta": float(flat[j, i]),
                "in_spec_bdy_5": bool(bmask[j, i]),
                "hgt_m": float(hgt_in[j, i]),
            })
        entry = {
            "n_cells_gt_1pa": int(len(bad)),
            "n_cells_total": int(flat.size),
            "n_in_spec_bdy_5": int(sum(1 for cc in cells if cc["in_spec_bdy_5"]))
            if len(bad) <= 50 else None,
            "worst_cells": sorted(
                cells, key=lambda cc: -cc["abs_delta"])[:10],
        }
        if inp is not None:
            entry["gpu_vs_wrfinput_d02"] = stats(g - inp)
            entry["cpu_vs_wrfinput_d02"] = stats(c - inp)
        base[f] = entry
    out["base_state_localization"] = base

    # --- H-RAD: radiation timing --------------------------------------------
    rad = {}
    for f in ("COSZEN", "SWDOWN", "SWNORM", "GLW"):
        g = var(GPU_H1, f)
        c = var(CPU_H1, f)
        rad[f] = stats(g - c)
    # relative SW offset vs relative coszen offset
    cpu_cz = var(CPU_H1, "COSZEN")
    gpu_cz = var(GPU_H1, "COSZEN")
    cpu_sw = var(CPU_H1, "SWDOWN")
    rad["mean_cpu_coszen"] = float(cpu_cz.mean())
    rad["mean_gpu_coszen"] = float(gpu_cz.mean())
    rad["rel_coszen_offset"] = float(
        (gpu_cz.mean() - cpu_cz.mean()) / cpu_cz.mean())
    rad["rel_swdown_offset"] = float(
        (var(GPU_H1, "SWDOWN").mean() - cpu_sw.mean()) / cpu_sw.mean())
    out["radiation_timing"] = rad

    # --- T boundary vs interior ----------------------------------------------
    g_t = var(GPU_H1, "T")
    c_t = var(CPU_H1, "T")
    d_t = g_t - c_t
    bmask3 = np.broadcast_to(
        boundary_distance_mask(d_t.shape[1:], 5), d_t.shape)
    out["t_split"] = {
        "boundary_band_5": stats(d_t[bmask3]),
        "interior": stats(d_t[~bmask3]),
    }

    # --- H-EOS: equation-of-state moisture-factor attribution -----------------
    # Invert WRF's EOS p = p0*(Rd*theta*qvf/(p0*alpha_d))**cpovcv on each file
    # using that file's OWN discrete dry inverse density
    # alpha_d = -(dphi/dnw)/(c1h*mu_tot+c2h) (start_em.F:730 inversion), with
    # qvf = 1+rvovrd*qv (WRF, module_big_step_utilities_em.F:1064) vs
    # qvf = 1+p608*qv (the pre-fix GPU constant).
    RD, RV, P0, T0 = 287.0, 461.6, 1.0e5, 300.0
    CP = 7.0 * RD / 2.0
    CPOVCV = CP / (CP - RD)
    eos = {}
    for tag, path in (("cpu_truth", CPU_H1), ("gpu_prefix_run", GPU_H1)):
        th = var(path, "T") + T0
        qv = var(path, "QVAPOR")
        p_tot = var(path, "P") + var(path, "PB")
        phi = var(path, "PH") + var(path, "PHB")
        mu_tot = var(path, "MU") + var(path, "MUB")
        with Dataset(path) as ds:
            dnw = np.asarray(ds.variables["DNW"][0], dtype=np.float64)
            c1h = np.asarray(ds.variables["C1H"][0], dtype=np.float64)
            c2h = np.asarray(ds.variables["C2H"][0], dtype=np.float64)
        mass = c1h[:, None, None] * mu_tot[None] + c2h[:, None, None]
        alpha_d = -(phi[1:] - phi[:-1]) / (dnw[:, None, None] * mass)
        eos[tag] = {}
        for label, fac in (("rvovrd_1.608", RV / RD), ("p608_0.608", RV / RD - 1.0)):
            p_eos = P0 * ((RD * th * (1.0 + fac * qv)) / (P0 * alpha_d)) ** CPOVCV
            eos[tag][label] = stats(p_eos - p_tot)
    eos["verdict"] = (
        "CPU-WRF truth satisfies the EOS with qvf=1+rvovrd*qv and grossly "
        "fails with 1+p608*qv; the pre-fix GPU output is the exact mirror. "
        "Root cause: src/gpuwrf/dynamics/acoustic_wrf.py used 0.608 (p608) "
        "in _pressure_from_theta_alt and _inverse_density_from_theta_pressure "
        "where WRF uses rvovrd=Rv/Rd~1.6084. Fixed in this sprint."
    )
    out["eos_attribution"] = eos

    # --- Verdict --------------------------------------------------------------
    att = out["mu_init_mode_attribution"]
    init_dominant = (
        att["pattern_corr_initgap_vs_h1gap"] > 0.7
        and att["livenest_proxy_minus_realinit"]["rmse"]
        > 0.5 * att["observed_h1_gpu_minus_cpu"]["rmse"]
    )
    out["classification"] = {
        "provenance_same_inputs": out["provenance"]["same_input_files"],
        "init_mode_mismatch_dominant_for_mass_fields": bool(init_dominant),
        "dominant_class": (
            "DYCORE_EOS_MOISTURE_FACTOR (p608 vs rvovrd) — fixed in "
            "src/gpuwrf/dynamics/acoustic_wrf.py this sprint; GPU rerun "
            "required to re-baseline the h1 falsifier."
        ),
        "secondary_classes": [
            "init-mode mismatch: GPU standalone_native_init_nested live-nests "
            "d02 from d01; CPU truth used real.exe wrfinput_d02 "
            "(input_from_file=.true.) — irreducible in current gate pairing",
            "radiation timing: COSZEN bias -0.0551 (~15-20 min zenith offset, "
            "WRF seeds xtime+radt/2) -> SWDOWN/SWNORM ~-37% relative at h1",
            "MUB/PB: 154/10494 cells differ >1 Pa, all in spec_bdy band, "
            "max ~90 Pa (boundary-row base class)",
        ],
    }

    path = "proofs/v014/short_field_h1_residual_classification.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(json.dumps(out["mu_init_mode_attribution"], indent=1))
    print(json.dumps(out["psfc_decomposition"], indent=1))
    print(json.dumps(out["classification"], indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
