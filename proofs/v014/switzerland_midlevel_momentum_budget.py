#!/usr/bin/env python
"""V0.14 Switzerland mid-level momentum budget — forecast-level (cumulative) lane.

Measures the 1-hour and 2-hour ACCUMULATED momentum forcing defect profile of the
GPU forecast (gpu_output_phys_tendf, reinit from CPU truth h36) against the CPU
truth, per level, interior depth-8 / depth-20.  Because the GPU run starts from
the IDENTICAL CPU h36 state, Δu(h37)/3600 s is the net per-level momentum forcing
bias (m/s^2) of the GPU model — the small cumulative defect that is invisible at
single-substep parity but drives the venting jet.

Also localizes the worst band horizontally (terrain/jet/2dx correlations) so the
defective term class can be identified.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

CPU = Path("/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu")
GPU = Path("/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable/gpu_output_phys_tendf")
OUT = Path(__file__).with_suffix(".json")

H = {36: "2023-01-16_12:00:00", 37: "2023-01-16_13:00:00", 38: "2023-01-16_14:00:00"}


def load(root: Path, hour: int, names: list[str]) -> dict[str, np.ndarray]:
    ds = Dataset(root / f"wrfout_d01_{H[hour]}")
    out = {n: np.asarray(ds.variables[n][0], dtype=np.float64) for n in names}
    ds.close()
    return out


def imask(ny: int, nx: int, depth: int) -> np.ndarray:
    m = np.zeros((ny, nx), dtype=bool)
    m[depth:-depth, depth:-depth] = True
    return m


def kprof(diff3d: np.ndarray, mask2d: np.ndarray) -> dict[str, list[float]]:
    # diff3d (nz, ny, nx) on the cropped A-grid
    mean = [float(np.mean(diff3d[k][mask2d])) for k in range(diff3d.shape[0])]
    rms = [float(np.sqrt(np.mean(diff3d[k][mask2d] ** 2))) for k in range(diff3d.shape[0])]
    return {"mean": mean, "rms": rms}


def main() -> int:
    names = ["U", "V", "W", "MU", "MUB", "PH", "PHB", "P", "PB", "T", "HGT", "MAPFAC_M", "F"]
    cpu36 = load(CPU, 36, names)
    cpu37 = load(CPU, 37, names)
    cpu38 = load(CPU, 38, names)
    gpu37 = load(GPU, 37, names)
    gpu38 = load(GPU, 38, names)

    nz, ny, nxp1 = cpu36["U"].shape
    nx = nxp1 - 1
    res: dict = {
        "schema": "v014_switzerland_midlevel_momentum_budget",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "anchor": "GPU reinit from CPU h36; gpu_output_phys_tendf h37/h38 vs CPU truth",
        "nz": nz,
    }

    # A-grid (unstagger) u/v for clean masking
    def ua(d):
        return 0.5 * (d["U"][:, :, :-1] + d["U"][:, :, 1:])

    def va(d):
        return 0.5 * (d["V"][:, :-1, :] + d["V"][:, 1:, :])

    m8 = imask(ny, nx, 8)
    m20 = imask(ny, nx, 20)

    for tag, g, c, dt_s in (("h37", gpu37, cpu37, 3600.0), ("h38", gpu38, cpu38, 7200.0)):
        du = ua(g) - ua(c)
        dv = va(g) - va(c)
        res[f"du_{tag}_d8"] = kprof(du, m8)
        res[f"du_{tag}_d20"] = kprof(du, m20)
        res[f"dv_{tag}_d8"] = kprof(dv, m8)
        res[f"dv_{tag}_d20"] = kprof(dv, m20)
        # implied net forcing bias profile, m/s^2 (cumulative/Δt)
        res[f"forcing_u_{tag}_d8_m_s2"] = [m / dt_s for m in res[f"du_{tag}_d8"]["mean"]]
        # theta / pressure drift profiles for context
        dth = g["T"] - c["T"]
        res[f"dtheta_{tag}_d8"] = kprof(dth, m8)
        res[f"dmu_{tag}_d8_mean"] = float(np.mean((g["MU"] - c["MU"])[m8]))

    # sanity: CPU h36->h37 own hourly du profile (scale reference)
    du_cpu = ua(cpu37) - ua(cpu36)
    res["du_cpu_h36_h37_d8"] = kprof(du_cpu, m8)

    # --- horizontal localization at the worst bands -------------------------
    du37 = ua(gpu37) - ua(cpu37)
    hgt = cpu36["HGT"]
    # band means (k chosen from flux-localizer bands)
    bands = {"k00_09": slice(0, 10), "k10_19": slice(10, 20), "k20_29": slice(20, 30), "k30_43": slice(30, nz)}
    loc = {}
    for bname, sl in bands.items():
        band = np.mean(du37[sl], axis=0)
        # terrain correlation (interior d8)
        bh = band[m8]
        hh = hgt[m8]
        corr_h = float(np.corrcoef(bh, hh)[0, 1])
        # jet correlation: |u| of CPU at band
        ju = np.mean(np.abs(ua(cpu37)[sl]), axis=0)[m8]
        corr_j = float(np.corrcoef(bh, ju)[0, 1])
        # 2dx roughness share of the GPU-CPU diff: high-pass via 5-point laplacian
        lap = np.abs(4 * band[1:-1, 1:-1] - band[:-2, 1:-1] - band[2:, 1:-1] - band[1:-1, :-2] - band[1:-1, 2:])
        loc[bname] = {
            "mean": float(np.mean(bh)),
            "rms": float(np.sqrt(np.mean(bh ** 2))),
            "corr_terrain": corr_h,
            "corr_abs_u_cpu": corr_j,
            "lap_rms": float(np.sqrt(np.mean(lap ** 2))),
        }
        # west->east profile (central third rows) of the band mean
        rows = slice(ny // 3, 2 * ny // 3)
        loc[bname]["west_east_profile_i0_40"] = [float(np.mean(band[rows, i])) for i in range(0, 40, 2)]
    res["du37_band_localization"] = loc

    # growth linearity: du(h38)/du(h37) per band (constant forcing -> ~2.0)
    du38 = ua(gpu38) - ua(cpu38)
    res["growth_ratio_band"] = {
        b: float(np.mean(np.mean(du38[sl], axis=0)[m8]) / (np.mean(np.mean(du37[sl], axis=0)[m8]) + 1e-30))
        for b, sl in bands.items()
    }

    OUT.write_text(json.dumps(res, indent=1))
    # console digest
    print("per-level du mean (GPU-CPU) h37 d8 [m/s]:")
    for k in range(nz):
        print(f"  k{k:02d} du={res['du_h37_d8']['mean'][k]:+.4f} rms={res['du_h37_d8']['rms'][k]:.4f} "
              f"dv={res['dv_h37_d8']['mean'][k]:+.4f} | du_h38={res['du_h38_d8']['mean'][k]:+.4f} "
              f"| cpu_own_du={res['du_cpu_h36_h37_d8']['mean'][k]:+.4f}")
    print(json.dumps(res["du37_band_localization"], indent=1))
    print("growth ratios:", res["growth_ratio_band"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
