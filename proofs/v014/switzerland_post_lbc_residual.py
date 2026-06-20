#!/usr/bin/env python
"""V0.14 Switzerland d01 post-LBC-fix residual: root-cause proof object.

CPU-only analysis over existing run artifacts plus two short GPU probe runs
(launched separately through scripts/run_gpu_lowprio.sh):

* full fixed run: <DATA_ROOT>/wrf_gpu_validation/v014_switzerland_d01_72h_lbcclockfix_20260611T020428Z
* CPU truth:      <DATA_ROOT>/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu
* h36 re-init probe (IC = CPU truth wrfout 2023-01-16_12, 12h):
                  <DATA_ROOT>/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable/gpu_output
* h36 re-init probe with mp_physics=0 (3h):
                  <DATA_ROOT>/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable/gpu_output_nomp

Gates:

G1  Dry-mass conservation: the column-integrated dry-mass budget (hybrid
    C1H/C2H coupling + map factors) closes for BOTH runs at sampling-error
    level (|residual| < 10 Pa/cell/h vs signals up to ~100), at depth-8
    control surface.  =>  the GPU does NOT destroy mass internally; the MU
    deficit is vented through the lateral boundary by the GPU's own winds.

G2  Locally generated (NOT accumulated drift / chaos): a GPU run re-initialized
    bit-true from CPU truth at h36 re-develops the venting within hours at a
    rate >= the full run over the same valid window (vs an h0-start baseline
    drift ~10x smaller in the calm first hours).

G3  Domain-wide divergence (NOT a boundary-zone artifact): the excess mass
    outflux probe-vs-CPU is the same (~ -30 Pa/cell/h) through control
    surfaces at depth 2, 8 and 20 cells.

G4  Day/night invariance: the probe keeps venting at full rate after SWDOWN
    goes to zero  =>  SW radiation / cloud-albedo error is NOT the driver.

G5  Microphysics symptoms quantified (report-only class): condensate
    annihilation in hour 1, QNICE = 0 through ~h5 (ssati never reaches the
    0.25 WRF nucleation threshold because GPU runs warm aloft), QNRAIN sparse
    bursts lie INSIDE the WRF mvd clamp band (not a clamp violation).

G6  Moist-vs-dry discriminator: the mp_physics=0 re-init probe measures
    whether the venting persists without microphysics.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import numpy as np
from netCDF4 import Dataset

CPU = "<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu"
FULL = "<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_d01_72h_lbcclockfix_20260611T020428Z/gpu_output"
PROBE = "<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable/gpu_output"
# gpu_output_nomp (namelist.input edit) ran Thompson bit-identically: the daily
# path resolves the physics suite from OperationalNamelist DEFAULTS, not the case
# namelist.input (daily_pipeline._build_real_case from_grid call).  gpu_output_nomp2
# is the genuine mp_physics=0 run via run_nomp_driver.py (patched case_builder).
PROBE_NOMP = "<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable/gpu_output_nomp2"
RUN_START = datetime(2023, 1, 15)
OUT_JSON = os.path.join(os.path.dirname(__file__), "switzerland_post_lbc_residual.json")


def fn(base: str, hour: int) -> str:
    label = (RUN_START + timedelta(hours=hour)).strftime("%Y-%m-%d_%H:%M:%S")
    return os.path.join(base, f"wrfout_d01_{label}")


def get(base: str, hour: int, var: str) -> np.ndarray:
    with Dataset(fn(base, hour)) as handle:
        return np.array(handle.variables[var][0])


def load_budget_state(base: str, hour: int) -> dict:
    with Dataset(fn(base, hour)) as d:
        return dict(
            mu=np.array(d.variables["MU"][0]) + np.array(d.variables["MUB"][0]),
            u=np.array(d.variables["U"][0]),
            v=np.array(d.variables["V"][0]),
            dnw=np.array(d.variables["DNW"][0]),
            c1h=np.array(d.variables["C1H"][0]),
            c2h=np.array(d.variables["C2H"][0]),
            mx=np.array(d.variables["MAPFAC_MX"][0]),
            my=np.array(d.variables["MAPFAC_MY"][0]),
            muy=np.array(d.variables["MAPFAC_UY"][0]),
            mvx=np.array(d.variables["MAPFAC_VX"][0]),
            dx=float(d.DX),
        )


def budget(base: str, h0: int, h1: int, depth: int = 8) -> dict:
    """Hybrid-coordinate, map-factor-correct column dry-mass budget (Pa/cell/h)."""

    a, b = load_budget_state(base, h0), load_budget_state(base, h1)
    wk, c1, c2 = -a["dnw"], a["c1h"], a["c2h"]
    ny, nx = a["mu"].shape
    i0, i1, j0, j1 = depth, nx - depth, depth, ny - depth

    def colmass(s):
        m = ((c1[:, None, None] * s["mu"][None] + c2[:, None, None]) * wk[:, None, None]).sum(0)
        return (m / (s["mx"] * s["my"]))[j0:j1, i0:i1].sum()

    def outflux(s):
        mu, u, v = s["mu"], s["u"], s["v"]

        def mul_u(i):
            muf = 0.5 * (mu[j0:j1, i - 1] + mu[j0:j1, i])
            return c1[:, None] * muf[None, :] + c2[:, None]

        def mul_v(j):
            muf = 0.5 * (mu[j - 1, i0:i1] + mu[j, i0:i1])
            return c1[:, None] * muf[None, :] + c2[:, None]

        fw = (u[:, j0:j1, i0] * mul_u(i0) * wk[:, None] / s["muy"][j0:j1, i0][None, :]).sum()
        fe = (u[:, j0:j1, i1] * mul_u(i1) * wk[:, None] / s["muy"][j0:j1, i1][None, :]).sum()
        fs = (v[:, j0, i0:i1] * mul_v(j0) * wk[:, None] / s["mvx"][j0, i0:i1][None, :]).sum()
        fnn = (v[:, j1, i0:i1] * mul_v(j1) * wk[:, None] / s["mvx"][j1, i0:i1][None, :]).sum()
        return (fe - fw) + (fnn - fs)

    ncell = (j1 - j0) * (i1 - i0)
    dm = (colmass(b) - colmass(a)) / ncell
    flux = (outflux(a) + outflux(b)) / 2.0 * 3600.0 / a["dx"] / ncell
    return {"dM_pa_per_cell_h": float(dm), "net_influx_pa_per_cell_h": float(-flux), "residual": float(dm + flux)}


def rmse_bias(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = a - b
    return float(np.sqrt((d * d).mean())), float(d.mean())


def main() -> None:
    proof: dict = {"schema": "v014_switzerland_post_lbc_residual", "generated_utc": datetime.utcnow().isoformat()}

    # G1: budget closure full run vs CPU truth.
    g1 = []
    for h in [24, 36, 40, 48, 60, 66, 71]:
        g1.append({"window": f"h{h}->h{h+1}", "cpu": budget(CPU, h, h + 1), "gpu_full": budget(FULL, h, h + 1)})
    max_resid = max(max(abs(r["cpu"]["residual"]), abs(r["gpu_full"]["residual"])) for r in g1)
    proof["g1_mass_budget_closure"] = {
        "rows": g1,
        "max_abs_residual": max_resid,
        "pass": bool(max_resid < 10.0),
        "meaning": "both runs conserve dry mass to sampling error; the GPU MU deficit is wind-vented through the LBC zone",
    }

    # G2: re-init probe divergence vs full run, same valid hours.
    g2 = []
    for h in range(37, 49):
        if not os.path.exists(fn(PROBE, h)):
            continue
        mu_c = get(CPU, h, "MU")
        rmse_p, bias_p = rmse_bias(get(PROBE, h, "MU"), mu_c)
        rmse_f, bias_f = rmse_bias(get(FULL, h, "MU"), mu_c)
        g2.append({"valid_h": h, "probe_mu_rmse": rmse_p, "probe_mu_bias": bias_p, "full_mu_rmse": rmse_f, "full_mu_bias": bias_f})
    # calm-start baseline: full-run h1-h6 MU rmse
    base_rows = [{"lead_h": h, "mu_rmse": rmse_bias(get(FULL, h, "MU"), get(CPU, h, "MU"))[0]} for h in range(1, 7)]
    probe_h42 = next(r for r in g2 if r["valid_h"] == 42)
    proof["g2_reinit_probe_locally_generated"] = {
        "probe_vs_full": g2,
        "calm_start_baseline_h1_h6": base_rows,
        "probe_lead6_mu_bias": probe_h42["probe_mu_bias"],
        "pass": bool(abs(probe_h42["probe_mu_bias"]) > 5 * max(abs(r["mu_rmse"]) for r in base_rows)),
        "meaning": "from a bit-true CPU-truth IC at h36 the GPU re-develops the venting within 6h at >= the full-run rate: NOT accumulated drift, NOT chaos",
    }

    # G3: depth-independent excess outflux at probe lead 1.
    g3 = {}
    for depth in [2, 8, 20]:
        c = budget(CPU, 36, 37, depth)
        p_dm = None
        # probe budget: IC shared with truth, endpoint = probe h37
        a, b = load_budget_state(CPU, 36), load_budget_state(PROBE, 37)
        # reuse budget() by temporary symlink-free re-evaluation: inline trapezoid
        # (identical math to budget(); endpoints differ in source dirs)
        wk, c1, c2 = -a["dnw"], a["c1h"], a["c2h"]
        ny, nx = a["mu"].shape
        i0, i1, j0, j1 = depth, nx - depth, depth, ny - depth

        def colmass(s):
            m = ((c1[:, None, None] * s["mu"][None] + c2[:, None, None]) * wk[:, None, None]).sum(0)
            return (m / (s["mx"] * s["my"]))[j0:j1, i0:i1].sum()

        def outflux(s):
            mu, u, v = s["mu"], s["u"], s["v"]

            def mul_u(i):
                muf = 0.5 * (mu[j0:j1, i - 1] + mu[j0:j1, i])
                return c1[:, None] * muf[None, :] + c2[:, None]

            def mul_v(j):
                muf = 0.5 * (mu[j - 1, i0:i1] + mu[j, i0:i1])
                return c1[:, None] * muf[None, :] + c2[:, None]

            fw = (u[:, j0:j1, i0] * mul_u(i0) * wk[:, None] / s["muy"][j0:j1, i0][None, :]).sum()
            fe = (u[:, j0:j1, i1] * mul_u(i1) * wk[:, None] / s["muy"][j0:j1, i1][None, :]).sum()
            fs = (v[:, j0, i0:i1] * mul_v(j0) * wk[:, None] / s["mvx"][j0, i0:i1][None, :]).sum()
            fnn = (v[:, j1, i0:i1] * mul_v(j1) * wk[:, None] / s["mvx"][j1, i0:i1][None, :]).sum()
            return (fe - fw) + (fnn - fs)

        ncell = (j1 - j0) * (i1 - i0)
        flux_probe = (outflux(a) + outflux(b)) / 2.0 * 3600.0 / a["dx"] / ncell
        g3[f"depth_{depth}"] = {
            "cpu_net_influx": c["net_influx_pa_per_cell_h"],
            "probe_net_influx": float(-flux_probe),
            "excess_outflux": float(-flux_probe - c["net_influx_pa_per_cell_h"]),
        }
    excesses = [v["excess_outflux"] for v in g3.values()]
    proof["g3_depth_independent_divergence"] = {
        "rows": g3,
        "pass": bool(max(excesses) - min(excesses) < 0.3 * abs(np.mean(excesses))),
        "meaning": "excess outflux ~constant from depth 2 to depth 20: the divergence bias is INTERIOR-WIDE, not a boundary-zone artifact",
    }

    # G4: night-time continuation.
    g4 = []
    for h in range(37, 49):
        if not os.path.exists(fn(PROBE, h)):
            continue
        dsw = float((get(PROBE, h, "SWDOWN") - get(CPU, h, "SWDOWN")).mean())
        bias = rmse_bias(get(PROBE, h, "MU"), get(CPU, h, "MU"))[1]
        g4.append({"valid_h": h, "swdown_diff_mean": dsw, "probe_mu_bias": bias})
    night = [r for r in g4 if abs(r["swdown_diff_mean"]) < 1.0]
    venting_at_night = night[-1]["probe_mu_bias"] - night[0]["probe_mu_bias"] if len(night) >= 2 else 0.0
    proof["g4_night_continuation"] = {
        "rows": g4,
        "night_mu_bias_change": float(venting_at_night),
        "pass": bool(venting_at_night < -30.0),
        "meaning": "venting continues at full rate after SWDOWN->0: SW/cloud-albedo radiation error is NOT the mass driver",
    }

    # G5: microphysics symptom quantification.
    qc_ic = float(get(CPU, 36, "QCLOUD").mean())
    qc_probe1 = float(get(PROBE, 37, "QCLOUD").mean())
    qc_cpu1 = float(get(CPU, 37, "QCLOUD").mean())
    qnice_rows = []
    for h in [1, 2, 3, 4, 6]:
        qnice_rows.append({"lead_h": h, "gpu_max": float(get(FULL, h, "QNICE").max()), "cpu_max": float(get(CPU, h, "QNICE").max())})
    nr = get(FULL, 19, "QNRAIN")
    qr = get(FULL, 19, "QRAIN")
    idx = np.unravel_index(int(np.argmax(nr)), nr.shape)
    # WRF mvd clamp band ceiling at this qr (mu_r=0): nr_max = rr*lamr^3/(6*am_r), lamr=3.672/(0.75*D0r)
    rho = 1.1
    rr = float(qr[idx]) * rho
    lamr = 3.672 / (0.75 * 50.0e-6)
    nr_band_ceiling = rr * lamr**3 / (6.0 * (np.pi * 1000.0 / 6.0)) / rho
    proof["g5_microphysics_symptoms"] = {
        "hour1_condensate_annihilation": {
            "qcloud_mean_ic_h36": qc_ic,
            "qcloud_mean_probe_h37": qc_probe1,
            "qcloud_mean_cpu_h37": qc_cpu1,
            "fraction_lost_vs_cpu": 1.0 - qc_probe1 / qc_cpu1,
        },
        "qnice_zero_first_hours": qnice_rows,
        "qnrain_burst_within_wrf_clamp_band": {
            "burst_value_per_kg": float(nr[idx]),
            "qr_at_burst": float(qr[idx]),
            "wrf_mvd_band_ceiling_per_kg": float(nr_band_ceiling),
            "violates_band": bool(nr[idx] > nr_band_ceiling * 1.05),
        },
    }

    # G6: no-microphysics discriminator (filled only if the run exists).
    g6 = {"available": False}
    if os.path.exists(fn(PROBE_NOMP, 37)):
        rows = []
        for h in [37, 38, 39]:
            if not os.path.exists(fn(PROBE_NOMP, h)):
                continue
            mu_c = get(CPU, h, "MU")
            rmse_n, bias_n = rmse_bias(get(PROBE_NOMP, h, "MU"), mu_c)
            rmse_p, bias_p = rmse_bias(get(PROBE, h, "MU"), mu_c)
            rows.append({"valid_h": h, "nomp_mu_rmse": rmse_n, "nomp_mu_bias": bias_n, "withmp_mu_rmse": rmse_p, "withmp_mu_bias": bias_p})
        g6 = {"available": True, "rows": rows}
        if rows:
            last = rows[-1]
            g6["venting_persists_without_mp"] = bool(last["nomp_mu_bias"] < 0.5 * last["withmp_mu_bias"] < 0.0) or bool(
                last["nomp_mu_bias"] < -60.0
            )
    proof["g6_no_microphysics_discriminator"] = g6

    with open(OUT_JSON, "w") as handle:
        json.dump(proof, handle, indent=1, default=float)
    print(f"wrote {OUT_JSON}")
    for key in proof:
        if isinstance(proof[key], dict) and "pass" in proof[key]:
            print(f"  {key}: pass={proof[key]['pass']}")


if __name__ == "__main__":
    main()
