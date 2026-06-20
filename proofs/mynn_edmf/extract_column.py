"""Extract a representative d03 daytime-land column from the equiv-T2 WRF case.

CPU-only. Reads one wrfout_d03 12z file (the exact case the manager's equiv-T2
diagnostic used), selects a representative convective land column, and writes the
WRF-faithful MYNN-EDMF driver inputs (th1, p1, rho1, dz1, ex1, tk1, u1, v1, w1,
qv, qc, qi, qke, surface hfx/qfx/ust/pblh/tsk/psfc) to a JSON the Fortran oracle
and the JAX port both consume.

The derived surface fluxes (flt/flq/flqv/fltv) follow module_bl_mynnedmf.F:869-876.
Run with: JAX_PLATFORMS=cpu python extract_column.py
"""

import json
from pathlib import Path

import numpy as np
import netCDF4 as nc

# WRF model constants (module_model_constants.F)
R_D = 287.0
CP = 7.0 * R_D / 2.0
R_V = 461.6
P608 = R_V / R_D - 1.0
P1000MB = 100000.0
RCP = R_D / CP
G = 9.81

WRFOUT = (
    "<DATA_ROOT>/canairy_meteo/runs/wrf_l3/"
    "20260521_18z_l3_24h_20260522T133443Z/wrfout_d03_2026-05-22_12:00:00"
)
OUT = str(Path(__file__).resolve().parent / "column_d03_12z.json")


def main():
    d = nc.Dataset(WRFOUT)
    g = lambda v: np.array(d.variables[v][0])

    XLAND = g("XLAND")
    HFX = g("HFX")
    QFX = g("QFX")
    LH = g("LH")
    PBLH = g("PBLH")
    UST = g("UST")
    TSK = g("TSK")
    PSFC = g("PSFC")

    land = XLAND < 1.5
    ny, nx = XLAND.shape
    jj, ii = np.where(land)
    interior = (
        (jj > 5) & (jj < ny - 6) & (ii > 5) & (ii < nx - 6)
        & (HFX[jj, ii] > 200) & (QFX[jj, ii] > 0)
    )
    cj, ci = jj[interior], ii[interior]
    med = np.median(HFX[cj, ci])
    k = int(np.argmin(np.abs(HFX[cj, ci] - med)))
    J, I = int(cj[k]), int(ci[k])

    # de-stagger U,V to mass points; W keep full (kte+1)
    U = g("U")  # (44,75,94) staggered x
    V = g("V")  # (44,76,93) staggered y
    W = g("W")  # (45,75,93) staggered z (interfaces)
    T = g("T")  # perturbation theta (theta - 300)
    QV = g("QVAPOR")
    QC = g("QCLOUD")
    QI = g("QICE")
    P = g("P")
    PB = g("PB")
    PH = g("PH")
    PHB = g("PHB")
    QKE = g("QKE")

    u_m = 0.5 * (U[:, J, I] + U[:, J, I + 1])
    v_m = 0.5 * (V[:, J, I] + V[:, J + 1, I])
    # W on mass levels: average the two bounding interfaces
    w_full = W[:, J, I]  # length 45 (interfaces)
    w_m = 0.5 * (w_full[:-1] + w_full[1:])  # length 44

    th = T[:, J, I] + 300.0  # full potential temperature (theta)
    pres = P[:, J, I] + PB[:, J, I]  # full pressure (Pa)
    qv = QV[:, J, I].astype(float)
    qc = QC[:, J, I].astype(float)
    qi = QI[:, J, I].astype(float)
    qke = QKE[:, J, I].astype(float)

    exner = (pres / P1000MB) ** RCP
    tk = th * exner
    rho = pres / (R_D * tk * (1.0 + P608 * qv / (1.0 + qv)))  # moist density approx

    # layer thickness dz from geopotential interfaces
    geo = (PH[:, J, I] + PHB[:, J, I]) / G  # geopotential height at interfaces (45)
    dz = np.diff(geo)  # (44)

    nz = th.shape[0]

    # surface
    hfx = float(HFX[J, I])
    qfx = float(QFX[J, I])
    ust = float(UST[J, I])
    pblh = float(PBLH[J, I])
    tsk = float(TSK[J, I])
    psfc = float(PSFC[J, I])

    # apply driver flux limiters (module_bl_mynnedmf_driver.F:381-397)
    hfx = min(max(hfx, -600.0), 1200.0)
    qfx = min(max(qfx, -3e-4), 9e-4)

    # WRF-faithful flux derivation (module_bl_mynnedmf.F:859-876)
    sqv0 = qv[0] / (1.0 + qv[0])  # specific humidity at kts
    cpm = CP * (1.0 + 0.84 * max(sqv0, 1e-8))
    th_sfc = tsk / exner[0]
    rho0 = float(rho[0])
    flqv = qfx / rho0
    flqc = 0.0
    flq = flqv + flqc
    flt = hfx / (rho0 * cpm) - 2.5e6 / CP * flqc / exner[0]
    fltv = flt + flqv * P608 * th_sfc

    wspd = float(max(np.sqrt(u_m[0] ** 2 + v_m[0] ** 2), 0.1))

    col = {
        "meta": {
            "wrfout": WRFOUT,
            "j": J,
            "i": I,
            "valid_time": "2026-05-22_12:00:00",
            "lead_h": 18,
            "nz": int(nz),
            "HFX_land_mean_in_case": float(HFX[land].mean()),
            "QFX_land_mean_in_case": float(QFX[land].mean()),
            "LH_land_mean_in_case": float(LH[land].mean()),
            "selected_HFX": hfx,
            "selected_QFX": qfx,
            "selected_LH": float(LH[J, I]),
            "selected_PBLH": pblh,
        },
        "profiles": {
            "u": u_m.tolist(),
            "v": v_m.tolist(),
            "w": w_m.tolist(),
            "th": th.tolist(),
            "p": pres.tolist(),
            "exner": exner.tolist(),
            "tk": tk.tolist(),
            "rho": rho.tolist(),
            "dz": dz.tolist(),
            "qv": qv.tolist(),
            "qc": qc.tolist(),
            "qi": qi.tolist(),
            "qke": qke.tolist(),
        },
        "surface": {
            "hfx": hfx,
            "qfx": qfx,
            "ust": ust,
            "pblh": pblh,
            "tsk": tsk,
            "psfc": psfc,
            "wspd": wspd,
            "xland": 1.0,
            "flt": flt,
            "flq": flq,
            "flqv": flqv,
            "fltv": fltv,
            "th_sfc": th_sfc,
            "cpm": cpm,
            "dx": 1000.0,
        },
        "config": {
            "delt": 3.0,
            "bl_mynn_edmf": 1,
            "bl_mynn_edmf_mom": 0,
            "bl_mynn_edmf_tke": 0,
            "bl_mynn_mixscalars": 1,
            "bl_mynn_cloudmix": 1,
            "bl_mynn_mixqt": 0,
            "env_subs": False,
            "bl_mynn_edmf_dd": 0,
        },
    }
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o

    with open(OUT, "w") as fh:
        json.dump(_clean(col), fh, indent=2)
    print(f"wrote {OUT}")
    print(f"col j={J} i={I} nz={nz} HFX={hfx:.1f} QFX={qfx:.3e} PBLH={pblh:.1f}")
    print(f"flqv={flqv:.4e} flt={flt:.5f} fltv={fltv:.5f} wspd={wspd:.3f} ust={ust:.4f}")
    print(f"qv[0..4]={qv[:5]}")
    print(f"th[0..4]={th[:5]}")


if __name__ == "__main__":
    main()
