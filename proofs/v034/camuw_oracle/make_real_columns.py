#!/usr/bin/env python3
"""Real-structured CAM-UW oracle columns from the Swiss original CPU-WRF 24 h reference.

Source frames (READ-ONLY): <USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu/wrfout_d01_2023-01-15_{00,12}:00:00
(42x42, 44 levels, Thompson mp=8 -> WRF passes the dummy scalar slot for QNC = 0; QNICE real).
Inputs are WRF REAL (float32 arithmetic). RTHRATLW is not in wrfout -> synthetic LW forcing
(-1.5e-5 K/s, -4e-4 K/s in the cloud-top layer): labelled real-structured, not a WRF state.
Usage: make_real_columns.py OUT.txt
"""
import sys
import netCDF4
import numpy as np

SRC = "<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu/wrfout_d01_2023-01-15_{hh}:00:00"
F = np.float32
G, RD, CP = F(9.81), F(287.0), F(1004.5)


def pick(d):
    hfx = d["HFX"][0]
    qc = d["QCLOUD"][0].sum(0)
    qi = d["QICE"][0].sum(0)
    hgt = d["HGT"][0]
    picks = [np.unravel_index(np.argmax(hfx), hfx.shape), np.unravel_index(np.argmin(hfx), hfx.shape),
             np.unravel_index(np.argmax(qc), qc.shape), np.unravel_index(np.argmax(qi), qi.shape),
             np.unravel_index(np.argmax(hgt), hgt.shape), np.unravel_index(np.argmin(hgt), hgt.shape),
             (10, 31), (30, 12)]
    out = []
    for p in picks:
        if p not in out:
            out.append(p)
    return out


def column(d, j, i):
    v = lambda k: np.asarray(d[k][0], F)  # noqa: E731
    u3 = v("U")
    v3 = v("V")
    u = F(0.5) * (u3[:, j, i] + u3[:, j, i + 1])
    vv = F(0.5) * (v3[:, j, i] + v3[:, j + 1, i])
    th = v("T")[:, j, i] + F(300.0)
    p = v("P")[:, j, i] + v("PB")[:, j, i]
    zw = (v("PH")[:, j, i] + v("PHB")[:, j, i]) / G
    z = F(0.5) * (zw[:-1] + zw[1:])
    exn = (p / F(1.0e5)) ** (RD / CP)
    t = th * exn
    qv = v("QVAPOR")[:, j, i]
    qc = v("QCLOUD")[:, j, i]
    qi = v("QICE")[:, j, i]
    qni = v("QNICE")[:, j, i]
    cf = v("CLDFRA")[:, j, i]
    rho = p / (RD * t * (F(1.0) + F(0.608) * qv))
    fnm = np.asarray(d["FNM"][0], F)
    fnp = np.asarray(d["FNP"][0], F)
    kx = p.shape[0]
    p8w = np.empty(kx + 1, F)
    p8w[0] = F(d["PSFC"][0, j, i])
    p8w[1:kx] = fnm[1:] * p[1:] + fnp[1:] * p[:-1]
    p8w[kx] = max(F(2.0) * p[kx - 1] - p8w[kx - 1], F(10.0))
    lw = np.full(kx, F(-1.5e-5))
    for k in range(kx - 1):
        if cf[k] > 0.5 and cf[k + 1] <= 0.5:
            lw[k] = F(-4.0e-4)
    hfx = F(d["HFX"][0, j, i])
    qfx = F(d["QFX"][0, j, i])
    ust = F(d["UST"][0, j, i])
    ht = F(d["HGT"][0, j, i])
    rows = [[hfx, qfx, ust, ht], u, vv, th, p, t, exn, rho, qv, qc, qi, np.zeros(kx, F), qni, cf, lw,
            np.zeros(kx, F), z, p8w, zw]
    return rows, kx


def main(out):
    cols, names, kx = [], [], None
    for hh in ("00", "12"):
        d = netCDF4.Dataset(SRC.format(hh=hh))
        d.set_auto_mask(False)
        for (j, i) in pick(d):
            rows, kx = column(d, j, i)
            cols.append(rows)
            names.append(f"swiss_{hh}z_j{j}_i{i}")
        d.close()
    with open(out, "w") as fh:
        fh.write(f"{len(cols)} {kx} 3 60.0\n")
        for rows in cols:
            for r in rows:
                fh.write(" ".join(repr(float(x)) for x in np.atleast_1d(np.asarray(r, F))) + "\n")
    with open(out + ".names", "w") as fh:
        fh.write("\n".join(names) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
