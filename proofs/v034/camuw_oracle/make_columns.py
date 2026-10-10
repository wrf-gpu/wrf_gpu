#!/usr/bin/env python3
"""Generate CAM-UW oracle input columns (all values float32-exact, WRF REAL).

Regimes cover every caleddy/zisocl branch class: surface-based convective CL (bflxs>0), stable
STL surface (bflxs<0), stratocumulus top (ql>1e-5 jump + LW cooling -> evhc/radf and SRCL),
elevated CL above a stable surface layer, two separated CLs (merging tests), ice cloud (latsub,
qi), elevated terrain (phis/ht), weak wind, and a deep moist neutral column.
Usage: make_columns.py OUT.txt
"""
import sys
import numpy as np

KX = 30
DT = 60.0
NSTEP = 3
G, RD, CP, RV, P0 = 9.81, 287.0, 1004.5, 461.6, 1.0e5
EP1 = RV / RD - 1.0


def f32(x):
    return np.asarray(x, dtype=np.float32)


def column(name, *, psfc, ht, theta_fn, q_fn, u_fn, v_fn, hfx, qfx, ust, qc_fn=None, qi_fn=None,
           cf_fn=None, lw_fn=None, wsed=0.0, nc=1.0e8, ni=1.0e5, ztop=14000.0):
    zi = ztop * (np.arange(KX + 1) / KX) ** 1.6          # height above ground, interfaces
    zm = 0.5 * (zi[1:] + zi[:-1])
    th = np.array([theta_fn(z) for z in zm])
    qv = np.array([q_fn(z) for z in zm])
    qc = np.array([qc_fn(z) if qc_fn else 0.0 for z in zm])
    qi = np.array([qi_fn(z) if qi_fn else 0.0 for z in zm])
    cf = np.array([cf_fn(z) if cf_fn else (1.0 if (qc[k] + qi[k]) > 0 else 0.0) for k, z in enumerate(zm)])
    lw = np.array([lw_fn(z) if lw_fn else -2.0e-5 for z in zm])
    pi = np.empty(KX + 1)
    pm = np.empty(KX)
    pi[0] = psfc
    for k in range(KX):
        pe = pi[k]
        for _ in range(3):
            pm_k = pe
            t = th[k] * (pm_k / P0) ** (RD / CP)
            tv = t * (1 + EP1 * qv[k] - qc[k] - qi[k])
            pi[k + 1] = pi[k] * np.exp(-G * (zi[k + 1] - zi[k]) / (RD * tv))
            pe = 0.5 * (pi[k] + pi[k + 1])
        pm[k] = pe
    exn = (pm / P0) ** (RD / CP)
    t = th * exn
    rho = pm / (RD * t * (1 + EP1 * qv - qc - qi))
    u = np.array([u_fn(z) for z in zm])
    v = np.array([v_fn(z) for z in zm])
    rows = [
        [hfx, qfx, ust, ht], u, v, th, pm, t, exn, rho, qv, qc, qi,
        np.full(KX, nc), np.full(KX, ni), cf, lw, np.full(KX, wsed), zm + ht, pi, zi + ht,
    ]
    return name, [f32(r) for r in rows]


def ml_theta(theta0, zml, lml, lft):
    return lambda z: theta0 + lml * min(z, zml) + lft * max(z - zml, 0.0)


def build():
    cols = []
    cols.append(column("convective_clear", psfc=100500.0, ht=0.0, theta_fn=ml_theta(301.0, 1200.0, -0.0005, 0.0045),
                       q_fn=lambda z: max(0.015 * np.exp(-z / 2500.0), 2e-5), u_fn=lambda z: 3.0 + 0.002 * z,
                       v_fn=lambda z: -1.0 + 0.0008 * z, hfx=320.0, qfx=1.2e-4, ust=0.42))
    cols.append(column("stable_nocturnal", psfc=101300.0, ht=0.0, theta_fn=ml_theta(284.0, 250.0, 0.012, 0.004),
                       q_fn=lambda z: max(0.006 * np.exp(-z / 1800.0), 2e-5), u_fn=lambda z: 2.0 + 8.0 * (1 - np.exp(-z / 300.0)),
                       v_fn=lambda z: 1.0 + 0.0005 * z, hfx=-35.0, qfx=-5.0e-6, ust=0.18))
    zinv = 1000.0
    cols.append(column("stratocumulus", psfc=101800.0, ht=0.0,
                       theta_fn=lambda z: 289.0 + (0.0 if z < zinv else 9.0 + 0.004 * (z - zinv)),
                       q_fn=lambda z: 0.0095 if z < zinv else max(0.004 * np.exp(-(z - zinv) / 2000.0), 2e-5),
                       qc_fn=lambda z: 4.0e-4 * (z - 600.0) / 400.0 if 600.0 < z < zinv else 0.0,
                       lw_fn=lambda z: -4.0e-4 if 800.0 < z < zinv else -1.0e-5,
                       u_fn=lambda z: 6.0 + 0.001 * z, v_fn=lambda z: 2.0, hfx=12.0, qfx=3.0e-5, ust=0.30, wsed=0.01))
    cols.append(column("elevated_cl_over_stable", psfc=99000.0, ht=0.0,
                       theta_fn=lambda z: 290.0 + 0.010 * min(z, 300.0) + (0.0 if z < 1600.0 else 0.005 * (z - 1600.0)) - 0.0002 * max(min(z, 1600.0) - 300.0, 0.0),
                       q_fn=lambda z: max(0.008 * np.exp(-z / 2200.0), 2e-5), u_fn=lambda z: 4.0 + 0.003 * z,
                       v_fn=lambda z: 0.5, hfx=-20.0, qfx=0.0, ust=0.25))
    cols.append(column("two_cls_merge", psfc=100800.0, ht=0.0,
                       theta_fn=lambda z: 297.0 + (-0.0008 * z if z < 700.0 else (-0.56 + 0.006 * (z - 700.0) if z < 1000.0 else (1.24 - 0.0004 * (z - 1000.0) if z < 2200.0 else 0.76 + 0.005 * (z - 2200.0)))),
                       q_fn=lambda z: max(0.013 * np.exp(-z / 2400.0), 2e-5), u_fn=lambda z: 5.0 + 0.0015 * z,
                       v_fn=lambda z: 2.0 - 0.0005 * z, hfx=180.0, qfx=8.0e-5, ust=0.35))
    cols.append(column("ice_cloud_cold", psfc=95000.0, ht=600.0, theta_fn=ml_theta(268.0, 500.0, 0.001, 0.006),
                       q_fn=lambda z: max(0.0025 * np.exp(-z / 2000.0), 1e-5),
                       qi_fn=lambda z: 6.0e-5 if 1500.0 < z < 3500.0 else 0.0,
                       qc_fn=lambda z: 2.0e-5 if 1500.0 < z < 2000.0 else 0.0,
                       lw_fn=lambda z: -6.0e-5 if 1500.0 < z < 3500.0 else -1.5e-5,
                       u_fn=lambda z: 10.0 + 0.002 * z, v_fn=lambda z: -3.0, hfx=40.0, qfx=1.0e-5, ust=0.5, ni=5.0e5))
    cols.append(column("high_terrain", psfc=82000.0, ht=1750.0, theta_fn=ml_theta(306.0, 1800.0, 0.0, 0.004),
                       q_fn=lambda z: max(0.007 * np.exp(-z / 2500.0), 2e-5), u_fn=lambda z: 1.5 + 0.004 * z,
                       v_fn=lambda z: 2.0 + 0.001 * z, hfx=420.0, qfx=6.0e-5, ust=0.55))
    cols.append(column("weak_wind_neutral_moist", psfc=100900.0, ht=120.0, theta_fn=ml_theta(295.0, 600.0, 0.001, 0.0035),
                       q_fn=lambda z: max(0.012 * np.exp(-z / 2600.0), 2e-5),
                       qc_fn=lambda z: 1.5e-5 if 1800.0 < z < 2300.0 else 0.0,
                       cf_fn=lambda z: 0.4 if 1800.0 < z < 2300.0 else 0.0,
                       u_fn=lambda z: 0.6 + 0.0002 * z, v_fn=lambda z: 0.3, hfx=5.0, qfx=2.0e-5, ust=0.05))
    cols.append(column("windy_stable_stl", psfc=101000.0, ht=0.0, theta_fn=ml_theta(288.0, 600.0, 0.003, 0.004),
                       q_fn=lambda z: max(0.007 * np.exp(-z / 2000.0), 2e-5), u_fn=lambda z: 1.0 + 16.0 * min(z, 600.0) / 600.0,
                       v_fn=lambda z: 0.5 + 0.004 * min(z, 600.0), hfx=-15.0, qfx=0.0, ust=0.40))
    cols.append(column("decoupled_sc", psfc=101500.0, ht=0.0,
                       theta_fn=lambda z: 287.0 + 0.006 * min(z, 400.0) + (0.0 if z < 1200.0 else 8.0 + 0.004 * (z - 1200.0)),
                       q_fn=lambda z: 0.0085 if z < 1200.0 else max(0.003 * np.exp(-(z - 1200.0) / 2000.0), 2e-5),
                       qc_fn=lambda z: 3.0e-4 if 850.0 < z < 1200.0 else 0.0,
                       lw_fn=lambda z: -6.0e-4 if 1000.0 < z < 1200.0 else -1.0e-5,
                       u_fn=lambda z: 7.0, v_fn=lambda z: 1.0 + 0.0002 * z, hfx=-8.0, qfx=0.0, ust=0.20, wsed=0.02))
    cols.append(column("fog_stable", psfc=101600.0, ht=30.0, theta_fn=ml_theta(281.0, 200.0, 0.008, 0.005),
                       q_fn=lambda z: 0.0062 if z < 200.0 else max(0.004 * np.exp(-z / 2000.0), 2e-5),
                       qc_fn=lambda z: 2.0e-4 if z < 150.0 else 0.0,
                       lw_fn=lambda z: -3.0e-4 if z < 150.0 else -1.0e-5,
                       u_fn=lambda z: 1.5 + 0.004 * z, v_fn=lambda z: 0.2, hfx=-12.0, qfx=-1.0e-6, ust=0.10))
    # found by searching the JAX port for a zisocl upward merge, then confirmed in the oracle
    a1, z1, d, b, a2, z2 = -0.0022148554776390884, 736.6718769884715, 393.25801993043353, 0.0028857384371180845, -0.0011733192309832186, 2033.5248091305693

    def th_mu(z):
        if z < z1:
            return 300 + a1 * z
        t1 = 300 + a1 * z1
        if z < z1 + d:
            return t1 + b * (z - z1)
        t2 = t1 + b * d
        if z < z2:
            return t2 + a2 * (z - z1 - d)
        return t2 + a2 * (z2 - z1 - d) + 0.005 * (z - z2)

    cols.append(column("merge_up_two_cls", psfc=100600.0, ht=0.0, theta_fn=th_mu,
                       q_fn=lambda z: max(0.010 * np.exp(-z / 2500.0), 2e-5), u_fn=lambda z: 4.0 + 0.001 * z,
                       v_fn=lambda z: 1.0, hfx=160.7564816181483, qfx=8e-5, ust=0.35))
    # found by search: surface-based SRCL (cloud only in the lowest layer, bflxs <= 0)
    top, qcv, qv0, qdrop, lwv = 3.6112576322956293, 7.895309744072663e-05, 0.008386404882559054, 0.0016703664093845594, -0.0008009355122375923
    cols.append(column("fog_surface_srcl", psfc=101700.0, ht=10.0, theta_fn=ml_theta(279.0, 300.0, 0.0216545853388077, 0.005),
                       q_fn=lambda z: qv0 if z < 6.0 else max(qv0 - qdrop, 2e-5),
                       qc_fn=lambda z: qcv if z < top else 0.0, lw_fn=lambda z: lwv if z < top else -1.0e-5,
                       u_fn=lambda z: 1.0 + 0.002 * z, v_fn=lambda z: 0.3, hfx=-13.964703316005904, qfx=-5e-7,
                       ust=0.29289326782041025))
    return cols


def main(out):
    cols = build()
    with open(out, "w") as fh:
        fh.write(f"{len(cols)} {KX} {NSTEP} {DT!r}\n")
        for name, rows in cols:
            for r in rows:
                fh.write(" ".join(repr(float(x)) for x in np.atleast_1d(r)) + "\n")
    with open(out + ".names", "w") as fh:
        fh.write("\n".join(n for n, _ in cols) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
