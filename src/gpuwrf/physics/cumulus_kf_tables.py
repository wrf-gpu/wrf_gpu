"""Kain-Fritsch-eta lookup tables (faithful port of KF_LUTAB in WRF
module_cu_kfeta.F).

The tables map saturation-equivalent potential temperature -> (temperature,
saturation mixing ratio) on a (KFNT x KFNP) grid, plus a log lookup (ALU) and
the base saturation theta_e curve (THE0K). They depend ONLY on the four SVP
constants, so they are deterministic and are built once at import time in
float64 numpy, then frozen.

This is a verbatim transcription of the Fortran iteration (secant solve, 11
iterations, tolerance 1e-3) so the JAX scheme indexes the SAME table the
Fortran oracle uses.
"""
from __future__ import annotations

import numpy as np

# WRF saturation-vapor-pressure constants (share/module_model_constants.F)
SVP1 = 0.6112
SVP2 = 17.67
SVP3 = 29.65
SVPT0 = 273.15

KFNT = 250
KFNP = 220


def _build_tables(svp1: float, svp2: float, svp3: float, svpt0: float):
    aliq = svp1 * 1000.0
    bliq = svp2
    cliq = svp2 * svpt0
    dliq = svp3

    dth = 1.0
    tmin = 150.0
    toler = 0.001
    plutop = 5000.0
    pbot = 110000.0

    rdthk = 1.0 / dth
    dpr = (pbot - plutop) / float(KFNP - 1)
    rdpr = 1.0 / dpr

    the0k = np.zeros(KFNP, dtype=np.float64)
    ttab = np.zeros((KFNT, KFNP), dtype=np.float64)
    qstab = np.zeros((KFNT, KFNP), dtype=np.float64)

    # base saturation equivalent potential temperature curve
    temp = tmin
    p = plutop - dpr
    for kp in range(KFNP):
        p = p + dpr
        es = aliq * np.exp((bliq * temp - cliq) / (temp - dliq))
        qs = 0.622 * es / (p - es)
        pi = (1.0e5 / p) ** (0.2854 * (1.0 - 0.28 * qs))
        the0k[kp] = temp * pi * np.exp((3374.6525 / temp - 2.5403) * qs * (1.0 + 0.81 * qs))

    # temperatures for each saturation equivalent potential temperature
    p = plutop - dpr
    for kp in range(KFNP):
        thes = the0k[kp] - dth
        p = p + dpr
        for it in range(KFNT):
            thes = thes + dth
            if it == 0:
                tgues = tmin
            else:
                tgues = ttab[it - 1, kp]
            es = aliq * np.exp((bliq * tgues - cliq) / (tgues - dliq))
            qs = 0.622 * es / (p - es)
            pi = (1.0e5 / p) ** (0.2854 * (1.0 - 0.28 * qs))
            thgues = tgues * pi * np.exp((3374.6525 / tgues - 2.5403) * qs * (1.0 + 0.81 * qs))
            f0 = thgues - thes
            t1 = tgues - 0.5 * f0
            t0 = tgues
            qs_last = qs
            for _itcnt in range(11):
                es = aliq * np.exp((bliq * t1 - cliq) / (t1 - dliq))
                qs = 0.622 * es / (p - es)
                pi = (1.0e5 / p) ** (0.2854 * (1.0 - 0.28 * qs))
                thtgs = t1 * pi * np.exp((3374.6525 / t1 - 2.5403) * qs * (1.0 + 0.81 * qs))
                f1 = thtgs - thes
                qs_last = qs
                if abs(f1) < toler:
                    break
                dt = f1 * (t1 - t0) / (f1 - f0)
                t0 = t1
                f0 = f1
                t1 = t1 - dt
            ttab[it, kp] = t1
            qstab[it, kp] = qs_last

    # log lookup table for tlog(emix/aliq)
    astrt = 1.0e-3
    ainc = 0.075
    alu = np.zeros(200, dtype=np.float64)
    a1 = astrt - ainc
    for i in range(200):
        a1 = a1 + ainc
        alu[i] = np.log(a1)

    return {
        "TTAB": ttab,
        "QSTAB": qstab,
        "THE0K": the0k,
        "ALU": alu,
        "RDPR": rdpr,
        "RDTHK": rdthk,
        "PLUTOP": plutop,
    }


# Built once at import, in float64, frozen.
_TABLES = _build_tables(SVP1, SVP2, SVP3, SVPT0)
TTAB = _TABLES["TTAB"]
QSTAB = _TABLES["QSTAB"]
THE0K = _TABLES["THE0K"]
ALU = _TABLES["ALU"]
RDPR = _TABLES["RDPR"]
RDTHK = _TABLES["RDTHK"]
PLUTOP = _TABLES["PLUTOP"]
