"""S1 (Opus) — terrain-aware dry base state + MUB/PHB.

FROZEN ENTRY SIGNATURE. Implements the real.exe base-state block
(module_initialize_real.F:3781-3835), the ``setup_base_state``-equivalent.

Algorithm (faithful spec; per-column over (i,j), terrain ht from SurfaceInit.hgt
— the SAME terrain S2 carries, so S1 and S2 MUST agree on ht; the driver passes
the single met_em-derived HGT_M to both, the freeze guarantees one source):
  p_surf = p00 * exp( -t00/a + ((t00/a)^2 - 2*g*ht/a/r_d)^0.5 )    (:3790)
  for k in 0..nz-1 (model order; uses c3h[k], c4h[k] from VerticalCoord1D):
     pb[k]   = c3h[k]*(p_surf - p_top) + c4h[k] + p_top             (:3795)
     temp    = max(tiso, t00 + a*log(pb[k]/p00))                    (:3796)
       (if pb[k] < p_strat: temp = tiso + a_strat*log(pb[k]/p_strat)) (:3797-3798)
     t_init[k] = temp*(p00/pb[k])^(r_d/cp) - t0                     (:3801)
     alb[k]    = (r_d/p1000mb)*(t_init[k]+t0)*(pb[k]/p1000mb)^cvpm  (:3802)
  mub = p_surf - p_top                                              (:3806)
  phb[0] = ht*g                                                     (:3813)
  hybrid_opt==0:
     phb[k] = phb[k-1] - dnw[k-1]*(c1h*mub+c2h)*alb[k-1]            (:3817)
  hybrid_opt>=1 (Canary=2):
     pfu=c3f[k]*mub+c4f[k]+p_top ; pfd=c3f[k-1]*mub+c4f[k-1]+p_top
     phm=c3h[k-1]*mub+c4h[k-1]+p_top
     phb[k] = phb[k-1] + alb[k-1]*phm*log(pfd/pfu)                  (:3821-3824)

Oracle: wrfinput PB / MUB / PHB for d01/d02/d03; tols ``WRFINPUT_TOLS`` PB/MUB/
PHB. Must match exactly modulo fp rounding (this is the hour-0 foundation).

FILE OWNERSHIP: S1 exclusive (see vertical_coord.py header).
"""

from __future__ import annotations

import numpy as np

from gpuwrf.init.real_init.types import (
    BaseStateColumns,
    CP,
    CVPM,
    G,
    P1000MB,
    R_D,
    T0,
    RealInitConfig,
    VerticalCoord1D,
)


def compute_base_state(
    config: RealInitConfig,
    vcoord: VerticalCoord1D,
    hgt: np.ndarray,
) -> BaseStateColumns:
    """Builds pb/alb/t_init/mub/phb from terrain + the 1D coordinate.

    ``hgt`` is the (ny, nx) terrain height (m) — the SAME field S2 puts in
    SurfaceInit.hgt; the driver sources it once from met_em HGT_M and passes it
    to both lanes (frozen single-source rule).
    """

    hgt64 = np.asarray(hgt, dtype=np.float64)
    if hgt64.ndim != 2:
        raise ValueError(f"hgt must be 2D (ny,nx); got shape {hgt64.shape}")

    p00 = float(config.base_pres)
    t00 = float(config.base_temp)
    lapse = float(config.base_lapse)
    p_top = float(config.p_top_pa)

    p_surf = p00 * np.exp(
        -t00 / lapse + np.sqrt((t00 / lapse) ** 2 - 2.0 * G * hgt64 / lapse / R_D)
    )
    mub = p_surf - p_top

    pb = (
        vcoord.c3h[:, None, None] * (p_surf[None, :, :] - p_top)
        + vcoord.c4h[:, None, None]
        + p_top
    )
    temp = np.maximum(
        float(config.iso_temp),
        t00 + lapse * np.log(pb / p00),
    )
    if config.base_pres_strat > 0.0:
        strat = pb < float(config.base_pres_strat)
        temp = np.where(
            strat,
            float(config.iso_temp)
            + float(config.base_lapse_strat) * np.log(pb / float(config.base_pres_strat)),
            temp,
        )
    t_init = temp * (p00 / pb) ** (R_D / CP) - T0
    alb = (R_D / P1000MB) * (t_init + T0) * (pb / P1000MB) ** CVPM

    nz = config.nz
    phb = np.empty((nz + 1, *hgt64.shape), dtype=np.float64)
    phb[0] = hgt64 * G
    if config.hybrid_opt == 0:
        for k in range(1, nz + 1):
            h = k - 1
            phb[k] = (
                phb[k - 1]
                - vcoord.dnw[h]
                * (vcoord.c1h[h] * mub + vcoord.c2h[h])
                * alb[h]
            )
    else:
        for k in range(1, nz + 1):
            h = k - 1
            pfu = vcoord.c3f[k] * mub + vcoord.c4f[k] + p_top
            pfd = vcoord.c3f[k - 1] * mub + vcoord.c4f[k - 1] + p_top
            phm = vcoord.c3h[h] * mub + vcoord.c4h[h] + p_top
            phb[k] = phb[k - 1] + alb[h] * phm * np.log(pfd / pfu)

    return BaseStateColumns(
        pb=pb,
        alb=alb,
        t_init=t_init,
        mub=mub,
        phb=phb,
    )
