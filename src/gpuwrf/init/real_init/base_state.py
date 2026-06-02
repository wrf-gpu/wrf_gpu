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

    raise NotImplementedError("v0.4.0 S1 (Opus): compute_base_state — frozen stub")
