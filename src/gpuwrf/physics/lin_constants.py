"""Purdue-Lin (WRF mp_physics=2) constants, faithful to WRF phys/module_mp_lin.F.

All values mirror the pristine WRF source exactly:

* The module ``PARAMETER`` block of ``module_mp_lin.F`` (xnor, xnos, xnog,
  rhograul, qi0, ql0, qs0, constb, constd, cdrag, avisc, ...).
* The WRF model constants bound at the ``CALL lin_et_al(...)`` site in
  ``module_microphysics_driver.F`` (g, cp, r_d=Rair, r_v=rvapor, xls, xlv,
  xlf, rhowater, rhosnow, ep_2, svp1, svp2, svp3, svpt0).
* The Bergeron-process lookup tables ``parama1``/``parama2`` (32 entries) and
  the ``ggamma`` 8-term Hastings polynomial gamma, reproduced verbatim so the
  JAX port's process coefficients are numerically identical to the Fortran.

Verified against /home/user/src/wrf_pristine/WRF/phys/module_mp_lin.F on
2026-06-04 (sha256 in proofs/v060/savepoints_lin/wrf_source_checksums.txt).
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------------
# WRF model constants (module_model_constants.F), bound at CALL lin_et_al(...)
# --------------------------------------------------------------------------
G = 9.81
RAIR = 287.0                  # r_d
CP = 7.0 * RAIR / 2.0         # 1004.5
RVAPOR = 461.6                # r_v
XLS = 2.85e6
XLV = 2.5e6
XLF = 3.50e5
RHOWATER = 1000.0
RHOSNOW = 100.0
EP2 = RAIR / RVAPOR           # ep_2
SVP1 = 0.6112
SVP2 = 17.67
SVP3 = 29.65
SVPT0 = 273.15

# --------------------------------------------------------------------------
# module_mp_lin.F PARAMETER block (PRIVATE)
# --------------------------------------------------------------------------
RH = 1.0
XNOR = 8.0e6                  # rain intercept
XNOS = 3.0e6                  # snow intercept
XNOG = 4.0e6                  # graupel intercept (Hobbs)
RHOGRAUL = 400.0              # graupel density (Hobbs)
QI0 = 1.0e-3
QL0 = 7.0e-4
QS0 = 6.0e-4
XMI50 = 4.8e-10
XMI40 = 2.46e-10
CONSTB = 0.8
CONSTD = 0.25
O6 = 1.0 / 6.0
CDRAG = 0.6
AVISC = 1.49628e-6
ADIFFWV = 8.7602e-5
AXKA = 1.4132e3
DI50 = 1.0e-4
XMI = 4.19e-13
CW = 4.187e3
VF1S = 0.78
VF2S = 0.31
XNI0 = 1.0e-2
XMNIN = 1.05e-18
BNI = 0.5
CI = 2.093e3

QVMIN = 1.0e-20

# --------------------------------------------------------------------------
# ggamma: WRF 8-term Hastings polynomial gamma (REAL FUNCTION ggamma), fp64.
# Reproduces module_mp_lin.F ggamma exactly (same coefficients, same range
# reduction TEMP=X..down to <=2 with PF accumulation).
# --------------------------------------------------------------------------
_GGAMMA_B = (
    -0.577191652, 0.988205891, -0.897056937, 0.918206857,
    -0.756704078, 0.482199394, -0.193527818, 0.035868343,
)


def ggamma(x: float) -> float:
    pf = 1.0
    temp = float(x)
    # range-reduce to (1,2]: WRF loops PF*=TEMP while TEMP>2, TEMP-=1
    n = 0
    while temp > 2.0:
        temp = temp - 1.0
        pf = pf * temp
        n += 1
        if n > 200:
            raise ValueError("ggamma input too large")
    g1to2 = 1.0
    temp = temp - 1.0
    for k1 in range(1, 9):
        g1to2 = g1to2 + _GGAMMA_B[k1 - 1] * temp ** k1
    return pf * g1to2


# --------------------------------------------------------------------------
# Derived gamma constants computed at the top of clphy1d (fp64 evaluation).
# --------------------------------------------------------------------------
CONSTA = 2115.0 * 0.01 ** (1 - CONSTB)   # rain fall-speed prefactor
CONSTC = 152.93 * 0.01 ** (1 - CONSTD)   # snow fall-speed prefactor
OCDRAG = 1.0 / CDRAG
EPISP0K = RH * EP2 * 1000.0 * SVP1

GAMBP4 = ggamma(CONSTB + 4.0)
GAMDP4 = ggamma(CONSTD + 4.0)
GAM4PT5 = ggamma(4.5)
CPOR = CP / RAIR
OXMI = 1.0 / XMI
GAMBP3 = ggamma(CONSTB + 3.0)
GAMDP3 = ggamma(CONSTD + 3.0)
GAMBP6 = ggamma(CONSTB + 6.0)
GAM3PT5 = ggamma(3.5)
GAM2PT75 = ggamma(2.75)
GAMBP5O2 = ggamma((CONSTB + 5.0) / 2.0)
GAMDP5O2 = ggamma((CONSTD + 5.0) / 2.0)
CWOXLF = CW / XLF

OBP4 = 1.0 / (CONSTB + 4.0)
BP3 = CONSTB + 3.0
BP5 = CONSTB + 5.0
BP6 = CONSTB + 6.0
ODP4 = 1.0 / (CONSTD + 4.0)
DP3 = CONSTD + 3.0
DP5 = CONSTD + 5.0
DP5O2 = 0.5 * (CONSTD + 5.0)

PI = math.acos(-1.0)
PIO4 = PI / 4.0
PIO6 = PI / 6.0
OCP = 1.0 / CP
OXLF = 1.0 / XLF
XLVOCP = XLV / CP
XLFOCP = XLF / CP

# --------------------------------------------------------------------------
# parama1 / parama2 Bergeron-process lookup tables (32 entries), DATA verbatim.
# Indexed by i1=int(-temp)+1 with linear interp ratio=-temp-(i1-1).
# --------------------------------------------------------------------------
PARAMA1 = (
    0.100e-10, 0.7939e-7, 0.7841e-6, 0.3369e-5, 0.4336e-5,
    0.5285e-5, 0.3728e-5, 0.1852e-5, 0.2991e-6, 0.4248e-6,
    0.7434e-6, 0.1812e-5, 0.4394e-5, 0.9145e-5, 0.1725e-4,
    0.3348e-4, 0.1725e-4, 0.9175e-5, 0.4412e-5, 0.2252e-5,
    0.9115e-6, 0.4876e-6, 0.3473e-6, 0.4758e-6, 0.6306e-6,
    0.8573e-6, 0.7868e-6, 0.7192e-6, 0.6513e-6, 0.5956e-6,
    0.5333e-6, 0.4834e-6,
)
PARAMA2 = (
    0.0100, 0.4006, 0.4831, 0.5320, 0.5307, 0.5319, 0.5249,
    0.4888, 0.3849, 0.4047, 0.4318, 0.4771, 0.5183, 0.5463,
    0.5651, 0.5813, 0.5655, 0.5478, 0.5203, 0.4906, 0.4447,
    0.4126, 0.3960, 0.4149, 0.4320, 0.4506, 0.4483, 0.4460,
    0.4433, 0.4413, 0.4382, 0.4361,
)

__all__ = [name for name in dir() if name.isupper() or name == "ggamma"]
