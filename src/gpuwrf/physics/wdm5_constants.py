"""WDM5 (WRF mp_physics=14) constants, faithful to WRF phys/module_mp_wdm5.F.

WDM5 = WSM5-style single-moment ICE/SNOW microphysics (NO graupel, NO hail) +
a DOUBLE-MOMENT warm rain (predicted cloud-droplet number Nc, rain number Nr,
and CCN number Nn) -- identical warm-rain double-moment machinery to WDM6, but
the cold-rain side is the 5-class WSM5 ice physics (rain + snow only in the
precipitating ``qrs`` array; cloud water + ice in ``qci``).

All values here mirror the pristine WRF source EXACTLY:

* The module ``parameter`` block of ``module_mp_wdm5.F`` (dtcldcr, n0r, n0s,
  avtr, bvtr, the warm-rain double-moment parameters ncrk1/ncrk2/di100/...).
* The derived ``save`` constants computed in ``wdm5init``.
* The WRF model constants bound at the ``CALL wdm5(...)`` site:
  g, cpd=cp, cpv, rd=r_d, rv=r_v, t0c=svpt0, ep1=ep_1, ep2=ep_2,
  qmin=epsilon, xls, xlv0=xlv, xlf0=xlf, den0=rhoair0, denr=rhowater,
  cliq, cice, psat, ccn0=ccn_conc.

TWO DIFFERENCES from WDM6's init that are LOAD-BEARING:

* ``qck1`` is the WSM6/WDM5 form ``.104*9.8*peaut/(xncr*denr)**(1/3)/xmyu*
  den0**(4/3)`` -- it divides by ``(xncr*denr)**(1/3)``, NOT ``denr**(1/3)``
  (the WDM6 form). WDM5's praut does NOT use qck1 -- it uses the LH/CP
  lencon/taucon form -- but qck1 is still computed in wdm5init, so we record
  it for completeness.
* ``qc0`` uses the single ``xncr = 3.e8`` (maritime droplet number), and there
  is NO ``xncr0/xncr1`` split and NO land/sea ``qcr`` switch (WDM5's praut is
  gated only by the mean-volume cloud diameter ``avedia > di15``).

There is NO graupel: no n0g/deng/avtg/bvtg/lamdagmax and no graupel slope
maxima.

The gamma constants are reproduced from WRF's ``rgmma`` infinite-product
function (10000 terms) evaluated in fp64 so the JAX port's init is numerically
identical to the Fortran init to ~fp64 precision. Verified against
/home/enric/src/wrf_pristine/WRF/phys/module_mp_wdm5.F.
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------------
# WRF model constants (module_model_constants.F), bound at CALL wdm5(...)
# --------------------------------------------------------------------------
G = 9.81
RD = 287.0
CPD = 7.0 * RD / 2.0          # cp = 1004.5
RV = 461.6
CPV = 4.0 * RV                # 1846.4
CLIQ = 4190.0
CICE = 2106.0
PSAT = 610.78
XLV0 = 2.5e6                  # xlv
XLS = 2.85e6
XLF0 = 3.50e5                 # xlf
T0C = 273.15                  # svpt0
EP1 = RV / RD - 1.0
EP2 = RD / RV
QMIN = 1.0e-15               # epsilon
DEN0 = 1.28                  # rhoair0
DENR = 1000.0                # rhowater
CCN0 = 1.0e8                 # ccn_conc default (Registry.EM_COMMON)

# --------------------------------------------------------------------------
# module_mp_wdm5.F module parameters (the `real, parameter, private` block)
# --------------------------------------------------------------------------
DTCLDCR = 120.0
N0R = 8.0e6
N0S = 2.0e6
N0SMAX = 1.0e11
DENS = 100.0                 # snow bulk density (wdm5init `dens` arg)
ALPHA = 0.12
AVTR = 841.9
BVTR = 0.8
AVTS = 11.72
BVTS = 0.41
LAMDACMAX = 5.0e5
LAMDACMIN = 2.0e4
LAMDARMAX = 5.0e4
LAMDARMIN = 2.0e3
LAMDASMAX = 1.0e5
R0 = 0.8e-5
PEAUT = 0.55
XNCR = 3.0e8                 # WDM5: single maritime droplet number
XMYU = 1.718e-5
DICON = 11.9
DIMAX = 500.0e-6
PFRZ1 = 100.0
PFRZ2 = 0.66
QCRMIN = 1.0e-9
NCMIN = 1.0e1
NRMIN = 1.0e-2
EACRC = 1.0
QS0 = 6.0e-4
SATMAX = 1.0048
ACTK = 0.6
ACTR = 1.5
NCRK1 = 3.03e3
NCRK2 = 2.59e15
DI100 = 1.0e-4
DI600 = 6.0e-4
DI2000 = 2000.0e-6
DI82 = 82.0e-6
DI15 = 15.0e-6

PI = 4.0 * math.atan(1.0)
XLV1 = CLIQ - CPV            # xlv1 = cl - cpv


def _rgmma(x: float) -> float:
    """WRF rgmma: reciprocal-gamma via infinite product (10000 terms), fp64.

    Reproduces module_mp_wdm5.F ``rgmma`` exactly (same algorithm, same
    iteration count). Returns 1/Gamma(x).
    """
    if x == 1.0:
        return 0.0
    euler = 0.577215664901532
    r = x * math.exp(euler * x)
    for i in range(1, 10001):
        y = float(i)
        r = r * (1.0 + x / y) * math.exp(-x / y)
    return 1.0 / r


# --------------------------------------------------------------------------
# Derived constants from wdm5init, fp64 evaluation.
# NOTE: WDM5 qck1 divides by (xncr*denr)**(1/3) (WSM6 form), NOT denr**(1/3)
# (the WDM6 form). qc0 uses the single xncr (no xncr0/xncr1 split).
# --------------------------------------------------------------------------
QC0 = 4.0 / 3.0 * PI * DENR * R0 ** 3 * XNCR / DEN0
QCK1 = 0.104 * 9.8 * PEAUT / (XNCR * DENR) ** (1.0 / 3.0) / XMYU * DEN0 ** (4.0 / 3.0)
PIDNC = PI * DENR / 6.0

# rain (double-moment): pvtr = avtr*g5pbr/24, pvtrn = avtr*g2pbr
BVTR1 = 1.0 + BVTR
BVTR2 = 2.0 + BVTR
BVTR3 = 3.0 + BVTR
BVTR4 = 4.0 + BVTR
BVTR5 = 5.0 + BVTR
BVTR6 = 6.0 + BVTR
BVTR7 = 7.0 + BVTR
BVTR2O5 = 2.5 + 0.5 * BVTR
BVTR3O5 = 3.5 + 0.5 * BVTR
G1PBR = _rgmma(BVTR1)
G2PBR = _rgmma(BVTR2)
G3PBR = _rgmma(BVTR3)
G4PBR = _rgmma(BVTR4)        # 17.837825
G5PBR = _rgmma(BVTR5)
G6PBR = _rgmma(BVTR6)
G7PBR = _rgmma(BVTR7)
G5PBRO2 = _rgmma(BVTR2O5)
G7PBRO2 = _rgmma(BVTR3O5)
PVTR = AVTR * G5PBR / 24.0
PVTRN = AVTR * G2PBR
EACRR = 1.0
PACRR = PI * N0R * AVTR * G3PBR * 0.25 * EACRR
PRECR1 = 2.0 * PI * 1.56
PRECR2 = 2.0 * PI * 0.31 * AVTR ** 0.5 * G7PBRO2
PIDN0R = PI * DENR * N0R
PIDNR = 4.0 * PI * DENR

XMMAX = (DIMAX / DICON) ** 2
ROQIMAX = 2.08e22 * DIMAX ** 8

# snow
BVTS1 = 1.0 + BVTS
BVTS2 = 2.5 + 0.5 * BVTS
BVTS3 = 3.0 + BVTS
BVTS4 = 4.0 + BVTS
G1PBS = _rgmma(BVTS1)
G3PBS = _rgmma(BVTS3)
G4PBS = _rgmma(BVTS4)
G5PBSO2 = _rgmma(BVTS2)
PVTS = AVTS * G4PBS / 6.0
PACRS = PI * N0S * AVTS * G3PBS * 0.25
PRECS1 = 4.0 * N0S * 0.65
PRECS2 = 4.0 * N0S * 0.44 * AVTS ** 0.5 * G5PBSO2
PIDN0S = PI * DENS * N0S
PACRC = PI * N0S * AVTS * G3PBS * 0.25 * EACRC

# slope maxima / minima (NO graupel)
RSLOPECMAX = 1.0 / LAMDACMAX
RSLOPERMAX = 1.0 / LAMDARMAX
RSLOPESMAX = 1.0 / LAMDASMAX
RSLOPERBMAX = RSLOPERMAX ** BVTR
RSLOPESBMAX = RSLOPESMAX ** BVTS
RSLOPEC2MAX = RSLOPECMAX * RSLOPECMAX
RSLOPER2MAX = RSLOPERMAX * RSLOPERMAX
RSLOPES2MAX = RSLOPESMAX * RSLOPESMAX
RSLOPEC3MAX = RSLOPEC2MAX * RSLOPECMAX
RSLOPER3MAX = RSLOPER2MAX * RSLOPERMAX
RSLOPES3MAX = RSLOPES2MAX * RSLOPESMAX

# effective-radius background/max (module_microphysics_driver.F defaults +
# module_model_constants RE_QC_BG/RE_QI_BG/RE_QS_BG)
RE_QC_BG = 2.49e-6
RE_QI_BG = 4.99e-6
RE_QS_BG = 9.99e-6
RE_QC_MAX = 50.0e-6
RE_QI_MAX = 125.0e-6
RE_QS_MAX = 999.0e-6

__all__ = [name for name in dir() if name.isupper()]
