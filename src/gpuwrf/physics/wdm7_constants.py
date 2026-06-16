"""WDM7 (WRF mp_physics=26) constants, faithful to WRF phys/module_mp_wdm7.F.

WDM7 = WDM6 (DOUBLE-MOMENT warm rain: predicted cloud-droplet number Nc, rain
number Nr, CCN number Nn) extended with a SEPARATE precipitating HAIL class
(Bae, Hong, Tao 2018). The ice / snow / graupel / HAIL classes are all
SINGLE-MOMENT (qh is carried; there is NO hail number Nh -- the WDM7 ``ncr``
number array has only 3 components: ncr(:,:,1)=Nn (CCN), ncr(:,:,2)=Nc (cloud),
ncr(:,:,3)=Nr (rain)). The precipitating mass array is qrs(:,:,1..4) =
(rain, snow, graupel, hail).

All values mirror the pristine WRF source EXACTLY:

* The module ``parameter`` block of ``module_mp_wdm7.F`` (dtcldcr, n0r, n0g,
  n0h, avtr/avts/avtg/avth, the warm-rain double-moment parameters
  ncrk1/ncrk2/di100/..., the hail intercept/density/fall-speed n0h/denh/
  avth/bvth, the hail collection efficiencies eachs/eachg, the hail drag
  coefficient cd, the hail slope maximum lamdahmax). NOTE: WDM7 uses the WDM6
  warm-rain limits (lamdarmax=5.0e4, lamdarmin=2.0e3) -- NOT the WSM7
  lamdarmax=8.0e4. The double-moment rain slope is min(1/lamdar, 1.e-3).
* The derived ``save`` constants computed in ``wdm7init`` (the same warm-rain +
  graupel prefactors as wdm6init, PLUS the hail prefactors pvth/pacrh/prech1/
  prech2/prech3/pidn0h that mirror wsm7init).
* The WRF model constants bound in ``module_microphysics_driver.F`` at the
  ``CALL wdm7(...)`` site (g, cpd=cp, cpv, rd=r_d, rv=r_v, t0c=svpt0, ep1=ep_1,
  ep2=ep_2, qmin=epsilon, xls, xlv0=xlv, xlf0=xlf, den0=rhoair0,
  denr=rhowater, cliq, cice, psat, ccn0=ccn_conc).

``ccn_conc`` defaults to 1.0E8 # m-3 (Registry.EM_COMMON ccn_conc default), so
the WDM7 CCN background is ``CCN0 = 1.0e8``.

The gamma constants are reproduced from WRF's ``rgmma`` infinite-product
function (10000 terms) evaluated in fp64 so the JAX port's init is numerically
identical to the Fortran init to ~fp64 precision. Verified against
/home/enric/src/wrf_pristine/WRF/phys/module_mp_wdm7.F on 2026-06-14.
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------------
# WRF model constants (module_model_constants.F), bound at CALL wdm7(...)
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
# module_mp_wdm7.F module parameters (the `real, parameter, private` block)
# --------------------------------------------------------------------------
DTCLDCR = 120.0
N0R = 8.0e6
N0G = 4.0e6                   # intercept parameter graupel
N0H = 4.0e4                   # intercept parameter hail
AVTR = 841.9
BVTR = 0.8
R0 = 0.8e-5
PEAUT = 0.55
XNCR = 3.0e8
XNCR0 = 5.0e7
XNCR1 = 5.0e8
XMYU = 1.718e-5
AVTS = 11.72
BVTS = 0.41
AVTG = 330.0
BVTG = 0.8
DENG = 500.0                 # density of graupel
AVTH = 285.0                 # terminal velocity constant for hail
BVTH = 0.8
DENH = 912.0                 # density of hail
N0SMAX = 1.0e11
LAMDACMAX = 5.0e5
LAMDACMIN = 2.0e4
LAMDARMAX = 5.0e4
LAMDARMIN = 2.0e3
LAMDASMAX = 1.0e5
LAMDAGMAX = 6.0e4
LAMDAHMAX = 2.0e4
DICON = 11.9
DIMAX = 500.0e-6
N0S = 2.0e6
ALPHA = 0.12
PFRZ1 = 100.0
PFRZ2 = 0.66
QCRMIN = 1.0e-9
NCMIN = 1.0e1
NRMIN = 1.0e-2
EACRC = 1.0
EACHS = 1.0                  # hail/snow collection efficiency
EACHG = 0.5                  # hail/graupel collection efficiency
DENS = 100.0                 # density of snow
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
T00 = 238.16
T01 = 273.16
CD = 0.6                     # drag coefficient for hailstone

PI = 4.0 * math.atan(1.0)
XLV1 = CLIQ - CPV            # xlv1 = cl - cpv


def _rgmma(x: float) -> float:
    """WRF rgmma: reciprocal-gamma via infinite product (10000 terms), fp64.

    Reproduces module_mp_wdm7.F ``rgmma`` exactly (same algorithm, same
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
# Derived constants from wdm7init, fp64 evaluation.
# NOTE: WDM7 qck1 (like WDM6) divides by denr**(1/3) (not (xncr*denr)**(1/3))
# because the double-moment praut uses nc explicitly.
# --------------------------------------------------------------------------
QC0 = 4.0 / 3.0 * PI * DENR * R0 ** 3 * XNCR0 / DEN0
QC1 = 4.0 / 3.0 * PI * DENR * R0 ** 3 * XNCR1 / DEN0
QCK1 = 0.104 * 9.8 * PEAUT / DENR ** (1.0 / 3.0) / XMYU * DEN0 ** (4.0 / 3.0)  # ~4706.082
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

# graupel
BVTG1 = 1.0 + BVTG
BVTG2 = 2.5 + 0.5 * BVTG
BVTG3 = 3.0 + BVTG
BVTG4 = 4.0 + BVTG
G1PBG = _rgmma(BVTG1)
G3PBG = _rgmma(BVTG3)
G4PBG = _rgmma(BVTG4)
G5PBGO2 = _rgmma(BVTG2)
G6PBGH = _rgmma(2.75)
PACRG = PI * N0G * AVTG * G3PBG * 0.25
PVTG = AVTG * G4PBG / 6.0
PRECG1 = 2.0 * PI * N0G * 0.78
PRECG2 = 2.0 * PI * N0G * 0.31 * AVTG ** 0.5 * G5PBGO2
PRECG3 = 2.0 * PI * N0G * 0.31 * G6PBGH * math.sqrt(math.sqrt(4.0 * DENG / 3.0 / CD))
PIDN0G = PI * DENG * N0G

# hail
BVTH2 = 2.5 + 0.5 * BVTH
BVTH3 = 3.0 + BVTH
BVTH4 = 4.0 + BVTH
G3PBH = _rgmma(BVTH3)
G4PBH = _rgmma(BVTH4)
G5PBHO2 = _rgmma(BVTH2)
PACRH = PI * N0H * AVTH * G3PBH * 0.25
PVTH = AVTH * G4PBH / 6.0
PRECH1 = 2.0 * PI * N0H * 0.78
PRECH2 = 2.0 * PI * N0H * 0.31 * AVTH ** 0.5 * G5PBHO2
PRECH3 = 2.0 * PI * N0H * 0.31 * G6PBGH * math.sqrt(math.sqrt(4.0 * DENH / 3.0 / CD))
PIDN0H = PI * DENH * N0H

# slope maxima / minima
RSLOPECMAX = 1.0 / LAMDACMAX
RSLOPERMAX = 1.0 / LAMDARMAX
RSLOPESMAX = 1.0 / LAMDASMAX
RSLOPEGMAX = 1.0 / LAMDAGMAX
RSLOPEHMAX = 1.0 / LAMDAHMAX
RSLOPERBMAX = RSLOPERMAX ** BVTR
RSLOPESBMAX = RSLOPESMAX ** BVTS
RSLOPEGBMAX = RSLOPEGMAX ** BVTG
RSLOPEHBMAX = RSLOPEHMAX ** BVTH
RSLOPEC2MAX = RSLOPECMAX * RSLOPECMAX
RSLOPER2MAX = RSLOPERMAX * RSLOPERMAX
RSLOPES2MAX = RSLOPESMAX * RSLOPESMAX
RSLOPEG2MAX = RSLOPEGMAX * RSLOPEGMAX
RSLOPEH2MAX = RSLOPEHMAX * RSLOPEHMAX
RSLOPEC3MAX = RSLOPEC2MAX * RSLOPECMAX
RSLOPER3MAX = RSLOPER2MAX * RSLOPERMAX
RSLOPES3MAX = RSLOPES2MAX * RSLOPESMAX
RSLOPEG3MAX = RSLOPEG2MAX * RSLOPEGMAX
RSLOPEH3MAX = RSLOPEH2MAX * RSLOPEHMAX

# effective-radius background/max (module_model_constants RE_QC_BG/.. + driver)
RE_QC_BG = 2.49e-6
RE_QI_BG = 4.99e-6
RE_QS_BG = 9.99e-6
RE_QC_MAX = 50.0e-6
RE_QI_MAX = 125.0e-6
RE_QS_MAX = 999.0e-6

__all__ = [name for name in dir() if name.isupper()]
