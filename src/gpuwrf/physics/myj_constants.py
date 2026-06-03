"""WRF-faithful constants for the Mellor-Yamada-Janjic (MYJ) PBL + Janjic Eta
surface-layer pair (``bl_pbl_physics=2`` / ``sf_sfclay_physics=2``).

All values are transcribed 1:1 from the UNMODIFIED pristine WRF sources:

* ``phys/module_bl_myjpbl.F`` (MYJ PBL parameters).
* ``phys/module_sf_myjsfc.F`` (Janjic surface-layer parameters).
* ``share/module_model_constants.F`` (shared model constants).

Derived parameters are computed here exactly as the Fortran ``PARAMETER``
expressions so the JAX port matches WRF to machine precision in fp64.
"""

from __future__ import annotations

import math


# --- share/module_model_constants.F -----------------------------------------
G = 9.81
R_D = 287.0
CP = 7.0 * R_D / 2.0
R_V = 461.6
CAPA = R_D / CP
RCAP = 1.0 / CAPA
RVOVRD = R_V / R_D
P608 = RVOVRD - 1.0
XLV = 2.5e6
XLS = 2.85e6
P1000MB = 100000.0
PQ0 = 379.90516
EPSQ2 = 0.2
EPSQ = 1.0e-12
A2 = 17.2693882
A3 = 273.16
A4 = 35.86

# --- common Janjic surface-layer + PBL parameters ----------------------------
VKARMAN = 0.4
PI = 3.1415926
ELOCP = 2.72e6 / CP
SEAFC = 0.98
PQ0SEA = PQ0 * SEAFC
BETA = 1.0 / 273.0
BTG = BETA * G
GLKBR = 10.0
GLKBS = 30.0
GRRS = GLKBR / GLKBS
SMALL = 0.35
CZIV = SMALL * GLKBS
QVISC = 2.1e-5
TVISC = 2.1e-5
VISC = 1.5e-5
RTVISC = 1.0 / TVISC
RVISC = 1.0 / VISC
RQVISC = 1.0 / QVISC
SQPR = 0.84
SQSC = 0.84
SQVISC = 258.2
ZQRZT = SQSC / SQPR
USTC = 0.7
USTR = 0.225
WWST = 1.2
WWST2 = WWST * WWST
USTFC = 0.018 / G
PIHF = 0.5 * PI
FH = 1.01
RIC = 0.505

FZQ1 = RTVISC * QVISC * ZQRZT
FZQ2 = RTVISC * QVISC * ZQRZT
FZT1 = RVISC * TVISC * SQPR
FZT2 = CZIV * GRRS * TVISC * SQPR
FZU1 = CZIV * VISC

# --- module_sf_myjsfc.F specific ---------------------------------------------
A2S = 17.2693882
A3S = 273.16
A4S = 35.86
EPSU2 = 1.0e-6
EPSUST = 1.0e-9
EPSZT = 1.0e-28
EXCML = 0.0001
EXCMS = 0.0001
ZTFC = 1.0
GOCP02 = G / CP * 2.0
GOCP10 = G / CP * 10.0
SFC_ITRMX = 5

# Surface-layer integral-function lookup table (MYJSFCINIT).
KZTM = 10001
KZTM2 = KZTM - 2
# After the K=KZTM iteration the Fortran resets ZTMAX1/2 to the final ZETA then
# subtracts EPS=1.e-6. ZTMIN1/2=-5.0 with ZRNG=6.0 spanning ZTMAX_nominal=1.0.
SFC_ZTMIN1 = -5.0
SFC_ZTMIN2 = -5.0
_SFC_ZTMAX_NOMINAL = 1.0
SFC_DZETA1 = (_SFC_ZTMAX_NOMINAL - SFC_ZTMIN1) / (KZTM - 1)
SFC_DZETA2 = (_SFC_ZTMAX_NOMINAL - SFC_ZTMIN2) / (KZTM - 1)
# ZTMAX after the loop = ZTMIN + (KZTM-1)*DZETA - EPS = 1.0 - 1.e-6.
SFC_EPS = 1.0e-6
SFC_ZTMAX1 = SFC_ZTMIN1 + (KZTM - 1) * SFC_DZETA1 - SFC_EPS
SFC_ZTMAX2 = SFC_ZTMIN2 + (KZTM - 1) * SFC_DZETA2 - SFC_EPS
FH01 = 1.0
FH02 = 1.0

# --- module_bl_myjpbl.F specific ---------------------------------------------
PBL_ITRMX = 5
RLIVWV = XLS / XLV
EPS1 = 1.0e-12
EPS2 = 0.0
EPSL = 0.32
EPSRU = 1.0e-7
EPSRS = 1.0e-7
EPSTRB = 1.0e-24
ALPH = 0.30
EL0MAX = 1000.0
EL0MIN = 1.0
ELFC = 0.23 * 0.5
ELZ0 = 0.0
ESQ = 5.0
PRT = 1.0

A1 = 0.659888514560862645
A2x = 0.6574209922667784586
B1 = 11.87799326209552761
B2 = 7.226971804046074028
C1 = 0.000830955950095854396
RB1 = 1.0 / B1

# Mixing-length / diffusion coefficient closure constants (MYJ level 2.5).
ADNH = 9.0 * A1 * A2x * A2x * (12.0 * A1 + 3.0 * B2) * BTG * BTG
ADNM = 18.0 * A1 * A1 * A2x * (B2 - 3.0 * A2x) * BTG
ANMH = -9.0 * A1 * A2x * A2x * BTG * BTG
ANMM = -3.0 * A1 * A2x * (3.0 * A2x + 3.0 * B2 * C1 + 18.0 * A1 * C1 - B2) * BTG
BDNH = 3.0 * A2x * (7.0 * A1 + B2) * BTG
BDNM = 6.0 * A1 * A1
BEQH = A2x * B1 * BTG + 3.0 * A2x * (7.0 * A1 + B2) * BTG
BEQM = -A1 * B1 * (1.0 - 3.0 * C1) + 6.0 * A1 * A1
BNMH = -A2x * BTG
BNMM = A1 * (1.0 - 3.0 * C1)
BSHH = 9.0 * A1 * A2x * A2x * BTG
BSHM = 18.0 * A1 * A1 * A2x * C1
BSMH = -3.0 * A1 * A2x * (3.0 * A2x + 3.0 * B2 * C1 + 12.0 * A1 * C1 - B2) * BTG
CESH = A2x
CESM = A1 * (1.0 - 3.0 * C1)

AEQH = (
    9.0 * A1 * A2x * A2x * B1 * BTG * BTG
    + 9.0 * A1 * A2x * A2x * (12.0 * A1 + 3.0 * B2) * BTG * BTG
)
AEQM = (
    3.0 * A1 * A2x * B1 * (3.0 * A2x + 3.0 * B2 * C1 + 18.0 * A1 * C1 - B2) * BTG
    + 18.0 * A1 * A1 * A2x * (B2 - 3.0 * A2x) * BTG
)

REQU = -AEQH / AEQM
EPSGH = 1.0e-9
EPSGM = REQU * EPSGH

UBRYL = (
    18.0 * REQU * A1 * A1 * A2x * B2 * C1 * BTG
    + 9.0 * A1 * A2x * A2x * B2 * BTG * BTG
) / (REQU * ADNM + ADNH)
UBRY = (1.0 + EPSRS) * UBRYL
UBRY3 = 3.0 * UBRY

AUBH = 27.0 * A1 * A2x * A2x * B2 * BTG * BTG - ADNH * UBRY3
AUBM = 54.0 * A1 * A1 * A2x * B2 * C1 * BTG - ADNM * UBRY3
BUBH = (9.0 * A1 * A2x + 3.0 * A2x * B2) * BTG - BDNH * UBRY3
BUBM = 18.0 * A1 * A1 * C1 - BDNM * UBRY3
CUBR = 1.0 - UBRY3
RCUBR = 1.0 / CUBR


__all__ = [name for name in dir() if not name.startswith("_") and name.isupper()
           or name in ("A1", "A2x", "B1", "B2", "C1")]
del math
