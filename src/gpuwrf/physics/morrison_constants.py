"""Morrison 2-moment microphysics constants (WRF mp_physics=10).

Faithful transcription of ``MORR_TWO_MOMENT_INIT`` in
``phys/module_mp_morr_two_moment.F`` for the default WRF configuration used by
the oracle driver: ``morr_rimed_ice=0`` -> ``IHAIL=0`` (graupel mode, not hail),
``IGRAUP=0`` (graupel included), ``ILIQ=0`` (ice included), ``INUC=0`` (Cooper/
Rasmussen ice nucleation), ``IACT=2`` (lognormal aerosol; only used when
``iinum=0``), ``iinum=1`` (constant droplet number ``NDCNST=250`` cm^-3).

All ``CONS*`` efficiency constants are the SAME GAMMA-and-power expressions as
the Fortran init, evaluated here in fp64 (the JAX port runs fp64). Against the
fp64 oracle (the same scheme promoted with -fdefault-real-8) these constants are
bit-faithful to ~1e-12; against the canonical fp32 oracle they differ only by
fp32 round-off, which the predeclared parity tolerance absorbs.

The Euler gamma function used by WRF Morrison is the Cody/Stoltz rational
minimax ``GAMMA``; for positive real arguments it agrees with the true gamma to
machine precision, so we use ``scipy.special``-free ``math.gamma`` via NumPy at
import time (host constant folding, no JAX tracing).
"""

from __future__ import annotations

import math

# ----------------------------------------------------------------------------
# WRF model constants (share/module_model_constants.F), as USEd by Morrison.
# ----------------------------------------------------------------------------
CP = 7.0 * 287.0 / 2.0      # = 1004.5  (cp = 7*r_d/2)
G = 9.81
R = 287.0                   # r_d
RV = 461.6                  # r_v
EP_2 = R / RV               # r_d/r_v

PI = 3.1415926535897932384626434
XXX = 0.9189385332046727417803297   # 0.5*log(2*pi), used in GAMMA for x>=12

# ----------------------------------------------------------------------------
# User switches frozen by MORR_TWO_MOMENT_INIT(morr_rimed_ice=0).
# ----------------------------------------------------------------------------
INUM = 1            # constant droplet number
IINUM = 1           # wrapper sets iinum=1 (no qndrop coupling)
NDCNST = 250.0      # cm^-3
IACT = 2
IBASE = 2
ISUB = 0
ILIQ = 0            # include ice
INUC = 0            # Rasmussen/Cooper nucleation
IGRAUP = 0          # include graupel
IHAIL = 0           # dense precip ice is graupel (not hail)

# ----------------------------------------------------------------------------
# Fall-speed parameters (V = A D^B). IHAIL=0 branch.
# ----------------------------------------------------------------------------
AI = 700.0
AC = 3.0e7
AS = 11.72
AR = 841.99667
BI = 1.0
BC = 2.0
BS = 0.41
BR = 0.8
AG = 19.3           # IHAIL=0
BG = 0.37           # IHAIL=0

# ----------------------------------------------------------------------------
# Physical constants.
# ----------------------------------------------------------------------------
RHOSU = 85000.0 / (287.15 * 273.15)
RHOW = 997.0
RHOI = 500.0
RHOSN = 100.0
RHOG = 400.0        # IHAIL=0
AIMM = 0.66
BIMM = 100.0
ECR = 1.0
DCS = 125.0e-6
MI0 = 4.0 / 3.0 * PI * RHOI * (10.0e-6) ** 3
MG0 = 1.6e-10
F1S = 0.86
F2S = 0.28
F1R = 0.78
F2R = 0.308         # fix 053011
QSMALL = 1.0e-14
EII = 0.1
ECI = 0.7
CPW = 4187.0

# Size distribution parameters.
CI = RHOI * PI / 6.0
DI = 3.0
CS = RHOSN * PI / 6.0
DS = 3.0
CG = RHOG * PI / 6.0
DG = 3.0

RIN = 0.1e-6
MMULT = 4.0 / 3.0 * PI * RHOI * (5.0e-6) ** 3

# Size limits for lambda.
LAMMAXI = 1.0 / 1.0e-6
LAMMINI = 1.0 / (2.0 * DCS + 100.0e-6)
LAMMAXR = 1.0 / 20.0e-6
LAMMINR = 1.0 / 2800.0e-6
LAMMAXS = 1.0 / 10.0e-6
LAMMINS = 1.0 / 2000.0e-6
LAMMAXG = 1.0 / 20.0e-6
LAMMING = 1.0 / 2000.0e-6

# Aerosol activation params (IACT=2; only used when iinum=0 -> not in this path,
# but NANEW1/NANEW2 appear in the droplet-number upper-bound clause).
MW = 0.018
OSM = 1.0
VI = 3.0
EPSM = 0.7
RHOA = 1777.0
MAP = 0.132
MA = 0.0284
RR = 8.3145
BACT = VI * OSM * EPSM * MW * RHOA / (MAP * RHOW)
RM1 = 0.052e-6
SIG1 = 2.04
NANEW1 = 72.2e6
RM2 = 1.3e-6
SIG2 = 2.5
NANEW2 = 1.8e6

_g = math.gamma

# ----------------------------------------------------------------------------
# Efficiency constants CONS1..CONS41 (identical expressions to the init).
# ----------------------------------------------------------------------------
CONS1 = _g(1.0 + DS) * CS
CONS2 = _g(1.0 + DG) * CG
CONS3 = _g(4.0 + BS) / 6.0
CONS4 = _g(4.0 + BR) / 6.0
CONS5 = _g(1.0 + BS)
CONS6 = _g(1.0 + BR)
CONS7 = _g(4.0 + BG) / 6.0
CONS8 = _g(1.0 + BG)
CONS9 = _g(5.0 / 2.0 + BR / 2.0)
CONS10 = _g(5.0 / 2.0 + BS / 2.0)
CONS11 = _g(5.0 / 2.0 + BG / 2.0)
CONS12 = _g(1.0 + DI) * CI
CONS13 = _g(BS + 3.0) * PI / 4.0 * ECI
CONS14 = _g(BG + 3.0) * PI / 4.0 * ECI
CONS15 = -1108.0 * EII * PI ** ((1.0 - BS) / 3.0) * RHOSN ** ((-2.0 - BS) / 3.0) / (4.0 * 720.0)
CONS16 = _g(BI + 3.0) * PI / 4.0 * ECI
CONS17 = 4.0 * 2.0 * 3.0 * RHOSU * PI * ECI * ECI * _g(2.0 * BS + 2.0) / (8.0 * (RHOG - RHOSN))
CONS18 = RHOSN * RHOSN
CONS19 = RHOW * RHOW
CONS20 = 20.0 * PI * PI * RHOW * BIMM
CONS21 = 4.0 / (DCS * RHOI)
CONS22 = PI * RHOI * DCS ** 3 / 6.0
CONS23 = PI / 4.0 * EII * _g(BS + 3.0)
CONS24 = PI / 4.0 * ECR * _g(BR + 3.0)
CONS25 = PI * PI / 24.0 * RHOW * ECR * _g(BR + 6.0)
CONS26 = PI / 6.0 * RHOW
CONS27 = _g(1.0 + BI)
CONS28 = _g(4.0 + BI) / 6.0
CONS29 = 4.0 / 3.0 * PI * RHOW * (25.0e-6) ** 3
CONS30 = 4.0 / 3.0 * PI * RHOW
CONS31 = PI * PI * ECR * RHOSN
CONS32 = PI / 2.0 * ECR
CONS33 = PI * PI * ECR * RHOG
CONS34 = 5.0 / 2.0 + BR / 2.0
CONS35 = 5.0 / 2.0 + BS / 2.0
CONS36 = 5.0 / 2.0 + BG / 2.0
CONS37 = 4.0 * PI * 1.38e-23 / (6.0 * PI * RIN)
CONS38 = PI * PI / 3.0 * RHOW
CONS39 = PI * PI / 36.0 * RHOW * BIMM
CONS40 = PI / 6.0 * BIMM
CONS41 = PI * PI * ECR * RHOW
