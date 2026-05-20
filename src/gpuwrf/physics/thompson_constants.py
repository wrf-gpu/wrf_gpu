"""WRF Thompson constants used by the M5-S1 column subset."""

from __future__ import annotations

import math


# Source: module_mp_thompson.F.pre lines 64-89, 183-225, 660-667.
T_0 = 273.15
PI = 3.1415926536
RHO_W = 1000.0
RHO_I = 890.0
RHO_G_MP8 = 400.0
NT_C = 100.0e6
NT_C_MAX = 1999.0e6
R1 = 1.0e-12
R2 = 1.0e-6
EPS = 1.0e-15
HGFR = 235.16
RV = 461.5
R_D = 287.04
CP = 1004.0
RHO_NOT = 101325.0 / (287.05 * 298.0)
LSUB = 2.834e6
LVAP0 = 2.5e6
LFUS = LSUB - LVAP0
XM0I = 1.0e-12
D0C = 1.0e-6
D0R = 50.0e-6
D0S = 300.0e-6
D0G = 350.0e-6
AM_R = PI * RHO_W / 6.0
AM_I = PI * RHO_I / 6.0
D0I = (XM0I / AM_I) ** (1.0 / 3.0)
AM_G_MP8 = PI * RHO_G_MP8 / 6.0

# Source: module_mp_thompson.F.pre lines 101-168, 670-725, 786-817.
BM_R = 3.0
BM_I = 3.0
BM_G = 3.0
MU_R = 0.0
MU_I = 0.0
MU_G = 0.0
OBMR = 1.0 / BM_R
OBMI = 1.0 / BM_I
OBMG = 1.0 / BM_G
AV_R = 4854.0
BV_R = 1.0
FV_R = 195.0
AV_S = 40.0
BV_S = 0.55
FV_S = 100.0
AV_G_MP8 = 143.204224
BV_G_MP8 = 0.640961647
SC = 0.632
SC3 = SC ** (1.0 / 3.0)
C_CUBE = 0.5
C_SQRD = 0.15
CRG2 = 1.0
CRG3 = 6.0
CRG9 = 6.0
CRG10 = 1.0
CRG11 = 2.0
ORG2 = 1.0 / CRG2
ORG3 = 1.0 / CRG3
CRE2 = 1.0
CRE9 = 4.0
CRE10 = 2.0
CRE11 = 3.0
CIG1 = 1.0
CIG2 = 6.0
CIG5 = 1.0
OIG1 = 1.0 / CIG1
OIG2 = 1.0 / CIG2
NU_C_MP8 = 12.0
CCG1_NU12 = 479001600.0
CCG2_NU12 = 1307674368000.0
CCG3_NU12 = 6402373705728000.0
OCG1_NU12 = 1.0 / CCG1_NU12
OCG2_NU12 = 1.0 / CCG2_NU12
T1_QR_QC = PI * 0.25 * AV_R * CRG9
T1_QR_EV = 0.78 * CRG10
T2_QR_EV = 0.308 * SC3 * math.sqrt(AV_R) * CRG11
T1_SUBL_QS = 0.86
T2_SUBL_QS = 0.28 * SC3 * math.sqrt(AV_S)
T1_MELT_QS = PI * 4.0 * C_SQRD / LFUS * 0.86
T2_MELT_QS = PI * 4.0 * C_SQRD / LFUS * 0.28 * SC3 * math.sqrt(AV_S)
T1_SUBL_QG = 0.86
T2_SUBL_QG = 0.28 * SC3 * math.sqrt(AV_G_MP8) * 2.0
T1_MELT_QG = PI * 4.0 * C_CUBE / LFUS * 0.86
T2_MELT_QG = PI * 4.0 * C_CUBE / LFUS * 0.28 * SC3 * math.sqrt(AV_G_MP8) * 2.0


def constant_table() -> dict[str, float]:
    """Returns scalar constants for tests and ADR/report generation."""

    names = (
        "T_0",
        "PI",
        "RHO_W",
        "RHO_I",
        "RHO_G_MP8",
        "NT_C",
        "NT_C_MAX",
        "R1",
        "R2",
        "EPS",
        "HGFR",
        "RV",
        "R_D",
        "CP",
        "RHO_NOT",
        "LSUB",
        "LVAP0",
        "LFUS",
        "XM0I",
        "D0C",
        "D0R",
        "D0S",
        "D0G",
        "AM_R",
        "AM_I",
        "D0I",
        "AM_G_MP8",
        "BM_R",
        "BM_I",
        "BM_G",
        "OBMR",
        "OBMI",
        "OBMG",
        "AV_R",
        "FV_R",
        "SC3",
        "C_CUBE",
        "C_SQRD",
        "T1_QR_QC",
        "T1_QR_EV",
        "T2_QR_EV",
    )
    return {name: float(globals()[name]) for name in names}


def assert_finite_constants() -> None:
    """Fails early if a transcribed scalar is not finite and positive where required."""

    for name, value in constant_table().items():
        if not math.isfinite(value):
            raise ValueError(f"{name} is not finite")
        if name not in {"EPS"} and value <= 0.0:
            raise ValueError(f"{name} must be positive")
