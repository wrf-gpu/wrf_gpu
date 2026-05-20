"""WRF Thompson constants used by the M5-S1 column subset."""

from __future__ import annotations

import math


# Source: module_mp_thompson.F.pre lines 64-89, 183-225, 660-667.
T_0 = 273.15
PI = 3.1415926536
RHO_W = 1000.0
RHO_I = 890.0
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
AM_R = PI * RHO_W / 6.0
AM_I = PI * RHO_I / 6.0
D0I = (XM0I / AM_I) ** (1.0 / 3.0)


def constant_table() -> dict[str, float]:
    """Returns scalar constants for tests and ADR/report generation."""

    names = (
        "T_0",
        "PI",
        "RHO_W",
        "RHO_I",
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
        "AM_R",
        "AM_I",
        "D0I",
    )
    return {name: float(globals()[name]) for name in names}


def assert_finite_constants() -> None:
    """Fails early if a transcribed scalar is not finite and positive where required."""

    for name, value in constant_table().items():
        if not math.isfinite(value):
            raise ValueError(f"{name} is not finite")
        if name not in {"EPS"} and value <= 0.0:
            raise ValueError(f"{name} must be positive")
