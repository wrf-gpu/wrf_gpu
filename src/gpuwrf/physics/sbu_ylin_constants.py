"""Module-level constants for the SBU-YLin microphysics (WRF mp_physics=13).

Faithful transcription of the ``PARAMETER`` block at the top of
``phys/module_mp_sbu_ylin.F`` (Lin et al. 1983 / Rutledge-Hobbs 1984 base with
the Stony-Brook-University Y. Lin snow/ice-Richardson and Liu-Daum
autoconversion modifications). Values are byte-for-byte the Fortran source;
derived gamma-function prefactors that the column kernel computes once from
``ggamma`` are precomputed here so the hot path carries no scalar gamma loop.
"""

from __future__ import annotations

import math

# --- user-tunable / PSD parameters (module_mp_sbu_ylin.F:22-24) ---------------
RH = 1.0
XNOR = 8.0e6          # rain intercept N0r [m^-4]
NT_C = 10.0e6         # fixed cloud-droplet number concentration [m^-3]

# --- gas constants / thermodynamics (lines 26-43) -----------------------------
RVAPOR = 461.5
ORV = 1.0 / RVAPOR
RAIR = 287.04
CP = 1004.0
GRAV = 9.81
RHOWATER = 1000.0
RHOSNOW = 100.0

SVP1 = 0.6112
SVP2 = 17.67
SVP3 = 29.65
SVPT0 = 273.15
EP1 = RVAPOR / RAIR - 1.0
EP2 = RAIR / RVAPOR

XLS = 2.834e6
XLV = 2.5e6
XLF = XLS - XLV

# --- microphysical scalars (lines 46-54) --------------------------------------
QI0 = 1.0e-3          # ice->snow aggregation threshold [kg/kg]
XMI50 = 4.8e-10       # mass of a 50 micron ice crystal [kg]
XMI40 = 2.46e-10      # mass of a 40 micron ice crystal [kg]
XNI0 = 1.0e-2
XMNIN = 1.05e-18      # mass of a natural ice nucleus [kg]
BNI = 0.5
DI50 = 1.0e-4         # diameter of a 50 micron ice crystal [m]
XMI = 4.19e-13        # mass of one cloud-ice crystal [kg]
BV_R = 0.8            # rain fall-speed exponent
BV_I = 0.25           # ice fall-speed exponent
O6 = 1.0 / 6.0
CDRAG = 0.6
AVISC = 1.49628e-6
ADIFFWV = 8.7602e-5
AXKA = 1.4132e3
CW = 4.187e3
CI = 2.093e3

# --- snow size-distribution coefficients (Heymsfield 2007, lines 313-318) -----
AM_C1, AM_C2, AM_C3 = 0.004, 6e-5, 0.15
BM_C1, BM_C2, BM_C3 = 1.85, 0.003, 1.25
AA_C1, AA_C2, AA_C3 = 1.28, -0.012, -0.6
BA_C1, BA_C2, BA_C3 = 1.5, 0.0075, 0.5
BEST_A, BEST_B = 1.08, 0.499

MU_S = 0.0
MU_I = 0.0
MU_R = 0.0

# --- ventilation factors (line 311) -------------------------------------------
VF1S, VF2S = 0.65, 0.44
VF1R, VF2R = 0.78, 0.31

QVMIN = 1.0e-20


def _ggamma(x: float) -> float:
    """Hastings polynomial gamma approximation -- EXACTLY the Fortran ``ggamma``.

    Reproduces ``module_mp_sbu_ylin.F:1713`` bit-pattern (reduction by integer
    steps to (1,2], then an 8-term Hastings series). Used only at module import
    to precompute constant-argument gamma values; the hot path never calls it.
    """

    b = (
        -0.577191652, 0.988205891, -0.897056937, 0.918206857,
        -0.756704078, 0.482199394, -0.193527818, 0.035868343,
    )
    pf = 1.0
    temp = x
    # DO J=1,200: IF (TEMP <= 2) exit; TEMP=TEMP-1; PF=PF*TEMP
    for _ in range(200):
        if temp <= 2.0:
            break
        temp = temp - 1.0
        pf = pf * temp
    g1to2 = 1.0
    temp = temp - 1.0
    for k1 in range(1, 9):
        g1to2 = g1to2 + b[k1 - 1] * temp ** k1
    return pf * g1to2


# Precomputed constant-argument gamma prefactors used by the column kernel.
GAMBP4 = _ggamma(BV_R + 4.0)
GAMDP4 = _ggamma(BV_I + 4.0)
GAMBP3 = _ggamma(BV_R + 3.0)
GAMBP6 = _ggamma(BV_R + 6.0)
GAMBP5O2 = _ggamma((BV_R + 5.0) / 2.0)
GAMDP5O2 = _ggamma((BV_I + 5.0) / 2.0)

# Rain/ice fall-speed prefactors (lines 365-366): av_r=2115*0.01**(1-bv_r) etc.
AV_R = 2115.0 * 0.01 ** (1.0 - BV_R)
AV_I = 152.93 * 0.01 ** (1.0 - BV_I)

# Liu-Daum autoconversion mean cloud-droplet shape (line 351).
MU_C = min(15.0, 1000.0e6 / NT_C + 2.0)
R6C = 10.0e-6
# gamma(4+mu_c)/gamma(1+mu_c) and gamma(7+mu_c)/gamma(1+mu_c) appear in lamc/Dc.
GGAMMA_4PMUC = _ggamma(4.0 + MU_C)
GGAMMA_1PMUC = _ggamma(1.0 + MU_C)
GGAMMA_7PMUC = _ggamma(6.0 + 1.0 + MU_C)

PI = math.acos(-1.0)
O6_AVR_GAMBP4 = O6 * AV_R * GAMBP4

# Bergeron-process tabulated crystal-growth coefficients (parama1/parama2).
PARAMA1_TABLE = (
    0.100e-10, 0.7939e-7, 0.7841e-6, 0.3369e-5, 0.4336e-5,
    0.5285e-5, 0.3728e-5, 0.1852e-5, 0.2991e-6, 0.4248e-6,
    0.7434e-6, 0.1812e-5, 0.4394e-5, 0.9145e-5, 0.1725e-4,
    0.3348e-4, 0.1725e-4, 0.9175e-5, 0.4412e-5, 0.2252e-5,
    0.9115e-6, 0.4876e-6, 0.3473e-6, 0.4758e-6, 0.6306e-6,
    0.8573e-6, 0.7868e-6, 0.7192e-6, 0.6513e-6, 0.5956e-6,
    0.5333e-6, 0.4834e-6,
)
PARAMA2_TABLE = (
    0.0100, 0.4006, 0.4831, 0.5320, 0.5307, 0.5319, 0.5249,
    0.4888, 0.3849, 0.4047, 0.4318, 0.4771, 0.5183, 0.5463,
    0.5651, 0.5813, 0.5655, 0.5478, 0.5203, 0.4906, 0.4447,
    0.4126, 0.3960, 0.4149, 0.4320, 0.4506, 0.4483, 0.4460,
    0.4433, 0.4413, 0.4382, 0.4361,
)

__all__ = [name for name in dir() if not name.startswith("_")]
