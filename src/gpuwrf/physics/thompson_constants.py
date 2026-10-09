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
TNO = 5.0
ATO = 0.304
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
AM_S = 0.069
AM_I = PI * RHO_I / 6.0
D0I = (XM0I / AM_I) ** (1.0 / 3.0)
AM_G_MP8 = PI * RHO_G_MP8 / 6.0

# Source: module_mp_thompson.F.pre lines 101-168, 670-725, 786-817.
BM_R = 3.0
BM_S = 2.0
BM_I = 3.0
BM_G = 3.0
MU_R = 0.0
MU_S = 0.6357
MU_I = 0.0
MU_G = 0.0
CIE2 = BM_I + MU_I + 1.0
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
CRG8 = math.gamma(BM_R + MU_R + BV_R + 3.0)
CRG9 = 6.0
CRG10 = 1.0
CRG11 = 2.0
ORG1 = 1.0 / CRG3
ORG2 = 1.0 / CRG2
ORG3 = 1.0 / CRG3
CRE1 = 4.0
CRE2 = 1.0
CRE9 = 4.0
CRE10 = 2.0
CRE11 = 3.0
# Sedimentation fall-speed gamma exponents/coefficients for rain and cloud ice.
# WRF module_mp_thompson.F:705-725 (cre/crg) and :687-702 (cie/cig), with
# mp=8 values mu_r=0, bm_r=3, bv_r=1, mu_i=0, bm_i=3, bv_i=1.
BV_R = 1.0  # WRF module_mp_thompson.F:144
BV_I = 1.0  # WRF module_mp_thompson.F:162
AV_I = 1493.9  # WRF module_mp_thompson.F:161
CRE3 = BM_R + MU_R + 1.0  # = 4
CRE6 = BM_R + MU_R + BV_R + 1.0  # = 5
CRE7 = BM_R * 0.5 + MU_R + BV_R + 1.0  # = 3.5
CRE8 = BM_R + MU_R + BV_R + 3.0  # = 7
CRE12 = BM_R * 0.5 + MU_R + 1.0  # = 2.5
CRG3_SED = math.gamma(CRE3)  # = 6 (== crg(3))
CRG6 = math.gamma(CRE6)  # = 24
CRG7 = math.gamma(CRE7)  # = Gamma(3.5)
CRG12 = math.gamma(CRE12)  # = Gamma(2.5)
CIE3 = BM_I + MU_I + BV_I + 1.0  # = 5
CIE6 = BM_I * 0.5 + MU_I + BV_I + 1.0  # = 3.5
CIE7 = BM_I * 0.5 + MU_I + 1.0  # = 2.5
CIG3 = math.gamma(CIE3)  # = 24 (cig(3))
CIG6 = math.gamma(CIE6)  # = Gamma(3.5) (cig(6))
CIG7 = math.gamma(CIE7)  # = Gamma(2.5) (cig(7))
CIG2_GAMMA = math.gamma(BM_I + MU_I + 1.0)  # = Gamma(4) = 6 (cig(2))
OIG2_SED = 1.0 / CIG2_GAMMA
# WRF module_mp_thompson.F.pre lines 104, 156, 763, and 767.
MU_G_MP8 = MU_G
CGE11 = 0.5 * (BV_G_MP8 + 5.0 + 2.0 * MU_G_MP8)
CGG11 = math.gamma(CGE11)
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
# Cloud-water terminal fall speed (WRF module_mp_thompson.F:163-164, 3658-3664).
# av_c/bv_c are the cloud droplet fall-speed coefficient/exponent; the two gamma
# closures ccg(4,nu_c)/ccg(5,nu_c) for the number/mass-weighted moments use the
# mp=8 default nu_c=12 (== NU_C_MP8 at nc=NT_C: NINT(1000e6/100e6)+2 = 12).
# cce(4,12)=n+bv_c+1=15 -> ccg(4,12)=Gamma(15); cce(5,12)=bm_r+n+bv_c+1=18 ->
# ccg(5,12)=Gamma(18).
AV_C = 0.316946e8  # WRF module_mp_thompson.F:163
BV_C = 2.0  # WRF module_mp_thompson.F:164
CCG4_NU12 = math.gamma(NU_C_MP8 + BV_C + 1.0)  # Gamma(15) = 87178291200.0
CCG5_NU12 = math.gamma(BM_R + NU_C_MP8 + BV_C + 1.0)  # Gamma(18) = 355687428096000.0
T1_QR_QC = PI * 0.25 * AV_R * CRG9
T1_QR_EV = 0.78 * CRG10
T2_QR_EV = 0.308 * SC3 * math.sqrt(AV_R) * CRG11
T1_SUBL_QS = 0.86
T2_SUBL_QS = 0.28 * SC3 * math.sqrt(AV_S)
T1_MELT_QS = PI * 4.0 * C_SQRD / LFUS * 0.86
T2_MELT_QS = PI * 4.0 * C_SQRD / LFUS * 0.28 * SC3 * math.sqrt(AV_S)
T1_SUBL_QG = 0.86
T2_SUBL_QG = 0.28 * SC3 * math.sqrt(AV_G_MP8) * CGG11
T1_MELT_QG = PI * 4.0 * C_CUBE / LFUS * 0.86
T2_MELT_QG = PI * 4.0 * C_CUBE / LFUS * 0.28 * SC3 * math.sqrt(AV_G_MP8) * CGG11

# ---- WRF cloud-ice collection by snow/rain (module_mp_thompson.F:2710-2734) ----
EF_SI = 0.05
EF_RI = 0.95
T1_QS_QI = PI * 0.25 * AV_S
T1_QR_QI = PI * 0.25 * AV_R * CRG9
T2_QR_QI = PI * 0.25 * AM_R * AV_R * CRG8

# ---- v0.15 cold-phase riming (WRF module_mp_thompson.F:2403-2440, 2758-2776) ----
# Snow collecting cloud water: prs_scw = rhof * t1_qs_qc * Ef_sw * rc * smoe,
# with t1_qs_qc = PI*0.25*av_s (line 794; the snow gamma moments live in smoe).
T1_QS_QC = PI * 0.25 * AV_S
# Graupel collecting cloud water (mu_g=0 single-density mp8 convention):
# prg_gcw = rhof * t1_qg_qc * Ef_gw * rc * N0_g * ilamg**cge(9), with
# cge(9) = bv_g + 3 + mu_g and t1_qg_qc = PI*0.25*av_g*cgg(9) (lines 766/2432).
CGE9 = BV_G_MP8 + 3.0 + MU_G
CGG9 = math.gamma(CGE9)
T1_QG_QC = PI * 0.25 * AV_G_MP8 * CGG9
# Mass-weighted graupel fall speed for the Stokes number (vtg of line 2421):
# vtg = rhof*av_g*Gamma(bv_g+mu_g+4)/Gamma(mu_g+4) * ilamg**bv_g.
CGG6_OVER_CGG3 = math.gamma(BV_G_MP8 + MU_G + 4.0) / math.gamma(MU_G + 4.0)
# WRF mp8/mp28 graupel: thompson_init without ng (physics_init :4521) replaces av_g/bv_g(idx_bg1) by
# av_g_old/bv_g_old (:459-464) BEFORE cge/cgg are built (:758); AV_G_MP8/BV_G_MP8 are the hail-aware
# (mp38) table entry. Native REAL uses these (thompson_column._graupel_constants).
AV_G_OLD = 442.0
BV_G_OLD = 0.89
CGE9_OLD = BV_G_OLD + 3.0 + MU_G
CGE11_OLD = 0.5 * (BV_G_OLD + 5.0 + 2.0 * MU_G_MP8)
CGG6_OVER_CGG3_OLD = math.gamma(BV_G_OLD + MU_G + 4.0) / math.gamma(MU_G + 4.0)
T1_QG_QC_OLD = PI * 0.25 * AV_G_OLD * math.gamma(CGE9_OLD)
T2_SUBL_QG_OLD = 0.28 * SC3 * math.sqrt(AV_G_OLD) * math.gamma(CGE11_OLD)
T2_MELT_QG_OLD = PI * 4.0 * C_CUBE / LFUS * 0.28 * SC3 * math.sqrt(AV_G_OLD) * math.gamma(CGE11_OLD)
RHO_W_RIME = 1000.0  # liquid water density in the graupel Stokes number
# t_Efsw snow-bin geometry (WRF lines 862-872): nbs=100 log bins from D0s=300um
# to 2 cm, Ds(n) = sqrt(xDx(n)*xDx(n+1)); the lookup uses Ds(1)/Ds(nbs).
NBS_EFSW = 100
_XDX_S_FIRST = 300.0e-6
_XDX_S_LAST = 0.02
_S_BIN_RATIO = math.exp(math.log(_XDX_S_LAST / _XDX_S_FIRST) / NBS_EFSW)
DS_FIRST = math.sqrt(_XDX_S_FIRST * (_XDX_S_FIRST * _S_BIN_RATIO))
DS_LAST = math.sqrt((_XDX_S_LAST / _S_BIN_RATIO) * _XDX_S_LAST)


# WRF's own REAL values of the Thompson mp8 constants the release (native REAL) path consumes, where the f64 folds
# above differ. WRF folds PARAMETERs and runs thompson_init / the mp_thompson constant sub-expressions in REAL, one
# rounding per operation, and builds every gamma constant with its own WGAMMA = EXP(GAMMLN(x)) (Numerical-Recipes
# Lanczos, GAMMLN rounded to REAL; module_mp_thompson.F:5325/:5371), not the exact gamma function. IEEE bits from
# gfortran 14.3 with the pristine FCOPTIM compiling WRF's declarations/statements and GAMMLN/WGAMMA verbatim
# (proofs/thompson/wrf_constants/wrf_constants.F90 -> wrf_constants.txt). Every value is exactly representable in
# REAL, so DOUBLE contexts see DBLE(REAL) as in WRF. Selected by
# thompson_column._wrf() under GPUWRF_THOMPSON_WRF_CONSTANTS (BD94); MP_RE takes AM_R / AM_I.
WRF_REAL_CONSTANTS = {
    "AM_R": 523.5988159179688,  # bits 1141040723, :128 am_r = PI*rho_w/6.0
    "AM_I": 466.0029296875,  # bits 1139343456, :137 am_i = PI*rho_i/6.0
    "D0I": 1.289843385166023e-05,  # bits 928540250, :666 D0i = (xm0i/am_i)**(1./bm_i)
    "CRG7": 3.3233511447906494,  # bits 1079292361, :719 crg(7) = WGAMMA(3.5)
    "CIG6": 3.3233511447906494,  # bits 1079292361, :699 cig(6) = WGAMMA(3.5)
    "CCG5_NU12": 355687448182784.0,  # bits 1470218104, :682 ccg(5,12) = WGAMMA(18.)
    "OCG2_NU12": 7.647166320318144e-13,  # bits 727138212, :684 ocg2(12) = 1./ccg(2,12)
    "T1_QR_QC": 22873.9375,  # bits 1186116576, :786 PI*.25*av_r * crg(9)
    "T1_QR_QI": 22873.9375,  # bits 1186116576, :787 PI*.25*av_r * crg(9)
    "T2_QR_QI": 1437212032.0,  # bits 1319851067, :788 PI*.25*am_r*av_r * crg(8)
    "T1_QS_QC": 31.41592788696289,  # bits 1106990034, :794 PI*.25*av_s
    "T1_QS_QI": 31.41592788696289,  # bits 1106990034, :797 PI*.25*av_s
    "T1_MELT_QS": 4.853478912991704e-06,  # bits 916642577, :808 PI*4.*C_sqrd*olfus * 0.86
    "T2_MELT_QS": 8.576598702347837e-06,  # bits 923788342, :809 PI*4.*C_sqrd*olfus * 0.28*Sc3*SQRT(av_s)
    "CGE9_OLD": 3.8899998664855957,  # bits 1081669058, :762 cge(9,idx_bg1) = mu_g + bv_g_old + 3.
    "CGG6_OLD": 20.36322784423828,  # bits 1101195236, :767 cgg(6,idx_bg1) = WGAMMA(cge(6,idx_bg1))
    "OGG3": 0.1666666716337204,  # bits 1042983595, :779 ogg3 = 1./cgg(3,1)
    "T1_QG_QC_OLD": 1817.2269287109375,  # bits 1155737411, :2432 PI*.25*av_g(idx_bg1) * cgg(9,idx_bg1)
    "T2_MELT_QG_OLD": 0.00018076917331200093,  # bits 960335067, :2808 PI*4.*C_cube*olfus * 0.28*Sc3*SQRT(av_g)*cgg(11)
    "LAMC_PREFIX": 142942333304832.0,  # bits 1459749211, :1833/:2173/:3659 nc*am_r*ccg(2,12)*ocg1(12), nc = Nt_c
    "DC_G_PREFIX": 16.98038101196289,  # bits 1099421650, :2181 (ccg(3,12)*ocg2(12))**obmr
    "D0R_X075": 3.749999814317562e-05,  # bits 941443409, :1894 D0r*0.75
    "D0S_X01": 3.000000106112566e-05,  # bits 939239555, :2670 0.1*D0s
    "PNR_WAU_DEN": 7.853981465189008e-09,  # bits 839314989, :2191 am_r*nu_c*10.*D0r*D0r*D0r
}


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
        "TNO",
        "ATO",
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
        "AM_S",
        "AM_I",
        "D0I",
        "AM_G_MP8",
        "BM_R",
        "BM_S",
        "BM_I",
        "BM_G",
        "CIE2",
        "CGE11",
        "CGG11",
        "OBMR",
        "OBMI",
        "OBMG",
        "ORG1",
        "AV_R",
        "FV_R",
        "SC3",
        "C_CUBE",
        "C_SQRD",
        "CRE1",
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
