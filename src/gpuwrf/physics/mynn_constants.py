"""Constants for the M5-S2 MYNN2.5 column kernel."""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

from jax import config


configure_jax_x64()


# WRF MYNN constants are declared in module_bl_mynnedmf.F90 lines 278-309.
PR = 0.74
G1 = 0.235
B1 = 24.0
B2 = 15.0
C2 = 0.729
C3 = 0.340
C4 = 0.0
C5 = 0.2
A1 = B1 * (1.0 - 3.0 * G1) / 6.0
C1 = G1 - 1.0 / (3.0 * A1 * 2.88449914061481660)
A2 = A1 * (G1 - C1) / (G1 * PR)
G2 = B2 / B1 * (1.0 - C3) + 2.0 * A1 / B1 * (3.0 - 2.0 * C2)
CC2 = 1.0 - C2
CC3 = 1.0 - C3
E1C = 3.0 * A2 * B2 * CC3
E2C = 9.0 * A1 * A2 * CC2
E3C = 9.0 * A2 * A2 * CC2 * (1.0 - C5)
E4C = 12.0 * A1 * A2 * CC2
E5C = 6.0 * A1 * A1

QMIN = 0.0
ZMAX = 1.0
SQFAC = 3.0
QKEMIN = 1.0e-5
TKE_EPS = 0.5 * QKEMIN
CKMOD = 1.0

# Shared thermodynamic constants imported by module_bl_mynn.F90 lines 257-267.
CP = 1004.0
GRAV = 9.81
KARMAN = 0.4
R_D = 287.04
TREF = 300.0
P608 = 0.608
GTR = GRAV / TREF

# MYNN local mixing-length option 2 constants from module_bl_mynnedmf.F90
# lines 2221-2350. M5-S2 uses this bounded local form with EDMF terms disabled.
LOCAL_CNS = 3.5   # surface-layer length coeff (cns) in stable conditions
LOCAL_ALP1 = 0.22
LOCAL_ALP2 = 0.30
LOCAL_ALP3 = 2.5
LOCAL_ALP4 = 5.0  # surface-layer length coeff (alp4) in unstable conditions
LOCAL_ELT_MIN = 10.0
LOCAL_ELT_MAX = 400.0
LOCAL_ELF_SOFT_MAX = 800.0
MIN_PBLH = 300.0
MAX_PBLH_TRANSITION = 600.0
CTAU = 1000.0

# MYNN NONLOCAL mixing-length option 1 (bl_mynn_mixlength=1, the WRF v4.7.1
# default and what the v0.9.0 oracle run used; see namelist.output
# BL_MYNN_MIXLENGTH=1). module_bl_mynnedmf.F:1753-1870 (CASE 1) + boulac_length
# (lines 2192-2338). Uses the BouLac mixing length in the free atmosphere.
NL_CNS = 3.5
NL_ALP1 = 0.23
NL_ALP2 = 0.30
NL_ALP3 = 2.5   # scaled by wt_u2 (hurricane shear taper; =2.5 for U<=20 m/s)
NL_ALP4 = 5.0
NL_ALP5 = 0.30  # weight on BouLac length in the free atmosphere
NL_ALP6 = 50.0  # mass-flux length coeff (EDMF off -> qkw_mf=0 -> unused)
NL_QKW_ELB_MIN = 0.18  # qkw_elb_min floor on qkw in elb/elf (stable)
NL_ELT_MIN = 8.0
NL_ELT_MAX = 400.0        # elt_max over LAND (WRF module_bl_mynnedmf.F:1804)
NL_ELT_MAX_WATER = 350.0  # elt_max base over WATER (WRF :1802; +100*hurricane)
NL_UONSET = 20.0      # hurricane-taper wind onset (m/s)
# Surface-layer stability-function coefficients for the MYNN TKE surface source
# (WRF module_bl_mynnedmf.F:275-276, 7714). cphm_unst feeds the unstable phim.
CPHM_UNST = 16.0
NL_BOULAC_LMAX = 1500.0  # BouLac soft length limit

# Bulk surface-layer stub constants. The real MYNN surface layer remains outside
# M5-S2 scope; these coefficients are conventional neutral bulk values.
BULK_CD = 1.3e-3
BULK_CH = 1.2e-3
BULK_CQ = 1.2e-3
MIN_WIND = 0.2
