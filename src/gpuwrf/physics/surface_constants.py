"""Constants for the WRF revised surface layer (``module_sf_sfclayrev.F``).

All values are cited to the WRF source on this workstation:
``/home/enric/src/wrf_pristine/WRF/phys/physics_mmm/sf_sfclayrev.F90`` (the CCPP
``sf_sfclayrev_run`` core that ``module_sf_sfclayrev.F`` delegates to) and the
namelist/registry constants WRF passes into ``sfclayrev`` from
``module_physics_init`` / ``module_model_constants``.

This replaces the prior MM5 ``sfclay`` constants used by the FAILED M12 surface
attempt. The revised scheme (Jimenez et al. 2012, MWR) uses Cheng & Brutsaert
(2005, CB05) integrated similarity functions and a bulk-Richardson Newton solve
(``zolri``) rather than the MM5 regime algebra.
"""

from __future__ import annotations


# --- thermodynamic / model constants (WRF module_model_constants.F) ---
# Passed into sfclayrev as cp, g, rovcp, r, xlv, p1000mb (sf_sfclayrev.F90:78-105).
CP_D = 1004.0          # cp
G = 9.81               # g (WRF uses 9.81, not 9.80665, in physics)
KARMAN = 0.4           # karman
P0_PA = 100000.0       # p1000mb
R_D = 287.0            # r (gas_constant)
R_D_OVER_CP = R_D / CP_D  # rovcp
XLV = 2.5e6            # xlv latent heat of vaporization

# Virtual-temperature coefficients (WRF ep_1 = R_v/R_d - 1 = 0.608, ep_2 = R_d/R_v).
EP1 = 0.608            # ep1 (passed as ep_1)
EP2 = 0.622            # ep2 = R_d/R_v
P608 = EP1             # alias used in virtual-temperature forms

# --- saturation-vapor-pressure constants (WRF SVP1/SVP2/SVP3/SVPT0) ---
# e1 = svp1*exp(svp2*(T-svpt0)/(T-svp3)); svp1 is in kPa. (sf_sfclayrev.F90:281,288)
SVP1_KPA = 0.6112
SVP2 = 17.67
SVP3_K = 29.65
SVPT0_K = 273.15

# --- sfclayrev module parameters (sf_sfclayrev.F90:12-14, 175-177) ---
VCONVC = 1.0
# MYNN surface layer (module_sf_mynn.F:83) uses a LARGER convective velocity-scale
# coefficient than sfclayrev. The MYNN-SL JAX path (surface_layer.py, the operational
# sf_sfclay_physics=5 scheme) MUST use this value; the sfclayrev-family schemes
# (sfclay_revised_mm5, sfclay_pleim_xiu) keep VCONVC=1.0.
VCONVC_MYNN = 1.25
CZO = 0.0185
OZO = 1.59e-5
XKA = 2.4e-5           # molecular thermal diffusivity for psiq (parameter xka)
PRT = 1.0              # turbulent Prandtl number (parameter prt)
SALINITY_FACTOR = 0.98  # salty-water saturation reduction (parameter salinity_factor)

# --- CB05 stability-function lookup table layout (sf_sfclayrev.F90:16,39-49) ---
# psim_stab/psih_stab/psim_unstab/psih_unstab are tabulated for n=0..1000 with a
# z/L step of 0.01 (zolf = n*0.01 stable, -n*0.01 unstable). Linear interpolation
# between table nodes; for |z/L| >= 10 the analytic "full" form is used.
SFCLAYREV_TABLE_N = 1000
SFCLAYREV_TABLE_DZOL = 0.01
SFCLAYREV_TABLE_ZOL_MAX = SFCLAYREV_TABLE_N * SFCLAYREV_TABLE_DZOL  # 10.0

# --- zolri Newton/secant iteration (sf_sfclayrev.F90:922-957) ---
# Fixed 10-iteration secant solve for z/L from bulk Richardson number. WRF caps Ri
# magnitude at 250 before the solve (sf_sfclayrev.F90:382-398).
ZOLRI_MAX_ITER = 10
ZOLRI_BR_CAP = 250.0

# Defaults used only when a prescribed surface field is absent (e.g. analytic
# smoke states). Real Canary cases supply xland/roughness/mavail from wrfinput.
DEFAULT_LAND_ROUGHNESS_M = 0.10
DEFAULT_WATER_ROUGHNESS_M = 2.85e-3
DEFAULT_DX_M = 3000.0
DEFAULT_PBLH_M = 1000.0
MIN_WIND_M_S = 0.1     # wspd floor (sf_sfclayrev.F90:353)


__all__ = [
    "CP_D",
    "CZO",
    "DEFAULT_DX_M",
    "DEFAULT_LAND_ROUGHNESS_M",
    "DEFAULT_PBLH_M",
    "DEFAULT_WATER_ROUGHNESS_M",
    "EP1",
    "EP2",
    "G",
    "KARMAN",
    "MIN_WIND_M_S",
    "OZO",
    "P0_PA",
    "P608",
    "PRT",
    "R_D",
    "R_D_OVER_CP",
    "SALINITY_FACTOR",
    "SFCLAYREV_TABLE_DZOL",
    "SFCLAYREV_TABLE_N",
    "SFCLAYREV_TABLE_ZOL_MAX",
    "SVP1_KPA",
    "SVP2",
    "SVP3_K",
    "SVPT0_K",
    "VCONVC",
    "VCONVC_MYNN",
    "XKA",
    "XLV",
    "ZOLRI_BR_CAP",
    "ZOLRI_MAX_ITER",
]
