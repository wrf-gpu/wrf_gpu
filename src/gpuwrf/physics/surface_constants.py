"""Constants shared by the M6 MM5 sfclay surface-layer implementation."""

from __future__ import annotations


CP_D = 1004.0
G = 9.80665
KARMAN = 0.40
P0_PA = 100000.0
P608 = 0.608
R_D = 287.0
R_D_OVER_CP = R_D / CP_D
XLV = 2.5e6

MIN_WIND_M_S = 0.1
MIN_ROUGHNESS_M = 1.0e-7
MAX_ROUGHNESS_M = 10.0
DEFAULT_LAND_ROUGHNESS_M = 0.10
DEFAULT_WATER_ROUGHNESS_M = 2.85e-3
DEFAULT_DX_M = 3000.0
DEFAULT_PBLH_M = 1000.0

SVP1_KPA = 0.6112
SVP2 = 17.67
SVP3_K = 29.65
SVPT0_K = 273.15
EP2 = 0.622
SALINITY_FACTOR = 0.98
THERMAL_DIFFUSIVITY_M2_S = 2.4e-5

# WRF module_sf_sfclay.F:5-7.
VCONVC = 1.0
CZO = 0.0185
OZO = 1.59e-5


__all__ = [
    "CP_D",
    "CZO",
    "DEFAULT_DX_M",
    "DEFAULT_LAND_ROUGHNESS_M",
    "DEFAULT_PBLH_M",
    "DEFAULT_WATER_ROUGHNESS_M",
    "EP2",
    "G",
    "KARMAN",
    "MAX_ROUGHNESS_M",
    "MIN_ROUGHNESS_M",
    "MIN_WIND_M_S",
    "OZO",
    "P0_PA",
    "P608",
    "R_D",
    "R_D_OVER_CP",
    "SALINITY_FACTOR",
    "SVP1_KPA",
    "SVP2",
    "SVP3_K",
    "SVPT0_K",
    "THERMAL_DIFFUSIVITY_M2_S",
    "VCONVC",
    "XLV",
]
