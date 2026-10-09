"""Constants for the M5-S3 RRTMG column kernels."""

from __future__ import annotations

import os
import numpy as np

_RRTMG_REAL_CONSTANTS = os.environ.get("GPUWRF_RRTMG_REAL_CONSTANTS", "0").strip().lower() in {"1", "true", "yes", "on"}

CP_AIR = 1004.0
GRAVITY = 9.80665
RD_AIR = 287.04
RV_OVER_RD_MINUS_ONE = 0.608
SECONDS_PER_DAY = 86400.0
SOLAR_CONSTANT = 1368.22
STEFAN_BOLTZMANN = 5.670374419e-8
AVOGADRO = 6.02214199e23
DRY_AIR_MOLECULAR_WEIGHT = 28.9660
WATER_VAPOR_MOLECULAR_WEIGHT_RATIO = 1.607793
CO2_VMR = 431.3824884728998e-6
CH4_VMR = 1774.0e-9
N2O_VMR = 319.0e-9
O2_VMR = 0.209488
O3_BACKGROUND_VMR = 8.0e-8

WRF_RRTMG_SW_BANDS = 14
WRF_RRTMG_LW_BANDS = 16
RRTMG_TOTAL_BANDS = WRF_RRTMG_SW_BANDS + WRF_RRTMG_LW_BANDS

MIN_COSZEN = 0.0
MIN_LAYER_MASS = 1.0e-6
MIN_OPTICAL_DEPTH = 1.0e-10
MAX_OPTICAL_DEPTH = 80.0

LW_DIFFUSIVITY_A0 = (
    1.66,
    1.55,
    1.58,
    1.66,
    1.54,
    1.454,
    1.89,
    1.33,
    1.668,
    1.66,
    1.66,
    1.66,
    1.66,
    1.66,
    1.66,
    1.66,
)
LW_DIFFUSIVITY_A1 = (
    0.0,
    0.25,
    0.22,
    0.0,
    0.13,
    0.446,
    -0.10,
    0.40,
    -0.006,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)
LW_DIFFUSIVITY_A2 = (
    0.0,
    -12.0,
    -11.7,
    0.0,
    -0.72,
    -0.243,
    0.19,
    -0.062,
    0.414,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
)
LW_TBLINT = 10000.0
LW_NTBL = 10000
LW_PADE = 0.278
LW_BPADE = 1.0 / LW_PADE
LW_EXP_EPS = 1.0e-20

_f = np.float32
_WRF_REAL_CONSTANTS = {
    "LW_BPADE": _f(1) / _f(.278),
    "GRAVITY": _f(9.8066),
    "CP_AIR": _f(7) * _f(287) / _f(2),
}
_WRF_HEATFAC = _f(9.8066) * _f(86400) / (_WRF_REAL_CONSTANTS["CP_AIR"] * _f(100))


def rrtmg_constant(name, dtype, legacy=None):
    if _RRTMG_REAL_CONSTANTS and np.dtype(dtype) == np.dtype(np.float32):
        return _WRF_REAL_CONSTANTS[name]
    return globals()[name] if legacy is None else legacy


def rrtmg_heating_rate(delta_flux, layer_mass, pressure_interfaces=None, *, shortwave=False):
    """Original REAL heatfac/pressure order; preserve the retained formula off."""
    # SW keeps its internal flux stream REAL even for the retained f64 entry.
    # Layer mass retains the entry's pressure precision and owns this choice.
    if not _RRTMG_REAL_CONSTANTS or np.dtype(layer_mass.dtype) != np.dtype(np.float32):
        return delta_flux / (layer_mass * CP_AIR)
    import jax.numpy as jnp
    from jax import lax
    if pressure_interfaces is None:
        raise ValueError("WRF REAL heating needs the original pressure interfaces")
    pz = jnp.asarray(pressure_interfaces, jnp.float32) / lax.optimization_barrier(jnp.asarray(100, jnp.float32))
    dp = pz[..., :-1] - pz[..., 1:]
    if shortwave:
        heating = delta_flux * lax.optimization_barrier(_WRF_HEATFAC / dp)
    else:
        heating = lax.optimization_barrier(_WRF_HEATFAC * delta_flux) / dp
    return lax.optimization_barrier(heating) / lax.optimization_barrier(jnp.asarray(86400, jnp.float32))


def rrtmg_cloud_water_path(q, layer_mass, pressure_interfaces=None, *, entry_dtype=None):
    """WRF driver cloud mass uses model g=9.81, distinct from RRTMG grav."""
    dtype = layer_mass.dtype if entry_dtype is None else entry_dtype
    if not _RRTMG_REAL_CONSTANTS or np.dtype(dtype) != np.dtype(np.float32):
        return q * layer_mass * 1000.0
    import jax.numpy as jnp
    from jax import lax
    if pressure_interfaces is None:
        raise ValueError("WRF cloud path needs its original pressure interfaces")
    barrier = lax.optimization_barrier
    pz = jnp.asarray(pressure_interfaces, jnp.float32) / barrier(jnp.float32(100))
    pdel = barrier(pz[..., :-1] - pz[..., 1:])
    # module_ra_rrtmg_{lw,sw}: q*pdel*100/gravmks*1000; gravmks=g.
    product = barrier(jnp.asarray(q, jnp.float32) * pdel)
    product = barrier(jnp.nextafter(product, product))
    product = barrier(product * jnp.float32(100))
    return barrier(product / barrier(jnp.float32(9.81))) * jnp.float32(1000)
