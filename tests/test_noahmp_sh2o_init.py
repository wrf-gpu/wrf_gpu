"""Unit test for the NOAHMP_INIT SH2O reconstruction (cold-start fix).

The Gen2 corpus wrfinput is PRE-NOAHMP_INIT: its SH2O over land is written as 0,
so a raw load reads warm soil as ALL-ICE and the first PHASECHANGE step "melts"
phantom ice -> craters TSLB -> overnight cold-start transient. ``_noahmp_init_sh2o``
reconstructs SH2O from SMOIS/TSLB exactly per module_sf_noahmpdrv.F:2069-2106.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from gpuwrf.io.noahmp_land_init import _noahmp_init_sh2o, _HLICE, _GRAV_INIT, _T0


class _Params:
    # 1-based per-soil-type tables (index 0 unused); soil type 6 (loam-ish).
    bexp = jnp.array([0.0, 2.79, 4.26, 4.74, 5.33, 3.86, 5.25])
    smcmax = jnp.array([0.0, 0.339, 0.421, 0.434, 0.476, 0.484, 0.439])
    psisat = jnp.array([0.0, 0.069, 0.036, 0.141, 0.759, 0.955, 0.355])  # positive mag
    isice = 15


def test_warm_soil_sh2o_equals_smois():
    # warm soil (TSLB > 273.149) => SH2O == SMOIS (all liquid), regardless of
    # the (pre-init zero) corpus SH2O.
    nsoil, ny, nx = 4, 3, 2
    smois = jnp.full((nsoil, ny, nx), 0.13)
    tslb = jnp.full((nsoil, ny, nx), 295.0)
    isltyp = jnp.full((ny, nx), 6, dtype=jnp.int32)
    ivgtyp = jnp.full((ny, nx), 5, dtype=jnp.int32)  # not glacier
    sh2o = np.asarray(_noahmp_init_sh2o(smois, tslb, isltyp, ivgtyp, _Params()))
    assert np.allclose(sh2o, 0.13), sh2o
    assert np.all(np.isfinite(sh2o))


def test_frozen_soil_sh2o_is_fk_capped_and_nan_safe():
    # cold soil (TSLB < 273.149) => SH2O = min(FK, SMOIS) <= SMOIS, finite.
    nsoil, ny, nx = 4, 2, 2
    smois = jnp.full((nsoil, ny, nx), 0.30)
    tslb = jnp.full((nsoil, ny, nx), 268.0)
    isltyp = jnp.full((ny, nx), 6, dtype=jnp.int32)
    ivgtyp = jnp.full((ny, nx), 5, dtype=jnp.int32)
    sh2o = np.asarray(_noahmp_init_sh2o(smois, tslb, isltyp, ivgtyp, _Params()))
    assert np.all(np.isfinite(sh2o))
    assert np.all(sh2o <= 0.30 + 1e-9)
    assert np.all(sh2o >= 0.0)
    # cross-check FK against the WRF closed form for soil type 6.
    bexp, smcmax, psisat = 5.25, 0.439, 0.355
    fk = ((_HLICE / (_GRAV_INIT * (-psisat))) * ((268.0 - _T0) / 268.0)) ** (-1.0 / bexp) * smcmax
    fk = max(fk, 0.02)
    assert np.allclose(sh2o, min(fk, 0.30)), (sh2o.flat[0], min(fk, 0.30))


def test_glacier_tile_sh2o_zero():
    nsoil, ny, nx = 4, 2, 2
    smois = jnp.full((nsoil, ny, nx), 0.5)
    tslb = jnp.full((nsoil, ny, nx), 295.0)
    isltyp = jnp.full((ny, nx), 6, dtype=jnp.int32)
    ivgtyp = jnp.full((ny, nx), 15, dtype=jnp.int32)  # ISICE
    sh2o = np.asarray(_noahmp_init_sh2o(smois, tslb, isltyp, ivgtyp, _Params()))
    assert np.allclose(sh2o, 0.0), sh2o
