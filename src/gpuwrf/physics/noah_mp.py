"""Prescribed Noah-MP subset for M6-S3 Option A.

This module intentionally does not implement prognostic Noah-MP. It packages
the Gen2 `wrfinput_d02` Noah-MP state fields needed by the surface-layer lower
boundary and applies only bounds/diagnostic derivations.
"""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

from dataclasses import dataclass
from typing import Any

from jax import config
import jax.numpy as jnp

from gpuwrf.physics.surface_constants import DEFAULT_LAND_ROUGHNESS_M, DEFAULT_WATER_ROUGHNESS_M


configure_jax_x64()


@dataclass(frozen=True)
class PrescribedNoahMPState:
    """Static/time-slice Noah-MP lower-boundary state."""

    t_skin: object
    soil_moisture: object
    soil_liquid: object
    soil_temperature: object
    xland: object
    landmask: object
    lakemask: object
    ivgtyp: object
    isltyp: object
    lu_index: object
    sst: object
    roughness_m: object
    mavail: object
    source: dict[str, Any]


# ``run/LANDUSE.TBL`` stores summer and winter columns. WRF chooses ``ISN``
# before the first surface-layer call (module_physics_init.F:1833-1835,
# 1958-1972). Keeping summer unconditionally changes winter land drag before
# Noah-MP has advanced even once.
_MODIFIED_IGBP_MODIS_NOAH_SFZ0_SUMMER_M = jnp.asarray(
    [
        DEFAULT_WATER_ROUGHNESS_M,  # category 0 is mapped to water by WRF before lookup.
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.05,
        0.06,
        0.05,
        0.15,
        0.12,
        0.3,
        0.15,
        0.8,
        0.14,
        0.001,
        0.01,
        0.0001,
        0.3,
        0.15,
        0.1,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
        0.8,
    ],
    dtype=jnp.float64,
)

_MODIFIED_IGBP_MODIS_NOAH_SFZ0_WINTER_M = jnp.asarray(
    [
        DEFAULT_WATER_ROUGHNESS_M,
        0.5, 0.5, 0.5, 0.5, 0.2,
        0.01, 0.01, 0.01, 0.15, 0.10,
        0.30, 0.05, 0.8, 0.05, 0.001,
        0.01, 0.0001, 0.30, 0.15, 0.05,
        *([0.8] * 30),
        *([0.8] * 11),
    ],
    dtype=jnp.float64,
)

_MODIFIED_IGBP_MODIS_NOAH_SLMO_SUMMER = jnp.asarray(
    [
        1.0,
        0.3,
        0.5,
        0.3,
        0.3,
        0.3,
        0.1,
        0.15,
        0.1,
        0.15,
        0.15,
        0.42,
        0.3,
        0.1,
        0.25,
        0.95,
        0.02,
        1.0,
        0.5,
        0.5,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
    ],
    dtype=jnp.float64,
)

_MODIFIED_IGBP_MODIS_NOAH_SLMO_WINTER = jnp.asarray(
    [
        1.0,
        0.60, 0.50, 0.60, 0.60, 0.60,
        0.20, 0.25, 0.20, 0.15, 0.30,
        0.725, 0.60, 0.10, 0.40, 0.95,
        0.05, 1.0, 0.60, 0.60, 0.05,
        *([0.02] * 30),
        *([0.10] * 11),
    ],
    dtype=jnp.float64,
)


def _seasonal_landuse_table(summer, winter, season: int):
    """Select WRF ``LANDUSE.TBL`` column ``ISN`` (1 summer, 2 winter)."""

    value = int(season)
    if value == 1:
        return summer
    if value == 2:
        return winter
    raise ValueError(
        f"WRF landuse season must be 1 (summer) or 2 (winter), got {season!r}"
    )


def _modified_igbp_modis_noah_lookup(lu_index, table):
    cat = jnp.rint(jnp.asarray(lu_index, dtype=jnp.float64)).astype(jnp.int32)
    cat = jnp.where(cat == 0, 17, cat)
    valid = (cat >= 1) & (cat < table.shape[0])
    return table[jnp.clip(cat, 0, table.shape[0] - 1)], valid


def roughness_from_prescribed_fields(
    xland,
    landmask,
    vegfra=None,
    cm=None,
    lu_index=None,
    *,
    season: int = 1,
):
    """Return a bounded roughness surrogate from prescribed Gen2 fields.

    When ``LU_INDEX`` is available for the MODIFIED_IGBP_MODIS_NOAH table, WRF
    cold-initialises ``ZNT`` from seasonal ``LANDUSE.TBL`` ``SFZ0/100`` before
    the first MYNN surface-layer call. ``season`` is WRF ``ISN`` (1 summer,
    2 winter). Older smoke callers without land-use categories keep the legacy
    CM/VEGFRA fallback.
    """

    xland = jnp.asarray(xland, dtype=jnp.float64)
    landmask = jnp.asarray(landmask, dtype=jnp.float64)
    if cm is not None:
        cm = jnp.asarray(cm, dtype=jnp.float64)
        usable_cm = cm > 1.0e-6
        neutral_z0 = 10.0 * jnp.exp(-0.40 / jnp.sqrt(jnp.maximum(cm, 1.0e-6)))
    else:
        usable_cm = jnp.zeros_like(xland, dtype=bool)
        neutral_z0 = jnp.zeros_like(xland, dtype=jnp.float64)
    if vegfra is None:
        land_z0 = jnp.ones_like(xland, dtype=jnp.float64) * DEFAULT_LAND_ROUGHNESS_M
    else:
        veg = jnp.clip(jnp.asarray(vegfra, dtype=jnp.float64) / 100.0, 0.0, 1.0)
        land_z0 = 0.02 + 0.18 * veg
    water_z0 = jnp.ones_like(xland, dtype=jnp.float64) * DEFAULT_WATER_ROUGHNESS_M
    surrogate = jnp.where((xland > 1.5) | (landmask < 0.5), water_z0, land_z0)
    fallback = jnp.where(usable_cm, neutral_z0, surrogate)
    if lu_index is not None:
        table = _seasonal_landuse_table(
            _MODIFIED_IGBP_MODIS_NOAH_SFZ0_SUMMER_M,
            _MODIFIED_IGBP_MODIS_NOAH_SFZ0_WINTER_M,
            season,
        )
        table_z0, valid = _modified_igbp_modis_noah_lookup(lu_index, table)
        fallback = jnp.where(valid, table_z0, fallback)
    return jnp.clip(fallback, 1.0e-7, 10.0)


def mavail_from_prescribed_fields(
    xland, landmask, smois, lu_index=None, *, season: int = 1,
):
    """Return WRF cold-start surface moisture availability for prescribed land."""

    xland = jnp.asarray(xland, dtype=jnp.float64)
    landmask = jnp.asarray(landmask, dtype=jnp.float64)
    smois = jnp.asarray(smois, dtype=jnp.float64)
    top_soil = smois[0] if smois.ndim == 3 else smois
    fallback = jnp.where((xland > 1.5) | (landmask < 0.5), 1.0, top_soil)
    if lu_index is not None:
        table = _seasonal_landuse_table(
            _MODIFIED_IGBP_MODIS_NOAH_SLMO_SUMMER,
            _MODIFIED_IGBP_MODIS_NOAH_SLMO_WINTER,
            season,
        )
        table_mavail, valid = _modified_igbp_modis_noah_lookup(lu_index, table)
        fallback = jnp.where(valid, table_mavail, fallback)
    return jnp.clip(fallback, 0.0, 1.0)


def prescribe_noah_mp_state(
    *,
    t_skin,
    smois,
    sh2o,
    tslb,
    xland,
    landmask,
    lakemask,
    ivgtyp,
    isltyp,
    lu_index,
    sst,
    vegfra=None,
    cm=None,
    season: int = 1,
    source: dict[str, Any] | None = None,
) -> PrescribedNoahMPState:
    """Package bounded Gen2 land state for the sfclay lower boundary."""

    t_skin = jnp.clip(jnp.asarray(t_skin, dtype=jnp.float64), 180.0, 340.0)
    smois = jnp.clip(jnp.asarray(smois, dtype=jnp.float64), 0.0, 1.0)
    sh2o = jnp.clip(jnp.asarray(sh2o, dtype=jnp.float64), 0.0, 1.0)
    tslb = jnp.clip(jnp.asarray(tslb, dtype=jnp.float64), 180.0, 340.0)
    xland = jnp.asarray(xland, dtype=jnp.float64)
    landmask = jnp.asarray(landmask, dtype=jnp.float64)
    lakemask = jnp.asarray(lakemask, dtype=jnp.float64)
    lu_index = jnp.asarray(lu_index)
    roughness = roughness_from_prescribed_fields(
        xland, landmask, vegfra=vegfra, cm=cm, lu_index=lu_index,
        season=season,
    )
    mavail = mavail_from_prescribed_fields(
        xland, landmask, smois, lu_index=lu_index, season=season,
    )
    return PrescribedNoahMPState(
        t_skin=t_skin,
        soil_moisture=smois,
        soil_liquid=sh2o,
        soil_temperature=tslb,
        xland=xland,
        landmask=landmask,
        lakemask=lakemask,
        ivgtyp=jnp.asarray(ivgtyp),
        isltyp=jnp.asarray(isltyp),
        lu_index=lu_index,
        sst=jnp.asarray(sst, dtype=jnp.float64),
        roughness_m=roughness,
        mavail=mavail,
        source={} if source is None else dict(source),
    )


__all__ = [
    "PrescribedNoahMPState",
    "mavail_from_prescribed_fields",
    "prescribe_noah_mp_state",
    "roughness_from_prescribed_fields",
]
