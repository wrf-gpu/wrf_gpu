"""Prescribed Noah-MP subset for M6-S3 Option A.

This module intentionally does not implement prognostic Noah-MP. It packages
the Gen2 `wrfinput_d02` Noah-MP state fields needed by the surface-layer lower
boundary and applies only bounds/diagnostic derivations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jax import config
import jax.numpy as jnp

from gpuwrf.physics.surface_constants import DEFAULT_LAND_ROUGHNESS_M, DEFAULT_WATER_ROUGHNESS_M


config.update("jax_enable_x64", True)


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


def roughness_from_prescribed_fields(xland, landmask, vegfra=None, cm=None):
    """Return a bounded roughness surrogate from prescribed Gen2 fields.

    Direct `ZNT` is absent in the pinned `wrfinput_d02`; ADR-012 records this
    deviation. `CM` is present but all zeros in the local fixture, so vegetation
    fraction and land/water masks provide the operational v0 surrogate.
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
    return jnp.clip(jnp.where(usable_cm, neutral_z0, surrogate), 1.0e-7, 10.0)


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
    roughness = roughness_from_prescribed_fields(xland, landmask, vegfra=vegfra, cm=cm)
    top_soil = smois[0] if smois.ndim == 3 else smois
    mavail = jnp.clip(jnp.where((xland > 1.5) | (landmask < 0.5), 1.0, top_soil), 0.0, 1.0)
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
        lu_index=jnp.asarray(lu_index),
        sst=jnp.asarray(sst, dtype=jnp.float64),
        roughness_m=roughness,
        mavail=mavail,
        source={} if source is None else dict(source),
    )


__all__ = ["PrescribedNoahMPState", "prescribe_noah_mp_state", "roughness_from_prescribed_fields"]
