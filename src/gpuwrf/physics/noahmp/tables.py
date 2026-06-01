"""Noah-MP parameter table loader (Sprint 0b) — INTERFACE FREEZE STUB.

Loads per-category lookup tables once per run from the WRF MPTABLE/SOILPARM/
GENPARM equivalents and bundles them as a frozen ``NoahMPParameters`` pytree
indexed by ``ivgtyp``/``isltyp`` (ADR §2.5). Prerequisite for energy/water/
phenology oracle tests. No host transfer at runtime: tables are device-resident
after load and gathered per column by category index.
"""

from __future__ import annotations

from typing import NamedTuple

import jax


class NoahMPParameters(NamedTuple):
    """FROZEN Noah-MP parameter bundle (per-category, gathered by index at use).

    Field set is the scoped subset (dveg=4, opt_crs=1, opt_run=3 Schaake); carbon /
    groundwater / crop parameters are CUT. Vegetation params are indexed by
    ``ivgtyp``; soil params by ``isltyp``; general params are scalars.
    """

    # vegetation (indexed by ivgtyp)
    rhol: jax.Array       # leaf reflectance (vis, nir)
    rhos: jax.Array       # stem reflectance
    taul: jax.Array       # leaf transmittance
    taus: jax.Array       # stem transmittance
    xl: jax.Array         # leaf orientation index
    rgl: jax.Array        # CANRES radiation stress parameter
    rsmin: jax.Array      # minimum stomatal resistance [s/m]
    hs: jax.Array         # CANRES vapor-pressure-deficit parameter
    rsmax: jax.Array      # maximum stomatal resistance [s/m]
    z0mvt: jax.Array      # momentum roughness by veg type [m]
    hvt: jax.Array        # canopy top height [m]
    hvb: jax.Array        # canopy bottom height [m]
    saim: jax.Array       # monthly stem area index (12)
    laim: jax.Array       # monthly leaf area index (12)
    sla: jax.Array        # specific leaf area
    shdfac: jax.Array     # green vegetation fraction (FVEG, dveg=4)
    # soil (indexed by isltyp)
    bexp: jax.Array       # Clapp-Hornberger b
    smcmax: jax.Array     # saturated (porosity) soil moisture
    smcref: jax.Array     # reference (field capacity) soil moisture
    smcwlt: jax.Array     # wilting-point soil moisture
    smcdry: jax.Array     # dry soil moisture
    dksat: jax.Array      # saturated hydraulic conductivity [m/s]
    dwsat: jax.Array      # saturated soil diffusivity
    psisat: jax.Array     # saturated matric potential [m]
    quartz: jax.Array     # soil quartz content
    # general (scalars)
    csoil: jax.Array      # soil heat capacity [J/m3/K]
    zbot: jax.Array       # deep-soil temperature depth [m]
    czil: jax.Array       # Zilitinkevich thermal-roughness coefficient


def load_noahmp_parameters(table_dir, *, scope_options: dict) -> NoahMPParameters:
    """Load and bundle Noah-MP parameter tables — STUB.

    ``table_dir`` points at the WRF parameter-table directory; ``scope_options``
    is the frozen iopt map (recorded into the restart, ADR §5). Sprint 0b body.
    """

    raise NotImplementedError("load_noahmp_parameters: Sprint 0b table loader")


__all__ = ["NoahMPParameters", "load_noahmp_parameters"]
