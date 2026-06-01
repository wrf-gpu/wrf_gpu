"""Noah-MP snow water / compaction / albedo aging (Sprint S3) — FREEZE STUB.

Ports SNOWWATER (module_sf_noahmplsm.F:6398-6535) and the snow-albedo aging
(opt_alb=2 CLASS, opt_snf=1 Jordan91 partition) for NSNOW=3, ISNOW in {-2,-1,0}.

Updates the snow-layer state: snowfall accumulation, compaction, layer
combine/divide, sublimation/frost, and melt routing -> ISNOW/SNICE/SNLIQ/SNOWH/
SNEQV/ZSNSO + snow age TAUSS and prior albedo ALBOLD.

FULLY PARALLEL: depends only on frozen ``types``; oracle = WRF snow-column
savepoint parity + a snow-water-equivalent conservation check.
"""

from __future__ import annotations

import jax

from gpuwrf.contracts.noahmp_state import NoahMPLandState, NoahMPStatic
from gpuwrf.physics.noahmp.types import NoahMPForcing


def noahmp_snow(
    land_state: NoahMPLandState,
    forcing: NoahMPForcing,
    static: NoahMPStatic,
    qsnow: jax.Array,
    imelt: jax.Array,
    qmelt: jax.Array,
    dt: float,
) -> NoahMPLandState:
    """Advance the snow column one ``dt`` (SNOWWATER + aging) — STUB.

    ``qsnow`` snowfall onto ground [mm/s], ``imelt`` per-layer phase-change flag,
    ``qmelt`` snowmelt rate [mm/s] (all from the energy step). Returns the land
    carry with snow fields updated; all other fields untouched.
    """

    raise NotImplementedError("noahmp_snow: Sprint S3 (SNOWWATER + albedo aging)")


__all__ = ["noahmp_snow"]
