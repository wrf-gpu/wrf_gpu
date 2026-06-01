"""Noah-MP table phenology (Sprint S5, dveg=4) — FREEZE STUB.

Ports PHENOLOGY (module_sf_noahmplsm.F:1255-1358) for the dveg=4 branch: monthly
LAI/SAI tables interpolated by Julian day and vegetation category, snow-burial
adjustment to exposed ELAI/ESAI, FVEG = SHDFAC, and the growing-season index IGS.
No dynamic vegetation / carbon (dveg=4 -> table-driven only).

FULLY PARALLEL: depends only on frozen ``types`` + the parameter tables (Sprint
0b). Smallest component; may be folded into the S1 worker. Oracle = table-
interpolation parity (LAI/SAI/ELAI/ESAI/FVEG) vs WRF for a category x julian sweep.
"""

from __future__ import annotations

from gpuwrf.contracts.noahmp_state import NoahMPLandState, NoahMPStatic
from gpuwrf.physics.noahmp.types import NoahMPForcing, NoahMPPhenology


def noahmp_phenology_table(
    land_state: NoahMPLandState,
    forcing: NoahMPForcing,
    static: NoahMPStatic,
) -> NoahMPPhenology:
    """Table phenology for dveg=4 — STUB.

    Returns LAI/SAI/ELAI/ESAI/FVEG/IGS. Reads SNOWH from ``land_state`` for the
    snow-burial adjustment and the Julian day from ``forcing``.
    """

    raise NotImplementedError("noahmp_phenology_table: Sprint S5 (table phenology, dveg=4)")


__all__ = ["noahmp_phenology_table"]
