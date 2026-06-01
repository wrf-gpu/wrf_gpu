"""Prognostic Noah-MP land-surface model (v0.2.0 P0-3) — INTERFACE FREEZE.

Package layout (see ``.agent/decisions/ADR-NOAHMP-INTERFACES.md`` §3):
  - ``types``         shared FROZEN NamedTuples (forcing, rad inputs, et, phenology, energy fluxes)
  - ``tables``        Noah-MP parameter table loader (Sprint 0b)
  - ``noahmp_driver`` top-level ``noah_mp_step`` orchestration (Sprint S6, serial-last)
  - ``energy``        canopy/ground energy balance — THE HFX FIX (Sprint S1, priority)
  - ``soil_thermo``   semi-implicit snow/soil temperature (Sprint S2)
  - ``snow``          snow water/compaction/aging (Sprint S3)
  - ``water_hydro``   Schaake96 soil hydrology + runoff (Sprint S4)
  - ``phenology``     table phenology, dveg=4 (Sprint S5)

All component bodies raise ``NotImplementedError`` at the freeze; signatures are
binding boundaries that each sprint implements and oracle-tests independently.
"""

from __future__ import annotations

__all__ = [
    "noahmp_driver",
    "energy",
    "soil_thermo",
    "snow",
    "water_hydro",
    "phenology",
    "tables",
    "types",
]
