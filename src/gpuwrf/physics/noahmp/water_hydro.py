"""Noah-MP Schaake96 soil hydrology + runoff (Sprint S4) — FREEZE STUB.

Ports WATER (module_sf_noahmplsm.F:5954-6261) restricted to the Schaake branch
(opt_run=3) and SOILWATER (:7234-7556), with opt_inf=1 / opt_frz=1 frozen-soil
treatment. NO GROUNDWATER (SIMGM/SIMTOP CUT).

Steps: canopy interception update (CANLIQ/CANICE/FWET), infiltration, Schaake96
surface + subsurface runoff, the Richards-like soil-moisture tridiagonal solve
(SOILWATER), and supercooled-liquid (SH2O <= SMC). Consumes the transpiration /
evaporation sinks from the energy step (S1) as soil-moisture withdrawals.

FULLY PARALLEL to author (savepoints supply the ET inputs as fixtures); integrates
serially with S1 (consumes its ET). Oracle = WRF SMC/SH2O/SMCWTD/SFCRUNOFF/UDRUNOFF
savepoint parity + a water-mass conservation check (the LH 18x over-flux must
collapse to ~1x once evaporation is soil-hydraulic + canopy-resistance limited).
"""

from __future__ import annotations

from gpuwrf.contracts.noahmp_state import NoahMPLandState, NoahMPStatic
from gpuwrf.physics.noahmp.types import NoahMPEtFluxes, NoahMPForcing


def noahmp_water_hydro(
    land_state: NoahMPLandState,
    forcing: NoahMPForcing,
    static: NoahMPStatic,
    et_fluxes: NoahMPEtFluxes,
    dt: float,
) -> NoahMPLandState:
    """Advance soil/canopy water one ``dt`` (Schaake96) — STUB.

    Returns the land carry with SMC/SH2O/SMCWTD/CANLIQ/CANICE/FWET/SFCRUNOFF/
    UDRUNOFF updated; thermal and snow fields untouched (those are S2/S3).
    """

    raise NotImplementedError("noahmp_water_hydro: Sprint S4 (Schaake96 hydrology + runoff)")


__all__ = ["noahmp_water_hydro"]
