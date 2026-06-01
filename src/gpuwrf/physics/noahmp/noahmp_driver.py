"""Noah-MP top-level driver (Sprint S6, serial-last) — INTERFACE FREEZE STUB.

Ports the NOAHMP_SFLX orchestration (module_sf_noahmplsm.F:450-1079) for the
scoped LAND-ONLY configuration. No physics here at the freeze; this file pins
the component call ORDER that S6 wires once S1-S5 have merged + passed their
oracle gates.

Call order (WRF NOAHMP_SFLX, scoped):
  1. phenology_table   -> LAI/SAI/ELAI/ESAI/FVEG/IGS               (S5)
  2. precip_heat       -> rain/snow partition (opt_snf=1) + QSNOW  (folded into driver)
  3. energy_canopy     -> FSH/FCEV/FGEV/FCTR/SSOIL/TRAD/.../ET     (S1)
       (energy calls soil_thermo (S2) internally as TSNOSOI for the STC update)
  4. snow              -> ISNOW/SNICE/SNLIQ/SNOWH/SNEQV/ZSNSO      (S3)
  5. water_hydro       -> SMC/SH2O/SMCWTD/runoff/canopy water      (S4, consumes ET from S1)

LAND-ONLY: ocean/lake columns are masked out upstream by the coupler
(``physics.noahmp_coupler``); only land columns reach this driver.
"""

from __future__ import annotations

from gpuwrf.contracts.noahmp_state import NoahMPFluxes, NoahMPLandState, NoahMPStatic
from gpuwrf.physics.noahmp.types import NoahMPForcing


def noah_mp_step(
    land_state: NoahMPLandState,
    forcing: NoahMPForcing,
    static: NoahMPStatic,
    dt: float,
) -> tuple[NoahMPLandState, NoahMPFluxes]:
    """One Noah-MP physics-timestep over all land columns — STUB.

    Pure functional pytree-in / pytree-out; no host/device transfer (GPU rule).
    ``dt`` is the physics (long) timestep in seconds. Returns the advanced land
    carry and the per-step fluxes the coupler blends into the land path.
    """

    raise NotImplementedError("noah_mp_step: Sprint S6 driver orchestration (serial-last)")


__all__ = ["noah_mp_step"]
