"""Noah-MP canopy/ground energy balance (Sprint S1, PRIORITY) — FREEZE STUB.

THE HFX-FIX COMPONENT. Ports NOAHMP_SFLX::ENERGY (module_sf_noahmplsm.F:1741-2396):
  - two-stream radiation (opt_rad) -> SAV/SAG/PAR/albedo  (RADIATION, in-module)
  - VEGE_FLUX (:3578-4170): canopy + canopy-air + ground energy balance over the
    vegetated tile -> solves TV (canopy), TAH (canopy-air), TG (ground); emits the
    sensible/latent partition. CANRES (:5141-5223, opt_crs=1 Ball-Berry) supplies
    stomatal resistance for transpiration.
  - BARE_FLUX (:4174-4479): bare-tile ground energy balance.
  - FVEG-weighted tile sum -> FSH/FCEV/FGEV/FCTR/SSOIL/TRAD/EMISSI/Z0WRF.
  - calls soil_thermo (Sprint S2, TSNOSOI) internally for the semi-implicit STC update.

WHY THIS IS THE FIX (proofs/v010_validation/hfx_overflux_root_cause.json):
  WRF HFX = FSH = rho*cpm*CH*(TAH - SFCTMP) from the CANOPY-AIR temperature TAH,
  which is ~5.5 K cooler than the radiative skin TSK at midday over dry sparse-veg
  land. The v0.1.0 bulk path used the radiative TSK and over-fluxed by exactly the
  gradient ratio 1.40. Solving the canopy-air energy balance for TAH closes it.

Inputs CH/CM come from sfclay (opt_sfc=1); this component consumes, not recomputes them.
"""

from __future__ import annotations

import jax

from gpuwrf.contracts.noahmp_state import NoahMPLandState, NoahMPStatic
from gpuwrf.physics.noahmp.types import (
    NoahMPEnergyFluxes,
    NoahMPEtFluxes,
    NoahMPForcing,
    NoahMPPhenology,
    NoahMPRadInputs,
)


def noahmp_radiation_twostream(
    land_state: NoahMPLandState,
    forcing: NoahMPForcing,
    static: NoahMPStatic,
    phen: NoahMPPhenology,
) -> NoahMPRadInputs:
    """Two-stream canopy radiation transfer (in-module to S1) — STUB.

    Exposed so S1 can unit-oracle-test the radiation sub-step (SAV/SAG/albedo)
    against WRF savepoints independently of the flux solve.
    """

    raise NotImplementedError("noahmp_radiation_twostream: Sprint S1 (two-stream)")


def noahmp_energy_canopy(
    land_state: NoahMPLandState,
    forcing: NoahMPForcing,
    static: NoahMPStatic,
    rad: NoahMPRadInputs,
    dt: float,
) -> tuple[NoahMPLandState, NoahMPEnergyFluxes, NoahMPEtFluxes]:
    """Canopy/ground surface-energy balance — STUB (THE HFX FIX).

    Solves TV/TAH/TG energy balances (VEGE_FLUX + BARE_FLUX), FVEG-weighted tile
    sum, and the internal semi-implicit STC update (via ``soil_thermo``). Returns
    the energy-updated land carry (tv/tg/tah/eah/tslb/tsno/t_skin/emiss/albedo/znt),
    the energy fluxes (FSH->HFX, latent partition, SSOIL->GRDFLX, TRAD->TSK), and
    the ET sinks consumed by ``water_hydro`` (S4).
    """

    raise NotImplementedError("noahmp_energy_canopy: Sprint S1 (canopy energy balance / HFX fix)")


__all__ = ["noahmp_radiation_twostream", "noahmp_energy_canopy"]
