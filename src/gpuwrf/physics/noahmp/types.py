"""FROZEN shared Noah-MP NamedTuples (ADR-NOAHMP-INTERFACES.md §3).

These are the data boundaries BETWEEN component sprints. They are frozen in
Sprint 0 so S1-S5 can be authored and oracle-tested in parallel without
touching each other's files. Field lists are binding; a component sprint that
needs an extra field must amend this file via the patch protocol, not silently.

All array fields are (ny, nx) on land columns unless noted; fp64; device-resident.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax


class NoahMPForcing(NamedTuple):
    """Atmosphere + radiation + clock forcing into Noah-MP (NOAHMP_SFLX IN args).

    Assembled by the coupler from the atmosphere lowest level, radiation
    (SOLDN/LWDN/COSZ), microphysics precip partition, and the run clock. All
    (ny, nx) on land columns except scalars ``julian``/``yearlen``.

    The trailing block (``pahv``..``foln``) is an ADDITIVE, backward-compatible
    amendment (Sprint S1 energy FIX, patch protocol): all default ``None`` so any
    pre-S1 caller continues to construct ``NoahMPForcing`` unchanged. S1 energy and
    its callers default them at the energy-function boundary (PAH=0 no-precip,
    O2/CO2 from STOMATA carbon block) when not supplied. S6/PRECIP_HEAT supplies
    the PAH* terms when precipitation is present; the coupler supplies o2air/co2air/
    foln from the WRF carbon block (``co2air = parameters%CO2 * SFCPRS`` etc.).
    """

    sfctmp: jax.Array     # lowest-level air temperature [K]
    sfcprs: jax.Array     # lowest-level air pressure [Pa]
    psfc: jax.Array       # surface pressure [Pa]
    uu: jax.Array         # lowest-level u wind [m/s]
    vv: jax.Array         # lowest-level v wind [m/s]
    qair: jax.Array       # lowest-level specific/mixing humidity [kg/kg]
    qc: jax.Array         # lowest-level cloud water [kg/kg]
    soldn: jax.Array      # downward shortwave at surface [W/m2]
    lwdn: jax.Array       # downward longwave at surface [W/m2]
    prcpconv: jax.Array   # convective precip rate [mm/s]
    prcpnonc: jax.Array   # non-convective (grid) precip rate [mm/s]
    prcpsnow: jax.Array   # snow precip rate [mm/s]
    prcpgrpl: jax.Array   # graupel precip rate [mm/s]
    prcphail: jax.Array   # hail precip rate [mm/s]
    cosz: jax.Array       # cosine solar zenith angle
    zlvl: jax.Array       # reference (forcing) height [m]
    julian: jax.Array     # day-of-year (scalar)
    yearlen: jax.Array    # days in year (scalar)
    # --- ADDITIVE (S1 energy FIX); default None -> energy boundary fills them ---
    pahv: Optional[jax.Array] = None   # precip-advected heat, vegetation net [W/m2]
    pahg: Optional[jax.Array] = None   # precip-advected heat, under-canopy ground [W/m2]
    pahb: Optional[jax.Array] = None   # precip-advected heat, bare ground [W/m2]
    o2air: Optional[jax.Array] = None  # atmos O2 partial pressure [Pa] (STOMATA)
    co2air: Optional[jax.Array] = None  # atmos CO2 partial pressure [Pa] (STOMATA)
    foln: Optional[jax.Array] = None   # foliage nitrogen concentration [%] (STOMATA)


class NoahMPRadInputs(NamedTuple):
    """Two-stream radiation outputs feeding the energy balance (in-module to S1).

    Produced inside ``energy`` (RADIATION/two-stream) and passed to VEGE_FLUX /
    BARE_FLUX. Exposed as a type so the energy sprint can unit-test the radiation
    sub-step against WRF savepoints independently.
    """

    sav: jax.Array        # solar radiation absorbed by vegetation [W/m2]
    sag: jax.Array        # solar radiation absorbed by ground [W/m2]
    parsun: jax.Array     # PAR absorbed by sunlit leaves [W/m2]
    parsha: jax.Array     # PAR absorbed by shaded leaves [W/m2]
    fsa: jax.Array        # total absorbed solar [W/m2]
    fsr: jax.Array        # reflected solar [W/m2]
    albedo: jax.Array     # broadband surface albedo (SALB)
    fsno: jax.Array       # snow-cover fraction


class NoahMPPhenology(NamedTuple):
    """Table-phenology outputs (Sprint S5, dveg=4)."""

    lai: jax.Array        # leaf area index [m2/m2]
    sai: jax.Array        # stem area index [m2/m2]
    elai: jax.Array       # exposed (snow-adjusted) LAI [m2/m2]
    esai: jax.Array       # exposed SAI [m2/m2]
    fveg: jax.Array       # green vegetation fraction (= SHDFAC, dveg=4)
    igs: jax.Array        # growing-season index


class NoahMPEnergyFluxes(NamedTuple):
    """Energy-balance outputs (Sprint S1). FSH is the HFX fix (FSH from TAH).

    ``canhs`` (ADDITIVE, default None) is the WRF canopy heat-storage change
    ``DTV*FVEG*HCV/DT`` (module_sf_noahmplsm.F:4062), required by the WRF energy
    closure check ``ERRENG`` (:1662). The coupler does not map it; it is returned
    for the closure proof and budget accounting.
    """

    fsh: jax.Array        # total sensible heat [W/m2] -> HFX
    fcev: jax.Array       # canopy evaporation latent heat [W/m2]
    fgev: jax.Array       # ground evaporation latent heat [W/m2]
    fctr: jax.Array       # transpiration latent heat [W/m2]
    ssoil: jax.Array      # ground heat flux [W/m2] -> GRDFLX
    fira: jax.Array       # net longwave [W/m2]
    trad: jax.Array       # radiative skin temperature [K] -> TSK
    emissi: jax.Array     # surface emissivity -> EMISS
    z0wrf: jax.Array      # combined roughness [m] -> ZNT
    chv: jax.Array        # canopy heat exchange coeff
    chb: jax.Array        # bare heat exchange coeff
    canhs: "jax.Array | None" = None  # canopy heat-storage change [W/m2] (closure)


class NoahMPEtFluxes(NamedTuple):
    """Evapotranspiration sinks passed from energy (S1) into water (S4)."""

    ecan: jax.Array       # canopy evaporation [kg/m2/s]
    etran: jax.Array      # transpiration [kg/m2/s]
    edir: jax.Array       # direct soil evaporation [kg/m2/s]
    qseva: jax.Array      # ground surface evaporation [kg/m2/s]
    btrani: jax.Array     # per-soil-layer transpiration factor (NSOIL, ny, nx)
    qsnow: jax.Array      # snowfall rate onto ground [mm/s]
    qmelt: jax.Array      # snowmelt rate [mm/s]
    imelt: jax.Array      # phase-change flag per snow+soil layer


__all__ = [
    "NoahMPForcing",
    "NoahMPRadInputs",
    "NoahMPPhenology",
    "NoahMPEnergyFluxes",
    "NoahMPEtFluxes",
]
