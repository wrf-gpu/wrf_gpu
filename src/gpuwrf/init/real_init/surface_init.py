"""S2 (GPT) — surface / skin / SST / coordinate / map-metric initial state.

FROZEN ENTRY SIGNATURE. Reproduces the real.exe surface handling
(module_initialize_real.F): the map-factor / Coriolis / lat-lon fields are
carried from the met_em artifact (S2 validates the passthrough), and the
skin-temperature / SST / TMN protection logic is reproduced faithfully:

  * SST sanity + fallback to tavgsfc/tsk/t2 when out of [150,400] K
    (module_initialize_real.F:2820-2860).
  * Over water, protect TSK with SST when SST in (170,400) (:2894-2905).
  * Seaice pre-adjust (adjust_for_seaice_pre, :3235) — for Canary, ice ~ absent,
    but reproduce the branch so a future cold case is correct.
  * TMN over water := SST; over land keep deep-soil temp (:3265-3335).
  * XLAND from LANDMASK (1=land -> XLAND=1; water -> XLAND=2), WRF convention.

The map factors / F / E / SINALPHA / COSALPHA / XLAT*/XLONG* come straight from
the met_em fields (the v0.3.0 schema carries MAPFAC_*X/*Y + MAPFAC_M/U/V + F/E +
SINALPHA/COSALPHA + XLAT_M/U/V etc). S2 maps schema names -> SurfaceInit fields
and confirms staggering. HGT comes from met_em HGT_M and MUST be the identical
array S1's base_state.compute_base_state receives (driver single-sources it).

Oracle: wrfinput TSK/SST/TMN/XLAND/MAPFAC_*/F/E/SINALPHA/COSALPHA/XLAT/XLONG/HGT;
tols ``WRFINPUT_TOLS``. The passthrough metrics should match near-exactly; the
SST/TSK/TMN protected fields match within the K tolerances.

FILE OWNERSHIP: this file + soil_init.py are S2's exclusive files. Do not edit
types.py, driver.py, or any S1/S3 file.
"""

from __future__ import annotations

from gpuwrf.init.real_init.types import RealInitConfig, SurfaceInit
from gpuwrf.init.metgrid_schema import MetEmArtifact


def compute_surface_init(
    config: RealInitConfig,
    metem: MetEmArtifact,
) -> SurfaceInit:
    """Builds the wrfinput-equivalent surface + metric fields.

    Consumes the FROZEN v0.3.0 ``MetEmArtifact``. The returned ``hgt`` is the
    canonical terrain the base-state lane also uses.
    """

    raise NotImplementedError("v0.4.0 S2 (GPT): compute_surface_init — frozen stub")
