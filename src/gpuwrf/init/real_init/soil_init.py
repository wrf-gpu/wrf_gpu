"""S2 (GPT) — soil thermodynamic + categorical initial state.

FROZEN ENTRY SIGNATURE. Reproduces real.exe ``process_soil_real`` +
``process_percent_cat_new`` (module_initialize_real.F:3009-3530, USE
module_soil_pre):

  * Vertical soil interpolation: met_em provides 2 metgrid soil layers
    (ST000010/ST010040/SM000010/SM010040 + the ST/SM layer stacks); Noah-MP
    needs ``config.num_soil_layers`` (Canary = 4). Reproduce module_soil_pre's
    interpolation onto the Noah-MP node depths ZS / thicknesses DZS
    (sf_surface_physics=4 selects the 4-layer 0.1/0.3/0.6/1.0 m set; confirm
    the exact ZS/DZS from module_sf_noahmpdrv / the wrfinput ZS/DZS oracle).
  * Categorical: dominant soil category ISLTYP from SCT_DOM / SOILCTOP, veg/
    landuse IVGTYP / LU_INDEX from LU_INDEX / LANDUSEF (process_percent_cat_new,
    :3045). Match isltyp to landmask: land cell must not be isoilwater; water
    cell forced to isoilwater (:3117-3140).
  * Soil-moisture floors / LSM-specific minima (:3342-3530); the Noah->Noah
    bad-moisture floor of 0.005 (:3403) etc — reproduce for Noah-MP path.

Oracle: wrfinput TSLB / SMOIS / ZS / DZS / ISLTYP / IVGTYP; tols ``WRFINPUT_TOLS``
(TSLB/SMOIS looser to absorb the 2->4-layer interpolation spread, but the
oracle is the real.exe 4-layer result, so the comparison IS faithful).

FILE OWNERSHIP: S2 exclusive (see surface_init.py header).
"""

from __future__ import annotations

from gpuwrf.init.real_init.types import RealInitConfig, SoilInit, SurfaceInit
from gpuwrf.init.metgrid_schema import MetEmArtifact


def compute_soil_init(
    config: RealInitConfig,
    metem: MetEmArtifact,
    surface: SurfaceInit,
) -> SoilInit:
    """Builds the wrfinput-equivalent soil temperature/moisture + categories.

    Takes ``surface`` so the soil categorical logic can use the consistent
    landmask/xland the surface lane produced.
    """

    raise NotImplementedError("v0.4.0 S2 (GPT): compute_soil_init — frozen stub")
