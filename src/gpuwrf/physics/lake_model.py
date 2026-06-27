"""G3 WRF lake-model scaffold: recognized, fail-closed.

WRF reference: ``phys/module_sf_lake.F`` dispatches ``Lake``/``LakeMain`` and
initializes with ``lakeini``. The model carries lake depth/category state, snow
layers, lake-water temperatures, soil/snow ice/liquid water, lake ice fraction,
Monin-Obukhov flux state, hydrology and tridiagonal thermal solves.

No faithful JAX kernel or pristine-WRF single-column oracle is shipped here.
Calling an entrypoint raises so ``sf_lake_physics=1`` cannot silently run the
wrong surface path.
"""

from __future__ import annotations

from typing import NamedTuple

LAKE_SOURCE = "<DATA_ROOT>/src/wrf_pristine/WRF/phys/module_sf_lake.F"
REGISTRY_SOURCE = "<DATA_ROOT>/src/wrf_pristine/WRF/Registry/Registry.EM_COMMON"

LAKE_CARRY_MEMBERS = (
    "lakedepth2d",
    "savedtke12d",
    "snowdp2d",
    "h2osno2d",
    "snl2d",
    "t_grnd2d",
    "t_lake3d",
    "lake_icefrac3d",
    "z_lake3d",
    "dz_lake3d",
    "t_soisno3d",
    "h2osoi_ice3d",
    "h2osoi_liq3d",
    "h2osoi_vol3d",
    "z3d",
    "dz3d",
    "zi3d",
    "watsat3d",
    "csol3d",
    "tkmg3d",
    "tkdry3d",
    "tksatu3d",
    "lakemask",
)


class LakeState(NamedTuple):
    """Minimal future-port handle for the lake carry/static payload."""

    carry_members: tuple[str, ...]
    static_tables: object = None
    carry: object = None


def lake_step(*args, **kwargs):
    """WRF lake-model entrypoint -- fail-closed until oracle/kernel work lands."""

    raise NotImplementedError(
        "G3 lake model is a fail-closed scaffold, not an operational physics "
        f"kernel. Required carry members: {len(LAKE_CARRY_MEMBERS)}; WRF source: "
        f"{LAKE_SOURCE}. Add a pristine-WRF single-column oracle and faithful JAX "
        "kernel before wiring sf_lake_physics into the operational scan."
    )


def lake_initial_state(*args, **kwargs):
    """WRF ``lakeini`` scaffold -- fail-closed until source-faithful init lands."""

    raise NotImplementedError(
        "G3 lakeini is a fail-closed scaffold. The WRF initialization path "
        "requires lake_depth/use_lakedepth/lake_min_elev handling and lake/snow/"
        "soil column carry creation from the input fields before it can run."
    )


__all__ = [
    "LAKE_SOURCE",
    "REGISTRY_SOURCE",
    "LAKE_CARRY_MEMBERS",
    "LakeState",
    "lake_step",
    "lake_initial_state",
]
