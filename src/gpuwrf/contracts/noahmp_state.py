"""Frozen prognostic Noah-MP land-surface state and flux contracts (v0.2.0 P0-3).

INTERFACE FREEZE ONLY — no physics. See ``.agent/decisions/ADR-NOAHMP-INTERFACES.md``.

These pytrees are SIBLINGS of ``contracts.state.State`` (like ``BaseState`` /
``BoundaryState``); the prognostic dycore ``State.__slots__`` is intentionally NOT
widened (ADR §2.3). Land state advances at the physics (long) timestep only, is
2-D / soil / snow, and is threaded alongside ``State`` by the operational driver.

Fixed dimensions (verified against ``/home/user/src/wrf_pristine`` Noah-MP):
- ``NSOIL = 4``  soil layers (module_sf_noahmpdrv.F:689-691).
- ``NSNOW = 3``  snow layers, ISNOW in {-2,-1,0} (module_sf_noahmpdrv.F:628).

Scope (NOAH-MP-SCOPING.md, LAND-ONLY): dveg=4, opt_run=3 (Schaake), opt_sfc=1,
opt_stc=1, opt_tbot=2, opt_snf=1, opt_alb=2, opt_crs/btr=1. Carbon, groundwater,
glacier, crop, urban, lake, irrigation CUT.

All arrays are device-resident JAX arrays, fp64 at construction. Shapes:
``surface_2d = (ny, nx)``, ``soil = (NSOIL, ny, nx)``, ``snow = (NSNOW, ny, nx)``,
``snowsoil = (NSNOW + NSOIL, ny, nx)``.
"""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

from typing import Any, NamedTuple

import jax
from jax import config

from .grid import GridSpec

configure_jax_x64()

NSOIL: int = 4
NSNOW: int = 3


@jax.tree_util.register_pytree_node_class
class NoahMPLandState:
    """FROZEN prognostic Noah-MP land carry (ADR §2.1). Land columns only.

    Threaded alongside ``State`` by the operational driver; advanced once per
    physics step by ``physics.noahmp.noahmp_driver.noah_mp_step``.

    Field units / WRF names are pinned in ADR-NOAHMP-INTERFACES.md §2.1. Soil
    fields are (NSOIL, ny, nx); snow fields are (NSNOW, ny, nx); ``zsnso`` is
    (NSNOW + NSOIL, ny, nx); everything else is (ny, nx). ``isnow`` is int32.
    """

    __slots__ = (
        # soil column (NSOIL)
        "tslb",       # STC(1:4)   soil temperature [K]
        "smois",      # SMC(1:4)   total soil moisture [m3/m3]
        "sh2o",       # SH2O(1:4)  liquid soil moisture [m3/m3]
        "smcwtd",     # SMCWTD     deep below-bottom soil moisture [m3/m3]
        # snow column (NSNOW) + bulk snow
        "isnow",      # ISNOW      active snow-layer count (-2,-1,0) [int32]
        "tsno",       # STC(-2:0)  snow-layer temperature [K]
        "snice",      # SNICE      snow-layer ice mass [kg/m2]
        "snliq",      # SNLIQ      snow-layer liquid mass [kg/m2]
        "zsnso",      # ZSNSO      snow+soil interface depths (<0) [m]
        "snowh",      # SNOWH      bulk snow depth [m]
        "sneqv",      # SNEQV      bulk snow water equivalent [kg/m2]
        "sneqvo",     # SNEQVO     prior-step SWE [kg/m2]
        "tauss",      # TAUSS      non-dimensional snow age
        "albold",     # ALBOLD     prior-step snow albedo
        # canopy "big-leaf" (single layer) — the HFX-fix variables
        "tv",         # TV         canopy (vegetation) temperature [K]
        "tg",         # TG         ground temperature [K]
        "tah",        # TAH        canopy-air temperature [K]  (HFX fix)
        "eah",        # EAH        canopy-air vapor pressure [Pa]
        "canliq",     # CANLIQ     intercepted canopy liquid [kg/m2]
        "canice",     # CANICE     intercepted canopy ice [kg/m2]
        "fwet",       # FWET       wetted canopy fraction
        "lai",        # LAI        leaf area index [m2/m2]
        "sai",        # SAI        stem area index [m2/m2]
        # exchange coeffs (supplied by sfclay, opt_sfc=1; carried for inout)
        "cm",         # CM         momentum drag coeff
        "ch",         # CH         heat drag coeff
        # surface diagnostics carried for the coupler / writer
        "t_skin",     # TSK = TRAD radiative skin temperature [K]
        "qsfc",       # QSFC       surface mixing ratio [kg/kg]
        "znt",        # ZNT = Z0WRF combined roughness [m]
        "emiss",      # EMISSI     surface emissivity
        "albedo",     # SALB       broadband surface albedo
        # accumulated runoff (Schaake, opt_run=3)
        "sfcrunoff",  # SFCRUNOFF  accumulated surface runoff [m]
        "udrunoff",   # UDRUNOFF   accumulated subsurface runoff [m]
    )

    def __init__(
        self,
        tslb: jax.Array,
        smois: jax.Array,
        sh2o: jax.Array,
        smcwtd: jax.Array,
        isnow: jax.Array,
        tsno: jax.Array,
        snice: jax.Array,
        snliq: jax.Array,
        zsnso: jax.Array,
        snowh: jax.Array,
        sneqv: jax.Array,
        sneqvo: jax.Array,
        tauss: jax.Array,
        albold: jax.Array,
        tv: jax.Array,
        tg: jax.Array,
        tah: jax.Array,
        eah: jax.Array,
        canliq: jax.Array,
        canice: jax.Array,
        fwet: jax.Array,
        lai: jax.Array,
        sai: jax.Array,
        cm: jax.Array,
        ch: jax.Array,
        t_skin: jax.Array,
        qsfc: jax.Array,
        znt: jax.Array,
        emiss: jax.Array,
        albedo: jax.Array,
        sfcrunoff: jax.Array,
        udrunoff: jax.Array,
    ) -> None:
        for name, value in zip(self.__slots__, (
            tslb, smois, sh2o, smcwtd, isnow, tsno, snice, snliq, zsnso, snowh,
            sneqv, sneqvo, tauss, albold, tv, tg, tah, eah, canliq, canice,
            fwet, lai, sai, cm, ch, t_skin, qsfc, znt, emiss, albedo,
            sfcrunoff, udrunoff,
        )):
            setattr(self, name, value)

    @classmethod
    def zeros(cls, grid: GridSpec) -> "NoahMPLandState":
        """Allocate a zero land carry — STUB (no physics init here)."""

        raise NotImplementedError("NoahMPLandState.zeros: Sprint 0b table loader / cold-init")

    def replace(self, **updates) -> "NoahMPLandState":
        """Return an updated land pytree with explicit field names (mirrors State)."""

        values = {name: getattr(self, name) for name in self.__slots__}
        values.update(updates)
        return type(self)(**values)

    def bytes(self) -> int:
        """Report persistent land-state bytes for the spacetime budget."""

        leaves, _ = jax.tree_util.tree_flatten(self)
        return int(sum(int(l.size) * int(l.dtype.itemsize) for l in leaves))

    def tree_flatten(self):
        return tuple(getattr(self, name) for name in self.__slots__), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class NoahMPStatic:
    """FROZEN read-only per-run Noah-MP inputs (ADR §2.4). Constructed from wrfinput.

    Holds category fields, soil-layer geometry, location, and the parameter
    tables. NOT advanced; never written in the timestep. ``parameters`` is the
    ``NoahMPParameters`` table bundle (Sprint 0b).
    """

    __slots__ = (
        "ivgtyp",     # int32 vegetation category
        "isltyp",     # int32 soil category
        "xland",      # 1 land / 2 water
        "landmask",
        "lakemask",
        "lu_index",   # int32 land-use index
        "tbot",       # deep-soil lower BC temperature [K]
        "dzs",        # soil-layer thicknesses [m] (len NSOIL)
        "zsoil",      # soil interface depths (<0) [m] (len NSOIL)
        "lat",        # latitude [deg]
        "dx_m",       # grid spacing [m]
        "parameters",  # NoahMPParameters bundle (Sprint 0b)
        # --- ADDITIVE (S6a integrate, patch protocol): per-column green-vegetation
        # fraction fields from wrfinput. SHDMAX/SHDFAC are 2-D input fields, NOT
        # MPTABLE parameters (arbiter = pristine WRF module_sf_noahmplsm.F:864 +
        # module_sf_noahmpdrv.F:752-753: FVEG=VEGFRA/100, FVGMAX=VEGMAX/100; for
        # dveg=4 FVEG=SHDMAX=FVGMAX). Default None so pre-S6a positional
        # NoahMPStatic(...) constructions (energy gate) keep working unchanged.
        "shdmax",     # annual-max green-veg fraction [0-1] (= VEGMAX/100); FVEG source for dveg=4
        "shdfac",     # instantaneous green-veg fraction [0-1] (= VEGFRA/100)
    )

    def __init__(
        self,
        ivgtyp: jax.Array,
        isltyp: jax.Array,
        xland: jax.Array,
        landmask: jax.Array,
        lakemask: jax.Array,
        lu_index: jax.Array,
        tbot: jax.Array,
        dzs: jax.Array,
        zsoil: jax.Array,
        lat: jax.Array,
        dx_m: Any,
        parameters: Any,
        shdmax: Any = None,
        shdfac: Any = None,
    ) -> None:
        for name, value in zip(self.__slots__, (
            ivgtyp, isltyp, xland, landmask, lakemask, lu_index, tbot, dzs,
            zsoil, lat, dx_m, parameters, shdmax, shdfac,
        )):
            setattr(self, name, value)

    def replace(self, **updates) -> "NoahMPStatic":
        values = {name: getattr(self, name) for name in self.__slots__}
        values.update(updates)
        return type(self)(**values)

    def tree_flatten(self):
        return tuple(getattr(self, name) for name in self.__slots__), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        return cls(*children)


class NoahMPFluxes(NamedTuple):
    """FROZEN per-step Noah-MP outputs consumed by the coupler (ADR §2.2).

    The ONLY object the coupling adapter reads back into model state and the PBL
    bottom boundary. WRF driver mapping (module_sf_noahmpdrv.F):
      hfx=FSH(:1223), grdflx=SSOIL(:1224), qfx=ECAN+ESOIL+ETRAN(:1205),
      lh=FCEV+FGEV+FCTR(:1206), tsk=TRAD(:1222), albedo=SALB(:1231),
      emiss=EMISSI(:1243), znt=Z0WRF(:1279-1280).
    All fields are (ny, nx) on land columns. Fluxes are W/m2 positive upward
    (hfx/lh/grdflx); qfx is kg/m2/s.
    """

    hfx: jax.Array       # FSH    sensible heat flux [W/m2]
    lh: jax.Array        # FCEV+FGEV+FCTR latent heat flux [W/m2]
    qfx: jax.Array       # ECAN+ESOIL+ETRAN moisture flux [kg/m2/s]
    grdflx: jax.Array    # SSOIL  ground heat flux [W/m2]
    tsk: jax.Array       # TRAD   radiative skin temperature [K]
    qsfc: jax.Array      # QSFC   surface mixing ratio [kg/kg]
    znt: jax.Array       # Z0WRF  roughness length [m]
    emiss: jax.Array     # EMISSI surface emissivity
    albedo: jax.Array    # SALB   surface albedo
    chs: jax.Array       # CHV/CHB blend, diagnostic heat exchange coeff
    # --- ADDITIVE (v0.9.0 Noah-MP 2-m LSM diagnostic; default None) ---
    # The land 2-m air temperature WRF writes back to T2 (module_surface_driver.F
    # :3469-3473), OVERWRITING the surface-layer MYNN/sfclay 2-m value over land:
    # t2 = FVEG*T2MV + (1-FVEG)*T2MB (veg) / T2MB (bare). The coupler routes ``t2``
    # over land. ``t2mv``/``t2mb`` carried for diagnostics/proofs. Default None so
    # pre-v0.9.0 callers construct NoahMPFluxes unchanged.
    t2: "jax.Array | None" = None     # land 2-m air temperature [K] -> T2
    t2mv: "jax.Array | None" = None   # 2-m air temp over vegetated tile [K]
    t2mb: "jax.Array | None" = None   # 2-m air temp over bare tile [K]


__all__ = [
    "NSOIL",
    "NSNOW",
    "NoahMPLandState",
    "NoahMPStatic",
    "NoahMPFluxes",
]
