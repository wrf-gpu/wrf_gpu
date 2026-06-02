"""S2 — native real.exe-equivalent surface / skin / SST / metric initial state.

FROZEN ENTRY SIGNATURE (``compute_surface_init``). This module reproduces the
WRF ``real.exe`` surface handling (``dyn_em/module_initialize_real.F`` +
``share/module_soil_pre.F``) for the Canary AIFS / Noah-MP case
(``sf_surface_physics=4``), consuming the frozen v0.3.0 ``MetEmArtifact``.

What real.exe does, and what we reproduce here
----------------------------------------------
The map factors / Coriolis F,E / SINALPHA / COSALPHA / XLAT*/XLONG* / HGT come
straight from the met_em artifact (geogrid statics) — real.exe passes them
through with no arithmetic (``module_initialize_real.F`` reads ``*_gc`` and
renames). The S0 recon + the oracle confirm these are *bit-exact* passthroughs,
so S2 maps the schema names to the :class:`SurfaceInit` fields and confirms the
staggering. ``hgt`` is sourced ONCE here from met_em ``HGT_M`` and is the
canonical terrain the driver also hands to S1's base-state lane (the
single-source-terrain rule in V0.4.0-S0-PLAN.md §3).

The skin-temperature / SST / deep-soil-temperature fields get the real.exe
water/skin protection + elevation-lapse logic, reproduced faithfully:

  * ``tsk`` starts at met_em ``SKINTEMP`` (``grid%tsk = grid%tsk_gc``,
    :709). With ``flag_sst==0`` (the Canary met_em carries no SST field) the
    SST-protect-over-water branch (:2900-2905) is a no-op. Then
    ``adjust_soil_temp_new`` (:1057) applies, ON LAND, the topography lapse
    ``tsk -= 0.0065*(ht - toposoil)`` (toposoil = met_em ``SOILHGT``).
  * ``tmn`` starts at met_em ``SOILTEMP`` (annual-mean deep-soil temp,
    ``grid%tmn = grid%tmn_gc``, :710). For Noah-MP, ``adjust_soil_temp_new``
    applies, ON LAND, the sea-level lapse ``tmn -= 0.0065*ht``
    (module_soil_pre.F:973). NOTE: this ``tmn`` (pre water-fill / pre
    reasonable-value fallback) is the value the soil lane must use as its
    300-cm interpolation endpoint, so it is exposed as ``tmn_soil_endpoint``.
  * ``fix_tsk_tmn`` (:3266): over water (``flag_sst==0`` ELSE branch) set
    ``tmn := tsk``; then the reasonable-value fallbacks (:3289, :3320) replace
    any out-of-[170,400] K ``tsk``/``tmn`` ON LAND with ``tsk`` (the order is
    AFTER soil interpolation, so the soil lane never sees the fallback value).
  * ``sst`` is, with ``flag_sst==0``, left equal to the (adjusted) ``tsk`` —
    real.exe leaves ``grid%sst`` at its skin-temp-seeded value, and the oracle
    confirms ``SST == TSK`` everywhere.
  * ``xland`` from the landmask (1=land -> XLAND=1; water -> XLAND=2),
    ``seaice`` / ``snow`` carried (both ~absent for the Canary case).

A loud guard rejects an unsupported case (an SST field present, or a non-Noah
LSM, or num_soil_layers != 4) rather than silently mis-initializing (the P2-2
loud-reject pattern). Everything is computed in fp64 (the project fp64 rule);
the driver downcasts at the final ``State`` pack.

VALIDATION: ``compute_surface_init`` vs real.exe wrfinput TSK/SST/TMN/XLAND/
MAPFAC_*/F/E/SINALPHA/COSALPHA/XLAT/XLONG/HGT, ≥10 cases across d01/d02/d03,
tolerances ``WRFINPUT_TOLS`` (see ``tests/init/real_init/test_s2_surface_soil.py``
and ``proofs/v040/s2_wrfinput_surface_soil_report.json``). Measured max-abs
errors are at the fp32-storage rounding floor (~1.5e-5 K; passthroughs 0.0).

FILE OWNERSHIP: this file + ``soil_init.py`` are S2's exclusive files. This
module does not import or mutate ``types.py``, ``driver.py``, or any S1/S3 file.
"""

from __future__ import annotations

import numpy as np

from gpuwrf.init.real_init.types import RealInitConfig, SurfaceInit
from gpuwrf.init.metgrid_schema import MetEmArtifact


# WRF tmn/tsk topography lapse rate (module_soil_pre.F:973 / :1057). Pinned.
_LAPSE = 0.0065  # K m^-1
# WRF "reasonable temperature" guard bounds (module_initialize_real.F:3289,3320).
_T_MIN = 170.0
_T_MAX = 400.0
# WRF SST sanity window for the (unused-here) flag_sst==1 protect branch.
_SST_LO = 170.0
_SST_HI = 400.0


def _met(metem: MetEmArtifact, name: str) -> np.ndarray:
    """Returns a met_em field as fp64 (raises if a mandatory field is absent)."""

    arr = metem.arrays.get(name)
    if arr is None:
        raise KeyError(
            f"v0.4.0 S2 surface_init: met_em artifact is missing required "
            f"field {name!r} (domain {metem.domain}, time {metem.valid_time})"
        )
    return np.asarray(arr, dtype=np.float64)


def compute_surface_init(
    config: RealInitConfig,
    metem: MetEmArtifact,
) -> SurfaceInit:
    """Builds the wrfinput-equivalent surface + metric fields (real.exe-faithful).

    Consumes the FROZEN v0.3.0 ``MetEmArtifact``. The returned ``hgt`` is the
    canonical terrain the base-state lane also uses. The returned ``tmn`` already
    has the over-water and reasonable-value fixes applied; the soil lane needs
    the *pre-fix* deep-soil endpoint, which is exposed as the extra (additive,
    freeze-compatible) attribute ``tmn_soil_endpoint`` on the returned object via
    a thin wrapper — see ``compute_surface_init_full`` below.
    """

    full = compute_surface_init_full(config, metem)
    return full.surface


# --- internal richer result (lets the soil lane reuse the pre-fix tmn) --------
class _SurfaceResult:
    """Internal S2 surface result: the frozen :class:`SurfaceInit` plus the
    intermediate fields the *soil* lane needs to stay byte-consistent with the
    surface lane (so both lanes share one landmask / one tsk / one pre-fix tmn).

    This is NOT a frozen interface; it lives entirely inside S2 (surface_init +
    soil_init) and never crosses a lane boundary. The driver only ever sees the
    frozen ``SurfaceInit``.
    """

    __slots__ = (
        "surface",
        "tsk",
        "sst",
        "tmn_soil_endpoint",
        "landmask",
        "land",
        "water",
        "ht",
        "toposoil",
    )

    def __init__(
        self,
        surface: SurfaceInit,
        tsk: np.ndarray,
        sst: np.ndarray,
        tmn_soil_endpoint: np.ndarray,
        landmask: np.ndarray,
        ht: np.ndarray,
        toposoil: np.ndarray,
    ) -> None:
        self.surface = surface
        self.tsk = tsk
        self.sst = sst
        self.tmn_soil_endpoint = tmn_soil_endpoint
        self.landmask = landmask
        self.land = landmask > 0.5
        self.water = ~self.land
        self.ht = ht
        self.toposoil = toposoil


def compute_surface_init_full(
    config: RealInitConfig,
    metem: MetEmArtifact,
) -> _SurfaceResult:
    """Same as :func:`compute_surface_init` but returns the richer internal
    result the soil lane consumes. Public-stable output is ``.surface``."""

    _reject_unsupported(config, metem)

    # ---- passthrough metric / coordinate / static fields (bit-exact) --------
    xlat = _met(metem, "XLAT_M")
    xlong = _met(metem, "XLONG_M")
    xlat_u = _met(metem, "XLAT_U")
    xlong_u = _met(metem, "XLONG_U")
    xlat_v = _met(metem, "XLAT_V")
    xlong_v = _met(metem, "XLONG_V")
    mapfac_m = _met(metem, "MAPFAC_M")
    mapfac_u = _met(metem, "MAPFAC_U")
    mapfac_v = _met(metem, "MAPFAC_V")
    mapfac_mx = _met(metem, "MAPFAC_MX")
    mapfac_my = _met(metem, "MAPFAC_MY")
    mapfac_ux = _met(metem, "MAPFAC_UX")
    mapfac_uy = _met(metem, "MAPFAC_UY")
    mapfac_vx = _met(metem, "MAPFAC_VX")
    mapfac_vy = _met(metem, "MAPFAC_VY")
    f = _met(metem, "F")
    e = _met(metem, "E")
    sinalpha = _met(metem, "SINALPHA")
    cosalpha = _met(metem, "COSALPHA")
    hgt = _met(metem, "HGT_M")  # single-source terrain (driver hands to S1)

    landmask = _met(metem, "LANDMASK")
    land = landmask > 0.5
    water = ~land

    # toposoil = met_em SOILHGT (the source-data surface elevation). real.exe
    # only blends it into ht under smooth_cg_topo (off here); we use it raw.
    toposoil = _met(metem, "SOILHGT")
    skintemp = _met(metem, "SKINTEMP")
    soiltemp = _met(metem, "SOILTEMP")  # annual-mean deep-soil temp (geogrid)

    # ---- TSK: skin temp + over-land topography lapse ------------------------
    # grid%tsk = SKINTEMP (:709). flag_sst==0 so the SST-protect branch
    # (:2900-2905) is a no-op. adjust_soil_temp_new (:1057): on land,
    # tsk -= 0.0065*(ht - toposoil).
    tsk = skintemp.copy()
    delev = hgt - toposoil
    tsk[land] = tsk[land] - _LAPSE * delev[land]

    # ---- TMN deep-soil endpoint (the value the soil lane interpolates to) ---
    # grid%tmn = SOILTEMP (:710). NOAHMP CASE in adjust_soil_temp_new
    # (module_soil_pre.F:973): on land, tmn -= 0.0065*ht. This is the deep
    # (300 cm) endpoint the soil interp uses BEFORE the over-water / reasonable
    # fixes (those run after process_soil_real).
    tmn_soil_endpoint = soiltemp.copy()
    tmn_soil_endpoint[land] = tmn_soil_endpoint[land] - _LAPSE * hgt[land]

    # ---- TMN final (post soil interp): water-fill + reasonable-value fix -----
    # fix_tsk_tmn (:3266): flag_sst==0 ELSE branch -> over water tmn := tsk.
    tmn = tmn_soil_endpoint.copy()
    tmn[water] = tsk[water]
    # reasonable-tmn guard (:3320): out-of-range on land -> tsk (tsk is in-range
    # for this case). Mirrors the tsk guard (:3289) which is a no-op here.
    bad_tmn = ((tmn < _T_MIN) | (tmn > _T_MAX)) & land
    tmn[bad_tmn] = tsk[bad_tmn]
    bad_tsk = (tsk < _T_MIN) | (tsk > _T_MAX)
    tsk[bad_tsk] = tmn[bad_tsk]

    # ---- SST: flag_sst==0 -> left equal to (adjusted) tsk -------------------
    sst = tsk.copy()

    # ---- XLAND from landmask (WRF convention 1=land, 2=water) ---------------
    xland = np.where(land, 1.0, 2.0)

    # ---- seaice / snow (carried; ~absent for the Canary case) ---------------
    seaice = _optional(metem, ("SEAICE", "XICE"))
    if seaice is None:
        seaice = np.zeros_like(landmask)
    snowh = _optional(metem, ("SNOWH",))
    if snowh is None:
        snowh = np.zeros_like(landmask)

    surface = SurfaceInit(
        xlat=xlat,
        xlong=xlong,
        xlat_u=xlat_u,
        xlong_u=xlong_u,
        xlat_v=xlat_v,
        xlong_v=xlong_v,
        mapfac_m=mapfac_m,
        mapfac_u=mapfac_u,
        mapfac_v=mapfac_v,
        mapfac_mx=mapfac_mx,
        mapfac_my=mapfac_my,
        mapfac_ux=mapfac_ux,
        mapfac_uy=mapfac_uy,
        mapfac_vx=mapfac_vx,
        mapfac_vy=mapfac_vy,
        f=f,
        e=e,
        sinalpha=sinalpha,
        cosalpha=cosalpha,
        hgt=hgt,
        tsk=tsk,
        sst=sst,
        tmn=tmn,
        xland=xland,
        landmask=landmask.copy(),
        snowh=snowh,
        seaice=seaice,
    )
    return _SurfaceResult(
        surface=surface,
        tsk=tsk,
        sst=sst,
        tmn_soil_endpoint=tmn_soil_endpoint,
        landmask=landmask,
        ht=hgt,
        toposoil=toposoil,
    )


def _optional(metem: MetEmArtifact, names: tuple[str, ...]) -> np.ndarray | None:
    for n in names:
        arr = metem.arrays.get(n)
        if arr is not None:
            return np.asarray(arr, dtype=np.float64)
    return None


def _reject_unsupported(config: RealInitConfig, metem: MetEmArtifact) -> None:
    """Loud-rejects cases v0.4.0 S2 does NOT faithfully handle (P2-2 pattern).

    The supported path is the Canary AIFS / Noah-MP isobaric-metgrid case:
    sf_surface_physics=4, num_soil_layers=4, no metgrid SST field (flag_sst=0).
    A future cold/ice/SST case must extend the SST-protect + seaice branches
    rather than silently running the wrong arithmetic.
    """

    if config.sf_surface_physics != 4:
        raise NotImplementedError(
            "v0.4.0 S2 supports sf_surface_physics=4 (Noah-MP) only; got "
            f"{config.sf_surface_physics}. The TSK/TMN lapse and the soil "
            "interp endpoints are scheme-specific."
        )
    if config.num_soil_layers != 4:
        raise NotImplementedError(
            "v0.4.0 S2 supports the Noah-MP 4-layer soil set only; got "
            f"num_soil_layers={config.num_soil_layers}."
        )
    if "SST" in metem.arrays:
        # The frozen schema does not list SST; if a future artifact adds it,
        # the flag_sst==1 SST-protect branches (TSK/TMN := SST over water) must
        # be wired before trusting this code.
        raise NotImplementedError(
            "v0.4.0 S2 assumes flag_sst==0 (no metgrid SST field). A met_em "
            "artifact carrying SST needs the flag_sst==1 protect branches "
            "(module_initialize_real.F:2900-2905, :3272-3275) wired first."
        )
