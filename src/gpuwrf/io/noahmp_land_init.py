"""Corpus -> prognostic Noah-MP land warm-start (Sprint S6b ACTIVATION).

Builds the FROZEN ``NoahMPLandState`` + ``NoahMPStatic`` (contracts.noahmp_state)
from a Gen2 corpus ``wrfinput_<domain>`` snapshot at t=0, so the operational
forecast can ACTIVATE prognostic Noah-MP over land instead of the prescribed-land
bulk path.

This is a WARM-START, not a synthetic cold-init: the Gen2 corpus wrfinput written
by the same WRF/Noah-MP that produced the CPU truth already carries the full
prognostic Noah-MP land state (TSLB/SMOIS/SH2O soil; TSNO/SNICE/SNLIQ/ZSNSO snow;
TV/TG/TAH/EAH/CANLIQ/CANICE/FWET canopy; LAI/XSAI phenology; SNOWH/SNEQVO/ALBOLD/
ISNOW snow-bulk; CM/CH drag). The handful of fields WRF does not persist to
wrfinput (SMCWTD deep below-bottom soil moisture, TAUSS snow age, QSFC surface
mixing ratio, ZNT roughness) are cold-initialised from their standard WRF/Noah-MP
``noahmplsm`` initial defaults (module_sf_noahmpdrv.F NOAHMP_INIT) -- NOT masked or
faked. Provenance for each field is recorded in the returned ``meta``.

The land tile is the only tile Noah-MP owns; ocean/water columns keep the
prescribed-SST sfclay bulk path verbatim (the coupler's where(is_land,...) switch).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.noahmp_state import NSNOW, NSOIL, NoahMPLandState, NoahMPStatic
from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.physics.noahmp.tables import load_noahmp_parameters


# Standard WRF MPTABLE / Noah-MP table directory (pristine WRF run/).
DEFAULT_TABLE_DIR = Path("/home/user/src/wrf_pristine/WRF/run")


def _i32(field) -> jnp.ndarray:
    return jnp.asarray(np.asarray(field).astype(np.int32), dtype=jnp.int32)


def _surface_2d(field) -> jnp.ndarray:
    """Squeeze a corpus field to surface (ny, nx) fp64."""
    a = np.squeeze(np.asarray(field, dtype=np.float64))
    return jnp.asarray(a, dtype=jnp.float64)


def _layered(field, nlayers: int) -> jnp.ndarray:
    """Squeeze a corpus field to (nlayers, ny, nx) fp64."""
    a = np.squeeze(np.asarray(field, dtype=np.float64))
    if a.ndim != 3 or a.shape[0] != nlayers:
        raise ValueError(f"expected ({nlayers}, ny, nx) layered field; got shape {a.shape}")
    return jnp.asarray(a, dtype=jnp.float64)


def build_noahmp_land_state(
    run_dir: str | Path,
    domain: str = "d02",
    *,
    table_dir: str | Path | None = None,
) -> tuple[NoahMPLandState, NoahMPStatic, dict[str, Any]]:
    """Warm-start the prognostic Noah-MP land carry from a corpus wrfinput.

    Returns ``(land_state, static, meta)``. ``meta`` records per-field provenance
    (wrfinput-loaded vs cold-init default) for the activation proof.
    """

    run = Gen2Run(Path(run_dir))
    present = set(run.wrfinput_variables(domain))
    tdir = Path(table_dir) if table_dir is not None else DEFAULT_TABLE_DIR

    def L(name):
        return run.load_wrfinput(domain, name, lazy=False)

    def has(name: str) -> bool:
        return name in present

    provenance: dict[str, str] = {}

    # ---- geometry / categories (static) ----
    xland = _surface_2d(L("XLAND"))
    ny, nx = xland.shape
    is_land = xland < 1.5

    landmask = _surface_2d(L("LANDMASK")) if has("LANDMASK") else jnp.where(is_land, 1.0, 0.0)
    lakemask = _surface_2d(L("LAKEMASK")) if has("LAKEMASK") else jnp.zeros((ny, nx), dtype=jnp.float64)
    ivgtyp = _i32(_surface_2d(L("IVGTYP")))
    isltyp = _i32(_surface_2d(L("ISLTYP")))
    lu_index = _i32(_surface_2d(L("LU_INDEX")))
    # deep-soil lower BC temperature: WRF TMN (module_sf_noahmpdrv.F TBOT). Fall back
    # to the bottom soil layer if a corpus lacks TMN.
    if has("TMN"):
        tbot = _surface_2d(L("TMN"))
        provenance["tbot"] = "wrfinput TMN"
    else:
        tbot = _layered(L("TSLB"), NSOIL)[-1]
        provenance["tbot"] = "cold-init: bottom TSLB layer (no TMN in corpus)"

    # soil-layer geometry. The wrfinput ``ZS`` field is the depth to soil-layer
    # CENTERS (e.g. [0.05, 0.25, 0.70, 1.50]); ``DZS`` is the layer THICKNESSES
    # ([0.10, 0.30, 0.60, 1.00]). Noah-MP's ZSOIL is the depth to soil-layer
    # INTERFACES (bottoms), built from the cumulative thicknesses exactly as WRF
    # does in module_sf_noahmpdrv.F:689-692:
    #     ZSOIL(1) = -DZS(1);  ZSOIL(K) = ZSOIL(K-1) - DZS(K)
    # -> ZSOIL = [-0.10, -0.40, -1.00, -2.00]. The previous code mapped the ZS
    # CENTERS straight onto ``zsoil`` ([-0.05, -0.25, -0.70, -1.50]), which made
    # every soil DZSNSO half-too-thin -- in particular the top soil thickness was
    # 0.05 m instead of 0.10 m. That doubled the surface ground conductance
    # CGH = 2*DF/DZSNSO(1) (energy.py BARE_FLUX/VEGE_FLUX) and inflated overnight
    # GRDFLX, driving the residual land cold bias. Build ZSOIL from DZS, WRF-faithful.
    dzs_arr = np.squeeze(np.asarray(L("DZS"), dtype=np.float64)) if has("DZS") else np.array([0.10, 0.30, 0.60, 1.00])
    dzs_np = np.abs(dzs_arr).reshape(NSOIL)
    zsoil_np = -np.cumsum(dzs_np)                     # WRF noahmpdrv:689-692
    zsoil = jnp.asarray(zsoil_np, dtype=jnp.float64)
    dzs = jnp.asarray(dzs_np, dtype=jnp.float64)

    lat = _surface_2d(L("XLAT"))
    dx_m = float(run.grid(domain).dx_m)

    # green-vegetation fractions (FVEG source; dveg=4 uses SHDMAX)
    shdmax = (_surface_2d(L("SHDMAX")) / 100.0) if has("SHDMAX") else None
    shdfac = (_surface_2d(L("VEGFRA")) / 100.0) if has("VEGFRA") else shdmax
    if shdmax is None and shdfac is not None:
        shdmax = shdfac

    parameters = load_noahmp_parameters(tdir)

    static = NoahMPStatic(
        ivgtyp=ivgtyp, isltyp=isltyp, xland=xland, landmask=landmask,
        lakemask=lakemask, lu_index=lu_index, tbot=tbot, dzs=dzs, zsoil=zsoil,
        lat=lat, dx_m=dx_m, parameters=parameters, shdmax=shdmax, shdfac=shdfac,
    )

    # ---- prognostic land carry ----
    tslb = _layered(L("TSLB"), NSOIL)
    smois = _layered(L("SMOIS"), NSOIL)
    provenance["tslb"] = "wrfinput TSLB"
    provenance["smois"] = "wrfinput SMOIS"

    # WRF NOAHMP_INIT initialises the liquid soil water SH2O from SMOIS + TSLB
    # (module_sf_noahmpdrv.F:2069-2106): SH2O = SMOIS where TSLB >= 273.149 K
    # (all liquid above freezing), else the FK frozen-water cap. The Gen2 corpus
    # wrfinput is the PRE-NOAHMP_INIT state -- its SH2O over land is written as 0
    # (so sice = SMOIS - SH2O = SMOIS => the column reads as ALL-ICE even at
    # 290-297 K). Loading that raw zero makes the first PHASECHANGE step "melt"
    # phantom ice, dumping a latent-heat sink that craters TSLB toward TFRZ and
    # drives the overnight cold-start transient. We therefore reconstruct SH2O
    # via the faithful NOAHMP_INIT relation -- the SAME liquid-water state WRF
    # integrates from (see wrfout t=0: SH2O == SMOIS over the warm Canary soil).
    sh2o = _noahmp_init_sh2o(smois, tslb, isltyp, ivgtyp, parameters)
    provenance["sh2o"] = (
        "NOAHMP_INIT reconstruction from SMOIS/TSLB (module_sf_noahmpdrv.F:2088-2106); "
        "corpus wrfinput SH2O is pre-init zero"
    )

    # --- WRF NOAHMP_INIT (module_sf_noahmpdrv.F:1827-2334) ---
    # The Gen2 corpus wrfinput is the PRE-NOAHMP_INIT state: TSLB/SMOIS/SH2O/TSK/
    # TMN/LAI/SHDMAX + categories are real, but the prognostic canopy/snow XY
    # fields (TV/TG/TAH/EAH/CANLIQ/CANICE/FWET/ISNOW/TSNO/SNICE/SNLIQ/ZSNSO/
    # SNEQVO/ALBOLD/CM/CH) are written as ZERO over land. This is exactly what
    # WRF's real.exe sees: NOAHMP_INIT then fills them from TSK/CANWAT/SNOW/SNOWH
    # before the first timestep. We replicate NOAHMP_INIT faithfully here so the
    # warm-start is the SAME initial land carry WRF integrates from -- NOT a
    # masked/faked seed. Fields are taken from the corpus when nonzero over land
    # (a spun-up wrfinput/restart), else cold-init per NOAHMP_INIT.
    tsk = _surface_2d(L("TSK"))
    canwat = _surface_2d(L("CANWAT")) if has("CANWAT") else jnp.zeros((ny, nx), dtype=jnp.float64)
    snowh = _surface_2d(L("SNOWH")) if has("SNOWH") else jnp.zeros((ny, nx), dtype=jnp.float64)
    sneqv = _surface_2d(L("SNOW")) if has("SNOW") else jnp.zeros((ny, nx), dtype=jnp.float64)

    def _init_over_land(corpus_name, default, *, is_temp=False):
        """Use the corpus field on land where it is plausibly initialised; else
        the NOAHMP_INIT default. A 0-K temperature / all-zero canopy field is the
        pre-init signature -> use the default. ``default`` is a (ny,nx) array."""
        if has(corpus_name):
            c = _surface_2d(L(corpus_name))
            valid = (c > 150.0) if is_temp else (jnp.abs(c) > 0.0)
            return jnp.where(valid, c, default)
        return default

    # canopy big-leaf (NOAHMP_INIT:2105-2122). tvxy=tgxy=tahxy=TSK; eahxy=2000;
    # canliqxy=CANWAT; canicexy=0; fwetxy=0. (snow-cap 273.15 branch is inert here
    # — Canary domain is snow-free, but we apply it for faithfulness.)
    snow_cap = (sneqv > 0.0) & (tsk > 273.15)
    tsk_capped = jnp.where(snow_cap, 273.15, tsk)
    tv = _init_over_land("TV", tsk_capped, is_temp=True)
    tg = _init_over_land("TG", tsk_capped, is_temp=True)
    tah = _init_over_land("TAH", tsk_capped, is_temp=True)
    eah = _init_over_land("EAH", jnp.full((ny, nx), 2000.0, dtype=jnp.float64))
    canliq = _init_over_land("CANLIQ", canwat)
    canice = _init_over_land("CANICE", jnp.zeros((ny, nx), dtype=jnp.float64))
    fwet = _init_over_land("FWET", jnp.zeros((ny, nx), dtype=jnp.float64))
    provenance["canopy_tv_tg_tah"] = "NOAHMP_INIT: =TSK (corpus wrfinput pre-init over land)"
    provenance["eah"] = "NOAHMP_INIT: 2000 Pa (corpus pre-init)"

    lai = _surface_2d(L("LAI")) if has("LAI") else jnp.zeros((ny, nx), dtype=jnp.float64)
    sai_name = "XSAI" if has("XSAI") else ("SAI" if has("SAI") else None)
    sai = _surface_2d(L(sai_name)) if sai_name else jnp.zeros((ny, nx), dtype=jnp.float64)
    provenance["sai"] = f"wrfinput {sai_name}" if sai_name else "cold-init: 0"

    # snow column via WRF SNOW_INIT (module_sf_noahmpdrv.F:2339-2439). For the
    # snow-free case (SNODEP<0.025) ISNOW=0, snow ZSNSO=0, soil ZSNSO=cumsum(zsoil).
    isnow, tsno, snice, snliq, zsnso = _wrf_snow_init(sneqv, snowh, tg, zsoil, ny, nx)
    provenance["snow"] = "WRF SNOW_INIT from SNOW/SNOWH/TG/ZSOIL (corpus pre-init over land)"

    sneqvo = jnp.zeros((ny, nx), dtype=jnp.float64)   # NOAHMP_INIT:2147
    albold = jnp.full((ny, nx), 0.65, dtype=jnp.float64)  # NOAHMP_INIT:2148
    tauss = jnp.zeros((ny, nx), dtype=jnp.float64)        # NOAHMP_INIT (TAUSSXY=0)
    provenance["sneqvo_albold_tauss"] = "NOAHMP_INIT: 0 / 0.65 / 0"

    # smcwtd: deep below-bottom soil moisture. NOAHMP_INIT (opt_run=3, no
    # groundwater) sets SMCWTDXY = bottom SMOIS layer.
    smcwtd = smois[-1]
    provenance["smcwtd"] = "NOAHMP_INIT opt_run=3: bottom SMOIS layer"

    # exchange coeffs: NOAHMP_INIT seeds CMXY=CHXY=0 (the driver overwrites them
    # with the sfclay-supplied CH/CM each step via the coupler seed). Use a small
    # positive floor so the very first step is well-posed before sfclay seeds.
    cm = jnp.full((ny, nx), 1.0e-4, dtype=jnp.float64)
    ch = jnp.full((ny, nx), 1.0e-4, dtype=jnp.float64)
    provenance["cm_ch"] = "NOAHMP_INIT: ~0 (sfclay seeds each step); 1e-4 floor at t=0"

    # surface diagnostics carried for coupler/writer
    t_skin = tsk
    qsfc = jnp.zeros((ny, nx), dtype=jnp.float64)   # OUTPUT diagnostic; recomputed each step
    znt = jnp.full((ny, nx), 0.05, dtype=jnp.float64)
    emiss = jnp.full((ny, nx), 0.97, dtype=jnp.float64)
    albedo = albold
    sfcrunoff = jnp.zeros((ny, nx), dtype=jnp.float64)
    udrunoff = jnp.zeros((ny, nx), dtype=jnp.float64)

    land_state = NoahMPLandState(
        tslb=tslb, smois=smois, sh2o=sh2o, smcwtd=smcwtd,
        isnow=isnow, tsno=tsno, snice=snice, snliq=snliq, zsnso=zsnso,
        snowh=snowh, sneqv=sneqv, sneqvo=sneqvo, tauss=tauss, albold=albold,
        tv=tv, tg=tg, tah=tah, eah=eah, canliq=canliq, canice=canice,
        fwet=fwet, lai=lai, sai=sai, cm=cm, ch=ch,
        t_skin=t_skin, qsfc=qsfc, znt=znt, emiss=emiss, albedo=albedo,
        sfcrunoff=sfcrunoff, udrunoff=udrunoff,
    )

    meta = {
        "source": "warm-start from corpus wrfinput",
        "run_dir": str(run_dir),
        "domain": domain,
        "wrfinput_file": str(run.wrfinput_file(domain)),
        "grid_shape_yx": [int(ny), int(nx)],
        "n_land_cells": int(np.sum(np.asarray(is_land))),
        "table_dir": str(tdir),
        "prognostic_state_real_from_corpus": sorted(
            v for v in (
                "TSLB", "SMOIS", "SH2O", "SNOW", "SNOWH", "CANWAT", "LAI", "XSAI",
                "TSK", "TMN", "SHDMAX", "VEGFRA", "ISLTYP", "IVGTYP", "LU_INDEX",
            ) if has(v)
        ),
        "init_note": (
            "corpus wrfinput is the PRE-NOAHMP_INIT state; canopy/snow prognostic "
            "XY fields are 0 over land, so they are built via a faithful "
            "module_sf_noahmpdrv.F NOAHMP_INIT replica (TV/TG/TAH=TSK, EAH=2000, "
            "CANLIQ=CANWAT, SNOW_INIT for ZSNSO/ISNOW/TSNO). This is the SAME "
            "land carry WRF integrates from."
        ),
        "cold_init_provenance": provenance,
    }
    return land_state, static, meta


def _wrf_snow_init(swe, snodep, tg, zsoil, ny: int, nx: int):
    """Vectorised WRF SNOW_INIT (module_sf_noahmpdrv.F:2339-2439).

    Returns ``(isnow, tsno, snice, snliq, zsnso)`` with snow layers (NSNOW) and
    soil layers (NSOIL). For SNODEP<0.025 m (the Canary snow-free case) ISNOW=0,
    snow layers are empty, and ZSNSO over soil is the cumulative ZSOIL interface
    depth. The deeper-snow branches are implemented faithfully for completeness.
    """
    swe = jnp.asarray(swe, dtype=jnp.float64)
    snodep = jnp.asarray(snodep, dtype=jnp.float64)
    tg = jnp.asarray(tg, dtype=jnp.float64)

    # ISNOW from SNODEP thresholds (note ISNOW is the NEGATIVE top-layer index).
    isnow = jnp.where(
        snodep < 0.025, 0,
        jnp.where(snodep <= 0.05, -1,
        jnp.where(snodep <= 0.10, -2,
        jnp.where(snodep <= 0.25, -2,
        jnp.where(snodep <= 0.45, -3, -3)))),
    ).astype(jnp.int32)

    # Per-layer snow thicknesses DZSNO for layers index 0 (top), -1, -2 (deepest).
    # Map WRF's [-2,-1,0] to array rows [0,1,2] = layers [-2,-1,0].
    z0 = jnp.where(snodep <= 0.05, snodep,
         jnp.where(snodep <= 0.10, snodep / 2.0,
         jnp.where(snodep <= 0.25, snodep - 0.05,
         jnp.where(snodep <= 0.45, 0.5 * (snodep - 0.05), snodep - 0.20 - 0.05))))
    zm1 = jnp.where(snodep <= 0.05, 0.0,
          jnp.where(snodep <= 0.10, snodep / 2.0,
          jnp.where(snodep <= 0.25, 0.05,
          jnp.where(snodep <= 0.45, 0.5 * (snodep - 0.05), 0.20))))
    zm2 = jnp.where(snodep <= 0.25, 0.0, 0.05)
    no_snow = snodep < 0.025
    z0 = jnp.where(no_snow, 0.0, z0)
    zm1 = jnp.where(no_snow, 0.0, zm1)
    zm2 = jnp.where(no_snow, 0.0, zm2)
    dzsno = jnp.stack([zm2, zm1, z0], axis=0)  # rows = layers [-2,-1,0]

    # active mask per layer row: row r (layer L=r-2) is active iff L > ISNOW-1
    # i.e. L >= ISNOW+1.  layer index for row r is (r - 2).
    layer_idx = jnp.arange(NSNOW)[:, None, None] - 2  # [-2,-1,0]
    active = layer_idx >= (isnow[None] + 1)
    safe_snodep = jnp.where(snodep > 0.0, snodep, 1.0)
    snice_rate = swe / safe_snodep
    snice = jnp.where(active, dzsno * snice_rate[None], 0.0)
    snliq = jnp.zeros((NSNOW, ny, nx), dtype=jnp.float64)
    tsno = jnp.where(active, jnp.broadcast_to(tg[None], (NSNOW, ny, nx)), 0.0)

    # DZSNSO: snow layers = -dzsno (active only); soil layers from zsoil.
    dzsnso_snow = jnp.where(active, -dzsno, 0.0)  # (NSNOW, ny, nx)
    dz_soil = jnp.empty((NSOIL,), dtype=jnp.float64)
    dz_soil = dz_soil.at[0].set(zsoil[0])
    for iz in range(1, NSOIL):
        dz_soil = dz_soil.at[iz].set(zsoil[iz] - zsoil[iz - 1])
    dzsnso_soil = jnp.broadcast_to(dz_soil[:, None, None], (NSOIL, ny, nx))
    dzsnso = jnp.concatenate([dzsnso_snow, dzsnso_soil], axis=0)  # (NSNOW+NSOIL, ny, nx)

    # ZSNSO = cumulative sum of DZSNSO over inactive-masked snow + soil. WRF starts
    # the cumulative sum at the topmost ACTIVE layer (ISNOW+1); inactive snow rows
    # contribute 0, so a plain cumsum of the masked dzsnso yields the same soil
    # interface depths and leaves inactive snow rows at 0.
    zsnso = jnp.cumsum(dzsnso, axis=0)
    # zero out the inactive (empty) snow rows so they read 0, matching WRF.
    snow_inactive = jnp.concatenate(
        [~active, jnp.zeros((NSOIL, ny, nx), dtype=bool)], axis=0)
    zsnso = jnp.where(snow_inactive, 0.0, zsnso)
    return isnow, tsno, snice, snliq, zsnso


# WRF NOAHMP_INIT soil-ice constants (module_sf_noahmpdrv.F:1988-1990).
_HLICE = 3.335e5   # latent heat of fusion [J/kg]
_GRAV_INIT = 9.81  # gravity used in the NOAHMP_INIT FK expression [m/s2]
_T0 = 273.15       # triple point [K]


def _noahmp_init_sh2o(smois, tslb, isltyp, ivgtyp, parameters):
    """Liquid soil water SH2O per WRF NOAHMP_INIT (module_sf_noahmpdrv.F:2069-2106).

    For each soil layer:
      * glacier veg tile (IVGTYP==ISICE): SH2O = 0 (all frozen);
      * TSLB >= 273.149 K: SH2O = SMOIS (all liquid);
      * TSLB <  273.149 K: SH2O = min(FK, SMOIS) with the explicit frozen-water
        cap ``FK = ((HLICE/(GRAV*(-PSISAT)))*((TSLB-T0)/TSLB))**(-1/BEXP) * SMCMAX``,
        floored at 0.02 (WRF :2096).
    Returns SH2O shaped like ``smois`` (NSOIL, ny, nx). Per-soil-type BEXP/SMCMAX/
    PSISAT are gathered from the (1-based) parameter tables by ISLTYP; PSISAT is
    the positive matric-potential magnitude (WRF uses ``-PSISAT``).
    """
    smois = jnp.asarray(smois, dtype=jnp.float64)
    tslb = jnp.asarray(tslb, dtype=jnp.float64)
    st = jnp.clip(jnp.asarray(isltyp, dtype=jnp.int32), 0, None)

    def _g(name):
        tab = jnp.asarray(getattr(parameters, name), dtype=jnp.float64)
        idx = jnp.clip(st, 0, tab.shape[0] - 1)
        return tab[idx]            # (ny, nx)

    bexp = _g("bexp")
    smcmax = _g("smcmax")
    psisat = _g("psisat")          # positive magnitude

    # clamp SMOIS to porosity (WRF :2089) and broadcast soil params over layers
    smois = jnp.minimum(smois, smcmax[None, ...])
    bexp_l = jnp.broadcast_to(bexp[None, ...], smois.shape)
    smcmax_l = jnp.broadcast_to(smcmax[None, ...], smois.shape)
    psisat_l = jnp.broadcast_to(psisat[None, ...], smois.shape)

    valid = (bexp_l > 0.0) & (smcmax_l > 0.0) & (psisat_l > 0.0)
    frozen = tslb < 273.149
    safe_bexp = jnp.where(bexp_l > 0.0, bexp_l, 1.0)
    safe_tslb = jnp.where(tslb > 1.0, tslb, 1.0)
    # FK base = (HLICE/(GRAV*(-PSISAT))) * ((TSLB-T0)/TSLB). For frozen soil both
    # factors are negative so the product is positive; the warm branch never reads
    # FK, so floor the base to keep the fractional power NaN-safe everywhere.
    fk_base = (_HLICE / (_GRAV_INIT * (-psisat_l))) * ((safe_tslb - _T0) / safe_tslb)
    fk_base = jnp.maximum(fk_base, 1.0e-30)
    fk = (fk_base ** (-1.0 / safe_bexp)) * smcmax_l
    fk = jnp.maximum(fk, 0.02)
    sh2o_frozen = jnp.minimum(fk, smois)
    sh2o = jnp.where(valid & frozen, sh2o_frozen, smois)

    # glacier (ISICE) tile: SMOIS=1 / SH2O=0 over land. ISICE from parameters.
    isice = int(getattr(parameters, "isice", 15))
    is_glacier = jnp.asarray(ivgtyp, dtype=jnp.int32) == isice
    sh2o = jnp.where(is_glacier[None, ...], 0.0, sh2o)
    return sh2o


def build_noahmp_params(static: NoahMPStatic):
    """Pre-build the per-run Noah-MP energy/radiation parameter bundles ONCE.

    Returns ``(energy_params, rad_params, nroot)``. ``nroot`` is the CONCRETE static
    root-depth slice bound (the energy kernel uses ``range(nroot)``); it is carried
    separately so the operational scan can reattach it to the (otherwise-traced)
    pre-built energy params without re-running the frozen ``build_energy_params``
    inside jit. Must be called EAGERLY (outside jit) with concrete ``static``.
    """
    from gpuwrf.physics.noahmp.noahmp_driver import build_energy_params
    scalar_shape = jnp.asarray(static.xland, dtype=jnp.float64).shape
    energy, rad = build_energy_params(static, scalar_shape)
    return energy, rad, int(energy.nroot)


__all__ = ["build_noahmp_land_state", "build_noahmp_params", "DEFAULT_TABLE_DIR"]
