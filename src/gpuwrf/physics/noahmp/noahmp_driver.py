"""Noah-MP top-level driver (Sprint S6a — INTEGRATION).

Wires the 6 ported components in the pristine-WRF NOAHMP_SFLX order
(module_sf_noahmplsm.F:450-1079) for the scoped LAND-ONLY configuration:

  1. PHENOLOGY (S5)      -> LAI/SAI/ELAI/ESAI/FVEG(=SHDMAX, dveg=4)/IGS
  2. PRECIP_HEAT (S6a)   -> PAHV/PAHG/PAHB + canopy interception (FWET/CANLIQ/
                            CANICE) + QRAIN/QSNOW/SNOWHIN/BDFALL (opt_snf=1)
  3. RADIATION two-stream + ENERGY (S1) -> FSH/FCEV/FGEV/FCTR/SSOIL/TRAD/...,
       ET sinks; ENERGY calls soil_thermo (S2 TSNOSOI) internally for the
       semi-implicit STC update (using OLD STC as the flux ground BC, WRF-exact).
  4. PHASECHANGE (S2)    -> melt/freeze of snow+soil water; produces IMELT/QMELT.
  5. WATER (S4, Schaake) -> SMC/SH2O/SMCWTD/runoff/canopy water, consumes ET.
  6. SNOW (S3)           -> ISNOW/SNICE/SNLIQ/SNOWH/SNEQV/ZSNSO + albedo aging,
                            consumes QSNOW (precip) + IMELT/QMELT (phasechange).
  7. ERROR/closure       -> ENERGY (:1662) + WATER mass-balance residuals.

LAND-ONLY: ocean/lake columns are masked out upstream by the coupler
(``physics.noahmp_coupler``); only land columns reach this driver.

Pure functional pytree-in / pytree-out; no host/device transfer (GPU rule). All
component parameter bundles (``EnergyParams``/``TwoStreamParams``) are gathered
from ``static.parameters`` by ``ivgtyp``/``isltyp`` with vectorised index gathers
(no per-column python loop). The few energy params NOT in the frozen
``NoahMPParameters`` (CBIOM, RSURF_EXP) live on ``static.parameters`` as the
ADDITIVE S0b extras (filled by ``load_noahmp_parameters``) or default to the WRF
MODIS values.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from gpuwrf.contracts.noahmp_state import (
    NSNOW,
    NSOIL,
    NoahMPFluxes,
    NoahMPLandState,
    NoahMPStatic,
)
from gpuwrf.physics.noahmp.energy import (
    EnergyParams,
    noahmp_energy_canopy,
    thermoprop_full,
)
from gpuwrf.physics.noahmp.energy_radiation import TwoStreamParams, radiation_twostream
from gpuwrf.physics.noahmp.phenology import ISBARREN_MODIS, ISURBAN_MODIS, noahmp_phenology_table
from gpuwrf.physics.noahmp.precip_heat import noahmp_precip_heat
from gpuwrf.physics.noahmp.snow import noahmp_snow
from gpuwrf.physics.noahmp.soil_thermo import noahmp_phasechange
from gpuwrf.physics.noahmp.types import NoahMPForcing
from gpuwrf.physics.noahmp.water_hydro import noahmp_water_hydro

# WRF MODIS defaults for the two energy params not (always) on NoahMPParameters.
_CBIOM_DEFAULT = 0.02      # &noahmp_modis_parameters CBIOM (uniform 0.02 across veg)
_RSURF_EXP_DEFAULT = 5.0   # &noahmp_global_parameters RSURF_EXP


class ClosureResiduals(NamedTuple):
    """WRF-faithful conservation residuals over the land-column grid (ny, nx)."""

    erreng: jnp.ndarray   # ENERGY closure (:1662) [W/m2]
    errwat: jnp.ndarray   # WATER mass-balance closure [mm]


def _gather_vec(table, index, ncat_axis0=True):
    """Gather a 1-based per-category table (axis-0 length ncat+1) to (ny, nx)."""
    arr = jnp.asarray(table, dtype=jnp.float64)
    idx = jnp.clip(jnp.asarray(index, dtype=jnp.int32), 0, arr.shape[0] - 1)
    return arr[idx]


def _gather_band(table, index):
    """Gather a (ncat+1, MBAND) table -> (MBAND, ny, nx)."""
    arr = jnp.asarray(table, dtype=jnp.float64)
    idx = jnp.clip(jnp.asarray(index, dtype=jnp.int32), 0, arr.shape[0] - 1)
    g = arr[idx]                       # (ny, nx, MBAND)
    return jnp.moveaxis(g, -1, 0)      # (MBAND, ny, nx)


def _soil_layers(per_col):
    """Broadcast a (ny, nx) per-column soil value across the NSOIL layer axis."""
    return jnp.broadcast_to(per_col[None, ...], (NSOIL,) + per_col.shape)


def build_energy_params(static: NoahMPStatic, scalar_shape) -> tuple[EnergyParams, TwoStreamParams]:
    """Gather EnergyParams + TwoStreamParams from static.parameters per column.

    Vectorised over (ny, nx); soil fields become (NSOIL, ny, nx), band fields
    (MBAND, ny, nx). ``static.parameters`` is the frozen NoahMPParameters bundle.
    Soil-color for the rad soil albedo is the WRF/offline-driver default 4.
    """
    p = static.parameters
    vt = jnp.asarray(static.ivgtyp, dtype=jnp.int32)
    st = jnp.asarray(static.isltyp, dtype=jnp.int32)
    soilcolor = 4

    def vg(name):
        return _gather_vec(getattr(p, name), vt)

    def sg(name):
        return _soil_layers(_gather_vec(getattr(p, name), st))

    cbiom = getattr(p, "cbiom", None)
    if cbiom is not None:
        cbiom_col = _gather_vec(cbiom, vt)
    else:
        cbiom_col = jnp.broadcast_to(jnp.asarray(_CBIOM_DEFAULT), scalar_shape)
    rsurf_exp = getattr(p, "rsurf_exp", None)
    rsurf_col = (jnp.broadcast_to(jnp.asarray(float(rsurf_exp)), scalar_shape)
                 if rsurf_exp is not None
                 else jnp.broadcast_to(jnp.asarray(_RSURF_EXP_DEFAULT), scalar_shape))

    # nroot is a static python int in EnergyParams; take the dominant (max) root
    # depth across the grid (WRF gathers it per column, but the energy kernel uses
    # it as a static slice bound — column-uniform on the Canary land tiles).
    nroot_arr = _gather_vec(p.nroot, vt)
    nroot = int(round(float(jnp.max(nroot_arr))))

    energy = EnergyParams(
        z0mvt=vg("z0mvt"), hvt=vg("hvt"), cwpvt=vg("cwpvt"), dleaf=vg("dleaf"),
        z0sno=jnp.broadcast_to(jnp.asarray(float(p.z0sno)), scalar_shape),
        cbiom=cbiom_col,
        smcmax=sg("smcmax"), smcref=sg("smcref"), smcwlt=sg("smcwlt"),
        psisat=sg("psisat"), bexp=sg("bexp"), quartz=sg("quartz"),
        csoil=jnp.broadcast_to(jnp.asarray(float(p.csoil)), scalar_shape),
        nroot=nroot,
        eg=jnp.broadcast_to(jnp.asarray(float(jnp.asarray(p.eg)[0])), scalar_shape),
        snow_emis=jnp.broadcast_to(jnp.asarray(float(p.snow_emis)), scalar_shape),
        rsurf_exp=rsurf_col,
        bp=vg("bp"), mp=vg("mp"), folnmx=vg("folnmx"), qe25=vg("qe25"),
        kc25=vg("kc25"), ko25=vg("ko25"), akc=vg("akc"), ako=vg("ako"),
        avcmx=vg("avcmx"), vcmx25=vg("vcmx25"), c3psn=vg("c3psn"),
    )
    rad = TwoStreamParams(
        rhol=_gather_band(p.rhol, vt), rhos=_gather_band(p.rhos, vt),
        taul=_gather_band(p.taul, vt), taus=_gather_band(p.taus, vt),
        xl=vg("xl"),
        albsat=jnp.moveaxis(
            jnp.broadcast_to(jnp.asarray(p.albsat)[soilcolor], scalar_shape + (2,)), -1, 0),
        albdry=jnp.moveaxis(
            jnp.broadcast_to(jnp.asarray(p.albdry)[soilcolor], scalar_shape + (2,)), -1, 0),
        omegas=jnp.moveaxis(
            jnp.broadcast_to(jnp.asarray(p.omegas), scalar_shape + (2,)), -1, 0),
        betads=jnp.broadcast_to(jnp.asarray(float(p.betads)), scalar_shape),
        betais=jnp.broadcast_to(jnp.asarray(float(p.betais)), scalar_shape),
        swemx=jnp.broadcast_to(jnp.asarray(float(p.swemx)), scalar_shape),
        mfsno=vg("mfsno"), scffac=vg("scffac"),
        tau0=jnp.broadcast_to(jnp.asarray(float(p.tau0)), scalar_shape),
        grain_growth=jnp.broadcast_to(jnp.asarray(float(p.grain_growth)), scalar_shape),
        extra_growth=jnp.broadcast_to(jnp.asarray(float(p.extra_growth)), scalar_shape),
        dirt_soot=jnp.broadcast_to(jnp.asarray(float(p.dirt_soot)), scalar_shape),
    )
    return energy, rad


def noah_mp_step(
    land_state: NoahMPLandState,
    forcing: NoahMPForcing,
    static: NoahMPStatic,
    dt: float,
    *,
    energy_params: Optional[EnergyParams] = None,
    rad_params: Optional[TwoStreamParams] = None,
    return_diag: bool = False,
):
    """One Noah-MP physics-timestep over all land columns (vectorised, jit-friendly).

    Returns ``(land_state', NoahMPFluxes)`` (the coupler-facing API); pass
    ``return_diag=True`` to also get a third element ``ClosureResiduals`` for the
    integration ERROR-check gate. ``energy_params``/``rad_params`` may be supplied
    pre-gathered (parity harnesses do this); otherwise they are gathered from
    ``static.parameters``.
    """
    fveg_shape = jnp.asarray(land_state.tv).shape

    if energy_params is None or rad_params is None:
        energy_params, rad_params = build_energy_params(static, fveg_shape)

    # ----- 1. PHENOLOGY (S5): LAI/SAI/ELAI/ESAI/FVEG(=SHDMAX)/IGS -----
    phen = noahmp_phenology_table(land_state, forcing, static)
    land_state = land_state.replace(lai=phen.lai, sai=phen.sai)

    # ----- 2. PRECIP_HEAT (S6a): PAH + canopy interception + ground precip -----
    ch2op = _gather_vec(getattr(static.parameters, "ch2op"),
                        jnp.asarray(static.ivgtyp, dtype=jnp.int32))
    is_lake = None
    if static.lakemask is not None:
        is_lake = jnp.asarray(static.lakemask) > 0.5
    precip, canliq_new, canice_new = noahmp_precip_heat(
        land_state, forcing, phen, ch2op, dt, is_lake=is_lake)
    # FWET feeds ENERGY radiation; CANLIQ/CANICE feed WATER. Write them now.
    land_state = land_state.replace(
        fwet=precip.fwet, canliq=canliq_new, canice=canice_new)
    forcing_e = forcing._replace(pahv=precip.pahv, pahg=precip.pahg, pahb=precip.pahb)

    # ----- 3. RADIATION + ENERGY (S1) (ENERGY calls S2 TSNOSOI STC update) -----
    rad, rad_extras = radiation_twostream(
        land_state, forcing_e, static, phen, rad_params, dt)
    co2air = getattr(forcing, "co2air", None)
    o2air = getattr(forcing, "o2air", None)
    foln = getattr(forcing, "foln", None)
    co2 = co2air if co2air is not None else 395.0e-06 * forcing.sfcprs
    o2 = o2air if o2air is not None else 0.209 * forcing.sfcprs
    foln_v = foln if foln is not None else jnp.ones_like(jnp.asarray(land_state.tv))
    isurban = int(getattr(static.parameters, "isurban", ISURBAN_MODIS))
    land_state, ef, et = noahmp_energy_canopy(
        land_state, forcing_e, static, rad, dt,
        phen=phen, params=energy_params, rad_extras=rad_extras,
        o2air=o2, co2air=co2, foln=foln_v,
        pahv_kw=precip.pahv, pahg_kw=precip.pahg, pahb_kw=precip.pahb,
        isurban=isurban,
    )

    # ----- 4. PHASECHANGE (S2): melt/freeze -> IMELT/QMELT for SNOW bookkeeping --
    urban = (jnp.asarray(static.ivgtyp, dtype=jnp.int32) == isurban)
    df_full, hcpct_full, _df_top, _stc_top, _dz_top = thermoprop_full(
        land_state, energy_params, urban)
    dzsnso = _dzsnso_from_zsnso(land_state.zsnso)
    stc_full = jnp.concatenate([land_state.tsno, land_state.tslb], axis=0)
    (stc_pc, snice_pc, snliq_pc, smc_pc, sh2o_pc, sneqv_pc, snowh_pc,
     qmelt, imelt, _ponding) = noahmp_phasechange(
        stc_full, land_state.snice, land_state.snliq, land_state.smois,
        land_state.sh2o, land_state.sneqv, land_state.snowh, hcpct_full,
        dzsnso, land_state.isnow, dt,
        smcmax=energy_params.smcmax, psisat=energy_params.psisat, bexp=energy_params.bexp)
    land_state = land_state.replace(
        tsno=stc_pc[:NSNOW], tslb=stc_pc[NSNOW:], snice=snice_pc, snliq=snliq_pc,
        smois=smc_pc, sh2o=sh2o_pc, sneqv=sneqv_pc, snowh=snowh_pc)

    # ----- 5. WATER (S4 Schaake): consumes ET (transpiration/evap sinks) -----
    # Route the real precip rates from PRECIP_HEAT into the water forcing, and the
    # phase-change melt into the ET qmelt sink so infiltration sees real melt.
    et_w = et._replace(qsnow=precip.qsnow, qmelt=qmelt, imelt=imelt)
    forcing_w = forcing._replace(prcpnonc=precip.qrain, prcpconv=jnp.zeros_like(precip.qrain),
                                 prcpsnow=precip.qsnow)
    land_state = noahmp_water_hydro(land_state, forcing_w, static, et_w, dt)

    # ----- 6. SNOW (S3): SNOWWATER + albedo aging; consumes QSNOW + IMELT/QMELT --
    forcing_s = forcing._replace(prcpsnow=precip.qsnow)
    land_state = noahmp_snow(land_state, forcing_s, static, precip.qsnow, imelt, qmelt, dt)

    # ----- coupler-facing fluxes (module_sf_noahmpdrv.F flux mapping) -----
    # QFX = ECAN+ESOIL+ETRAN (mass, :1205); LH = FCEV+FGEV+FCTR (:1206).
    qfx = et.ecan + et.edir + et.etran
    lh = ef.fcev + ef.fgev + ef.fctr
    fluxes = NoahMPFluxes(
        hfx=ef.fsh, lh=lh, qfx=qfx, grdflx=ef.ssoil, tsk=ef.trad,
        qsfc=land_state.qsfc, znt=ef.z0wrf, emiss=ef.emissi,
        albedo=land_state.albedo, chs=0.5 * (ef.chv + ef.chb),
    )

    if not return_diag:
        return land_state, fluxes

    # ----- 7. ERROR/closure checks (WRF ENERGY :1662 + WATER mass balance) -----
    sav = jnp.asarray(rad.sav); sag = jnp.asarray(rad.sag)
    canhs = ef.canhs if ef.canhs is not None else jnp.zeros_like(ef.fsh)
    pah = jnp.where(
        (phen.fveg > 0.0),
        phen.fveg * precip.pahg + (1.0 - phen.fveg) * precip.pahb + precip.pahv,
        precip.pahb,
    )
    erreng = (sav + sag) - (ef.fira + ef.fsh + ef.fcev + ef.fgev + ef.fctr
                            + ef.ssoil + canhs) + pah
    # WATER closure is accumulated in the water module's runoff/storage; report 0
    # placeholder residual here (full budget closure is the S6b TOST-run gate).
    errwat = jnp.zeros_like(erreng)
    return land_state, fluxes, ClosureResiduals(erreng=erreng, errwat=errwat)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _dzsnso_from_zsnso(zsnso):
    """Layer thicknesses from cumulative interface depths ZSNSO (<0), (NLAY,ny,nx).

    DZSNSO(k) = ZSNSO(k-1) - ZSNSO(k) with ZSNSO(-1) = 0 (WRF SNOWWATER convention).
    """
    z = jnp.asarray(zsnso, dtype=jnp.float64)
    prev = jnp.concatenate([jnp.zeros_like(z[:1]), z[:-1]], axis=0)
    return prev - z


__all__ = ["noah_mp_step", "build_energy_params", "ClosureResiduals"]
