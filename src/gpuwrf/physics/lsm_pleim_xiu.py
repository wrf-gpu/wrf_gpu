"""WRF Pleim-Xiu land-surface model (``sf_surface_physics=7``).

This is the v0.17 per-scheme lane port of WRF's ``phys/module_sf_pxlsm.F``
column physics: the subroutine ``SURFPX`` (lines 1059-1503) and its callee
``QFLUX`` (lines 1505-1671).  PX is the ISBA-style two-layer (1 cm / 1 m) soil
moisture + soil temperature + canopy-water land model of Pleim & Xiu (1995) and
Xiu & Pleim (2001), coupled to the PX surface layer and ACM2 PBL.

The column kernel ``pxlsm_column`` is a pure ``jnp`` function,
``jax.jit``/``jax.vmap``-traceable (no Python branching on data, no host
allocations).  It is faithful to ``SURFPX``/``QFLUX`` for the operational WRF
configuration the generated pristine-WRF fp64 oracle exercises:

* ``NUDGEX = 0`` -- soil/temperature nudging disabled (the SMASS path is not
  ported; WGNUDG = W2NUDG = T2NUD = 0, module_sf_pxlsm.F:1358-1361),
* ``XICE = 0`` -- no sea-ice (the ``XICE1 > 0.5`` skin-T cap, line 1400, and the
  ice soil-moisture path never trigger),
* land columns (``IFLAND < 1.5``) advance ``TG``/``T2``/``WG``/``W2``/``WR``;
  ocean/lake columns (``IFLAND >= 1.5``) keep their soil carry unchanged exactly
  as every ``IF (IFLAND .LT. 1.5)`` block in ``SURFPX`` skips them, while the
  unconditional RADNET/HFX/QFX/RA/QST/EG/2m-diagnostic code still runs.

The 11 ISBA soil constants (``wwlt, wfc, wres, cgsat, wsat, b, c1sat, c2r,
asoil, jp, c3``) and the per-column vegetation/surface parameters are STATIC
inputs (the WRF ``SOILPROP``/``VEGELAND`` setup is out of scope -- the oracle
sets them per case, the operational hook supplies them), mirroring how
``lsm_noah_classic`` takes its ``REDPRM`` block as an explicit static bundle.

The driver prepares ``QSS`` (ground saturation mixing ratio, with the
below-freezing ``ES`` branch, module_sf_pxlsm.F:624-629) and ``BETAP`` (the
bare-soil beta factor, lines 632-634) from the *current* ``TG``/``WG`` before
each sub-step, so both are recomputed inside the ``lax.fori_loop`` over the
sub-time steps, exactly like the Fortran ``DO IT=1,NTSPS`` loop (lines 621-657).
The sub-step count ``ntsps = INT(DT/(DTPBLX+1e-6)+1)`` (DTPBLX = 40 s) depends
only on the time step, so it is a Python-level static argument.

All math is fp64.  Cited to
``<USER_HOME>/src/wrf_pristine/WRF/phys/module_sf_pxlsm.F`` (SURFPX 1059-1503,
QFLUX 1505-1671, driver input prep 426-667) and
``share/module_model_constants.F`` for the thermodynamic constants.
"""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

from typing import NamedTuple

import jax
from jax import config
import jax.numpy as jnp


configure_jax_x64()


# --- module_model_constants.F constants (verified against the pristine tree) ---
RD = 287.04          # module_sf_pxlsm.F:7  RD (used for ROVCP, local to PX)
CPD = 1004.67        # module_sf_pxlsm.F:7  CPD
ROVCP = RD / CPD     # module_sf_pxlsm.F:10
R_D = 287.0          # share/module_model_constants.F:19  r_d (used in CQ1/CQ2)
STBOLT = 5.67051e-8  # share/module_model_constants.F:84  Stefan-Boltzmann
KARMAN = 0.4         # share/module_model_constants.F:82  von Karman
SVP1 = 0.6112        # cb
SVP2 = 17.67
SVP3 = 29.65
SVPT0 = 273.15
EP_2 = 287.0 / 461.6  # share/module_model_constants.F:81  R_d/R_v (= 0.6217504)

# --- SURFPX / module_sf_pxlsm.F parameters ---
CRANKP = 0.5          # Crank-Nicolson factor (line 12)
DENW = 1000.0         # water density kg/m^3 (line 14)
TAUINV = 1.0 / 86400.0  # 1/day (line 15)
T2TFAC = 1.0 / 10.0   # deep-soil response factor (line 16)
PI = 3.1415926        # line 17
PR0 = 0.95            # line 18
ZOBS = 1.5            # screen height (m) (line 1194)
GAMAH = 16.0          # line 1196
BETAH = 5.0           # line 1197
SIGF = 0.5            # rain interception (line 1198)
CV = 1.2e-5           # vegetation heat-capacity factor K-m^2/J (line 1192)
CT_SNOW = 2.0e-5      # line 1204
CT_IMPERV = 3.268e-6  # line 1224
DWAT = 0.2178         # cm^2/s at 273.15 K (line 1199)

# --- QFLUX parameters (module_sf_pxlsm.F:1582-1584) ---
RSMAX = 5000.0        # s/m
FTMIN = 0.0000001     # m/s
F3MIN = 0.25

DTPBLX = 40.0         # max PX sub-step (s), module_sf_pxlsm.F:326


def ntsps_substeps(dt: float) -> int:
    """WRF PX sub-step count ``INT(DT/(DTPBLX+1e-6)+1)`` (module_sf_pxlsm.F:618).

    Depends only on the surface time step, so it resolves to a Python int at
    trace time and the kernel loops it with ``lax.fori_loop`` -- no
    data-dependent control flow.
    """

    return int(dt / (DTPBLX + 0.000001) + 1.0)


class PleimXiuStatic(NamedTuple):
    """Read-only per-column PX static inputs (soil + vegetation + surface).

    The 11 ISBA soil constants follow the Noilhan & Mahfouf (1996) analytic
    formulation WRF ``SOILPROP`` produces (module_sf_pxlsm.F:1904-1915); the
    vegetation/surface parameters come from ``VEGELAND``.  All are constant over
    the sub-time loop.
    """

    # vegetation / surface
    vegfrc: jax.Array      # vegetation coverage (0..1)
    lai: jax.Array         # leaf area index
    imperv: jax.Array      # impervious fraction (%)
    canfra: jax.Array      # canopy/tree fraction (%)
    rstmin: jax.Array      # minimum stomatal resistance (s/m)
    emissi: jax.Array      # surface emissivity
    znt: jax.Array         # roughness length (m)
    wetfra: jax.Array      # wetlands fraction (0..1)
    hc_snow: jax.Array     # snow heat-capacity term (passthrough)
    snow_fra: jax.Array    # fractional snow area (0..1)
    # ISBA soil constants
    wwlt: jax.Array        # wilting point
    wfc: jax.Array         # field capacity
    wres: jax.Array        # residual soil moisture
    cgsat: jax.Array       # saturated soil heat-capacity coefficient
    wsat: jax.Array        # saturation soil moisture
    b: jax.Array           # Clapp-Hornberger B
    c1sat: jax.Array       # surface-layer C1 at saturation
    c2r: jax.Array         # restoring C2 reference
    asoil: jax.Array       # equilibrium-moisture A coefficient
    jp: jax.Array          # equilibrium-moisture exponent
    c3: jax.Array          # drainage coefficient
    ds1: jax.Array         # surface soil-layer thickness (m), DZS(1)
    ds2: jax.Array         # root-zone soil-layer thickness (m), DZS(2)


def _qflux(
    dens1, qv1, ta1, rg, raw, qss,
    vegfrc, isnow, ifland, lai, betap,
    wg, w2, wr,
    rstmin, wwlt, wfc, rsoil, rinc,
):
    """Faithful port of WRF ``QFLUX`` (module_sf_pxlsm.F:1505-1671).

    Returns ``(eg, er, etr, cq4, rs, fass, sigg)``.  Bare-soil evaporation
    ``EG`` is computed unconditionally; canopy ``ER``/transpiration ``ETR`` and
    the implicit-TG coefficient ``CQ4`` only contribute on the
    ``IFLAND<1.5 .AND. VEGFRC>0`` vegetated-land path (line 1597 onward), which
    we realise with ``jnp.where`` masks.
    """

    veg_land = (ifland < 1.5) & (vegfrc > 0.0)

    # GROUND EVAPORATION (DEPOSITION). RSOIL -> 0 when QSS < QV1 (dew). (1593-1594)
    rsoil_eff = jnp.where(qss < qv1, 0.0, rsoil)
    eg = dens1 * (qss - qv1) * (
        (1.0 - vegfrc) / (raw + rsoil_eff)
        + vegfrc / (raw + rinc + rsoil_eff)
    )

    # CANOPY (1597-1612). WRMAX>0 needs VEGFRC>0; guard the divide.
    wrmax = 0.2e-3 * vegfrc * lai
    wrmax_safe = jnp.where(wrmax > 0.0, wrmax, 1.0)
    delta = jnp.where(wr <= 0.0, 0.0, wr / wrmax_safe)
    sigg = jnp.where(qss >= qv1, delta, 1.0)
    er = jnp.where(veg_land, dens1 * vegfrc * sigg * (qss - qv1) / raw, 0.0)

    # TRANSPIRATION (1617-1668).
    # F1 radiation factor (1620-1627).
    f1max = jnp.where(rstmin > 130.0, 1.0 - 0.02 * lai, 1.0 - 0.07 * lai)
    par = 0.45 * rg * 4.566
    f1 = f1max * (1.0 - jnp.exp(-0.0017 * par))
    f1 = jnp.maximum(f1, rstmin / RSMAX)
    # F2 soil-moisture factor (1630-1633).
    w2avail = w2 - wwlt
    w2mxav = wfc - wwlt
    f2 = 1.0 / (1.0 + jnp.exp(-5.0 * (w2avail / w2mxav - (w2mxav / 3.0 + wwlt))))
    # F4 air-temperature factor (1637-1641).
    f4 = jnp.where(
        ta1 <= 302.15,
        1.0 / (1.0 + jnp.exp(-0.41 * (ta1 - 282.05))),
        1.0 / (1.0 + jnp.exp(0.5 * (ta1 - 314.0))),
    )
    ftot = lai * f1 * f2 * f4

    # Stomatal + aerodynamic conductance, humidity factor F3 (1648-1668).
    fshelt = 1.0
    gs = ftot / (rstmin * fshelt)
    ga = 1.0 / raw
    f3 = 0.5 * (gs - ga + jnp.sqrt(
        ga * ga + ga * gs * (4.0 * qv1 / qss - 2.0) + gs * gs)) / gs
    f3 = jnp.minimum(jnp.maximum(f3, F3MIN), 1.0)
    rs_veg = 1.0 / (gs * f3)
    fx = jnp.where(rg < 0.00001, 0.0, 30.0 * f1 * f4 * lai / (rstmin * fshelt))
    etr_veg = dens1 * vegfrc * (1.0 - sigg) * (qss - qv1) / (raw + rs_veg)
    cq4_veg = dens1 * vegfrc * ((1.0 - sigg) / (raw + rs_veg) + sigg / raw)

    etr = jnp.where(veg_land, etr_veg, 0.0)
    cq4 = jnp.where(veg_land, cq4_veg, 0.0)
    fass = jnp.where(veg_land, fx, 0.0)
    rs = jnp.where(veg_land, rs_veg, 0.0)
    # SIGG only meaningful on the veg path; off-path ER/ETR use it = 0 anyway.
    sigg = jnp.where(veg_land, sigg, 0.0)
    return eg, er, etr, cq4, rs, fass, sigg


def _surfpx_substep(
    dtpbl, ifland, isnow, soldn, gsw, lwdn, emissi, z1, mol, znt, ust,
    psurf, dens1, qv1, qss, ta1, theta1, precip, cpair, betap,
    vegfrc, lai, imperv, canfra, rstmin, hc_snow, snow_fra, wetfra,
    wwlt, wfc, wres, cgsat, wsat, b, c1sat, c2r, asoil, jp, c3, ds1, ds2,
    qst12, tg, t2, wg, w2, wr,
):
    """One ``SURFPX`` sub-step (NUDGEX=0, XICE=0). module_sf_pxlsm.F:1059-1503.

    Returns updated ``(tg, t2, wg, w2, wr)`` plus the diagnostic outputs
    ``(radnet, grdflx, hfx, qfx, lh, eg, er, etr, qst, capg, rs, ra, ta2, qa2,
    psih)``.  ``QSS``/``BETAP`` are supplied by the caller (recomputed each
    sub-step from the current TG/WG).
    """

    is_land = ifland < 1.5

    radnet = soldn - (emissi * (STBOLT * tg ** 4 - lwdn))     # line 1228
    cpot = (100.0 / psurf) ** ROVCP                           # line 1230
    thetag = tg * cpot

    zol = z1 / mol                                            # line 1233
    zobol = ZOBS / mol
    zntol = znt / mol

    # PSIH stability functions (1238-1263).
    y = (1.0 - GAMAH * zol) ** 0.5
    y0 = (1.0 - GAMAH * zobol) ** 0.5
    ynt = (1.0 - GAMAH * zntol) ** 0.5
    psih15_u = 2.0 * jnp.log((y + 1.0) / (y0 + 1.0))
    psih_u = 2.0 * jnp.log((y + 1.0) / (ynt + 1.0))
    psiob_u = 2.0 * jnp.log((y0 + 1.0) / (ynt + 1.0))
    phih_u = 1.0 / y

    psih_s = jnp.where((zol - zntol) <= 1.0,
                       -BETAH * (zol - zntol),
                       1.0 - BETAH - (zol - zntol))
    psiob_s = jnp.where((zobol - zntol) <= 1.0,
                        -BETAH * (zobol - zntol),
                        1.0 - BETAH - (zobol - zntol))
    psih15_s = psih_s - psiob_s
    phih_s = jnp.where(zol <= 1.0, 1.0 + BETAH * zol, BETAH + zol)

    unstable = mol < 0.0
    psih = jnp.where(unstable, psih_u, psih_s)
    psiob = jnp.where(unstable, psiob_u, psiob_s)
    psih15 = jnp.where(unstable, psih15_u, psih15_s)
    phih = jnp.where(unstable, phih_u, phih_s)

    # RA / RAH / RAW (1268-1270).
    ra = PR0 * (jnp.log(z1 / znt) - psih) / (KARMAN * ust)
    rah = ra + 5.0 / ust
    raw = ra + 4.503 / ust

    # RSOIL over land (1271-1278). XICE=0 so the land branch always applies here.
    ldry = 1.75 * ds1 * (jnp.exp((1.0 - wg / wsat) ** 5) - 1.0) / 1.718
    dp = DWAT * 1.0e-4 * wsat ** 2 * (1.0 - wres / wsat) ** (2.0 + 3.0 / b)
    rsoil = jnp.where(is_land, ldry / dp, 0.0)

    # Wetlands soil-moisture floor on W2 (1283-1287).
    wetsat = wsat
    sm2 = wetfra * wetsat
    w2 = jnp.where(is_land, jnp.maximum(sm2, w2), w2)

    # In-canopy resistance (1291-1292).
    hcan = znt * 10.0
    rinc = 14.0 * lai * hcan / ust

    # Moisture flux (1296-1300).
    eg, er, etr, cq4, rs, fass, sigg = _qflux(
        dens1, qv1, ta1, soldn, raw, qss,
        vegfrc, isnow, ifland, lai, betap,
        wg, w2, wr, rstmin, wwlt, wfc, rsoil, rinc,
    )

    # Total evaporation and turbulent moisture scale (1305-1306).
    et = eg + er + etr
    qst = -et / (dens1 * ust)

    # Latent heat: sublimation unless warm & no snow (1308-1310).
    lv = jnp.where((isnow < 0.5) & (tg > 273.15),
                   (2.501 - 0.00237 * (tg - 273.15)) * 1.0e6,
                   2.83e6)
    qfx = et
    lh = lv * qfx

    # Surface sensible heat flux (1318-1320).
    tst = (theta1 - thetag) / (ust * rah)
    hf = ust * tst
    hfx = jnp.maximum(-dens1 * cpair * hf, -250.0)

    # Diagnosed 2-m T and Q (1325-1334).
    qst1 = 0.5 * (qst + qst12 / phih)
    ta2 = (thetag + tst * (PR0 / KARMAN * (jnp.log(ZOBS / znt) - psiob) + 5.0)) / cpot
    qa2 = qv1 - qst1 * PR0 / KARMAN * (jnp.log(z1 / ZOBS) - psih15)
    qa2 = jnp.where(qa2 <= 0.0, qv1, qa2)
    # RH2MOD (1332-1334) is only needed by nudging (NUDGEX=0); skipped.

    # Heat capacity / ground heat flux over land (1336-1351).
    w2cg = jnp.maximum(w2, wwlt)
    cg = cgsat * 1.0e-6 * (wsat / w2cg) ** (0.5 * b / jnp.log(10.0))
    imf = jnp.maximum(0.0, imperv / 100.0)
    vegf = (1.0 - imf) * vegfrc
    soilf = (1.0 - imf) * (1.0 - vegfrc)
    ct = 1.0 / (imf / CT_IMPERV + vegf / CV + soilf / cg)
    ct = 1.0 / (snow_fra / CT_SNOW + (1.0 - snow_fra) / ct)
    capg = jnp.where(is_land, 1.0 / ct, 0.0)
    soilflx = 2.0 * PI * TAUINV * (tg - t2)
    grdflx = jnp.where(is_land, soilflx / ct, 0.0)

    # NUDGEX=0 -> WGNUDG = W2NUDG = T2NUD = 0 (1358-1361).
    wgnudg = 0.0
    w2nudg = 0.0
    t2nud = 0.0

    # --- Crank-Nicolson TG / T2 update over land (1381-1408). ---
    cq1 = (1.0 - 0.622 * lv * CRANKP / (R_D * tg)) * qss
    cq2 = 0.622 * lv * qss * CRANKP / (R_D * tg * tg)
    cq3bg = dens1 * (1.0 - vegfrc) / (raw + rsoil)
    cq3vw = dens1 * vegfrc * sigg / raw
    cq3vg = dens1 * vegfrc / (raw + rsoil + rinc)
    cq3 = cq3bg + cq3vw + cq3vg
    coeffnp1 = 1.0 + dtpbl * CRANKP * (
        4.0 * emissi * STBOLT * tg ** 3 * ct
        + dens1 * cpair / rah * cpot * ct
        + 2.0 * PI * TAUINV
    ) + dtpbl * (ct * lv * cq2 * (cq3 + cq4))
    coeffn = ct * (
        gsw + emissi * (STBOLT * (4.0 * CRANKP - 1.0) * tg * tg * tg * tg + lwdn)
        + dens1 * cpair / rah * (theta1 - (1.0 - CRANKP) * thetag)
        - lv * (cq3 * (cq1 - qv1) + cq4 * (cq1 - qv1))
    ) - 2.0 * PI * TAUINV * ((1.0 - CRANKP) * tg - t2)
    tsnew = (tg + dtpbl * coeffn) / coeffnp1
    # XICE=0 -> the `XICE1>0.5` skin cap (line 1400) never applies.
    tshlf = 0.5 * (tsnew + tg)
    t2new = (t2 + dtpbl * TAUINV * T2TFAC * (tshlf - (1.0 - CRANKP) * t2)
             + dtpbl * t2nud) / (1.0 + dtpbl * TAUINV * T2TFAC * CRANKP)
    tg_land = tsnew
    t2_land = t2new

    # --- WR / W2 update over land (XICE=0) (1413-1454). ---
    wrmax = 0.2e-3 * vegfrc * lai
    wrmax_pos = wrmax > 0.0
    wrmax_safe = jnp.where(wrmax_pos, wrmax, 1.0)
    pc0 = vegfrc * SIGF * precip
    dwr = (wrmax - wr) / dtpbl
    pnet = pc0 - er / DENW
    roff = jnp.where(pnet > dwr, pnet - dwr, 0.0)
    pc = jnp.where(pnet > dwr, pc0 - roff, pc0)
    # branch on QSS<QV1 (dew) vs evaporation (1427-1436)
    tendwr = pc - er / DENW
    wrnew_dew = wr + dtpbl * tendwr
    cof1 = dens1 / DENW * vegfrc * (qss - qv1) / raw
    cfnp1wr = 1.0 + dtpbl * cof1 * CRANKP / wrmax_safe
    cfnwr = pc - cof1 * (1.0 - CRANKP) * wr / wrmax_safe
    wrnew_evap = (wr + dtpbl * cfnwr) / cfnp1wr
    wrnew = jnp.where(qss < qv1, wrnew_dew, wrnew_evap)
    # WRMAX<=0 -> PC=0, WRNEW=0 (1437-1440)
    pc = jnp.where(wrmax_pos, pc, 0.0)
    wrnew = jnp.where(wrmax_pos, wrnew, 0.0)

    # W2 tendency (1443-1450).
    pg = DENW * (precip - pc)
    tendw2 = (1.0 / (DENW * ds2) * (pg - eg - etr)
              - c3 / ds2 * TAUINV * jnp.maximum(0.0, w2 - wfc)
              + (w2nudg + wgnudg) / ds2)
    w2new = w2 + dtpbl * tendw2
    w2new = jnp.minimum(w2new, wsat)
    w2new = jnp.maximum(w2new, wres)
    w2hlf = 0.5 * (w2 + w2new)
    w2_land = w2new
    wr_land = jnp.minimum(wrmax, wrnew)

    # --- Surface soil-moisture WG update over land, XICE=0 (1459-1500). ---
    # Snow path: WG=WSAT (1462-1463). Otherwise the Sakaguchi-Zeng diffusion.
    w2rel = w2hlf / wsat
    # C1: WG>WWLT vs Giard-Bazile (1466-1481).
    c1_wet = ds1 * c1sat * (wsat / wg) ** (0.5 * b + 1.0)
    zy2 = c1sat * (wsat / wwlt) ** (0.5 * b + 1.0)
    c1max0 = (1.19 * wwlt - 5.09) * tg - 146.0 * wwlt + 1786.0
    c1max = jnp.maximum(jnp.maximum(c1max0, zy2), 10.0)
    zly = jnp.log(c1max / 10.0)
    zza = -jnp.log(zy2 / 10.0)
    zzb = 2.0 * wwlt * zly
    zdel = 4.0 * (zly + zza) * zly * wwlt ** 2
    za = (-zzb + jnp.sqrt(zdel)) / (2.0 * zza)
    zb = za ** 2 / zly
    c1_dry = ds1 * c1max * jnp.exp(-(wg - za) ** 2 / zb)
    c1 = jnp.where(wg > wwlt, c1_wet, c1_dry)

    c2 = c2r * w2hlf / (wsat - w2hlf + 1.0e-11)
    weq = jnp.where(
        w2hlf >= wsat,
        wsat,
        w2hlf - asoil * wsat * w2rel ** jp * (1.0 - w2rel ** (8.0 * jp)),
    )
    cfnp1 = 1.0 + dtpbl * c2 * TAUINV * CRANKP
    cfn = (c1 / (DENW * ds1) * (pg - eg)
           - c2 * TAUINV * ((1.0 - CRANKP) * wg - weq) + wgnudg / ds1)
    wgnew = jnp.maximum((wg + dtpbl * cfn) / cfnp1, wres)
    wg_snow = wsat
    wg_diff = jnp.minimum(wgnew, wsat)
    wg_land = jnp.where(isnow > 0.5, wg_snow, wg_diff)

    # Apply land updates; water columns keep their carry unchanged.
    tg_out = jnp.where(is_land, tg_land, tg)
    t2_out = jnp.where(is_land, t2_land, t2)
    w2_out = jnp.where(is_land, w2_land, w2)
    wr_out = jnp.where(is_land, wr_land, wr)
    wg_out = jnp.where(is_land, wg_land, wg)

    return (
        tg_out, t2_out, wg_out, w2_out, wr_out,
        radnet, grdflx, hfx, qfx, lh, eg, er, etr, qst, capg, rs, ra, ta2, qa2,
        psih,
    )


def _qss_betap(tg, wg, psurf, ifland, isnow, wfc):
    """Driver-side QSS + BETAP prep (module_sf_pxlsm.F:624-634).

    QSS from the current ground temperature with the below-freezing ``ES``
    branch; BETAP the Lee & Pielke (1992) bare-soil beta factor from the current
    surface soil moisture.  Both are recomputed every sub-step.
    """

    es = jnp.where(
        tg <= SVPT0,
        SVP1 * jnp.exp(22.514 - 6.15e3 / tg),
        SVP1 * jnp.exp(SVP2 * (tg - SVPT0) / (tg - SVP3)),
    )
    qss = es * 0.622 / (psurf - es)

    betap_wet = 0.25 * (1.0 - jnp.cos(wg / wfc * PI)) ** 2
    use_betap = (ifland < 1.5) & (isnow < 0.5) & (wg <= wfc)
    betap = jnp.where(use_betap, betap_wet, 1.0)
    return qss, betap


def pxlsm_column(
    soldn,      # downward shortwave (W/m^2), SOLDN = GSW/(1-albedo)
    gsw,        # net shortwave at ground (W/m^2)
    lwdn,       # downward longwave (W/m^2)
    z1,         # lowest half-level height (m), ZLVL = 0.5*dz
    rmol,       # 1/MOL raw (pre-clamp); MOLX = clamp(1/RMOL, +-1000)
    ust_in,     # friction velocity (m/s) (pre MAX(.,0.005))
    psurf,      # surface pressure in cb (= PSFC/1000)
    dens1,      # air density at first layer (kg/m^3)
    qv1,        # air vapor mixing ratio at first layer (kg/kg)
    ta1,        # air temperature at first layer (K)
    theta1,     # potential temperature at first layer (K)
    precip,     # precipitation rate (m/s)
    cpair,      # specific heat of moist air, CPD*(1+0.84*qv1) (J/kg/K)
    qst12,      # 2-level turbulent moisture-gradient scale
    ifland,     # 1=land, 2=water mask
    isnow,      # snow-cover flag (>=0.5 = snow)
    xice,       # sea-ice fraction (pinned to 0 per contract; not branched on)
    *,
    static: PleimXiuStatic,
    dt: float,
    ntsps: int,
    tg, t2, wg, w2, wr,
):
    """Faithful per-column port of WRF ``SURFPX`` + ``QFLUX`` (NUDGEX=0, XICE=0).

    ``ifland`` (1=land, 2=water) and ``isnow`` (snow-cover flag) are per-column
    masks.  ``xice`` is accepted for interface symmetry but the contract pins it
    to 0 (no sea-ice); it is not branched on.  The sub-time loop recomputes
    ``QSS``/``BETAP`` each step, exactly like the Fortran ``DO IT`` loop.
    """

    del xice  # XICE=0 per contract; the ice branches are unreachable.
    s = static
    ustar = jnp.maximum(ust_in, 0.005)             # driver line 506

    # MOLX clamp (driver lines 604-610). RMOL>0 -> min(1/RMOL,1000);
    # RMOL<0 -> max(1/RMOL,-1000); RMOL==0 -> 1000.
    inv = jnp.where(rmol != 0.0, 1.0 / jnp.where(rmol != 0.0, rmol, 1.0), 1000.0)
    molx = jnp.where(
        rmol > 0.0, jnp.minimum(inv, 1000.0),
        jnp.where(rmol < 0.0, jnp.maximum(inv, -1000.0), 1000.0),
    )

    rnsub = 1.0 / float(ntsps)
    dtpbl = dt * rnsub

    def substep(_it, carry):
        tg_c, t2_c, wg_c, w2_c, wr_c, diag = carry
        del diag
        qss, betap = _qss_betap(tg_c, wg_c, psurf, ifland, isnow, s.wfc)
        out = _surfpx_substep(
            dtpbl, ifland, isnow, soldn, gsw, lwdn, s.emissi, z1, molx, s.znt,
            ustar, psurf, dens1, qv1, qss, ta1, theta1, precip, cpair, betap,
            s.vegfrc, s.lai, s.imperv, s.canfra, s.rstmin, s.hc_snow, s.snow_fra,
            s.wetfra, s.wwlt, s.wfc, s.wres, s.cgsat, s.wsat, s.b, s.c1sat,
            s.c2r, s.asoil, s.jp, s.c3, s.ds1, s.ds2, qst12,
            tg_c, t2_c, wg_c, w2_c, wr_c,
        )
        tg_n, t2_n, wg_n, w2_n, wr_n = out[0], out[1], out[2], out[3], out[4]
        return (tg_n, t2_n, wg_n, w2_n, wr_n, out[5:])

    diag0 = tuple(jnp.zeros_like(tg) for _ in range(15))
    tg, t2, wg, w2, wr, diag = jax.lax.fori_loop(
        0, ntsps, substep, (tg, t2, wg, w2, wr, diag0)
    )
    (radnet, grdflx, hfx, qfx, lh, eg, er, etr, qst, capg, rs, ra, ta2, qa2,
     psih) = diag

    # Driver post-processing (module_sf_pxlsm.F:659-667).
    is_land = ifland < 1.5
    grdflx = jnp.where(is_land, grdflx, 0.0)
    tsk = tg                                       # land: TSK=TSLB(1); water: TG kept
    canwat = wr * 1000.0
    raw_final = ra + 4.503 / ustar
    qsfc = qfx * raw_final / dens1 + qv1

    return {
        "tg": tg, "t2": t2, "wg": wg, "w2": w2, "wr": wr,
        "tsk": tsk, "canwat": canwat, "qsfc": qsfc,
        "radnet": radnet, "grdflx": grdflx, "hfx": hfx, "qfx": qfx, "lh": lh,
        "eg": eg, "er": er, "etr": etr, "qst": qst, "capg": capg, "rs": rs,
        "ra": ra, "ta2": ta2, "qa2": qa2,
    }


def pxlsm_columns(
    soldn, gsw, lwdn, z1, rmol, ust_in, psurf, dens1, qv1, ta1, theta1,
    precip, cpair, qst12, ifland, isnow,
    tg, t2, wg, w2, wr,
    *,
    # static vegetation/surface
    vegfrc, lai, imperv, canfra, rstmin, emissi, znt, wetfra, hc_snow, snow_fra,
    # static ISBA soil constants
    wwlt, wfc, wres, cgsat, wsat, b, c1sat, c2r, asoil, jp, c3, ds1, ds2,
    dt: float,
    ntsps: int,
    xice=0.0,
):
    """Batched (``vmap``) PX-LSM entry. 2-D fields are ``(ncol,)``.

    The static soil/vegetation parameters are per-column ``(ncol,)`` vectors
    (mirroring ``slab_columns``/``sflx``); ``dt`` is a Python float and
    ``ntsps`` a Python static int.  Returns a dict of stacked outputs with the
    soil carry exposed as ``tslb`` = stack(TG, T2) and ``smois`` = stack(WG, W2).
    """

    def one(
        soldn_, gsw_, lwdn_, z1_, rmol_, ust_, psurf_, dens1_, qv1_, ta1_,
        theta1_, precip_, cpair_, qst12_, ifland_, isnow_, tg_, t2_, wg_, w2_,
        wr_, xice_,
        vegfrc_, lai_, imperv_, canfra_, rstmin_, emissi_, znt_, wetfra_,
        hc_snow_, snow_fra_, wwlt_, wfc_, wres_, cgsat_, wsat_, b_, c1sat_,
        c2r_, asoil_, jp_, c3_, ds1_, ds2_,
    ):
        st = PleimXiuStatic(
            vegfrc=vegfrc_, lai=lai_, imperv=imperv_, canfra=canfra_,
            rstmin=rstmin_, emissi=emissi_, znt=znt_, wetfra=wetfra_,
            hc_snow=hc_snow_, snow_fra=snow_fra_, wwlt=wwlt_, wfc=wfc_,
            wres=wres_, cgsat=cgsat_, wsat=wsat_, b=b_, c1sat=c1sat_, c2r=c2r_,
            asoil=asoil_, jp=jp_, c3=c3_, ds1=ds1_, ds2=ds2_,
        )
        return pxlsm_column(
            soldn_, gsw_, lwdn_, z1_, rmol_, ust_, psurf_, dens1_, qv1_, ta1_,
            theta1_, precip_, cpair_, qst12_, ifland_, isnow_, xice_,
            static=st, dt=dt, ntsps=ntsps,
            tg=tg_, t2=t2_, wg=wg_, w2=w2_, wr=wr_,
        )

    ncol = jnp.shape(tg)[0]
    xice_arr = jnp.broadcast_to(jnp.asarray(xice, dtype=jnp.float64), (ncol,))
    out = jax.vmap(one, in_axes=(0,) * 45)(
        soldn, gsw, lwdn, z1, rmol, ust_in, psurf, dens1, qv1, ta1, theta1,
        precip, cpair, qst12, ifland, isnow, tg, t2, wg, w2, wr, xice_arr,
        vegfrc, lai, imperv, canfra, rstmin, emissi, znt, wetfra, hc_snow,
        snow_fra, wwlt, wfc, wres, cgsat, wsat, b, c1sat, c2r, asoil, jp, c3,
        ds1, ds2,
    )
    out["tslb"] = jnp.stack([out["tg"], out["t2"]], axis=-1)
    out["smois"] = jnp.stack([out["wg"], out["w2"]], axis=-1)
    return out


__all__ = [
    "PleimXiuStatic",
    "ntsps_substeps",
    "pxlsm_column",
    "pxlsm_columns",
]
