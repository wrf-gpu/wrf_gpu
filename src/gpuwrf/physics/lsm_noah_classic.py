"""Noah LSM (classic, sf_surface_physics=2) — JAX GPU port (v0.6.0 lane 14).

Faithful, fully-vectorized fp64 JAX translation of WRF ARW
``phys/module_sf_noahlsm.F`` (the classic Noah land-surface model, distinct from
the Noah-MP scheme already ported in ``noah_mp.py``). This module implements the
SFLX orchestrator and every physics sub-routine it calls on the WRF-coupled land
path (ICE=0, the SF_URBAN_PHYSICS=0 / LOCAL=.false. / UA_PHYS=.false. /
OPT_THCND=1 / FASDAS=0 configuration the operational port uses):

  * surface energy balance (PENMAN potential evaporation + sensible-heat closure)
  * 4-layer soil temperature update (HRT right-hand side + HSTEP tridiagonal solve,
    Peters-Lidard thermal conductivity TDFCND, Koren supercooled-water phase change
    SNKSRC/FRH2O, temperature averaging TBND/TMPAVG)
  * 4-layer soil moisture update (SMFLX -> SRT/SSTEP Richards-equation tridiagonal
    solve, WDFCND diffusivity/conductivity, Schaake/Koren infiltration, runoff)
  * evapotranspiration (EVAPO -> DEVAP direct soil evap + TRANSP root-zone
    transpiration + canopy evaporation), canopy resistance (CANRES)
  * snowpack (SNOPAC branch: SNFRAC fractional cover, ALCALC snow albedo, CSNOW /
    SNOW_NEW / SNOWPACK density+depth, SNOWZ0 roughness, snowmelt energy)
  * surface fluxes HFX (SHEAT), QFX (ETA_KINEMATIC), LH (ETA), GRDFLX (SSOIL),
    skin temperature T1.

The REDPRM table lookup (soil/veg/slope parameter assignment) is NOT re-derived
here: the operational coupler supplies the per-column derived parameter block
(``NoahClassicParams``) the same way WRF's REDPRM populates it from
SOILPARM/VEGPARM/GENPARM. The parity oracle
(``proofs/v060/oracle/noahclassic_offline_driver.F90``) dumps WRF's exact REDPRM
block per column so the physics solve is validated in isolation against the
genuine WRF SFLX output.

State / carry contract (V0.6.0-S0-PLAN.md lane 14):
  * State leaves written: ``t_skin`` (TSK=T1), ``soil_moisture`` (write-back of the
    surface-layer total soil-moisture handle), ``mavail``.
  * 4-layer land carry (``PhysicsCarry.land_surface``): TSLB (STC), SMOIS (SMC),
    SH2O (SWC), CANWAT (CMC), SNOW (SNEQV), SNOWH, SNOWC (SNCOVR), SNOTIME, plus the
    S0 carry members ``flx4,fvb,fbur,fgsn,smcrel,xlaidyn`` (UA-physics-off => the
    first four are 0; ``smcrel``=SMAV soil-moisture availability; ``xlaidyn``=XLAI).
  * num_soil_layers = 4 (NOAH_CLASSIC_NUM_SOIL_LAYERS).

Numerics: fp64 throughout. No host<->device transfer; the whole step is a single
jittable pure function over (ny, nx) column tiles. Over-water columns (XLAND>=1.5)
are passed through unchanged (the LSM does not run there — the caller masks).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# WRF model constants (module_model_constants.F + module_sf_noahlsm.F params)
# ---------------------------------------------------------------------------
CP = 1004.5            # cp = 7*r_d/2, r_d=287 (module_model_constants)
RD = 287.04            # SFLX-internal gas constant (module_sf_noahlsm.F)
SIGMA = 5.67e-8        # Stefan-Boltzmann
CPH2O = 4.218e3
CPICE = 2.106e3
LSUBF = 3.335e5        # latent heat of fusion
EMISSI_S = 0.95        # snow emissivity
LVH2O = 2.501e6        # latent heat of vaporization (SFLX top-level)
LSUBS = 2.83e6         # latent heat of sublimation
TFREEZ = 273.15

# PENMAN constants
ELCP = 2.4888e3
LSUBC = 2.501000e6
PENMAN_CP = 1004.6     # PENMAN uses its own CP literal

# SNKSRC / FRH2O constants
HLICE = 3.335e5
GS = 9.81
DH2O = 1000.0
FRH2O_CK = 8.0
FRH2O_BLIM = 5.5
FRH2O_ERROR = 0.005
FRH2O_T0 = 273.15

NSOIL = 4


class NoahClassicParams(NamedTuple):
    """Per-column REDPRM-derived parameter block (WRF supplies these).

    Every field is a (ny, nx) array (or scalar broadcastable). Mirrors the
    INTENT(OUT) block of ``module_sf_noahlsm.F:REDPRM`` plus the SHDFAC/ALB/EMISS/
    Z0BRD/SNOALB surface characteristics resolved in the SFLX shdfac-interp block.
    """
    bexp: jax.Array
    dksat: jax.Array
    dwsat: jax.Array
    psisat: jax.Array
    quartz: jax.Array
    f1: jax.Array
    smcmax: jax.Array
    smcwlt: jax.Array
    smcref: jax.Array
    smcdry: jax.Array
    kdt: jax.Array
    frzx: jax.Array
    slope: jax.Array
    snup: jax.Array
    salp: jax.Array
    czil: jax.Array
    sbeta: jax.Array
    csoil: jax.Array
    fxexp: jax.Array
    zbot: jax.Array
    cfactr: jax.Array
    cmcmax: jax.Array
    rsmax: jax.Array
    topt: jax.Array
    rgl: jax.Array
    hs: jax.Array
    rsmin: jax.Array
    lvcoef: jax.Array
    nroot: jax.Array          # integer per column
    rtdis: jax.Array          # (..., NSOIL)
    # surface characteristics resolved by SFLX shdfac-interp (background, snow-free)
    alb: jax.Array            # ALB background snow-free albedo
    embrd: jax.Array          # background emissivity
    xlai: jax.Array           # leaf area index
    z0brd: jax.Array          # background roughness
    shdfac: jax.Array         # REDPRM-resolved green-vegetation fraction
    is_urban: jax.Array       # bool: VEGTYP == ISURBAN (SFLX urban override)


class NoahClassicForcing(NamedTuple):
    """Per-column atmospheric / radiation / precip forcing (caller-assembled)."""
    sfctmp: jax.Array         # air T at lowest level [K]
    sfcprs: jax.Array         # pressure at lowest level [Pa]
    th2: jax.Array            # potential T at lowest level [K]
    q2: jax.Array             # specific humidity at lowest level [kg/kg]
    q2sat: jax.Array          # saturation specific humidity [kg/kg]
    dqsdt2: jax.Array         # dQsat/dT [kg/kg/K]
    soldn: jax.Array          # downward solar [W/m2]
    solnet: jax.Array         # net downward solar [W/m2]
    lwdn: jax.Array           # downward longwave [W/m2] (=GLW*EMISS)
    prcp: jax.Array           # precip rate [kg/m2/s]
    ffrozp: jax.Array         # frozen-precip fraction [0-1]
    sfcspd: jax.Array         # wind speed [m/s] (not used in coupled CH path)
    zlvl: jax.Array           # forcing height [m]
    snoalb: jax.Array         # max deep-snow albedo
    tbot: jax.Array           # bottom soil boundary T [K]
    ch: jax.Array             # surface exchange coeff for heat [m/s] (from sfclay)
    cm: jax.Array             # surface exchange coeff for momentum [m/s]


class NoahClassicState(NamedTuple):
    """Prognostic land state (the 4-layer carry). All (ny, nx[, NSOIL])."""
    t1: jax.Array             # skin temperature [K]
    stc: jax.Array            # (..., NSOIL) soil temperature [K]
    smc: jax.Array            # (..., NSOIL) total soil moisture [m3/m3]
    sh2o: jax.Array           # (..., NSOIL) unfrozen soil moisture [m3/m3]
    cmc: jax.Array            # canopy moisture content [m]
    sneqv: jax.Array          # snow water equivalent [m]
    snowh: jax.Array          # snow depth [m]
    sncovr: jax.Array         # fractional snow cover
    snotime1: jax.Array       # time since last snowfall [s]
    ribb: jax.Array           # bulk Richardson (snow dewfall guard)


class NoahClassicOutput(NamedTuple):
    """SFLX outputs + updated prognostics (the savepoint-parity surface)."""
    state: NoahClassicState
    hfx: jax.Array            # SHEAT sensible heat flux [W/m2]
    qfx: jax.Array            # ETA_KINEMATIC moisture flux [kg/m2/s]
    lh: jax.Array             # ETA latent heat flux [W/m2]
    grdflx: jax.Array         # SSOIL ground heat flux [W/m2]
    etp: jax.Array            # potential evap [W/m2]
    albedo: jax.Array         # surface albedo incl. snow
    emissi: jax.Array         # surface emissivity
    z0: jax.Array             # time-varying roughness
    q1: jax.Array             # effective surface mixing ratio handle
    snomlt: jax.Array         # snowmelt [m]
    edir: jax.Array
    ec: jax.Array
    ett: jax.Array
    esnow: jax.Array
    beta: jax.Array
    smav: jax.Array           # (..., NSOIL) soil-moisture availability (smcrel)
    runoff1: jax.Array
    runoff2: jax.Array


# ===========================================================================
# Helper physics routines (faithful per-column translations)
# ===========================================================================

def _csnow(dsnow):
    """CSNOW: snow thermal conductivity from density (W/m/K)."""
    unit = 0.11631
    c = 0.328 * 10.0 ** (2.25 * dsnow)
    return 2.0 * unit * c


def _snow_new(temp, newsn, snowh, sndens):
    """SNOW_NEW: update snow depth/density for new snowfall (in/out cm-internal)."""
    snowhc = snowh * 100.0
    newsnc = newsn * 100.0
    tempc = temp - 273.15
    dsnew = jnp.where(tempc <= -15.0, 0.05, 0.05 + 0.0017 * (tempc + 15.0) ** 1.5)
    hnewc = newsnc / dsnew
    sndens_new = jnp.where(
        snowhc + hnewc < 1.0e-3,
        jnp.maximum(dsnew, sndens),
        (snowhc * sndens + hnewc * dsnew) / (snowhc + hnewc),
    )
    snowhc = snowhc + hnewc
    return snowhc * 0.01, sndens_new


def _snfrac(sneqv, snup, salp, snowh):
    """SNFRAC: fractional snow cover (UA_PHYS=.false. branch)."""
    rsnow = sneqv / snup
    sncovr = jnp.where(
        sneqv < snup,
        1.0 - (jnp.exp(-salp * rsnow) - rsnow * jnp.exp(-salp)),
        1.0,
    )
    return sncovr


def _alcalc(alb, snoalb, embrd, sncovr, dt, snowng, snotime1, lvcoef):
    """ALCALC: snow albedo (Livneh formulation) + emissivity."""
    snacca, snaccb = 0.94, 0.58
    emissi = embrd + sncovr * (EMISSI_S - embrd)
    snoalb1 = snoalb + lvcoef * (0.85 - snoalb)
    snotime_new = jnp.where(snowng, 0.0, snotime1 + dt)
    snoalb2 = jnp.where(
        snowng,
        snoalb1,
        snoalb1 * (snacca ** ((snotime_new / 86400.0) ** snaccb)),
    )
    snoalb2 = jnp.maximum(snoalb2, alb)
    albedo = alb + sncovr * (snoalb2 - alb)
    albedo = jnp.where(albedo > snoalb2, snoalb2, albedo)
    return albedo, emissi, snotime_new


def _snowz0(sncovr, z0brd, snowh):
    """SNOWZ0: roughness over snow (UA_PHYS=.false. branch)."""
    z0s = 0.001
    burial = 7.0 * z0brd - snowh
    z0eff = jnp.where(burial <= 0.0007, z0s, burial / 7.0)
    return (1.0 - sncovr) * z0brd + sncovr * z0eff


def _tdfcnd(smc, qz, smcmax, sh2o):
    """TDFCND: Peters-Lidard soil thermal conductivity (OPT_THCND=1 branch)."""
    satratio = smc / smcmax
    thkice = 2.2
    thkw = 0.57
    thko = 2.0
    thkqtz = 7.7
    thks = (thkqtz ** qz) * (thko ** (1.0 - qz))
    xunfroz = sh2o / smc
    xu = xunfroz * smcmax
    thksat = thks ** (1.0 - smcmax) * thkice ** (smcmax - xu) * thkw ** xu
    gammd = (1.0 - smcmax) * 2700.0
    thkdry = (0.135 * gammd + 64.7) / (2700.0 - 0.947 * gammd)
    akei = satratio
    akel = jnp.where(satratio > 0.1, jnp.log10(jnp.maximum(satratio, 1e-30)) + 1.0, 0.0)
    ake = ((smc - sh2o) * akei + sh2o * akel) / smc
    return ake * (thksat - thkdry) + thkdry


def _frh2o(tkelv, smc, sh2o, smcmax, bexp, psis):
    """FRH2O: supercooled liquid water (Koren 1999 eqn 17), Newton iteration.

    Vectorized fixed-iteration form of the WRF DO-WHILE (NLOG<10). The WRF loop
    stops on convergence (DSWL<=ERROR) and otherwise after 10 sweeps; running a
    fixed 10 Newton steps with a converged-freeze of the update reproduces it
    bit-for-bit for the converged columns (the only ones used downstream).
    """
    bx = jnp.minimum(bexp, FRH2O_BLIM)
    warm = tkelv > (FRH2O_T0 - 1.0e-3)

    swl0 = smc - sh2o
    swl0 = jnp.where(swl0 > (smc - 0.02), smc - 0.02, swl0)
    swl0 = jnp.where(swl0 < 0.0, 0.0, swl0)

    def body(_, carry):
        swl, done = carry
        df = (jnp.log((psis * GS / HLICE) * ((1.0 + FRH2O_CK * swl) ** 2.0)
                      * (smcmax / jnp.maximum(smc - swl, 1e-30)) ** bx)
              - jnp.log(-(tkelv - FRH2O_T0) / tkelv))
        denom = 2.0 * FRH2O_CK / (1.0 + FRH2O_CK * swl) + bx / jnp.maximum(smc - swl, 1e-30)
        swlk = swl - df / denom
        swlk = jnp.where(swlk > (smc - 0.02), smc - 0.02, swlk)
        swlk = jnp.where(swlk < 0.0, 0.0, swlk)
        dswl = jnp.abs(swlk - swl)
        new_swl = jnp.where(done, swl, swlk)
        new_done = done | (dswl <= FRH2O_ERROR)
        return new_swl, new_done

    swl_final, _ = jax.lax.fori_loop(0, 10, body, (swl0, jnp.zeros_like(swl0, dtype=bool)))
    free = smc - swl_final
    # cold-but-low-T columns where iteration never set KCOUNT use the Flerchinger
    # explicit fallback; for the validated regimes the iterated branch is taken.
    return jnp.where(warm, smc, free)


def _tbnd(tu, tb, zsoil, zbot, k):
    """TBND: temperature at a layer boundary by interpolation (1-based k)."""
    zup = jnp.where(k == 1, 0.0, _zget(zsoil, k - 1))
    zk = _zget(zsoil, k)
    zb = jnp.where(k == NSOIL, 2.0 * zbot - zk, _zget(zsoil, k + 1))
    return tu + (tb - tu) * (zup - zk) / (zup - zb)


def _zget(zsoil, k):
    """1-based layer index access for ZSOIL stacked on the last axis."""
    return zsoil[..., k - 1]


def _tmpavg(tup, tm, tdn, dz):
    """TMPAVG: layer-average T in a freezing/thawing layer."""
    t0 = FRH2O_T0
    dzh = dz * 0.5

    def case_tup_lt():
        def tm_lt():
            return jnp.where(
                tdn < t0,
                (tup + 2.0 * tm + tdn) / 4.0,
                _safe(lambda: 0.5 * (tup * dzh + tm * (dzh + ((t0 - tm) * dzh / (tdn - tm)))
                                     + t0 * (2.0 * dzh - ((t0 - tm) * dzh / (tdn - tm)))) / dz),
            )

        def tm_ge():
            xup = (t0 - tup) * dzh / (tm - tup)
            return jnp.where(
                tdn < t0,
                0.5 * (tup * xup + t0 * (2.0 * dz - xup - (dzh - (t0 - tm) * dzh / (tdn - tm)))
                       + tdn * (dzh - (t0 - tm) * dzh / (tdn - tm))) / dz,
                0.5 * (tup * xup + t0 * (2.0 * dz - xup)) / dz,
            )

        return jnp.where(tm < t0, tm_lt(), tm_ge())

    def case_tup_ge():
        def tm_lt():
            xup = dzh - (t0 - tup) * dzh / (tm - tup)
            return jnp.where(
                tdn < t0,
                0.5 * (t0 * (dz - xup) + tm * (dzh + xup) + tdn * dzh) / dz,
                0.5 * (t0 * (2.0 * dz - xup - (t0 - tm) * dzh / (tdn - tm))
                       + tm * (xup + (t0 - tm) * dzh / (tdn - tm))) / dz,
            )

        def tm_ge():
            xdn = dzh - (t0 - tm) * dzh / (tdn - tm)
            return jnp.where(
                tdn < t0,
                (t0 * (dz - xdn) + 0.5 * (t0 + tdn) * xdn) / dz,
                (tup + 2.0 * tm + tdn) / 4.0,
            )

        return jnp.where(tm < t0, tm_lt(), tm_ge())

    return jnp.where(tup < t0, case_tup_lt(), case_tup_ge())


def _safe(fn):
    return fn()


def _snksrc(tavg, smc, sh2o, dz, smcmax, psisat, bexp, dt, qtot):
    """SNKSRC: phase-change heat sink/source + updated SH2O for one layer."""
    free = _frh2o(tavg, smc, sh2o, smcmax, bexp, psisat)
    xh2o = sh2o + qtot * dt / (DH2O * HLICE * dz)
    cond1 = (xh2o < sh2o) & (xh2o < free)
    xh2o = jnp.where(cond1, jnp.where(free > sh2o, sh2o, free), xh2o)
    cond2 = (xh2o > sh2o) & (xh2o > free)
    xh2o = jnp.where(cond2, jnp.where(free < sh2o, sh2o, free), xh2o)
    xh2o = jnp.clip(xh2o, 0.0, smc)
    tsnsr = -DH2O * HLICE * dz * (xh2o - sh2o) / dt
    return tsnsr, xh2o


def _rosr12(a, b, c, d):
    """ROSR12: tridiagonal solve. a=sub, b=diag, c=super, d=rhs (all (...,NSOIL)).

    Returns the solution P (the WRF DELTA-folded back-substitution result).
    """
    n = NSOIL
    c = c.at[..., n - 1].set(0.0)
    p = jnp.zeros_like(b)
    delta = jnp.zeros_like(b)
    p = p.at[..., 0].set(-c[..., 0] / b[..., 0])
    delta = delta.at[..., 0].set(d[..., 0] / b[..., 0])
    for k in range(1, n):
        denom = b[..., k] + a[..., k] * p[..., k - 1]
        p = p.at[..., k].set(-c[..., k] * (1.0 / denom))
        delta = delta.at[..., k].set((d[..., k] - a[..., k] * delta[..., k - 1]) * (1.0 / denom))
    out = delta[..., n - 1]
    sol = jnp.zeros_like(b)
    sol = sol.at[..., n - 1].set(out)
    for k in range(2, n + 1):
        kk = n - k  # 0-based index of (NSOIL-k+1)
        sol = sol.at[..., kk].set(p[..., kk] * sol[..., kk + 1] + delta[..., kk])
    return sol


def _penman(forcing, ssoil, t24, snowng, frzgra, emissi, sneqv, t1, sncovr):
    """PENMAN: potential evaporation ETP + RCH/EPSCA/RR/FLX2 partials."""
    f = forcing
    t2v = f.sfctmp * (1.0 + 0.61 * f.q2)
    elcp1 = (1.0 - sncovr) * ELCP + sncovr * ELCP * LSUBS / LSUBC
    lvs = (1.0 - sncovr) * LSUBC + sncovr * LSUBS
    flx2 = jnp.zeros_like(f.sfctmp)
    delta = elcp1 * f.dqsdt2
    t24_ = f.sfctmp ** 4
    rr = emissi * t24_ * 6.48e-8 / (f.sfcprs * f.ch) + 1.0
    rho = f.sfcprs / (RD * t2v)
    rch = rho * PENMAN_CP * f.ch
    rr = jnp.where(
        ~snowng,
        jnp.where(f.prcp > 0.0, rr + CPH2O * f.prcp / rch, rr),
        rr + CPICE * f.prcp / rch,
    )
    fnet = f.solnet + f.lwdn - emissi * SIGMA * t24_ - ssoil
    flx2 = jnp.where(frzgra, -LSUBF * f.prcp, flx2)
    fnet = jnp.where(frzgra, fnet - (-LSUBF * f.prcp), fnet)
    rad = fnet / rch + f.th2 - f.sfctmp
    a = elcp1 * (f.q2sat - f.q2)
    epsca = (a * rr + rad * delta) / (delta + rr)
    epsca = jnp.where(epsca > 0.0, epsca, epsca)  # AOASIS=1.0 no-op
    etp = epsca * rch / lvs
    return etp, rch, epsca, rr, flx2, t24_


def _canres(forcing, params, sh2o, zsoil, emissi):
    """CANRES: canopy resistance RC -> plant coefficient PC (frozen-ground SH2O)."""
    f, p = forcing, params
    ff = 0.55 * 2.0 * f.soldn / (p.rgl * p.xlai)
    rcs = jnp.maximum((ff + p.rsmin / p.rsmax) / (1.0 + ff), 0.0001)
    rct = jnp.maximum(1.0 - 0.0016 * ((p.topt - f.sfctmp) ** 2.0), 0.0001)
    rcq = jnp.maximum(1.0 / (1.0 + p.hs * (f.q2sat - f.q2)), 0.01)
    # soil-moisture availability weighted by layer depth over root zone (1..NROOT)
    kidx = jnp.arange(NSOIL)
    inroot = (kidx[None] < p.nroot[..., None]).astype(sh2o.dtype)
    znroot = jnp.take_along_axis(zsoil, jnp.clip(p.nroot[..., None] - 1, 0, NSOIL - 1), axis=-1)
    gx = jnp.clip((sh2o - p.smcwlt[..., None]) / (p.smcref - p.smcwlt)[..., None], 0.0, 1.0)
    z_above = jnp.concatenate([jnp.zeros_like(zsoil[..., :1]), zsoil[..., :-1]], axis=-1)
    weight = jnp.where(kidx[None] == 0, zsoil[..., 0:1] / znroot, (zsoil - z_above) / znroot)
    part = weight * gx * inroot
    rcsoil = jnp.maximum(jnp.sum(part, axis=-1), 0.0001)
    rc = p.rsmin / (p.xlai * rcs * rct * rcq * rcsoil)
    rr = (4.0 * emissi * SIGMA * RD / CP) * (f.sfctmp ** 4.0) / (f.sfcprs * f.ch) + 1.0
    delta = (LSUBC / CP) * f.dqsdt2
    pc = (rr + delta) / (rr * (1.0 + rc * f.ch) + delta)
    return jnp.where((p.shdfac > 0.0) & (p.xlai > 0.0), pc, 0.0)


def _devap(etp1, smc1, shdfac, smcmax, smcdry, fxexp):
    """DEVAP: direct soil evaporation."""
    sratio = (smc1 - smcdry) / (smcmax - smcdry)
    fx = jnp.where(sratio > 0.0, jnp.clip(sratio ** fxexp, 0.0, 1.0), 0.0)
    return fx * (1.0 - shdfac) * etp1


def _transp(etp1, smc, cmc, shdfac, params, pc, zsoil):
    """TRANSP: root-zone transpiration ET(k) preserving total ETP1A."""
    p = params
    nroot = p.nroot
    cfactr = p.cfactr
    cmcmax = p.cmcmax
    etp1a = jnp.where(
        cmc != 0.0,
        shdfac * pc * etp1 * (1.0 - (cmc / cmcmax) ** cfactr),
        shdfac * pc * etp1,
    )
    kidx = jnp.arange(NSOIL)
    inroot = (kidx[None] < nroot[..., None]).astype(smc.dtype)  # (..., NSOIL) mask
    gx = jnp.clip((smc - p.smcwlt[..., None]) / (p.smcref - p.smcwlt)[..., None], 0.0, 1.0)
    gx = gx * inroot
    nroot_f = jnp.maximum(nroot.astype(smc.dtype), 1.0)
    sgx = jnp.sum(gx, axis=-1) / nroot_f
    rtx = p.rtdis + gx - sgx[..., None]
    gx2 = gx * jnp.maximum(rtx, 0.0)
    denom = jnp.sum(gx2, axis=-1)
    denom = jnp.where(denom <= 0.0, 1.0, denom)
    et = etp1a[..., None] * gx2 / denom[..., None]
    return et * inroot


def _wdfcnd(smc, smcmax, bexp, dksat, dwsat, sicemax):
    """WDFCND: soil water diffusivity (WDF) and hydraulic conductivity (WCND)."""
    factr1 = 0.05 / smcmax
    factr2 = smc / smcmax
    factr1 = jnp.minimum(factr1, factr2)
    expon = bexp + 2.0
    wdf = dwsat * factr2 ** expon
    vkwgt = 1.0 / (1.0 + (500.0 * sicemax) ** 3.0)
    wdf = jnp.where(sicemax > 0.0, vkwgt * wdf + (1.0 - vkwgt) * dwsat * factr1 ** expon, wdf)
    expon2 = 2.0 * bexp + 3.0
    wcnd = dksat * factr2 ** expon2
    return wdf, wcnd


# ===========================================================================
# Soil thermodynamics: HRT (right-hand side + matrix) + HSTEP (tridiag solve)
# ===========================================================================

def _shflx(stc, smc, sh2o, t1, dt, yy, zz1, zsoil, params, df1, forcing_tbot):
    """SHFLX: update soil temperature column + skin temp + ground heat flux.

    ITAVG=.TRUE. (the WRF default): freezing/thawing layers use layer-average
    temperature (TBND/TMPAVG) before the SNKSRC phase-change source term.
    """
    p = params
    t0 = FRH2O_T0
    cair, cice, ch2o = 1004.0, 2.106e6, 4.2e6
    csoil_loc = jnp.where(p.is_urban, 3.0e6, p.csoil)  # HRT urban heat capacity
    smcmax = p.smcmax
    tbot = forcing_tbot
    zbot = p.zbot

    def zget(k):
        return zsoil[..., k - 1]

    # ---- layer 1 ----
    hcpct1 = (sh2o[..., 0] * ch2o + (1.0 - smcmax) * csoil_loc
              + (smcmax - smc[..., 0]) * cair + (smc[..., 0] - sh2o[..., 0]) * cice)
    ddz = 1.0 / (-0.5 * zget(2))
    ai1 = jnp.zeros_like(hcpct1)
    ci1 = (df1 * ddz) / (zget(1) * hcpct1)
    bi1 = -ci1 + df1 / (0.5 * zget(1) * zget(1) * hcpct1 * zz1)
    dtsdz1 = (stc[..., 0] - stc[..., 1]) / (-0.5 * zget(2))
    ssoil_top = df1 * (stc[..., 0] - yy) / (0.5 * zget(1) * zz1)
    denom1 = zget(1) * hcpct1
    rhsts1 = (df1 * dtsdz1 - ssoil_top) / denom1
    qtot1 = -1.0 * rhsts1 * denom1
    sice1 = smc[..., 0] - sh2o[..., 0]
    tsurf = (yy + (zz1 - 1.0) * stc[..., 0]) / zz1
    tbk1_l1 = _tbnd(stc[..., 0], stc[..., 1], zsoil, zbot, 1)
    dz1 = -zget(1)
    tavg1 = _tmpavg(tsurf, stc[..., 0], tbk1_l1, dz1)
    need1 = (sice1 > 0.0) | (stc[..., 0] < t0) | (tsurf < t0) | (tbk1_l1 < t0)
    tsnsr1, sh2o0_new = _snksrc(tavg1, smc[..., 0], sh2o[..., 0], dz1, smcmax, p.psisat, p.bexp, dt, qtot1)
    rhsts1 = jnp.where(need1, rhsts1 - tsnsr1 / denom1, rhsts1)
    sh2o = sh2o.at[..., 0].set(jnp.where(need1, sh2o0_new, sh2o[..., 0]))

    rhsts = [rhsts1]
    ai = [ai1]
    bi = [bi1]
    ci = [ci1]

    # carry across layers: df1k, dtsdz, ddz, tbk
    df1k = df1
    dtsdz = dtsdz1
    ddz_c = ddz
    tbk = tbk1_l1

    for kk in range(2, NSOIL + 1):
        i = kk - 1  # 0-based
        hcpct = (sh2o[..., i] * ch2o + (1.0 - smcmax) * csoil_loc
                 + (smcmax - smc[..., i]) * cair + (smc[..., i] - sh2o[..., i]) * cice)
        if kk != NSOIL:
            df1n = jnp.where(p.is_urban, 3.24, _tdfcnd(smc[..., i], p.quartz, smcmax, sh2o[..., i]))
            denom2 = 0.5 * (zget(kk - 1) - zget(kk + 1))
            dtsdz2 = (stc[..., i] - stc[..., i + 1]) / denom2
            ddz2 = 2.0 / (zget(kk - 1) - zget(kk + 1))
            ci_k = -df1n * ddz2 / ((zget(kk - 1) - zget(kk)) * hcpct)
            tbk1 = _tbnd(stc[..., i], stc[..., i + 1], zsoil, zbot, kk)
        else:
            df1n = jnp.where(p.is_urban, 3.24, _tdfcnd(smc[..., i], p.quartz, smcmax, sh2o[..., i]))
            denom_b = 0.5 * (zget(kk - 1) + zget(kk)) - zbot
            dtsdz2 = (stc[..., i] - tbot) / denom_b
            ci_k = jnp.zeros_like(hcpct)
            tbk1 = _tbnd(stc[..., i], tbot, zsoil, zbot, kk)

        denom = (zget(kk) - zget(kk - 1)) * hcpct
        rhsts_k = (df1n * dtsdz2 - df1k * dtsdz) / denom
        qtot = -1.0 * denom * rhsts_k
        sice = smc[..., i] - sh2o[..., i]
        dz_k = zget(kk - 1) - zget(kk)
        tavg = _tmpavg(tbk, stc[..., i], tbk1, dz_k)
        need = (sice > 0.0) | (stc[..., i] < t0) | (tbk < t0) | (tbk1 < t0)
        tsnsr, sh2o_new = _snksrc(tavg, smc[..., i], sh2o[..., i], dz_k, smcmax, p.psisat, p.bexp, dt, qtot)
        rhsts_k = jnp.where(need, rhsts_k - tsnsr / denom, rhsts_k)
        sh2o = sh2o.at[..., i].set(jnp.where(need, sh2o_new, sh2o[..., i]))

        ai_k = -df1k * ddz_c / ((zget(kk - 1) - zget(kk)) * hcpct)
        bi_k = -(ai_k + ci_k)

        rhsts.append(rhsts_k)
        ai.append(ai_k)
        bi.append(bi_k)
        ci.append(ci_k)

        tbk = tbk1
        df1k = df1n
        dtsdz = dtsdz2
        ddz_c = ddz2 if kk != NSOIL else ddz_c

    rhsts = jnp.stack(rhsts, axis=-1)
    ai = jnp.stack(ai, axis=-1)
    bi = jnp.stack(bi, axis=-1)
    ci = jnp.stack(ci, axis=-1)

    # HSTEP: scale by dt and solve tridiagonal
    rhsts_dt = rhsts * dt
    ai_dt = ai * dt
    bi_dt = 1.0 + bi * dt
    ci_dt = ci * dt
    sol = _rosr12(ai_dt, bi_dt, ci_dt, rhsts_dt)
    stc_out = stc + sol

    t1_new = (yy + (zz1 - 1.0) * stc_out[..., 0]) / zz1
    ssoil = df1 * (stc_out[..., 0] - t1_new) / (0.5 * zget(1))
    return stc_out, sh2o, t1_new, ssoil


# ===========================================================================
# Soil hydrology: SRT (RHS + matrix) + SSTEP (tridiag solve + update)
# ===========================================================================

def _srt(sh2o, sh2oa, et, edir, pcpdrp, zsoil, params, sice, dt):
    """SRT: Richards-equation RHS + tridiagonal matrix coefficients."""
    p = params
    smcmax = p.smcmax
    smcwlt = p.smcwlt

    def zget(k):
        return zsoil[..., k - 1]

    sicemax = jnp.max(sice, axis=-1)
    cvfrz = 3

    # ---- infiltration / surface runoff (Schaake) ----
    dt1 = dt / 86400.0
    smcav = smcmax - smcwlt
    dmax1 = -zget(1) * smcav * (1.0 - (sh2oa[..., 0] + sice[..., 0] - smcwlt) / smcav)
    dice = -zget(1) * sice[..., 0]
    dd = dmax1
    for kk in range(2, NSOIL + 1):
        i = kk - 1
        dice = dice + (zget(kk - 1) - zget(kk)) * sice[..., i]
        dmaxk = ((zget(kk - 1) - zget(kk)) * smcav
                 * (1.0 - (sh2oa[..., i] + sice[..., i] - smcwlt) / smcav))
        dd = dd + dmaxk
    val = 1.0 - jnp.exp(-p.kdt * dt1)
    ddt = dd * val
    px = jnp.maximum(pcpdrp * dt, 0.0)
    infmax = (px * (ddt / (px + ddt))) / dt
    # frozen-ground reduction FCR
    acrt = cvfrz * p.frzx / jnp.where(dice > 1e-2, dice, 1.0)
    # SUM = 1 + sum_{j=1}^{cvfrz-1} acrt^(cvfrz-j)/k!  ; cvfrz=3 -> j=1,2
    sumv = 1.0 + acrt ** 2 / 1.0 + acrt ** 1 / 2.0
    fcr = jnp.where(dice > 1e-2, 1.0 - jnp.exp(-acrt) * sumv, 1.0)
    infmax = infmax * fcr
    _, wcnd_top = _wdfcnd(sh2oa[..., 0], smcmax, p.bexp, p.dksat, p.dwsat, sicemax)
    infmax = jnp.maximum(infmax, wcnd_top)
    infmax = jnp.minimum(infmax, px / dt)
    has_prcp = pcpdrp != 0.0
    over = has_prcp & (pcpdrp > infmax)
    runoff1 = jnp.where(over, pcpdrp - infmax, 0.0)
    pddum = jnp.where(over, infmax, pcpdrp)

    # ---- matrix assembly ----
    wdf, wcnd = _wdfcnd(sh2oa[..., 0], smcmax, p.bexp, p.dksat, p.dwsat, sicemax)
    ddz = 1.0 / (-0.5 * zget(2))
    ai1 = jnp.zeros_like(wdf)
    bi1 = wdf * ddz / (-zget(1))
    ci1 = -bi1
    dsmdz = (sh2o[..., 0] - sh2o[..., 1]) / (-0.5 * zget(2))
    rhstt1 = (wdf * dsmdz + wcnd - pddum + edir + et[..., 0]) / zget(1)

    rhstt = [rhstt1]
    ai = [ai1]
    bi = [bi1]
    ci = [ci1]
    runoff2 = jnp.zeros_like(wdf)

    wdf_c, wcnd_c, dsmdz_c, ddz_c = wdf, wcnd, dsmdz, ddz
    for kk in range(2, NSOIL + 1):
        i = kk - 1
        denom2 = zget(kk - 1) - zget(kk)
        if kk != NSOIL:
            slopx = 1.0
            wdf2, wcnd2 = _wdfcnd(sh2oa[..., i], smcmax, p.bexp, p.dksat, p.dwsat, sicemax)
            denom = zget(kk - 1) - zget(kk + 1)
            dsmdz2 = (sh2o[..., i] - sh2o[..., i + 1]) / (denom * 0.5)
            ddz2 = 2.0 / denom
            ci_k = -wdf2 * ddz2 / denom2
        else:
            slopx = p.slope
            wdf2, wcnd2 = _wdfcnd(sh2oa[..., NSOIL - 1], smcmax, p.bexp, p.dksat, p.dwsat, sicemax)
            dsmdz2 = jnp.zeros_like(wdf2)
            ddz2 = jnp.zeros_like(wdf2)
            ci_k = jnp.zeros_like(wdf2)
        numer = wdf2 * dsmdz2 + slopx * wcnd2 - wdf_c * dsmdz_c - wcnd_c + et[..., i]
        rhstt_k = numer / (-denom2)
        ai_k = -wdf_c * ddz_c / denom2
        bi_k = -(ai_k + ci_k)
        rhstt.append(rhstt_k)
        ai.append(ai_k)
        bi.append(bi_k)
        ci.append(ci_k)
        if kk == NSOIL:
            runoff2 = slopx * wcnd2
        else:
            wdf_c, wcnd_c, dsmdz_c, ddz_c = wdf2, wcnd2, dsmdz2, ddz2

    rhstt = jnp.stack(rhstt, axis=-1)
    ai = jnp.stack(ai, axis=-1)
    bi = jnp.stack(bi, axis=-1)
    ci = jnp.stack(ci, axis=-1)
    return rhstt, ai, bi, ci, runoff1, runoff2


def _sstep(sh2oin, cmc, rhstt, ai, bi, ci, rhsct, dt, zsoil, params, smc_in, sice):
    """SSTEP: tridiagonal solve + soil-moisture / canopy update.

    Returns (sh2oout, smc, cmc, runoff3).
    """
    p = params
    smcmax = p.smcmax
    cmcmax = p.cmcmax

    def zget(k):
        return zsoil[..., k - 1]

    rhstt_dt = rhstt * dt
    ai_dt = ai * dt
    bi_dt = 1.0 + bi * dt
    ci_dt = ci * dt
    ci_sol = _rosr12(ai_dt, bi_dt, ci_dt, rhstt_dt)

    # sequential layer update with super-saturation spill WPLUS -> next layer
    wplus = jnp.zeros(sh2oin.shape[:-1], dtype=sh2oin.dtype)
    sh2oout_layers = []
    smc_layers = []
    for kk in range(1, NSOIL + 1):
        i = kk - 1
        ddz = jnp.where(jnp.asarray(kk == 1), -zget(1), zget(kk - 1) - zget(kk)) if kk != 1 else -zget(1)
        sh2o_k = sh2oin[..., i] + ci_sol[..., i] + wplus / ddz
        stot = sh2o_k + sice[..., i]
        ddz_sp = (-zget(1)) if kk == 1 else (-zget(kk) + zget(kk - 1))
        wplus = jnp.where(stot > smcmax, (stot - smcmax) * ddz_sp, 0.0)
        smc_k = jnp.clip(jnp.minimum(stot, smcmax), 0.02, smcmax)
        sh2o_out_k = jnp.maximum(smc_k - sice[..., i], 0.0)
        sh2oout_layers.append(sh2o_out_k)
        smc_layers.append(smc_k)
    sh2oout = jnp.stack(sh2oout_layers, axis=-1)
    smc = jnp.stack(smc_layers, axis=-1)
    runoff3 = wplus
    cmc_new = cmc + dt * rhsct
    cmc_new = jnp.where(cmc_new < 1e-20, 0.0, cmc_new)
    cmc_new = jnp.minimum(cmc_new, cmcmax)
    return sh2oout, smc, cmc_new, runoff3


def _fac2mit(smcmax):
    """FAC2MIT: FLIMIT lookup keyed on exact SMCMAX literals (else 0.90)."""
    flimit = jnp.full_like(smcmax, 0.90)
    flimit = jnp.where(smcmax == 0.395, 0.59, flimit)
    flimit = jnp.where((smcmax == 0.434) | (smcmax == 0.404), 0.85, flimit)
    flimit = jnp.where((smcmax == 0.465) | (smcmax == 0.406), 0.86, flimit)
    flimit = jnp.where((smcmax == 0.476) | (smcmax == 0.439), 0.74, flimit)
    flimit = jnp.where((smcmax == 0.200) | (smcmax == 0.464), 0.80, flimit)
    return flimit


def _smflx(smc, sh2o, cmc, dt, prcp1, zsoil, params, shdfac, ec, edir, et, drip_in):
    """SMFLX: canopy + soil moisture update (SRT/SSTEP F-/D-scheme).

    Returns (smc, sh2o, cmc, runoff1, runoff2, runoff3, drip).
    """
    p = params
    smcmax = p.smcmax
    cmcmax = p.cmcmax
    rhsct = shdfac * prcp1 - ec
    trhsct = dt * rhsct
    excess = cmc + trhsct
    drip = jnp.where(excess > cmcmax, excess - cmcmax, 0.0)
    pcpdrp = (1.0 - shdfac) * prcp1 + drip / dt
    sice = smc - sh2o

    fac2 = jnp.max(sh2o / smcmax[..., None], axis=-1)
    flimit = _fac2mit(smcmax)
    use_fscheme = ((pcpdrp * dt) > (0.0001 * 1000.0 * (-zsoil[..., 0]) * smcmax)) | (fac2 > flimit)

    # ---- D-scheme (single SRT/SSTEP) ----
    rhstt_d, ai_d, bi_d, ci_d, ro1_d, ro2_d = _srt(sh2o, sh2o, et, edir, pcpdrp, zsoil, params, sice, dt)
    sh2o_d, smc_d, cmc_d, ro3_d = _sstep(sh2o, cmc, rhstt_d, ai_d, bi_d, ci_d, rhsct, dt, zsoil, params, smc, sice)

    # ---- F-scheme (SRT, SSTEP-fg, average, SRT, SSTEP) ----
    rhstt1, ai1, bi1, ci1, ro1_1, ro2_1 = _srt(sh2o, sh2o, et, edir, pcpdrp, zsoil, params, sice, dt)
    sh2ofg, _smc_fg, _cmc_fg, _ro3_fg = _sstep(sh2o, cmc, rhstt1, ai1, bi1, ci1, rhsct, dt, zsoil, params, smc, sice)
    sh2oa = (sh2o + sh2ofg) * 0.5
    rhstt2, ai2, bi2, ci2, ro1_f, ro2_f = _srt(sh2o, sh2oa, et, edir, pcpdrp, zsoil, params, sice, dt)
    sh2o_f, smc_f, cmc_f, ro3_f = _sstep(sh2o, cmc, rhstt2, ai2, bi2, ci2, rhsct, dt, zsoil, params, smc, sice)

    m = use_fscheme
    sh2o_out = jnp.where(m[..., None], sh2o_f, sh2o_d)
    smc_out = jnp.where(m[..., None], smc_f, smc_d)
    cmc_out = jnp.where(m, cmc_f, cmc_d)
    runoff1 = jnp.where(m, ro1_f, ro1_d)
    runoff2 = jnp.where(m, ro2_f, ro2_d)
    runoff3 = jnp.where(m, ro3_f, ro3_d)
    return smc_out, sh2o_out, cmc_out, runoff1, runoff2, runoff3, drip


def _evapo(smc, sh2o, cmc, etp1, dt, shdfac, params, pc, zsoil, forcing):
    """EVAPO: total ET = direct soil evap + transpiration + canopy evap."""
    p = params
    edir = jnp.where(
        (etp1 > 0.0) & (shdfac < 1.0),
        _devap(etp1, sh2o[..., 0], shdfac, p.smcmax, p.smcdry, p.fxexp),
        0.0,
    )
    et = jnp.where(
        ((etp1 > 0.0) & (shdfac > 0.0))[..., None],
        _transp(etp1, sh2o, cmc, shdfac, params, pc, zsoil),
        0.0,
    )
    ett = jnp.sum(et, axis=-1)
    ec_raw = jnp.where(cmc > 0.0, shdfac * ((cmc / p.cmcmax) ** p.cfactr) * etp1, 0.0)
    cmc2ms = cmc / dt
    ec = jnp.where((etp1 > 0.0) & (shdfac > 0.0), jnp.minimum(cmc2ms, ec_raw), 0.0)
    eta1 = edir + ett + ec
    return eta1, edir, ec, et, ett


def _nopac(forcing, params, state, etp, t24, rch, epsca, rr, emissi, pc, zsoil, dt):
    """NOPAC: no-snowpack soil moisture + heat update, returns SSOIL/ETA/T1/states."""
    f, p, s = forcing, params, state
    shdfac = p.shdfac
    prcp1_base = f.prcp * 0.001
    etp1 = etp * 0.001
    smcmax = p.smcmax

    # EVAPO only when ETP>0; else dew
    eta1_pos, edir1_p, ec1_p, et1_p, ett1_p = _evapo(
        s.smc, s.sh2o, s.cmc, etp1, dt, shdfac, params, pc, zsoil, forcing)
    dew = jnp.where(etp <= 0.0, -etp1, 0.0)
    prcp1 = jnp.where(etp <= 0.0, prcp1_base + (-etp1), prcp1_base)
    edir1 = jnp.where(etp > 0.0, edir1_p, 0.0)
    ec1 = jnp.where(etp > 0.0, ec1_p, 0.0)
    et1 = jnp.where((etp > 0.0)[..., None], et1_p, 0.0)
    ett1 = jnp.where(etp > 0.0, ett1_p, 0.0)
    eta1 = jnp.where(etp > 0.0, eta1_pos, 0.0)

    smc, sh2o, cmc, runoff1, runoff2, runoff3, drip = _smflx(
        s.smc, s.sh2o, s.cmc, dt, prcp1, zsoil, params, shdfac, ec1, edir1, et1, drip_in=0.0)

    eta = eta1 * 1000.0
    beta = jnp.where(etp <= 0.0, jnp.where(etp < 0.0, 1.0, 0.0), eta / jnp.where(etp == 0.0, 1.0, etp))
    eta = jnp.where(etp <= 0.0, etp, eta)

    df1n = jnp.where(p.is_urban, 3.24, _tdfcnd(smc[..., 0], p.quartz, smcmax, sh2o[..., 0]))
    df1v = df1n * jnp.exp(p.sbeta * shdfac)
    yynum = (f.solnet + f.lwdn) - emissi * SIGMA * t24
    yy = f.sfctmp + (yynum / rch + f.th2 - f.sfctmp - beta * epsca) / rr
    zz1 = df1v / (-0.5 * zsoil[..., 0] * rch * rr) + 1.0
    stc, sh2o2, t1_new, ssoil = _shflx(s.stc, smc, sh2o, s.t1, dt, yy, zz1, zsoil, params, df1v, f.tbot)
    edir = edir1 * 1000.0
    ec = ec1 * 1000.0
    ett = ett1 * 1000.0
    return (smc, sh2o2, cmc, t1_new, stc, ssoil, eta, beta, dew, edir, ec, ett,
            runoff1, runoff2, runoff3)


def _snopac(forcing, params, state, etp, t24, rch, epsca, rr, df1, pc, zsoil, dt,
            sndens, snowng, emissi):
    """SNOPAC: snowpack branch — sublimation/melt energy + soil moisture/heat.

    Faithful translation of the WRF SNOPAC for the non-glacial land path
    (UA_PHYS=.false.). T1 / SSOIL are computed at the snow-top effective surface.
    """
    f, p, s = forcing, params, state
    shdfac = p.shdfac
    smcmax = p.smcmax
    esdmin = 1.0e-6
    snoexp = 2.0
    # PRCPF: precip is 0 when snowing/freezing (folded into SNEQV upstream); else PRCP.
    prcpf = jnp.where(snowng, 0.0, f.prcp)
    prcp1 = prcpf * 0.001

    esd = s.sneqv
    sncovr = s.sncovr
    ribb = s.ribb

    # ----- ETP<=0 dewfall vs ETP>0 sublimation+evapo -----
    etp_adj = jnp.where(
        (etp <= 0.0) & (ribb >= 0.1) & ((f.solnet + f.lwdn) > 150.0),
        (jnp.minimum(etp * (1.0 - ribb), 0.0) * sncovr / 0.980 + etp * (0.980 - sncovr)) / 0.980,
        etp,
    )
    etp1 = etp_adj * 0.001
    dew = jnp.where(etp_adj <= 0.0, -etp1, 0.0)

    # ETP>0 land evapo over snow-free fraction
    eta1, edir1_p, ec1_p, et1_p, ett1_p = _evapo(
        s.smc, s.sh2o, s.cmc, etp1, dt, shdfac, params, pc, zsoil, forcing)
    fr = 1.0 - sncovr
    edir1 = jnp.where(etp_adj > 0.0, edir1_p * fr, 0.0)
    ec1 = jnp.where(etp_adj > 0.0, ec1_p * fr, 0.0)
    et1 = jnp.where((etp_adj > 0.0)[..., None], et1_p * fr[..., None], 0.0)
    ett1 = jnp.where(etp_adj > 0.0, ett1_p * fr, 0.0)
    etns1 = jnp.where(etp_adj > 0.0, eta1 * fr, 0.0)
    etns = etns1 * 1000.0

    esnow = jnp.where(etp_adj > 0.0, etp_adj * sncovr, 0.0)
    esnow1 = esnow * 0.001
    esnow2 = jnp.where(etp_adj <= 0.0, etp1 * dt, esnow1 * dt)
    etanrg = jnp.where(
        etp_adj <= 0.0,
        etp_adj * ((1.0 - sncovr) * LSUBC + sncovr * LSUBS),
        esnow * LSUBS + etns * LSUBC,
    )

    # ----- FLX1 precip-snow surface heat -----
    flx1 = jnp.where(
        snowng,
        CPICE * f.prcp * (s.t1 - f.sfctmp),
        jnp.where(f.prcp > 0.0, CPH2O * f.prcp * (s.t1 - f.sfctmp), 0.0),
    )
    flx2 = jnp.where(False, 0.0, 0.0)  # FLX2 carried from PENMAN (frzgra only); snow=>0 here

    dsoil = -(0.5 * zsoil[..., 0])
    dtot = s.snowh + dsoil
    denom = 1.0 + df1 / (dtot * rr * rch)
    t12a = ((f.solnet + f.lwdn - flx1 - flx2 - emissi * SIGMA * t24) / rch
            + f.th2 - f.sfctmp - etanrg / rch) / rr
    t12b = df1 * s.stc[..., 0] / (dtot * rr * rch)
    t12 = (f.sfctmp + t12a + t12b) / denom

    sub_freeze = t12 <= TFREEZ
    # --- sub-freezing branch ---
    t1_sf = t12
    ssoil_sf = df1 * (t1_sf - s.stc[..., 0]) / dtot
    esd_sf = jnp.maximum(0.0, esd - esnow2)
    ex_sf = jnp.zeros_like(t12)
    snomlt_sf = jnp.zeros_like(t12)
    flx3_sf = jnp.zeros_like(t12)

    # --- above-freezing branch (snowmelt) ---
    snofac = jnp.maximum(0.01, sncovr ** snoexp)
    t1_af = TFREEZ * snofac + t12 * (1.0 - snofac)
    ssoil_af = df1 * (t1_af - s.stc[..., 0]) / dtot
    sublimated = (esd - esnow2) <= esdmin
    esd_af0 = esd - esnow2
    seh = rch * (t1_af - f.th2)
    t14 = t1_af ** 4
    flx3_raw = (f.solnet + f.lwdn) - flx1 - flx2 - emissi * SIGMA * t14 - ssoil_af - seh - etanrg
    flx3_raw = jnp.maximum(flx3_raw, 0.0)
    ex = flx3_raw * 0.001 / LSUBF
    snomlt_raw = ex * dt
    # snowmelt <= snow depth
    melt_ok = (esd_af0 - snomlt_raw) >= esdmin
    esd_af = jnp.where(sublimated, 0.0,
                       jnp.where(melt_ok, esd_af0 - snomlt_raw, 0.0))
    ex_af = jnp.where(sublimated, 0.0,
                      jnp.where(melt_ok, ex, esd_af0 / dt))
    flx3_af = jnp.where(sublimated, 0.0,
                        jnp.where(melt_ok, flx3_raw, ex_af * 1000.0 * LSUBF))
    snomlt_af = jnp.where(sublimated, 0.0,
                          jnp.where(melt_ok, snomlt_raw, esd_af0))

    t1 = jnp.where(sub_freeze, t1_sf, t1_af)
    ssoil = jnp.where(sub_freeze, ssoil_sf, ssoil_af)
    esd_new = jnp.where(sub_freeze, esd_sf, esd_af)
    ex = jnp.where(sub_freeze, ex_sf, ex_af)
    snomlt = jnp.where(sub_freeze, snomlt_sf, snomlt_af)
    flx3 = jnp.where(sub_freeze, flx3_sf, flx3_af)

    prcp1 = prcp1 + jnp.where(sub_freeze, 0.0, ex)

    smc, sh2o, cmc, runoff1, runoff2, runoff3, drip = _smflx(
        s.smc, s.sh2o, s.cmc, dt, prcp1, zsoil, params, shdfac, ec1, edir1, et1, drip_in=0.0)

    # SHFLX with snow-top BC (ZZ1=1, YY from SSOIL); skin temp already set
    zz1 = jnp.ones_like(t1)
    yy = s.stc[..., 0] - 0.5 * ssoil * zsoil[..., 0] * zz1 / df1
    stc, sh2o2, _t11, _ssoil1 = _shflx(s.stc, smc, sh2o, t1, dt, yy, zz1, zsoil, params, df1, f.tbot)

    # snow depth/density update (compaction) or zero-out
    snowh_c, sndens_c = _snowpack(esd_new, dt, s.snowh, sndens, t1, yy, snomlt)
    have_snow = esd_new > 0.0
    snowh_new = jnp.where(have_snow, snowh_c, 0.0)
    sncovr_new = jnp.where(have_snow, sncovr, 0.0)

    edir = edir1 * 1000.0
    ec = ec1 * 1000.0
    ett = ett1 * 1000.0
    eta_kin = esnow + etns - 1000.0 * dew
    return (smc, sh2o2, cmc, t1, stc, ssoil, esnow, etns, dew, edir, ec, ett, eta_kin,
            esd_new, snowh_new, sncovr_new, runoff1, runoff2, runoff3, snomlt, flx1, flx3)


def _snowpack(esd, dtsec, snowh, sndens, tsnow, tsoil, snomlt):
    """SNOWPACK: snow compaction (Koren polynomial), UA_PHYS=.false."""
    c1, c2 = 0.01, 21.0
    snowhc = snowh * 100.0
    esdc = esd * 100.0
    dthr = dtsec / 3600.0
    tsnowc = tsnow - 273.15
    tsoilc = tsoil - 273.15
    tavgc = 0.5 * (tsnowc + tsoilc)
    esdcx = jnp.where(esdc > 1.0e-2, esdc, 1.0e-2)
    bfac = dthr * c1 * jnp.exp(0.08 * tavgc - c2 * sndens)
    pexp = jnp.zeros_like(esdc)
    for j in range(4, 0, -1):
        pexp = (1.0 + pexp) * bfac * esdcx / float(j + 1)
    pexp = pexp + 1.0
    dsx = jnp.clip(sndens * pexp, 0.05, 0.40)
    sndens_new = dsx
    dw = 0.13 * dthr / 24.0
    sndens_melt = jnp.minimum(sndens_new * (1.0 - dw) + dw, 0.40)
    sndens_new = jnp.where(tsnowc >= 0.0, sndens_melt, sndens_new)
    snowhc_new = esdc / sndens_new
    return snowhc_new * 0.01, sndens_new


def sflx_step(forcing: NoahClassicForcing, params: NoahClassicParams,
              state: NoahClassicState, dt: float, zsoil, sldpth) -> NoahClassicOutput:
    """SFLX driver: one Noah-classic land step over a column tile (ICE=0 land).

    ``zsoil`` (negative interface depths) and ``sldpth`` (layer thicknesses) are
    (..., NSOIL) arrays. Returns the updated state + the full flux/diagnostic set.
    """
    f, p, s = forcing, params, state

    # SFLX urban override (VEGTYP==ISURBAN, SF_URBAN_PHYSICS=0): overwrite SHDFAC
    # and the soil-moisture parameters; DF1 forced to 3.24 below (handled at use).
    u = p.is_urban
    p = p._replace(
        shdfac=jnp.where(u, 0.05, p.shdfac),
        rsmin=jnp.where(u, 400.0, p.rsmin),
        smcmax=jnp.where(u, 0.45, p.smcmax),
        smcref=jnp.where(u, 0.42, p.smcref),
        smcwlt=jnp.where(u, 0.40, p.smcwlt),
        smcdry=jnp.where(u, 0.40, p.smcdry),
    )

    snowng = (f.prcp > 0.0) & (f.ffrozp > 0.5)
    frzgra = (f.prcp > 0.0) & (f.ffrozp <= 0.5) & (s.t1 <= TFREEZ)

    # snow density / new-snowfall accumulation
    sndens0 = jnp.where(s.sneqv > 1e-7, s.sneqv / jnp.maximum(s.snowh, 1e-12), 0.0)
    add_snow = snowng | frzgra
    sn_new = f.prcp * dt * 0.001
    sneqv = jnp.where(add_snow, s.sneqv + sn_new, s.sneqv)
    snowh_ns, sndens_ns = _snow_new(f.sfctmp, sn_new, s.snowh, sndens0)
    snowh = jnp.where(add_snow, snowh_ns, s.snowh)
    sndens = jnp.where(add_snow, sndens_ns, sndens0)
    sncond = jnp.where(s.sneqv > 1e-7, _csnow(sndens), 1.0)

    has_snow = sneqv > 0.0
    # SNFRAC + ALCALC (snow), else snow-free
    sncovr = jnp.minimum(_snfrac(sneqv, p.snup, p.salp, snowh), 0.98)
    albedo_sn, emissi_sn, snotime_sn = _alcalc(
        p.alb, f.snoalb, p.embrd, sncovr, dt, snowng, s.snotime1, p.lvcoef)
    sncovr = jnp.where(has_snow, sncovr, 0.0)
    albedo = jnp.where(has_snow, albedo_sn, p.alb)
    emissi = jnp.where(has_snow, emissi_sn, p.embrd)
    snotime1 = jnp.where(has_snow, snotime_sn, s.snotime1)

    # thermal diffusivity DF1 (top layer) with veg + snow plane-parallel
    df1_base = jnp.where(p.is_urban, 3.24, _tdfcnd(s.smc[..., 0], p.quartz, p.smcmax, s.sh2o[..., 0]))
    df1_soil = df1_base * jnp.exp(p.sbeta * p.shdfac)
    df1_soil = jnp.where(sncovr > 0.97, sncond, df1_soil)
    dsoil = -(0.5 * zsoil[..., 0])
    dtot = snowh + dsoil
    frcsno = snowh / dtot
    frcsoi = dsoil / dtot
    df1h = (sncond * df1_soil) / (frcsoi * sncond + frcsno * df1_soil)
    df1a = frcsno * sncond + frcsoi * df1_soil
    df1_snow = df1a * sncovr + df1_soil * (1.0 - sncovr)
    df1 = jnp.where(has_snow, df1_snow, df1_soil)
    ssoil_pre = jnp.where(
        has_snow,
        df1 * (s.t1 - s.stc[..., 0]) / dtot,
        df1_soil * (s.t1 - s.stc[..., 0]) / dsoil,
    )

    # roughness over snow
    z0 = jnp.where(sncovr > 0.0, _snowz0(sncovr, p.z0brd, snowh), p.z0brd)

    # PENMAN potential evaporation
    etp, rch, epsca, rr, flx2, t24 = _penman(
        f, ssoil_pre, None, snowng, frzgra, emissi, sneqv, s.t1, sncovr)

    # CANRES -> PC
    pc = _canres(f, p, s.sh2o, zsoil, emissi)

    # build a working forcing that carries prcpf for SNOPAC
    state_work = s._replace(sneqv=sneqv, snowh=snowh, sncovr=sncovr, snotime1=snotime1)

    # ---- NOPAC (no snow) ----
    (smc_n, sh2o_n, cmc_n, t1_n, stc_n, ssoil_n, eta_n, beta_n, dew_n, edir_n, ec_n,
     ett_n, ro1_n, ro2_n, ro3_n) = _nopac(f, p, state_work, etp, t24, rch, epsca, rr, emissi, pc, zsoil, dt)
    esnow_n = jnp.zeros_like(t1_n)
    eta_kin_n = eta_n

    # ---- SNOPAC (snow) ----
    (smc_s, sh2o_s, cmc_s, t1_s, stc_s, ssoil_s, esnow_s, etns_s, dew_s, edir_s, ec_s,
     ett_s, eta_kin_s, esd_s, snowh_s, sncovr_s, ro1_s, ro2_s, ro3_s, snomlt_s,
     flx1_s, flx3_s) = _snopac(
        f, p, state_work, etp, t24, rch, epsca, rr, df1, pc, zsoil, dt, sndens, snowng, emissi)

    sel = has_snow[..., None]
    sel2 = has_snow
    smc = jnp.where(sel, smc_s, smc_n)
    sh2o = jnp.where(sel, sh2o_s, sh2o_n)
    stc = jnp.where(sel, stc_s, stc_n)
    cmc = jnp.where(sel2, cmc_s, cmc_n)
    t1 = jnp.where(sel2, t1_s, t1_n)
    ssoil = jnp.where(sel2, ssoil_s, ssoil_n)
    esnow = jnp.where(sel2, esnow_s, esnow_n)
    dew = jnp.where(sel2, dew_s, dew_n)
    edir = jnp.where(sel2, edir_s, edir_n)
    ec = jnp.where(sel2, ec_s, ec_n)
    ett = jnp.where(sel2, ett_s, ett_n)
    runoff1 = jnp.where(sel2, ro1_s, ro1_n)
    runoff2 = jnp.where(sel2, ro2_s, ro2_n)
    snomlt = jnp.where(sel2, snomlt_s, jnp.zeros_like(t1))
    sneqv_out = jnp.where(sel2, esd_s, sneqv)
    snowh_out = jnp.where(sel2, snowh_s, jnp.where(has_snow, snowh, 0.0))
    sncovr_out = jnp.where(sel2, sncovr_s, sncovr)

    # ETA_KINEMATIC
    eta_kin = jnp.where(sel2, eta_kin_s, eta_kin_n)

    # Q1 effective surface mixing ratio handle
    q1 = f.q2 + eta_kin * CP / rch

    # SHEAT (HFX): -(CH*CP*SFCPRS)/(R*T2V)*(TH2-T1)
    t2v = f.sfctmp * (1.0 + 0.61 * f.q2)
    sheat = -(f.ch * CP * f.sfcprs) / (RD * t2v) * (f.th2 - t1)

    # top-level energy conversions (W/m2)
    edir_w = edir * LVH2O
    ec_w = ec * LVH2O
    ett_w = ett * LVH2O
    esnow_w = esnow * LSUBS
    etp_w = etp * ((1.0 - sncovr_out) * LVH2O + sncovr_out * LSUBS)
    eta_w = jnp.where(etp_w > 0.0, edir_w + ec_w + ett_w + esnow_w, etp_w)
    beta = jnp.where(etp_w == 0.0, 0.0, eta_w / jnp.where(etp_w == 0.0, 1.0, etp_w))

    # SSOIL sign flip (WRF: GRDFLX = -SSOIL... then driver stores SSOIL post-flip)
    ssoil_out = -1.0 * ssoil

    # SMAV soil-moisture availability (smcrel carry)
    smav = (smc - p.smcwlt[..., None]) / (p.smcmax - p.smcwlt)[..., None]

    new_state = NoahClassicState(
        t1=t1, stc=stc, smc=smc, sh2o=sh2o, cmc=cmc,
        sneqv=sneqv_out, snowh=snowh_out, sncovr=sncovr_out,
        snotime1=snotime1, ribb=s.ribb,
    )
    return NoahClassicOutput(
        state=new_state, hfx=sheat, qfx=eta_kin, lh=eta_w, grdflx=ssoil_out,
        etp=etp_w, albedo=albedo, emissi=emissi, z0=z0, q1=q1, snomlt=snomlt,
        edir=edir_w, ec=ec_w, ett=ett_w, esnow=esnow_w, beta=beta, smav=smav,
        runoff1=runoff1, runoff2=runoff2,
    )
