"""Noah-MP semi-implicit snow/soil temperature (Sprint S2).

WRF-faithful port of the pure-thermal column step from pristine
``<USER_HOME>/src/wrf_pristine/WRF/phys/noahmp/src/module_sf_noahmplsm.F`` for the
scoped configuration (``opt_stc=1`` semi-implicit, ``opt_tbot=2`` Noah deep-soil
lower BC, ``opt_frz=1`` NY06 supercooled liquid). NSOIL=4, NSNOW=3, fp64.

Routines ported (1:1 with WRF, vectorized over the ``(ny, nx)`` land tile, layer
axis 0 over the ``NSNOW + NSOIL`` snow/soil column):

- ``noahmp_thermoprop``  -> THERMOPROP (:2400-2510) + CSNOW (:2514-2569) +
  TDFCND (:2573-2680): thermal conductivity ``DF`` and volumetric heat capacity
  ``HCPCT`` per layer (Peters-Lidard soil, Stieglitz snow) and the snow/soil
  interface DF blend.
- ``noahmp_soil_thermo`` -> TSNOSOI (:5258-5371) = HRT (:5375-5473) tridiagonal
  assembly + HSTEP (:5477-5530) / ROSR12 (:5534-5591) semi-implicit solve, with
  the ``opt_tbot=2`` Noah deep-soil bottom BC from TBOT.
- ``noahmp_phasechange`` -> PHASECHANGE (:5595-5810): melt/freeze energy
  redistribution with the ``opt_frz=1`` NY06 supercooled-liquid threshold; resets
  STC to TFRZ where phase change occurs, returns updated SNICE/SNLIQ/SMC/SH2O and
  QMELT/IMELT. Exposed (NOT one of the two FROZEN signatures, which carry only the
  thermal solve) so the energy/water sprints can call it without amending the
  frozen ``soil_thermo`` interface.

The two frozen signatures (``noahmp_thermoprop``, ``noahmp_soil_thermo``) are
unchanged from the Sprint-0 freeze. ``noahmp_thermoprop_fact`` and
``noahmp_phasechange`` are additive helpers (no interface widening).

FULLY PARALLEL: depends only on the frozen ``contracts.noahmp_state`` shapes and
the ``physics.noahmp.tables`` parameter bundle (read off ``static.parameters``).
"""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

import jax
from jax import config
import jax.numpy as jnp

configure_jax_x64()

# ---------------------------------------------------------------------------
# Physical constants — pristine WRF module_sf_noahmplsm.F:204-220.
# ---------------------------------------------------------------------------
GRAV: float = 9.80616      # gravity [m/s2]
TFRZ: float = 273.16       # freezing/melting point [K]
HFUS: float = 0.3336e06    # latent heat of fusion [J/kg]
CWAT: float = 4.188e06     # volumetric heat capacity of water [J/m3/K]
CICE: float = 2.094e06     # volumetric heat capacity of ice [J/m3/K]
CPAIR: float = 1004.64     # heat capacity dry air at const pressure [J/kg/K]
TKWAT: float = 0.6         # thermal conductivity of water [W/m/K]
TKICE: float = 2.2         # thermal conductivity of ice [W/m/K]
DENH2O: float = 1000.0     # density of water [kg/m3]
DENICE: float = 917.0      # density of ice [kg/m3]
ZBOT: float = -8.0         # depth of lower BC from soil surface [m], NEGATIVE.
# WRF GENPARM ``ZBOT_DATA = -8.0`` is loaded verbatim into ``parameters%ZBOT``
# (module_sf_noahmpdrv.F:1687; the glacier module hard-codes ``ZBOT = -8.0``).
# ZBOT enters TSNOSOI as ``ZBOTSNO = parameters%ZBOT - SNOWH`` and the HRT bottom
# gradient ``DTSDZ(NSOIL)=(STC-TBOT)/(0.5*(ZSNSO(NSOIL-1)+ZSNSO(NSOIL))-ZBOTSNO)``
# (module_sf_noahmplsm.F:5444). With ZSNSO<0 the denominator is
# ``0.5*(-0.7-1.5) - (-8.0) = +6.9`` (deep heat flows DOWN to the cooler 8 m BC).
# The previous ``+8.0`` flipped the denominator to ``-9.1`` -- WRONG SIGN -- which
# inverted the deep-soil heat exchange. The single-step soil savepoint hid this
# because the oracle replica used the same +8.0 (a self-consistent tautology on
# this term).

NSOIL: int = 4
NSNOW: int = 3
NLAY: int = NSNOW + NSOIL  # 7: snow layers [0..2] over soil layers [3..6]

# Layer-axis-0 index of the first soil layer (STC index 1 in WRF's -2:4 array).
_SOIL_TOP: int = NSNOW


# ===========================================================================
# THERMOPROP / CSNOW / TDFCND  (DF, HCPCT)
# ===========================================================================
def _tdfcnd(smc: jax.Array, sh2o: jax.Array, smcmax: jax.Array, quartz: jax.Array) -> jax.Array:
    """Soil thermal conductivity DF [W/m/K] — TDFCND (:2573-2680), Peters-Lidard.

    ``smc``/``sh2o`` are total / liquid volumetric soil moisture; ``smcmax``,
    ``quartz`` are per-soil-layer parameters. Fully vectorized, branch-free via
    ``jnp.where`` (Kersten-number frozen/unfrozen split + saturation-ratio gate).
    """

    thkw = 0.57
    thko = 2.0
    thkqtz = 7.7

    satratio = smc / smcmax

    thks = (thkqtz ** quartz) * (thko ** (1.0 - quartz))

    # Unfrozen volume fraction for saturation (prevent divide-by-zero, D. Mocko).
    xunfroz = jnp.where(smc > 0.0, sh2o / smc, 1.0)
    xu = xunfroz * smcmax

    thksat = thks ** (1.0 - smcmax) * TKICE ** (smcmax - xu) * thkw ** xu

    gammd = (1.0 - smcmax) * 2700.0
    thkdry = (0.135 * gammd + 64.7) / (2700.0 - 0.947 * gammd)

    # Kersten number AKE: frozen -> SATRATIO; unfrozen -> log10(SATRATIO)+1 (or 0).
    ake_unfrozen = jnp.where(satratio > 0.1, jnp.log10(jnp.maximum(satratio, 1e-30)) + 1.0, 0.0)
    is_frozen = (sh2o + 0.0005) < smc
    ake = jnp.where(is_frozen, satratio, ake_unfrozen)

    return ake * (thksat - thkdry) + thkdry


def _csnow(snice, snliq, dzsnso_snow):
    """Snow thermal conductivity TKSNO and volumetric heat capacity CVSNO — CSNOW (:2514-2569).

    Stieglitz (Yen 1965) ``TKSNO = 3.2217e-6 * BDSNOI**2``. ``dzsnso_snow`` is the
    snow-layer thickness slice (>0 in active layers). Inactive layers (thickness 0)
    are guarded so the bulk density stays finite; they are masked out downstream.
    """

    safe_dz = jnp.where(dzsnso_snow > 0.0, dzsnso_snow, 1.0)
    snicev = jnp.minimum(1.0, snice / (safe_dz * DENICE))
    epore = 1.0 - snicev
    snliqv = jnp.minimum(epore, snliq / (safe_dz * DENH2O))

    bdsnoi = (snice + snliq) / safe_dz
    cvsno = CICE * snicev + CWAT * snliqv
    tksno = 3.2217e-6 * bdsnoi ** 2.0
    return tksno, cvsno


def noahmp_thermoprop(
    land_state,
    static,
    fsno: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Thermal conductivity DF and volumetric heat capacity HCPCT per layer.

    Ports THERMOPROP (module_sf_noahmplsm.F:2400-2510) for the land branch
    (``IST=1``, no urban, no lake). Returns ``(df, hcpct)`` shaped
    ``(NSNOW + NSOIL, ny, nx)`` (snow layers above soil; only ``isnow`` active).

    Reads SMC/SH2O/SNICE/SNLIQ/SNOWH/ISNOW/ZSNSO from ``land_state`` and the
    per-category soil parameters (SMCMAX, QUARTZ, CSOIL) gathered onto the tile in
    ``static.parameters`` (already indexed by ``isltyp``, ADR §2.5). ``fsno`` is
    accepted for signature parity (snow-cover fraction is not used in the land
    THERMOPROP branch; the snow/soil DF blend uses SNOWH directly).
    """

    del fsno  # land THERMOPROP branch uses SNOWH, not the FSNO fraction
    p = static.parameters
    smcmax = p.smcmax       # (NSOIL, ny, nx)
    quartz = p.quartz       # (NSOIL, ny, nx)
    csoil = p.csoil         # scalar or (ny, nx)

    smc = land_state.smois  # (NSOIL, ny, nx)
    sh2o = land_state.sh2o  # (NSOIL, ny, nx)
    snice = land_state.snice  # (NSNOW, ny, nx)
    snliq = land_state.snliq  # (NSNOW, ny, nx)
    snowh = land_state.snowh  # (ny, nx)
    isnow = land_state.isnow  # (ny, nx) int, in {-2,-1,0}
    zsnso = land_state.zsnso  # (NLAY, ny, nx) interface depths (<0)

    dzsnso = _dzsnso_from_zsnso(zsnso)  # (NLAY, ny, nx) thicknesses (>0)

    # --- snow layers (indices 0..NSNOW-1) ---
    tksno, cvsno = _csnow(snice, snliq, dzsnso[:NSNOW])
    # active-snow mask: WRF loops IZ = ISNOW+1, 0  -> layer-axis index >= NSNOW+isnow
    snow_idx = jnp.arange(NSNOW).reshape(NSNOW, 1, 1)
    snow_active = snow_idx >= (NSNOW + isnow)  # broadcast over (ny,nx)
    df_snow = jnp.where(snow_active, tksno, 0.0)
    hcpct_snow = jnp.where(snow_active, cvsno, 0.0)

    # --- soil layers (indices NSNOW..NLAY-1) ---
    sice = smc - sh2o
    hcpct_soil = (
        sh2o * CWAT
        + (1.0 - smcmax) * csoil
        + (smcmax - smc) * CPAIR
        + sice * CICE
    )
    df_soil = _tdfcnd(smc, sh2o, smcmax, quartz)

    df = jnp.concatenate([df_snow, df_soil], axis=0)
    hcpct = jnp.concatenate([hcpct_snow, hcpct_soil], axis=0)

    # --- snow/soil interface DF blend (THERMOPROP:2503-2507) ---
    # Top soil layer = index NSNOW; top snow layer (when present) = index NSNOW-1.
    no_snow = isnow == 0
    df_soil1 = df[_SOIL_TOP]
    dz_soil1 = dzsnso[_SOIL_TOP]
    df_snow0 = df[_SOIL_TOP - 1]
    dz_snow0 = dzsnso[_SOIL_TOP - 1]

    df1_nosnow = (df_soil1 * dz_soil1 + 0.35 * snowh) / (snowh + dz_soil1)
    df1_snow = (df_soil1 * dz_soil1 + df_snow0 * dz_snow0) / (dz_snow0 + dz_soil1)
    df_top_soil = jnp.where(no_snow, df1_nosnow, df1_snow)
    df = df.at[_SOIL_TOP].set(df_top_soil)

    return df, hcpct


def noahmp_thermoprop_fact(hcpct: jax.Array, dzsnso: jax.Array, dt: float) -> jax.Array:
    """Phase-change factor FACT = DT/(HCPCT*DZSNSO) (THERMOPROP:2497-2499).

    Separate so PHASECHANGE can recompute it from the THERMOPROP outputs without
    widening the frozen ``noahmp_thermoprop`` return tuple. Inactive layers
    (``hcpct==0``) are guarded to keep FACT finite; they carry no energy.
    """

    safe = jnp.where(hcpct > 0.0, hcpct * dzsnso, 1.0)
    return dt / safe


# ===========================================================================
# TSNOSOI = HRT + HSTEP + ROSR12  (semi-implicit STC solve, opt_stc=1)
# ===========================================================================
def _dzsnso_from_zsnso(zsnso: jax.Array) -> jax.Array:
    """Layer thicknesses (>0) from interface depths ZSNSO (<0).

    DZSNSO(K) = ZSNSO(K-1) - ZSNSO(K) with ZSNSO(0)=0 at the snow surface
    (WRF builds DZSNSO the same way; module_sf_noahmplsm.F snow/soil geometry).
    """

    zero = jnp.zeros_like(zsnso[:1])
    z_above = jnp.concatenate([zero, zsnso[:-1]], axis=0)
    return z_above - zsnso


def noahmp_soil_thermo(
    stc: jax.Array,
    df: jax.Array,
    hcpct: jax.Array,
    ssoil: jax.Array,
    tbot: jax.Array,
    zsnso: jax.Array,
    dzsnso: jax.Array,
    isnow: jax.Array,
    dt: float,
) -> jax.Array:
    """Semi-implicit snow/soil temperature solve (TSNOSOI, opt_stc=1/opt_tbot=2).

    Ports TSNOSOI (:5258-5371): HRT tridiagonal assembly (:5375-5473) + HSTEP/ROSR12
    semi-implicit solve (:5477-5591). ``stc`` is ``(NSNOW + NSOIL, ny, nx)`` (snow
    above soil; only ``isnow`` active), driven by the ground heat flux ``ssoil`` at
    the active top and the Noah deep-soil BC at ``tbot``. ``zsnso``/``dzsnso`` are
    interface depths (<0) / thicknesses (>0); ``isnow`` in {-2,-1,0}. Returns the
    updated ``stc``.

    Pure thermal: no water movement, no melt (that is ``noahmp_phasechange``).
    """

    # ZBOTSNO = ZBOT - SNOWH (TSNOSOI:5314). SNOWH = -ZSNSO at the soil-surface
    # interface = -zsnso[NSNOW-1] (top of soil = bottom of snow). No-snow -> SNOWH=0.
    snowh = jnp.where(isnow < 0, -zsnso[_SOIL_TOP - 1], 0.0)
    zbotsno = ZBOT - snowh  # (ny, nx)

    ai, bi, ci, rhsts = _hrt(stc, tbot, zbotsno, df, hcpct, ssoil, zsnso, isnow)

    # HSTEP (:5506-5511): scale by DT and form the implicit system, then ROSR12.
    rhsts_dt = rhsts * dt
    ai_dt = ai * dt
    bi_dt = 1.0 + bi * dt
    ci_dt = ci * dt

    delta = _rosr12(ai_dt, bi_dt, ci_dt, rhsts_dt, isnow)
    return stc + delta


def _hrt(stc, tbot, zbot, df, hcpct, ssoil, zsnso, isnow):
    """HRT (:5375-5473): tridiagonal coefficients AI/BI/CI + RHS, opt_stc=1/opt_tbot=2.

    Vectorized over the column (axis 0). Inactive snow layers (index < NSNOW+isnow)
    are decoupled to the identity row (AI=CI=BI=0 pre-DT-scale, RHSTS=0) so the
    DT-scaled HSTEP system leaves them unchanged. The active top is ``ISNOW+1``
    (axis index ``NSNOW+isnow``); the bottom is the deepest soil layer ``NSOIL``.
    PHI (solar penetration) is 0 for the soil/land branch.
    """

    nz = stc.shape[0]
    idx = jnp.arange(nz).reshape(nz, 1, 1)
    top = NSNOW + isnow          # axis index of ISNOW+1 (active top)
    active = idx >= top
    is_top = idx == top
    is_bottom = idx == (nz - 1)

    # Neighbour shifts along the column.
    stc_p1 = jnp.concatenate([stc[1:], stc[-1:]], axis=0)   # STC(K+1)
    df_m1 = jnp.concatenate([df[:1], df[:-1]], axis=0)      # DF(K-1)
    zsnso_m1 = jnp.concatenate([zsnso[:1], zsnso[:-1]], axis=0)   # ZSNSO(K-1)
    zsnso_p1 = jnp.concatenate([zsnso[1:], zsnso[-1:]], axis=0)   # ZSNSO(K+1)

    eps = 1e-30

    # top: DENOM=-ZSNSO(K)*HCPCT; TEMP1=-ZSNSO(K+1); DDZ=2/TEMP1; DTSDZ=2*(STC-STC+1)/TEMP1
    denom_top = -zsnso * hcpct
    temp1_top = -zsnso_p1
    safe_t1_top = jnp.where(jnp.abs(temp1_top) > eps, temp1_top, eps)
    ddz_top = 2.0 / safe_t1_top
    dtsdz_top = 2.0 * (stc - stc_p1) / safe_t1_top

    # interior: DENOM=(ZSNSO(K-1)-ZSNSO(K))*HCPCT; TEMP1=ZSNSO(K-1)-ZSNSO(K+1)
    denom_int = (zsnso_m1 - zsnso) * hcpct
    temp1_int = zsnso_m1 - zsnso_p1
    safe_t1_int = jnp.where(jnp.abs(temp1_int) > eps, temp1_int, eps)
    ddz_int = 2.0 / safe_t1_int
    dtsdz_int = 2.0 * (stc - stc_p1) / safe_t1_int

    # bottom (opt_tbot=2): DTSDZ=(STC-TBOT)/(0.5*(ZSNSO(K-1)+ZSNSO(K))-ZBOT); BOTFLX=-DF*DTSDZ
    denom_bot = (zsnso_m1 - zsnso) * hcpct
    dtsdz_bot = (stc - tbot) / (0.5 * (zsnso_m1 + zsnso) - zbot)
    botflx = -df * dtsdz_bot

    ddz = jnp.where(is_top, ddz_top, jnp.where(is_bottom, 0.0, ddz_int))
    dtsdz = jnp.where(is_top, dtsdz_top, jnp.where(is_bottom, dtsdz_bot, dtsdz_int))
    denom = jnp.where(is_top, denom_top, jnp.where(is_bottom, denom_bot, denom_int))

    # EFLUX:
    #  top      : DF*DTSDZ - SSOIL          (PHI=0)
    #  interior : DF*DTSDZ - DF(K-1)*DTSDZ(K-1)
    #  bottom   : -BOTFLX  - DF(K-1)*DTSDZ(K-1)
    dtsdz_m1 = jnp.concatenate([dtsdz[:1], dtsdz[:-1]], axis=0)
    eflux_top = df * dtsdz - ssoil
    eflux_int = df * dtsdz - df_m1 * dtsdz_m1
    eflux_bot = -botflx - df_m1 * dtsdz_m1
    eflux = jnp.where(is_top, eflux_top, jnp.where(is_bottom, eflux_bot, eflux_int))

    # --- AI/BI/CI (opt_stc=1: BI = -CI at top) ---
    ddz_m1 = jnp.concatenate([ddz[:1], ddz[:-1]], axis=0)  # DDZ(K-1)
    safe_denom = jnp.where(jnp.abs(denom) > eps, denom, eps)

    ci_top = -df * ddz / safe_denom
    ai_int = -df_m1 * ddz_m1 / safe_denom
    ci_int = -df * ddz / safe_denom
    ai_bot = -df_m1 * ddz_m1 / safe_denom

    ai = jnp.where(is_top, 0.0, jnp.where(is_bottom, ai_bot, ai_int))
    ci = jnp.where(is_top, ci_top, jnp.where(is_bottom, 0.0, ci_int))
    bi = -(ai + ci)
    # top opt_stc=1 special-case: BI = -CI (AI is 0 there, so -(ai+ci)==-ci already)

    rhsts = eflux / (-safe_denom)

    # Decouple inactive (above-top) layers -> identity row.
    ai = jnp.where(active, ai, 0.0)
    ci = jnp.where(active, ci, 0.0)
    bi = jnp.where(active, bi, 0.0)
    rhsts = jnp.where(active, rhsts, 0.0)

    return ai, bi, ci, rhsts


def _rosr12(a, b, c, d, isnow):
    """ROSR12 (:5534-5591): forward elimination + back substitution over axis 0.

    Solves the tridiagonal system [A,B,C] x = D and returns the per-layer increment
    (the WRF ``CI``/``P`` increment added to STC). Above-top inactive rows have
    B=1, A=C=D=0 (from the HSTEP DT-scaling of the decoupled HRT rows), so starting
    the ROSR12 recurrence at axis row 0 is algebraically identical to starting at
    WRF's ``NTOP = ISNOW+1`` for the active block; the result is re-masked to the
    active rows at the end for safety.

    Fixed-length ``lax.scan`` over the full ``NLAY`` column (jit/GPU friendly).
    """

    eps = 1e-30
    nz = a.shape[0]

    # ROSR12 forward: P(NTOP) = -C(NTOP)/B(NTOP); DELTA(NTOP) = D(NTOP)/B(NTOP).
    b0 = jnp.where(jnp.abs(b[0]) > eps, b[0], eps)
    p0 = -c[0] / b0
    delta0 = d[0] / b0

    def fwd(carry, row):
        p_prev, delta_prev = carry
        ak, bk, ck, dk = row
        denom = bk + ak * p_prev
        denom = jnp.where(jnp.abs(denom) > eps, denom, eps)
        pk = -ck / denom
        deltak = (dk - ak * delta_prev) / denom
        return (pk, deltak), (pk, deltak)

    rows = (a[1:], b[1:], c[1:], d[1:])
    _, (p_tail, delta_tail) = jax.lax.scan(fwd, (p0, delta0), rows)

    p = jnp.concatenate([p0[None], p_tail], axis=0)
    delta = jnp.concatenate([delta0[None], delta_tail], axis=0)

    # Back substitution: P(NSOIL)=DELTA(NSOIL); P(KK)=P(KK)*P(KK+1)+DELTA(KK).
    x_last = delta[-1]

    def bwd(x_next, row):
        p_k, delta_k = row
        x_k = p_k * x_next + delta_k
        return x_k, x_k

    rev = (p[:-1][::-1], delta[:-1][::-1])
    _, x_rev = jax.lax.scan(bwd, x_last, rev)
    x = jnp.concatenate([x_rev[::-1], x_last[None]], axis=0)

    idx = jnp.arange(nz).reshape(nz, 1, 1)
    active = idx >= (NSNOW + isnow)
    return jnp.where(active, x, 0.0)


# ===========================================================================
# PHASECHANGE (opt_frz=1 NY06 supercooled liquid)  — additive (not frozen-sig)
# ===========================================================================
def noahmp_phasechange(
    stc: jax.Array,
    snice: jax.Array,
    snliq: jax.Array,
    smc: jax.Array,
    sh2o: jax.Array,
    sneqv: jax.Array,
    snowh: jax.Array,
    hcpct: jax.Array,
    dzsnso: jax.Array,
    isnow: jax.Array,
    dt: float,
    *,
    smcmax: jax.Array,
    psisat: jax.Array,
    bexp: jax.Array,
):
    """Melt/freeze of snow & soil water — PHASECHANGE (:5595-5810), opt_frz=1 land branch.

    Faithful port of the land (``IST=1``) branch: NY06 supercooled-liquid threshold
    (``opt_frz=1``), per-layer energy residual HM, melt/freeze mass XM, ice/liquid
    repartition, and STC reset to TFRZ where phase change occurs. The
    ``ISNOW==0 .AND. SNEQV>0`` no-snow-layer melt special case (:5736-5753) and the
    BARLAGE snow-layer cleanup (:5784-5788) are included.

    Inputs are full-column ``(NSNOW+NSOIL, ny, nx)`` for ``stc``/``hcpct``/``dzsnso``,
    snow slices ``(NSNOW, ny, nx)`` for ``snice``/``snliq``, soil slices
    ``(NSOIL, ny, nx)`` for ``smc``/``sh2o``; soil params ``smcmax``/``psisat``/
    ``bexp`` are ``(NSOIL, ny, nx)``. ``psisat`` is the (positive) saturated matric
    potential magnitude (WRF passes ``PSISAT`` directly to the NY06 expression).
    Returns ``(stc, snice, snliq, smc, sh2o, sneqv, snowh, qmelt, imelt, ponding)``.

    NOTE: intentionally OUTSIDE the two frozen ``soil_thermo`` signatures (which
    carry only the thermal solve). The energy/water sprints call it directly.
    """

    nz = stc.shape[0]
    idx = jnp.arange(nz).reshape(nz, 1, 1)
    top = NSNOW + isnow
    active = idx >= top
    is_snow = idx < _SOIL_TOP          # axis index < NSNOW

    fact = noahmp_thermoprop_fact(hcpct, dzsnso, dt)

    # --- layer masses MICE/MLIQ (snow in kg/m2, soil converted from volumetric) ---
    dz_soil = dzsnso[_SOIL_TOP:]
    mliq_soil = sh2o * dz_soil * 1000.0
    mice_soil = (smc - sh2o) * dz_soil * 1000.0
    mliq = jnp.concatenate([snliq, mliq_soil], axis=0)
    mice = jnp.concatenate([snice, mice_soil], axis=0)

    wice0 = mice
    wliq0 = mliq  # noqa: F841  (kept for WRF-name fidelity; not read after init)
    wmass0 = mice + mliq

    # --- supercooled liquid threshold SUPERCOOL (soil only, opt_frz=1 NY06) ---
    below_frz_soil = stc[_SOIL_TOP:] < TFRZ
    smp = HFUS * (TFRZ - stc[_SOIL_TOP:]) / (GRAV * jnp.maximum(stc[_SOIL_TOP:], 1e-6))
    supercool_soil = smcmax * (smp / psisat) ** (-1.0 / bexp)
    supercool_soil = supercool_soil * dz_soil * 1000.0
    supercool_soil = jnp.where(below_frz_soil, supercool_soil, 0.0)
    supercool = jnp.concatenate([jnp.zeros_like(snice), supercool_soil], axis=0)

    # --- IMELT flag (:5699-5713) ---
    melt = (mice > 0.0) & (stc >= TFRZ)
    refreeze = (mliq > supercool) & (stc < TFRZ)
    imelt = jnp.where(melt, 1, jnp.where(refreeze, 2, 0))

    # no-snow-layer (ISNOW==0, SNEQV>0) top-soil melt flag at J==1 (axis NSNOW)
    no_snow_layer = (isnow == 0) & (sneqv > 0.0)
    j_is_top_soil = idx == _SOIL_TOP
    imelt = jnp.where(no_snow_layer & j_is_top_soil & (stc >= TFRZ), 1, imelt)

    has_phase = (imelt > 0) & active

    # --- energy residual HM and STC reset to TFRZ (:5717-5732) ---
    hm = jnp.where(has_phase, (stc - TFRZ) / fact, 0.0)
    stc = jnp.where(has_phase, TFRZ, stc)

    # cancel spurious melt/freeze (sign disagrees with HM)
    cancel = (((imelt == 1) & (hm < 0.0)) | ((imelt == 2) & (hm > 0.0))) & has_phase
    hm = jnp.where(cancel, 0.0, hm)
    imelt = jnp.where(cancel, 0, imelt)
    has_phase = (imelt > 0) & active

    xm = hm * dt / HFUS

    # --- no-layer snow melt special case (:5736-5753), only at top-soil index ---
    temp1 = sneqv
    xm_top = xm[_SOIL_TOP]
    do_nolayer = no_snow_layer & (xm_top > 0.0)
    sneqv_new = jnp.where(do_nolayer, jnp.maximum(0.0, temp1 - xm_top), sneqv)
    safe_temp1 = jnp.where(temp1 > 0.0, temp1, 1.0)
    propor = jnp.where(temp1 > 0.0, sneqv_new / safe_temp1, 0.0)
    snowh_pr = jnp.maximum(0.0, propor * snowh)
    snowh_pr = jnp.minimum(jnp.maximum(snowh_pr, sneqv_new / 500.0), sneqv_new / 50.0)
    snowh_new = jnp.where(do_nolayer, snowh_pr, snowh)
    heatr_top = hm[_SOIL_TOP] - HFUS * (temp1 - sneqv_new) / dt
    xm_top_new = jnp.where(heatr_top > 0.0, heatr_top * dt / HFUS, 0.0)
    hm_top_new = jnp.where(heatr_top > 0.0, heatr_top, 0.0)
    qmelt = jnp.where(do_nolayer, jnp.maximum(0.0, temp1 - sneqv_new) / dt, 0.0)
    ponding = jnp.where(do_nolayer, temp1 - sneqv_new, 0.0)

    hm = hm.at[_SOIL_TOP].set(jnp.where(do_nolayer, hm_top_new, hm[_SOIL_TOP]))
    xm = xm.at[_SOIL_TOP].set(jnp.where(do_nolayer, xm_top_new, xm[_SOIL_TOP]))
    sneqv = sneqv_new
    snowh = snowh_new

    # --- ice/liquid repartition per layer (:5757-5798) ---
    do_repart = has_phase & (jnp.abs(hm) > 0.0)

    xm_pos = xm > 0.0
    mice_melt = jnp.maximum(0.0, wice0 - xm)  # XM>0 melting

    # XM<0 refreezing: snow vs soil branch
    mice_snow_rf = jnp.minimum(wmass0, wice0 - xm)
    mice_soil_rf = jnp.where(
        wmass0 < supercool,
        0.0,
        jnp.maximum(jnp.minimum(wmass0 - supercool, wice0 - xm), 0.0),
    )
    mice_freeze = jnp.where(is_snow, mice_snow_rf, mice_soil_rf)

    mice_new = jnp.where(do_repart, jnp.where(xm_pos, mice_melt, mice_freeze), mice)
    mliq_new = jnp.where(do_repart, jnp.maximum(0.0, wmass0 - mice_new), mliq)

    heatr = jnp.where(do_repart, hm - HFUS * (wice0 - mice_new) / dt, 0.0)

    # STC update from residual heat (:5780-5790)
    stc = jnp.where(do_repart & (jnp.abs(heatr) > 0.0), stc + fact * heatr, stc)
    # snow-layer cleanup: liquid&ice both present -> TFRZ; all-melted -> TFRZ
    snow_both = is_snow & (mliq_new * mice_new > 0.0)
    snow_gone = is_snow & (mice_new == 0.0)
    stc = jnp.where(do_repart & (snow_both | snow_gone), TFRZ, stc)

    # accumulate snowmelt (snow layers, J<1) into QMELT (:5794-5796)
    qmelt = qmelt + jnp.sum(
        jnp.where(do_repart & is_snow, jnp.maximum(0.0, wice0 - mice_new) / dt, 0.0),
        axis=0,
    )

    mice = mice_new
    mliq = mliq_new

    # --- write back snow (kg/m2) and soil (volumetric) ---
    snliq = mliq[:_SOIL_TOP]
    snice = mice[:_SOIL_TOP]
    sh2o = mliq[_SOIL_TOP:] / (1000.0 * dz_soil)
    smc = (mliq[_SOIL_TOP:] + mice[_SOIL_TOP:]) / (1000.0 * dz_soil)

    return stc, snice, snliq, smc, sh2o, sneqv, snowh, qmelt, imelt, ponding


__all__ = [
    "noahmp_thermoprop",
    "noahmp_thermoprop_fact",
    "noahmp_soil_thermo",
    "noahmp_phasechange",
]
