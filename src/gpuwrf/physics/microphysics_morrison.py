"""JAX Morrison 2-moment bulk microphysics (WRF mp_physics=10).

Faithful port of WRF ``phys/module_mp_morr_two_moment.F`` for the default WRF
configuration (``morr_rimed_ice=0`` -> graupel mode; ``IGRAUP=0``, ``ILIQ=0``,
``INUC=0``, ``iinum=1`` constant droplet number ``NDCNST=250`` cm^-3). It ports
the wrapper ``MP_MORR_TWO_MOMENT`` (th<->t conversion, internal rho recompute,
surface-precip binding) and the column microphysics ``MORR_TWO_MOMENT_MICRO``,
preserving WRF's exact process order:

  per column k (vectorized over the vertical):
    1. XXLV, XXLS, CPM, EVS/EIS (POLYSVP), QVS/QVI, RHO=P/(R*T), cumulus-N add
    2. subsaturation removal of trace qc/qr/qi/qs/qg; QSMALL zeroing
    3. fall-speed prefactors AIN/ARN/ASN/ACN/AGN; skip-empty mask (GOTO 200)
    4. T >= 273.15 branch:  small-snow/graupel melt-to-rain, slope params,
         warm process rates (PRC/PRA/NRAGG/PRE), snow+graupel melting
         (PSMLT/PGMLT/EVPMS/EVPMG), rain-collection enhancement (PRACS/PRACG),
         per-species conservation ratios, tendencies + number melting/sub
       T <  273.15 branch:  slope params, contact+immersion freezing (MNUCCC),
         autoconv (PRC), aggregation (NSAGG), riming (PSACWS/PSACWG/PSACWI),
         rain-snow/rain-graupel collection (PRACS/PRACG/PSACR), rime-splinter
         (Hallett-Mossop, NMULTS/NMULTG/NMULTR/NMULTRG), riming->graupel
         conversion (PGSACW/PGRACS), rain freezing (MNUCCR), ice autoconv to
         snow (NPRCI/PRCI), ice-snow accretion (PRAI), rain-ice collection
         (PIACR/PRACI[S]), ice nucleation (Cooper, NNUCCD/MNUCCD), dep/sub
         (PRD/PRDS/PRDG with FUDGEF clamp), per-species conservation, tendencies
    5. saturation adjustment PCC (liquid), number sublimation NSUBI/S/R/G
    6. NSTEP-split semi-Lagrangian sedimentation (Reisner 1998) for all species
    7. final state update; ice->snow if mean diam > 2*dcs; instantaneous melt of
       ice (T>=273.15), homogeneous freezing of cloud (T<=233.15) and rain;
       slope recompute; effective radii; ice/droplet number bounds
  surface precip bound: RAINNCV=PRECRT, SNOWNCV=ice+snow, GRAUPELNCV=graupel,
                        SR = SNOWRT/(PRECRT+1e-12).

The adapter returns a frozen ``PhysicsTendency`` with ``state_replacements`` for
theta and the moist species + number concentrations (Morrison is an in-place
scheme), ``accumulator_increments`` for surface precip, and ``diagnostics`` for
the effective radii.

Validation: per-column WRF savepoint parity against the real Fortran scheme
(proofs/v060/oracle). Defaults to fp64; the WRF scheme runs fp32, so parity is
to a predeclared physical tolerance, never bitwise (run_morrison_parity.py).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from gpuwrf.contracts.physics_interfaces import PhysicsTendency
from gpuwrf.physics import morrison_constants as C


# ===========================================================================
# POLYSVP: saturation vapor pressure (Pa).  TYPE: 0 = liquid, 1 = ice.
# Flatau et al. 1992 polynomial above the Goff-Gratch switch temperature,
# Goff-Gratch below.  Vectorized with jnp.where on the switch.
# ===========================================================================
_A_ICE = (6.11147274, 0.503160820, 0.188439774e-1, 0.420895665e-3,
          0.615021634e-5, 0.602588177e-7, 0.385852041e-9, 0.146898966e-11,
          0.252751365e-14)
_A_LIQ = (6.11239921, 0.443987641, 0.142986287e-1, 0.264847430e-3,
          0.302950461e-5, 0.206739458e-7, 0.640689451e-10, -0.952447341e-13,
          -0.976195544e-15)


def _flatau(dt, coeffs):
    # Horner evaluation matching the Fortran nested form, *100 -> Pa.
    res = coeffs[8]
    for i in range(7, -1, -1):
        res = coeffs[i] + dt * res
    return res * 100.0


def polysvp(t, itype):
    """Saturation vapor pressure (Pa). itype=0 liquid, itype=1 ice."""
    dt = t - 273.15
    if itype == 1:
        flat = _flatau(dt, _A_ICE)
        gg = 10.0 ** (-9.09718 * (273.16 / t - 1.0)
                      - 3.56654 * jnp.log10(273.16 / t)
                      + 0.876793 * (1.0 - t / 273.16)
                      + jnp.log10(6.1071)) * 100.0
        return jnp.where(t >= 195.8, flat, gg)
    else:
        flat = _flatau(dt, _A_LIQ)
        gg = 10.0 ** (-7.90298 * (373.16 / t - 1.0)
                      + 5.02808 * jnp.log10(373.16 / t)
                      - 1.3816e-7 * (10.0 ** (11.344 * (1.0 - t / 373.16)) - 1.0)
                      + 8.1328e-3 * (10.0 ** (-3.49149 * (373.16 / t - 1.0)) - 1.0)
                      + jnp.log10(1013.246)) * 100.0
        return jnp.where(t >= 202.0, flat, gg)


# ===========================================================================
# GAMMA: Euler gamma for positive real argument (Cody/Stoltz rational minimax),
# vectorized.  Morrison only ever calls GAMMA with positive arguments (PGAM+k,
# 1+B*, etc.), so the negative-argument reflection branch is not reproduced.
# ===========================================================================
_GP = (-1.71618513886549492533811e0, 2.47656508055759199108314e1,
       -3.79804256470945635097577e2, 6.29331155312818442661052e2,
       8.66966202790413211295064e2, -3.14512729688483675254357e4,
       -3.61444134186911729807069e4, 6.64561438202405440627855e4)
_GQ = (-3.08402300119738975254353e1, 3.15350626979604161529144e2,
       -1.01515636749021914166146e3, -3.10777167157231109440444e3,
       2.25381184209801510330112e4, 4.75584627752788110767815e3,
       -1.34659959864969306392456e5, -1.15132259675553483497211e5)
_GC = (-1.910444077728e-03, 8.4171387781295e-04, -5.952379913043012e-04,
       7.93650793500350248e-04, -2.777777777777681622553e-03,
       8.333333333333333331554247e-02, 5.7083835261e-03)


def gamma_fn(x):
    """Euler gamma for x > 0 (Cody/Stoltz), vectorized over a JAX array."""
    x = jnp.asarray(x)
    # --- small argument 0 < y < 12 path uses range reduction to (1,2) ---
    # n = number of reduction multiplications for 1 <= y < 12
    # We replicate the algorithm branch-free with jnp.where over the regions.
    y = x
    # region masks
    lt1 = y < 1.0
    ge12 = y >= 12.0

    # ---- 1 <= y < 12 (and 0<y<1 via y->y+1): build z and the (1,2) minimax ----
    # For y<1:   z = y,        adjust = divide result by original y (=y1)
    # For 1<=y<12: n=int(y)-1, yr = y - n, z = yr - 1, then multiply result by
    #              y, y+1, ..., (n terms) where the running y starts at yr.
    y1 = y
    # number of integer reductions for 1<=y<12
    n = jnp.where(lt1, 0, jnp.floor(y).astype(jnp.int32) - 1)
    yr = jnp.where(lt1, y + 1.0, y - n.astype(y.dtype))
    z = jnp.where(lt1, y, yr - 1.0)

    xnum = jnp.zeros_like(y)
    xden = jnp.ones_like(y)
    for i in range(8):
        xnum = (xnum + _GP[i]) * z
        xden = xden * z + _GQ[i]
    res_small = xnum / xden + 1.0

    # adjust for 0<y<1: res = res / y1   (y1 = original y)
    res_small = jnp.where(lt1, res_small / y1, res_small)

    # adjust for 2<=y<12: multiply by yr, yr+1, ..., (n terms).
    # n ranges 0..10 for y in [1,12). Unroll up to 11 multiplications, each
    # applied only while the iteration index < n, with the running factor yr+i.
    nmax = 11
    res_red = res_small
    yy = yr
    for i in range(nmax):
        do_mul = (i < n) & (~lt1)
        res_red = jnp.where(do_mul, res_red * yy, res_red)
        yy = yy + 1.0
    res_small = res_red

    # ---- y >= 12 path: asymptotic (Stirling) ----
    ysq = y * y
    s = _GC[6]
    for i in range(6):
        s = s / ysq + _GC[i]
    s = s / y - y + C.XXX
    s = s + (y - 0.5) * jnp.log(y)
    res_big = jnp.exp(s)

    return jnp.where(ge12, res_big, res_small)


# ===========================================================================
# Slope-limited size-distribution parameters for one species.
# Returns (lam, n0, n_adjusted) with WRF's lambda min/max clamping that also
# back-adjusts the number concentration.  Generic over the WRF forms:
#   rain:    lam=(pi*rhow*N/q)^(1/3),  n0=N*lam, clamp adjusts n0 & N
#   snow:    lam=(CONS1*N/q)^(1/DS),   n0=N*lam
#   graupel: lam=(CONS2*N/q)^(1/DG),   n0=N*lam
#   ice:     lam=(CONS12*N/q)^(1/DI),  n0=N*lam
# clamp: N_new = n0_new / lam_clamp where n0_new = lam_clamp^4 * q / cmass
# ===========================================================================
def _slope_generic(q, n, cmass, inv_pow, lammin, lammax, qsmall):
    """Generic slope for rain(cmass=pi*rhow)/snow(CONS1)/graupel(CONS2)/ice(CONS12)."""
    active = q >= qsmall
    qs = jnp.where(active, q, 1.0)
    ns = jnp.maximum(n, 0.0)
    lam = (cmass * jnp.where(active, ns, 0.0) / qs) ** inv_pow
    # guard lam where inactive
    lam = jnp.where(active, lam, lammin)
    too_small = lam < lammin
    too_big = lam > lammax
    lam_c = jnp.where(too_small, lammin, jnp.where(too_big, lammax, lam))
    n0_c = lam_c ** 4 * jnp.where(active, q, 0.0) / cmass
    n_c = jnp.where(too_small | too_big, n0_c / lam_c, ns)
    n0 = jnp.where(too_small | too_big, n0_c, jnp.where(active, ns * lam, 0.0))
    lam_out = jnp.where(active, lam_c, 0.0)
    n_out = jnp.where(active, n_c, n)
    n0_out = jnp.where(active, n0, 0.0)
    return lam_out, n0_out, n_out


def _slope_droplet(qc, nc, t, pres, qsmall):
    """Cloud-droplet PGAM (Martin et al. 1994), LAMC, slope clamp, CDIST1."""
    active = qc >= qsmall
    dum = pres / (287.15 * t)
    pgam = 0.0005714 * (nc / 1.0e6 * dum) + 0.2714
    pgam = 1.0 / (pgam * pgam) - 1.0
    pgam = jnp.clip(pgam, 2.0, 10.0)
    g_p1 = gamma_fn(pgam + 1.0)
    g_p4 = gamma_fn(pgam + 4.0)
    qcs = jnp.where(active, qc, 1.0)
    lamc = (C.CONS26 * nc * g_p4 / (qcs * g_p1)) ** (1.0 / 3.0)
    lammin = (pgam + 1.0) / 60.0e-6
    lammax = (pgam + 1.0) / 1.0e-6
    too_small = lamc < lammin
    too_big = lamc > lammax
    lamc_c = jnp.where(too_small, lammin, jnp.where(too_big, lammax, lamc))
    # nc back-adjust on clamp: nc = exp(3*log(lamc)+log(qc)+log(g_p1)-log(g_p4))/CONS26
    nc_adj = jnp.exp(3.0 * jnp.log(lamc_c) + jnp.log(jnp.where(active, qc, 1.0))
                     + jnp.log(g_p1) - jnp.log(g_p4)) / C.CONS26
    nc_out = jnp.where(active & (too_small | too_big), nc_adj, nc)
    lamc_out = jnp.where(active, lamc_c, 0.0)
    cdist1 = jnp.where(active, nc_out / g_p1, 0.0)
    pgam_out = jnp.where(active, pgam, 0.0)
    return lamc_out, pgam_out, cdist1, nc_out, g_p1, g_p4


@partial(jax.jit, static_argnames=())
def morrison_run(th, qv, qc, qr, qi, qs, qg, ni, ns, nr, ng, pii, p, dz, w, dt):
    """Run one Morrison microphysics step on a batch of columns.

    All array args have shape (ncol, kx), bottom-up in k (k=0 lowest layer),
    matching the oracle. Returns a dict of post-scheme arrays and surface precip.
    """
    th = jnp.asarray(th)
    f = th.dtype
    qv = jnp.asarray(qv, f); qc = jnp.asarray(qc, f); qr = jnp.asarray(qr, f)
    qi = jnp.asarray(qi, f); qs = jnp.asarray(qs, f); qg = jnp.asarray(qg, f)
    ni = jnp.asarray(ni, f); ns = jnp.asarray(ns, f); nr = jnp.asarray(nr, f)
    ng = jnp.asarray(ng, f); pii = jnp.asarray(pii, f); p = jnp.asarray(p, f)
    dz = jnp.asarray(dz, f); w = jnp.asarray(w, f)
    dt = jnp.asarray(dt, f)

    QSMALL = C.QSMALL
    R = C.R; RV = C.RV; CP = C.CP; EP_2 = C.EP_2; PI = C.PI

    # wrapper: T = TH*PII
    t = th * pii

    # ----------------------------------------------------------------------
    # Setup: latent heats, CPM, saturation, RHO. (cumulus tendencies are zero.)
    # ----------------------------------------------------------------------
    xxlv = 3.1484e6 - 2370.0 * t
    xxls = 3.15e6 - 2370.0 * t + 0.3337e6
    cpm = CP * (1.0 + 0.887 * qv)

    evs = jnp.minimum(0.99 * p, polysvp(t, 0))
    eis = jnp.minimum(0.99 * p, polysvp(t, 1))
    eis = jnp.where(eis > evs, evs, eis)
    qvs = EP_2 * evs / (p - evs)
    qvi = EP_2 * eis / (p - eis)
    qvqvs = qv / qvs
    qvqvsi = qv / qvi
    rho = p / (R * t)

    # (cumulus-N additions skipped: qrcu/qscu/qicu are all zero)

    # subsaturation removal of trace qc/qr (liquid) and qi/qs/qg (ice)
    def _sub_remove_liq(qv_, t_, qx, thr):
        rm = (qvqvs < 0.9) & (qx < thr)
        qv2 = jnp.where(rm, qv_ + qx, qv_)
        t2 = jnp.where(rm, t_ - qx * xxlv / cpm, t_)
        qx2 = jnp.where(rm, 0.0, qx)
        return qv2, t2, qx2

    def _sub_remove_ice(qv_, t_, qx, thr):
        rm = (qvqvsi < 0.9) & (qx < thr)
        qv2 = jnp.where(rm, qv_ + qx, qv_)
        t2 = jnp.where(rm, t_ - qx * xxls / cpm, t_)
        qx2 = jnp.where(rm, 0.0, qx)
        return qv2, t2, qx2

    qv, t, qr = _sub_remove_liq(qv, t, qr, 1.0e-8)
    qv, t, qc = _sub_remove_liq(qv, t, qc, 1.0e-8)
    qv, t, qi = _sub_remove_ice(qv, t, qi, 1.0e-8)
    qv, t, qs = _sub_remove_ice(qv, t, qs, 1.0e-8)
    qv, t, qg = _sub_remove_ice(qv, t, qg, 1.0e-8)

    xlf = xxls - xxlv

    # QSMALL zeroing of mass+number
    def _qsmall_zero(qx, nx):
        m = qx < QSMALL
        return jnp.where(m, 0.0, qx), jnp.where(m, 0.0, nx)

    qc, nc0 = _qsmall_zero(qc, jnp.zeros_like(qc))  # nc not prognostic (iinum=1)
    qr, nr = _qsmall_zero(qr, nr)
    qi, ni = _qsmall_zero(qi, ni)
    qs, ns = _qsmall_zero(qs, ns)
    qg, ng = _qsmall_zero(qg, ng)

    # fall-speed prefactors with density correction
    mu = 1.496e-6 * t ** 1.5 / (t + 120.0)
    dum_dc = (C.RHOSU / rho) ** 0.54
    ain = (C.RHOSU / rho) ** 0.35 * C.AI
    arn = dum_dc * C.AR
    asn = dum_dc * C.AS
    acn = C.G * C.RHOW / (18.0 * mu)
    agn = dum_dc * C.AG
    kap = 1.414e3 * mu
    dv = 8.794e-5 * t ** 1.81 / p
    sc = mu / (rho * dv)

    dqsdt = xxlv * qvs / (RV * t * t)
    dqsidt = xxls * qvi / (RV * t * t)
    abi = 1.0 + dqsidt * xxls / cpm
    ab = 1.0 + dqsdt * xxlv / cpm

    # GOTO 200 mask: no hydrometeors and subsaturated -> skip ALL micro + sat-adj
    empty = ((qc < QSMALL) & (qi < QSMALL) & (qs < QSMALL)
             & (qr < QSMALL) & (qg < QSMALL))
    skip200 = empty & (((t < 273.15) & (qvqvsi < 0.999))
                       | ((t >= 273.15) & (qvqvs < 0.999)))
    do_cell = ~skip200

    # constant droplet number nc (iinum=1): NDCNST cm-3 -> kg-1
    nc = C.NDCNST * 1.0e6 / rho

    warm = t >= 273.15

    # ----------------------------------------------------------------------
    # Tendency accumulators (start at zero, like the Fortran *3DTEN arrays).
    # ----------------------------------------------------------------------
    z = jnp.zeros_like(t)
    qv_ten = z; t_ten = z; qc_ten = z; qr_ten = z
    qi_ten = z; qni_ten = z; qg_ten = z
    nc_ten = z; ni_ten = z; ns_ten = z; nr_ten = z; ng_ten = z

    # =====================================================================
    # WARM BRANCH (T >= 273.15)
    # =====================================================================
    # small snow/graupel melt to rain (Q< 1e-6) for warm cells
    melt_sn = warm & (qs < 1.0e-6)
    qr = jnp.where(melt_sn, qr + qs, qr)
    nr = jnp.where(melt_sn, nr + ns, nr)
    t = jnp.where(melt_sn, t - qs * xlf / cpm, t)
    qs = jnp.where(melt_sn, 0.0, qs)
    ns = jnp.where(melt_sn, 0.0, ns)
    melt_g = warm & (qg < 1.0e-6)
    qr = jnp.where(melt_g, qr + qg, qr)
    nr = jnp.where(melt_g, nr + ng, nr)
    t = jnp.where(melt_g, t - qg * xlf / cpm, t)
    qg = jnp.where(melt_g, 0.0, qg)
    ng = jnp.where(melt_g, 0.0, ng)

    # warm GOTO 300 mask: no warm hydrometeors -> skip warm process rates
    warm_active = warm & do_cell & ~(
        (qc < QSMALL) & (qs < 1.0e-8) & (qr < QSMALL) & (qg < 1.0e-8))

    nsw = jnp.maximum(ns, 0.0); ncw = jnp.maximum(nc, 0.0)
    nrw = jnp.maximum(nr, 0.0); ngw = jnp.maximum(ng, 0.0)

    lamr_w, n0rr_w, nr_w = _slope_generic(qr, nrw, PI * C.RHOW, 1.0 / 3.0,
                                          C.LAMMINR, C.LAMMAXR, QSMALL)
    lams_w, n0s_w, ns_w = _slope_generic(qs, nsw, C.CONS1, 1.0 / C.DS,
                                         C.LAMMINS, C.LAMMAXS, QSMALL)
    lamg_w, n0g_w, ng_w = _slope_generic(qg, ngw, C.CONS2, 1.0 / C.DG,
                                         C.LAMMING, C.LAMMAXG, QSMALL)
    lamc_w, pgam_w, _cd_w, ncw2, _gp1w, _gp4w = _slope_droplet(qc, ncw, t, p, QSMALL)

    # apply number clamps from slope (only on warm_active cells)
    nr = jnp.where(warm_active, nr_w, nr)
    ns = jnp.where(warm_active, ns_w, ns)
    ng = jnp.where(warm_active, ng_w, ng)
    nc = jnp.where(warm_active, ncw2, nc)

    one = jnp.ones_like(t)
    eps = 1.0e-30

    # autoconversion (KK2000)
    qc_ge6 = qc >= 1.0e-6
    prc_w = jnp.where(qc_ge6, 1350.0 * qc ** 2.47 * (nc / 1.0e6 * rho) ** (-1.79), 0.0)
    nprc1_w = jnp.where(qc_ge6, prc_w / C.CONS29, 0.0)
    nprc_w = jnp.where(qc_ge6, prc_w / (qc / jnp.where(nc > 0, nc, one)), 0.0)
    nprc_w = jnp.minimum(nprc_w, nc / dt)
    nprc1_w = jnp.minimum(nprc1_w, nprc_w)
    prc_w = jnp.where(qc_ge6, prc_w, 0.0)

    # accretion (KK2000)
    qrqc8 = (qr >= 1.0e-8) & (qc >= 1.0e-8)
    pra_w = jnp.where(qrqc8, 67.0 * (qc * qr) ** 1.15, 0.0)
    npra_w = jnp.where(qrqc8, pra_w / (qc / jnp.where(nc > 0, nc, one)), 0.0)

    # self-collection of rain (with breakup)
    qr8 = qr >= 1.0e-8
    inv_lamr = jnp.where(lamr_w > 0, 1.0 / jnp.where(lamr_w > 0, lamr_w, one), 0.0)
    br_dum = jnp.where(inv_lamr < 300.0e-6, 1.0,
                       2.0 - jnp.exp(2300.0 * (inv_lamr - 300.0e-6)))
    nragg_w = jnp.where(qr8, -5.78 * br_dum * nr * qr * rho, 0.0)

    # rain evaporation (only if subsaturated)
    qr_qs = qr >= QSMALL
    epsr_w = jnp.where(qr_qs,
                       2.0 * PI * n0rr_w * rho * dv
                       * (C.F1R / (lamr_w * lamr_w + eps)
                          + C.F2R * (arn * rho / mu) ** 0.5 * sc ** (1.0 / 3.0)
                          * C.CONS9 / (lamr_w ** C.CONS34 + eps)), 0.0)
    pre_w = jnp.where(qv < qvs, jnp.minimum(epsr_w * (qv - qvs) / ab, 0.0), 0.0)

    # collection of snow by rain (for melting enhancement) -> PRACS
    qr_qs8 = (qr >= 1.0e-8) & (qs >= 1.0e-8)
    ums_s = jnp.minimum(asn * C.CONS3 / (lams_w ** C.BS + eps), 1.2 * dum_dc)
    umr_s = jnp.minimum(arn * C.CONS4 / (lamr_w ** C.BR + eps), 9.1 * dum_dc)
    pracs_w = jnp.where(
        qr_qs8,
        C.CONS41 * (((1.2 * umr_s - 0.95 * ums_s) ** 2 + 0.08 * ums_s * umr_s) ** 0.5
                    * rho * n0rr_w * n0s_w / (lamr_w ** 3 + eps)
                    * (5.0 / (lamr_w ** 3 * lams_w + eps)
                       + 2.0 / (lamr_w ** 2 * lams_w ** 2 + eps)
                       + 0.5 / (lamr_w * lams_w ** 3 + eps))), 0.0)

    # collection of graupel by rain -> PRACG (+ NPRACG with shed-drop subtraction)
    qr_qg8 = (qr >= 1.0e-8) & (qg >= 1.0e-8)
    umg_g = jnp.minimum(agn * C.CONS7 / (lamg_w ** C.BG + eps), 20.0 * dum_dc)
    umr_g = jnp.minimum(arn * C.CONS4 / (lamr_w ** C.BR + eps), 9.1 * dum_dc)
    ung_g = jnp.minimum(agn * C.CONS8 / (lamg_w ** C.BG + eps), 20.0 * dum_dc)
    unr_g = jnp.minimum(arn * C.CONS6 / (lamr_w ** C.BR + eps), 9.1 * dum_dc)
    pracg_w = jnp.where(
        qr_qg8,
        C.CONS41 * (((1.2 * umr_g - 0.95 * umg_g) ** 2 + 0.08 * umg_g * umr_g) ** 0.5
                    * rho * n0rr_w * n0g_w / (lamr_w ** 3 + eps)
                    * (5.0 / (lamr_w ** 3 * lamg_w + eps)
                       + 2.0 / (lamr_w ** 2 * lamg_w ** 2 + eps)
                       + 0.5 / (lamr_w * lamg_w ** 3 + eps))), 0.0)
    npracg_w = jnp.where(
        qr_qg8,
        C.CONS32 * rho * (1.7 * (unr_g - ung_g) ** 2 + 0.3 * unr_g * ung_g) ** 0.5
        * n0rr_w * n0g_w * (1.0 / (lamr_w ** 3 * lamg_w + eps)
                            + 1.0 / (lamr_w ** 2 * lamg_w ** 2 + eps)
                            + 1.0 / (lamr_w * lamg_w ** 3 + eps)), 0.0)
    # shed drops (1 mm) subtraction: NPRACG = NPRACG - PRACG/5.2e-7
    npracg_w = jnp.where(qr_qg8, npracg_w - pracg_w / 5.2e-7, 0.0)

    # snow melting
    qs8 = qs >= 1.0e-8
    dum_psmlt = -C.CPW / xlf * (t - 273.15) * pracs_w
    psmlt_w = jnp.where(
        qs8,
        2.0 * PI * n0s_w * kap * (273.15 - t) / xlf
        * (C.F1S / (lams_w * lams_w + eps)
           + C.F2S * (asn * rho / mu) ** 0.5 * sc ** (1.0 / 3.0)
           * C.CONS10 / (lams_w ** C.CONS35 + eps)) + dum_psmlt, 0.0)
    sub_s = qs8 & (qvqvs < 1.0)
    epss_w = jnp.where(
        sub_s,
        2.0 * PI * n0s_w * rho * dv
        * (C.F1S / (lams_w * lams_w + eps)
           + C.F2S * (asn * rho / mu) ** 0.5 * sc ** (1.0 / 3.0)
           * C.CONS10 / (lams_w ** C.CONS35 + eps)), 0.0)
    evpms_w = jnp.where(sub_s, (qv - qvs) * epss_w / ab, 0.0)
    evpms_w = jnp.where(sub_s, jnp.maximum(evpms_w, psmlt_w), 0.0)
    psmlt_w = jnp.where(sub_s, psmlt_w - evpms_w, psmlt_w)

    # graupel melting
    qg8 = qg >= 1.0e-8
    dum_pgmlt = -C.CPW / xlf * (t - 273.15) * pracg_w
    pgmlt_w = jnp.where(
        qg8,
        2.0 * PI * n0g_w * kap * (273.15 - t) / xlf
        * (C.F1S / (lamg_w * lamg_w + eps)
           + C.F2S * (agn * rho / mu) ** 0.5 * sc ** (1.0 / 3.0)
           * C.CONS11 / (lamg_w ** C.CONS36 + eps)) + dum_pgmlt, 0.0)
    sub_g = qg8 & (qvqvs < 1.0)
    epsg_w = jnp.where(
        sub_g,
        2.0 * PI * n0g_w * rho * dv
        * (C.F1S / (lamg_w * lamg_w + eps)
           + C.F2S * (agn * rho / mu) ** 0.5 * sc ** (1.0 / 3.0)
           * C.CONS11 / (lamg_w ** C.CONS36 + eps)), 0.0)
    evpmg_w = jnp.where(sub_g, (qv - qvs) * epsg_w / ab, 0.0)
    evpmg_w = jnp.where(sub_g, jnp.maximum(evpmg_w, pgmlt_w), 0.0)
    pgmlt_w = jnp.where(sub_g, pgmlt_w - evpmg_w, pgmlt_w)

    # reset PRACG/PRACS to zero (only used for melting enhancement)
    pracg_w = jnp.zeros_like(t)
    pracs_w = jnp.zeros_like(t)

    # conservation QC (warm): PRC+PRA
    dum_qc = (prc_w + pra_w) * dt
    cons_qc = (dum_qc > qc) & (qc >= QSMALL)
    ratio = jnp.where(cons_qc, qc / jnp.where(dum_qc != 0, dum_qc, one), 1.0)
    prc_w = prc_w * ratio
    pra_w = pra_w * ratio
    # conservation QNI (snow)
    dum_sn = (-psmlt_w - evpms_w + pracs_w) * dt
    cons_sn = (dum_sn > qs) & (qs >= QSMALL)
    rsn = jnp.where(cons_sn, qs / jnp.where(dum_sn != 0, dum_sn, one), 1.0)
    psmlt_w = psmlt_w * rsn; evpms_w = evpms_w * rsn; pracs_w = pracs_w * rsn
    # conservation QG (graupel)
    dum_g = (-pgmlt_w - evpmg_w + pracg_w) * dt
    cons_g = (dum_g > qg) & (qg >= QSMALL)
    rg = jnp.where(cons_g, qg / jnp.where(dum_g != 0, dum_g, one), 1.0)
    pgmlt_w = pgmlt_w * rg; evpmg_w = evpmg_w * rg; pracg_w = pracg_w * rg
    # conservation QR (PRE negative)
    dum_qr = (-pracs_w - pracg_w - pre_w - pra_w - prc_w + psmlt_w + pgmlt_w) * dt
    cons_qr = (dum_qr > qr) & (qr >= QSMALL)
    rqr = jnp.where(cons_qr & (pre_w != 0),
                    (qr / dt + pracs_w + pracg_w + pra_w + prc_w - psmlt_w - pgmlt_w)
                    / jnp.where(pre_w != 0, -pre_w, one), 1.0)
    pre_w = jnp.where(cons_qr, pre_w * rqr, pre_w)

    # number melting/sub (warm)
    nsubr_w = jnp.where(pre_w < 0.0,
                        jnp.maximum(-1.0, pre_w * dt / jnp.where(qr > 0, qr, one))
                        * nr / dt, 0.0)
    nsmlts_w = jnp.where((evpms_w + psmlt_w) < 0.0,
                         jnp.maximum(-1.0, (evpms_w + psmlt_w) * dt / jnp.where(qs > 0, qs, one))
                         * ns / dt, 0.0)
    nsmltr_w = jnp.where(psmlt_w < 0.0,
                         jnp.maximum(-1.0, psmlt_w * dt / jnp.where(qs > 0, qs, one))
                         * ns / dt, 0.0)
    ngmltg_w = jnp.where((evpmg_w + pgmlt_w) < 0.0,
                         jnp.maximum(-1.0, (evpmg_w + pgmlt_w) * dt / jnp.where(qg > 0, qg, one))
                         * ng / dt, 0.0)
    ngmltr_w = jnp.where(pgmlt_w < 0.0,
                         jnp.maximum(-1.0, pgmlt_w * dt / jnp.where(qg > 0, qg, one))
                         * ng / dt, 0.0)

    # warm tendencies (applied only where warm_active)
    qv_w = -pre_w - evpms_w - evpmg_w
    t_w = (pre_w * xxlv + (evpms_w + evpmg_w) * xxls
           + (psmlt_w + pgmlt_w - pracs_w - pracg_w) * xlf) / cpm
    qc_w = -pra_w - prc_w
    qr_w = pre_w + pra_w + prc_w - psmlt_w - pgmlt_w + pracs_w + pracg_w
    qni_w = psmlt_w + evpms_w - pracs_w
    qg_w = pgmlt_w + evpmg_w - pracg_w
    nc_w = -npra_w - nprc_w
    # NR tendency (Fortran line 2010): NPRC1 + NRAGG - NPRACG; NPRACG is computed
    # in the warm rain-graupel collection block (with shed-drop subtraction) even
    # though PRACG mass is reset to 0. Then + (NSUBR - NSMLTR - NGMLTR) from
    # number melting/sublimation (Fortran line 2044).
    nr_w_t = (nprc1_w + nragg_w - npracg_w) + (nsubr_w - nsmltr_w - ngmltr_w)
    ns_w_t = nsmlts_w
    ng_w_t = ngmltg_w

    wa = warm_active
    qv_ten = jnp.where(wa, qv_ten + qv_w, qv_ten)
    t_ten = jnp.where(wa, t_ten + t_w, t_ten)
    qc_ten = jnp.where(wa, qc_ten + qc_w, qc_ten)
    qr_ten = jnp.where(wa, qr_ten + qr_w, qr_ten)
    qni_ten = jnp.where(wa, qni_ten + qni_w, qni_ten)
    qg_ten = jnp.where(wa, qg_ten + qg_w, qg_ten)
    nc_ten = jnp.where(wa, nc_ten + nc_w, nc_ten)
    nr_ten = jnp.where(wa, nr_ten + nr_w_t, nr_ten)
    ns_ten = jnp.where(wa, ns_ten + ns_w_t, ns_ten)
    ng_ten = jnp.where(wa, ng_ten + ng_w_t, ng_ten)

    # =====================================================================
    # COLD BRANCH (T < 273.15)
    # =====================================================================
    cold = (~warm) & do_cell
    out = _cold_branch(
        cold, t, qv, qc, qr, qi, qs, qg, nc, ni, ns, nr, ng,
        qvs, qvi, qvqvs, qvqvsi, ab, abi, rho, dv, mu, sc, kap,
        ain, arn, asn, agn, acn, dum_dc, xxlv, xxls, xlf, cpm, p, dt)
    (qv_c, t_c, qc_c, qr_c, qi_c, qni_c, qg_c,
     nc_c, ni_c, ns_c, nr_c, ng_c,
     ni_clamp, ns_clamp, nr_clamp, ng_clamp) = out

    qv_ten = jnp.where(cold, qv_ten + qv_c, qv_ten)
    t_ten = jnp.where(cold, t_ten + t_c, t_ten)
    qc_ten = jnp.where(cold, qc_ten + qc_c, qc_ten)
    qr_ten = jnp.where(cold, qr_ten + qr_c, qr_ten)
    qi_ten = jnp.where(cold, qi_ten + qi_c, qi_ten)
    qni_ten = jnp.where(cold, qni_ten + qni_c, qni_ten)
    qg_ten = jnp.where(cold, qg_ten + qg_c, qg_ten)
    nc_ten = jnp.where(cold, nc_ten + nc_c, nc_ten)
    ni_ten = jnp.where(cold, ni_ten + ni_c, ni_ten)
    ns_ten = jnp.where(cold, ns_ten + ns_c, ns_ten)
    nr_ten = jnp.where(cold, nr_ten + nr_c, nr_ten)
    ng_ten = jnp.where(cold, ng_ten + ng_c, ng_ten)
    # apply slope number clamps on cold cells
    ni = jnp.where(cold, ni_clamp, ni)
    ns = jnp.where(cold, ns_clamp, ns)
    nr = jnp.where(cold, nr_clamp, nr)
    ng = jnp.where(cold, ng_clamp, ng)

    # =====================================================================
    # SATURATION ADJUSTMENT (liquid) PCC, all do_cell cells (warm 300 & cold both)
    # =====================================================================
    dumt = t + dt * t_ten
    dumqv = qv + dt * qv_ten
    dum_svp = jnp.minimum(0.99 * p, polysvp(dumt, 0))
    dumqss = EP_2 * dum_svp / (p - dum_svp)
    dumqc = jnp.maximum(qc + dt * qc_ten, 0.0)
    dums = dumqv - dumqss
    pcc = dums / (1.0 + xxlv ** 2 * dumqss / (cpm * RV * dumt ** 2)) / dt
    pcc = jnp.where(pcc * dt + dumqc < 0.0, -dumqc / dt, pcc)
    pcc = jnp.where(do_cell, pcc, 0.0)
    qv_ten = qv_ten - pcc
    t_ten = t_ten + pcc * xxlv / cpm
    qc_ten = qc_ten + pcc

    # =====================================================================
    # SEDIMENTATION (Reisner 1998 split-step). Operates over the column.
    # =====================================================================
    (qr_st, qi_st, qni_st, qc_st, qg_st,
     ni_sed, ns_sed, nr_sed, nc_sed, ng_sed,
     precrt, snowrt, snowprt, grplprt) = _sedimentation(
        qc, qi, qs, qr, qg, nc, ni, ns, nr, ng,
        qc_ten, qi_ten, qni_ten, qr_ten, qg_ten,
        nc_ten, ni_ten, ns_ten, nr_ten, ng_ten,
        t, p, rho, dz, dt, do_cell)

    # add sedimentation number tendencies to the eulerian ones
    ni_ten = ni_ten + ni_sed
    ns_ten = ns_ten + ns_sed
    nr_ten = nr_ten + nr_sed
    nc_ten = nc_ten + nc_sed
    ng_ten = ng_ten + ng_sed
    # add sedimentation mass tendencies
    qr_ten = qr_ten + qr_st
    qi_ten = qi_ten + qi_st
    qc_ten = qc_ten + qc_st
    qg_ten = qg_ten + qg_st
    qni_ten = qni_ten + qni_st

    # =====================================================================
    # FINAL STATE UPDATE + instantaneous processes + slope recompute + Reff
    # =====================================================================
    res = _finalize(t, qv, qc, qi, qs, qr, qg, nc, ni, ns, nr, ng,
                    qc_ten, qi_ten, qni_ten, qr_ten, qg_ten,
                    t_ten, qv_ten,
                    nc_ten, ni_ten, ns_ten, nr_ten, ng_ten,
                    xxlv, xxls, xlf, cpm, p, rho, dt, do_cell)
    (t, qv, qc, qi, qs, qr, qg, nc, ni, ns, nr, ng,
     effc, effi, effs, effr, effg) = res

    # wrapper: TH = T/PII
    th_out = t / pii

    # surface precip binding (PRECPRT1D etc. are mm; SR ratio)
    rainncv = precrt
    snowncv = snowprt
    graupelncv = grplprt
    sr = snowrt / (precrt + 1.0e-12)

    return {
        "th": th_out, "qv": qv, "qc": qc, "qr": qr, "qi": qi, "qs": qs, "qg": qg,
        "ni": ni, "ns": ns, "nr": nr, "ng": ng,
        "effc": effc, "effi": effi, "effs": effs, "effr": effr, "effg": effg,
        "rainncv": rainncv, "snowncv": snowncv, "graupelncv": graupelncv, "sr": sr,
    }


# Cold-branch and sedimentation/finalize are large; defined below.
from gpuwrf.physics._morrison_cold import _cold_branch  # noqa: E402
from gpuwrf.physics._morrison_sed import _sedimentation, _finalize  # noqa: E402


# ===========================================================================
# Adapter: frozen PhysicsTendency per the S0 interface.
# ===========================================================================
def morrison_tendency(th, qv, qc, qr, qi, qs, qg, ni, ns, nr, ng,
                      pii, p, dz, w, dt):
    """Run Morrison and return a frozen PhysicsTendency (state_replacements)."""
    out = morrison_run(th, qv, qc, qr, qi, qs, qg, ni, ns, nr, ng, pii, p, dz, w, dt)
    return PhysicsTendency(
        state_replacements={
            "theta": out["th"], "qv": out["qv"], "qc": out["qc"], "qr": out["qr"],
            "qi": out["qi"], "qs": out["qs"], "qg": out["qg"],
            "Ni": out["ni"], "Ns": out["ns"], "Nr": out["nr"], "Ng": out["ng"],
        },
        accumulator_increments={
            "rain_acc": out["rainncv"], "snow_acc": out["snowncv"],
            "graupel_acc": out["graupelncv"],
        },
        diagnostics={
            "re_cloud": out["effc"], "re_ice": out["effi"], "re_snow": out["effs"],
        },
    )
