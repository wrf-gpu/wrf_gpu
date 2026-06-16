"""JAX port of the WRF MYNN-EDMF Dynamic Multi-Plume (DMP) mass-flux scheme.

Faithful transcription of `phys/module_bl_mynnedmf.F90:DMP_mf` (WRF v4 pristine,
`/home/enric/src/wrf_pristine/WRF/phys/module_bl_mynnedmf.F`) restricted to the
operational d03 configuration:

  bl_mynn_edmf      = 1   (mass-flux ON)
  bl_mynn_edmf_mom  = 1   (WRF default; momentum MF -> s_awu/s_awv)
  bl_mynn_edmf_tke  = 0   (no TKE MF)
  bl_mynn_mixscalars= 1   (scalar MF on; but qnc/qni/aerosols not carried here)
  bl_mynn_mixqt     = 0   (mix qv/qc separately -> the "MIX WATER VAPOR ONLY" path)
  env_subs          = .false. (-> sub_sqv = det_sqv = 0; only s_awqv matters)
  bl_mynn_edmf_dd   = 0   (no downdraft -> all sd_* = 0)

Under this config the mass-flux terms entering `mynn_tendencies` are `s_awqv1`
(the updraft total-water minus updraft-condensate flux), `s_awthl1`, and the
momentum fluxes `s_awu1`/`s_awv1`. This module produces the interface-staggered
solver arrays consumed by the augmented implicit solves in
`mynn_pbl._apply_mean_tendencies`.

Inputs/outputs use specific-content moisture (sqv/sqc/sqw), matching WRF: the
DMP integration carries qt1=sqw, qv1=sqv, qc1=sqc.

All arrays are batched over an arbitrary leading column dimension `(..., nz)`.
Interface arrays have length `nz+1` with index 0 = surface (always 0), index k =
top of model layer k-1. fp64 throughout.

WRF file:line anchors are cited inline.
"""

from __future__ import annotations

import os
from functools import partial

import jax
from jax import config
import jax.numpy as jnp
from jax import lax

config.update("jax_enable_x64", True)


def _cond_niter() -> int:
    """Condensation fixed-point iteration cap (v0.15 S1 host-removal knob).

    Default ``16`` (v0.15) is the WRF-faithful cap and the dominant v0.15 wall
    win (-70 ms/step, the only strong device axis; S1 bisect row D).  WRF's own
    cap is 50 (``module_bl_mynnedmf.F:6794-6851``) BUT it carries an
    ``if (abs(QC-QCold)<diff) exit`` early-out and "usually converges in < 8
    iterations"; the port dropped the data-dependent exit (it would lower to a
    capture-breaking ``while``), so the WRF-faithful fixed cap is SMALLER than
    50, not larger: ``16`` keeps the residual below WRF's own exit threshold
    (diff=1e-6) wherever the iteration CONVERGES (measured 1.6e-7 worst case);
    in the warm+very-moist corner the WRF iteration itself does not converge
    (qs-feedback gain beats the 0.5 damping) and qc50 vs qc16 are different
    phase samples of the oscillation -- WRF's 50th iterate is equally arbitrary
    there.  This is therefore a TIERED (not bitwise) identity change: the niter
    oracle (proofs/perf/v015/cond_niter_oracle.json) bounds the convergent
    regime, and the frozen-tolerance field gate PASSES at 0.06% of the manifest
    limits (proofs/perf/v015/tiered_gate_cond16.json / tiered_gate_combined.json).
    Rollback to the WRF hard cap with ``GPUWRF_MYNN_COND_NITER=50``.
    """

    return max(1, int(os.environ.get("GPUWRF_MYNN_COND_NITER", "16")))


def _cond_unroll() -> bool:
    """Lower the condensation fixed point as a Python-unrolled chain (v0.15 S1).

    Default OFF (v0.15): keep the v0.14 ``lax.fori_loop`` lowering.  Setting it
    ON peels the fixed point into a straight-line chain and removes the innermost
    ``while`` so the EDMF plume body fuses into one kernel; it is numerically the
    SAME (bitwise-proven at niter=16 by the cond-niter oracle
    ``qc16_unrolled_vs_fori_bitwise=true`` in
    proofs/perf/v015/cond_niter_oracle.json -- loop peeling, no reassociation).

    Why default OFF despite the S1 ship candidate using it: peeling 16 EDMF
    condensation bodies (each already carrying the unrolled EDMF level loop)
    triggers an XLA "Very slow compile" pathology -- a >10 min first-compile of
    a huge HLO, reproduced on this branch at both THOMAS_UNROLL=8 and =45.  The
    DOMINANT wall win is the niter cut itself (50->16 = -70 ms; S1 bisect), which
    the fori lowering captures too (16 iterations instead of 50 either way).  The
    unroll adds only a marginal fusion benefit on top and is not worth the
    compile-ergonomics cost as a default.  Set ``GPUWRF_MYNN_COND_UNROLL=1`` to
    opt into the fully-fused variant (e.g. for a long production run where the
    one-time compile amortizes and command-buffer capture is also engaged).
    """

    return os.environ.get("GPUWRF_MYNN_COND_UNROLL", "0") == "1"


def _edmf_level_unroll() -> int:
    """Unroll factor for the 42-level EDMF plume-rise ``lax.scan`` (v0.15 S1).

    Default ``1`` is the exact v0.14 lowering.  Values > 1 peel the level loop
    (same-op-order class as the Thomas unroll); a value >= nz-2 (44-2 = 42 for
    the d01 case -- use e.g. 64) fully flattens the scan so no ``while`` thunk
    remains and the EDMF chain joins the step's command buffer.  Fusion
    boundaries shift => TIERED identity regime, not bitwise.
    """

    return max(1, int(os.environ.get("GPUWRF_MYNN_EDMF_LEVEL_UNROLL", "1")))

# ---- WRF model constants (module_model_constants.F) ----
R_D = 287.0
CP = 7.0 * R_D / 2.0
R_V = 461.6
P608 = R_V / R_D - 1.0
RVOVRD = R_V / R_D
EP_2 = R_D / R_V
P1000MB = 100000.0
RCP = R_D / CP
GRAV = 9.81
GTR = GRAV / 300.0  # gtr = grav/tref, tref=300
XLV = 2.5e6
XLVCP = XLV / CP
T0C = 273.15
TICE = 240.0
ONETHIRD = 1.0 / 3.0

# ---- DMP_mf parameters (module_bl_mynnedmf.F:5686-5776) ----
NUP = 8
ATOT = 0.10
LMAX = 1000.0
LMIN = 300.0
DCUT = 1.2
DNEGGERS = -1.9
PWMIN = 0.1
PWMAX = 0.4
Z0_PLUME = 50.0
CSIGMA = 1.34
FLUXPORTION = 0.75
# env_subs=.false. -> exc_fac land = 0.58, water = 0.58*4.0
# (module_bl_mynnedmf.F90:6052-6063).
EXC_FAC_LAND = 0.58
EXC_FAC_WATER = 0.58 * 4.0


def _qsat_blend(t, p):
    """WRF `qsat_blend` (module_bl_mynnedmf.F:7619-7668), liquid/ice blend."""
    J = (0.611583699e3, 0.444606896e2, 0.143177157e1, 0.264224321e-1,
         0.299291081e-3, 0.203154182e-5, 0.702620698e-8, 0.379534310e-11,
         -0.321582393e-13)
    K = (0.609868993e3, 0.499320233e2, 0.184672631e1, 0.402737184e-1,
         0.565392987e-3, 0.521693933e-5, 0.307839583e-7, 0.105785160e-9,
         0.161444444e-12)
    xc = jnp.maximum(-80.0, t - T0C)

    def horner(coef):
        acc = coef[8]
        for c in coef[7::-1]:
            acc = c + xc * acc
        return acc

    esl = jnp.minimum(horner(J), p * 0.15)
    esi = jnp.minimum(horner(K), p * 0.15)
    rslf = 0.622 * esl / jnp.maximum(p - esl, 1e-5)
    rsif = 0.622 * esi / jnp.maximum(p - esi, 1e-5)
    chi = ((T0C - 6.0) - t) / ((T0C - 6.0) - TICE)
    qs_liq = rslf
    qs_ice = rsif
    qs_blend = (1.0 - chi) * rslf + chi * rsif
    out = jnp.where(t >= (T0C - 6.0), qs_liq,
                    jnp.where(t <= TICE, qs_ice, qs_blend))
    return out


def _condensation_edmf(qt, thl, p, zagl, niter=None):
    """WRF `condensation_edmf` (line 6794): zero/one moist adjustment for a plume.

    Returns (thv, qc). fp64. Fixed iteration count (JAX-friendly; WRF caps at 50
    and converges in <8). The `if (abs(QC-QCold)<diff) exit` early-out is dropped
    (extra iterations are idempotent once converged).

    v0.15 S1 knobs (see `_cond_niter`/`_cond_unroll` docstrings): the default
    (niter=50, fori) is the exact v0.14 numerics AND lowering; the S1 candidate
    (GPUWRF_MYNN_COND_NITER=16 + GPUWRF_MYNN_COND_UNROLL=1) is the WRF-faithful
    convergence cap, Python-unrolled so the innermost ``while`` -- 82% of all
    kernel time and the largest launch family -- disappears into one fusion."""
    if niter is None:
        niter = _cond_niter()
    exn = (p / P1000MB) ** RCP
    qc = jnp.zeros_like(qt)

    def body(_, qc):
        t = exn * thl + XLVCP * qc
        qs = _qsat_blend(t, p)
        return 0.5 * qc + 0.5 * jnp.maximum(qt - qs, 0.0)

    if _cond_unroll():
        for _ in range(int(niter)):
            qc = body(None, qc)
    else:
        qc = lax.fori_loop(0, niter, body, qc)
    t = exn * thl + XLVCP * qc
    qs = _qsat_blend(t, p)
    qc = jnp.maximum(qt - qs, 0.0)
    qc = jnp.where(zagl < 100.0, 0.0, qc)
    thv = (thl + XLVCP * qc) * (1.0 + qt * (RVOVRD - 1.0) - RVOVRD * qc)
    return thv, qc


def _single_column_dmp_mf(sqw, sqv, sqc, u, v, w, th, thl, thv, tk, qke,
                          p, exner, rho, dz, zw, ust, flt, fltv, flq, flqv,
                          pblh, ts, xland, psig_shcu, *, dx, dt):
    """Port of DMP_mf for ONE column. All inputs are 1-D arrays length nz
    (interfaces zw length nz+1). Returns dict of solver arrays.

    Mirrors module_bl_mynnedmf.F:DMP_mf for the operational config
    (momentum_opt=1, tke_opt=0, env_subs=.false., no chem). Carries qt1=sqw,
    qv1=sqv, qc1=sqc (specific contents).

    The full WRF `DMP_mf` argument list is preserved so the JAX call mirrors the
    Fortran interface; under this config several args are intentionally unused:
    ``tk``/``exner`` (only WRF diagnostics), ``qke``/``ust`` (TKE-MF path off),
    ``flqv`` (only separate vapor bookkeeping; ``flq`` drives the plume excess),
    and ``dt`` (only env_subs subsidence, which is off). They are
    accepted-and-ignored rather than dropped.
    """
    del tk, qke, exner, ust, flqv, dt  # WRF-interface args unused in this config
    dtype = th.dtype
    dx = jnp.asarray(dx, dtype=dtype)
    psig_shcu = jnp.asarray(psig_shcu, dtype=dtype)
    nz = th.shape[-1]
    is_water = (xland - 1.5) >= 0.0
    qv1 = sqv  # WRF names: qv1==sqv, qt1==sqw, qc1==sqc inside DMP_mf
    qt1 = sqw
    del sqc  # qc1 unused: surface updraft UPQC(1,ip)=0, plume qc from condensation_edmf

    # ---- activation: maxw / Psig_w (lines 5855-5879) ----
    zagl_mid = zw[:-1] + 0.5 * dz  # length nz
    below = zagl_mid <= (pblh + 500.0)
    wpbl = jnp.where(w < 0.0, 2.0 * w, w)
    maxw = jnp.max(jnp.where(below, jnp.abs(wpbl), 0.0))
    maxw = jnp.maximum(0.0, maxw - 1.0)
    psig_w = jnp.maximum(0.0, 1.0 - maxw)
    psig_w = jnp.minimum(psig_w, psig_shcu)

    fltv2 = jnp.where((psig_w == 0.0) & (fltv > 0.0), -fltv, fltv)

    # ---- superadiabatic check up through ~50 m (lines 5892-5918) ----
    # WRF superadiabatic test (lines 5899-5918): the k=0 criterion is
    #   dthvdz = (thv0 - tvs)/(0.5*dz0) < hux, with hux=-0.003 over land
    #   and -0.001 over water,
    #   tvs = ts*(1+p608*qv0)  -- requires the skin temperature ts.
    # This is the governing check (the k=1..k50 ladder uses hux=-0.0005 and only
    # adds levels while contiguous; for 1km d03 the first mass level is ~25 m so
    # k=0 dominates).
    #
    # When ts (skin temperature) IS supplied (ts>0, the WRF coupling / oracle
    # path) we use the exact WRF criterion. When ts is NOT available (ts<=0
    # sentinel) we fall back to the physically-equivalent buoyancy-flux criterion
    # ``fltv2 > 0``: a positive surface virtual-heat flux IS by definition an
    # unstable (superadiabatic) surface layer, which is the condition WRF's test
    # detects. This avoids fabricating a skin temperature from an underdetermined
    # kinematic-flux/ustar relation (which needs the surface exchange coefficient
    # ch we don't carry in the standalone column).
    tvs = ts * (1.0 + P608 * qv1[0])
    dthvdz0 = (thv[0] - tvs) / (0.5 * dz[0])
    hux0 = jnp.where(is_water, -0.001, -0.003)
    superad = jnp.where(ts > 0.0, dthvdz0 < hux0, fltv2 > 0.0)

    # ---- plume widths (lines 5933-5975) ----
    maxwidth_dx = jnp.minimum(dx * DCUT, LMAX)
    maxwidth_pbl = jnp.minimum(1.1 * pblh, LMAX)
    cloud_base = 9000.0  # no SGS cloud deck height carried -> default
    cloud_factor = jnp.where(is_water, 0.9, 0.5)
    maxwidth_cld = jnp.minimum(LMAX, jnp.maximum(cloud_factor * cloud_base, 400.0))
    wspd_pbl = jnp.sqrt(jnp.maximum(u[0] ** 2 + v[0] ** 2, 0.01))
    maxwidth_flx_land = 1000.0 * (0.6 * jnp.tanh((fltv - 0.040) / 0.04) + 0.5)
    maxwidth_flx_water = 1000.0 * (0.6 * jnp.tanh((fltv - 0.007) / 0.02) + 0.5)
    maxwidth_flx = jnp.maximum(
        jnp.minimum(jnp.where(is_water, maxwidth_flx_water, maxwidth_flx_land), 1000.0),
        0.0)
    maxwidth = jnp.minimum(jnp.minimum(maxwidth_dx, maxwidth_pbl),
                           jnp.minimum(maxwidth_cld, maxwidth_flx))
    minwidth = LMIN

    active = (fltv2 > 0.002) & (maxwidth > minwidth) & superad

    # ---- number density coefficient C (lines 5981-5989) ----
    dl = (maxwidth - minwidth) / (NUP - 1)
    ip = jnp.arange(NUP)
    l_arr = minwidth + dl * ip  # plume diameters
    cn = jnp.sum(l_arr ** DNEGGERS * (l_arr * l_arr) / (dx * dx) * dl)
    C = ATOT / jnp.maximum(cn, jnp.asarray(1e-30, dtype=dtype))

    # ---- area fraction acfac (lines 5992-6008, land) ----
    acfac_land = 0.5 * jnp.tanh((fltv2 - 0.020) / 0.05) + 0.5
    acfac_water = 0.5 * jnp.tanh((fltv2 - 0.012) / 0.03) + 0.5
    acfac = jnp.where(is_water, acfac_water, acfac_land)
    ac_wsp = jnp.where(wspd_pbl <= 10.0, 1.0,
                       1.0 - jnp.minimum(jnp.maximum(wspd_pbl - 13.0, 0.0) / 10.0, 1.0))
    acfac = jnp.minimum(acfac, ac_wsp)

    # initial plume area UPA0 (lines 6012-6019)
    N_arr = C * l_arr ** DNEGGERS
    upa0 = N_arr * l_arr * l_arr / (dx * dx) * dl * acfac  # length NUP

    # ---- surface updraft excess scales (lines 6033-6069) ----
    wstar = jnp.maximum(1e-2, (GTR * fltv2 * pblh) ** ONETHIRD)
    qstar = jnp.maximum(flq, 1e-5) / wstar
    thstar = flt / wstar
    exc_fac = jnp.where(is_water, EXC_FAC_WATER, EXC_FAC_LAND) * ac_wsp
    sigmaw = CSIGMA * wstar * (Z0_PLUME / pblh) ** ONETHIRD * (1.0 - 0.8 * Z0_PLUME / pblh)
    sigmaqt = CSIGMA * qstar * (Z0_PLUME / pblh) ** ONETHIRD
    sigmath = CSIGMA * thstar * (Z0_PLUME / pblh) ** ONETHIRD
    wmin_s = jnp.minimum(sigmaw * PWMIN, 0.1)
    wmax_s = jnp.minimum(sigmaw * PWMAX, 0.5)

    # surface updraft properties at interface 1 (between k=0 & 1) (lines 6079-6104)
    ipf = (ip + 1).astype(dtype)
    upw0 = wmin_s + ipf / NUP * (wmax_s - wmin_s)  # length NUP
    # interface-averaged env values between level 0 and 1
    a01 = dz[1] / (dz[0] + dz[1])
    b01 = dz[0] / (dz[0] + dz[1])
    u01 = u[0] * a01 + u[1] * b01
    v01 = v[0] * a01 + v[1] * b01
    thv01 = thv[0] * a01 + thv[1] * b01
    thl01 = thl[0] * a01 + thl[1] * b01
    qt01 = qt1[0] * a01 + qt1[1] * b01
    exc_heat = exc_fac * upw0 * sigmath / sigmaw
    exc_moist = exc_fac * upw0 * sigmaqt / sigmaw
    upthl0 = thl01 + exc_heat
    upthv0 = thv01 + exc_heat
    upqt0 = qt01 + exc_moist
    upu0 = jnp.full((NUP,), u01)
    upv0 = jnp.full((NUP,), v01)
    upqc0 = jnp.zeros((NUP,), dtype=dtype)

    # rhoz on interfaces (lines 6120-6123): rhoz(k)=(rho(k)*dz(k+1)+rho(k+1)*dz(k))/(dz(k+1)+dz(k))
    rhoz_mid = (rho[:-1] * dz[1:] + rho[1:] * dz[:-1]) / (dz[1:] + dz[:-1])  # len nz-1, idx k -> interface above level k
    rhoz = jnp.concatenate([rhoz_mid, rho[-1:]])  # length nz: rhoz[k] valid for k=0..nz-1

    # ---- per-plume vertical integration (lines 6128-6330) ----
    # We scan k from 1..nz-2 (WRF: kts+1..kte-1, 0-based 1..nz-2), carrying the
    # updraft state. State per plume: (w, thl, qt, qc, thv, u, v, area, alive, ktop_flag)
    # Outputs accumulated on interfaces: UPA(k), UPW(k), UPQT(k), UPQC(k), UPTHL(k), UPTHV(k)
    # for k in 1..nz-2 (the surface interface k=0 carries upw0 etc.)

    l_per_plume = minwidth + dl * ip.astype(dtype)  # length NUP

    def plume_scan(l, a0, w0, thl0, qt0, qc0, u0, v0):
        # carry: w_prev, thl_prev, qt_prev, qc_prev, u_prev, v_prev, area_prev, alive
        carry0 = (w0, thl0, qt0, qc0, u0, v0, a0, jnp.array(True))

        def step(carry, k):
            w_p, thl_p, qt_p, qc_p, u_p, v_p, area_p, alive = carry
            # entrainment (lines 6136-6156)
            wmin_e = 0.3 + l * 0.0005
            ent = 0.33 / (jnp.minimum(jnp.maximum(w_p, wmin_e), 0.9) * l)
            ent = jnp.maximum(ent, 0.0003)
            ent = jnp.where(zw[k] >= jnp.minimum(pblh + 1500.0, 4000.0),
                            ent + (zw[k] - jnp.minimum(pblh + 1500.0, 4000.0)) * 5.0e-6, ent)
            ent = jnp.minimum(ent, 0.9 / (zw[k + 1] - zw[k]))

            entexp = ent * (zw[k + 1] - zw[k])
            # interface-averaged env values between level k and k+1 (for Pk, THVk)
            ak = dz[k + 1] / (dz[k + 1] + dz[k])
            bk = dz[k] / (dz[k + 1] + dz[k])
            # Entrainment uses MASS-level qt1(k)/thl(k) (WRF line 6167-6168):
            #   QTn = UPQT(k-1)*(1-EntExp) + qt1(k)*EntExp
            qtn = qt_p * (1.0 - entexp) + qt1[k] * entexp
            thln = thl_p * (1.0 - entexp) + thl[k] * entexp

            pk = p[k] * ak + p[k + 1] * bk
            thvn, qcn = _condensation_edmf(qtn, thln, pk, zw[k + 1])

            thvk = thv[k] * ak + thv[k + 1] * bk
            B = GRAV * (thvn / thvk - 1.0)
            bcoeff = jnp.where(B > 0.0, 0.15, 0.2)

            dzc = jnp.minimum(zw[k] - zw[k - 1], 250.0)
            wterm = (-2.0 * ent * w_p + bcoeff * B / jnp.maximum(w_p, 0.2)) * dzc
            wn = w_p + wterm
            # symmetric accel limiter (lines 6233-6239)
            lim = jnp.minimum(1.25 * (zw[k] - zw[k - 1]) / 200.0, 2.0)
            wn = jnp.minimum(wn, w_p + lim)
            wn = jnp.maximum(wn, w_p - lim)
            wn = jnp.minimum(jnp.maximum(wn, 0.0), 3.0)

            un = u_p * (1.0 - entexp * 0.3333) + u[k] * entexp * 0.3333
            vn = v_p * (1.0 - entexp * 0.3333) + v[k] * entexp * 0.3333

            still = alive & (wn > 0.0)
            # plume area constant while alive (line 6315: UPA(K)=UPA(K-1))
            area_n = jnp.where(still, area_p, 0.0)

            # emit interface values at level k (only if still alive)
            emit_a = jnp.where(still, area_p, 0.0)
            emit_w = jnp.where(still, wn, 0.0)
            emit_qt = jnp.where(still, qtn, 0.0)
            emit_qc = jnp.where(still, qcn, 0.0)
            emit_thl = jnp.where(still, thln, 0.0)

            new_carry = (jnp.where(still, wn, w_p),
                         jnp.where(still, thln, thl_p),
                         jnp.where(still, qtn, qt_p),
                         jnp.where(still, qcn, qc_p),
                         jnp.where(still, un, u_p),
                         jnp.where(still, vn, v_p),
                         area_n, still)
            emit_u = jnp.where(still, un, 0.0)
            emit_v = jnp.where(still, vn, 0.0)
            return new_carry, (emit_a, emit_w, emit_qt, emit_qc, emit_thl, emit_u, emit_v)

        ks = jnp.arange(1, nz - 1)
        # v0.15 S1: optional level-loop peel (see `_edmf_level_unroll`); the
        # default 1 lowers byte-equal to the pre-v0.15 `lax.scan(step, c, ks)`.
        _lvl_u = _edmf_level_unroll()
        _, (ea, ew, eqt, eqc, ethl, eu, ev) = lax.scan(
            step, carry0, ks, unroll=_lvl_u if _lvl_u > 1 else False
        )
        # ea[i] corresponds to WRF level K = i+1 (Fortran kts+1..kte-1), i.e. the
        # scanned levels 1..nz-2 (0-based). Build full UP arrays of length nz with:
        #   UP[0] = surface updraft (upw0 etc., WRF UPW(1,ip))
        #   UP[1..nz-2] = scan results
        #   UP[nz-1] = 0 (not integrated)
        return ea, ew, eqt, eqc, ethl, eu, ev

    # vmap over plumes (all per-plume surface arrays are length NUP)
    EA_s, EW_s, EQT_s, EQC_s, ETHL_s, EU_s, EV_s = jax.vmap(plume_scan)(
        l_per_plume, upa0, upw0, upthl0, upqt0, upqc0, upu0, upv0)
    # *_s shape (NUP, nz-2) for levels K=1..nz-2 (0-based).

    # Prepend the surface updraft (WRF UPW(1,ip)=upw0) at 0-based level K=0, and
    # append a zero top level -> full UP arrays length nz (index = WRF level K).
    def full_up(surf, scan):
        return jnp.concatenate(
            [surf[:, None], scan, jnp.zeros((NUP, 1), dtype=dtype)], axis=1)  # (NUP, nz)
    UPA = full_up(upa0, jnp.broadcast_to(upa0[:, None], (NUP, nz - 2)) * (EW_s > 0))
    # WRF: UPA(K)=UPA(K-1) while alive, else plume stops -> area persists on live
    # levels. Reconstruct area as upa0 on live (EW_s>0) scanned levels, 0 above.
    UPW = full_up(upw0, EW_s)
    UPQT = full_up(upqt0, EQT_s)
    UPQC = full_up(upqc0, EQC_s)
    UPTHL = full_up(upthl0, ETHL_s)
    UPU = full_up(upu0, EU_s)
    UPV = full_up(upv0, EV_s)

    # ---- assemble s_aw* (lines 6363-6382): s_aw1(k+1) += rhoz(k)*UPA(K)*UPW(K)*Psig_w
    # for K=kts..kte-1 (0-based 0..nz-2). rhoz_dmp(K) is the interface ABOVE level K.
    rhoz_dmp = jnp.concatenate([rhoz_mid, rho[-1:]])  # length nz; [K]=interface above level K
    Kmask = (jnp.arange(nz) <= nz - 2).astype(dtype)[None, :]  # K=0..nz-2
    upa_w = UPA * UPW  # (NUP, nz)
    wgt = rhoz_dmp[None, :] * upa_w * Kmask
    s_aw_inner = jnp.sum(wgt, axis=0) * psig_w          # length nz; index K
    s_awqt_inner = jnp.sum(wgt * UPQT, axis=0) * psig_w
    s_awqc_inner = jnp.sum(wgt * UPQC, axis=0) * psig_w
    s_awthl_inner = jnp.sum(wgt * UPTHL, axis=0) * psig_w
    s_awu_inner = jnp.sum(wgt * UPU, axis=0) * psig_w
    s_awv_inner = jnp.sum(wgt * UPV, axis=0) * psig_w
    # WRF writes these to s_aw1(K+1): shift up by one -> length nz+1, [0]=0
    def to_iface(inner):
        return jnp.concatenate([jnp.zeros((1,)), inner])  # length nz+1, drops last
    s_aw = to_iface(s_aw_inner)[: nz + 1]
    s_awqt = to_iface(s_awqt_inner)[: nz + 1]
    s_awqc = to_iface(s_awqc_inner)[: nz + 1]
    s_awthl = to_iface(s_awthl_inner)[: nz + 1]
    s_awu = to_iface(s_awu_inner)[: nz + 1]
    s_awv = to_iface(s_awv_inner)[: nz + 1]
    s_awqv = s_awqt - s_awqc  # line 6380

    # ---- flux limiter (lines 6423-6461) ----
    dzi0 = 0.5 * (dz[0] + dz[1])
    flx1 = jnp.where(s_aw[1] != 0.0,
                     jnp.maximum(s_aw[1] * (thv[0] - thv[1]) / dzi0, 1.0e-6),
                     0.0)
    flt2 = jnp.maximum(fltv, 0.0)
    need = (flx1 > FLUXPORTION * flt2 / dz[0]) & (flx1 > 0.0)
    adjustment = jnp.where(need,
                           jnp.maximum(0.01, FLUXPORTION * flt2 / dz[0] / jnp.maximum(flx1, 1e-30)),
                           1.0)
    s_aw = s_aw * adjustment
    s_awqt = s_awqt * adjustment
    s_awqc = s_awqc * adjustment
    s_awqv = s_awqv * adjustment
    s_awthl = s_awthl * adjustment
    s_awu = s_awu * adjustment
    s_awv = s_awv * adjustment

    # zero everything if not active
    zero1 = jnp.zeros((nz + 1,), dtype=dtype)
    s_aw = jnp.where(active, s_aw, zero1)
    s_awqv = jnp.where(active, s_awqv, zero1)
    s_awqt = jnp.where(active, s_awqt, zero1)
    s_awqc = jnp.where(active, s_awqc, zero1)
    s_awthl = jnp.where(active, s_awthl, zero1)
    s_awu = jnp.where(active, s_awu, zero1)
    s_awv = jnp.where(active, s_awv, zero1)

    # edmf_a / maxmf diagnostics (mean over plumes; lines 6470-6491)
    edmf_a_inner = jnp.sum(UPA, axis=0) * psig_w        # length nz
    edmf_aw_inner = jnp.sum(upa_w, axis=0) * psig_w
    edmf_a_inner = jnp.where(active, edmf_a_inner, 0.0)
    maxmf = jnp.max(jnp.where(active, edmf_aw_inner, 0.0))

    # mean in-plume properties (WRF lines 870-889): area-weighted plume means
    # edmf_qt1/edmf_thl1/edmf_qc1 = sum(UPA*UPX)/sum(UPA) where sum(UPA)>0; only
    # edmf_a is multiplied by Psig_w. Consumed by the DMP shallow-cu
    # cldfra/qc_bl overwrite (mynn_sgs_cloud.dmp_shallow_cu_overwrite).
    upa_sum = jnp.sum(UPA, axis=0)                      # length nz, pre-Psig
    safe = jnp.maximum(upa_sum, jnp.asarray(1e-30, dtype=dtype))
    has_a = upa_sum > 0.0
    edmf_qc_inner = jnp.where(has_a, jnp.sum(UPA * UPQC, axis=0) / safe, 0.0)
    edmf_qt_inner = jnp.where(has_a, jnp.sum(UPA * UPQT, axis=0) / safe, 0.0)
    edmf_thl_inner = jnp.where(has_a, jnp.sum(UPA * UPTHL, axis=0) / safe, 0.0)
    edmf_qc_inner = jnp.where(active, edmf_qc_inner, 0.0)
    edmf_qt_inner = jnp.where(active, edmf_qt_inner, 0.0)
    edmf_thl_inner = jnp.where(active, edmf_thl_inner, 0.0)

    return {
        "s_aw": s_aw,
        "s_awqv": s_awqv,
        "s_awqt": s_awqt,
        "s_awqc": s_awqc,
        "s_awthl": s_awthl,
        "s_awu": s_awu,
        "s_awv": s_awv,
        "edmf_a": edmf_a_inner,
        "edmf_qc": edmf_qc_inner,
        "edmf_qt": edmf_qt_inner,
        "edmf_thl": edmf_thl_inner,
        "maxmf": maxmf,
        "active": active.astype(dtype),
        "psig_w": psig_w,
    }


def dmp_mf_columns(sqw, sqv, sqc, u, v, w, th, thl, thv, tk, qke,
                   p, exner, rho, dz, zw, ust, flt, fltv, flq, flqv,
                   pblh, ts, dx, xland, dt, psig_shcu=None):
    """Batched DMP_mf over a leading column dimension.

    All profile args shape (B, nz); surface args (flt, fltv, flq, flqv, ust,
    pblh, ts, xland) shape (B,); zw shape (B, nz+1). Returns dict of (B, nz+1)
    solver arrays (plus diagnostics). dx scalar."""
    B = th.shape[0]
    if psig_shcu is None:
        psig_shcu = jnp.ones((B,), dtype=th.dtype)

    return jax.vmap(
        lambda *a: _single_column_dmp_mf(*a[:-1], dx=dx, dt=dt, psig_shcu=a[-1])
    )(sqw, sqv, sqc, u, v, w, th, thl, thv, tk, qke, p, exner, rho, dz, zw,
      ust, flt, fltv, flq, flqv, pblh, ts, xland, psig_shcu)
