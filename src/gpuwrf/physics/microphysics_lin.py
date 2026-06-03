"""JAX Purdue-Lin single-moment 6-class microphysics (WRF mp_physics=2).

Faithful port of WRF ``phys/module_mp_lin.F`` (the ``lin_et_al`` wrapper +
``clphy1d`` 1-D cloud microphysics + ``satadj`` Newton saturation adjustment +
the ``parama1``/``parama2`` Bergeron tables + ``ggamma``).

WRF process order is preserved exactly per column:

  1. cap negatives; tem/temcc; qsw/qsi (Tetens liquid / Murray ice); theiz
  2. ADAPTIVE-SUBSTEP terminal-velocity sedimentation, in order:
        rain, snow, graupel, cloud-ice  (each a Courant-limited while-loop)
     accumulating surface precip pptrain/pptsnow/pptgraul/pptice
  3. per-cell microphysics process block (DO 2000):
        T<0C snow processes (psaut,psfw,psfi,praci,piacr,psaci,psacw,
                              psdep/pssub,pracs,psacr) or
        T>0C snow processes (psacw,psacr,psmlt,psmltevp);
        rain processes (praut,pracw,prevp);
        T<0C graupel (pgaut,pgfr,pgacw,pgaci,pgacr,pgacs,pgwet,pgdep/pgsub,...)
        or T>0C graupel (pgacw,pgacr,pgmlt,pgmltevp,pgacs)
  4. conservation feedback (depletion clamps) + state update (T<0C / T>0C)
  5. saturation adjustment (satadj Newton) when supersaturated
  6. ice/water melt/freeze (pihom,pimlt,pidw) + second satadj
  7. qv<qvmin padding (k>=2)

The scheme works on POTENTIAL TEMPERATURE ``thz`` internally (theiz is the
ice-liquid equivalent potential temperature carried through the process
updates). This port returns a ``PhysicsTendency`` with ``state_replacements``
for the updated theta + 6 moist species and ``accumulator_increments`` for the
surface precip (mm).

Validation: per-column WRF savepoint parity against the real Fortran scheme
(proofs/v060/savepoints_lin, run_lin_parity.py). The port defaults to fp64; the
WRF scheme runs in default single precision, so parity is to a predeclared
physical tolerance, never bitwise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.contracts.physics_interfaces import PhysicsTendency
from gpuwrf.physics import lin_constants as C


# --------------------------------------------------------------------------
# Saturation vapor pressure helpers (es in Pa) -- module_mp_lin.F inline forms
# --------------------------------------------------------------------------
def _es_liquid(tem):
    # es = 1000.*svp1*exp( svp2*temcc/(tem-svp3) ), temcc = tem-273.15
    temcc = tem - 273.15
    return 1000.0 * C.SVP1 * jnp.exp(C.SVP2 * temcc / (tem - C.SVP3))


def _es_ice(tem):
    # es = 1000.*svp1*exp( 21.8745584*(tem-273.16)/(tem-7.66) )
    return 1000.0 * C.SVP1 * jnp.exp(21.8745584 * (tem - 273.16) / (tem - 7.66))


def _qfromes(es, prez):
    return C.EP2 * es / (prez - es)


# --------------------------------------------------------------------------
# parama1 / parama2: 32-entry Bergeron table, linear interp (JAX-vectorized).
# WRF: i1=int(-temp)+1 (1-based), ratio=-temp-float(i1-1),
#      out = a(i1) + ratio*(a(i1p1)-a(i1))   [Fortran int() truncates toward 0]
# --------------------------------------------------------------------------
_PA1 = jnp.asarray(C.PARAMA1, dtype=jnp.float64)
_PA2 = jnp.asarray(C.PARAMA2, dtype=jnp.float64)


def _parama(temp, table):
    # temp is <= 0 on the calling paths => -temp >= 0; int() truncates toward 0.
    mtemp = -temp
    i0 = jnp.floor(mtemp).astype(jnp.int32)        # = int(-temp) for -temp>=0
    ratio = mtemp - i0.astype(temp.dtype)          # = -temp - float(i1-1)
    i0 = jnp.clip(i0, 0, _PA1.shape[0] - 2)         # i1-1 (0-based); guard top
    a_i = table[i0]
    a_ip1 = table[i0 + 1]
    return a_i + ratio * (a_ip1 - a_i)


# --------------------------------------------------------------------------
# Adaptive-substep terminal-velocity sedimentation (one species).
# Mirrors the WRF `notlast` while-loop EXACTLY: at each substep recompute the
# fall speeds, find [min_q,max_q] (the active layer span), pick a Courant dt
# (del_tv), and apply the downward flux sweep within that span. Surface flux
# accumulates into ppt; flux past min_q (when min_q>1) deposits into min_q-1.
#
# `vt_fn(q, k)` returns the terminal velocity for cell k given mixing ratio q
# (already > thresh). The threshold is 1.0e-8 (WRF qXz>1.0e-8).
# Returns (q_new, ppt, vtold) where vtold is the last-iteration fall speed
# (WRF reuses vtXold(k) for the precZ flux diagnostic; we keep it for fidelity
# although precZ is not part of the parity surface).
# --------------------------------------------------------------------------
def _sediment_species(q, rho, dzw, zz, zsfc, dtb, vt_of_q):
    km = q.shape[0]
    kidx = jnp.arange(km)
    thresh = 1.0e-8
    # cell-top thickness for the Courant limit: 0.9*(zz(k)-zz(k-1)), with
    # zz(0)-zsfc for k==1 (0-based k==0).
    zprev = jnp.concatenate([jnp.asarray([zsfc], q.dtype), zz[:-1]])
    dz_courant = zz - zprev  # (zz(k)-zz(k-1)) with zz(-1)=zsfc

    def vtold_all(qz):
        # fall speed for all cells, 0 where qz<=thresh; WRF only computes for
        # k=kts..kte-1 (k<km-1); top cell (k==km-1) is excluded from min/max.
        vt = jnp.where(qz > thresh, vt_of_q(jnp.maximum(qz, thresh)), 0.0)
        vt = jnp.where(kidx < km - 1, vt, 0.0)
        return vt

    def cond(state):
        qz, ppt, t_del_tv, notlast, vtlast = state
        return notlast

    def body(state):
        qz, ppt, t_del_tv, notlast, vtlast = state
        vt = vtold_all(qz)
        active = (qz > thresh) & (kidx < km - 1)
        any_active = jnp.any(active)
        # min_q / max_q over active cells
        big = km
        small = -1
        min_q = jnp.min(jnp.where(active, kidx, big))
        max_q = jnp.max(jnp.where(active, kidx, small))
        # del_tv = min over active of 0.9*dz_courant/vt, capped at dtb
        cour = jnp.where(active & (vt > 0.0),
                         0.9 * dz_courant / jnp.maximum(vt, 1e-30), dtb)
        del_tv0 = jnp.minimum(dtb, jnp.min(jnp.where(active, cour, dtb)))

        proceed = any_active & (max_q >= min_q)

        # t_del_tv bookkeeping (only when proceeding)
        t_new = t_del_tv + del_tv0
        over = t_new >= dtb
        del_tv = jnp.where(over, dtb + del_tv0 - t_new, del_tv0)
        notlast_next = jnp.where(proceed, jnp.logical_not(over), False)

        # downward flux sweep from max_q to min_q.
        # WRF: fluxin starts 0 at top (max_q); for k=max_q..min_q step -1:
        #   fluxout=rho*vt*qz; flux=(fluxin-fluxout)/rho/dzw; qz+=del_tv*flux;
        #   fluxin=fluxout. The carried fluxin is the flux ENTERING cell k from
        #   above (= fluxout of k+1). Implement as a top-down scan over all k,
        #   masking to [min_q,max_q].
        def scan_step(fluxin, k_from_top):
            k = max_q - k_from_top  # descend from max_q down to 0
            in_span = (k >= min_q) & (k <= max_q)
            fluxout = rho[k] * vt[k] * qz[k]
            flux = (fluxin - fluxout) / rho[k] / dzw[k]
            dq = jnp.where(in_span & proceed, del_tv * flux, 0.0)
            # fluxin for next (lower) cell becomes this cell's fluxout, but only
            # within span; outside span carry stays 0 (matches WRF reset each call)
            fluxin_next = jnp.where(in_span, fluxout, fluxin)
            return fluxin_next, (k, dq)

        # iterate k_from_top = 0..km-1 so k = max_q, max_q-1, ... ; outside span
        # contributes nothing. The fluxin that exits the bottom of min_q is the
        # final carry value (fluxout of min_q).
        fluxin_final, (ks, dqs) = jax.lax.scan(
            scan_step, jnp.asarray(0.0, q.dtype), jnp.arange(km))
        # apply dq scattered back to the right cells
        dq_by_k = jnp.zeros(km, q.dtype).at[ks].add(dqs)
        qz_upd = qz + dq_by_k
        qz_upd = jnp.maximum(qz_upd, 0.0)  # WRF amax1(0.,qsz) for s/g/i (rain: no clamp)
        # the carried fluxin out of min_q is fluxin_final (fluxout of min_q).
        fluxin_at_min = fluxin_final
        # surface accumulation vs deposit into min_q-1
        at_surface = (min_q == 0)
        ppt_inc = jnp.where(proceed & at_surface, fluxin_at_min * del_tv, 0.0)
        # deposit into min_q-1 when min_q>0
        idx_below = jnp.maximum(min_q - 1, 0)
        deposit = jnp.where(
            proceed & jnp.logical_not(at_surface),
            del_tv * fluxin_at_min / rho[idx_below] / dzw[idx_below], 0.0)
        qz_upd = qz_upd.at[idx_below].add(deposit)

        ppt_next = ppt + ppt_inc
        qz_out = jnp.where(proceed, qz_upd, qz)
        return (qz_out, ppt_next, jnp.where(proceed, t_new, t_del_tv),
                notlast_next, vt)

    init = (q, jnp.asarray(0.0, q.dtype), jnp.asarray(0.0, q.dtype),
            jnp.asarray(True), jnp.zeros(km, q.dtype))
    qz, ppt, _, _, vtlast = jax.lax.while_loop(cond, body, init)
    return qz, ppt


# --------------------------------------------------------------------------
# satadj (Newton iteration, 20 steps fixed). One cell. Returns (qvz,qlz,qiz).
# Mirrors module_mp_lin.F SUBROUTINE satadj exactly.
# --------------------------------------------------------------------------
def _satadj_cell(qvz, qlz, qiz, prez, theiz, tothz):
    xlvocp = C.XLVOCP
    xlfocp = C.XLFOCP

    tem = tothz * (theiz - (xlvocp * qvz - xlfocp * qiz) / tothz)
    tem_noliqice = tem - xlvocp * qlz - (xlvocp + xlfocp) * qiz

    es_liq = _es_liquid(tem_noliqice)
    qsat_nl_liq = _qfromes(es_liq, prez)
    qsat_nl_ice = (C.EPISP0K / prez
                   * jnp.exp(21.8745584 * (tem_noliqice - 273.15) / (tem_noliqice - 7.66)))
    qsat_noliqice = jnp.where(tem_noliqice > 273.15, qsat_nl_liq, qsat_nl_ice)

    qpz = qvz + qlz + qiz

    # branch A: subsaturated (qpz < qsat_noliqice) -> all to vapor
    # branch B: do the Newton solve
    qlpqi = qlz + qiz
    big = qlpqi >= 1.0e-5
    ratql_big = qlz / jnp.where(qlpqi > 0.0, qlpqi, 1.0)
    ratqi_big = qiz / jnp.where(qlpqi > 0.0, qlpqi, 1.0)
    t0 = 273.15
    t1 = 248.15
    tmp1 = jnp.clip((t0 - tem) / (t0 - t1), 0.0, 1.0)
    ratqi_small = tmp1
    ratql_small = 1.0 - tmp1
    ratql = jnp.where(big, ratql_big, ratql_small)
    ratqi = jnp.where(big, ratqi_big, ratqi_small)

    def newton_body(n, carry):
        tsat, absft, qvsbar = carry
        denom1 = 1.0 / (tsat - C.SVP3)
        denom2 = 1.0 / (tsat - 7.66)
        es1 = 1000.0 * C.SVP1 * jnp.exp(C.SVP2 * denom1 * (tsat - C.SVPT0))
        qswz = _qfromes(es1, prez)
        es2 = 1000.0 * C.SVP1 * jnp.exp(21.8745584 * denom2 * (tsat - 273.15))
        qsiz_cold = _qfromes(es2, prez)
        # if tem<273.15: qsiz=qsiz_cold; if tem<233.15: qswz=qsiz_cold
        qsiz = jnp.where(tem < 273.15, qsiz_cold, qswz)
        qswz = jnp.where(tem < 233.15, qsiz_cold, qswz)
        qvsbar_new = ratql * qswz + ratqi * qsiz
        # WRF: if(absft<0.01) skip the update for the rest (go to 300). We emulate
        # by freezing tsat/qvsbar once converged.
        converged = absft < 0.01
        dqvsbar = (ratql * qswz * C.SVP2 * 243.5 * denom1 * denom1
                   + ratqi * qsiz * 21.8745584 * 265.5 * denom2 * denom2)
        ftsat = (tsat + (xlvocp + ratqi * xlfocp) * qvsbar_new
                 - tothz * theiz - xlfocp * ratqi * (qvz + qlz + qiz))
        dftsat = 1.0 + (xlvocp + ratqi * xlfocp) * dqvsbar
        tsat_new = tsat - ftsat / dftsat
        absft_new = jnp.abs(ftsat)
        tsat_out = jnp.where(converged, tsat, tsat_new)
        qvsbar_out = jnp.where(converged, qvsbar, qvsbar_new)
        absft_out = jnp.where(converged, absft, absft_new)
        return (tsat_out, absft_out, qvsbar_out)

    tsat0 = tem
    qvsbar0 = jnp.asarray(0.0, qvz.dtype)
    tsat_f, absft_f, qvsbar_f = jax.lax.fori_loop(
        0, 20, newton_body, (tsat0, jnp.asarray(1.0, qvz.dtype), qvsbar0))

    # 300 continue: if qpz>qvsbar -> distribute; else all vapor
    sat = qpz > qvsbar_f
    qvz_B = jnp.where(sat, qvsbar_f, qpz)
    qiz_B = jnp.where(sat, ratqi * (qpz - qvsbar_f), 0.0)
    qlz_B = jnp.where(sat, ratql * (qpz - qvsbar_f), 0.0)

    sub = qpz < qsat_noliqice  # branch A condition
    qvz_out = jnp.where(sub, qpz, qvz_B)
    qiz_out = jnp.where(sub, 0.0, qiz_B)
    qlz_out = jnp.where(sub, 0.0, qlz_B)
    return qvz_out, qlz_out, qiz_out


# vectorize satadj over the column
_satadj_col = jax.vmap(_satadj_cell, in_axes=(0, 0, 0, 0, 0, 0))


# --------------------------------------------------------------------------
# Core single-column Lin scheme (clphy1d for one column).
# --------------------------------------------------------------------------
def _lin_column(thz, qvz, qlz, qrz, qiz, qsz, qgz, rho, pii, prez, zz, dzw, zsfc, dtb):
    km = thz.shape[0]
    kidx = jnp.arange(km)
    orho = 1.0 / rho
    sqrho = jnp.sqrt(1.29 * orho)   # sqrt(rhoe_s/rho), rhoe_s=1.29
    tothz = pii
    oprez = 1.0 / prez
    odtb = 1.0 / dtb
    gindex = 1.0  # graupel active

    # cap negatives (WRF amax1)
    qlz = jnp.maximum(qlz, 0.0)
    qiz = jnp.maximum(qiz, 0.0)
    qvz = jnp.maximum(qvz, C.QVMIN)
    qsz = jnp.maximum(qsz, 0.0)
    qrz = jnp.maximum(qrz, 0.0)
    qgz = jnp.maximum(qgz, 0.0)

    tem = thz * tothz
    temcc = tem - 273.15
    es_l = _es_liquid(tem)
    qswz = _qfromes(es_l, prez)
    es_i = _es_ice(tem)
    qsiz_cold = _qfromes(es_i, prez)
    cold0 = tem < 273.15
    qsiz = jnp.where(cold0, qsiz_cold, qswz)
    qswz = jnp.where(cold0 & (temcc < -40.0), qsiz_cold, qswz)

    theiz = thz + (C.XLVOCP * qvz - C.XLFOCP * qiz) / tothz

    # ---- adaptive-substep sedimentation (rain, snow, graupel, ice) ----
    def vt_rain(q):
        tmp1 = jnp.sqrt(C.PI * C.RHOWATER * C.XNOR / rho / q)
        tmp1 = jnp.sqrt(tmp1)  # = lambdar^... matching WRF sqrt(sqrt(...))
        return C.O6 * C.CONSTA * C.GAMBP4 * sqrho / tmp1 ** C.CONSTB

    def vt_snow(q):
        tmp1 = jnp.sqrt(C.PI * C.RHOSNOW * C.XNOS / rho / q)
        tmp1 = jnp.sqrt(tmp1)
        return C.O6 * C.CONSTC * C.GAMDP4 * sqrho / tmp1 ** C.CONSTD

    def vt_graupel(q):
        tmp1 = jnp.sqrt(C.PI * C.RHOGRAUL * C.XNOG / rho / q)
        tmp1 = jnp.sqrt(tmp1)
        term0 = jnp.sqrt(4.0 * C.G * C.RHOGRAUL * 0.33334 / rho / C.CDRAG)
        return C.O6 * C.GAM4PT5 * term0 * jnp.sqrt(1.0 / tmp1)

    def vt_ice(q):
        return 3.29 * (rho * q) ** 0.16  # Heymsfield and Donner

    qrz, pptrain = _sediment_species(qrz, rho, dzw, zz, zsfc, dtb, vt_rain)
    qsz, pptsnow = _sediment_species(qsz, rho, dzw, zz, zsfc, dtb, vt_snow)
    qgz, pptgraul = _sediment_species(qgz, rho, dzw, zz, zsfc, dtb, vt_graupel)
    qiz, pptice = _sediment_species(qiz, rho, dzw, zz, zsfc, dtb, vt_ice)

    # ---- per-cell microphysics process block (DO 2000) ----
    qvz, qlz, qrz, qiz, qsz, qgz, theiz = _process_block(
        thz, qvz, qlz, qrz, qiz, qsz, qgz, rho, orho, sqrho, tothz, prez,
        oprez, tem, temcc, qswz, qsiz, theiz, dtb, odtb, gindex)

    # recompute thz from theiz at end (WRF updates thz inside; we recompute final)
    thz_out = theiz - (C.XLVOCP * qvz - C.XLFOCP * qiz) / tothz

    return {
        "th": thz_out, "qv": qvz, "qc": qlz, "qr": qrz,
        "qi": qiz, "qs": qsz, "qg": qgz,
        # surface precip: pptrain/snow/graul/ice are in METERS (flux*dt/rho_w
        # implicitly via flux units kg/m2/s * s / rho... see below conversion).
        "pptrain": pptrain, "pptsnow": pptsnow,
        "pptgraul": pptgraul, "pptice": pptice,
    }


# --------------------------------------------------------------------------
# The microphysics process block: per-cell, fully vectorized over k. Reproduces
# the DO 2000 loop's snow/rain/graupel process diagnostics, conservation
# clamps, state update, satadj, and melt/freeze with masks for the `go to`s.
# --------------------------------------------------------------------------
def _process_block(thz, qvz, qlz, qrz, qiz, qsz, qgz, rho, orho, sqrho, tothz,
                   prez, oprez, tem, temcc, qswz, qsiz, theiz, dtb, odtb, gindex):
    km = thz.shape[0]
    pi = C.PI
    pio4 = C.PIO4
    pio6 = C.PIO6

    qvoqswz = qvz / qswz
    qvoqsiz = qvz / qsiz

    qvzodt = jnp.maximum(0.0, odtb * qvz)
    qlzodt = jnp.maximum(0.0, odtb * qlz)
    qizodt = jnp.maximum(0.0, odtb * qiz)
    qszodt = jnp.maximum(0.0, odtb * qsz)
    qrzodt = jnp.maximum(0.0, odtb * qrz)
    qgzodt = jnp.maximum(0.0, odtb * qgz)

    rs0 = C.EP2 * 1000.0 * C.SVP1 / (prez - 1000.0 * C.SVP1)

    # skip flag: unsaturated AND no condensate (WRF go to 2000)
    tmp_cond = qiz + qlz + qsz + qrz + qgz * gindex
    skip = (qvz + qlz + qiz < qsiz) & (tmp_cond == 0.0)
    active = jnp.logical_not(skip)

    # ---- slopes / terminal velocities (process-block re-eval) ----
    thresh = 1.0e-8
    have_r = qrz > thresh
    have_s = qsz > thresh
    have_g = qgz > thresh
    lamr = jnp.sqrt(jnp.sqrt(pi * C.RHOWATER * C.XNOR * orho / jnp.maximum(qrz, thresh)))
    olambdar = jnp.where(have_r, 1.0 / lamr, 0.0)
    vtr = jnp.where(have_r, C.O6 * C.CONSTA * C.GAMBP4 * sqrho * olambdar ** C.CONSTB, 0.0)
    lams = jnp.sqrt(jnp.sqrt(pi * C.RHOSNOW * C.XNOS * orho / jnp.maximum(qsz, thresh)))
    olambdas = jnp.where(have_s, 1.0 / lams, 0.0)
    vts = jnp.where(have_s, C.O6 * C.CONSTC * C.GAMDP4 * sqrho * olambdas ** C.CONSTD, 0.0)
    lamg = jnp.sqrt(jnp.sqrt(pi * C.RHOGRAUL * C.XNOG * orho / jnp.maximum(qgz, thresh)))
    olambdag = jnp.where(have_g, 1.0 / lamg, 0.0)
    term0g = jnp.sqrt(4.0 * C.G * C.RHOGRAUL * 0.33334 * orho * C.OCDRAG)
    vtg = jnp.where(have_g, C.O6 * C.GAM4PT5 * term0g * jnp.sqrt(olambdag), 0.0)

    # ---- viscosity / diffusivity / conductivity ----
    viscmu = C.AVISC * tem ** 1.5 / (tem + 120.0)
    visc = viscmu * orho
    diffwv = C.ADIFFWV * tem ** 1.81 * oprez
    schmidt = visc / diffwv
    xka = C.AXKA * viscmu

    cold = tem < 273.15

    # ============== SNOW PROCESSES ==============
    # --- T<0C branch ---
    # (1) psaut
    alpha1 = 1.0e-3 * jnp.exp(0.025 * temcc)
    tmp_qic = -7.6 + 4.0 * jnp.exp(-0.2443e-3 * (jnp.abs(temcc) - 20.0) ** 2.455)
    qic = jnp.where(temcc < -20.0, 1.0e-3 * jnp.exp(tmp_qic) * orho, C.QI0)
    psaut = jnp.maximum(0.0, odtb * (qiz - qic) * (1.0 - jnp.exp(-alpha1 * dtb)))
    # (2) psfw + (3) psfi (Bergeron), only qlz>1e-10
    temc1 = jnp.maximum(-30.99, temcc)
    a1 = _parama(temc1, _PA1)
    a2 = _parama(temc1, _PA2)
    tmp1b = 1.0 - a2
    a1m = a1 * 0.001 ** tmp1b
    odtberg = (a1m * tmp1b) / (C.XMI50 ** tmp1b - C.XMI40 ** tmp1b)
    vti50 = C.CONSTC * C.DI50 ** C.CONSTD * sqrho
    eiw = 1.0
    save1_b = a1m * C.XMI50 ** a2
    save2_b = 0.25 * pi * eiw * rho * C.DI50 * C.DI50 * vti50
    tmp2_b = save1_b + save2_b * qlz
    xni50mx = jnp.where(tmp2_b > 0.0, qlzodt / tmp2_b, 0.0)
    xni50_a = qiz * (1.0 - jnp.exp(-dtb * odtberg)) / C.XMI50
    xni50 = jnp.minimum(xni50_a, xni50mx)
    tmp3_b = odtb * tmp2_b / save2_b * (1.0 - jnp.exp(-save2_b * xni50 * dtb))
    psfw_raw = jnp.minimum(tmp3_b, qlzodt)
    psfi_raw = jnp.minimum(xni50 * C.XMI50 - psfw_raw, qizodt)
    has_ql = qlz > 1.0e-10
    psfw = jnp.where(has_ql, psfw_raw, 0.0)
    psfi = jnp.where(has_ql, psfi_raw, 0.0)
    # (4) praci + (5) piacr (qrz>0)
    eri = 1.0
    save1_r = pio4 * eri * C.XNOR * C.CONSTA * sqrho
    tmp1_r = save1_r * C.GAMBP3 * olambdar ** C.BP3
    praci_raw = qizodt * (1.0 - jnp.exp(-tmp1_r * dtb))
    tmp2_r = qiz * save1_r * rho * pio6 * C.RHOWATER * C.GAMBP6 * C.OXMI * olambdar ** C.BP6
    piacr_raw = jnp.minimum(tmp2_r, qrzodt)
    have_qr = qrz > 0.0
    praci_c = jnp.where(have_qr, praci_raw, 0.0)
    piacr_c = jnp.where(have_qr, piacr_raw, 0.0)
    # (6) psaci + (7) psacw + (8) psdep/pssub (qsz>0)
    esi = jnp.exp(0.025 * temcc)
    save1_s = pio4 * C.XNOS * C.CONSTC * C.GAMDP3 * sqrho * olambdas ** C.DP3
    psaci_raw = qizodt * (1.0 - jnp.exp(-esi * save1_s * dtb))
    esw = 1.0
    psacw_raw_cold = qlzodt * (1.0 - jnp.exp(-esw * save1_s * dtb))
    tmpa_s = C.RVAPOR * xka * tem * tem
    tmpb_s = C.XLS * C.XLS * rho * qsiz * diffwv
    tmpc_s = tmpa_s * qsiz * diffwv
    abi = 2.0 * pi * (qvoqsiz - 1.0) * tmpc_s / (tmpa_s + tmpb_s)
    tmp1_sd = C.CONSTC * sqrho * olambdas ** C.DP5 / visc
    tmp2_sd = abi * C.XNOS * (C.VF1S * olambdas * olambdas
                              + C.VF2S * schmidt ** 0.33334 * C.GAMDP5O2 * jnp.sqrt(tmp1_sd))
    tmp3_sd = odtb * (qvz - qsiz)
    sub_branch = tmp3_sd <= 0.0
    tmp2_clamped = jnp.maximum(tmp2_sd, tmp3_sd)
    pssub_raw = jnp.minimum(0.0, jnp.maximum(tmp2_clamped, -qszodt))
    psdep_raw = jnp.minimum(tmp2_sd, tmp3_sd)
    have_qs = qsz > 0.0
    psaci_cold = jnp.where(have_qs, psaci_raw, 0.0)
    psacw_cold = jnp.where(have_qs, psacw_raw_cold, 0.0)
    psdep = jnp.where(have_qs & jnp.logical_not(sub_branch), psdep_raw, 0.0)
    pssub = jnp.where(have_qs & sub_branch, pssub_raw, 0.0)
    # (9) pracs + (10) psacr (qsz>0 and qrz>0)
    tmpa = olambdar * olambdar
    tmpb = olambdas * olambdas
    tmpc = olambdar * olambdas
    esr = 1.0
    tmp1_pr = pi * pi * esr * C.XNOR * C.XNOS * jnp.abs(vtr - vts) * orho
    tmp2_pr = tmpb * tmpb * olambdar * (5.0 * tmpb + 2.0 * tmpc + 0.5 * tmpa)
    pracs_raw = jnp.minimum(tmp1_pr * C.RHOSNOW * tmp2_pr, qszodt)
    tmp3_pr = tmpa * tmpa * olambdas * (5.0 * tmpa + 2.0 * tmpc + 0.5 * tmpb)
    psacr_cold_raw = jnp.minimum(tmp1_pr * C.RHOWATER * tmp3_pr, qrzodt)
    qsqr = have_qs & have_qr
    pracs_cold = jnp.where(qsqr, pracs_raw, 0.0)
    psacr_cold = jnp.where(qsqr, psacr_cold_raw, 0.0)

    # --- T>0C branch ---
    # (1) psacw
    tmp1_sw = esw * pio4 * C.XNOS * C.CONSTC * C.GAMDP3 * sqrho * olambdas ** C.DP3
    psacw_warm = jnp.where(have_qs, qlzodt * (1.0 - jnp.exp(-tmp1_sw * dtb)), 0.0)
    # (2) psacr
    tmp2_swr = tmpa * tmpa * olambdas * (5.0 * tmpa + 2.0 * tmpc + 0.5 * tmpb)
    psacr_warm = jnp.where(have_qs, jnp.minimum(tmp1_pr * C.RHOWATER * tmp2_swr, qrzodt), 0.0)
    # (3) psmlt
    delrs = rs0 - qvz
    term1_sm = 2.0 * pi * orho * (C.XLV * diffwv * rho * delrs - xka * temcc)
    tmp2_sm = C.XNOS * (C.VF1S * olambdas * olambdas
                        + C.VF2S * schmidt ** 0.33334 * C.GAMDP5O2 * jnp.sqrt(tmp1_sd))
    tmp3_sm = term1_sm * C.OXLF * tmp2_sm - C.CWOXLF * temcc * (psacw_warm + psacr_warm)
    psmlt_warm = jnp.where(have_qs, jnp.maximum(jnp.minimum(0.0, tmp3_sm), -qszodt), 0.0)
    # (4) psmltevp
    tmpa_se = C.RVAPOR * xka * tem * tem
    tmpb_se = C.XLV * C.XLV * rho * qswz * diffwv
    tmpc_se = tmpa_se * qswz * diffwv
    tmpd_se = jnp.minimum(0.0, (qvoqswz - 0.90) * qswz * odtb)
    abr_se = 2.0 * pi * (qvoqswz - 0.90) * tmpc_se / (tmpa_se + tmpb_se)
    tmp2_sev = abr_se * C.XNOS * (C.VF1S * olambdas * olambdas
                                  + C.VF2S * schmidt ** 0.33334 * C.GAMDP5O2 * jnp.sqrt(tmp1_sd))
    tmp3_sev = jnp.maximum(jnp.minimum(0.0, tmp2_sev), tmpd_se)
    psmltevp_warm = jnp.where(have_qs, jnp.maximum(tmp3_sev, -qszodt), 0.0)

    # select snow-branch process values by cold/warm
    psaut = jnp.where(cold, psaut, 0.0)
    psfw = jnp.where(cold, psfw, 0.0)
    psfi = jnp.where(cold, psfi, 0.0)
    praci = jnp.where(cold, praci_c, 0.0)
    piacr = jnp.where(cold, piacr_c, 0.0)
    psaci = jnp.where(cold, psaci_cold, 0.0)
    psacw = jnp.where(cold, psacw_cold, psacw_warm)
    psdep = jnp.where(cold, psdep, 0.0)
    pssub = jnp.where(cold, pssub, 0.0)
    pracs = jnp.where(cold, pracs_cold, 0.0)
    psacr = jnp.where(cold, psacr_cold, psacr_warm)
    psmlt = jnp.where(cold, 0.0, psmlt_warm)
    psmltevp = jnp.where(cold, 0.0, psmltevp_warm)

    # ============== RAIN PROCESSES (both branches) ==============
    araut = 0.001
    praut = jnp.maximum(0.0, odtb * (qlz - C.QL0) * (1.0 - jnp.exp(-araut * dtb)))
    erw = 1.0
    tmp1_rw = pio4 * erw * C.XNOR * C.CONSTA * sqrho * C.GAMBP3 * olambdar ** C.BP3
    pracw = qlzodt * (1.0 - jnp.exp(-tmp1_rw * dtb))
    # prevp
    tmpa_re = C.RVAPOR * xka * tem * tem
    tmpb_re = C.XLV * C.XLV * rho * qswz * diffwv
    tmpc_re = tmpa_re * qswz * diffwv
    dqsdt = qswz * C.XLV / (C.RVAPOR * tem ** 2)
    tmpd_re = jnp.minimum(0.0, 0.9 * odtb * (qvz + qlz - qswz) / (1.0 + C.XLVOCP * dqsdt))
    abr_re = 2.0 * pi * (qvoqswz - 1.0) * tmpc_re / (tmpa_re + tmpb_re)
    vf1r = 0.78
    vf2r = 0.31
    tmp1_re = C.CONSTA * sqrho * olambdar ** C.BP5 / visc
    tmp2_re = abr_re * C.XNOR * (vf1r * olambdar * olambdar
                                 + vf2r * schmidt ** 0.33334 * C.GAMBP5O2 * jnp.sqrt(tmp1_re))
    tmp3_re = jnp.maximum(jnp.minimum(0.0, tmp2_re), tmpd_re)
    prevp = jnp.maximum(tmp3_re, -qrzodt)

    # ============== GRAUPEL PROCESSES ==============
    pgaut, pgfr, pgacw, pgaci, pgacr, pgacs, pgacip, pgacrp, pgacsp, pgwet, \
        pgsub, pgdep, pgmlt, pgmltevp, delta4 = _graupel_block(
            cold, qlz, qiz, qrz, qsz, qgz, rho, orho, sqrho, visc, schmidt, xka,
            diffwv, tem, temcc, qvz, qvoqsiz, qvoqswz, qsiz, qswz, rs0,
            olambdar, olambdas, olambdag, vtr, vts, vtg,
            qlzodt, qizodt, qrzodt, qszodt, qgzodt, odtb, dtb)

    # ============== CONSERVATION FEEDBACK + STATE UPDATE ==============
    new = _conservation_and_update(
        cold, gindex, dtb, odtb, tothz, prez, orho, qvz, qlz, qiz, qrz, qsz, qgz, theiz,
        qvzodt, qlzodt, qizodt, qrzodt, qszodt, qgzodt,
        psaut, psfw, psfi, praci, piacr, psaci, psacw, psdep, pssub, pracs, psacr,
        psmlt, psmltevp, praut, pracw, prevp,
        pgaut, pgfr, pgacw, pgaci, pgacr, pgacs, pgacip, pgacrp, pgacsp, pgwet,
        pgsub, pgdep, pgmlt, pgmltevp, delta4)

    qvz_n, qlz_n, qiz_n, qrz_n, qsz_n, qgz_n, theiz_n, pclw_n, pvapor_n = new

    # apply only where active; skipped cells keep their (post-sediment) values.
    qvz = jnp.where(active, qvz_n, qvz)
    qlz = jnp.where(active, qlz_n, qlz)
    qiz = jnp.where(active, qiz_n, qiz)
    qrz = jnp.where(active, qrz_n, qrz)
    qsz = jnp.where(active, qsz_n, qsz)
    qgz = jnp.where(active, qgz_n, qgz)
    theiz = jnp.where(active, theiz_n, theiz)

    # ---- qv<qvmin padding (WRF do k=kts+1,kte) ----
    kidx = jnp.arange(km)
    below_min = (qvz < C.QVMIN) & (kidx >= 1)
    qlz = jnp.where(below_min, 0.0, qlz)
    qiz = jnp.where(below_min, 0.0, qiz)
    qvz = jnp.where(below_min, jnp.maximum(C.QVMIN, qvz + qlz + qiz), qvz)

    return qvz, qlz, qrz, qiz, qsz, qgz, theiz


# placeholder modules filled below
from gpuwrf.physics._lin_graupel import _graupel_block  # noqa: E402
from gpuwrf.physics._lin_update import _conservation_and_update  # noqa: E402


# --------------------------------------------------------------------------
# vmap + jit the single-column scheme across columns.
# --------------------------------------------------------------------------
_lin_columns = jax.jit(
    jax.vmap(_lin_column,
             in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None)),
    static_argnums=(13,))


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def lin_run(th, qv, qc, qr, qi, qs, qg, rho, pii, p, z, dz8w, delt):
    """Run Purdue-Lin on a batch of columns (shape (ncol, nlev)).

    Inputs are POTENTIAL-TEMPERATURE-based (th in K), matching the WRF
    ``lin_et_al`` interface. ``z`` is geometric height of each level (m), used
    by the Courant-limited sedimentation; ``dz8w`` is the layer thickness (m).
    Returns a dict with updated th, qv, qc, qr, qi, qs, qg and surface precip
    increments (mm) named rainncv/snowncv/graupelncv/sr.
    """
    zsfc = jnp.zeros(th.shape[0], dtype=th.dtype)  # ht=0 in the oracle
    out = _lin_columns(th, qv, qc, qr, qi, qs, qg, rho, pii, p, z, dz8w, zsfc,
                       float(delt))
    # surface precip: WRF lin_et_al sets pptX = flux*del_tv summed; flux units
    # are kg/m2/s (rho*vt*q) so pptX is kg/m2 = mm of water-equivalent already
    # divided by rho_water? No: WRF then assigns RAINNCV = pptrain+...; the
    # ppt accumulation `fluxin*del_tv` has units (kg m^-2 s^-1)*s = kg m^-2 = mm.
    # (lin_et_al comment: "unit is transferred from m to mm" -- but the flux
    # form already yields mm; the scheme does NOT multiply by 1000 again here.)
    rainncv = out["pptrain"] + out["pptsnow"] + out["pptgraul"] + out["pptice"]
    snowncv = out["pptsnow"] + out["pptice"]
    graupelncv = out["pptgraul"]
    sr = (out["pptice"] + out["pptsnow"] + out["pptgraul"]) / (
        out["pptice"] + out["pptsnow"] + out["pptgraul"] + out["pptrain"] + 1.0e-12)
    return {
        "th": out["th"], "qv": out["qv"], "qc": out["qc"], "qr": out["qr"],
        "qi": out["qi"], "qs": out["qs"], "qg": out["qg"],
        "rainncv": rainncv, "snowncv": snowncv, "graupelncv": graupelncv,
        "sr": sr,
    }


def lin_physics_tendency(theta, qv, qc, qr, qi, qs, qg, pii, rho, p, z, dz, delt):
    """Lin adapter returning a frozen PhysicsTendency (in-place replacements)."""
    out = lin_run(theta, qv, qc, qr, qi, qs, qg, rho, pii, p, z, dz, delt)
    return PhysicsTendency(
        state_replacements={
            "theta": out["th"], "qv": out["qv"], "qc": out["qc"],
            "qr": out["qr"], "qi": out["qi"], "qs": out["qs"], "qg": out["qg"],
        },
        accumulator_increments={
            "rain_acc": out["rainncv"],
            "snow_acc": out["snowncv"],
            "graupel_acc": out["graupelncv"],
        },
        diagnostics={"sr": out["sr"]},
    )


__all__ = ["lin_run", "lin_physics_tendency"]
