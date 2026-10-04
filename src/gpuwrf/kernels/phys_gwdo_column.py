"""WRF REAL GWDO (``gwd_opt=1``) in one column-resident Pallas program per lane tile.

Spec: pristine ``phys/physics_mmm/bl_gwdo.F90`` (``bl_gwdo_run``; WRF builds it
with ``kind_phys = selected_real_kind(6)``, i.e. REAL). Each lane owns one
column; the mountain-top search, low-level averages, flow-blocking descent and
the saturation recursion run as per-lane loops in registers, in the Fortran
operation order. Layout is the State's ``(K, B)`` (level-major, columns
contiguous), so every level load is coalesced across lanes.

The kernel returns the UNSCALED ``taud`` profile plus the per-column ``dtfac``
limiter and ``xn``/``yn`` direction factors. WRF's final
``taud*dtfac -> dtaux/dtauy -> rotate`` is elementwise and is applied by
:func:`gwdo_epilogue` (it fuses into the consumer), so the kernel needs no
second pass over the profile once ``dtfac`` (a min over ``k<=kbl``) is known.
Flag: ``GPUWRF_GWDO_NATIVE_REAL=1`` (default off) routes
``coupling.physics_couplers.gwdo_adapter`` here.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

TX = 128
_NWDIR = (6, 7, 5, 8, 2, 3, 1, 4)  # WRF data nwdir


def native_enabled() -> bool:
    return os.environ.get("GPUWRF_GWDO_NATIVE_REAL", "0") == "1"


def _kernel(u_ref, v_ref, t_ref, q_ref, pl_ref, pk_ref, z_ref, pi_ref,
            var_ref, oc_ref, oa1_ref, oa2_ref, oa3_ref, oa4_ref,
            ol1_ref, ol2_ref, ol3_ref, ol4_ref, sina_ref, cosa_ref, dx_ref, dt_ref,
            taud_ref, dtfac_ref, xn_ref, yn_ref, *, nk, ncol):
    f32, i32 = jnp.float32, jnp.int32
    c = lambda value: jnp.asarray(value, f32)
    lane = pl.program_id(0) * TX + jnp.arange(TX, dtype=i32)
    # Tail lanes alias the last column and store identical values (no masks).
    col = jnp.minimum(lane, ncol - 1)
    zeros = jnp.zeros((TX,), f32)
    izeros = jnp.zeros((TX,), i32)
    false = jnp.zeros((TX,), jnp.bool_)

    def at(k):
        return jnp.broadcast_to(jnp.asarray(k, i32), (TX,))

    ld1 = lambda ref: plt.load(ref.at[col])
    ld = lambda ref, k: plt.load(ref.at[at(k), col])

    def tile_any(mask):
        return jnp.max(mask.astype(i32)) > 0

    # --- WRF constants as the driver passes them (module_model_constants) ---
    g_, rd_, rv_ = c(9.81), c(287.0), c(461.6)
    cp_ = c(7.0) * rd_ / c(2.0)
    fv_ = rv_ / rd_ - c(1.0)
    pi_ = c(3.141592653)
    ric, dw2min, rimin, bnv2min = c(0.25), c(1.0), c(-100.0), c(1.0e-5)
    efmin, efmax, gmax, veleps, frc, ce, cg = (c(0.0), c(10.0), c(1.0), c(1.0),
                                               c(1.0), c(0.8), c(0.5))
    frmax, olmin, odmin, odmax = c(10.0), c(1.0e-5), c(0.1), c(10.0)
    one, half, two = c(1.0), c(0.5), c(2.0)
    rrd = one / rd_
    gocp = g_ / cp_
    deltim = plt.load(dt_ref.at[at(0)])

    var, oc1 = ld1(var_ref), ld1(oc_ref)
    sina, cosa, dxm = ld1(sina_ref), ld1(cosa_ref), ld1(dx_ref)

    def level(k):
        """Per-level phy quantities (WRF:254-273), rotated to earth-relative."""
        up, vp = ld(u_ref, k), ld(v_ref, k)
        t1, q1, prsl, prslk = ld(t_ref, k), ld(q_ref, k), ld(pl_ref, k), ld(pk_ref, k)
        vtj = t1 * (one + fv_ * q1)
        vtk = vtj / prslk
        rho = rrd * prsl / vtj
        u1 = up * cosa - vp * sina
        v1 = up * sina + vp * cosa
        return dict(u1=u1, v1=v1, t1=t1, vtj=vtj, vtk=vtk, rho=rho, prsl=prsl,
                    zl=ld(z_ref, k))

    def interface(lo, hi):
        """usqj/bnv2 of the k..k+1 interface (WRF:366-378)."""
        ti = two / (lo['t1'] + hi['t1'])
        rdz = one / (hi['zl'] - lo['zl'])
        tem1 = lo['u1'] - hi['u1']
        tem2 = lo['v1'] - hi['v1']
        dw2 = tem1 * tem1 + tem2 * tem2
        shr2 = jnp.maximum(dw2, dw2min) * rdz * rdz
        bvf2 = g_ * (gocp + rdz * (hi['vtj'] - lo['vtj'])) * ti
        usqj = jnp.maximum(bvf2 / shr2, rimin)
        bnv2 = two * g_ * rdz * (hi['vtk'] - lo['vtk']) / (hi['vtk'] + lo['vtk'])
        return usqj, bnv2

    # --- mountain-top index kbl (WRF:276-302); 1-based like the Fortran ---
    zl1 = ld(z_ref, 0)
    zlowtop = two * var

    def top_cond(s):
        k, found = s[0], s[1]
        return (k < nk) & tile_any(~found & (zlowtop > c(0.0)))

    def top_body(s):
        k, found, klowtop = s
        hit = ~found & (zlowtop > c(0.0)) & (ld(z_ref, k) - zl1 >= zlowtop)
        return k + 1, found | hit, jnp.where(hit, k + 2, klowtop)

    _, _, klowtop = lax.while_loop(top_cond, top_body, (jnp.asarray(1, i32), false, izeros))
    kbl = jnp.maximum(jnp.minimum(klowtop, nk), 2)  # max(min(kbl,kpblmax),kpblmin)
    kb0 = kbl - 1                                   # 0-based mountain-top level
    prsi1 = ld(pi_ref, 0)
    delks = one / (prsi1 - plt.load(pi_ref.at[kb0, col]))
    delks1 = one / (ld(pl_ref, 0) - plt.load(pl_ref.at[kb0, col]))
    kmax = jnp.max(kb0)                             # tile bound for k < kbl loops

    # --- low-level means + weighted bnv2/usqj (WRF:305-322, 417-431) ---
    def low_body(k, s):
        ubar, vbar, rhobar, bnv2_1, usqj_1 = s
        lo, hi = level(k), level(k + 1)
        below = k < kb0                               # Fortran k < kbl
        rcsks = (ld(pi_ref, k) - ld(pi_ref, k + 1)) * delks
        usqj, bnv2 = interface(lo, hi)
        rdelks = (lo['prsl'] - hi['prsl']) * delks1   # == wtkbj at k = 1
        return (jnp.where(below, ubar + rcsks * lo['u1'], ubar),
                jnp.where(below, vbar + rcsks * lo['v1'], vbar),
                jnp.where(below, rhobar + rcsks * lo['rho'], rhobar),
                jnp.where(below, bnv2_1 + bnv2 * rdelks, bnv2_1),
                jnp.where(below, usqj_1 + usqj * rdelks, usqj_1))

    ubar, vbar, rhobar, bnv2_1, usqj_1 = lax.fori_loop(
        0, kmax, low_body, (zeros, zeros, zeros, zeros, zeros))

    # --- low-level wind direction -> oa/ol/olp/od/dxy (WRF:329-362) ---
    oa4 = [ld1(r) for r in (oa1_ref, oa2_ref, oa3_ref, oa4_ref)]
    ol4 = [ld1(r) for r in (ol1_ref, ol2_ref, ol3_ref, ol4_ref)]
    fdir = c(8.0) / (two * pi_)
    wdir = jnp.arctan2(ubar, vbar) + pi_
    x = fdir * wdir                                  # >= 0: nint == floor(x)+(frac>=.5)
    xf = jnp.floor(x)
    idir = lax.rem(xf.astype(i32) + (x - xf >= half).astype(i32), 8)
    nwd = izeros
    for i, value in enumerate(_NWDIR):
        nwd = jnp.where(idir == i, value, nwd)
    m = lax.rem(nwd - 1, 4)
    pick = lambda vals: jnp.where(m == 0, vals[0], jnp.where(
        m == 1, vals[1], jnp.where(m == 2, vals[2], vals[3])))
    oa = (1 - 2 * lax.div(nwd - 1, 4)).astype(f32) * pick(oa4)
    ol = pick(ol4)
    olp = pick([ol4[1], ol4[0], ol4[3], ol4[2]])
    od = olp / jnp.maximum(ol, olmin)
    od = jnp.maximum(jnp.minimum(od, odmax), odmin)
    diag = jnp.sqrt(dxm * dxm + dxm * dxm)
    dxy = jnp.where(m < 2, dxm, diag)
    dxyp = dxy                                       # dxy4p = permuted dxy4: same values

    ulow = jnp.maximum(jnp.sqrt(ubar * ubar + vbar * vbar), one)
    rulow = one / ulow

    def velco_of(lo, hi):
        """WRF:387-396 (component of the layer wind along the low-level wind)."""
        vel = half * ((lo['u1'] + hi['u1']) * ubar + (lo['v1'] + hi['v1']) * vbar)
        vel = vel * rulow
        return jnp.where((vel < veleps) & (vel > c(0.0)), veleps, vel)

    # --- ldrag (WRF:400-437): velco <= 0 anywhere below kbl ---
    def drag_body(k, ldrag):
        vel = velco_of(level(k), level(k + 1))
        return ldrag | ((k < kb0) & (vel <= c(0.0)))

    ldrag = lax.fori_loop(0, kmax, drag_body, false)
    ldrag = ldrag | (bnv2_1 <= c(0.0)) | (ulow == one) | (var <= c(0.0))
    act = ~ldrag

    # --- Froude number and base stress (WRF:447-477) ---
    bnv = jnp.sqrt(jnp.where(act, bnv2_1, one))
    fr = jnp.minimum(bnv * rulow * var * od, frmax)
    xn = jnp.where(act, ubar * rulow, c(0.0))
    yn = jnp.where(act, vbar * rulow, c(0.0))
    efact = jnp.power(oa + two, ce * fr / frc)
    efact = jnp.minimum(jnp.maximum(efact, efmin), efmax)
    coefm = jnp.power(one + ol, oa + one)
    xlinv = coefm / dxm                              # cleff = dxmeter
    tem = fr * fr * oc1
    gfobnv = gmax * tem / ((tem + cg) * bnv)
    taub = jnp.where(act, xlinv * rhobar * ulow * ulow * ulow * gfobnv * efact, c(0.0))

    # --- flow-blocking descent (WRF:541-570): k = kbl .. kpblmin ---
    zl_kbl = plt.load(z_ref.at[kb0, col])

    def fb_cond(s):
        k, kblk = s[0], s[1]
        # Lanes with k > kb0 are still WAITING for their band: pending until hit.
        return (k >= 1) & tile_any(act & (kblk == 0))

    def fb_body(s):
        k, kblk, fbdpe, zblk = s
        lo = level(k)
        hi = level(jnp.minimum(k + 1, nk - 1))
        _, bnv2 = interface(lo, hi)
        bnv2 = jnp.where(k + 1 < nk, bnv2, c(0.0))   # bnv2(kte) is never set (0)
        live = act & (kblk == 0) & (k <= kb0)
        dpe = bnv2 * (zl_kbl - lo['zl']) * (ld(pi_ref, k) - ld(pi_ref, k + 1)) / g_ / lo['rho']
        fbdpe = jnp.where(live, fbdpe + dpe, fbdpe)
        fbdke = half * (lo['u1'] * lo['u1'] + lo['v1'] * lo['v1'])
        hit = live & (fbdpe >= fbdke)
        kblk = jnp.where(hit, jnp.minimum(k + 1, kbl), kblk)
        zblk = jnp.where(hit, lo['zl'] - zl1, zblk)
        return k - 1, kblk, fbdpe, zblk

    _, kblk, _, zblk = lax.while_loop(
        fb_cond, fb_body, (jnp.max(jnp.where(act, kb0, 0)), izeros, zeros, zeros))
    has_blk = act & (kblk != 0)
    fbdcd = jnp.maximum(two - one / od, c(0.0))
    taufb1 = (half * rhobar * coefm / (dxm * dxm) * fbdcd * dxyp * olp * zblk
              * (ulow * ulow))
    tautem = taufb1 / jnp.maximum(kblk - 1, 1).astype(f32)
    taufb1 = jnp.where(has_blk, taufb1, c(0.0))

    # --- saturation recursion + d(tau)/dp + dtfac (WRF:481-532, 585-603) ---
    lev0 = level(0)

    def main_body(k, s):
        lo, tp, tf, icrilv, dtfac = s
        k1 = k + 1                                   # Fortran level of `lo`
        hi = level(jnp.minimum(k + 1, nk - 1))
        has_hi = k + 1 < nk
        usqj, bnv2 = interface(lo, hi)
        vel = velco_of(lo, hi)
        ge = k1 >= kbl
        in_sat = (k1 >= 2) & (k1 <= nk - 1)
        icrilv = jnp.where(in_sat & ge, icrilv | (usqj < ric) | (vel <= c(0.0)), icrilv)
        brvf = jnp.sqrt(jnp.maximum(bnv2, bnv2min))
        active = in_sat & ge & act & ~icrilv & (tp > c(0.0))
        temv = one / vel
        tem1 = coefm / dxy * (hi['rho'] + lo['rho']) * brvf * vel * half
        hd = jnp.sqrt(tp / tem1)
        fro = brvf * hd * temv
        tem2 = jnp.sqrt(usqj)
        temr = one + tem2 * fro
        rim = usqj * (one - fro) / (temr * temr)
        temc = two + one / tem2
        hds = vel * (two * jnp.sqrt(temc) - temc) / brvf
        existing = jnp.where(k1 + 1 <= kbl, taub, c(0.0))  # taup(k+1) init (WRF:481-485)
        existing = jnp.where(has_hi, existing, c(0.0))     # taup(kte+1) is never set
        tp_hi = jnp.where(active, jnp.where(rim <= ric, tem1 * hds * hds, tp), existing)
        tf_hi = jnp.where(has_blk & (k1 + 1 <= kblk), tf - tautem, c(0.0))
        tot_lo = jnp.where(has_blk, tp + tf, tp)
        tot_hi = jnp.where(has_blk, tp_hi + tf_hi, tp_hi)
        dp = ld(pi_ref, k) - ld(pi_ref, k + 1)
        taud = one * (tot_hi - tot_lo) * g_ / dp
        lim = (k1 <= kbl) & (k1 <= nk - 1) & (taud != c(0.0))
        dtfac = jnp.where(lim, jnp.minimum(dtfac, jnp.abs(vel / (deltim * taud))), dtfac)
        plt.store(taud_ref.at[at(k), col], taud)
        return hi, tp_hi, tf_hi, icrilv, dtfac

    s = lax.fori_loop(0, nk, main_body, (lev0, taub, taufb1, false, jnp.full((TX,), 1.0, f32)))
    plt.store(dtfac_ref.at[col], s[-1])
    plt.store(xn_ref.at[col], xn)
    plt.store(yn_ref.at[col], yn)


_COLS = ("uproj", "vproj", "t1", "q1", "prsl", "prslk", "zl", "prsi")
_STATS = ("var", "con", "oa1", "oa2", "oa3", "oa4", "ol1", "ol2", "ol3", "ol4",
          "sina", "cosa", "dxmeter")


def gwdo_profile_native(cols, stats, deltim, *, interpret=None):
    """Kernel call. ``cols``: (K,B) level-major columns (``prsi`` (K+1,B)),
    ``stats``: (B,) statics; returns ``(taud, dtfac, xn, yn)`` in float32."""
    if interpret is None:
        interpret = jax.default_backend() == "cpu"
    args = [jnp.asarray(cols[name], jnp.float32) for name in _COLS]
    args += [jnp.asarray(stats[name], jnp.float32) for name in _STATS]
    args.append(jnp.asarray(deltim, jnp.float32).reshape((1,)))
    nk, ncol = args[0].shape

    def kernel(*refs):
        with jax.enable_x64(False):
            _kernel(*refs, nk=nk, ncol=ncol)

    shapes = (jax.ShapeDtypeStruct((nk, ncol), jnp.float32),) + tuple(
        jax.ShapeDtypeStruct((ncol,), jnp.float32) for _ in range(3))
    return pl.pallas_call(kernel, out_shape=shapes, grid=(pl.cdiv(ncol, TX),),
                          interpret=interpret, name="gwdo_column_fp32",
                          compiler_params=plt.CompilerParams(num_warps=4))(*args)


def gwdo_epilogue(taud, dtfac, xn, yn, sina, cosa, *, prsi=None):
    """WRF:597-641 with rublten/rvblten entering as zero: limiter, x/y split, rotation."""
    taud = taud * dtfac
    dudt = taud * xn
    dvdt = taud * yn
    rublten = dudt * cosa + dvdt * sina
    rvblten = -(dudt * sina) + dvdt * cosa
    if prsi is None:
        return rublten, rvblten
    # WRF:623-640: integrate earth-relative stresses over pressure thickness,
    # then rotate the surface sums. Reuse the limited profile from this call.
    prsi = jnp.asarray(prsi, jnp.float32)
    delp = prsi[:-1] - prsi[1:]
    minus_rgrav = -jnp.asarray(1.0, jnp.float32) / jnp.asarray(9.81, jnp.float32)
    dusfc = jnp.sum(dudt * delp, axis=0) * minus_rgrav
    dvsfc = jnp.sum(dvdt * delp, axis=0) * minus_rgrav
    return rublten, rvblten, dusfc * cosa + dvsfc * sina, -dusfc * sina + dvsfc * cosa


def gwdo_tendencies_native(cols, stats, deltim, *, interpret=None):
    """Grid-relative GWDO tendencies (K,B) from level-major WRF phy columns."""
    taud, dtfac, xn, yn = gwdo_profile_native(cols, stats, deltim, interpret=interpret)
    sina = jnp.asarray(stats["sina"], jnp.float32)
    cosa = jnp.asarray(stats["cosa"], jnp.float32)
    rublten, rvblten, dusfcg, dvsfcg = gwdo_epilogue(
        taud, dtfac, xn, yn, sina, cosa, prsi=cols["prsi"])
    return {"rublten": rublten, "rvblten": rvblten, "dusfcg": dusfcg, "dvsfcg": dvsfcg}


def gwdo_tendencies_from_state(state, dt, statics, grid, *, return_surface_stress=False):
    """``couplers.gwdo_tendencies`` on the native path: WRF ``phy_prep`` inputs in REAL + kernel.

    Inputs follow module_big_step_utilities_em.F:4830-4935 (``use_theta_m=1``):
    ``th=theta_m/(1+Rv/Rd*qv)``, ``pi=(p/p1000mb)**rcp``, ``z_at_w=ph/g`` with WRF
    ``g=9.81``, ``p8w`` from the shared WRF helper (couplers._wrf_phy_prep_p8w, also used
    by the default adapter since B42). The level-major State layout is used directly
    (no column transposes). Returns REAL A-grid tendencies ``(nz, ny, nx)``.
    """
    from gpuwrf.coupling import physics_couplers as pc

    f32 = jnp.float32
    c = lambda value: jnp.asarray(value, f32)
    nz, ny, nx = state.theta.shape
    r_d = c(287.0)
    rcp = r_d / (c(7.0) * r_d / c(2.0))
    qv = jnp.asarray(state.qv, f32)
    th = jnp.asarray(state.theta, f32) / (c(1.0) + c(461.6) / r_d * qv)
    p = jnp.asarray(state.p, f32)
    pii = jnp.power(p / c(100000.0), rcp)
    u = jnp.asarray(state.u, f32)
    v = jnp.asarray(state.v, f32)
    z_at_w = jnp.asarray(state.ph, f32) / c(pc.WRF_PHYSICS_G)
    z = c(0.5) * (z_at_w[:-1] + z_at_w[1:])
    p8w = pc._wrf_phy_prep_p8w(p, z_at_w, grid.metrics.fnm, grid.metrics.fnp)
    flat = lambda a: a.reshape((a.shape[0], ny * nx))
    cols = dict(uproj=flat(c(0.5) * (u[:, :, :-1] + u[:, :, 1:])),
                vproj=flat(c(0.5) * (v[:, :-1, :] + v[:, 1:, :])),
                t1=flat(th * pii), q1=flat(qv), prsl=flat(p), prslk=flat(pii),
                zl=flat(z), prsi=flat(p8w))
    stats = dict(var=statics.var, con=statics.oc1, oa1=statics.oa1, oa2=statics.oa2,
                 oa3=statics.oa3, oa4=statics.oa4, ol1=statics.ol1, ol2=statics.ol2,
                 ol3=statics.ol3, ol4=statics.ol4, sina=statics.sina, cosa=statics.cosa,
                 dxmeter=statics.dxmeter)
    stats = {k: jnp.asarray(v, f32) for k, v in stats.items()}
    taud, dtfac, xn, yn = gwdo_profile_native(cols, stats, dt)
    out = gwdo_epilogue(taud, dtfac, xn, yn, stats["sina"], stats["cosa"],
                        prsi=cols["prsi"] if return_surface_stress else None)
    tendencies = out[0].reshape((nz, ny, nx)), out[1].reshape((nz, ny, nx))
    if return_surface_stress:
        return (*tendencies, out[2].reshape((ny, nx)), out[3].reshape((ny, nx)))
    return tendencies


__all__ = ["gwdo_tendencies_from_state", "gwdo_profile_native", "gwdo_epilogue",
           "gwdo_tendencies_native", "native_enabled"]
