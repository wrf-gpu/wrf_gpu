"""WRF RRTMG SW band solve in ONE Pallas launch per band: optics -> reftra -> vrtqdr -> g-sums.

Per band, WRF ``spcvmc_sw`` (module_ra_rrtmg_sw.F) forms the delta-scaled clear and
cloudy layer optics, the Eddington ``reftra_sw`` of both, the McICA-mixed total layer
values, and runs ``vrtqdr_sw`` TWICE -- total sky and clear sky (zrefc/ztrac) -- before
accumulating the g-point fluxes.  The XLA path materialises every one of those
``(ncol, nlay, 16)`` intermediates; here they stay in registers.  Both streams share
the layer optics of one launch; only the upward (bottom-up) adding pass is kept per
g-point (four scratch arrays), the downward pass recomputes the layer values and
emits g-summed fluxes directly (summation order is the only intended difference).

Numerics follow ``rrtmg_sw._sw_band_tile_fluxes`` / ``_sw_band_scan_optics_fluxes``
(REAL entry, float32) term for term; cloud scattering coefficients that depend only
on the band are formed by the caller.
"""

from functools import partial
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pt

TX = 8          # 8 columns x 16 g lanes = 128 elements = one per thread at 4 warps
WARPS = 4

F32 = jnp.float32


def _c(x):
    return F32(x)


def _lookup(od):
    """`rrtmg_sw._sw_transmittance_lookup` (WRF rrsw_tbl exp lookup + low-tau expansion)."""
    one = _c(1.0)
    tau = jnp.minimum(od, _c(500.0))
    tblint = _c(10000.0)
    bpade = _c(1.0 / 0.278)
    tblind = tau / (bpade + tau)
    idx = jnp.clip((tblint * tblind + _c(0.5)).astype(jnp.int32), 0, 10000)
    tfn = idx.astype(F32) / tblint
    tau_tbl = bpade * tfn / jnp.maximum(one - tfn, _c(1.0e-30))
    value = jnp.maximum(jnp.exp(-tau_tbl), _c(1.0e-20))
    value = jnp.where(idx == 0, one, value)
    value = jnp.where(idx == 10000, _c(1.0e-20), value)
    expansion = one - tau + _c(0.5) * tau * tau
    return jnp.where(tau <= _c(0.06), expansion, value)


def _wrf_reftra_enabled():
    from gpuwrf.physics.rrtmg_sw import _sw_reftra_wrf_enabled
    return _sw_reftra_wrf_enabled()


def _ssa_upper():
    return 1.0 if _wrf_reftra_enabled() else 0.999999


def _reftra(tau, omega, asymmetry, mu0, active):
    """`rrtmg_sw._reftra_eddington` on register blocks (mu0 already broadcast)."""
    one = _c(1.0)
    tau = jnp.maximum(tau, _c(1.0e-10))
    omega = jnp.clip(omega, _c(0.0), _c(_ssa_upper()))
    asymmetry = jnp.clip(asymmetry, _c(-0.999999), _c(0.999999))
    mu0 = jnp.maximum(mu0, _c(1.0e-6))
    g3 = _c(3.0) * asymmetry
    if _wrf_reftra_enabled():
        gamma1 = (_c(8.0) - omega * (_c(5.0) + g3)) * _c(0.25)
        gamma2 = _c(3.0) * omega * (one - asymmetry) * _c(0.25)
    else:
        gamma1 = (_c(7.0) - omega * (_c(4.0) + g3)) * _c(0.25)
        gamma2 = -(one - omega * (_c(4.0) - g3)) * _c(0.25)
    gamma3 = (_c(2.0) - g3 * mu0) * _c(0.25)
    gamma4 = one - gamma3
    ratio = asymmetry / jnp.maximum(one - asymmetry, _c(1.0e-12))
    zwo_denom = one - (one - omega) * (ratio * ratio)
    zwo = jnp.where((omega > _c(0.0)) & (jnp.abs(zwo_denom) > _c(1.0e-12)),
                    omega / jnp.where(zwo_denom == _c(0.0), one, zwo_denom), _c(0.0))
    conservative = zwo >= _c(0.9999995)

    za = gamma1 * mu0
    za_cons = za - gamma3
    zgt = gamma1 * tau
    exp_mu = _lookup(tau / mu0)
    pref_cons = (zgt - za_cons * (one - exp_mu)) / (one + zgt)
    ptra_cons = one - pref_cons
    prefd_cons = zgt / (one + zgt)
    ptrad_cons = one - prefd_cons
    low = exp_mu == one
    pref_cons = jnp.where(low, _c(0.0), pref_cons)
    ptra_cons = jnp.where(low, one, ptra_cons)
    prefd_cons = jnp.where(low, _c(0.0), prefd_cons)
    ptrad_cons = jnp.where(low, one, ptrad_cons)

    za1 = gamma1 * gamma4 + gamma2 * gamma3
    za2 = gamma1 * gamma3 + gamma2 * gamma4
    zrk = jnp.sqrt(jnp.maximum(gamma1 * gamma1 - gamma2 * gamma2, _c(1.0e-14)))
    zrp = zrk * mu0
    zrp1 = one + zrp
    zrm1 = one - zrp
    zrk2 = _c(2.0) * zrk
    zrpp = one - zrp * zrp
    zrkg = zrk + gamma1
    zr1 = zrm1 * (za2 + zrk * gamma3)
    zr2 = zrp1 * (za2 - zrk * gamma3)
    zr3 = zrk2 * (gamma3 - za2 * mu0)
    zr4 = zrpp * zrkg
    zr5 = zrpp * (zrk - gamma1)
    zt1 = zrp1 * (za1 + zrk * gamma4)
    zt2 = zrm1 * (za1 - zrk * gamma4)
    zt3 = zrk2 * (gamma4 + za1 * mu0)
    zbeta = (gamma1 - zrk) / jnp.maximum(zrkg, _c(1.0e-12))

    exp_h = _lookup(zrk * tau)
    # f32 max(x, 1e-300) == max(x, 0) (the literal flushes to zero in REAL).
    inv_exp_h = one / jnp.maximum(exp_h, _c(0.0))
    inv_exp_mu = one / jnp.maximum(exp_mu, _c(0.0))
    zdenr = zr4 * inv_exp_h + zr5 * exp_h
    zdent = zr4 * inv_exp_h + zr5 * exp_h
    singular = (zdenr >= _c(-1.0e-8)) & (zdenr <= _c(1.0e-8))
    pref_non = omega * (zr1 * inv_exp_h - zr2 * exp_h - zr3 * exp_mu) / jnp.where(singular, one, zdenr)
    ptra_non = exp_mu - exp_mu * omega * (zt1 * inv_exp_h - zt2 * exp_h - zt3 * inv_exp_mu) / jnp.where(
        jnp.abs(zdent) > _c(0.0), zdent, one)
    pref_non = jnp.where(singular, _c(1.0e-8), pref_non)
    ptra_non = jnp.where(singular, exp_mu, ptra_non)
    exp_2 = exp_h * exp_h
    zdend = one / jnp.maximum((one - zbeta * exp_2) * zrkg, _c(1.0e-12))
    prefd_non = gamma2 * (one - exp_2) * zdend
    ptrad_non = zrk2 * exp_h * zdend

    pref = jnp.where(conservative, pref_cons, pref_non)
    ptra = jnp.where(conservative, ptra_cons, ptra_non)
    prefd = jnp.where(conservative, prefd_cons, prefd_non)
    ptrad = jnp.where(conservative, ptrad_cons, ptrad_non)
    identity = active <= _c(0.0)
    return (jnp.where(identity, _c(0.0), pref), jnp.where(identity, _c(0.0), prefd),
            jnp.where(identity, one, ptra), jnp.where(identity, one, ptrad))


def _kernel(tau_gas, tau_ray, cloud, incloud, cfac, gmask, sflux, colf, band_ref,
            down, up, direct, clear_down, clear_up, su, sud, scu, scud, *, nlay, ng, ngin, nbands_loop=None):
    """``nbands_loop=None``: one band (``band_ref``), ``tau`` ``(npad, nlay, ngin)``.  Otherwise all
    ``nbands_loop`` bands in one launch (``tau`` ``(npad, nlay, nbands, ngin)``); each band's partial is
    written once to ``out[col, band, lev]`` (no cross-thread read-modify-write) and summed by the caller."""
    rr = pl.program_id(0) * TX + jnp.arange(TX)
    rows = rr[:, None]
    points = jnp.arange(ng)[None, :]
    # Inputs keep their ngin g-points; pad lanes re-read the last one (no copy) and are
    # neutralised by gmask = 0 (identity layers) and a zero source flux.
    gin = jnp.minimum(points, ngin - 1)
    zero, one, eps, tiny = _c(0.0), _c(1.0), _c(1.0e-12), _c(1.0e-10)
    mu = colf[rows, 0]
    alb = colf[rows, 1] + jnp.zeros((TX, ng), F32)
    outs = (down, up, direct, clear_down, clear_up)

    if nbands_loop is None:
        def sunlit():
            _solve_band(band_ref[0], lambda j: (tau_gas[rows, j, gin], tau_ray[rows, j, gin]), None)
    else:
        def sunlit():
            def per_band(b, carry):
                _solve_band(b, lambda j: (tau_gas[rows, j, b, gin], tau_ray[rows, j, b, gin]), b)
                return carry
            jax.lax.fori_loop(0, nbands_loop, per_band, 0)

    def dark():
        # WRF RRTMG_SWRAD dorrsw=.false. (coszen <= 0): no SW transfer, fluxes exactly zero.
        def clear(lev, carry):
            for out in outs:
                if nbands_loop is None:
                    out[rr, lev] = jnp.zeros((TX,), F32)
                else:
                    for b in range(nbands_loop):
                        out[rr, b, lev] = jnp.zeros((TX,), F32)
            return carry
        jax.lax.fori_loop(0, nlay + 1, clear, 0)

    def put(out, lev, value, slot):
        if slot is None:
            out[rr, lev] = value
        else:
            out[rr, slot, lev] = value

    def _solve_band(band, taus, slot):
        top = jnp.where(points < ngin, (mu * colf[rows, 2]) * sflux[rows, band, gin], zero)
        m = gmask[band, points] + jnp.zeros((TX, ng), F32)
        mu_b = mu + jnp.zeros((TX, ng), F32)
        mu_d = jnp.maximum(mu_b, _c(1.0e-6))
        # Band cloud coefficients (caller-formed): ext, delta-scale tau factor, omega, asymmetry.
        lx, ix, sx = cfac[0, band], cfac[1, band], cfac[2, band]
        ld, id_, sd = cfac[3, band], cfac[4, band], cfac[5, band]
        lo, io, so = cfac[6, band], cfac[7, band], cfac[8, band]
        la, ia, sa = cfac[9, band], cfac[10, band], cfac[11, band]

        def layer(k):
            j = nlay - 1 - k            # inputs are bottom-up layers; the solve is top-down
            tg, tr = taus(j)
            c = cloud[rows, j, band, gin]
            # clear (gas + Rayleigh), delta-scaled with g = 0 (WRF :8330-8360)
            tco = tg + tr
            oco = jnp.clip(tr / jnp.maximum(tco, tiny), zero, _c(_ssa_upper()))
            f = zero * zero
            dt = jnp.maximum(one - f * oco, eps)
            tcl = dt * tco
            ocl = jnp.clip((one - f) * oco / dt, zero, _c(_ssa_upper()))
            acl = jnp.clip((zero - f) / jnp.maximum(one - f, eps), _c(-0.999999), _c(0.999999)) + jnp.zeros_like(tco)
            # cloud (cldprmc in-cloud paths x McICA amount)
            tl = ld * ((incloud[rows, j, 0] * lx) * c)
            ti = id_ * ((incloud[rows, j, 1] * ix) * c)
            ts = sd * ((incloud[rows, j, 2] * sx) * c)
            sl, si, ss = tl * lo, ti * io, ts * so
            tcd = tl + ti + ts
            scd = sl + si + ss
            ocd = jnp.clip(scd / jnp.maximum(tcd, tiny), zero, _c(_ssa_upper()))
            ocd = jnp.where(c > zero, ocd, one)
            acd = jnp.where(scd > tiny, (sl * la + si * ia + ss * sa) / jnp.maximum(scd, tiny), zero)
            stot = tcl * ocl + tcd * ocd
            ttot = jnp.maximum(tcl + tcd, tiny)
            otot = jnp.clip(stot / jnp.maximum(ttot, tiny), zero, _c(_ssa_upper()))
            atot = jnp.where(stot > tiny, (tcl * ocl * acl + tcd * ocd * acd) / jnp.maximum(stot, tiny), zero)
            tcl = jnp.maximum(tcl, tiny) * m
            ocl = ocl * m
            acl = acl * m
            ttot = ttot * m
            otot = otot * m
            atot = atot * m
            cl = jnp.clip(c, zero, one)
            cloud_active = m * (cl > _c(1.0e-12)).astype(F32)
            rc, rdc, tc, tdc = _reftra(tcl, ocl, acl, mu_b, m)
            rk, rdk, tk, tdk = _reftra(ttot, otot, atot, mu_b, cloud_active)
            w = one - cl
            r = w * rc + cl * rk
            rd = w * rdc + cl * rdk
            t = w * tc + cl * tk
            td = w * tdc + cl * tdk
            bcl = _lookup(tcl / mu_d)
            bcd = _lookup(ttot / mu_d)
            b = jnp.where(m > zero, (w * bcl + cl * bcd) * m, one)
            bc = jnp.where(m > zero, bcl, one)
            return (r, rd, t, td, b), (rc, rdc, tc, tdc, bc)

        # Upward adding pass (vrtqdr_sw bottom-up), both streams; surface = albedo.
        su[rows, nlay, points] = alb
        sud[rows, nlay, points] = alb
        scu[rows, nlay, points] = alb
        scud[rows, nlay, points] = alb

        def reflect(i, carry):
            u, ud, cu, cud = carry
            k = nlay - 1 - i
            (r, rd, t, td, b), (rc, rdc, tc, tdc, bc) = layer(k)
            refl = one / jnp.maximum(one - ud * rd, eps)
            nu = r + td * ((t - b) * ud + b * u) * refl
            nud = rd + td * td * ud * refl
            reflc = one / jnp.maximum(one - cud * rdc, eps)
            ncu = rc + tdc * ((tc - bc) * cud + bc * cu) * reflc
            ncud = rdc + tdc * tdc * cud * reflc
            su[rows, k, points] = nu
            sud[rows, k, points] = nud
            scu[rows, k, points] = ncu
            scud[rows, k, points] = ncud
            return nu, nud, ncu, ncud

        jax.lax.fori_loop(0, nlay, reflect, (alb, alb, alb, alb))

        def emit(k, d, tr_, df, u, ud):
            refl = one / jnp.maximum(one - df * ud, eps)
            fup = (d * u + (tr_ - d) * ud) * refl
            fdn = d + (tr_ - d + d * u * df) * refl
            return jnp.sum(fdn * top, axis=1), jnp.sum(fup * top, axis=1)

        def store(k, d, tr_, df, dc, trc, dfc):
            lev = nlay - k              # output is bottom-up interfaces
            fdn, fup = emit(k, d, tr_, df, su[rows, k, points], sud[rows, k, points])
            cdn, cup = emit(k, dc, trc, dfc, scu[rows, k, points], scud[rows, k, points])
            put(down, lev, fdn, slot)
            put(up, lev, fup, slot)
            put(direct, lev, jnp.sum(d * top, axis=1), slot)
            put(clear_down, lev, cdn, slot)
            put(clear_up, lev, cup, slot)

        def transmit(k, carry):
            d, tr_, df, dc, trc, dfc = carry
            store(k, d, tr_, df, dc, trc, dfc)
            (r, rd, t, td, b), (rc, rdc, tc, tdc, bc) = layer(k)
            refl = one / jnp.maximum(one - rd * df, eps)
            ntr = d * t + td * ((tr_ - d) + d * r * df) * refl
            ndf = rd + td * td * df * refl
            reflc = one / jnp.maximum(one - rdc * dfc, eps)
            ntrc = dc * tc + tdc * ((trc - dc) + dc * rc * dfc) * reflc
            ndfc = rdc + tdc * tdc * dfc * reflc
            return d * b, ntr, ndf, dc * bc, ntrc, ndfc

        ones = jnp.ones((TX, ng), F32)
        zeros = jnp.zeros((TX, ng), F32)
        final = jax.lax.fori_loop(0, nlay, transmit, (ones, ones, zeros, ones, ones, zeros))
        store(nlay, *final)

    jax.lax.cond(jnp.max(mu) > zero, sunlit, dark)


def _launch(tau_gas, tau_ray, cloud_amount, incloud, cloud_coeffs, gpoint_mask, sfluxzen,
            coszen, surface_albedo, source_scale, band, *, all_bands, interpret):
    tau_suffix = 3 if all_bands else 2
    leading, nlay, ngin = tau_gas.shape[:-tau_suffix], tau_gas.shape[-tau_suffix], tau_gas.shape[-1]
    ncol = math.prod(leading) if leading else 1
    npad = ((ncol + TX - 1) // TX) * TX
    nbands = cloud_amount.shape[-2]
    # Lanes beyond the input g-points (power-of-two pad) carry mask 0 and zero source
    # flux: identity layers whose fluxes add exactly +0 to the g-sum.
    ng = 1 << (ngin - 1).bit_length()

    def pack(value, suffix):
        a = jnp.broadcast_to(value, leading + suffix).reshape((ncol,) + suffix).astype(F32)
        return a if npad == ncol else jnp.pad(a, ((0, npad - ncol),) + ((0, 0),) * len(suffix))

    tau_shape = (nlay, nbands, ngin) if all_bands else (nlay, ngin)
    colf = jnp.stack([coszen, surface_albedo, source_scale], axis=-1)
    gmask = jnp.pad(jnp.asarray(gpoint_mask, F32), ((0, 0), (0, ng - ngin)))
    args = [pack(tau_gas, tau_shape), pack(tau_ray, tau_shape), pack(cloud_amount, (nlay, nbands, ngin)),
            pack(incloud, (nlay, incloud.shape[-1])), jnp.asarray(cloud_coeffs, F32), gmask, pack(sfluxzen, (nbands, ngin)),
            pack(colf, (3,)), jnp.reshape(jnp.asarray(band, jnp.int32), (1,))]
    flux = jax.ShapeDtypeStruct((npad, nbands, nlay + 1) if all_bands else (npad, nlay + 1), F32)
    scratch = jax.ShapeDtypeStruct((npad, nlay + 1, ng), F32)
    kernel = partial(_kernel_mp_re if incloud.shape[-1] == 6 else _kernel, nlay=nlay, ng=ng, ngin=ngin, nbands_loop=nbands if all_bands else None)
    result = pl.pallas_call(kernel, grid=(npad // TX,),
                            out_shape=[flux] * 5 + [scratch] * 4, interpret=interpret,
                            compiler_params=pt.CompilerParams(num_warps=WARPS),
                            name="rrtmg_sw_allband_sums" if all_bands else "rrtmg_sw_band_sums")(*args)
    fields = result[:5]
    if all_bands:
        # Band-order sum (((b0 + b1) + b2) ...), the band-scan carry's association.
        def band_sum(x):
            acc = x[:, 0]
            for i in range(1, nbands):
                acc = acc + x[:, i]
            return acc
        fields = tuple(band_sum(x) for x in fields)
    return tuple(x[:ncol].reshape(leading + (nlay + 1,)) for x in fields)


def sw_band_flux_sums(tau_gas, tau_ray, cloud_amount, incloud, cloud_coeffs, gpoint_mask, sfluxzen,
                      coszen, surface_albedo, source_scale, band, *, interpret=False):
    """One band's g-summed ``(down, up, direct, clear_down, clear_up)`` over ``nlay+1`` interfaces.

    ``tau_gas``/``tau_ray``: ``(..., nlay, ng)`` bottom-up; ``cloud_amount``: ``(..., nlay, nbands, ng)``
    McICA; ``incloud``: ``(..., nlay, 3)`` liquid/ice/snow in-cloud paths; ``cloud_coeffs``: ``(12, nbands)``
    rows ext(l,i,s), delta tau factor(l,i,s), omega(l,i,s), asymmetry(l,i,s); ``gpoint_mask``: ``(nbands, ng)``;
    ``sfluxzen``: ``(..., nbands, ng)``; ``coszen``/``surface_albedo``/``source_scale``: ``(...)``; ``band``
    traced int.  Outputs are float32, bottom-up interfaces.
    """
    return _launch(tau_gas, tau_ray, cloud_amount, incloud, cloud_coeffs, gpoint_mask, sfluxzen,
                   coszen, surface_albedo, source_scale, band, all_bands=False, interpret=interpret)


def sw_allband_flux_sums(tau_gas, tau_ray, cloud_amount, incloud, cloud_coeffs, gpoint_mask, sfluxzen,
                         coszen, surface_albedo, source_scale, *, interpret=False):
    """All bands in ONE launch (``tau_gas``/``tau_ray`` ``(..., nlay, nbands, ng)``), summed over bands
    in band order 0..nbands-1 exactly like the band-scan carry; same outputs as :func:`sw_band_flux_sums`."""
    return _launch(tau_gas, tau_ray, cloud_amount, incloud, cloud_coeffs, gpoint_mask, sfluxzen,
                   coszen, surface_albedo, source_scale, 0, all_bands=True, interpret=interpret)


def _kernel_mp_re(tau_gas, tau_ray, cloud, incloud, cfac, gmask, sflux, colf, band_ref,
            down, up, direct, clear_down, clear_up, su, sud, scu, scud, *, nlay, ng, ngin, nbands_loop=None):
    """``nbands_loop=None``: one band (``band_ref``), ``tau`` ``(npad, nlay, ngin)``.  Otherwise all
    ``nbands_loop`` bands in one launch (``tau`` ``(npad, nlay, nbands, ngin)``); each band's partial is
    written once to ``out[col, band, lev]`` (no cross-thread read-modify-write) and summed by the caller."""
    rr = pl.program_id(0) * TX + jnp.arange(TX)
    rows = rr[:, None]
    points = jnp.arange(ng)[None, :]
    # Inputs keep their ngin g-points; pad lanes re-read the last one (no copy) and are
    # neutralised by gmask = 0 (identity layers) and a zero source flux.
    gin = jnp.minimum(points, ngin - 1)
    zero, one, eps, tiny = _c(0.0), _c(1.0), _c(1.0e-12), _c(1.0e-10)
    mu = colf[rows, 0]
    alb = colf[rows, 1] + jnp.zeros((TX, ng), F32)
    outs = (down, up, direct, clear_down, clear_up)

    if nbands_loop is None:
        def sunlit():
            _solve_band(band_ref[0], lambda j: (tau_gas[rows, j, gin], tau_ray[rows, j, gin]), None)
    else:
        def sunlit():
            def per_band(b, carry):
                _solve_band(b, lambda j: (tau_gas[rows, j, b, gin], tau_ray[rows, j, b, gin]), b)
                return carry
            jax.lax.fori_loop(0, nbands_loop, per_band, 0)

    def dark():
        # WRF RRTMG_SWRAD dorrsw=.false. (coszen <= 0): no SW transfer, fluxes exactly zero.
        def clear(lev, carry):
            for out in outs:
                if nbands_loop is None:
                    out[rr, lev] = jnp.zeros((TX,), F32)
                else:
                    for b in range(nbands_loop):
                        out[rr, b, lev] = jnp.zeros((TX,), F32)
            return carry
        jax.lax.fori_loop(0, nlay + 1, clear, 0)

    def put(out, lev, value, slot):
        if slot is None:
            out[rr, lev] = value
        else:
            out[rr, slot, lev] = value

    def _solve_band(band, taus, slot):
        top = jnp.where(points < ngin, (mu * colf[rows, 2]) * sflux[rows, band, gin], zero)
        m = gmask[band, points] + jnp.zeros((TX, ng), F32)
        mu_b = mu + jnp.zeros((TX, ng), F32)
        mu_d = jnp.maximum(mu_b, _c(1.0e-6))
        # Band cloud coefficients (caller-formed): ext, delta-scale tau factor, omega, asymmetry.

        def layer(k):
            j = nlay - 1 - k            # inputs are bottom-up layers; the solve is top-down
            def coeff(slot, radius, ice):
                start, step, length = (5., 3., 46) if ice else (2.5, 1., 58)
                idx = jnp.clip(jnp.floor((radius - _c(start)) / _c(step)).astype(jnp.int32), 0, length - 2)
                frac = (radius - (_c(start) + idx.astype(F32) * _c(step))) / _c(step)
                return cfac[slot, band, idx] + frac * (cfac[slot, band, idx + 1] - cfac[slot, band, idx])
            rl, ri, rs = incloud[rr, j, 3], incloud[rr, j, 4], incloud[rr, j, 5]
            lx, lssa, lasy = (coeff(n, rl, False) for n in (0, 1, 2))
            ix, issa, iasy, ifwd = (coeff(n, ri, True) for n in (3, 4, 5, 6))
            sx, sssa, sasy, sfwd = (coeff(n, rs, True) for n in (3, 4, 5, 6))
            def scale_coeff(ssa, asym, forward):
                den = jnp.maximum(one - forward * ssa, eps)
                om = jnp.clip(ssa * (one - forward) / den, zero, _c(_ssa_upper()))
                asy = jnp.clip((asym - forward) / jnp.maximum(one - forward, eps), _c(-.999999), _c(.999999))
                return den[:, None], om[:, None], asy[:, None]
            ld, lo, la = scale_coeff(lssa, lasy, lasy * lasy)
            id_, io, ia = scale_coeff(issa, iasy, jnp.minimum(iasy, ifwd + _c(.5) / issa))
            sd, so, sa = scale_coeff(sssa, sasy, jnp.minimum(sasy, sfwd + _c(.5) / sssa))
            lx, ix, sx = lx[:, None], ix[:, None], sx[:, None]
            tg, tr = taus(j)
            c = cloud[rows, j, band, gin]
            # clear (gas + Rayleigh), delta-scaled with g = 0 (WRF :8330-8360)
            tco = tg + tr
            oco = jnp.clip(tr / jnp.maximum(tco, tiny), zero, _c(_ssa_upper()))
            f = zero * zero
            dt = jnp.maximum(one - f * oco, eps)
            tcl = dt * tco
            ocl = jnp.clip((one - f) * oco / dt, zero, _c(_ssa_upper()))
            acl = jnp.clip((zero - f) / jnp.maximum(one - f, eps), _c(-0.999999), _c(0.999999)) + jnp.zeros_like(tco)
            # cloud (cldprmc in-cloud paths x McICA amount)
            tl = ld * ((incloud[rows, j, 0] * lx) * c)
            ti = id_ * ((incloud[rows, j, 1] * ix) * c)
            ts = sd * ((incloud[rows, j, 2] * sx) * c)
            sl, si, ss = tl * lo, ti * io, ts * so
            tcd = tl + ti + ts
            scd = sl + si + ss
            ocd = jnp.clip(scd / jnp.maximum(tcd, tiny), zero, _c(_ssa_upper()))
            ocd = jnp.where(c > zero, ocd, one)
            acd = jnp.where(scd > tiny, (sl * la + si * ia + ss * sa) / jnp.maximum(scd, tiny), zero)
            stot = tcl * ocl + tcd * ocd
            ttot = jnp.maximum(tcl + tcd, tiny)
            otot = jnp.clip(stot / jnp.maximum(ttot, tiny), zero, _c(_ssa_upper()))
            atot = jnp.where(stot > tiny, (tcl * ocl * acl + tcd * ocd * acd) / jnp.maximum(stot, tiny), zero)
            tcl = jnp.maximum(tcl, tiny) * m
            ocl = ocl * m
            acl = acl * m
            ttot = ttot * m
            otot = otot * m
            atot = atot * m
            cl = jnp.clip(c, zero, one)
            cloud_active = m * (cl > _c(1.0e-12)).astype(F32)
            rc, rdc, tc, tdc = _reftra(tcl, ocl, acl, mu_b, m)
            rk, rdk, tk, tdk = _reftra(ttot, otot, atot, mu_b, cloud_active)
            w = one - cl
            r = w * rc + cl * rk
            rd = w * rdc + cl * rdk
            t = w * tc + cl * tk
            td = w * tdc + cl * tdk
            bcl = _lookup(tcl / mu_d)
            bcd = _lookup(ttot / mu_d)
            b = jnp.where(m > zero, (w * bcl + cl * bcd) * m, one)
            bc = jnp.where(m > zero, bcl, one)
            return (r, rd, t, td, b), (rc, rdc, tc, tdc, bc)

        # Upward adding pass (vrtqdr_sw bottom-up), both streams; surface = albedo.
        su[rows, nlay, points] = alb
        sud[rows, nlay, points] = alb
        scu[rows, nlay, points] = alb
        scud[rows, nlay, points] = alb

        def reflect(i, carry):
            u, ud, cu, cud = carry
            k = nlay - 1 - i
            (r, rd, t, td, b), (rc, rdc, tc, tdc, bc) = layer(k)
            refl = one / jnp.maximum(one - ud * rd, eps)
            nu = r + td * ((t - b) * ud + b * u) * refl
            nud = rd + td * td * ud * refl
            reflc = one / jnp.maximum(one - cud * rdc, eps)
            ncu = rc + tdc * ((tc - bc) * cud + bc * cu) * reflc
            ncud = rdc + tdc * tdc * cud * reflc
            su[rows, k, points] = nu
            sud[rows, k, points] = nud
            scu[rows, k, points] = ncu
            scud[rows, k, points] = ncud
            return nu, nud, ncu, ncud

        jax.lax.fori_loop(0, nlay, reflect, (alb, alb, alb, alb))

        def emit(k, d, tr_, df, u, ud):
            refl = one / jnp.maximum(one - df * ud, eps)
            fup = (d * u + (tr_ - d) * ud) * refl
            fdn = d + (tr_ - d + d * u * df) * refl
            return jnp.sum(fdn * top, axis=1), jnp.sum(fup * top, axis=1)

        def store(k, d, tr_, df, dc, trc, dfc):
            lev = nlay - k              # output is bottom-up interfaces
            fdn, fup = emit(k, d, tr_, df, su[rows, k, points], sud[rows, k, points])
            cdn, cup = emit(k, dc, trc, dfc, scu[rows, k, points], scud[rows, k, points])
            put(down, lev, fdn, slot)
            put(up, lev, fup, slot)
            put(direct, lev, jnp.sum(d * top, axis=1), slot)
            put(clear_down, lev, cdn, slot)
            put(clear_up, lev, cup, slot)

        def transmit(k, carry):
            d, tr_, df, dc, trc, dfc = carry
            store(k, d, tr_, df, dc, trc, dfc)
            (r, rd, t, td, b), (rc, rdc, tc, tdc, bc) = layer(k)
            refl = one / jnp.maximum(one - rd * df, eps)
            ntr = d * t + td * ((tr_ - d) + d * r * df) * refl
            ndf = rd + td * td * df * refl
            reflc = one / jnp.maximum(one - rdc * dfc, eps)
            ntrc = dc * tc + tdc * ((trc - dc) + dc * rc * dfc) * reflc
            ndfc = rdc + tdc * tdc * dfc * reflc
            return d * b, ntr, ndf, dc * bc, ntrc, ndfc

        ones = jnp.ones((TX, ng), F32)
        zeros = jnp.zeros((TX, ng), F32)
        final = jax.lax.fori_loop(0, nlay, transmit, (ones, ones, zeros, ones, ones, zeros))
        store(nlay, *final)

    jax.lax.cond(jnp.max(mu) > zero, sunlit, dark)
