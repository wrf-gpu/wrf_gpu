"""WRF-native fp32 LW g-point recurrences inside a single Pallas kernel.

All-sky and clear-sky downward/upward sweeps share one launch. Layer-local
coefficients are recomputed on the upward sweep, avoiding the reference's ten
domain-wide scan-output buffers. The final column cloud flag stays in registers.
"""

import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

TX = 4
# Band-sum kernel: 8 columns x 16 g lanes = 128 elements = one per thread at the default 4 warps
# (TX = 4 leaves half the threads replicated); bitwise == TX 4 on GPU (lever-phys LP02, d01/d02 day/night).
TX_SUMS = 8


def _lookup(tau, s):
    zero, one, half, six, small, bpade, ntab, eps = [s[i] for i in range(8)]
    index = jnp.clip((ntab * (tau / (bpade + tau)) + half).astype(jnp.int32), 0, 10000)
    f = index.astype(jnp.float32) / ntab
    end = index == 10000
    # The endpoint does not evaluate the singular Pade denominator in WRF.
    table_tau = jnp.where(end, jnp.float32(1e10), bpade * f / jnp.where(end, one, one - f))
    table_tau = jnp.where(index == 0, zero, table_tau)
    exp = jnp.where(end, eps, jnp.maximum(jnp.exp(-table_tau), eps))
    tiny = table_tau < small
    safe_tau = jnp.where(tiny | end, one, table_tau)
    safe_denom = jnp.where(tiny | end, one, one - exp)
    tfn = jnp.where(end, one, jnp.where(tiny, table_tau / six,
                    one - (one + one) * (one / safe_tau - exp / safe_denom)))
    return table_tau, exp, tfn


def _coefficients(tau, frac, cldf, taucld, sec, plank, pdn, pup, cloud, s):
    zero, one, half, six, small = [s[i] for i in range(5)]
    depth = jnp.maximum(sec * tau, zero)
    odcld = jnp.where(cldf == one, sec * taucld, zero)
    ef = (one - jnp.exp(-odcld)) * cldf
    total = depth + odcld
    gas_tau, gas_exp, gas_tfn = _lookup(depth, s)
    _, tot_exp_b, tot_tfn_b = _lookup(total, s)
    _, tot_exp_c, tot_tfn_c = _lookup(gas_tau + odcld, s)
    gas_small = depth <= small
    case_a = total < small
    a_small = depth - half * depth * depth
    a_large = one - gas_exp
    t_small = depth / six
    trans = jnp.where(gas_small, a_small, a_large)
    tfn_gas = jnp.where(gas_small, t_small, gas_tfn)
    atot = jnp.where(case_a, total - half * total * total,
                    jnp.where(gas_small, one - tot_exp_b, one - tot_exp_c))
    tfn_total = jnp.where(case_a, total / six, jnp.where(gas_small, tot_tfn_b, tot_tfn_c))
    bbd = frac * (plank + pdn * tfn_gas)
    bbu = frac * (plank + pup * tfn_gas)
    bbdtot = frac * (plank + pdn * tfn_total)
    bbutot = frac * (plank + pup * tfn_total)
    # A cloudy layer uses the same small-gas expansion for cases A and B.
    trans_cld = jnp.where(case_a | gas_small, a_small, a_large)
    gas_tfn_cld = jnp.where(case_a | gas_small, t_small, gas_tfn)
    bbd_cld = frac * (plank + pdn * gas_tfn_cld)
    bbu_cld = frac * (plank + pup * gas_tfn_cld)
    trans_all = jnp.where(cloud, trans_cld, trans)
    bbd_all = jnp.where(cloud, bbd_cld, bbd)
    bbu_all = jnp.where(cloud, bbu_cld, bbu)
    return trans_all, atot, bbd_all, bbu_all, bbdtot, bbutot, ef, trans, bbd, bbu, jnp.where(cloud, gas_tfn_cld, tfn_gas)


def _kernel(tau, frac, cldf, taucld, sec, plank, planklev, plankbnd,
            emiss, cloud, valid, scalars, down, up, clear_down, clear_up,
            tfn_out, *, nlay, ng):
    rr = pl.program_id(0) * TX + jnp.arange(TX)
    rows = rr[:, None]
    points = jnp.arange(ng)[None, :]
    s = tuple(scalars[i] for i in range(9))
    zero, one = s[0], s[1]
    scale = s[8] * valid[points]
    secv = sec[rr][:, None]
    def terms(k):
        b = plank[rr, k][:, None]
        dn = planklev[rr, k][:, None] - b
        du = planklev[rr, k + 1][:, None] - b
        cloudy = cloud[rr, k][:, None]
        values = _coefficients(tau[rows, k, points], frac[rows, k, points],
            cldf[rows, k, points], taucld[rows, k, points], secv, b, dn, du, cloudy, s)
        return values, cloudy
    down[rows, nlay, points] = jnp.zeros((TX, ng), jnp.float32)
    clear_down[rows, nlay, points] = jnp.zeros((TX, ng), jnp.float32)
    def descend(i, carry):
        rad, clr, seen = carry
        k = nlay - 1 - i
        (a, at, bd, bu, bdt, but, ef, ac, bdc, buc, tfn), cloudy = terms(k)
        cf = cldf[rows, k, points]
        gas_src = bd * a
        rad_cloud = rad - rad * (a + ef * (one - a)) + gas_src + cf * (bdt * at - gas_src)
        rad_clear = rad + (bd - rad) * a
        new_rad = jnp.where(cloudy, rad_cloud, rad_clear)
        seen = seen | cloudy
        new_clr = jnp.where(seen, clr + (bdc - clr) * ac, new_rad)
        down[rows, k, points] = new_rad * scale
        clear_down[rows, k, points] = new_clr * scale
        tfn_out[rows, k, points] = tfn * valid[points]
        return new_rad, new_clr, seen
    zeros = jnp.zeros((TX, ng), jnp.float32)
    rd, cd, column_cloud = jax.lax.fori_loop(0, nlay, descend, (zeros, zeros, jnp.zeros((TX, ng), bool)))
    surface = frac[rows, 0, points] * plankbnd[rr][:, None]
    reflect = one - emiss[rr][:, None]
    ru = surface + reflect * rd
    cu = surface + reflect * cd
    up[rows, 0, points] = ru * scale
    clear_up[rows, 0, points] = cu * scale
    def ascend(k, carry):
        rad, clr = carry
        (a, at, bd, bu, bdt, but, ef, ac, bdc, buc, _), cloudy = terms(k)
        cf = cldf[rows, k, points]
        gas_src = bu * a
        rad_cloud = rad - rad * (a + ef * (one - a)) + gas_src + cf * (but * at - gas_src)
        rad_clear = rad + (bu - rad) * a
        new_rad = jnp.where(cloudy, rad_cloud, rad_clear)
        # WRF retains the FINAL downward iclddn throughout the upward sweep.
        new_clr = jnp.where(column_cloud, clr + (buc - clr) * ac, new_rad)
        up[rows, k + 1, points] = new_rad * scale
        clear_up[rows, k + 1, points] = new_clr * scale
        return new_rad, new_clr
    jax.lax.fori_loop(0, nlay, ascend, (ru, cu))


def lw_band_fluxes(state, tau, frac, cldf, taucld, sec, scale, valid,
                   plank, planklev, plankbnd, cloud_layer, with_clear_sky,
                   *, interpret=False):
    """Drop-in band transfer seam, returning fluxes plus the oracle tfn field."""
    leading, nlay, ng = tau.shape[:-2], tau.shape[-2], tau.shape[-1]
    if ng & (ng - 1):
        raise ValueError('Pallas g-point width must be a power of two')
    ncol = math.prod(leading) if leading else 1
    npad = ((ncol + TX - 1) // TX) * TX
    def pack(value, suffix, dtype=jnp.float32):
        a = jnp.broadcast_to(value, leading + suffix).reshape((ncol,) + suffix).astype(dtype)
        return jnp.pad(a, ((0, npad - ncol),) + ((0, 0),) * len(suffix))
    args = [pack(x, (nlay, ng)) for x in (tau, frac, cldf, taucld)]
    args += [pack(sec, ()), pack(plank, (nlay,)), pack(planklev, (nlay + 1,)),
             pack(plankbnd, ()), pack(state.surface_emissivity, ()),
             pack(cloud_layer, (nlay,), bool), jnp.asarray(valid, jnp.float32)]
    from gpuwrf.physics.rrtmg_constants import LW_BPADE, LW_EXP_EPS
    # Runtime constants preserve expression association through Triton lowering.
    args += [jnp.stack([jnp.float32(x) for x in (0, 1, .5, 6, .06, LW_BPADE, 10000, LW_EXP_EPS)]
                       + [jnp.asarray(scale, jnp.float32)] + [jnp.float32(0)] * 7)]
    flux_shape = jax.ShapeDtypeStruct((npad, nlay + 1, ng), jnp.float32)
    layer_shape = jax.ShapeDtypeStruct((npad, nlay, ng), jnp.float32)
    result = pl.pallas_call(lambda *r: _kernel(*r, nlay=nlay, ng=ng),
        grid=(npad // TX,), out_shape=[flux_shape] * 4 + [layer_shape],
        interpret=interpret)(*args)
    fields = tuple(x[:ncol].reshape(leading + (x.shape[1], ng)).astype(tau.dtype) for x in result[:5])
    return fields[0], fields[1], (fields[2] if with_clear_sky else None), (fields[3] if with_clear_sky else None), fields[4]


_NG_GLOBAL = 140


def _kernel_sums(tau, frac, mask, clw, ciw, csw, sec, plank, planklev, plankbnd,
                 emiss, cloud, valid, scalars, band_i, band_f,
                 down, up, clear_down, clear_up, *, nlay, ng):
    """:func:`_kernel` with the WRF `cldprmc` cloud optics built per g-point from
    the global McICA mask and in-cloud paths (taucmc = clw*liq + ciw*ice +
    csw*snow where cloudy, module_ra_rrtmg_lw.F cldprmc), and g-summed fluxes
    (the caller's band reduction) instead of per-g-point buffers (BP57)."""
    rr = pl.program_id(0) * TX_SUMS + jnp.arange(TX_SUMS)
    rows = rr[:, None]
    points = jnp.arange(ng)[None, :]
    s = tuple(scalars[i] for i in range(9))
    zero, one = s[0], s[1]
    scale = s[8] * valid[points]
    secv = sec[rr][:, None]
    offset, count = band_i[0], band_i[1]
    liq, ice, snow = band_f[0], band_f[1], band_f[2]
    in_band = points < count
    gidx = jnp.minimum(offset + points, _NG_GLOBAL - 1)

    def cloud_terms(k):
        cloudy_g = mask[rows, k, gidx] & in_band
        path = clw[rr, k][:, None] * liq + ciw[rr, k][:, None] * ice + csw[rr, k][:, None] * snow
        return cloudy_g.astype(jnp.float32), jnp.where(cloudy_g, path, zero)

    def terms(k):
        b = plank[rr, k][:, None]
        dn = planklev[rr, k][:, None] - b
        du = planklev[rr, k + 1][:, None] - b
        cloudy = cloud[rr, k][:, None]
        cf, tc = cloud_terms(k)
        values = _coefficients(tau[rows, k, points], frac[rows, k, points],
            cf, tc, secv, b, dn, du, cloudy, s)
        return values, cloudy, cf

    gsum = lambda x: jnp.sum(x * scale, axis=1)  # noqa: E731
    down[rr, nlay] = jnp.zeros((TX_SUMS,), jnp.float32)
    clear_down[rr, nlay] = jnp.zeros((TX_SUMS,), jnp.float32)

    def descend(i, carry):
        rad, clr, seen = carry
        k = nlay - 1 - i
        (a, at, bd, bu, bdt, but, ef, ac, bdc, buc, _tfn), cloudy, cf = terms(k)
        gas_src = bd * a
        rad_cloud = rad - rad * (a + ef * (one - a)) + gas_src + cf * (bdt * at - gas_src)
        rad_clear = rad + (bd - rad) * a
        new_rad = jnp.where(cloudy, rad_cloud, rad_clear)
        seen = seen | cloudy
        new_clr = jnp.where(seen, clr + (bdc - clr) * ac, new_rad)
        down[rr, k] = gsum(new_rad)
        clear_down[rr, k] = gsum(new_clr)
        return new_rad, new_clr, seen
    zeros = jnp.zeros((TX_SUMS, ng), jnp.float32)
    rd, cd, column_cloud = jax.lax.fori_loop(0, nlay, descend, (zeros, zeros, jnp.zeros((TX_SUMS, ng), bool)))
    surface = frac[rows, 0, points] * plankbnd[rr][:, None]
    reflect = one - emiss[rr][:, None]
    ru = surface + reflect * rd
    cu = surface + reflect * cd
    up[rr, 0] = gsum(ru)
    clear_up[rr, 0] = gsum(cu)

    def ascend(k, carry):
        rad, clr = carry
        (a, at, bd, bu, bdt, but, ef, ac, bdc, buc, _), cloudy, cf = terms(k)
        gas_src = bu * a
        rad_cloud = rad - rad * (a + ef * (one - a)) + gas_src + cf * (but * at - gas_src)
        rad_clear = rad + (bu - rad) * a
        new_rad = jnp.where(cloudy, rad_cloud, rad_clear)
        # WRF retains the FINAL downward iclddn throughout the upward sweep.
        new_clr = jnp.where(column_cloud, clr + (buc - clr) * ac, new_rad)
        up[rr, k + 1] = gsum(new_rad)
        clear_up[rr, k + 1] = gsum(new_clr)
        return new_rad, new_clr
    jax.lax.fori_loop(0, nlay, ascend, (ru, cu))


def _packer(leading):
    ncol = math.prod(leading) if leading else 1
    npad = ((ncol + TX_SUMS - 1) // TX_SUMS) * TX_SUMS
    def pack(value, suffix, dtype=jnp.float32):
        a = jnp.broadcast_to(value, leading + suffix).reshape((ncol,) + suffix).astype(dtype)
        return jnp.pad(a, ((0, npad - ncol),) + ((0, 0),) * len(suffix))
    return ncol, npad, pack


def pack_lw_cloud(cloudy_global, clw, ciw, csw):
    """Band-invariant kernel operands, packed ONCE outside the band loop: the McICA
    mask ``(..., nlay, 140)`` (WRF global g order) and in-cloud paths ``(..., nlay)``."""
    leading, nlay = cloudy_global.shape[:-2], cloudy_global.shape[-2]
    _, _, pack = _packer(leading)
    return (pack(cloudy_global, (nlay, _NG_GLOBAL), bool),
            *(pack(x, (nlay,)) for x in (clw, ciw, csw)))


def lw_band_flux_sums(state, tau, frac, packed_cloud, band, sec, scale, valid,
                      plank, planklev, plankbnd, cloud_layer, with_clear_sky,
                      *, gpoint_counts, cloud, interpret=False):
    """Band transfer returning the g-summed ``(down, up, clear_down, clear_up)`` fluxes.

    ``packed_cloud`` = :func:`pack_lw_cloud` of the McICA mask and in-cloud paths;
    ``band`` may be traced.
    """
    leading, nlay, ng = tau.shape[:-2], tau.shape[-2], tau.shape[-1]
    if ng & (ng - 1):
        raise ValueError('Pallas g-point width must be a power of two')
    ncol, npad, pack = _packer(leading)
    counts = jnp.asarray(gpoint_counts, jnp.int32)
    offsets = jnp.asarray([sum(gpoint_counts[:b]) for b in range(len(gpoint_counts))], jnp.int32)
    band_i = jnp.stack([offsets[band], counts[band]])
    band_f = jnp.stack([jnp.asarray(cloud.liquid, jnp.float32)[band], jnp.asarray(cloud.ice, jnp.float32)[band],
                        jnp.asarray(cloud.snow, jnp.float32)[band]])
    args = [pack(x, (nlay, ng)) for x in (tau, frac)] + list(packed_cloud)
    args += [pack(sec, ()), pack(plank, (nlay,)), pack(planklev, (nlay + 1,)),
             pack(plankbnd, ()), pack(state.surface_emissivity, ()),
             pack(cloud_layer, (nlay,), bool), jnp.asarray(valid, jnp.float32)]
    from gpuwrf.physics.rrtmg_constants import LW_BPADE, LW_EXP_EPS
    args += [jnp.stack([jnp.float32(x) for x in (0, 1, .5, 6, .06, LW_BPADE, 10000, LW_EXP_EPS)]
                       + [jnp.asarray(scale, jnp.float32)] + [jnp.float32(0)] * 7), band_i, band_f]
    flux_shape = jax.ShapeDtypeStruct((npad, nlay + 1), jnp.float32)
    result = pl.pallas_call(lambda *r: _kernel_sums(*r, nlay=nlay, ng=ng),
        grid=(npad // TX_SUMS,), out_shape=[flux_shape] * 4, interpret=interpret,
        name="rrtmg_lw_band_sums")(*args)
    fields = tuple(x[:ncol].reshape(leading + (nlay + 1,)).astype(tau.dtype) for x in result)
    if not with_clear_sky:
        return fields[:2]
    return fields
