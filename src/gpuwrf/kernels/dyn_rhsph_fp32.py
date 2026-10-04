"""WRF rhs_ph (specified, order <= 6, non-hydrostatic, rigid lid) as one REAL stencil.

Same expressions as dynamics/core/rhs_ph.rhs_ph_wrf on the GPUWRF_DYN_REAL_ALL
path: WRF's literal ph/phb association (E133), vertical term 3 from the destaggered
omega, the gw term, and the order-6/4/2 tiered horizontal advection with the
specified row/column map (columns 2 and nx-3 get no x advection), accumulated
(-term3) + gw - adv per face; the lid face kde is 0. Flag GPUWRF_DYN_GLUE_FUSED (rhsph).
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pt

F = jnp.float32


def _kernel(u, v, ww, ph, phb, w, mut, muu, muv, c1f, c2f, fnm, fnp, rdnw, msfty, msfux, msfvy, out,
            *, nz, ny, nx, rdx, rdy, g, block):
    flat = pl.program_id(0) * block + jnp.arange(block, dtype=jnp.int32)
    k, j, i = flat // (ny * nx), flat // nx % ny, flat % nx
    valid = flat < (nz + 1) * ny * nx
    interior = valid & (k >= 1) & (k <= nz - 1)
    zero = F(0)
    vec = lambda ref, kk: pt.load(ref.at[jnp.clip(kk, 0, nz - 1)], mask=valid, other=zero)
    vecf = lambda ref, kk: pt.load(ref.at[jnp.clip(kk, 0, nz)], mask=valid, other=zero)
    m2 = lambda ref, jj, ii: pt.load(ref.at[jj, ii], mask=valid, other=zero)

    def face(ref, kk, jj, ii):  # zero outside the domain (zero-padded stencils, never selected)
        ok = interior & (jj >= 0) & (jj < ny) & (ii >= 0) & (ii < nx)
        return pt.load(ref.at[jnp.clip(kk, 0, nz), jnp.clip(jj, 0, ny - 1), jnp.clip(ii, 0, nx - 1)], mask=ok, other=zero)

    # --- term 3: -(fnm(k)*wdwn(k) + fnp(k)*wdwn(k-1)), wdwn on mass levels ---
    def wdwn(m):
        dphi = ((face(ph, m + 1, j, i) - face(ph, m, j, i)) + face(phb, m + 1, j, i)) - face(phb, m, j, i)
        dest = F(.5) * (face(ww, m + 1, j, i) + face(ww, m, j, i))
        return dest * vec(rdnw, m) * dphi
    term3 = vec(fnm, k) * wdwn(k) + vec(fnp, k) * wdwn(k - 1)
    result = -term3
    # --- term 4: (c1f*mut + c2f) * g * w / msfty ---
    mass_f = vecf(c1f, k) * m2(mut, j, i) + vecf(c2f, k)
    result = result + mass_f * F(g) * face(w, k, j, i) * (F(1) / m2(msfty, j, i))
    # --- terms 1, 2: horizontal advection (specified order <= 6) ---
    c1, c2 = vecf(c1f, k), vecf(c2f, k)

    def flow_y(jj):  # v-face jj (0..ny)
        vp = pt.load(v.at[k, jj, i], mask=interior, other=zero) + pt.load(v.at[jnp.clip(k - 1, 0, nz - 1), jj, i], mask=interior, other=zero)
        return (c1 * m2(muv, jj, i) + c2) * vp * m2(msfvy, jj, i)

    def flow_x(ii):  # u-face ii (0..nx)
        up = pt.load(u.at[k, j, ii], mask=interior, other=zero) + pt.load(u.at[jnp.clip(k - 1, 0, nz - 1), j, ii], mask=interior, other=zero)
        return (c1 * m2(muu, j, ii) + c2) * up * m2(msfux, j, ii)

    def d(ref, off, axis):
        return face(ref, k, j + off, i) if axis == 1 else face(ref, k, j, i + off)

    def diffs(axis):
        a = lambda o: d(ph, o, axis)
        b = lambda o: d(phb, o, axis)
        d1, d2, d3 = a(1) - a(-1), a(2) - a(-2), a(3) - a(-3)
        b1, b2, b3 = b(1) - b(-1), b(2) - b(-2), b(3) - b(-3)
        sten6 = (F(45) * d1 - F(9) * d2 + d3 + F(45) * b1 - F(9) * b2 + b3) / F(60)
        sten4 = (F(8) * d1 - d2 + F(8) * b1 - b2) / F(12)
        dn = (b(1) - b(0)) + a(1) - a(0)
        ds = (b(0) - b(-1)) + a(0) - a(-1)
        return sten6, sten4, dn, ds

    s6, s4, dn, ds = diffs(1)
    fn, fs = flow_y(j + 1), flow_y(j)
    adv_y = jnp.where((j >= 3) & (j <= ny - 4), (fn + fs) * s6, zero)
    adv_y = adv_y + jnp.where((j == 2) | (j == ny - 3), (fn + fs) * s4, zero)
    adv_y = adv_y + jnp.where((j == 1) | (j == ny - 2), fn * dn + fs * ds, zero)
    s6x, _, de, dw = diffs(2)
    fe, fw = flow_x(i + 1), flow_x(i)
    adv_x = jnp.where((i >= 3) & (i <= nx - 4), (fe + fw) * s6x, zero)
    adv_x = adv_x + jnp.where((i == 1) | (i == nx - 2), fe * de + fw * dw, zero)
    mt = m2(msfty, j, i)
    adv = F(0.25 * rdy) / mt * adv_y + F(0.25 * rdx) / mt * adv_x
    result = result + -adv
    pt.store(out.at[k, j, i], jnp.where(interior, result, zero), mask=valid)


def rhs_ph_fp32(u, v, ww, ph, phb, w, mut, muu, muv, c1f, c2f, fnm, fnp, rdnw, msfty, msfux, msfvy,
                rdx, rdy, g, *, interpret=False, block=256, warps=4):
    nzp1, ny, nx = ph.shape
    arrays = (u, v, ww, ph, phb, w, mut, muu, muv, c1f, c2f, fnm, fnp, rdnw, msfty, msfux, msfvy)
    if any(a.dtype != jnp.float32 for a in arrays):
        raise TypeError("rhs_ph_fp32 requires WRF REAL fp32 inputs")
    from gpuwrf.kernels.dyn_real_fp32 import pin_rows
    kernel = partial(_kernel, nz=nzp1 - 1, ny=ny, nx=nx, rdx=rdx, rdy=rdy, g=g, block=block)
    return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(ph.shape, jnp.float32),
                          grid=((nzp1 * ny * nx + block - 1) // block,), name="b_diff_rhs_ph_fp32",
                          interpret=interpret, compiler_params=pt.CompilerParams(num_warps=warps))(*pin_rows(arrays))
