"""WRF h5/v3 SPECIFIED momentum advection (advect_u/v/w) as one REAL stencil per field.

Same arithmetic as the XLA specified paths in dynamics/flux_advection.py
(advect_u_flux / advect_v_flux / advect_w_flux under GPUWRF_DYN_ADVECTION_FP32):
the transporting velocity at every face, the tiered order-5 faces with zero-fill
neighbours (specified_flux_faces, upstream rule on the normal component), the
masked divergences, the order-3 vertical flux and the face/row ownership masks
are evaluated inside the kernel instead of as shifted/padded full arrays.
Accumulation order x, y, z (as the XLA path). Flag GPUWRF_DYN_GLUE_FUSED.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pt

F = jnp.float32


def _tiered(qm3, qm2, qm1, qc, qp1, qp2, v, m, extent, upstream):
    """specified_flux_faces value at face m (between cells m-1, m) times v."""
    sign = jnp.where(v >= F(0), F(1), F(-1))
    f6 = (F(37) * (qc + qm1) - F(8) * (qp1 + qm2) + (qp2 + qm3)) / F(60)
    f5 = f6 - sign * ((qp2 - qm3) - F(5) * (qp1 - qm2) + F(10) * (qc - qm1)) / F(60)
    f4 = (F(7) * (qc + qm1) - (qp1 + qm2)) / F(12)
    f3 = f4 + sign * ((qp1 - qm2) - F(3) * (qc - qm1)) / F(12)
    f2 = F(.5) * (qc + qm1)
    if upstream:
        low = F(.5) * (qc + jnp.where(qc < F(0), qc, qm1))
        high = F(.5) * (qm1 + jnp.where(qm1 > F(0), qm1, qc))
        f2 = jnp.where(m == 1, low, jnp.where(m == extent - 1, high, f2))
    value = jnp.where((m >= 3) & (m <= extent - 3), f5,
                      jnp.where((m == 2) | (m == extent - 2), f3,
                                jnp.where((m == 1) | (m == extent - 1), f2, F(0))))
    return v * value


def _flux3(qm2, qm1, qc, qp1, v):
    """flux3(q, -v) * v as in _vertical_flux_div_3 (sign(-v); v == 0 gives 0)."""
    sign = jnp.where(v <= F(0), F(1), F(-1))
    f4 = (F(7) * (qc + qm1) - (qp1 + qm2)) / F(12)
    return v * (f4 + sign * ((qp1 - qm2) - F(3) * (qc - qm1)) / F(12))


def _index(block, n0, n1, n2):
    flat = pl.program_id(0) * block + jnp.arange(block, dtype=jnp.int32)
    return flat // (n1 * n2), flat // n2 % n1, flat % n2, flat < n0 * n1 * n2


def _loader(valid, dims):
    def load(ref, zz, yy, xx):
        ok = valid & (zz >= 0) & (zz < dims[0]) & (yy >= 0) & (yy < dims[1]) & (xx >= 0) & (xx < dims[2])
        return pt.load(ref.at[zz, yy, xx], mask=ok, other=F(0))
    return load


def _vertical3(load_q, load_vel, k, nz, fzm, fzp, valid):
    """-[vflux(k+1)-vflux(k)] of _vertical_flux_div_3 without the rdzw factor."""
    q = [load_q(k + d) for d in (-2, -1, 0, 1, 2)]

    def vflux(shift):
        face = k + shift
        v = load_vel(face)
        inside = valid & (face > 0) & (face < nz)
        wm = pt.load(fzm.at[face], mask=inside, other=F(0))
        wp = pt.load(fzp.at[face], mask=inside, other=F(0))
        qm2, qm1, qc, qp1 = q[shift:shift + 4]
        second = v * (wm * qc + wp * qm1)
        return jnp.where((face >= 2) & (face <= nz - 2), _flux3(qm2, qm1, qc, qp1, v),
                         jnp.where((face == 1) | (face == nz - 1), second, F(0)))
    return vflux(1) - vflux(0)


def _u_kernel(u, ru, rv, rom, msf, rdzw, fzm, fzp, out, *, nz, ny, nxs, rdx, rdy, block):
    k, y, x, valid = _index(block, nz, ny, nxs)
    load = _loader(valid, (nz, ny, nxs))
    load_rv = _loader(valid, (nz, ny + 1, nxs - 1))
    load_rom = _loader(valid, (nz + 1, ny, nxs - 1))
    uq = lambda dk=0, dy=0, dx=0: load(u, k + dk, y + dy, x + dx)
    qx = [uq(dx=d) for d in range(-3, 4)]
    qy = [uq(dy=d) for d in range(-3, 4)]

    def xface(s):
        m = x + s
        v = F(.5) * (load(ru, k, y, m) + load(ru, k, y, m - 1))
        return _tiered(*qx[s:s + 6], jnp.where(m < nxs, v, F(0)), m, nxs, True)

    def yface(s):
        m = y + s
        v = F(.5) * (load_rv(rv, k, m, x) + load_rv(rv, k, m, x - 1))
        return _tiered(*qy[s:s + 6], jnp.where(m < ny, v, F(0)), m, ny, False)

    map_value = pt.load(msf.at[y, x], mask=valid, other=F(0))
    divx = jnp.where((x >= 1) & (x <= nxs - 2), xface(1) - xface(0), F(0))
    divy = jnp.where((y >= 1) & (y <= ny - 2), yface(1) - yface(0), F(0))
    tend = -map_value * F(rdx) * divx
    tend = tend - map_value * F(rdy) * divy
    romv = lambda face: F(.5) * (load_rom(rom, face, y, x) + load_rom(rom, face, y, x - 1))
    inverse_dz = pt.load(rdzw.at[k], mask=valid, other=F(0))
    tend = tend + -inverse_dz * _vertical3(lambda kk: load(u, kk, y, x), romv, k, nz, fzm, fzp, valid)
    pt.store(out.at[k, y, x], jnp.where((x >= 1) & (x <= nxs - 2), tend, F(0)), mask=valid)


def _v_kernel(v, ru, rv, rom, msfy, msfx, rdzw, fzm, fzp, out, *, nz, nys, nx, rdx, rdy, block):
    k, y, x, valid = _index(block, nz, nys, nx)
    load = _loader(valid, (nz, nys, nx))
    load_ru = _loader(valid, (nz, nys - 1, nx + 1))
    load_rom = _loader(valid, (nz + 1, nys - 1, nx))
    vq = lambda dk=0, dy=0, dx=0: load(v, k + dk, y + dy, x + dx)
    qx = [vq(dx=d) for d in range(-3, 4)]
    qy = [vq(dy=d) for d in range(-3, 4)]

    def xface(s):
        m = x + s
        vel = F(.5) * (load_ru(ru, k, y, m) + load_ru(ru, k, y - 1, m))
        return _tiered(*qx[s:s + 6], jnp.where(m < nx, vel, F(0)), m, nx, False)

    def yface(s):
        m = y + s
        vel = F(.5) * (load(rv, k, m, x) + load(rv, k, m - 1, x))
        return _tiered(*qy[s:s + 6], jnp.where(m < nys, vel, F(0)), m, nys, True)

    my = pt.load(msfy.at[y, x], mask=valid, other=F(0))
    mx = pt.load(msfx.at[y, x], mask=valid, other=F(1))
    divx = jnp.where((x >= 1) & (x <= nx - 2), xface(1) - xface(0), F(0))
    divy = jnp.where((y >= 1) & (y <= nys - 2), yface(1) - yface(0), F(0))
    tend = -my * F(rdx) * divx
    tend = tend - my * F(rdy) * divy
    romv = lambda face: F(.5) * (load_rom(rom, face, y, x) + load_rom(rom, face, y - 1, x))
    inverse_dz = pt.load(rdzw.at[k], mask=valid, other=F(0))
    tend = tend + (my / mx) * (-inverse_dz * _vertical3(lambda kk: load(v, kk, y, x), romv, k, nz, fzm, fzp, valid))
    pt.store(out.at[k, y, x], jnp.where((y >= 1) & (y <= nys - 2), tend, F(0)), mask=valid)


def _w_kernel(w, ru, rv, rom, msf, rdn, fzm, fzp, out, *, nzp1, ny, nx, rdx, rdy, block):
    nz = nzp1 - 1
    k, y, x, valid = _index(block, nzp1, ny, nx)
    load = _loader(valid, (nzp1, ny, nx))
    load_m = _loader(valid, (nz, ny, nx))
    wq = lambda dk=0, dy=0, dx=0: load(w, k + dk, y + dy, x + dx)
    qx = [wq(dx=d) for d in range(-3, 4)]
    qy = [wq(dy=d) for d in range(-3, 4)]
    vecload = lambda ref, kk: pt.load(ref.at[jnp.clip(kk, 0, nz - 1)], mask=valid, other=F(0))
    wm, wp = vecload(fzm, k), vecload(fzp, k)
    wm_top, wp_top = vecload(fzm, jnp.full_like(k, nz - 1)), vecload(fzp, jnp.full_like(k, nz - 1))

    def full_level(ref, yy, xx):
        """_mass_to_full_levels(..., extrapolate_top=True) at face k."""
        lo, hi = load_m(ref, k - 1, yy, xx), load_m(ref, k, yy, xx)
        top_lo, top_hi = load_m(ref, k - 2, yy, xx), load_m(ref, k - 1, yy, xx)
        interior = wm * hi + wp * lo
        top = (F(2) - wm_top) * top_hi - wp_top * top_lo
        return jnp.where(k == 0, hi, jnp.where(k == nz, top, interior))

    def xface(s):
        m = x + s
        return _tiered(*qx[s:s + 6], jnp.where(m < nx, full_level(ru, y, m), F(0)), m, nx, False)

    def yface(s):
        m = y + s
        return _tiered(*qy[s:s + 6], jnp.where(m < ny, full_level(rv, m, x), F(0)), m, ny, False)

    map_value = pt.load(msf.at[y, x], mask=valid, other=F(0))
    divx = jnp.where((x >= 1) & (x <= nx - 2), xface(1) - xface(0), F(0))
    divy = jnp.where((y >= 1) & (y <= ny - 2), yface(1) - yface(0), F(0))
    tend = -map_value * F(rdx) * divx
    tend = tend - map_value * F(rdy) * divy
    # _vertical_flux_div_w(top_lid=False): vflux(f) at faces f = k, k+1.
    zq = [wq(dk=d) for d in (-2, -1, 0, 1, 2)]  # w(k-2..k+2); face k+s reads zq[s:s+4] = w(f-2..f+1)

    def vflux(shift):
        f = k + shift
        rf = lambda ff: _loader(valid, (nzp1, ny, nx))(rom, ff, y, x)
        vel = F(.5) * (rf(f) + rf(f - 1))
        qm2, qm1, qc, qp1 = zq[shift:shift + 4]  # w(f-2..f+1)
        second = vel * (F(.5) * (qc + qm1))
        sign = jnp.where(-vel > F(0), F(1), jnp.where(-vel < F(0), F(-1), F(0)))
        f4 = (F(7) * (qc + qm1) - (qp1 + qm2)) / F(12)
        third = vel * (f4 + sign * ((qp1 - qm2) - F(3) * (qc - qm1)) / F(12))
        return jnp.where((f >= 2) & (f <= nz - 1), third, jnp.where((f == 1) | (f == nz), second, F(0)))

    inverse_dn = vecload(rdn, k)
    vtop = vflux(0)  # vflux(k) for the lid pickup when k == nz
    vert = jnp.where((k >= 1) & (k <= nz - 1), -inverse_dn * (vflux(1) - vtop),
                     jnp.where(k == nz, F(2) * vecload(rdn, jnp.full_like(k, nz - 1)) * vtop, F(0)))
    tend = tend + vert
    pt.store(out.at[k, y, x], jnp.where(k == 0, F(0), tend), mask=valid)


def _call(kernel, out_shape, arrays, name, block, warps, interpret):
    if any(a.dtype != jnp.float32 for a in arrays):
        raise TypeError(name + " requires WRF REAL fp32 inputs")
    n = 1
    for d in out_shape:
        n *= d
    from gpuwrf.kernels.dyn_real_fp32 import pin_rows
    return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(out_shape, jnp.float32),
        grid=((n + block - 1) // block,), name=name, interpret=interpret,
        compiler_params=pt.CompilerParams(num_warps=warps))(*pin_rows(arrays))


def advect_u_fp32(u, ru_full, rv_full, rom, msfux_f, rdzw, fzm, fzp, rdx, rdy, *, interpret=False, block=256, warps=4):
    nz, ny, nxs = u.shape
    kernel = partial(_u_kernel, nz=nz, ny=ny, nxs=nxs, rdx=rdx, rdy=rdy, block=block)
    return _call(kernel, u.shape, (u, ru_full, rv_full, rom, msfux_f, rdzw, fzm, fzp), "b_diff_advect_u_fp32", block, warps, interpret)


def advect_v_fp32(v, ru_full, rv_full, rom, msfvy_f, msfvx_f, rdzw, fzm, fzp, rdx, rdy, *, interpret=False, block=256, warps=4):
    nz, nys, nx = v.shape
    kernel = partial(_v_kernel, nz=nz, nys=nys, nx=nx, rdx=rdx, rdy=rdy, block=block)
    return _call(kernel, v.shape, (v, ru_full, rv_full, rom, msfvy_f, msfvx_f, rdzw, fzm, fzp), "b_diff_advect_v_fp32", block, warps, interpret)


def advect_w_fp32(w, ru, rv, rom, msftx, rdn, fzm, fzp, rdx, rdy, *, interpret=False, block=256, warps=4):
    nzp1, ny, nx = w.shape
    kernel = partial(_w_kernel, nzp1=nzp1, ny=ny, nx=nx, rdx=rdx, rdy=rdy, block=block)
    return _call(kernel, w.shape, (w, ru, rv, rom, msftx, rdn, fzm, fzp), "b_diff_advect_w_fp32", block, warps, interpret)
