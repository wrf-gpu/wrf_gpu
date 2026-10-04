"""Large-step u/v face tendencies (PGF + Coriolis + curvature) as one REAL stencil per stagger.

Same expressions and association as dynamics/core/rk_addtend_dry.py
large_step_horizontal_pgf / large_step_coriolis / large_step_horizontal_curvature on
the native REAL path (non-hydrostatic, specified/nested): edge-padded face pairs,
the dpn face extrapolation (cf1..cf3 bottom, cfn/cfn1 top under the rigid lid),
coupled ru/rv/rw quads, ownership masks. The kernel adds the three terms to the
incoming u_t/v_t in the caller's order (pgf, coriolis, curvature), replacing the
padded/shifted XLA glue. Flag GPUWRF_DYN_GLUE_FUSED.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pt

F = jnp.float32
RERADIUS = 1.0 / 6370.0e3


def _index(block, n0, n1, n2):
    flat = pl.program_id(0) * block + jnp.arange(block, dtype=jnp.int32)
    return flat // (n1 * n2), flat // n2 % n1, flat % n2, flat < n0 * n1 * n2


def _dpn(psum, coef, fnm, fnp, face, nz, top_lid, valid):
    """_dpn_faces at ``face`` from pair sums psum(level)."""
    cf1, cf2, cf3, cfn, cfn1 = (pt.load(coef.at[i]) for i in range(5))
    vec = lambda ref, kk: pt.load(ref.at[jnp.clip(kk, 0, nz - 1)], mask=valid, other=F(0))
    bottom = F(.5) * (cf1 * psum(jnp.zeros_like(face)) + cf2 * psum(jnp.ones_like(face)) + cf3 * psum(jnp.full_like(face, 2)))
    interior = F(.5) * (vec(fnm, face) * psum(face) + vec(fnp, face) * psum(face - 1))
    top = F(.5) * (cfn * psum(jnp.full_like(face, nz - 1)) + cfn1 * psum(jnp.full_like(face, nz - 2))) if top_lid else F(0)
    return jnp.where(face == 0, bottom, jnp.where(face == nz, top, interior))


def _u_kernel(ut, u, v, w, ph, p, pb, al, alt, php, cq, mub, mup, mus, msfux, msfuy, msfvx, msfty,
              f, e, cosa, c1h, c2h, c1f, c2f, rdnw, fnm, fnp, coef, *outs,
              nz, ny, nx, rdx, rdy, top_lid, curvature, split, block):
    k, j, i, valid = _index(block, nz, ny, nx + 1)
    il, ir = jnp.clip(i - 1, 0, nx - 1), jnp.clip(i, 0, nx - 1)
    m3 = lambda ref, kk, jj, ii: pt.load(ref.at[jnp.clip(kk, 0, nz - 1), jj, ii], mask=valid, other=F(0))
    f3 = lambda ref, kk, jj, ii: pt.load(ref.at[jnp.clip(kk, 0, nz), jj, ii], mask=valid, other=F(0))
    m2 = lambda ref, jj, ii: pt.load(ref.at[jj, ii], mask=valid, other=F(0))
    vk = lambda ref, kk: pt.load(ref.at[jnp.clip(kk, 0, nz - 1)], mask=valid, other=F(0))
    vf = lambda ref, kk: pt.load(ref.at[jnp.clip(kk, 0, nz)], mask=valid, other=F(0))
    c1, c2 = vk(c1h, k), vk(c2h, k)
    msf_u = m2(msfux, j, i) / m2(msfuy, j, i)
    # --- PGF x ---
    ph_term = (f3(ph, k + 1, j, ir) - f3(ph, k + 1, j, il)) + (f3(ph, k, j, ir) - f3(ph, k, j, il))
    p_term = (m3(alt, k, j, il) + m3(alt, k, j, ir)) * (m3(p, k, j, ir) - m3(p, k, j, il))
    pb_term = (m3(al, k, j, il) + m3(al, k, j, ir)) * (m3(pb, k, j, ir) - m3(pb, k, j, il))
    muu = F(.5) * ((m2(mub, j, il) + m2(mup, j, il)) + (m2(mub, j, ir) + m2(mup, j, ir)))
    mass_u = c1 * muu + c2
    dpx = msf_u * F(.5) * F(rdx) * mass_u * (ph_term + p_term + pb_term)
    psum = lambda kk: m3(p, kk, j, il) + m3(p, kk, j, ir)
    dpn = lambda face: _dpn(psum, coef, fnm, fnp, face, nz, top_lid, valid)
    bracket = vk(rdnw, k) * (dpn(k + 1) - dpn(k)) - F(.5) * (c1 * (m2(mup, j, il) + m2(mup, j, ir)))
    dpx = dpx + msf_u * F(rdx) * (m3(php, k, j, ir) - m3(php, k, j, il)) * bracket
    if split:  # moisture-coupled cq stays in XLA: dp, cor, curv are separate outputs
        pt.store(outs[0].at[k, j, i], dpx, mask=valid)
    else:
        result = pt.load(ut.at[k, j, i], mask=valid, other=F(0))
        result = result + -pt.load(cq.at[k, j, i], mask=valid, other=F(0)) * dpx
    # --- Coriolis (u) ---
    muv = lambda jj, ii: F(.5) * (m2(mus, jnp.clip(jj - 1, 0, ny - 1), ii) + m2(mus, jnp.clip(jj, 0, ny - 1), ii))
    rv = lambda jj, ii: pt.load(v.at[k, jj, ii], mask=valid, other=F(0)) * (c1 * muv(jj, ii) + c2) / m2(msfvx, jj, ii)
    rv_quad = F(.25) * (((rv(j, il) + rv(j, ir)) + rv(j + 1, il)) + rv(j + 1, ir))
    rw = lambda kk, ii: f3(w, kk, j, ii) * (vf(c1f, kk) * m2(mus, j, ii) + vf(c2f, kk)) / m2(msfty, j, ii)
    rw_quad = F(.25) * (((rw(k, il) + rw(k + 1, il)) + rw(k, ir)) + rw(k + 1, ir))
    face_u = lambda ref: F(.5) * (m2(ref, j, il) + m2(ref, j, ir))
    cor = msf_u * face_u(f) * rv_quad
    cor = cor - face_u(e) * face_u(cosa) * rw_quad
    edge = jnp.where((i == 0) | (i == nx), F(0), F(1))
    if split:
        pt.store(outs[1].at[k, j, i], cor * edge, mask=valid)
    else:
        result = result + cor * edge
    # --- curvature (u), owned faces 1..nx-1 ---
    if curvature:
        def vxgm(mm):
            mm = jnp.clip(mm, 0, nx - 1)
            uu = F(.5) * (pt.load(u.at[k, j, mm], mask=valid, other=F(0)) + pt.load(u.at[k, j, mm + 1], mask=valid, other=F(0)))
            vv = F(.5) * (pt.load(v.at[k, j, mm], mask=valid, other=F(0)) + pt.load(v.at[k, j + 1, mm], mask=valid, other=F(0)))
            return (uu * (m2(msfvx, j + 1, mm) - m2(msfvx, j, mm)) * F(rdy)
                    - vv * (m2(msfuy, j, mm + 1) - m2(msfuy, j, mm)) * F(rdx))
        owned = F(.5) * (vxgm(i) + vxgm(i - 1)) * rv_quad - pt.load(u.at[k, j, i], mask=valid, other=F(0)) * F(RERADIUS) * rw_quad
        curv = jnp.where((i >= 1) & (i <= nx - 1), owned, F(0))
    else:
        curv = jnp.zeros_like(rw_quad)
    if split:
        pt.store(outs[2].at[k, j, i], curv, mask=valid)
    else:
        result = result + curv
        pt.store(outs[0].at[k, j, i], result, mask=valid)


def _v_kernel(vt, u, v, w, ph, p, pb, al, alt, php, cq, mub, mup, mus, msfvy, msfvx, msfuy, msfty,
              f, e, sina, c1h, c2h, c1f, c2f, rdnw, fnm, fnp, coef, *outs,
              nz, ny, nx, rdx, rdy, top_lid, curvature, split, block):
    k, j, i, valid = _index(block, nz, ny + 1, nx)
    js, jn = jnp.clip(j - 1, 0, ny - 1), jnp.clip(j, 0, ny - 1)
    m3 = lambda ref, kk, jj, ii: pt.load(ref.at[jnp.clip(kk, 0, nz - 1), jj, ii], mask=valid, other=F(0))
    f3 = lambda ref, kk, jj, ii: pt.load(ref.at[jnp.clip(kk, 0, nz), jj, ii], mask=valid, other=F(0))
    m2 = lambda ref, jj, ii: pt.load(ref.at[jj, ii], mask=valid, other=F(0))
    vk = lambda ref, kk: pt.load(ref.at[jnp.clip(kk, 0, nz - 1)], mask=valid, other=F(0))
    vf = lambda ref, kk: pt.load(ref.at[jnp.clip(kk, 0, nz)], mask=valid, other=F(0))
    c1, c2 = vk(c1h, k), vk(c2h, k)
    msf_v = m2(msfvy, j, i) / pt.load(msfvx.at[j, i], mask=valid, other=F(1))
    # --- PGF y ---
    ph_term = (f3(ph, k + 1, jn, i) - f3(ph, k + 1, js, i)) + (f3(ph, k, jn, i) - f3(ph, k, js, i))
    p_term = (m3(alt, k, js, i) + m3(alt, k, jn, i)) * (m3(p, k, jn, i) - m3(p, k, js, i))
    pb_term = (m3(al, k, js, i) + m3(al, k, jn, i)) * (m3(pb, k, jn, i) - m3(pb, k, js, i))
    muv = F(.5) * ((m2(mub, js, i) + m2(mup, js, i)) + (m2(mub, jn, i) + m2(mup, jn, i)))
    mass_v = c1 * muv + c2
    dpy = msf_v * F(.5) * F(rdy) * mass_v * (ph_term + p_term + pb_term)
    psum = lambda kk: m3(p, kk, js, i) + m3(p, kk, jn, i)
    dpn = lambda face: _dpn(psum, coef, fnm, fnp, face, nz, top_lid, valid)
    bracket = vk(rdnw, k) * (dpn(k + 1) - dpn(k)) - F(.5) * (c1 * (m2(mup, js, i) + m2(mup, jn, i)))
    dpy = dpy + msf_v * F(rdy) * (m3(php, k, jn, i) - m3(php, k, js, i)) * bracket
    if split:  # moisture-coupled cq stays in XLA: dp, cor, curv are separate outputs
        pt.store(outs[0].at[k, j, i], dpy, mask=valid)
    else:
        result = pt.load(vt.at[k, j, i], mask=valid, other=F(0))
        result = result + -pt.load(cq.at[k, j, i], mask=valid, other=F(0)) * dpy
    # --- Coriolis (v) ---
    muu = lambda jj, ii: F(.5) * (m2(mus, jj, jnp.clip(ii - 1, 0, nx - 1)) + m2(mus, jj, jnp.clip(ii, 0, nx - 1)))
    ru = lambda jj, ii: pt.load(u.at[k, jj, ii], mask=valid, other=F(0)) * (c1 * muu(jj, ii) + c2) / m2(msfuy, jj, ii)
    ru_quad = F(.25) * (((ru(js, i) + ru(js, i + 1)) + ru(jn, i)) + ru(jn, i + 1))
    rw = lambda kk, jj: f3(w, kk, jj, i) * (vf(c1f, kk) * m2(mus, jj, i) + vf(c2f, kk)) / m2(msfty, jj, i)
    rw_quad = F(.25) * (((rw(k, js) + rw(k + 1, js)) + rw(k, jn)) + rw(k + 1, jn))
    face_v = lambda ref: F(.5) * (m2(ref, js, i) + m2(ref, jn, i))
    cor = -msf_v * face_v(f) * ru_quad
    cor = cor + msf_v * face_v(e) * face_v(sina) * rw_quad
    edge = jnp.where((j == 0) | (j == ny), F(0), F(1))
    if split:
        pt.store(outs[1].at[k, j, i], cor * edge, mask=valid)
    else:
        result = result + cor * edge
    # --- curvature (v), owned rows 1..ny-1 ---
    if curvature:
        def vxgm(mm):
            mm = jnp.clip(mm, 0, ny - 1)
            uu = F(.5) * (pt.load(u.at[k, mm, i], mask=valid, other=F(0)) + pt.load(u.at[k, mm, i + 1], mask=valid, other=F(0)))
            vv = F(.5) * (pt.load(v.at[k, mm, i], mask=valid, other=F(0)) + pt.load(v.at[k, mm + 1, i], mask=valid, other=F(0)))
            return (uu * (m2(msfvx, mm + 1, i) - m2(msfvx, mm, i)) * F(rdy)
                    - vv * (m2(msfuy, mm, i + 1) - m2(msfuy, mm, i)) * F(rdx))
        owned = (F(-.5) * (vxgm(j) + vxgm(j - 1)) * ru_quad
                 - msf_v * pt.load(v.at[k, j, i], mask=valid, other=F(0)) * F(RERADIUS) * rw_quad)
        curv = jnp.where((j >= 1) & (j <= ny - 1), owned, F(0))
    else:
        curv = jnp.zeros_like(rw_quad)
    if split:
        pt.store(outs[2].at[k, j, i], curv, mask=valid)
    else:
        result = result + curv
        pt.store(outs[0].at[k, j, i], result, mask=valid)


def large_step_uv_fp32(ut, vt, u, v, w, ph, p, pb, al, alt, php, cqu, cqv, mub, mup, mus, metrics, coef,
                       *, rdx, rdy, top_lid, curvature, split=False, interpret=False, block=256, warps=4):
    """Return (u_t + pgf_u + cor_u + curv_u, v_t + pgf_v + cor_v + curv_v), REAL.

    ``split``: the moisture-coupled cq never becomes a Pallas operand; returns
    ((dp_u, cor_u, curv_u), (dp_v, cor_v, curv_v)) with pgf = -cq*dp applied by the caller.
    """
    nz, ny, nx = p.shape
    m = metrics
    if split:  # ut/vt and cqu/cqv are not read in split mode
        ut = vt = cqu = cqv = jnp.zeros((1, 1, 1), jnp.float32)
    common = (m.c1h, m.c2h, m.c1f, m.c2f, m.rdnw, m.fnm, m.fnp, coef)
    u_args = (ut, u, v, w, ph, p, pb, al, alt, php, cqu, mub, mup, mus, m.msfux, m.msfuy, m.msfvx, m.msfty,
              m.f, m.e, m.cosa) + common
    v_args = (vt, u, v, w, ph, p, pb, al, alt, php, cqv, mub, mup, mus, m.msfvy, m.msfvx, m.msfuy, m.msfty,
              m.f, m.e, m.sina) + common
    if any(a.dtype != jnp.float32 for a in u_args + v_args):
        raise TypeError("large_step_uv_fp32 requires WRF REAL fp32 inputs")
    kw = dict(nz=nz, ny=ny, nx=nx, rdx=rdx, rdy=rdy, top_lid=top_lid, curvature=curvature, split=split, block=block)
    params = pt.CompilerParams(num_warps=warps)
    from gpuwrf.kernels.dyn_real_fp32 import pin_rows

    def call(kernel, shape, args, name):
        n = shape[0] * shape[1] * shape[2]
        one = jax.ShapeDtypeStruct(shape, jnp.float32)
        return pl.pallas_call(partial(kernel, **kw), out_shape=(one, one, one) if split else one,
                              grid=((n + block - 1) // block,), name=name, interpret=interpret,
                              compiler_params=params)(*pin_rows(args))
    return (call(_u_kernel, (nz, ny, nx + 1), u_args, "b_diff_large_step_u_fp32"),
            call(_v_kernel, (nz, ny + 1, nx), v_args, "b_diff_large_step_v_fp32"))
