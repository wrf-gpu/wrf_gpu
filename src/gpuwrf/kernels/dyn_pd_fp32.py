"""WRF ``advect_scalar_pd`` (positive-definite scalar advection) in native REAL.

Specified/nested lateral boundaries, h5/v3, ``time_step > 0``, ``tenddec`` false
(module_advect_em.F:6069-7885).  Two launches for ALL species of a batch:

1. ``scale`` per cell: low-order (donor-cell) and antidiffusive face fluxes are
   recomputed from the stencils, ``ph_low`` and ``flux_out`` formed exactly as WRF
   (:7733/:7748) and ``scale = max(0, ph_low/(flux_out+eps))`` where
   ``flux_out > ph_low`` inside WRF's limiter range (cells 2..ide-2 / 2..jde-2);
   1 elsewhere (multiplying a face by 1 is exact).
2. tendency per cell: faces recomputed, each antidiffusive face scaled by its
   DONOR cell (WRF :7764-7772 applies every face at most once, by sign), then
   ``-rdzw*dz`` over the full tile, ``-msftx*rdx*dx`` on cells 2..ide-2 and
   ``-msftx*rdy*dy`` on rows 2..jde-2, in WRF's accumulation order (:7790-7885).

Index map (0-based): mass cell x <-> WRF i=x+1; x face f is the left face of
cell f (WRF face i=f+1).  WRF computes x faces 2..ide-1, i.e. f in [1, nx-1];
f=1/nx-1 second order, f=2/nx-2 third order, else fifth.  Same in y.  Vertical
faces kz in [1, nz-1] (kz=1/nz-1 second order with fzm/fzp, else flux3(-rom)).
Returns the raw WRF tendency (no msfty factor), stacked like the input.
"""
from functools import partial
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pt

_F = np.float32
C3760 = _F(37.0) / _F(60.0)
C215 = _F(2.0) / _F(15.0)
C160 = _F(1.0) / _F(60.0)
C712 = _F(7.0) / _F(12.0)
C112 = _F(1.0) / _F(12.0)
EPS = _F(1.0e-20)


def _c(value):
    return jnp.float32(value)


def _sign(ua):
    # Fortran sign(1., ua) for the time_step>0 branch.
    return jnp.where(ua >= _c(0), _c(1), _c(-1))


def _flux5(qm3, qm2, qm1, qi, qp1, qp2, ua):
    f6 = _c(C3760) * (qi + qm1) - _c(C215) * (qp1 + qm2) + _c(C160) * (qp2 + qm3)
    return f6 - _sign(ua) * _c(C160) * (((qp2 - qm3) - _c(5) * (qp1 - qm2)) + _c(10) * (qi - qm1))


def _flux3(qm2, qm1, qi, qp1, ua):
    f4 = _c(C712) * (qi + qm1) - _c(C112) * (qp1 + qm2)
    return f4 + _sign(ua) * _c(C112) * ((qp1 - qm2) - _c(3) * (qi - qm1))


def _upwind(q_im1, q_i, cr):
    a = jnp.abs(cr)
    return (_c(0.5) * jnp.minimum(_c(1), cr + a)) * q_im1 + (_c(0.5) * jnp.maximum(_c(-1), cr - a)) * q_i


class _Grid:
    """Gather helpers over whole-array refs (b-core dyn_flux_fp32 style)."""

    def __init__(self, r, s, k, y, x, valid, nz, ny, nx):
        self.r, self.s, self.k, self.y, self.x, self.valid = r, s, k, y, x, valid
        self.nz, self.ny, self.nx = nz, ny, nx

    def f4(self, name, dk=0, dy=0, dx=0, other=0.0):
        k, y, x = self.k + dk, self.y + dy, self.x + dx
        m = self.valid & (k >= 0) & (k < self.nz) & (y >= 0) & (y < self.ny) & (x >= 0) & (x < self.nx)
        return pt.load(self.r[name].at[self.s, k, y, x], mask=m, other=_c(other))


    def f2(self, name, dy=0, dx=0):
        y, x = self.y + dy, self.x + dx
        m = self.valid & (y >= 0) & (y < self.ny) & (x >= 0) & (x < self.nx)
        return pt.load(self.r[name].at[y, x], mask=m, other=_c(1))

    def f1(self, name, kk, size):
        m = self.valid & (kk >= 0) & (kk < size)
        return pt.load(self.r[name].at[kk], mask=m, other=_c(1))


def _x_face(g, shift, rdx, dt):
    """(antidiffusive fq, low-order fql) at x face f = x + shift (left face of cell f)."""
    f = g.x + shift
    nx = g.nx
    q = {d: g.f4("q", dx=shift + d) for d in (-3, -2, -1, 0, 1, 2)}
    qo_m1, qo_0 = g.f4("qold", dx=shift - 1), g.f4("qold", dx=shift)
    inside = (f >= 1) & (f <= nx - 1)
    vel = pt.load(g.r["ru"].at[g.k, g.y, f], mask=g.valid & inside, other=_c(0))
    c1, c2 = g.f1("c1", g.k, g.nz), g.f1("c2", g.k, g.nz)
    mx_a, mx_b = g.f2("msfty", dx=shift), g.f2("msfty", dx=shift - 1)
    mut_a, mut_b = g.f2("mut", dx=shift), g.f2("mut", dx=shift - 1)
    dx = (_c(2) / (mx_a + mx_b)) / _c(rdx)
    mu = _c(0.5) * ((c1 * mut_a + c2) + (c1 * mut_b + c2))
    # WRF quirk at i == ids+1 (:6718): vel = ru/mu; cr = vel*dt/dx.
    cr = jnp.where(f == 1, ((vel / mu) * _c(dt)) / dx, ((vel * _c(dt)) / dx) / mu)
    fql = (mu * (dx / _c(dt))) * _upwind(qo_m1, qo_0, cr)
    second = (_c(0.5) * vel) * (q[0] + q[-1])
    third = vel * _flux3(q[-2], q[-1], q[0], q[1], vel)
    fifth = vel * _flux5(q[-3], q[-2], q[-1], q[0], q[1], q[2], vel)
    hi = jnp.where((f == 1) | (f == nx - 1), second,
                   jnp.where((f == 2) | (f == nx - 2), third, fifth))
    fq = hi - fql
    return jnp.where(inside, fq, _c(0)), jnp.where(inside, fql, _c(0))


def _y_face(g, shift, rdy, dt):
    f = g.y + shift
    ny = g.ny
    q = {d: g.f4("q", dy=shift + d) for d in (-3, -2, -1, 0, 1, 2)}
    qo_m1, qo_0 = g.f4("qold", dy=shift - 1), g.f4("qold", dy=shift)
    inside = (f >= 1) & (f <= ny - 1)
    vel = pt.load(g.r["rv"].at[g.k, f, g.x], mask=g.valid & inside, other=_c(0))
    c1, c2 = g.f1("c1", g.k, g.nz), g.f1("c2", g.k, g.nz)
    my_a, my_b = g.f2("msftx", dy=shift), g.f2("msftx", dy=shift - 1)
    mut_a, mut_b = g.f2("mut", dy=shift), g.f2("mut", dy=shift - 1)
    dy = (_c(2) / (my_a + my_b)) / _c(rdy)
    mu = _c(0.5) * ((c1 * mut_a + c2) + (c1 * mut_b + c2))
    cr = ((vel * _c(dt)) / dy) / mu
    fql = (mu * (dy / _c(dt))) * _upwind(qo_m1, qo_0, cr)
    second = (_c(0.5) * vel) * (q[0] + q[-1])
    third = vel * _flux3(q[-2], q[-1], q[0], q[1], vel)
    fifth = vel * _flux5(q[-3], q[-2], q[-1], q[0], q[1], q[2], vel)
    hi = jnp.where((f == 1) | (f == ny - 1), second,
                   jnp.where((f == 2) | (f == ny - 2), third, fifth))
    fq = hi - fql
    return jnp.where(inside, fq, _c(0)), jnp.where(inside, fql, _c(0))


def _z_face(g, shift, dt):
    kz = g.k + shift
    nz = g.nz
    q = {d: g.f4("q", dk=shift + d) for d in (-2, -1, 0, 1)}
    qo_m1, qo_0 = g.f4("qold", dk=shift - 1), g.f4("qold", dk=shift)
    inside = (kz >= 1) & (kz <= nz - 1)
    vel = pt.load(g.r["rom"].at[kz, g.y, g.x], mask=g.valid & inside, other=_c(0))
    rz_a, rz_b = g.f1("rdzw", kz, nz), g.f1("rdzw", kz - 1, nz)
    c1, c2 = g.f1("c1", kz, nz), g.f1("c2", kz, nz)
    mut = g.f2("mut")
    dz = _c(2) / (rz_a + rz_b)
    mu = _c(0.5) * ((c1 * mut + c2) + (c1 * mut + c2))
    cr = ((vel * _c(dt)) / dz) / mu
    fql = (mu * (dz / _c(dt))) * _upwind(qo_m1, qo_0, cr)
    wm, wp = g.f1("fzm", kz, g.r["fzm"].shape[0]), g.f1("fzp", kz, g.r["fzp"].shape[0])
    second = vel * (wm * q[0] + wp * q[-1])
    third = vel * _flux3(q[-2], q[-1], q[0], q[1], -vel)
    hi = jnp.where((kz == 1) | (kz == nz - 1), second, third)
    fq = hi - fql
    return jnp.where(inside, fq, _c(0)), jnp.where(inside, fql, _c(0))


def _index(block, ns, nz, ny, nx):
    flat = pl.program_id(0) * block + jnp.arange(block, dtype=jnp.int32)
    x = flat % nx
    y = flat // nx % ny
    k = flat // (nx * ny) % nz
    s = flat // (nx * ny * nz)
    return s, k, y, x, flat < ns * nz * ny * nx


_IN = ("q", "qold", "ru", "rv", "rom", "mut", "mub", "muold", "msftx", "msfty", "c1", "c2", "rdzw", "fzm", "fzp")


def _scale_kernel(*refs, ns, nz, ny, nx, rdx, rdy, dt, block):
    r = dict(zip(_IN, refs[:-1]))
    out = refs[-1]
    s, k, y, x, valid = _index(block, ns, nz, ny, nx)
    g = _Grid(r, s, k, y, x, valid, nz, ny, nx)
    fx0, fxl0 = _x_face(g, 0, rdx, dt)
    fx1, fxl1 = _x_face(g, 1, rdx, dt)
    fy0, fyl0 = _y_face(g, 0, rdy, dt)
    fy1, fyl1 = _y_face(g, 1, rdy, dt)
    fz0, fzl0 = _z_face(g, 0, dt)
    fz1, fzl1 = _z_face(g, 1, dt)
    c1, c2 = g.f1("c1", k, nz), g.f1("c2", k, nz)
    mx, my = g.f2("msftx"), g.f2("msfty")
    rz = g.f1("rdzw", k, nz)
    zero = _c(0)
    ph_low = ((c1 * g.f2("mub") + c2) + (c1 * g.f2("muold"))) * g.f4("qold") - _c(dt) * (
        (mx * my) * (_c(rdx) * (fxl1 - fxl0) + _c(rdy) * (fyl1 - fyl0)) + (my * rz) * (fzl1 - fzl0))
    flux_out = _c(dt) * (
        (mx * my) * (_c(rdx) * (jnp.maximum(zero, fx1) - jnp.minimum(zero, fx0))
                     + _c(rdy) * (jnp.maximum(zero, fy1) - jnp.minimum(zero, fy0)))
        + (my * rz) * (jnp.minimum(zero, fz1) - jnp.maximum(zero, fz0)))
    in_range = (x >= 1) & (x <= nx - 2) & (y >= 1) & (y <= ny - 2)
    limited = in_range & (flux_out > ph_low)
    scale = jnp.where(limited, jnp.maximum(zero, ph_low / (flux_out + _c(EPS))), _c(1))
    pt.store(out.at[s, k, y, x], scale, mask=valid)


def _tend_kernel(*refs, ns, nz, ny, nx, rdx, rdy, dt, block):
    r = dict(zip(_IN + ("scale",), refs[:-1]))
    out = refs[-1]
    s, k, y, x, valid = _index(block, ns, nz, ny, nx)
    g = _Grid(r, s, k, y, x, valid, nz, ny, nx)

    def sc(dk=0, dy=0, dx=0):
        return g.f4("scale", dk=dk, dy=dy, dx=dx, other=1.0)

    s0 = sc()
    zero = _c(0)

    def lim(fq, s_pos, s_neg):
        return jnp.where(fq > zero, s_pos * fq, jnp.where(fq < zero, s_neg * fq, fq))

    fx0, fxl0 = _x_face(g, 0, rdx, dt)
    fx1, fxl1 = _x_face(g, 1, rdx, dt)
    fy0, fyl0 = _y_face(g, 0, rdy, dt)
    fy1, fyl1 = _y_face(g, 1, rdy, dt)
    fz0, fzl0 = _z_face(g, 0, dt)
    fz1, fzl1 = _z_face(g, 1, dt)
    fx0 = lim(fx0, sc(dx=-1), s0)
    fx1 = lim(fx1, s0, sc(dx=1))
    fy0 = lim(fy0, sc(dy=-1), s0)
    fy1 = lim(fy1, s0, sc(dy=1))
    fz0 = lim(fz0, s0, sc(dk=-1))      # face k:   fq>0 -> cell k,   fq<0 -> cell k-1
    fz1 = lim(fz1, sc(dk=1), s0)       # face k+1: fq>0 -> cell k+1, fq<0 -> cell k
    rz = g.f1("rdzw", k, nz)
    mx = g.f2("msftx")
    tend = zero - rz * (((fz1 - fz0) + fzl1) - fzl0)
    xin = (x >= 1) & (x <= nx - 2)
    tend = jnp.where(xin, tend - mx * (_c(rdx) * (((fx1 - fx0) + fxl1) - fxl0)), tend)
    yin = (y >= 1) & (y <= ny - 2)
    tend = jnp.where(yin, tend - mx * (_c(rdy) * (((fy1 - fy0) + fyl1) - fyl0)), tend)
    pt.store(out.at[s, k, y, x], tend, mask=valid)


# --- species loop inside the program (GPUWRF_DYN_PD_SPECIES_LOOP): lanes over (k, y, x),
# species-independent face geometry/velocity/Courant terms computed once per cell, the
# species loop carries only the q-dependent stencils.  Same expressions and grouping.

class _SGrid(_Grid):
    def q4(self, name, s, dk=0, dy=0, dx=0, other=0.0):
        k, y, x = self.k + dk, self.y + dy, self.x + dx
        m = self.valid & (k >= 0) & (k < self.nz) & (y >= 0) & (y < self.ny) & (x >= 0) & (x < self.nx)
        return pt.load(self.r[name].at[s, k, y, x], mask=m, other=_c(other))


def _x_geom(g, shift, rdx, dt):
    f = g.x + shift
    nx = g.nx
    inside = (f >= 1) & (f <= nx - 1)
    vel = pt.load(g.r["ru"].at[g.k, g.y, f], mask=g.valid & inside, other=_c(0))
    c1, c2 = g.f1("c1", g.k, g.nz), g.f1("c2", g.k, g.nz)
    mx_a, mx_b = g.f2("msfty", dx=shift), g.f2("msfty", dx=shift - 1)
    mut_a, mut_b = g.f2("mut", dx=shift), g.f2("mut", dx=shift - 1)
    dx = (_c(2) / (mx_a + mx_b)) / _c(rdx)
    mu = _c(0.5) * ((c1 * mut_a + c2) + (c1 * mut_b + c2))
    cr = jnp.where(f == 1, ((vel / mu) * _c(dt)) / dx, ((vel * _c(dt)) / dx) / mu)
    return dict(shift=shift, axis="x", inside=inside, vel=vel, cr=cr, wfl=mu * (dx / _c(dt)),
                half=_c(0.5) * vel, sel2=(f == 1) | (f == nx - 1), sel3=(f == 2) | (f == nx - 2))


def _y_geom(g, shift, rdy, dt):
    f = g.y + shift
    ny = g.ny
    inside = (f >= 1) & (f <= ny - 1)
    vel = pt.load(g.r["rv"].at[g.k, f, g.x], mask=g.valid & inside, other=_c(0))
    c1, c2 = g.f1("c1", g.k, g.nz), g.f1("c2", g.k, g.nz)
    my_a, my_b = g.f2("msftx", dy=shift), g.f2("msftx", dy=shift - 1)
    mut_a, mut_b = g.f2("mut", dy=shift), g.f2("mut", dy=shift - 1)
    dy = (_c(2) / (my_a + my_b)) / _c(rdy)
    mu = _c(0.5) * ((c1 * mut_a + c2) + (c1 * mut_b + c2))
    cr = ((vel * _c(dt)) / dy) / mu
    return dict(shift=shift, axis="y", inside=inside, vel=vel, cr=cr, wfl=mu * (dy / _c(dt)),
                half=_c(0.5) * vel, sel2=(f == 1) | (f == ny - 1), sel3=(f == 2) | (f == ny - 2))


def _z_geom(g, shift, dt):
    kz = g.k + shift
    nz = g.nz
    inside = (kz >= 1) & (kz <= nz - 1)
    vel = pt.load(g.r["rom"].at[kz, g.y, g.x], mask=g.valid & inside, other=_c(0))
    rz_a, rz_b = g.f1("rdzw", kz, nz), g.f1("rdzw", kz - 1, nz)
    c1, c2 = g.f1("c1", kz, nz), g.f1("c2", kz, nz)
    mut = g.f2("mut")
    dz = _c(2) / (rz_a + rz_b)
    mu = _c(0.5) * ((c1 * mut + c2) + (c1 * mut + c2))
    cr = ((vel * _c(dt)) / dz) / mu
    wm, wp = g.f1("fzm", kz, g.r["fzm"].shape[0]), g.f1("fzp", kz, g.r["fzp"].shape[0])
    return dict(shift=shift, axis="z", inside=inside, vel=vel, cr=cr, wfl=mu * (dz / _c(dt)),
                wm=wm, wp=wp, sel2=(kz == 1) | (kz == nz - 1))


def _face_s(g, geo, s):
    """(fq, fql) of one face for species s from its precomputed geometry (= _x/_y/_z_face)."""
    sh, axis, vel = geo["shift"], geo["axis"], geo["vel"]
    if axis == "z":
        q = {d: g.q4("q", s, dk=sh + d) for d in (-2, -1, 0, 1)}
        qo_m1, qo_0 = g.q4("qold", s, dk=sh - 1), g.q4("qold", s, dk=sh)
        fql = geo["wfl"] * _upwind(qo_m1, qo_0, geo["cr"])
        second = vel * (geo["wm"] * q[0] + geo["wp"] * q[-1])
        third = vel * _flux3(q[-2], q[-1], q[0], q[1], -vel)
        hi = jnp.where(geo["sel2"], second, third)
    else:
        key = "dx" if axis == "x" else "dy"
        q = {d: g.q4("q", s, **{key: sh + d}) for d in (-3, -2, -1, 0, 1, 2)}
        qo_m1, qo_0 = g.q4("qold", s, **{key: sh - 1}), g.q4("qold", s, **{key: sh})
        fql = geo["wfl"] * _upwind(qo_m1, qo_0, geo["cr"])
        second = geo["half"] * (q[0] + q[-1])
        third = vel * _flux3(q[-2], q[-1], q[0], q[1], vel)
        fifth = vel * _flux5(q[-3], q[-2], q[-1], q[0], q[1], q[2], vel)
        hi = jnp.where(geo["sel2"], second, jnp.where(geo["sel3"], third, fifth))
    fq = hi - fql
    inside = geo["inside"]
    return jnp.where(inside, fq, _c(0)), jnp.where(inside, fql, _c(0))


def _cell_index(block, nz, ny, nx):
    flat = pl.program_id(0) * block + jnp.arange(block, dtype=jnp.int32)
    x = flat % nx
    y = flat // nx % ny
    k = flat // (nx * ny)
    return k, y, x, flat < nz * ny * nx


def _geoms(g, rdx, rdy, dt):
    return (_x_geom(g, 0, rdx, dt), _x_geom(g, 1, rdx, dt), _y_geom(g, 0, rdy, dt),
            _y_geom(g, 1, rdy, dt), _z_geom(g, 0, dt), _z_geom(g, 1, dt))


def _scale_kernel_sl(*refs, ns, nz, ny, nx, rdx, rdy, dt, block):
    r = dict(zip(_IN, refs[:-1]))
    out = refs[-1]
    k, y, x, valid = _cell_index(block, nz, ny, nx)
    g = _SGrid(r, None, k, y, x, valid, nz, ny, nx)
    geo = _geoms(g, rdx, rdy, dt)
    c1, c2 = g.f1("c1", k, nz), g.f1("c2", k, nz)
    mx, my = g.f2("msftx"), g.f2("msfty")
    rz = g.f1("rdzw", k, nz)
    mass = (c1 * g.f2("mub") + c2) + (c1 * g.f2("muold"))
    mxmy, myrz = mx * my, my * rz
    in_range = (x >= 1) & (x <= nx - 2) & (y >= 1) & (y <= ny - 2)
    zero = _c(0)

    def species(s, carry):
        (fx0, fxl0), (fx1, fxl1), (fy0, fyl0), (fy1, fyl1), (fz0, fzl0), (fz1, fzl1) = (
            _face_s(g, gm, s) for gm in geo)
        ph_low = mass * g.q4("qold", s) - _c(dt) * (
            mxmy * (_c(rdx) * (fxl1 - fxl0) + _c(rdy) * (fyl1 - fyl0)) + myrz * (fzl1 - fzl0))
        flux_out = _c(dt) * (
            mxmy * (_c(rdx) * (jnp.maximum(zero, fx1) - jnp.minimum(zero, fx0))
                    + _c(rdy) * (jnp.maximum(zero, fy1) - jnp.minimum(zero, fy0)))
            + myrz * (jnp.minimum(zero, fz1) - jnp.maximum(zero, fz0)))
        limited = in_range & (flux_out > ph_low)
        scale = jnp.where(limited, jnp.maximum(zero, ph_low / (flux_out + _c(EPS))), _c(1))
        pt.store(out.at[s, k, y, x], scale, mask=valid)
        return carry

    jax.lax.fori_loop(0, ns, species, 0)


def _tend_kernel_sl(*refs, ns, nz, ny, nx, rdx, rdy, dt, block):
    r = dict(zip(_IN + ("scale",), refs[:-1]))
    out = refs[-1]
    k, y, x, valid = _cell_index(block, nz, ny, nx)
    g = _SGrid(r, None, k, y, x, valid, nz, ny, nx)
    geo = _geoms(g, rdx, rdy, dt)
    rz = g.f1("rdzw", k, nz)
    mx = g.f2("msftx")
    xin = (x >= 1) & (x <= nx - 2)
    yin = (y >= 1) & (y <= ny - 2)
    zero = _c(0)

    def lim(fq, s_pos, s_neg):
        return jnp.where(fq > zero, s_pos * fq, jnp.where(fq < zero, s_neg * fq, fq))

    def species(s, carry):
        def sc(dk=0, dy=0, dx=0):
            return g.q4("scale", s, dk=dk, dy=dy, dx=dx, other=1.0)
        s0 = sc()
        (fx0, fxl0), (fx1, fxl1), (fy0, fyl0), (fy1, fyl1), (fz0, fzl0), (fz1, fzl1) = (
            _face_s(g, gm, s) for gm in geo)
        fx0 = lim(fx0, sc(dx=-1), s0)
        fx1 = lim(fx1, s0, sc(dx=1))
        fy0 = lim(fy0, sc(dy=-1), s0)
        fy1 = lim(fy1, s0, sc(dy=1))
        fz0 = lim(fz0, s0, sc(dk=-1))
        fz1 = lim(fz1, sc(dk=1), s0)
        tend = zero - rz * (((fz1 - fz0) + fzl1) - fzl0)
        tend = jnp.where(xin, tend - mx * (_c(rdx) * (((fx1 - fx0) + fxl1) - fxl0)), tend)
        tend = jnp.where(yin, tend - mx * (_c(rdy) * (((fy1 - fy0) + fyl1) - fyl0)), tend)
        pt.store(out.at[s, k, y, x], tend, mask=valid)
        return carry

    jax.lax.fori_loop(0, ns, species, 0)


def advect_scalar_pd_fp32(q, qold, ru, rv, rom, mut, muold, c1, c2, msftx, msfty,
                          rdzw, fzm, fzp, *, rdx, rdy, dt, mub=None,
                          interpret=False, block=256, warps=4):
    """Native WRF advect_scalar_pd tendency for stacked species ``(S, nz, ny, nx)``.

    ``ru``: (nz, ny, nx[+1]) x-face coupled u (face f = left face of cell f);
    ``rv``: (nz, ny[+1], nx); ``rom``: (nz+1, ny, nx).  ``mut`` current full dry
    mass; ``muold`` the start-of-step mass -- a perturbation when ``mub`` is given
    (WRF form), else the full old mass (``mub`` taken as 0).
    """
    if q.ndim != 4 or q.shape != qold.shape:
        raise ValueError("advect_scalar_pd_fp32 expects stacked (S, nz, ny, nx) q/qold")
    ns, nz, ny, nx = q.shape
    if mub is None:
        mub = jnp.zeros_like(mut)
    arrays = (q, qold, ru, rv, rom, mut, mub, muold, msftx, msfty, c1, c2, rdzw, fzm, fzp)
    if any(a.dtype != jnp.float32 for a in arrays):
        raise TypeError("advect_scalar_pd_fp32 requires WRF REAL (float32) operands")
    if ru.shape[:2] != (nz, ny) or ru.shape[2] < nx or rv.shape[0] != nz or rv.shape[1] < ny or rv.shape[2] != nx:
        raise ValueError(f"inconsistent face velocities ru{ru.shape} rv{rv.shape}")
    if rom.shape != (nz + 1, ny, nx) or fzm.shape[0] < nz or fzp.shape[0] < nz:
        raise ValueError("rom must be (nz+1, ny, nx); fzm/fzp need >= nz entries (faces 1..nz-1)")
    species_loop = os.environ.get("GPUWRF_DYN_PD_SPECIES_LOOP", "0") == "1"
    total = (nz if species_loop else ns * nz) * ny * nx
    grid = ((total + block - 1) // block,)
    params = dict(ns=ns, nz=nz, ny=ny, nx=nx, rdx=float(rdx), rdy=float(rdy), dt=float(dt), block=block)
    shape = jax.ShapeDtypeStruct(q.shape, jnp.float32)
    cp = pt.CompilerParams(num_warps=warps)
    if species_loop:
        scale = pl.pallas_call(partial(_scale_kernel_sl, **params), out_shape=shape, grid=grid,
                               name="b_core_pd_scale_sl_fp32", interpret=interpret, compiler_params=cp)(*arrays)
        return pl.pallas_call(partial(_tend_kernel_sl, **params), out_shape=shape, grid=grid,
                              name="b_core_pd_tend_sl_fp32", interpret=interpret, compiler_params=cp)(*arrays, scale)
    scale = pl.pallas_call(partial(_scale_kernel, **params), out_shape=shape, grid=grid,
                           name="b_carry_pd_scale_fp32", interpret=interpret, compiler_params=cp)(*arrays)
    return pl.pallas_call(partial(_tend_kernel, **params), out_shape=shape, grid=grid,
                          name="b_carry_pd_tend_fp32", interpret=interpret, compiler_params=cp)(*arrays, scale)
