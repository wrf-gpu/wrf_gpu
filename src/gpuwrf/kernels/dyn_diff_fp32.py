"""Direct WRF REAL sixth-order stencils for specified/nested domains.

Each output reads the immutable input stencil. No roll/packing buffers or
domain-wide level loops. Momentum returns the rk_addtend effective tendency;
scalar returns the forward mass-coupled tendency.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pt


def _sixth(q, mut, c1, c2, mx, my, coef, out, *, name, nz, ny, nx,
           mass_ny, mass_nx, monotonic, block):
    idx = pl.program_id(0) * block + jnp.arange(block, dtype=jnp.int32)
    x, y, k = idx % nx, idx // nx % ny, idx // (nx * ny)
    valid = idx < nz * ny * nx
    owned = valid & (x >= 3) & (x < nx - 3) & (y >= 3) & (y < ny - 3)
    if name == "w":
        owned = owned & (k > 0) & (k < nz - 1)
    zero, half, quarter = jnp.float32(0), jnp.float32(.5), jnp.float32(.25)
    a = pt.load(c1.at[k], mask=valid, other=zero)
    b = pt.load(c2.at[k], mask=valid, other=zero)

    def mass(dx=0, dy=0):
        xx, yy = x + dx, y + dy
        m = pt.load(mut.at[yy, xx], mask=owned & (xx >= 0) &
                    (xx < mass_nx) & (yy >= 0) & (yy < mass_ny), other=zero)
        return a * m + b

    def flux(axis):
        def load(offset):
            xx, yy = (x + offset, y) if axis == 2 else (x, y + offset)
            return pt.load(q.at[k, yy, xx], mask=owned, other=zero)
        m3, m2, m1, qi, p1, p2, p3 = [load(i) for i in range(-3, 4)]
        f0 = jnp.float32(10) * (qi - m1) - jnp.float32(5) * (p1 - m2) + (p2 - m3)
        f1 = jnp.float32(10) * (p1 - qi) - jnp.float32(5) * (p2 - m1) + (p3 - m2)
        if monotonic:
            f0 = jnp.where(f0 * (qi - m1) <= zero, zero, f0)
            f1 = jnp.where(f1 * (p1 - qi) <= zero, zero, f1)
        return f0, f1

    fx0, fx1 = flux(2)
    fy0, fy1 = flux(1)
    center = mass()
    if name == "u":
        x0, x1 = mass(-1), center
        y0 = quarter * (mass(-1, -1) + mass(0, -1) + mass(-1) + center)
        y1 = quarter * (mass(-1) + center + mass(-1, 1) + mass(0, 1))
    elif name == "v":
        y0, y1 = mass(0, -1), center
        x0 = quarter * (mass(-1, -1) + mass(0, -1) + mass(-1) + center)
        x1 = quarter * (mass(0, -1) + mass(1, -1) + center + mass(1))
    else:
        x0, x1 = half * (mass(-1) + center), half * (center + mass(1))
        y0, y1 = half * (mass(0, -1) + center), half * (center + mass(0, 1))
    mapx = pt.load(mx.at[y, x], mask=valid, other=zero)
    mapy = pt.load(my.at[y, x], mask=valid, other=zero)
    coefficient = pt.load(coef.at[()])
    result = coefficient * mapx * (x1 * fx1 - x0 * fx0)
    result = result + coefficient * mapy * (y1 * fy1 - y0 * fy0)
    if name in ("u", "w"):
        result = result / mapy
    elif name == "v":
        result = result * (jnp.float32(1) / mapx)
    pt.store(out.at[k, y, x], jnp.where(owned, result, zero), mask=valid)


def sixth_order_fp32(field, mut, c1, c2, mapx, mapy, *, name, dt,
                     factor, monotonic=True, interpret=False, block=256):
    """WRF slopeopt=0, specified/nested stencil (u/v/w/m staggering)."""
    if name not in ("u", "v", "w", "m"):
        raise ValueError("sixth-order field name must be u/v/w/m")
    arrays = (field, mut, c1, c2, mapx, mapy)
    if any(a.dtype != jnp.float32 for a in arrays):
        raise TypeError("native diffusion requires WRF REAL fp32 arrays")
    nz, ny, nx = field.shape
    expected_mass = (ny - int(name == "v"), nx - int(name == "u"))
    if mut.shape != expected_mass or c1.shape != (nz,) or c2.shape != (nz,):
        raise ValueError("inconsistent mass/staggering/vertical coefficients")
    if mapx.shape != (ny, nx) or mapy.shape != (ny, nx):
        raise ValueError("map factors must have the field's horizontal staggering")
    # WRF forms this coefficient in REAL, not a host float64 expression.
    coef = jnp.asarray(factor, jnp.float32) * jnp.float32(.015625) / (
        jnp.float32(2) * jnp.asarray(dt, jnp.float32))
    kernel = partial(_sixth, name=name, nz=nz, ny=ny, nx=nx,
                     mass_ny=mut.shape[0], mass_nx=mut.shape[1],
                     monotonic=monotonic, block=block)
    return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(field.shape, jnp.float32),
        grid=((nz * ny * nx + block - 1) // block,), name="b_diff_sixth_" + name,
        interpret=interpret, compiler_params=pt.CompilerParams(num_warps=4))(*arrays, coef)


def _horizontal(q, kh, mass, tx, ty, ux, uy, vx, vy, spacing, out, *,
                name, nz, ny, nx, block):
    idx = pl.program_id(0) * block + jnp.arange(block, dtype=jnp.int32)
    x, y, k = idx % nx, idx // nx % ny, idx // (nx * ny)
    valid = idx < nz * ny * nx
    owned = valid & (x > 0) & (x < nx-1) & (y > 0) & (y < ny-1)
    if name == "w":
        owned = owned & (k > 0) & (k < nz-1)
    zero, half, quarter = jnp.float32(0), jnp.float32(.5), jnp.float32(.25)

    def load3(ref, dx=0, dy=0, dk=0):
        return pt.load(ref.at[k+dk, y+dy, x+dx], mask=owned, other=zero)
    def load2(ref, dx=0, dy=0):
        return pt.load(ref.at[y+dy, x+dx], mask=owned, other=jnp.float32(1))
    def quad(ref, offsets):
        # Preserve the WRF four-point association.
        return quarter * sum(load3(ref, dx, dy, dk) for dx,dy,dk in offsets)

    rdx, rdy = pt.load(spacing.at[0]), pt.load(spacing.at[1])
    if name == "u":
        x0 = load2(tx,-1)/load2(ty,-1) * load3(mass,-1) * load3(kh,-1) * rdx
        x1 = load2(tx)/load2(ty) * load3(mass) * load3(kh) * rdx
        south = ((0,0,0),(0,-1,0),(-1,-1,0),(-1,0,0))
        north = ((0,0,0),(0,1,0),(-1,1,0),(-1,0,0))
        y0 = (load2(uy)+load2(uy,dy=-1))/(load2(ux)+load2(ux,dy=-1)) * quad(mass,south) * quad(kh,south) * rdy
        y1 = (load2(uy)+load2(uy,dy=1))/(load2(ux)+load2(ux,dy=1)) * quad(mass,north) * quad(kh,north) * rdy
        maps = load2(ux)*load2(uy)
    elif name == "v":
        west = ((0,0,0),(0,-1,0),(-1,-1,0),(-1,0,0))
        east = ((0,0,0),(0,-1,0),(1,-1,0),(1,0,0))
        x0 = (load2(vx)+load2(vx,-1))/(load2(vy)+load2(vy,-1)) * quad(mass,west) * quad(kh,west) * rdx
        x1 = (load2(vx)+load2(vx,1))/(load2(vy)+load2(vy,1)) * quad(mass,east) * quad(kh,east) * rdx
        # Literal WRF V y-direction has no mass multiplier.
        y0 = load2(ty,dy=-1)/load2(tx,dy=-1) * load3(kh,dy=-1) * rdy
        y1 = load2(ty)/load2(tx) * load3(kh) * rdy
        maps = load2(vx)*load2(vy)
    else:
        if name == "w":
            kx0 = quad(kh,((0,0,0),(-1,0,0),(0,0,-1),(-1,0,-1)))
            kx1 = quad(kh,((1,0,0),(0,0,0),(1,0,-1),(0,0,-1)))
            ky0 = quad(kh,((0,0,0),(0,-1,0),(0,0,-1),(0,-1,-1)))
            ky1 = quad(kh,((0,1,0),(0,0,0),(0,1,-1),(0,0,-1)))
        else:
            kx0, kx1 = half*(load3(kh)+load3(kh,-1)), half*(load3(kh,1)+load3(kh))
            ky0, ky1 = half*(load3(kh)+load3(kh,dy=-1)), half*(load3(kh,dy=1)+load3(kh))
        x0 = load2(ux)/load2(uy) * kx0 * (half*(load3(mass)+load3(mass,-1))) * rdx
        x1 = load2(ux,1)/load2(uy,1) * kx1 * (half*(load3(mass,1)+load3(mass))) * rdx
        y0 = load2(vy)/load2(vx) * ky0 * (half*(load3(mass)+load3(mass,dy=-1))) * rdy
        y1 = load2(vy,dy=1)/load2(vx,dy=1) * ky1 * (half*(load3(mass,dy=1)+load3(mass))) * rdy
        maps = load2(tx)*load2(ty)
    qi = load3(q)
    value = maps * rdx * (x1*(load3(q,1)-qi) - x0*(qi-load3(q,-1)))
    value = value + maps * rdy * (y1*(load3(q,dy=1)-qi) - y0*(qi-load3(q,dy=-1)))
    pt.store(out.at[k,y,x], jnp.where(owned,value,zero), mask=valid)


def horizontal_diffusion_fp32(field, kh, mass, tx, ty, ux, uy, vx, vy, *,
                             name, dx, dy, interpret=False, block=256):
    """Specified/nested WRF horizontal_diffusion(_3dmp) direct stencil."""
    arrays = (field, kh, mass, tx, ty, ux, uy, vx, vy)
    if name not in ("u","v","w","m"):
        raise ValueError("horizontal diffusion field must be u/v/w/m")
    if any(a.dtype != jnp.float32 for a in arrays):
        raise TypeError("native horizontal diffusion requires WRF REAL fp32")
    nz,ny,nx=field.shape
    kernel=partial(_horizontal,name=name,nz=nz,ny=ny,nx=nx,block=block)
    spacing=jnp.asarray([1./dx,1./dy],jnp.float32)
    return pl.pallas_call(kernel,out_shape=jax.ShapeDtypeStruct(field.shape,jnp.float32),
        grid=((nz*ny*nx+block-1)//block,),name="b_diff_horizontal_"+name,
        interpret=interpret,compiler_params=pt.CompilerParams(num_warps=4))(*arrays,spacing)
