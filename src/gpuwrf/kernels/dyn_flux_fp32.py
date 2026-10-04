"""Direct WRF order-5 horizontal face stencil in native REAL precision.

No packing, shifts, concatenation or temporary neighbour arrays. The separate
flux output is required because adjacent threads read the same input cells.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pt


def _face_kernel(q, velocity, out, *, nz, ny, nx, axis, upstream, block):
    flat = pl.program_id(0)*block+jnp.arange(block,dtype=jnp.int32)
    x, y, k = flat%nx, flat//nx%ny, flat//(nx*ny)
    valid = flat<nz*ny*nx
    pos, extent = (x,nx) if axis==2 else (y,ny)

    def load(offset):
        coord = pos+offset
        indices = (k,y,coord) if axis==2 else (k,coord,x)
        return pt.load(q.at[indices],mask=valid&(coord>=0)&(coord<extent),other=0.)
    qm3,qm2,qm1,qi,qp1,qp2 = (load(i) for i in (-3,-2,-1,0,1,2))
    v = pt.load(velocity.at[k,y,x],mask=valid,other=0.)
    sign = jnp.where(v>=0.,jnp.float32(1.),jnp.float32(-1.))
    f6 = (37.*(qi+qm1)-8.*(qp1+qm2)+(qp2+qm3))/60.
    f5 = f6-sign*((qp2-qm3)-5.*(qp1-qm2)+10.*(qi-qm1))/60.
    f4 = (7.*(qi+qm1)-(qp1+qm2))/12.
    f3 = f4+sign*((qp1-qm2)-3.*(qi-qm1))/12.
    f2 = .5*(qi+qm1)
    if upstream:
        low = .5*(qi+jnp.where(qi<0.,qi,qm1))
        high = .5*(qm1+jnp.where(qm1>0.,qm1,qi))
        f2 = jnp.where(pos==1,low,jnp.where(pos==extent-1,high,f2))
    value = jnp.where((pos>=3)&(pos<=extent-3),f5,
                     jnp.where((pos==2)|(pos==extent-2),f3,
                               jnp.where((pos==1)|(pos==extent-1),f2,0.)))
    pt.store(out.at[k,y,x],v*value,mask=valid)


def specified_flux_faces_fp32(field, velocity, axis, *, upstream=False,
                             interpret=False, block=256, warps=4):
    if field.dtype!=jnp.float32 or velocity.dtype!=jnp.float32:
        raise TypeError("native flux requires fp32 field and velocity")
    if field.ndim!=3 or field.shape!=velocity.shape or axis not in (1,2):
        raise ValueError("horizontal flux requires matching (z,y,x) arrays and axis1/2")
    nz,ny,nx = field.shape
    kernel = partial(_face_kernel,nz=nz,ny=ny,nx=nx,axis=axis,
                     upstream=upstream,block=block)
    return pl.pallas_call(kernel,out_shape=jax.ShapeDtypeStruct(field.shape,jnp.float32),
        grid=((nz*ny*nx+block-1)//block,),name="b_core_flux_face_fp32",
        interpret=interpret,compiler_params=pt.CompilerParams(num_warps=warps))(field,velocity)


def _scalar_kernel(q, ru, rv, rom, map_factor, rdzw, fzm, fzp, out,
                   *, nz, ny, nx, rdx, rdy, block):
    flat = pl.program_id(0) * block + jnp.arange(block, dtype=jnp.int32)
    x, y, k = flat % nx, flat // nx % ny, flat // (nx * ny)
    valid = flat < nz * ny * nx
    zero, one, minus_one = jnp.float32(0), jnp.float32(1), jnp.float32(-1)

    def qload(dk=0, dy=0, dx=0):
        zz, yy, xx = k + dk, y + dy, x + dx
        return pt.load(q.at[zz, yy, xx],
                       mask=valid & (zz >= 0) & (zz < nz) &
                       (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx), other=zero)

    qi = qload()

    def horizontal(axis, velocity):
        pos, extent = (x, nx) if axis == 2 else (y, ny)
        values = [qload(dx=i) if axis == 2 else qload(dy=i)
                  for i in (-3, -2, -1)] + [qi] + [
                  qload(dx=i) if axis == 2 else qload(dy=i) for i in (1, 2, 3)]

        def face(shift):
            m = pos + shift
            xx, yy = (x + shift, y) if axis == 2 else (x, y + shift)
            v = pt.load(velocity.at[k, yy, xx], mask=valid & (m < extent), other=zero)
            qm3, qm2, qm1, qc, qp1, qp2 = values[shift:shift + 6]
            sign = jnp.where(v >= zero, one, minus_one)
            f6 = (jnp.float32(37) * (qc + qm1) - jnp.float32(8) * (qp1 + qm2) + (qp2 + qm3)) / jnp.float32(60)
            f5 = f6 - sign * ((qp2 - qm3) - jnp.float32(5) * (qp1 - qm2) + jnp.float32(10) * (qc - qm1)) / jnp.float32(60)
            f4 = (jnp.float32(7) * (qc + qm1) - (qp1 + qm2)) / jnp.float32(12)
            f3 = f4 + sign * ((qp1 - qm2) - jnp.float32(3) * (qc - qm1)) / jnp.float32(12)
            f2 = jnp.float32(.5) * (qc + qm1)
            value = jnp.where((m >= 3) & (m <= extent - 3), f5,
                             jnp.where((m == 2) | (m == extent - 2), f3,
                                       jnp.where((m == 1) | (m == extent - 1), f2, zero)))
            return v * value

        difference = face(1) - face(0)
        return jnp.where((pos >= 1) & (pos <= extent - 2), difference, zero)

    zvalues = [qload(dk=-2), qload(dk=-1), qi, qload(dk=1), qload(dk=2)]

    def vertical(shift):
        face = k + shift
        v = pt.load(rom.at[face, y, x], mask=valid, other=zero)
        qm2, qm1, qc, qp1 = zvalues[shift:shift + 4]
        sign = jnp.where(v <= zero, one, minus_one)  # WRF flux3(..., -rom)
        f4 = (jnp.float32(7) * (qc + qm1) - (qp1 + qm2)) / jnp.float32(12)
        f3 = f4 + sign * ((qp1 - qm2) - jnp.float32(3) * (qc - qm1)) / jnp.float32(12)
        inside = valid & (face > 0) & (face < nz)
        wm = pt.load(fzm.at[face], mask=inside, other=zero)
        wp = pt.load(fzp.at[face], mask=inside, other=zero)
        f2 = wm * qc + wp * qm1
        value = jnp.where((face >= 2) & (face <= nz - 2), f3,
                          jnp.where((face == 1) | (face == nz - 1), f2, zero))
        return v * value

    map_value = pt.load(map_factor.at[y, x], mask=valid, other=zero)
    inverse_dz = pt.load(rdzw.at[k], mask=valid, other=zero)
    # Preserve WRF's y, x, z accumulation order. Outputs cannot alias neighbour input.
    result = -map_value * jnp.float32(rdy) * horizontal(1, rv)
    result = result - map_value * jnp.float32(rdx) * horizontal(2, ru)
    result = result - inverse_dz * (vertical(1) - vertical(0))
    pt.store(out.at[k, y, x], result, mask=valid)


def advect_scalar_flux_fp32(field, ru, rv, rom, map_factor, rdzw, fzm, fzp,
                            rdx, rdy, *, interpret=False, block=256, warps=4):
    """One direct stencil for pristine WRF h5/v3 specified scalar tendency."""
    nz, ny, nx = field.shape
    arrays = (field, ru, rv, rom, map_factor, rdzw, fzm, fzp)
    if any(a.dtype != jnp.float32 for a in arrays):
        raise TypeError("native scalar flux requires WRF REAL fp32 inputs")
    if ru.shape != field.shape or rv.shape != field.shape or rom.shape != (nz+1, ny, nx):
        raise ValueError("inconsistent transporting velocity shapes")
    kernel = partial(_scalar_kernel, nz=nz, ny=ny, nx=nx, rdx=rdx, rdy=rdy, block=block)
    return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(field.shape, jnp.float32),
        grid=((nz*ny*nx+block-1)//block,), name="b_core_scalar_flux_fp32", interpret=interpret,
        compiler_params=pt.CompilerParams(num_warps=warps))(*arrays)


def _scalar_kernel_stacked(q, ru, rv, rom, map_factor, rdzw, fzm, fzp, out,
                           *, ns, nz, ny, nx, rdx, rdy, block):
    """Species-stacked ``_scalar_kernel``: velocities/metrics loaded once per cell.

    Per species the arithmetic (expressions and order) is exactly the
    single-species kernel's; only the species-invariant loads are hoisted.
    """
    flat = pl.program_id(0) * block + jnp.arange(block, dtype=jnp.int32)
    x, y, k = flat % nx, flat // nx % ny, flat // (nx * ny)
    valid = flat < nz * ny * nx
    zero, one, minus_one = jnp.float32(0), jnp.float32(1), jnp.float32(-1)

    def hvel(axis, velocity, shift):
        pos, extent = (x, nx) if axis == 2 else (y, ny)
        m = pos + shift
        xx, yy = (x + shift, y) if axis == 2 else (x, y + shift)
        return pt.load(velocity.at[k, yy, xx], mask=valid & (m < extent), other=zero)

    vy = (hvel(1, rv, 0), hvel(1, rv, 1))
    vx = (hvel(2, ru, 0), hvel(2, ru, 1))
    vz, wz = [], []
    for shift in (0, 1):
        face = k + shift
        vz.append(pt.load(rom.at[face, y, x], mask=valid, other=zero))
        inside = valid & (face > 0) & (face < nz)
        wz.append((pt.load(fzm.at[face], mask=inside, other=zero),
                   pt.load(fzp.at[face], mask=inside, other=zero)))
    map_value = pt.load(map_factor.at[y, x], mask=valid, other=zero)
    inverse_dz = pt.load(rdzw.at[k], mask=valid, other=zero)

    for s in range(ns):
        def qload(dk=0, dy=0, dx=0):
            zz, yy, xx = k + dk, y + dy, x + dx
            return pt.load(q.at[s, zz, yy, xx],
                           mask=valid & (zz >= 0) & (zz < nz) &
                           (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx), other=zero)

        qi = qload()

        def horizontal(axis, vels):
            pos, extent = (x, nx) if axis == 2 else (y, ny)
            values = [qload(dx=i) if axis == 2 else qload(dy=i)
                      for i in (-3, -2, -1)] + [qi] + [
                      qload(dx=i) if axis == 2 else qload(dy=i) for i in (1, 2, 3)]

            def face(shift):
                m = pos + shift
                v = vels[shift]
                qm3, qm2, qm1, qc, qp1, qp2 = values[shift:shift + 6]
                sign = jnp.where(v >= zero, one, minus_one)
                f6 = (jnp.float32(37) * (qc + qm1) - jnp.float32(8) * (qp1 + qm2) + (qp2 + qm3)) / jnp.float32(60)
                f5 = f6 - sign * ((qp2 - qm3) - jnp.float32(5) * (qp1 - qm2) + jnp.float32(10) * (qc - qm1)) / jnp.float32(60)
                f4 = (jnp.float32(7) * (qc + qm1) - (qp1 + qm2)) / jnp.float32(12)
                f3 = f4 + sign * ((qp1 - qm2) - jnp.float32(3) * (qc - qm1)) / jnp.float32(12)
                f2 = jnp.float32(.5) * (qc + qm1)
                value = jnp.where((m >= 3) & (m <= extent - 3), f5,
                                 jnp.where((m == 2) | (m == extent - 2), f3,
                                           jnp.where((m == 1) | (m == extent - 1), f2, zero)))
                return v * value

            difference = face(1) - face(0)
            return jnp.where((pos >= 1) & (pos <= extent - 2), difference, zero)

        zvalues = [qload(dk=-2), qload(dk=-1), qi, qload(dk=1), qload(dk=2)]

        def vertical(shift):
            face = k + shift
            v = vz[shift]
            qm2, qm1, qc, qp1 = zvalues[shift:shift + 4]
            sign = jnp.where(v <= zero, one, minus_one)  # WRF flux3(..., -rom)
            f4 = (jnp.float32(7) * (qc + qm1) - (qp1 + qm2)) / jnp.float32(12)
            f3 = f4 + sign * ((qp1 - qm2) - jnp.float32(3) * (qc - qm1)) / jnp.float32(12)
            wm, wp = wz[shift]
            f2 = wm * qc + wp * qm1
            value = jnp.where((face >= 2) & (face <= nz - 2), f3,
                              jnp.where((face == 1) | (face == nz - 1), f2, zero))
            return v * value

        result = -map_value * jnp.float32(rdy) * horizontal(1, vy)
        result = result - map_value * jnp.float32(rdx) * horizontal(2, vx)
        result = result - inverse_dz * (vertical(1) - vertical(0))
        pt.store(out.at[s, k, y, x], result, mask=valid)


def advect_scalar_flux_fp32_stacked(fields, ru, rv, rom, map_factor, rdzw, fzm, fzp,
                                    rdx, rdy, *, interpret=False, block=256, warps=4):
    """All species of one call in ONE launch, species loop inside each program."""
    ns, nz, ny, nx = fields.shape
    arrays = (fields, ru, rv, rom, map_factor, rdzw, fzm, fzp)
    if any(a.dtype != jnp.float32 for a in arrays):
        raise TypeError("native scalar flux requires WRF REAL fp32 inputs")
    if ru.shape != (nz, ny, nx) or rv.shape != (nz, ny, nx) or rom.shape != (nz+1, ny, nx):
        raise ValueError("inconsistent transporting velocity shapes")
    kernel = partial(_scalar_kernel_stacked, ns=ns, nz=nz, ny=ny, nx=nx, rdx=rdx, rdy=rdy, block=block)
    return pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(fields.shape, jnp.float32),
        grid=((nz*ny*nx+block-1)//block,), name="b_core_scalar_flux_fp32_stacked", interpret=interpret,
        compiler_params=pt.CompilerParams(num_warps=warps))(*arrays)
