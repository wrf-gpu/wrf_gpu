"""Column-resident Thompson fill-down and explicit sedimentation.

One Triton program owns one column. Profiles stay in registers across the
adaptive substeps; there is no grid-wide level/substep loop. The small gather
primitive maps register neighbours to Triton's native tt.gather (Pallas's
Triton backend does not currently lower JAX gather).
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt
from jax.extend import core
from jax.interpreters import mlir
from jax._src.pallas.triton import lowering
from jaxlib.triton import dialect as tt


_gather = core.Primitive("thompson_register_gather")
_gather.def_impl(lambda x, i: jnp.take(x, i, mode="clip"))
_gather.def_abstract_eval(lambda x, i: jax.core.ShapedArray(i.shape, x.dtype))
mlir.register_lowering(
    _gather, mlir.lower_fun(lambda x, i: jnp.take(x, i, mode="clip"),
                          multiple_results=False))


@lowering.register_lowering(_gather)
def _gather_triton(ctx, x, i):
    return tt.gather(x, i, axis=0)


def _load(ref, index, **kwargs):
    return plt.load(ref.at[index], **kwargs)


def _store(ref, index, value, **kwargs):
    return plt.store(ref.at[index], value, **kwargs)


def fill_down(vt, active, *, interpret=False):
    """WRF top-to-bottom inheritance, including a zero model-top speed."""
    shape = vt.shape
    nz = shape[-1]
    ncol = math.prod(shape[:-1])
    width = 1 << (nz - 1).bit_length()

    def kernel(v, a, out):
        col = pl.program_id(0)
        k = jnp.arange(width, dtype=jnp.int32)
        is_active = _load(a, (col, k), mask=k < nz, other=False)
        nearest = jnp.where(is_active, k, nz)
        # Parallel suffix-min finds the first active level at/above k.
        for distance in (1 << d for d in range(width.bit_length() - 1)):
            above = _gather.bind(nearest, (k + distance) % width)
            nearest = jnp.minimum(nearest, jnp.where(k + distance < nz, above, nz))
        filled = _load(v, (col, jnp.minimum(nearest, nz - 1)),
                         mask=nearest < nz, other=0.0)
        _store(out, (col, k), filled, mask=k < nz)

    return pl.pallas_call(
        kernel, grid=(ncol,), out_shape=jax.ShapeDtypeStruct((ncol, nz), vt.dtype),
        interpret=interpret, compiler_params=plt.CompilerParams(num_warps=2),
        name="thompson_fill_down",
    )(vt.reshape(ncol, nz), active.reshape(ncol, nz)).reshape(shape)


def sediment_one_species(q, num, vt_mass, vt_num, dz, rho, dt, nstep,
                         surface_threshold, *, interpret=False, work_dtype=None):
    """Existing explicit flux scheme, with a private adaptive column loop.

    Arithmetic follows _sed_one_species; input/output storage dtypes are
    preserved. work_dtype=float32 selects the WRF REAL flux arithmetic without
    changing the DOUBLE source/rate work. No NSED_MAX truncation is applied.
    """
    shape = q.shape
    nz = shape[-1]
    ncol = math.prod(shape[:-1])
    width = 1 << (nz - 1).bit_length()
    dtype = (jnp.result_type(q, num, vt_mass, vt_num, dz, rho)
             if work_dtype is None else work_dtype)
    arrays = [jnp.asarray(a, dtype).reshape(ncol, nz)
              for a in (q, num, vt_mass, vt_num, dz, rho)]
    steps = jnp.asarray(nstep, dtype).reshape(ncol)
    scalars = jnp.asarray([dt, surface_threshold], dtype)

    def kernel(qref, nref, vmref, vnref, dzref, rhoref, stepsref, sref,
               qout, nout, pout):
        col = pl.program_id(0)
        k = jnp.arange(width, dtype=jnp.int32)
        valid = k < nz
        qc = _load(qref, (col, k), mask=valid, other=0.0)
        nc = _load(nref, (col, k), mask=valid, other=0.0)
        vm = _load(vmref, (col, k), mask=valid, other=0.0)
        vn = _load(vnref, (col, k), mask=valid, other=0.0)
        dzc = _load(dzref, (col, k), mask=valid, other=1.0)
        rhoc = _load(rhoref, (col, k), mask=valid, other=1.0)
        stepsc = _load(stepsref, (col,))
        dts = _load(sref, (0,)) / stepsc
        threshold = _load(sref, (1,))
        neighbour = (k + 1) % width

        def body(carry):
            iteration, qa, na, ppt = carry
            fq = vm * jnp.maximum(qa * rhoc, 0.0)
            fn = vn * jnp.maximum(na * rhoc, 0.0)
            fqa = jnp.where(k + 1 < nz, _gather.bind(fq, neighbour), 0.0)
            fna = jnp.where(k + 1 < nz, _gather.bind(fn, neighbour), 0.0)
            qb = jnp.maximum(qa + (fqa - fq) / dzc / rhoc * dts, 0.0)
            nb = jnp.maximum(na + (fna - fn) / dzc / rhoc * dts, 0.0)
            surface_density = jnp.sum(jnp.where(k == 0, qb * rhoc, 0.0))
            surface_flux = jnp.sum(jnp.where(k == 0, fq, 0.0))
            ppt += jnp.where(surface_density > threshold, surface_flux * dts, 0.0)
            return iteration + 1, qb, nb, ppt

        _, qc, nc, ppt = lax.while_loop(
            lambda carry: carry[0] < stepsc, body,
            (jnp.asarray(0, jnp.int32), qc, nc, jnp.asarray(0.0, dtype)))
        _store(qout, (col, k), qc, mask=valid)
        _store(nout, (col, k), nc, mask=valid)
        _store(pout, (col,), ppt)

    out_shapes = (jax.ShapeDtypeStruct((ncol, nz), dtype),
                  jax.ShapeDtypeStruct((ncol, nz), dtype),
                  jax.ShapeDtypeStruct((ncol,), dtype))
    qo, no, ppt = pl.pallas_call(
        kernel, grid=(ncol,), out_shape=out_shapes, interpret=interpret,
        compiler_params=plt.CompilerParams(num_warps=2),
        name="thompson_sediment_column",
    )(*arrays, steps, scalars)
    return qo.reshape(shape).astype(q.dtype), no.reshape(shape).astype(num.dtype), ppt.reshape(shape[:-1])
