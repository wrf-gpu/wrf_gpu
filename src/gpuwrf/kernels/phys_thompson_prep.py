"""Fuse fall-speed prep, explicit sedimentation, and final REAL work per column.

The retained Thompson functions supply the distribution/melt/finish equations.
Only their level-local fill/flux integration is replaced with register work.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from gpuwrf.kernels.phys_thompson_sedimentation import _gather, _load, _store


class _CoefficientRef:
    """Scalar reads from the small runtime coefficient stack."""
    def __init__(self, ref, offset, dtype=None):
        self.ref, self.offset, self.dtype = ref, offset, dtype

    def __getitem__(self, index):
        value = _load(self.ref, (self.offset + index,))
        return value if self.dtype is None else value.astype(self.dtype)

    def astype(self, dtype):
        return _CoefficientRef(self.ref, self.offset, dtype)


def _fill_registers(vt, active, nz):
    width = vt.shape[0]
    k = jnp.arange(width, dtype=jnp.int32)
    nearest = jnp.where(active & (k < nz), k, nz)
    for distance in (1 << d for d in range(width.bit_length() - 1)):
        above = _gather.bind(nearest, (k + distance) % width)
        nearest = jnp.minimum(nearest, jnp.where(k + distance < nz, above, nz))
    work = vt.astype(jnp.float32)
    value = _gather.bind(work, jnp.minimum(nearest, width - 1))
    return jnp.where(nearest < nz, value, 0.0).astype(vt.dtype)


def _sed_registers(q, num, vm, vn, dz, rho, dt, steps, threshold, nz):
    q, num, vm, vn, dz, rho = (x.astype(jnp.float32) for x in (q, num, vm, vn, dz, rho))
    k = jnp.arange(q.shape[0], dtype=jnp.int32)
    neighbour = (k + 1) % q.shape[0]
    dts = dt.astype(jnp.float32) / steps.astype(jnp.float32)

    def body(carry):
        iteration, qa, na, ppt = carry
        fq = vm * jnp.maximum(qa * rho, 0.0)
        fn = vn * jnp.maximum(na * rho, 0.0)
        fqa = jnp.where(k + 1 < nz, _gather.bind(fq, neighbour), 0.0)
        fna = jnp.where(k + 1 < nz, _gather.bind(fn, neighbour), 0.0)
        qb = jnp.maximum(qa + (fqa - fq) / dz / rho * dts, 0.0)
        nb = jnp.maximum(na + (fna - fn) / dz / rho * dts, 0.0)
        surface_density = jnp.sum(jnp.where(k == 0, qb * rho, 0.0))
        flux = jnp.sum(jnp.where(k == 0, fq, 0.0))
        ppt += jnp.where(surface_density > threshold, flux * dts, 0.0)
        return iteration + 1, qb, nb, ppt

    _, q, num, ppt = lax.while_loop(lambda c: c[0] < steps, body,
                                   (jnp.asarray(0, jnp.int32), q, num, jnp.asarray(0., jnp.float32)))
    return q, num, ppt


def sediment_and_finish(state, dt, vts_boost=None, *, interpret=False):
    from gpuwrf.physics import thompson_column as tc

    shape = state.qv.shape
    nz, ncol = shape[-1], math.prod(shape[:-1])
    width = 1 << (nz - 1).bit_length()
    keys = tc.ThompsonColumnState.__slots__
    arrays = [getattr(state, key).reshape(ncol, nz) for key in keys]
    if any(a.dtype != jnp.float32 for a in arrays):
        raise ValueError("fused prep expects REAL work state")
    boost = jnp.ones_like(state.qs) if vts_boost is None else vts_boost
    coefficient_stack = jnp.concatenate([tc.THOMPSON_TABLES.snow_sa,
                                         tc.THOMPSON_TABLES.snow_sb,
                                         tc.THOMPSON_TABLES.cse])
    scalars = jnp.asarray([dt, tc.RR_SURF_THRESHOLD], jnp.float32)
    defaults = dict(T=280., p=100000., rho=1., dz=250.)

    def kernel(*refs):
        input_refs = refs[:len(keys)]
        bref, cref, sref = refs[len(keys):len(keys)+3]
        output_refs = refs[len(keys)+3:len(keys)*2+3]
        precip_refs = refs[len(keys)*2+3:]
        col = pl.program_id(0)
        k = jnp.arange(width, dtype=jnp.int32)
        valid = k < nz
        values = {key: _load(ref, (col, k), mask=valid, other=defaults.get(key, 0.))
                  for key, ref in zip(keys, input_refs)}
        current = tc.ThompsonColumnState(**values)
        boost_values = _load(bref, (col, k), mask=valid, other=1.)
        tables = SimpleNamespace(snow_sa=_CoefficientRef(cref, 0),
                                 snow_sb=_CoefficientRef(cref, 10),
                                 cse=_CoefficientRef(cref, 20))
        step_dt = _load(sref, (0,))
        threshold = _load(sref, (1,))
        speeds = tc._fall_speeds(current, boost_values,
                                fill_down=lambda v, a: _fill_registers(v, a, nz), tables=tables)
        vr, vnr, vi, vni, vs, vg, vng = speeds
        dz = jnp.maximum(current.dz, 1.)
        rho = jnp.maximum(current.rho, tc.R1)
        nr = tc._clamp_rain_number(current.qr, current.Nr, rho)
        ni = tc._balance_ice_number(current.qi, current.Ni, rho)
        qr, nr, rain = _sed_registers(current.qr, nr, vr, vnr, dz, rho, step_dt,
                                      tc._nstep_per_column(vr, vnr, dz, step_dt), threshold, nz)
        qi, ni, ice = _sed_registers(current.qi, ni, vi, vni, dz, rho, step_dt,
                                    tc._nstep_per_column(vi, vi, dz, step_dt), threshold, nz)
        qs, ns, snow = _sed_registers(current.qs, current.Ns, vs, vs, dz, rho, step_dt,
                                     tc._nstep_per_column(vs, vs, dz, step_dt), threshold, nz)
        qg, ng, graupel = _sed_registers(current.qg, current.Ng, vg, vng, dz, rho, step_dt,
                                        tc._nstep_per_column(vg, vg, dz, step_dt), threshold, nz)
        current = current.replace(qr=qr, Nr=nr, qi=qi, Ni=ni, qs=qs, Ns=ns, qg=qg, Ng=ng)
        current = tc._real_state(tc._instant_melt_freeze(current, step_dt))
        current = tc._real_state(tc._finish(current))
        for key, ref in zip(keys, output_refs):
            _store(ref, (col, k), getattr(current, key), mask=valid)
        for value, ref in zip((rain, ice, snow, graupel), precip_refs):
            _store(ref, (col,), value)

    output_shapes = [jax.ShapeDtypeStruct((ncol, nz), jnp.float32) for _ in keys]
    output_shapes += [jax.ShapeDtypeStruct((ncol,), jnp.float32) for _ in range(4)]
    outputs = pl.pallas_call(kernel, grid=(ncol,), out_shape=output_shapes,
                            interpret=interpret, compiler_params=plt.CompilerParams(num_warps=2),
                            name="thompson_sediment_prep_finish")(
                                *arrays, boost.reshape(ncol, nz), coefficient_stack, scalars)
    result = state.replace(**{key: array.reshape(shape) for key, array in zip(keys, outputs[:len(keys)])})
    precip = {key: array.reshape(shape[:-1]) for key, array in
              zip(("rain", "ice", "snow", "graupel"), outputs[len(keys):])}
    return result, precip
