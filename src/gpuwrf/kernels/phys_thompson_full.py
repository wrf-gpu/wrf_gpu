"""Full Thompson source/sink call in one column-resident Pallas kernel.

The current retained function code is reused in a private trace namespace.
Runtime table refs replace only lookup operations; column reductions use the
physical level count and explicit sedimentation remains in registers. Keeping
current code objects also keeps the frozen oracle's source mutations live.
"""
from __future__ import annotations

import inspect
import math
import os
from types import FunctionType

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plt

from gpuwrf.kernels.phys_thompson_prep import _fill_registers, _sed_registers
from gpuwrf.kernels.phys_thompson_sedimentation import _gather, _load, _store


def _any(values, axis=None, keepdims=False):
    # Installed Triton lowering supports integer max/min, not reduce_or/and.
    return jnp.max((values != 0).astype(jnp.int32), axis=axis, keepdims=keepdims) != 0


class _RegisterNumpy:
    any = staticmethod(_any)

    @staticmethod
    def all(values, axis=None, keepdims=False):
        return jnp.min((values != 0).astype(jnp.int32), axis=axis, keepdims=keepdims) != 0

    def __getattr__(self, name):
        return getattr(jnp, name)


class _Channels:
    def __init__(self, table, prefix):
        self.table, self.prefix = table, prefix

    def __getitem__(self, key):
        if not isinstance(key, tuple) or key[0] is not Ellipsis:
            raise ValueError("channel lookup requires [..., constant channel]")
        return self.table.read(self.prefix + (key[-1],))


class _Table:
    def __init__(self, ref, valid, dtype=None):
        self.ref, self.valid, self.dtype = ref, valid, dtype
        self.shape = ref.shape

    def read(self, index):
        value = _load(self.ref, index, mask=self.valid, other=0.)
        return value if self.dtype is None else value.astype(self.dtype)

    def __getitem__(self, index):
        value = _load(self.ref, (index,))
        return value if self.dtype is None else value.astype(self.dtype)

    def astype(self, dtype):
        return _Table(self.ref, self.valid, dtype)


def _namespace(tc, tables, cold, nz, width):
    """Reuse original equations without mutating module bindings or JIT caches."""
    env = dict(vars(tc))
    register_numpy = _RegisterNumpy()
    env["jnp"] = register_numpy

    def replace_default(value):
        if value is tc.THOMPSON_TABLES:
            return tables
        if value is tc.COLD_COLLECTION_TABLES:
            return cold
        return value

    for name, fn in list(env.items()):
        foreign_hook = name in ("_wrf_l_qc_any", "_wrf_cloud_sed_band", "_cloud_sed_rho_stages")
        if inspect.isfunction(fn) and (fn.__module__ == tc.__name__ or foreign_hook):
            defaults = tuple(replace_default(x) for x in fn.__defaults__) if fn.__defaults__ else None
            globals_ = env if fn.__module__ == tc.__name__ else dict(fn.__globals__)
            if globals_ is not env and globals_.get("jnp") is jnp:
                globals_["jnp"] = register_numpy
            closure = fn.__closure__
            if closure:
                def cell(value):
                    return (lambda: value).__closure__[0]
                closure = tuple(cell(register_numpy) if c.cell_contents is jnp else c for c in closure)
            copy = FunctionType(fn.__code__, globals_, fn.__name__, defaults, closure)
            copy.__kwdefaults__ = ({key: replace_default(value) for key, value in fn.__kwdefaults__.items()}
                                   if fn.__kwdefaults__ else None)
            env[name] = copy
    env["THOMPSON_TABLES"], env["COLD_COLLECTION_TABLES"] = tables, cold
    env["_full_column_enabled"] = lambda: False
    env["_sed_prep_fused_enabled"] = lambda: False
    env["_fill_down"] = lambda vt, active: _fill_registers(vt, active, nz)
    env["_sed_one_species"] = lambda q, num, vm, vn, dz, rho, dt, steps: _sed_registers(
        q, num, vm, vn, dz, rho, jnp.asarray(dt, jnp.float32), steps,
        jnp.asarray(tc.RR_SURF_THRESHOLD, jnp.float32), nz)
    env["_take2"] = lambda table, i, j: table.read((i, j))
    env["_take3_last"] = lambda table, i, j: _Channels(table, (i, j))
    env["_take4"] = lambda table, i, j, k, m: table.read((i, j, k, m))
    env["_take_qrfz"] = lambda table, i, j, k: _Channels(
        table, ((i.astype(jnp.int32) * tc.N_R1_TABLE + j.astype(jnp.int32)) * tc.N_TC_TABLE
                 + k.astype(jnp.int32),))
    k = jnp.arange(width, dtype=jnp.int32)
    valid = k < nz

    # All production L_qc reductions are ANY. The frozen ALL mutant needs true
    # padding (its neutral element) so it still operates on physical levels.
    lqc = env["_wrf_l_qc_any"]
    pad_qc = 2. * tc.R1 if "all" in tc._wrf_l_qc_any.__code__.co_names else 0.
    env["_wrf_l_qc_any"] = lambda qp, rp, qo, br: lqc(
        jnp.where(valid, qp, pad_qc), jnp.where(valid, rp, 1.),
        jnp.where(valid, qo, pad_qc), jnp.where(valid, br, False))
    if tc._wrf_cloud_sed_band.__name__ == "_wrf_cloud_sed_band":
        def band(rc, dz):
            below = jnp.cumsum(dz) - dz
            visited = (below <= 500.) & (k <= nz - 2)
            ksed = jnp.max(jnp.where(visited & (rc > tc.R2), k, 0))
            return k <= ksed
        env["_wrf_cloud_sed_band"] = band

    def cloud(state, dt, cloud_sed_on=None, cloud_rho=None):
        rho_rc, rho_f, rho_o = (state.rho,) * 3 if cloud_rho is None else cloud_rho
        dtype = jnp.result_type(state.qc, rho_rc, state.dz)
        rho = jnp.maximum(rho_rc, tc.R1).astype(dtype)
        rho_update = jnp.maximum(rho_o, tc.R1).astype(dtype)
        dz = jnp.maximum(state.dz, 1.).astype(dtype)
        qc = state.qc.astype(dtype)
        vt = env["_cloud_water_fall_speed"](state, rho_rc, rho_f).astype(dtype)
        if cloud_sed_on is not None:
            vt = jnp.where(cloud_sed_on, vt, 0.)
        active_band = env["_wrf_cloud_sed_band"](jnp.maximum(qc * rho, tc.R1), dz)
        flux = jnp.where(active_band, vt * jnp.maximum(qc * rho, 0.), 0.)
        above = jnp.where(k + 1 < nz, _gather.bind(flux, (k + 1) % width), 0.)
        out = jnp.where(active_band, jnp.maximum(qc + (above - flux) / dz / rho_update * dt, 0.), qc)
        surface = jnp.sum(jnp.where(k == 0, flux, 0.)) * dt
        return out.astype(state.qc.dtype), surface.astype(jnp.float64)

    env["_sed_cloud_water"] = cloud
    early_exit = os.environ.get("GPUWRF_THOMPSON_FULL_COLUMN_EARLY_EXIT", "0")
    if early_exit == "1":
        _inactive_source_branches(env, tc, valid)
    # Bitwise-exact like the ice/cold exit, so they follow it unless set explicitly (A/B).
    if os.environ.get("GPUWRF_THOMPSON_FULL_COLUMN_WARM_EXIT", early_exit) == "1":
        _inactive_warm_branches(env, tc, valid)
    if os.environ.get("GPUWRF_THOMPSON_FULL_COLUMN_SED_EXIT", early_exit) == "1":
        _inactive_sedimentation(env, tc, valid)
    return env


def _inactive_sedimentation(env, tc, valid):
    """Skip fall speeds in columns without rain/ice/snow/graupel above R1.

    Each speed is masked by its species' > R1 test, so all are exactly zero
    there; the retained sediment/number/precip code runs with zero speeds.
    """
    sed_fn, sediment = env["_sedimentation"], env["_sediment_with_speeds"]

    def sedimentation(state, dt, vts_boost=None, cloud_sed_on=None, cloud_rho=None):
        active = _any(valid & ((state.qr > tc.R1) | (state.qi > tc.R1)
                               | (state.qs > tc.R1) | (state.qg > tc.R1)))
        zero = jnp.zeros_like(state.qr, dtype=jnp.float64)
        return jax.lax.cond(
            active, lambda: sed_fn(state, dt, vts_boost, cloud_sed_on, cloud_rho),
            lambda: sediment(state, dt, (zero,) * 7, cloud_sed_on, cloud_rho))

    env["_sedimentation"] = sedimentation


def _inactive_warm_branches(env, tc, valid):
    """Skip warm-rain and evaporation rates in columns without cloud/rain above R1.

    Every rate is exactly zero there (prr_wau needs qc > R1, the others rain
    > R1), so the inactive branch applies exact zeros through the retained
    update code: the rain-number cap/floor still run, values are bitwise equal.
    """
    warm_fn, evap_fn = env["_warm_rain_collection"], env["_rain_evaporation"]
    apply_warm, apply_evap = env["_apply_warm_rain_rates"], env["_apply_rain_evaporation"]
    air = env["_air_properties"]

    def warm(state, dt, tables=env["THOMPSON_TABLES"]):
        active = _any(valid & ((state.qc > tc.R1) | (state.qr > tc.R1)))
        zero64 = jnp.zeros_like(state.qc, dtype=jnp.float64)
        return jax.lax.cond(active, lambda: warm_fn(state, dt, tables),
                            lambda: apply_warm(state, dt, zero64, zero64, zero64,
                                               jnp.zeros_like(state.qc, dtype=jnp.float32)))

    def evaporation(state, dt, skip_evaporation=False, graupel_melt=0.0):
        active = _any(valid & (state.qr > tc.R1))

        def inactive():
            _tempc, _diffu, _visco, _tcond, lvap, ocp, *_ = air(state)
            return apply_evap(state, jnp.zeros_like(state.qr, dtype=jnp.float64), lvap, ocp)

        return jax.lax.cond(active, lambda: evap_fn(state, dt, skip_evaporation, graupel_melt), inactive)

    env["_warm_rain_collection"], env["_rain_evaporation"] = warm, evaporation


def _inactive_source_branches(env, tc, valid):
    """Skip only families whose retained process gates all exclude transfer."""
    cold_fn = env["_cold_collection_rates"]
    ice_fn = env["_ice_sources_with_process_flags"]

    def cold(state, dt, tables):
        active = _any(valid & ((state.qs > tc.R1) | (state.qg > tc.R1)))
        return jax.lax.cond(active,
                            lambda: tuple(x.astype(jnp.float64) for x in cold_fn(state, dt, tables)),
                            lambda: tuple(jnp.zeros_like(state.qv, dtype=jnp.float64) for _ in range(8)))

    def ice(state, dt, tables=env["THOMPSON_TABLES"], cold_collection_rates=None):
        rates = (tuple(jnp.zeros_like(state.qv, dtype=jnp.float64) for _ in range(8))
                 if cold_collection_rates is None else cold_collection_rates)
        # The retained helper rounds every per-process increment to REAL (WRF REAL
        # tendencies), so an inactive column keeps REAL T/qv and only gets the REAL
        # density recompute and number caps below; the gate is evaluated in REAL as
        # the helper's own nucleation test (RSLF/RSIF are REAL functions).
        temp, vapor = state.T, state.qv
        rho = tc.density_from_pressure_temperature(state.p, temp, vapor)
        ssati = vapor / tc.saturation_mixing_ratio_ice(state.p, temp) - 1.
        ssatw = vapor / tc.saturation_mixing_ratio_liquid(state.p, temp) - 1.
        cold_cell = temp < tc.T_0
        nucleation = cold_cell & ((ssati >= .25) | ((ssatw > tc.EPS) & (temp < 253.15)))
        freezing = cold_cell & ((state.qc > tc.R1) | (state.qr > tc.R1))
        existing = (state.qi > tc.R1) | (state.qs > tc.R1) | (state.qg > tc.R1)
        active = _any(valid & (existing | freezing | nucleation))
        for rate in rates:
            active = active | _any(valid & (rate != 0.))

        def full():
            out, melt, boost, scaled = ice_fn(state, dt, tables, rates)
            return (env["_real_state"](out), melt.astype(jnp.float32), boost.astype(jnp.float32),
                    tuple(x.astype(jnp.float64) for x in scaled))

        def inactive():
            rr = jnp.maximum(state.qr * rho, tc.R1)
            nr_max = tc.CRG2 * tc.ORG3 * rr * ((3. + .672) / (tc.D0R * .75)) ** 3. / tc.AM_R / rho
            ni = jnp.minimum(state.Ni, 999.e3 / state.rho)            # freeze-stage ceiling (entry rho)
            ni = jnp.minimum(jnp.maximum(0., ni), 999.e3 / rho)        # final ceiling (recomputed rho)
            out = state.replace(rho=rho, Ni=ni, Nr=jnp.maximum(0., jnp.minimum(state.Nr, nr_max)))
            return (env["_real_state"](out), jnp.zeros_like(vapor, dtype=jnp.float32),
                    jnp.ones_like(vapor, dtype=jnp.float32), tuple(x.astype(jnp.float64) for x in rates))

        return jax.lax.cond(active, full, inactive)

    env["_cold_collection_rates"], env["_ice_sources_with_process_flags"] = cold, ice


def full_column(state, dt, *, interpret=False, return_events=False):
    """Retained return contract, or optional device counters for validation.

    Diagnostic counters count physical cells before the retained fallback
    selection and never modify equations, masks or returned state.
    """
    from gpuwrf.physics import thompson_column as tc

    shape = state.qv.shape
    nz, ncol = shape[-1], math.prod(shape[:-1])
    width = 1 << (nz - 1).bit_length()
    keys = tc.ThompsonColumnState.__slots__
    arrays = [getattr(state, key).reshape(ncol, nz) for key in keys]
    table_arrays = list(tc.THOMPSON_TABLES)
    cold_arrays = list(tc.COLD_COLLECTION_TABLES)
    ntables, ncold = len(table_arrays), len(cold_arrays)
    precip_keys = ("rain", "snow", "graupel", "ice", "cloudw")
    defaults = dict(T=280., p=100000., rho=1., dz=250.)

    def kernel(*refs):
        col = pl.program_id(0)
        k = jnp.arange(width, dtype=jnp.int32)
        valid = k < nz
        source = tc.ThompsonColumnState(**{
            key: _load(ref, (col, k), mask=valid, other=defaults.get(key, 0.))
            for key, ref in zip(keys, refs[:len(keys)])})
        offset = len(keys)
        tables = tc.ThompsonTableBundle(*[_Table(ref, valid) for ref in refs[offset:offset+ntables]])
        offset += ntables
        cold = type(tc.COLD_COLLECTION_TABLES)(*[_Table(ref, valid) for ref in refs[offset:offset+ncold]])
        offset += ncold
        env = _namespace(tc, tables, cold, nz, width)
        events = {}
        if return_events:
            select = env["_select_state"]

            def counted_select(mask, good, fallback):
                finite = jnp.ones_like(valid)
                for key in keys:
                    finite = finite & jnp.isfinite(getattr(good, key))
                events["inadmissible"] = jnp.sum(valid & ~mask, dtype=jnp.int32)
                events["pre_mask_nonfinite"] = jnp.sum(valid & ~finite, dtype=jnp.int32)
                return select(mask, good, fallback)

            env["_select_state"] = counted_select
            input_finite = jnp.ones_like(valid)
            for key in keys:
                input_finite = input_finite & jnp.isfinite(getattr(source, key))
            events["input_nonfinite"] = jnp.sum(valid & ~input_finite, dtype=jnp.int32)
        out, ppt = env["_thompson_source_sink_body"](source, dt, False, sediment=True)
        for key, ref in zip(keys, refs[offset:offset+len(keys)]):
            _store(ref, (col, k), getattr(out, key), mask=valid)
        offset += len(keys)
        for key, ref in zip(precip_keys, refs[offset:offset+len(precip_keys)]):
            _store(ref, (col,), ppt[key])
        if return_events:
            for index, key in enumerate(("input_nonfinite", "inadmissible", "pre_mask_nonfinite")):
                _store(refs[-1], (col, jnp.asarray(index, jnp.int32)), events[key])

    output_shapes = [jax.ShapeDtypeStruct((ncol, nz), array.dtype) for array in arrays]
    output_shapes += [jax.ShapeDtypeStruct((ncol,), jnp.float64) for _ in precip_keys]
    if return_events:
        output_shapes += [jax.ShapeDtypeStruct((ncol, 3), jnp.int32)]
    outputs = pl.pallas_call(kernel, grid=(ncol,), out_shape=output_shapes,
                            interpret=interpret, compiler_params=plt.CompilerParams(num_warps=2),
                            name="thompson_full_column")(*arrays, *table_arrays, *cold_arrays)
    out = state.replace(**{key: value.reshape(shape) for key, value in zip(keys, outputs[:len(keys)])})
    ppt = {key: value.reshape(shape[:-1]) for key, value in zip(precip_keys, outputs[len(keys):len(keys)+len(precip_keys)])}
    if return_events:
        return out, ppt, outputs[-1]
    return out, ppt
