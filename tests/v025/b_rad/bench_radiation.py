"""Short, same-work full-call or mask benchmark on real ALISIOS columns.

Run only with scripts/with_gpu_lock.sh --label b-rad and timeout. Snapshot
columns are copied from CPU-WRF, never generated or replicated. WRF fidelity
is established separately by test_mcica.py's frozen oracles. The legacy arm
here is a regression comparison, not a WRF fidelity oracle.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import jax
import jax.numpy as jnp
from jax import lax
from netCDF4 import Dataset
import numpy as np
import nvtx

from gpuwrf.kernels.rad_mcica import lw_cloud_mask, sw_cloud_mask
from gpuwrf.physics import rrtmg_lw as lw, rrtmg_sw as sw


CASE = Path('<DATA_ROOT>/alisios/runs/20260725_18z/wn2_wrf/alisios_operational_d01_121x71_d02_268x118_oldgrid_v1/attempt_001/cpu_case')


def load_columns(domain, hour='12'):
    path = CASE / f'wrfout_{domain}_2026-07-26_{hour}:00:00'
    with Dataset(path) as f:
        def get(name):
            return np.asarray(f.variables[name][0], dtype=np.float64)

        def col(v):
            return jnp.asarray(np.moveaxis(v, 0, -1).reshape(-1, v.shape[0]))

        p = get('P') + get('PB')
        temp = (get('T') + 300) * (p / 100000) ** (287 / 1004.5)
        qv = get('QVAPOR')
        dz = np.diff((get('PH') + get('PHB')) / 9.81, axis=0)
        rho = p / (287 * temp * (1 + 0.61 * qv))
        # WRF mass-grid hydrostatic pressure supplied to radiation.
        rad_p = get('P_HYD')
        common = [col(v) for v in (temp, rad_p, qv, get('QCLOUD'), get('QICE'),
                                  get('QSNOW'), get('QGRAUP'), get('CLDFRA'))]
        sw_state = sw.RRTMGSWColumnState(*common, jnp.asarray(get('ALBEDO').reshape(-1)),
                                      jnp.asarray(get('COSZEN').reshape(-1)), col(dz), col(rho))
        ptop = float(np.asarray(f.variables['P_TOP'][:]).reshape(-1)[0])
        lw_state = lw.RRTMGLWColumnState(*common, jnp.asarray(get('TSK').reshape(-1)),
                                      jnp.asarray(get('EMISS').reshape(-1)), col(dz), col(rho),
                                      top_pressure_pa=ptop)
    manifest = dict(snapshot=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    domain=domain, shape=list(p.shape), top_pressure_pa=ptop,
                    precision='fp64 legacy solver; jump RNG native fp32 or legacy fp64',
                    tile_columns=1024, clear_sky_outputs=False,
                    interfaces='solver midpoint reconstruction; snapshot P_HYD mass levels',
                    gases='solver historical defaults', cadence='isolated one SW+LW call; no forecast cadence')
    return sw_state, lw_state, manifest


def mask_inputs(family, state):
    if family == 'lw':
        _, _, buffer = lw._lw_extended_pressure_profiles(state.p, state.top_pressure_pa, None)
        p = jnp.concatenate((state.p, buffer), axis=-1)
        cf = jnp.pad(state.cloud_fraction, ((0, 0), (0, buffer.shape[-1])))
        return p, cf
    # The SW solver adds one clear top layer; pressure determines RNG seeds
    # only from the bottom four layers, but nlay controls stream positions.
    p = jnp.concatenate((state.p, state.p[:, -1:] * 0.5), axis=-1)
    cf = jnp.pad(state.cloud_fraction, ((0, 0), (0, 1)))
    gm = jnp.asarray([[g < cnt for g in range(12)] for cnt in sw._SW_GPOINT_COUNTS], dtype=jnp.float64)
    return p, cf, gm


def timed(exe, args, label, repeats):
    jax.block_until_ready(exe(*args))
    values = []
    with nvtx.annotate(label):
        for _ in range(repeats):
            start = time.perf_counter()
            result = exe(*args)
            jax.block_until_ready(result)
            values.append(time.perf_counter() - start)
    return result, dict(seconds=values, median_s=statistics.median(values),
                        range=label, repeats=repeats)


def compare(a, b):
    left, right = jax.tree.leaves(a), jax.tree.leaves(b)
    return dict(finite=all(np.isfinite(np.asarray(v)).all() for v in right),
                bitwise=all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(left, right)),
                max_abs=max(float(np.max(np.abs(np.asarray(x) - np.asarray(y))))
                            for x, y in zip(left, right)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', choices=['d01', 'd02'], required=True)
    parser.add_argument('--mode', choices=['mask', 'full'], default='full')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--with-clear-sky', action='store_true')
    parser.add_argument('--snapshot-hour', choices=['00', '12'], default='12')
    opts = parser.parse_args()
    assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
    assert jax.devices()[0].platform == 'gpu'
    sw_state, lw_state, manifest = load_columns(opts.domain, opts.snapshot_hour)
    manifest['clear_sky_outputs'] = opts.with_clear_sky
    record = dict(manifest=manifest, gpu=str(jax.devices()[0]), jax=jax.__version__,
                  cache=os.environ.get('JAX_COMPILATION_CACHE_DIR'), mode=opts.mode, families={})
    sources = (Path(__file__), Path(lw.__file__), Path(sw.__file__),
               Path('src/gpuwrf/kernels/rad_mcica.py'))
    record['source_sha256'] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    record['cuda_cache'] = os.environ.get('CUDA_CACHE_PATH', '~/.nv/ComputeCache (default)')
    for family, module, state, old, new in (
        ('lw', lw, lw_state, lw._lw_mcica_random_cloud_mask, lw_cloud_mask),
        ('sw', sw, sw_state, sw._mcica_random_overlap_mask, sw_cloud_mask),
    ):
        results, rows = {}, {}
        for arm in ('serial', 'jump_legacy', 'jump_native'):
            legacy = arm == 'jump_legacy'
            if arm == 'serial':
                helper = old
            else:
                def helper(*args, _legacy=legacy, _new=new, **kw):
                    if family == 'lw':
                        # Preserve mask output bytes in the isolated mask
                        # benchmark too; only RNG conversion changes by arm.
                        kw.setdefault('output_dtype', jnp.float64)
                    return _new(*args, legacy_fp64=_legacy, **kw)
            if opts.mode == 'mask':
                args = mask_inputs(family, state)
                fn = jax.jit(helper)
            else:
                name = '_lw_mcica_random_cloud_mask' if family == 'lw' else '_mcica_random_overlap_mask'
                setattr(module, name, helper)
                fn = getattr(module, f'solve_rrtmg_{family}_column')
                fn.clear_cache()
                args = (state,)
            start = time.perf_counter()
            kwargs = {'with_clear_sky': opts.with_clear_sky} if opts.mode == 'full' else {}
            exe = fn.lower(*args, **kwargs).compile()
            compile_s = time.perf_counter() - start
            label = f'BR_{opts.mode}_{opts.domain}_h{opts.snapshot_hour}_{family}_{arm}'
            result, row = timed(exe, args, label, opts.repeats)
            row['compile_s'] = compile_s
            row['memory_analysis'] = str(exe.memory_analysis())
            results[arm] = jax.device_get(result)
            rows[arm] = row
            del result, exe
            print(label, row['median_s'], flush=True)
        rows['legacy_regression'] = compare(results['serial'], results['jump_legacy'])
        rows['native_regression'] = compare(results['serial'], results['jump_native'])
        record['families'][family] = rows
        opts.out.write_text(json.dumps(record, indent=2) + '\n')
        setattr(module, '_lw_mcica_random_cloud_mask' if family == 'lw' else '_mcica_random_overlap_mask', old)
        assert rows['legacy_regression']['finite'] and rows['legacy_regression']['bitwise']
        assert rows['native_regression']['finite']


if __name__ == '__main__':
    main()
