"""Reference/fused full SW+LW families on matched real ALISIOS snapshots."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import jax

from bench_radiation import load_columns, timed, compare
from gpuwrf.kernels import rad_mcica, rad_lw_transfer, rad_sw_quadrature
from gpuwrf.physics import rrtmg_lw as lw, rrtmg_sw as sw


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain', choices=['d01', 'd02'], required=True)
    p.add_argument('--hour', choices=['00', '12'], default='12')
    p.add_argument('--out', type=Path, required=True)
    opts = p.parse_args()
    assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
    assert jax.devices()[0].platform == 'gpu'
    sw_state, lw_state, manifest = load_columns(opts.domain, opts.hour)
    manifest.update(clear_sky_outputs=True, rng='jump-ahead legacy-fp64 in both arms')
    sources = [Path(__file__), Path(lw.__file__), Path(sw.__file__),
               Path(rad_mcica.__file__), Path(rad_lw_transfer.__file__), Path(rad_sw_quadrature.__file__)]
    record = dict(manifest=manifest, source_sha256={str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in sources},
                  xla_flags=os.environ.get('XLA_FLAGS'), cache=os.environ.get('JAX_COMPILATION_CACHE_DIR'), families={})
    lw._lw_mcica_random_cloud_mask = lambda p, c, **kw: rad_mcica.lw_cloud_mask(p, c, legacy_fp64=True, **kw)
    sw._mcica_random_overlap_mask = lambda p, c, g: rad_mcica.sw_cloud_mask(p, c, g, legacy_fp64=True)
    for family, mod, state, flag in [('lw', lw, lw_state, '_FUSED_TRANSFER'),
                                    ('sw', sw, sw_state, '_FUSED_QUADRATURE')]:
        results, rows = {}, {}
        for arm in ('reference', 'fused'):
            setattr(mod, flag, arm == 'fused')
            fn = getattr(mod, f'solve_rrtmg_{family}_column')
            fn.clear_cache()
            start = time.perf_counter()
            exe = fn.lower(state, with_clear_sky=True).compile()
            compile_s = time.perf_counter() - start
            label = f'BR_both_{opts.domain}_h{opts.hour}_{family}_{arm}'
            result, row = timed(exe, (state,), label, 3)
            row.update(compile_s=compile_s, memory_analysis=str(exe.memory_analysis()))
            rows[arm] = row
            results[arm] = jax.device_get(result)
            print(label, row['median_s'], 'compile', compile_s, flush=True)
            del result, exe
        rows['regression'] = compare(results['reference'], results['fused'])
        record['families'][family] = rows
        opts.out.write_text(json.dumps(record, indent=2) + '\n')
        assert rows['regression']['finite']
    record['fused_wall_sum_s'] = sum(x['fused']['median_s'] for x in record['families'].values())
    opts.out.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
