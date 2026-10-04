"""Real-column full LW call: reference versus fused native transfer sweeps."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import jax

from bench_radiation import load_columns, timed, compare
from gpuwrf.kernels import rad_mcica, rad_lw_transfer
from gpuwrf.physics import rrtmg_lw as lw


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain', choices=['d01', 'd02'], required=True)
    p.add_argument('--hour', choices=['00', '12'], default='12')
    p.add_argument('--out', type=Path, required=True)
    opts = p.parse_args()
    assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
    assert jax.devices()[0].platform == 'gpu'
    _, state, manifest = load_columns(opts.domain, opts.hour)
    manifest.update(clear_sky_outputs=True, rng='jump-ahead legacy-fp64 in both arms')
    sources = [Path(__file__), Path(lw.__file__), Path(rad_mcica.__file__), Path(rad_lw_transfer.__file__)]
    record = dict(manifest=manifest, source_sha256={str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in sources},
                  xla_flags=os.environ.get('XLA_FLAGS'), cache=os.environ.get('JAX_COMPILATION_CACHE_DIR'), arms={})
    lw._lw_mcica_random_cloud_mask = lambda p, c, **kw: rad_mcica.lw_cloud_mask(p, c, legacy_fp64=True, **kw)
    original = lw._lw_rtrnmc_band_fluxes
    results = {}
    for arm in ('reference', 'fused'):
        lw._lw_rtrnmc_band_fluxes = original if arm == 'reference' else rad_lw_transfer.lw_band_fluxes
        fn = lw.solve_rrtmg_lw_column
        fn.clear_cache()
        start = time.perf_counter()
        exe = fn.lower(state, with_clear_sky=True).compile()
        compile_s = time.perf_counter() - start
        result, row = timed(exe, (state,), f'BR_fused_{opts.domain}_h{opts.hour}_lw_{arm}', 3)
        row.update(compile_s=compile_s, memory_analysis=str(exe.memory_analysis()))
        record['arms'][arm] = row
        results[arm] = jax.device_get(result)
        print(arm, row['median_s'], 'compile', compile_s, flush=True)
        del result, exe
    record['regression'] = compare(results['reference'], results['fused'])
    opts.out.write_text(json.dumps(record, indent=2) + '\n')
    assert record['regression']['finite']


if __name__ == '__main__':
    main()
