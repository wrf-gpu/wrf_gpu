"""Frozen WRF radiation gates and same-input component receipts.

Run directly: pytest's v025 CPU guard cannot establish a GPU gate. OFF arms
must use the same import alias for both immutable source snapshots (E58).
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

if os.environ.get('JAX_PLATFORMS') == 'cpu':
    assert not Path('/tmp/wrf_gpu2_quiet').exists(), 'quiet window: defer CPU gate before JAX import'

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.coupling import physics_couplers as C
from gpuwrf.physics import rrtmg_lw as LW, rrtmg_sw as SW
from gpuwrf.validation import tier1_rrtmg as T1
from precision_inventory import inventory


def digest(tree):
    values = [np.asarray(v) for v in jax.tree.leaves(tree)]
    return hashlib.sha256(b''.join(
        str(v.dtype).encode() + str(v.shape).encode() + v.tobytes()
        for v in values)).hexdigest()


def radiation_root(state, grid, clock, static, land):
    return C.rrtmg_theta_tendency(
        state, grid, clock_base=clock, radiation_static=static, land_state=land,
        lead_seconds=jnp.float64(0.0), _with_diagnostics=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('oracle', 'off', 'root'), required=True)
    parser.add_argument('--backend', choices=('cpu', 'gpu'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--domain', choices=('d01', 'd02'), default='d01')
    parser.add_argument('--lower-only', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert jax.devices()[0].platform == args.backend
    if args.backend == 'gpu':
        assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
    else:
        assert not Path('/tmp/wrf_gpu2_quiet').exists()
        from jax.experimental import pallas as pl
        original = pl.pallas_call
        pl.pallas_call = lambda *a, **kw: original(*a, **dict(kw, interpret=True))
    record = dict(backend=jax.devices()[0].platform,
                  device=str(jax.devices()[0]), affinity=sorted(os.sched_getaffinity(0)),
                  flags={k: v for k, v in os.environ.items() if k.startswith('GPUWRF_')},
                  cache=os.environ.get('GPUWRF_JAX_CACHE_DIR'),
                  sources={m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                           for m in (C, LW, SW)})
    if args.mode in ('oracle', 'off'):
        if args.mode == 'oracle':
            record['lw'] = T1.run_tier1_lw(args.out / 'lw.json')
            record['sw'] = T1.run_tier1_sw(args.out / 'sw.json')
            assert record['lw']['pass'] and record['sw']['pass']
        record['entries'] = {}
        for name, mod, loader in (('lw', LW, T1.load_lw_fixture_state),
                                  ('sw', SW, T1.load_sw_fixture_state)):
            state, _ = loader()
            fn = getattr(mod, f'solve_rrtmg_{name}_column')
            lowered = fn.lower(state, with_clear_sky=True)
            hlo = lowered.compiler_ir('hlo').as_hlo_text()
            (args.out / f'{name}.hlo.txt').write_text(hlo)
            counts, nodes = inventory(hlo, Path.cwd())
            result = jax.block_until_ready(fn(state, with_clear_sky=True))
            np.savez(args.out / f'{name}_outputs.npz', **{
                str(i): np.asarray(v) for i, v in enumerate(jax.tree.leaves(result))})
            record['entries'][name] = dict(hlo_sha256=hashlib.sha256(hlo.encode()).hexdigest(),
                output_sha256=digest(result), finite=all(np.isfinite(np.asarray(v)).all()
                                                       for v in jax.tree.leaves(result)),
                f64_compute=Counter(n['opcode'] for n in nodes if n['category'] == 'compute'),
                f64_arithmetic=Counter(n['opcode'] for n in nodes
                    if n['category'] == 'compute' and n['opcode'] not in ('call', 'conditional', 'while')))
            assert record['entries'][name]['finite']
            if args.mode == 'oracle':
                # Calls/control with a retained f64 result are interface nodes; all
                # callee bodies are already included in the recursive HLO audit.
                assert not record['entries'][name]['f64_arithmetic'], record['entries'][name]
            if name == 'lw' and args.mode == 'oracle':
                truth = json.loads((Path(__file__).parents[1] / 'b_rad/data/lw_driver_clear_flux.json').read_text())
                expected = np.asarray(truth['fluxes'])
                record['clear'] = {}
                for i, field in enumerate(truth['fields']):
                    actual = np.asarray(getattr(result, field))
                    error = np.abs(actual - expected[..., i])
                    allowance = truth['flux_tolerance']['atol'] + truth['flux_tolerance']['rtol'] * np.abs(expected[..., i])
                    record['clear'][field] = dict(max_abs=float(error.max()),
                        finite=bool(np.isfinite(actual).all()), passed=bool(np.all(error <= allowance)))
                assert all(v['finite'] and v['passed'] for v in record['clear'].values())
    else:
        assert args.inputs is not None
        with args.inputs.open('rb') as f:
            state, grid, clock, static, land, manifest = pickle.load(f)[args.domain]
        operands = jax.tree.map(jnp.asarray, (state, grid, clock, static, land))
        record['input_sha256'] = digest(operands)
        record['manifest'] = manifest
        fn = jax.jit(radiation_root)
        lowered = fn.lower(*operands)
        hlo = lowered.compiler_ir('hlo').as_hlo_text()
        (args.out / 'root.hlo.txt').write_text(hlo)
        from jaxlib import _jax
        print_options = _jax.HloPrintOptions()
        print_options.print_metadata = True
        print_options.print_large_constants = False
        attributed = lowered.compiler_ir('hlo').as_hlo_module().to_string(print_options)
        (args.out / 'root.attributed.hlo.txt').write_text(attributed)
        counts, nodes = inventory(attributed, Path.cwd())
        record['hlo_sha256'] = hashlib.sha256(hlo.encode()).hexdigest()
        record['f64_compute'] = dict(Counter(
            (n['source_file'].split('/')[-1] + ':' + n['function'] + ':' + n['opcode'])
            for n in nodes if n['category'] == 'compute'))
        (args.out / 'nodes.json').write_text(json.dumps(nodes, indent=2) + '\n')
        if not args.lower_only:
            start = time.perf_counter()
            exe = lowered.compile()
            record['compile_s'] = time.perf_counter() - start
            optimized = exe.as_text()
            (args.out / 'optimized.hlo.txt').write_text(optimized)
            _, optimized_nodes = inventory(optimized, Path.cwd())
            record['optimized_f64_ops'] = dict(Counter(
                n['opcode'] for n in optimized_nodes if n['category'] == 'compute'))
            result = jax.block_until_ready(exe(*operands))
            samples = []
            if args.backend == 'gpu':
                import ctypes
                import site
                nvtx_path = next(Path(base) / 'nvidia/nvtx/lib/libnvToolsExt.so.1'
                                 for base in site.getsitepackages()
                                 if (Path(base) / 'nvidia/nvtx/lib/libnvToolsExt.so.1').is_file())
                nvtx = ctypes.CDLL(str(nvtx_path))
                nvtx.nvtxRangePushA.argtypes = [ctypes.c_char_p]
                nvtx.nvtxRangePop.argtypes = []
                cudart = ctypes.CDLL('/usr/local/cuda/lib64/libcudart.so')
                assert cudart.cudaProfilerStart() == 0
            for _ in range(7):
                start = time.perf_counter()
                if args.backend == 'gpu':
                    nvtx.nvtxRangePushA(b'BP43_radiation_root')
                    try:
                        result = jax.block_until_ready(exe(*operands))
                    finally:
                        nvtx.nvtxRangePop()
                else:
                    result = jax.block_until_ready(exe(*operands))
                samples.append(time.perf_counter() - start)
            if args.backend == 'gpu':
                assert cudart.cudaProfilerStop() == 0
            record['wall_s'] = samples
            record['median_s'] = float(np.median(samples))
            record['memory'] = str(exe.memory_analysis())
            record['finite'] = all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(result))
            record['output_sha256'] = digest(result)
            np.savez(args.out / 'root_outputs.npz', **{
                str(i): np.asarray(v) for i, v in enumerate(jax.tree.leaves(result))})
            assert record['finite']
    (args.out / 'receipt.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k not in ('sources', 'flags', 'f64_compute')}, indent=2))


if __name__ == '__main__':
    main()
