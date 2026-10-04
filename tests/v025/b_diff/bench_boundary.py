"""Matched real-PROD dry-boundary replay after the pristine GPU gate passes."""
import argparse
import ctypes
import json
import statistics
import sys
import time
from pathlib import Path

source = Path(sys.argv[sys.argv.index('--source')+1])
sys.path.insert(0, str(source/'tests/v025/b_diff'))
import probe_boundary as probe  # resolves --arm before importing gpuwrf/JAX


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('native', 'legacy'), required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--domain', choices=('d01', 'd02'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    jax = probe.jax
    assert jax.devices()[0].platform == 'gpu', jax.devices()
    variant = 'physical' if args.domain == 'd01' else 'coupled'
    state, metrics, cfg, meta = probe.load(args.domain, variant)
    dt = meta['resolved_dt_s']
    fn = jax.jit(lambda s, m: probe.dry(s, m, cfg, dt, dt,
                 variant == 'coupled', args.domain == 'd02'))
    for _ in range(8):
        output = fn(state, metrics)
        jax.block_until_ready(output)
    samples = []
    for _ in range(40):
        start = time.perf_counter_ns()
        output = fn(state, metrics)
        jax.block_until_ready(output)
        samples.append((time.perf_counter_ns()-start)/1000)
    record = {'platform': 'gpu', 'device': str(jax.devices()[0]),
              'arm': args.arm, 'domain': args.domain, 'variant': variant,
              'immutable_source': str(args.source),
              'source_fixture_sha256': meta['fixture_sha256'],
              'resolved_dt_s': dt, 'cadence_s': cfg.update_cadence_s,
              'shape': list(state.theta.shape), 'calls': len(samples),
              'wall_us_median': statistics.median(samples),
              'wall_us_samples': samples,
              'scope': 'standalone dry boundary; no whole-forecast speed claim'}
    args.out.write_text(json.dumps(record, indent=2)+'\n')  # persist before capture/export
    if args.profile:
        cudart = ctypes.CDLL('/usr/local/cuda/lib64/libcudart.so')
        assert cudart.cudaProfilerStart() == 0
        for _ in range(20):
            output = fn(state, metrics)
            jax.block_until_ready(output)
        assert cudart.cudaProfilerStop() == 0
        record['profile_calls'] = 20
        args.out.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'wall_us_samples'}), flush=True)


if __name__ == '__main__':
    main()
