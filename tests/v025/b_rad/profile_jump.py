"""One real 1024-column tile for ncu attribution of parallel McICA."""

import argparse
import os

import jax
import nvtx

from bench_radiation import load_columns, mask_inputs
from gpuwrf.kernels.rad_mcica import lw_cloud_mask, sw_cloud_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=['lw', 'sw'], default='lw')
    args = parser.parse_args()
    assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
    assert jax.devices()[0].platform == 'gpu'
    sw, lw, _ = load_columns('d02')
    inputs = mask_inputs(args.family, lw if args.family == 'lw' else sw)
    tile = tuple(value[:1024] if index < 2 else value for index, value in enumerate(inputs))
    fn = lw_cloud_mask if args.family == 'lw' else sw_cloud_mask
    exe = jax.jit(fn).lower(*tile).compile()
    jax.block_until_ready(exe(*tile))
    with nvtx.annotate('BR_roofline'):
        jax.block_until_ready(exe(*tile))


if __name__ == '__main__':
    main()
