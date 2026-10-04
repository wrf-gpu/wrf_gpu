"""Asserted-GPU Noah-MP component timing on actual PROD day/night shapes."""
import argparse
import ctypes
import hashlib
import json
import os
import pickle
from pathlib import Path
import sys
import time

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--native', choices=['0','1'], required=True)
    p.add_argument('--case', help='Limit the arm to one prepared workload')
    args = p.parse_args()
    if Path('/tmp/wrf_gpu2_quiet').exists():
        return 125
    os.environ['GPUWRF_NOAHMP_NATIVE_REAL'] = args.native
    sys.path.insert(0, str(args.repo/'src'))
    import gpuwrf
    import jax
    import nvtx
    from gpuwrf.physics.noahmp.noahmp_driver import noah_mp_step
    from gpuwrf.physics.noahmp.precision import real_tree
    assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
    assert jax.devices()[0].platform == 'gpu'
    args.out.mkdir(parents=True, exist_ok=True)
    with args.inputs.open('rb') as f:
        payload = pickle.load(f)
    records, prepared = {}, {}
    for name, (land, forcing, static, ep, rp, dt) in payload.items():
        if args.case and name != args.case:
            continue
        if Path('/tmp/wrf_gpu2_quiet').exists():
            return 125
        land, forcing, static, ep, rp = jax.tree.map(
            lambda x: jax.device_put(x) if hasattr(x,'dtype') else x,
            (land,forcing,static,ep,rp))
        land, forcing, static, ep, rp = real_tree((land,forcing,static,ep,rp))
        fn = jax.jit(lambda ls, ff: noah_mp_step(ls,ff,static,dt,energy_params=ep,rad_params=rp))
        t0 = time.perf_counter()
        compiled = fn.lower(land,forcing).compile()
        setup = time.perf_counter()-t0
        result = compiled(land,forcing)
        jax.block_until_ready(result)
        host = jax.device_get(result)
        finite = all(np.isfinite(x).all() for x in jax.tree.leaves(host)
                     if hasattr(x,'dtype') and np.issubdtype(x.dtype,np.floating))
        assert finite, name
        # Preserve full outputs for field regression and repeatability receipts.
        leaves = jax.tree.leaves(host)
        np.savez(args.out/(name+'_outputs.npz'), **{f'leaf_{i}':x for i,x in enumerate(leaves)})
        modules = compiled.runtime_executable().hlo_modules()
        for i,m in enumerate(modules):
            (args.out/f'{name}_{i}.optimized.hlo').write_text(m.to_string())
        compiled(land,forcing)
        wall = []
        for _ in range(5):
            t0 = time.perf_counter()
            jax.block_until_ready(compiled(land,forcing))
            wall.append(time.perf_counter()-t0)
        records[name] = dict(shape=list(land.tv.shape),dt=dt,compile_s=setup,
                             wall_s=wall,finite=finite,
                             input_sha256=hashlib.sha256(b''.join(np.asarray(x).tobytes() for x in jax.tree.leaves((land,forcing,static,ep,rp)))).hexdigest())
        prepared[name] = (compiled,land,forcing)
    cudart = ctypes.CDLL('/usr/local/cuda/lib64/libcudart.so')
    assert cudart.cudaProfilerStart() == 0
    try:
        for name,(fn,land,forcing) in prepared.items():
            for i in range(3):
                with nvtx.annotate(f'NOAHMP:{args.native}:{name}:{i}'):
                    jax.block_until_ready(fn(land,forcing))
    finally:
        cudart.cudaProfilerStop()
    report = dict(backend='gpu',device=str(jax.devices()[0]),native=args.native,
                  xla_flags=os.environ.get('XLA_FLAGS'),records=records,
                  inputs=str(args.inputs),inputs_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
                  scope='isolated whole NoahMP step; real workload, frozen savepoints are fidelity gate')
    (args.out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
