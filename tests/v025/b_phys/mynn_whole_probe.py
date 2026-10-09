"""Asserted-GPU whole-MYNN probe on retained real day/night state and fluxes."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--inputs', required=True, type=Path)
parser.add_argument('--domain', required=True, choices=('d01', 'd02'))
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--mode', required=True, choices=('baseline', 'plume', 'fp32', 'native'))
parser.add_argument('--profile', action='store_true')
parser.add_argument('--cpu-lower', action='store_true')
args = parser.parse_args()
os.environ['GPUWRF_EDMF_FUSED_PLUME'] = '0' if args.mode == 'baseline' else '1'
os.environ['GPUWRF_MYNN_FP32_PLUME'] = '0' if args.mode == 'baseline' else '1'
os.environ['GPUWRF_MYNN_FP32_COLUMNS'] = '1' if args.mode == 'native' else '0'
if not args.cpu_lower and os.environ.get('GPUWRF_GPU_LOCK_HELD') != '1':
    raise RuntimeError('requires GPU lock')
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.physics import mynn_pbl as P
from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
if not args.cpu_lower:
    assert jax.devices()[0].platform == 'gpu'

args.output.mkdir(parents=True, exist_ok=True)
meta = json.loads((args.inputs / f'{args.domain}.json').read_text())
path = args.inputs / f'{args.domain}_whole.npz'
with np.load(path) as data:
    arrays = {name: jnp.asarray(data[name]) for name in data.files}
if args.mode in ('fp32', 'native'):
    arrays = {name: value.astype(jnp.float32) for name, value in arrays.items()}
state = P.MynnPBLColumnState(**{
    name: None if name in ('exner', 'ni') and f'state_{name}' not in arrays else arrays[f'state_{name}']
    for name in P.MynnPBLColumnState.__slots__})
flux = SurfaceFluxes(**{name: arrays[f'flux_{name}'] for name in SurfaceFluxes._fields if f'flux_{name}' in arrays})

def step(state, flux):
    if args.mode == 'native':
        return P.step_mynn_pbl_column_with_pblh(state, meta['dt'], debug=False, surface=flux, edmf=True, dx=meta['dx'])
    if args.mode == 'fp32':
        with jax.enable_x64(False):
            return P._tiled_mynn_step(state, meta['dt'], False, flux, True, meta['dx'])
    return P._tiled_mynn_step(state, meta['dt'], False, flux, True, meta['dx'])

start = time.perf_counter()
lowered = jax.jit(step).lower(state, flux)
hlo = lowered.compiler_ir(dialect='hlo').as_hlo_text()
(args.output / 'hlo.txt').write_text(hlo)
record = dict(mode=args.mode, backend=jax.default_backend(),
              device_platform=jax.devices()[0].platform,
              source_commit=os.popen('git rev-parse HEAD').read().strip(),
              input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              shape=meta['shape'], dt=meta['dt'], dx=meta['dx'], edmf=True,
              cache=os.environ.get('GPUWRF_JAX_CACHE_DIR'),
              f64_hlo_lines=len(re.findall(r'= f64\[', hlo)))
if args.cpu_lower:
    record['scope'] = 'CPU lowering only, no GPU measurement or fidelity claim'
else:
    compiled = lowered.compile()
    jax.block_until_ready(compiled(state, flux))
    record['compile_first_s'] = time.perf_counter() - start
    if args.profile:
        cuda = ctypes.CDLL('libcudart.so.12')
        assert cuda.cudaProfilerStart() == 0
        with jax.profiler.TraceAnnotation(f'BP_MYNN_WHOLE:{args.domain}:{args.mode}'):
            result = compiled(state, flux)
            jax.block_until_ready(result)
        assert cuda.cudaProfilerStop() == 0
    else:
        samples = []
        for _ in range(7):
            start = time.perf_counter()
            result = compiled(state, flux)
            jax.block_until_ready(result)
            samples.append((time.perf_counter() - start) * 1000)
        record['wall_ms_samples'] = samples
        record['wall_ms_median'] = statistics.median(samples)
    leaves = jax.tree_util.tree_leaves(result)
    record['nonfinite_outputs'] = int(jax.device_get(sum(jnp.sum(~jnp.isfinite(v)) for v in leaves)))
    record['output_dtypes'] = sorted({str(v.dtype) for v in leaves})
    np.savez(args.output / 'outputs.npz', **{
        name: np.asarray(getattr(result[0], name)) for name in P.MynnPBLColumnState.__slots__
        if getattr(result[0], name) is not None}, pblh=np.asarray(result[1]))
(args.output / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
