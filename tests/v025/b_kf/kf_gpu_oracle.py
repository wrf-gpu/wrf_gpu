"""Run the frozen WRF KF bounds on native or interpreted column programs.

Direct Python entry avoids tests/v025/conftest.py's forced CPU backend.
Uses the strict p0_4 tendency bounds for all five v060 savepoint regimes.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--interpret', action='store_true')
parser.add_argument('--batch', action='store_true')
args = parser.parse_args()
sys.path.insert(0, str(args.source / 'src'))
os.environ.setdefault('JAX_PALLAS_USE_MOSAIC_GPU', 'false')
os.environ['GPUWRF_KF_COLUMN_FP32'] = '1'
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf._x64_config import configure_jax_x64
from gpuwrf.kernels.phys_kf_column import kf_column
from gpuwrf.physics.cumulus_kf import kf_eta_para

configure_jax_x64()
if not args.interpret:
    assert jax.devices()[0].platform == 'gpu'
    assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
spec = importlib.util.spec_from_file_location('frozen', args.source / 'tests/test_kf_cumulus_oracle.py')
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)
data = [json.loads((args.source / f'proofs/v060/savepoints/kf_case_{cid}.json').read_text())
        for cid in range(1, 6)]
scalars = data[0]['scalars']
assert all(all(d['scalars'][key] == scalars[key] for key in ('DT', 'DX', 'KX')) for d in data)
keys = ('T', 'QV', 'P', 'DZ', 'RHO', 'W0AVG', 'U', 'V')
inputs = [jnp.asarray([d['columns'][key] for d in data], jnp.float32) for key in keys]
if args.interpret:
    one = lambda *a: kf_column(*a, scalars['DT'], scalars['DX'], scalars['KX'], interpret=True)
else:
    # Exercise the actual default-off dispatch seam, including its JIT/vmap wiring.
    one = lambda *a: kf_eta_para(*a, scalars['DT'], scalars['DX'], scalars['KX'])
fn = jax.jit(jax.vmap(one) if args.batch else one)
proof = dict(source=str(args.source), backend=jax.default_backend(), device=str(jax.devices()[0]),
             cases=[], failures=[], interpret=args.interpret, batch=args.batch,
             tolerances=dict(tendency_relative=oracle.TEND_REL,
                             tendency_abs_floor=oracle.TEND_ABS_FLOOR,
                             rain_relative=oracle.RAINCV_REL, rain_abs=oracle.RAINCV_ABS,
                             category='ISHALL exact; CUTOP/CUBOT abs<0.5', nca_abs=1e-6))
started = time.perf_counter()
try:
    batch_out = jax.block_until_ready(fn(*inputs)) if args.batch else None
    for row, d in enumerate(data):
        cid = row + 1
        s, c = d['scalars'], d['columns']
        out = ({key: value[row] for key, value in batch_out.items()} if args.batch else
               jax.block_until_ready(fn(*(value[row] for value in inputs))))
        out = {key: np.asarray(value) for key, value in out.items()}
        rec = dict(case=cid, fields={},
                   categories={key: float(out[key]) for key in ('ISHALL', 'CUTOP', 'CUBOT')})
        for key in oracle.TEND_FIELDS:
            ok, mad, mrd = oracle._check_field(out[key], c[key])
            rec['fields'][key] = dict(pass_=bool(ok), max_abs=mad, max_rel=mrd)
            if not ok:
                proof['failures'].append(f'case{cid} {key} abs={mad} rel={mrd}')
        if int(out['ISHALL']) != int(round(s['SHALL'])):
            proof['failures'].append(f'case{cid} ISHALL')
        for key in ('CUTOP', 'CUBOT'):
            if abs(float(out[key]) - s[key]) >= .5:
                proof['failures'].append(f'case{cid} {key}')
        if abs(float(out['TIMEC']) - s['TIMEC']) > 1e-6:
            proof['failures'].append(f'case{cid} TIMEC')
        if abs(float(out['NCA']) - s['NCA']) > 1e-6:
            proof['failures'].append(f'case{cid} NCA')
        rain_err = abs(float(out['RAINCV']) - s['RAINCV'])
        tol = max(oracle.RAINCV_REL * abs(s['RAINCV']), oracle.RAINCV_ABS)
        rec['RAINCV'] = dict(value=float(out['RAINCV']), oracle=s['RAINCV'],
                             max_abs=rain_err, tolerance=tol)
        if rain_err > tol:
            proof['failures'].append(f'case{cid} RAINCV')
        if not all(np.isfinite(value).all() for value in out.values()):
            proof['failures'].append(f'case{cid} nonfinite')
        proof['cases'].append(rec)
        print(json.dumps(rec), flush=True)
    proof['verdict'] = 'PASS' if not proof['failures'] else 'FAIL'
except Exception:
    proof['verdict'] = 'ERROR'
    proof['error'] = traceback.format_exc()
    print(proof['error'], flush=True)
proof['elapsed_s'] = time.perf_counter() - started
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(proof, indent=2) + '\n')
print(proof['verdict'], proof['failures'], flush=True)
sys.exit(0 if proof['verdict'] == 'PASS' else 1)
