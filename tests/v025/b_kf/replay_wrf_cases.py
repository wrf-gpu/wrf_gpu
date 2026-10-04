"""Replay pristine WRF binary oracle fixtures, preserving the frozen tolerances.

Profiles are eight REAL columns followed by DT/DX/warm_rain/f_qi/f_qs.
Expected rows contain six tendencies plus seven scalar diagnostics.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--fixture', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--levels', type=int, default=40)
p.add_argument('--gpu', action='store_true')
a = p.parse_args()
sys.path.insert(0, str(a.source / 'src'))
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf._x64_config import configure_jax_x64
from gpuwrf.kernels.phys_kf_column import kf_column
configure_jax_x64()
if a.gpu:
    assert jax.devices()[0].platform == 'gpu'
    assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
else:
    assert {device.platform for device in jax.devices()} == {'cpu'}
nz = a.levels
x = np.fromfile(a.fixture / 'inputs.bin', dtype='<f4').reshape(-1, nz * 8 + 5)
expected = np.fromfile(a.fixture / 'expected.bin', dtype='<f4').reshape(-1, nz * 6 + 7)
manifest = json.loads((a.fixture / 'manifest.json').read_text())
fields = ('RTHCUTEN', 'RQVCUTEN', 'RQCCUTEN', 'RQRCUTEN', 'RQICUTEN', 'RQSCUTEN')
scalars = ('RAINCV', 'PRATEC', 'NCA', 'CUTOP', 'CUBOT', 'ISHALL', 'TIMEC')
profiles = x[:, :nz * 8].reshape(-1, 8, nz)
actual = {}
started = time.monotonic()
for options in sorted(set(tuple(row) for row in x[:, nz * 8 + 2:])):
    wr, qi, qs = map(bool, options)
    rows = np.flatnonzero(np.all(x[:, nz * 8 + 2:] == options, axis=1))
    def one(*args):
        return kf_column(*args[:8], args[8], args[9], nz,
                         warm_rain=wr, f_qi=qi, f_qs=qs, interpret=not a.gpu)
    fn = jax.jit(jax.vmap(one))
    out = jax.block_until_ready(fn(*(jnp.asarray(profiles[rows, i]) for i in range(8)),
                                  jnp.asarray(x[rows, nz * 8]), jnp.asarray(x[rows, nz * 8 + 1])))
    for key, value in out.items():
        array = np.asarray(value)
        if key not in actual:
            actual[key] = np.empty((len(x),) + array.shape[1:], dtype=array.dtype)
        actual[key][rows] = array
results = []
for row in range(len(x)):
    result = dict(index=row, scenario=manifest['columns'][row], failures=[], fields={}, scalars={})
    for i, key in enumerate(fields):
        truth = expected[row, i * nz:(i + 1) * nz]
        error = float(np.max(np.abs(actual[key][row] - truth)))
        scale = max(float(np.max(np.abs(truth))), 1e-9)
        passed = error <= 1e-9 or error <= .002 * scale
        result['fields'][key] = dict(max_abs=error, max_rel=error / scale, pass_=passed)
        if not passed:
            result['failures'].append(key)
    for i, key in enumerate(scalars):
        value, truth = float(actual[key][row]), float(expected[row, 6 * nz + i])
        error = abs(value - truth)
        tolerance = max(1e-4, .003 * abs(truth)) if key == 'RAINCV' else (
            .499 if key in ('CUTOP', 'CUBOT') else (1e-6 if key in ('NCA', 'TIMEC') else 0.))
        passed = key == 'PRATEC' or error <= tolerance
        result['scalars'][key] = dict(actual=value, wrf=truth, max_abs=error, pass_=passed)
        if not passed:
            result['failures'].append(key)
    results.append(result)
nonfinite = int(sum(np.count_nonzero(~np.isfinite(value)) for value in actual.values()))
record = dict(source=str(a.source), fixture=str(a.fixture), backend=jax.default_backend(),
              actual_device=str(jax.devices()[0]), rows=results,
              failed_rows=[row['index'] for row in results if row['failures']],
              nonfinite=nonfinite, elapsed_s=time.monotonic() - started)
record['verdict'] = 'PASS' if not record['failed_rows'] and not nonfinite else 'FAIL'
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({key: value for key, value in record.items() if key != 'rows'}), flush=True)
sys.exit(0 if record['verdict'] == 'PASS' else 1)
