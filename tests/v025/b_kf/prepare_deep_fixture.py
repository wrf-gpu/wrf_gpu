"""Tile an independently validated deep WRF sounding for ACTIVE column timing.

This is a frozen oracle scaling fixture, not real PROD forecast weather.
The stored W0AVG is the value already consumed by WRF after its CPS update.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--columns', type=int, default=8400)
a = p.parse_args()
file = a.source / 'proofs/v060/savepoints/kf_case_1.json'
d = json.loads(file.read_text())
assert int(d['scalars']['SHALL']) == 0
keys = dict(T0='T', QV0='QV', P0='P', DZQ='DZ', RHOE='RHO',
            w0avg='W0AVG', U0='U', V0='V')
arrays = {key: np.tile(np.asarray(d['columns'][field], np.float64), (a.columns, 1))
          for key, field in keys.items()}
arrays['w'] = np.zeros_like(arrays['w0avg'])  # unused: W0AVG already after CPS
arrays['nca'] = np.full(a.columns, -100.)
a.output.mkdir(parents=True, exist_ok=True)
np.savez(a.output / 'd01.npz', **arrays)
metadata = dict(shape=[1, a.columns, int(d['scalars']['KX'])], dt=d['scalars']['DT'],
                dx=d['scalars']['DX'], fixture_kind='frozen WRF deep column scaling',
                preparation='unchanged pristine WRF v060 case1 profiles, tiled columns',
                w0avg_after_cps=True, expected_ishall=0, expected_active_columns=a.columns,
                oracle_path=str(file), oracle_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                scope='ACTIVE deep timing fixture; NOT real PROD forecast input',
                input_sha256=hashlib.sha256((a.output / 'd01.npz').read_bytes()).hexdigest())
(a.output / 'd01.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(metadata))
