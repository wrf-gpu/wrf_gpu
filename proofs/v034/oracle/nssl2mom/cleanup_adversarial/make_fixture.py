#!/usr/bin/env python3
"""Collect the adversarial harness outputs into cleanup_adversarial_{fp32,fp64}.json."""
import hashlib, json, re
from pathlib import Path

H = Path(__file__).resolve().parent
rows = H.joinpath('adv_in.txt').read_text().split('\n')
n = int(rows[0])
vals = [[float(x) for x in r.split()] for r in rows[1:1 + n]]
an_in = [[v[il] for v in vals] for il in range(18)]
dn = [v[18] for v in vals]
t77 = [v[19] for v in vals]
for mode in ('fp32', 'fp64'):
    txt = (H / f'build_{mode}' / 'out.txt').read_text().split('\n')
    cq, sv, t0, dbz, re_ = {}, {}, {}, {}, {}
    for line in txt:
        m = re.match(r'^(CQ|SV):(\d+)=(.*)$', line)
        if m:
            (cq if m.group(1) == 'CQ' else sv)[int(m.group(2))] = [float(x) for x in m.group(3).split()]
            continue
        m = re.match(r'^(SVT0|DBZ|RE):(\d+)=(.*)$', line)
        if m:
            d = {'SVT0': t0, 'DBZ': dbz, 'RE': re_}[m.group(1)]
            d[int(m.group(2))] = [float(x) for x in m.group(3).split()]
    out = {
        'schema': 'wrf-v034-o1-nssl2mom-cleanup-adversarial-v1', 'mode': mode, 'n': n, 'dtp': 54.0,
        'provenance': 'pristine module_mp_nssl_2mom.F (sha256 29f42e76...) + one public statement; '
                      'harness cleanup_dev/adv/cleanup_adv.F90; inputs gen_inputs.py (seed 20261010)',
        'an_in': an_in, 'dn': dn, 't77': t77,
        'calcnfromq': [[cq[k][il] for k in range(1, n + 1)] for il in range(18)],
        'smallvalues': [[sv[k][il] for k in range(1, n + 1)] for il in range(18)],
        'smallvalues_t0': [t0[k][0] for k in range(1, n + 1)],
        'dbz': [dbz[k][0] for k in range(1, n + 1)],
        're': [re_[k] for k in range(1, n + 1)],
    }
    p = H / f'cleanup_adversarial_{mode}.json'
    p.write_text(json.dumps(out))
    print('wrote', p, hashlib.sha256(p.read_bytes()).hexdigest()[:12])
