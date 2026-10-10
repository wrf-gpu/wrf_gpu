#!/usr/bin/env python3
"""Freeze the module constants produced by the PRISTINE nssl_2mom_init (WRF default mp=18).

Input : proofs/v034/f2_oracles/nssl_2mom/nssl{,_fp64}_case_1.json ("module_vars", dumped by the
        print-only instrumented oracle right after nssl_2mom_init; E26-proven bit-identical build)
Output: src/gpuwrf/physics/nssl2mom/data/init_constants.json
        {"provenance": ..., "decls": {name: [type, kind, dims]}, "fp32": {name: values}, "fp64": {...}}
Large init tables (>4096 elements: tabqvs/tabqis 1e6, ciacrratio..., gamxinflu) are NOT frozen; the
port recomputes them at the use site (mathfun.gaminterp, saturation formulas).
"""
import hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SP = REPO / 'proofs/v034/f2_oracles/nssl_2mom'
OUT = REPO / 'src/gpuwrf/physics/nssl2mom/data/init_constants.json'
decls = {}
for line in (HERE / 'module_vars_dims.txt').read_text().split('\n'):
    if line.strip():
        name, typ, kind, par, dims = line.split()
        decls[name] = [typ, kind, dims]
out = {'provenance': {
    'source': 'WRF phys/module_mp_nssl_2mom.F (pristine)',
    'source_sha256': '29f42e76fbeca027a3ffb2ac805b148aaa1425c7f99ed1d8f19a90ebc1d015f8',
    'init_call': 'nssl_2mom_init(nssl_params=Registry defaults, ipctmp=5, density/hail/ccn on, icdx=icdxhl=6, ccn_is_ccna)',
    'oracle': 'proofs/v034/oracle/nssl2mom (instrumented copy, print-only, E26 bit-identical)'},
    'decls': {}}
for mode, prefix in (('fp32', 'nssl'), ('fp64', 'nssl_fp64')):
    d = json.loads((SP / f'{prefix}_case_1.json').read_text())
    mv = d['module_vars']
    out[mode] = mv
    out['savepoint_sha256_' + mode] = hashlib.sha256((SP / f'{prefix}_case_1.json').read_bytes()).hexdigest()
    for k in mv:
        out['decls'][k] = decls[k]
OUT.write_text(json.dumps(out, sort_keys=True))
print('wrote', OUT, len(out['fp32']), 'vars', OUT.stat().st_size, 'bytes')
