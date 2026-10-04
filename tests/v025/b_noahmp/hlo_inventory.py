"""Audit whole-column REAL arithmetic and preserve default-off HLO for comparison.

Uses an actual frozen daytime WRF column. Preoptimized CPU HLO is an inventory,
not a GPU operation count or timing measurement.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--modes', default='0,1')
    args = p.parse_args()
    if Path('/tmp/wrf_gpu2_quiet').exists():
        return 125
    sys.path.insert(0, str(args.repo/'src'))
    import gpuwrf
    import jax
    import jax.numpy as jnp
    assert jax.devices()[0].platform == 'cpu'
    proof = args.repo/'proofs/noahmp/energy_savepoint_gate.py'
    spec = importlib.util.spec_from_file_location('energy_frozen_builder',proof)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    from gpuwrf.physics.noahmp.noahmp_driver import noah_mp_step
    cols = json.loads((proof.parent/'savepoints_energy.json').read_text())['columns']
    col = next(c for c in cols if c['name']=='daytime_veg9')
    full = next(c for c in json.loads((proof.parent/'savepoints_all.json').read_text())['columns'] if c['name']==col['name'])
    state, forcing = gate.build_state(col), gate.build_forcing(col)
    static = gate.build_static(col).replace(parameters=gate._P,
        shdmax=jnp.full((1,1),full['shdmax']), shdfac=jnp.full((1,1),full['shdfac']))
    ep,rp = gate.build_params(col['vegtyp'],col['isltyp'])
    report = {}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    for mode in args.modes.split(','):
        assert mode in ('0','1')
        os.environ['GPUWRF_NOAHMP_NATIVE_REAL'] = mode
        jax.clear_caches()
        lowered = jax.jit(lambda ls,ff:noah_mp_step(ls,ff,static,90.,energy_params=ep,rad_params=rp)).lower(state,forcing)
        text = lowered.compiler_ir('hlo').as_hlo_text()
        path = args.out.with_name(args.out.stem+'_'+mode+'.hlo')
        path.write_text(text)
        ops = []
        for line in text.splitlines():
            m = re.search(r'= f64\[[^]]*\](?:\{[^}]*\})? ([a-z-]+)\(',line)
            if m and m[1] not in ('parameter','constant','convert','copy','get-tuple-element'):
                ops.append(m[1])
        report[mode] = dict(f64_compute_count=len(ops),opcodes=dict(Counter(ops)),
                            hlo_sha256=hashlib.sha256(text.encode()).hexdigest(),
                            scope='preoptimized CPU whole-column actual dayveg9; boundary parameters excluded')
        print(mode,report[mode],flush=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
