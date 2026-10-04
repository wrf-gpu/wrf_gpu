"""Score the pre-registered native balance rule against pristine WRF REAL.

Registration NP04/water_gate_registration.json predates both measurements.
The existing fp64 gate is loaded unchanged and its original verdict retained.
The pristine SOILWATER caller and native standalone boundary use BTRANI=0.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--oracle', type=Path, required=True)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--backend', choices=['cpu', 'gpu'], required=True)
    args = p.parse_args()
    if Path('/tmp/wrf_gpu2_quiet').exists():
        return 125
    sys.path.insert(0, str(args.repo / 'src'))
    import gpuwrf
    import jax
    from gpuwrf.physics.noahmp.precision import native_real_enabled
    assert native_real_enabled()
    assert jax.devices()[0].platform == args.backend
    path = args.repo / 'proofs/noahmp/water_savepoint_gate.py'
    spec = importlib.util.spec_from_file_location('unchanged_water', path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    original = gate.noahmp_water_hydro
    captured = {}

    def capture(land, forcing, static, et, dt):
        result = original(land, forcing, static, et, dt)
        captured.update(land=land, result=result, dt=dt)
        return result

    gate.noahmp_water_hydro = capture
    args.out.mkdir(parents=True, exist_ok=True)
    fixture = args.out / 'savepoints_all.json'
    if not fixture.exists():
        fixture.symlink_to(path.parent / fixture.name)
    gate.HERE = args.out
    original_rc = gate.main()
    cols = json.loads(fixture.read_text())['columns']
    oracle = np.loadtxt(args.oracle)
    assert oracle.shape == (len(cols), 12)
    assert np.array_equal(oracle[:, 0], np.arange(1, len(cols)+1))
    land, result = captured['land'], captured['result']
    initial = np.asarray(land.smois, dtype=np.float64).reshape(4, -1)
    smc = np.asarray(result.smois, dtype=np.float64).reshape(4, -1)
    runsrf = np.asarray(result.sfcrunoff, dtype=np.float64).reshape(-1)*1000
    runsub = np.asarray(result.udrunoff, dtype=np.float64).reshape(-1)*1000
    rows = []
    for i, c in enumerate(cols):
        sink = 0 if c['state_in']['sneqv'] > 0 else -c['wrf']['et']['edir']*captured['dt']
        residual = abs(float(np.dot(smc[:, i]-initial[:, i], [.05,.20,.45,.80])*1000 + runsrf[i]+runsub[i]-sink))
        wrf_residual = abs(float(np.dot(oracle[i, 1:5]-initial[:, i], [.05,.20,.45,.80])*1000 + oracle[i, 9]+oracle[i, 10]-sink))
        limit = min(1.5*wrf_residual, .1)
        rows.append(dict(name=c['name'], native_residual_mm=residual,
                         pristine_real_residual_mm=wrf_residual, limit_mm=limit,
                         pristine_runtime_real_residual_mm=float(oracle[i, 11]),
                         pass_=bool(np.isfinite(residual) and residual <= limit),
                         smc_max_abs_delta=float(np.max(np.abs(smc[:, i]-oracle[i, 1:5])))))
    report = dict(backend=args.backend, device=str(jax.devices()[0]),
                  registration=json.loads(args.registration.read_text()),
                  registration_sha256=hashlib.sha256(args.registration.read_bytes()).hexdigest(),
                  original_fp64_gate_rc=original_rc,
                  oracle_sha256=hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
                  fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                  columns=rows, npass=sum(r['pass_'] for r in rows),
                  ncolumns=len(rows), all_pass=all(r['pass_'] for r in rows))
    (args.out / 'registered_water_real_gate.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    return int(not report['all_pass'])


if __name__ == '__main__':
    raise SystemExit(main())
