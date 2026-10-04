"""Active-rain pristine WRF SFLX oracle through the operational rate producer.

Inputs retain the six real Canary day/night vegetation columns, with prescribed
rainfall and an initially dry canopy. Reference outputs come from the pristine
REAL4 Noah-MP objects; see precipitation_savepoints.json provenance. Existing
integration/energy tolerances are unchanged. New canopy state tolerances are
1e-5 absolute + 1e-5 relative; ECAN uses the energy FCEV tolerance divided by
2.5e6 J/kg. --delete-precip removes the rate producer for a behavioral mutant.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'src'))
import jax.numpy as jnp
import integration_step_gate as gate
from gpuwrf.runtime.noahmp_precipitation import precipitation_from_step, seed_precipitation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--delete-precip', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = json.loads((HERE / 'precipitation_savepoints.json').read_text())
    (args.out / 'savepoints_all.json').write_text(json.dumps(data))
    gate.HERE = args.out
    captured = []
    original = gate.noah_mp_step

    def observed(*a, **kw):
        forcing = kw['forcing']
        # All prescribed precipitation is warm rain in this oracle. These are
        # instantaneous mm amounts, the actual operational producer interface.
        dt = kw['dt']
        amounts = dict(rain=forcing.prcpnonc * dt, snow=jnp.zeros_like(forcing.prcpnonc),
                       ice=jnp.zeros_like(forcing.prcpnonc), graupel=jnp.zeros_like(forcing.prcpnonc))
        rates = precipitation_from_step(amounts, forcing.prcpconv, dt)
        if args.delete_precip:
            rates = seed_precipitation(kw['land_state'])
        kw['forcing'] = forcing._replace(**rates._asdict())
        kw['history'] = True
        result = original(*a, **kw)
        captured.append(result)
        return result

    gate.noah_mp_step = observed
    mapping_rc = gate.main()
    land, fluxes, _ = captured[0]
    columns = data['columns']
    got = {'CANLIQ': np.asarray(land.canliq).ravel(), 'FWET': np.asarray(land.fwet).ravel(),
           'ECAN': np.asarray(fluxes.history['ECAN']).ravel(), 'EVC': np.asarray(fluxes.history['EVC']).ravel()}
    ref = {'CANLIQ': np.array([c['wrf']['water_out']['canliq'] for c in columns]),
           'FWET': np.array([c['wrf']['water_out']['fwet'] for c in columns]),
           'ECAN': np.array([c['wrf']['et']['ecan'] for c in columns]),
           'EVC': np.array([c['wrf']['energy_out']['fcev'] for c in columns])}
    tolerances = {'CANLIQ': (1e-5, 1e-5), 'FWET': (1e-5, 1e-5), 'ECAN': (4e-7, .05), 'EVC': (1., .05)}
    fields = {}
    for name, (atol, rtol) in tolerances.items():
        error = np.abs(got[name] - ref[name])
        passed = np.isfinite(got[name]) & (error <= atol + rtol * np.abs(ref[name]))
        fields[name] = dict(pass_count=int(passed.sum()), count=len(columns),
                            max_abs=float(error.max()), failed_columns=np.flatnonzero(~passed).tolist(),
                            atol=atol, rtol=rtol)
    passed = mapping_rc == 0 and all(f['pass_count'] == f['count'] for f in fields.values())
    report = dict(passed=passed, mapping_rc=mapping_rc, deletion=args.delete_precip, fields=fields)
    (args.out / 'precipitation_gate.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
