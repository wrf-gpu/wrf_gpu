"""Prepare real PROD day/night surface columns with the retained land loader.

History files are exposed as input files only in a private directory. All source
files remain read-only. Metadata records the loader's cold-init fields and the
zero precipitation rate (instantaneous rates are absent from history output).
These are workloads; the frozen WRF savepoints are the independent field gate.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys

import numpy as np
from netCDF4 import Dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True, type=Path)
    p.add_argument('--case', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    sys.path.insert(0, str(args.repo / 'src'))
    import gpuwrf
    import jax
    import jax.numpy as jnp
    from gpuwrf.io.noahmp_land_init import build_noahmp_land_state
    from gpuwrf.physics.noahmp.noahmp_driver import build_energy_params
    from gpuwrf.physics.noahmp.types import NoahMPForcing
    assert jax.devices()[0].platform == 'cpu'
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {}
    for label, timestamp in [('night','2026-07-26_00:00:00'),
                             ('day','2026-07-26_12:00:00')]:
        for domain in ('d01','d02'):
            if Path('/tmp/wrf_gpu2_quiet').exists():
                return 125
            history = args.case / f'wrfout_{domain}_{timestamp}'
            private = args.out / f'{label}_{domain}'
            private.mkdir(exist_ok=True)
            cached = private / 'columns.pkl'
            if cached.exists():
                with cached.open('rb') as f:
                    payload[f'{label}_{domain}'] = pickle.load(f)
                continue
            for name, target in [(f'wrfinput_{domain}', history),
                                 ('namelist.input', args.case/'namelist.input')]:
                link = private / name
                if not link.exists():
                    link.symlink_to(target)
            land, static, meta = build_noahmp_land_state(private, domain=domain)
            with Dataset(history) as d:
                def a(name):
                    return np.asarray(d.variables[name][0], dtype=np.float64)
                pressure = (a('P')+a('PB'))[0]
                qv = a('QVAPOR')[0]
                temperature = (a('T')[0]+300)*(pressure/1.e5)**(287./1004.5)
                u, v = a('U')[0], a('V')[0]
                geopt = a('PH')+a('PHB')
                dz = (geopt[1]-geopt[0])/9.81
                zero = np.zeros_like(pressure)
                forcing = NoahMPForcing(
                    sfctmp=temperature, sfcprs=pressure, psfc=a('PSFC'),
                    uu=.5*(u[:,:-1]+u[:,1:]), vv=.5*(v[:-1,:]+v[1:,:]),
                    qair=qv/(1+qv), qc=a('QCLOUD')[0], soldn=a('SWDOWN'),
                    lwdn=a('GLW'), cosz=a('COSZEN'), zlvl=.5*dz,
                    prcpconv=zero, prcpnonc=zero, prcpsnow=zero,
                    prcpgrpl=zero, prcphail=zero,
                    # Operational WRF clock is zero-based fractional year-day.
                    julian=np.asarray(206. if label=='night' else 206.5),
                    yearlen=np.asarray(365.))
            ep, rp = build_energy_params(static, land.tv.shape)
            item = (land, forcing, static, ep, rp, 54. if domain=='d01' else 18.)
            payload[f'{label}_{domain}'] = jax.tree.map(
                lambda x: np.asarray(x) if hasattr(x,'dtype') else x, item)
            with cached.open('wb') as f:
                pickle.dump(payload[f'{label}_{domain}'],f,protocol=5)
            manifest = dict(history=str(history), shape=list(land.tv.shape),
                            loader_metadata=meta, dt=item[-1],
                            forcing='dry WRF T+300 and pressure Exner, QVAPOR/(1+QVAPOR), staggered mean U/V, PH-derived ZLVL',
                            precipitation='instantaneous rates absent from history; zero rates',
                            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            scope='real workload; frozen external savepoints score fidelity')
            (private/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str)+'\n')
            print(label,domain,land.tv.shape,flush=True)
    with (args.out/'columns.pkl').open('wb') as f:
        pickle.dump(payload,f,protocol=5)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
