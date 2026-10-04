"""CPU-only real WRF inputs through the retained KF column preparation."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/v025'))
from real_state import load_real_snapshot
import jax
import jax.numpy as jnp
from gpuwrf.contracts import state as S
from gpuwrf.coupling import scan_adapters as A

S._gpu_device = lambda: jax.devices('cpu')[0]


def prepare(case, output, domain, time):
    snapshot = load_real_snapshot(case, domain=domain, wrfout_name=f'wrfout_{domain}_{time}')
    state = snapshot.state
    nz, ny, nx = state.theta.shape
    columns = lambda x: np.asarray(jnp.moveaxis(x, 0, -1)).reshape(ny*nx,nz)
    rho = A._rho_from_state(state)
    temperature = A._temperature_from_theta(state.theta, state.p)
    z = state.ph.astype(jnp.float64) / A.GRAVITY_M_S2
    dz = jnp.maximum(z[1:] - z[:-1],1.0)
    arrays = dict(T0=columns(temperature), QV0=columns(state.qv), P0=columns(state.p),
                  DZQ=columns(dz), RHOE=columns(rho), U0=columns(A._u_mass(state)),
                  V0=columns(A._v_mass(state)), w=columns(A._w_mass(state)))
    # Exact retained fresh-cloud carry; both arms use the same seeded inputs.
    arrays['w0avg'] = np.zeros_like(arrays['w'])
    arrays['nca'] = np.full(ny*nx,-100.0)
    assert all(np.isfinite(v).all() for v in arrays.values())
    output.mkdir(parents=True,exist_ok=True)
    np.savez(output/f'{domain}.npz',**arrays)
    metadata = dict(provenance=snapshot.provenance,shape=[ny,nx,nz],
                    preparation='retained scan_adapters.kf_adapter expressions',
                    carry='fresh initial_kf_carry: w0avg=0, nca=-100',
                    dt=54.0 if domain=='d01' else 18.0,
                    dx=9000.0 if domain=='d01' else 3000.0,
                    product_active=domain=='d01',
                    note='d02 size is a scalability probe; KF is disabled there in PROD')
    (output/f'{domain}.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(domain,metadata['shape'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--valid-time',default='2026-07-26_00:00:00')
    args=parser.parse_args()
    for domain in ('d01','d02'):
        prepare(args.case,args.output,domain,args.valid_time)
