"""Prepare real d01 KF columns using the current production adapter wiring."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--case', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--valid-time', required=True)
args = p.parse_args()
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/v025'))
from real_state import load_real_snapshot
import jax
import jax.numpy as jnp
from gpuwrf.contracts import state as S
from gpuwrf.coupling import scan_adapters as A

assert jax.devices()[0].platform == 'cpu'
S._gpu_device = lambda: jax.devices('cpu')[0]
snapshot = load_real_snapshot(args.case, domain='d01',
                              wrfout_name=f'wrfout_d01_{args.valid_time}')
state = snapshot.state
nz, ny, nx = state.theta.shape
columns = lambda x: np.asarray(jnp.moveaxis(x, 0, -1)).reshape(ny * nx, nz)
rho = A._wrf_phy_prep_rho_from_state(state, snapshot.metrics)
temp = A._temperature_from_theta(A._dry_theta_view(state), state.p)
z = state.ph.astype(jnp.float64) / A.GRAVITY_M_S2
arrays = dict(T0=columns(temp), QV0=columns(state.qv), P0=columns(state.p),
              DZQ=columns(jnp.maximum(z[1:] - z[:-1], 1.0)), RHOE=columns(rho),
              U0=columns(A._u_mass(state)), V0=columns(A._v_mass(state)),
              w=columns(A._w_mass(state)))
arrays['w0avg'] = np.zeros_like(arrays['w'])
arrays['nca'] = np.full(ny * nx, -100.0)
assert all(np.isfinite(value).all() for value in arrays.values())
args.output.mkdir(parents=True, exist_ok=True)
np.savez(args.output / 'd01.npz', **arrays)
metadata = dict(provenance=snapshot.provenance, shape=[ny, nx, nz], dt=54.0,
                dx=9000.0, product_active=True,
                preparation='kf_adapter: dry theta + grid-backed WRF phy_prep total rho',
                carry='fresh w0avg=0,nca=-100; step_kf_column applies CPS mean',
                input_sha256=hashlib.sha256((args.output / 'd01.npz').read_bytes()).hexdigest())
(args.output / 'd01.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps(metadata))
