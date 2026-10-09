"""Prepare actual WRF columns on both PROD shapes for isolated kernel bakeoffs.

Host-only preparation via the existing State/phy_prep coupler. These samples
provide real workload inputs; independent WRF column fixtures remain the oracle.
"""
import argparse
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts/v025'))
from real_state import load_real_snapshot
import jax

from gpuwrf.contracts import state as state_contract
from gpuwrf.coupling import physics_couplers as C
from gpuwrf.physics import mynn_edmf as E, mynn_pbl as P

# CPU-only fixture preparation; State's normal GPU placement remains unchanged.
if jax.default_backend() != 'cpu':
    raise RuntimeError('prepare_mynn_columns requires JAX_PLATFORMS=cpu')
state_contract._gpu_device = lambda: jax.devices('cpu')[0]


def prepare(case, output, domain, file, whole=False):
    snapshot = load_real_snapshot(case, domain=domain, wrfout_name=file)
    if 'QKE' in snapshot.provenance['absent_variables']:
        raise RuntimeError('real turbulence field QKE is missing')
    grid = SimpleNamespace(metrics=snapshot.metrics)
    column = C._mynn_column_from_state(snapshot.state, grid)
    ny, nx, nz = column.theta.shape
    column = C._flatten_columns_to_batch(column, ny, nx)
    flux = C._flatten_columns_to_batch(C._surface_fluxes_from_state(snapshot.state), ny, nx)
    pblh = P._get_pblh(column, 2.0 * column.tke)
    arguments = {}
    original = E.dmp_mf_columns

    def collect(*args, **kwargs):
        arguments.update(inspect.signature(original).bind(*args, **kwargs).arguments)
        return None

    E.dmp_mf_columns = collect
    try:
        P._edmf_arrays_from_state(column, flux, flux.fltv, pblh,
                                 54.0 if domain == 'd01' else 18.0,
                                 9000.0 if domain == 'd01' else 3000.0)
    finally:
        E.dmp_mf_columns = original
    arrays = {key: np.asarray(value) for key, value in arguments.items()}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError('nonfinite workload inputs')
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / f'{domain}.npz', **arrays)
    if whole:
        whole_arrays = {f'state_{name}': np.asarray(getattr(column, name))
                        for name in column.__slots__ if getattr(column, name) is not None}
        whole_arrays.update({f'flux_{name}': np.asarray(getattr(flux, name))
                             for name in flux._fields})
        if not all(np.isfinite(value).all() for value in whole_arrays.values()):
            raise RuntimeError('nonfinite whole-MYNN inputs')
        np.savez(output / f'{domain}_whole.npz', **whole_arrays)
    meta = dict(provenance=snapshot.provenance, shape=[ny,nx,nz],
                workload='real CPU-WRF output transformed by retained phy_prep coupler',
                surface_flux_note=snapshot.provenance['surface_flux_handles'],
                dx=float(arrays['dx']), dt=float(arrays['dt']),
                oracle_scope='workload inputs only; not independent expected output')
    (output / f'{domain}.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(domain, meta['shape'], meta['dx'], meta['dt'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--whole', action='store_true', help='Also save the complete MYNN state and surface flux')
    parser.add_argument('--valid-time', help='Exact timestamp in the WRF history filename')
    args = parser.parse_args()
    for domain in ('d01', 'd02'):
        pattern = f'wrfout_{domain}_{args.valid_time}' if args.valid_time else f'wrfout_{domain}_*'
        files = sorted(args.case.glob(pattern))
        if not files:
            raise FileNotFoundError(pattern)
        file = files[0].name
        prepare(args.case, args.output, domain, file, args.whole)
