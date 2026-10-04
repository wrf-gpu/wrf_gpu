"""B45: flow_dep_bdy vs unchanged pristine WRF REAL4 on real PROD d01 edge columns (CPU).

Operands: real d01 qv/qc/qr/Ni fields and the real u/v (signs drive inflow /
outflow on all four sides). The operator is copy/select, so the gate is
BIT-EXACT on REAL values. Controls that must FAIL: flipped velocity signs,
and the released behaviour (spec+relax of the wrfbdy record).
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests/v025/b_diff')]
os.environ.setdefault('JAX_ENABLE_X64', 'true')
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.coupling import boundary_apply as bd
from pristine_boundary import Oracle, build

DATA = Path('<USER_HOME>/wrf_gpu2_lanes/b-diff/boundary_fixtures/d01.npz')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert jax.devices()[0].platform == 'cpu'
    z = np.load(DATA)
    get = lambda n: z['state_'+n] if 'state_'+n in z.files else z['physical_'+n]
    u, v = get('u'), get('v')
    cfg = bd.BoundaryConfig()
    path = build(a.out.parent/'pristine')
    rows = {}
    for name in ('qv', 'qc', 'qr', 'Ni'):
        field = get(name)
        oracle = Oracle(path, field.shape)
        oracle.set('field', field)
        oracle.set('ru', u)
        oracle.set('rv', v)
        oracle.run('flow_dep_bdy')
        ref = oracle.field('field', field.shape)
        got = np.asarray(bd.flow_dep_bdy(jnp.asarray(field), jnp.asarray(u), jnp.asarray(v), cfg)).astype(np.float32)
        flipped = np.asarray(bd.flow_dep_bdy(jnp.asarray(field), -jnp.asarray(u), -jnp.asarray(v), cfg)).astype(np.float32)
        released = np.asarray(bd._apply_3d(jnp.asarray(field), jnp.asarray(get(name+'_bdy')), 54.0, 54.0, cfg)).astype(np.float32)
        ring = np.zeros(field.shape[1:], bool)
        ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
        nonzero_edge = int(np.count_nonzero(np.asarray(field)[:, ring]))
        outflow = {'S': int((v[:, 0, :] < 0).sum()), 'N': int((v[:, -1, :] > 0).sum()),
                   'W': int((u[:, :, 0] < 0).sum()), 'E': int((u[:, :, -1] > 0).sum())}
        rows[name] = dict(bit_exact=bool(np.array_equal(got, ref)),
                          max_abs=float(np.max(np.abs(got-ref))),
                          interior_untouched=bool(np.array_equal(got[:, ~ring], np.asarray(field, np.float32)[:, ~ring])),
                          control_flipped_fails=not np.array_equal(flipped, ref),
                          control_released_fails=not np.array_equal(released, ref),
                          outflow_cells_by_side=outflow, nonzero_ring_values=nonzero_edge)
        print(name, rows[name], flush=True)
    # Fields with zero edge values (hydrometeors at t=0) are exact but not
    # discriminating; the controls must fail on every field that has ring data.
    discriminating = [n for n, r in rows.items() if r['nonzero_ring_values']]
    passed = bool(discriminating) and all(r['bit_exact'] and r['interior_untouched'] for r in rows.values()) and all(
        rows[n]['control_flipped_fails'] and rows[n]['control_released_fails'] for n in discriminating)
    result = dict(passed=passed, rows=rows, discriminating=discriminating, fixture=str(DATA), platform='cpu',
                  scope='pristine module_bc.F flow_dep_bdy (gfortran REAL4, -ffp-contract=off) vs port, real PROD d01 fields + u/v')
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
