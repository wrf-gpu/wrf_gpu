"""Score GWDO implementations against the pristine WRF oracle columns.

Arms: legacy64 / legacy32 (physics.gwd_gwdo.gwdo_columns in f64 / f32) and
native (kernels.phys_gwdo_column, WRF REAL). Each arm gets the identical
float32 phy_prep columns that the pristine driver consumed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FIELDS = ('rublten', 'rvblten')


def load(path: Path):
    z = np.load(path)
    cols = {k[3:]: z[k] for k in z.files if k.startswith('in_')}
    stat = {k[3:]: z[k] for k in z.files if k.startswith('st_')}
    wrf = {k[4:]: z[k] for k in z.files if k.startswith('wrf_')}
    return cols, stat, wrf, float(z['dt'])


def run_legacy(cols, stat, dt, dtype):
    import jax.numpy as jnp
    from gpuwrf.physics.gwd_gwdo import GWDOColumnState, GWDOStatics, gwdo_columns
    c = {k: jnp.asarray(v.T, dtype) for k, v in cols.items()}
    s = {k: jnp.asarray(v, dtype) for k, v in stat.items()}
    out = gwdo_columns(GWDOColumnState(**c), GWDOStatics(
        var=s['var'], oc1=s['con'], oa1=s['oa1'], oa2=s['oa2'], oa3=s['oa3'], oa4=s['oa4'],
        ol1=s['ol1'], ol2=s['ol2'], ol3=s['ol3'], ol4=s['ol4'], sina=s['sina'],
        cosa=s['cosa'], dxmeter=s['dxmeter']), dt)
    return {f: np.asarray(getattr(out, f)).T for f in FIELDS}


def run_native(cols, stat, dt):
    from gpuwrf.kernels.phys_gwdo_column import gwdo_tendencies_native
    return {k: np.asarray(v) for k, v in gwdo_tendencies_native(cols, stat, dt).items()}


def score(arm, wrf):
    out = {}
    for f in FIELDS:
        ref = wrf[f].astype(np.float64)
        got = arm[f].astype(np.float64)
        err = np.abs(got - ref)
        scale = float(np.abs(ref).max())
        colerr = err.max(axis=0)
        colref = np.abs(ref).max(axis=0)
        bad_rel = colerr > 1e-3 * np.maximum(colref, 1e-6)
        # amended at review: fail only if ALSO abs > 1e-4 x field max (ill-conditioned columns)
        bad = bad_rel & (colerr > 1e-4 * scale)
        out[f] = dict(max_abs_err=float(err.max()), rms_err=float(np.sqrt((err ** 2).mean())),
                      max_rel_to_field_max=float(err.max() / scale) if scale else 0.0,
                      field_max=scale, columns_over_1e3_rel=int(bad_rel.sum()), columns_failing_amended_rule=int(bad.sum()),
                      worst_column=int(colerr.argmax()), nonfinite=int((~np.isfinite(got)).sum()))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--oracle', type=Path, required=True)
    p.add_argument('--arms', default='legacy64,legacy32,native')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    import jax
    jax.config.update('jax_enable_x64', True)
    result = dict(platform=jax.devices()[0].platform, cases={})
    for case in ('night', 'day'):
        cols, stat, wrf, dt = load(args.oracle / case / 'columns.npz')
        active = int((np.abs(wrf['rublten']).max(0) + np.abs(wrf['rvblten']).max(0) > 0).sum())
        rec = dict(active_columns=active)
        for arm in args.arms.split(','):
            if arm == 'legacy64':
                res = run_legacy(cols, stat, dt, np.float64)
            elif arm == 'legacy32':
                res = run_legacy(cols, stat, dt, np.float32)
            else:
                res = run_native(cols, stat, dt)
            rec[arm] = score(res, wrf)
        result['cases'][case] = rec
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
