"""Additional pristine boundary gates, using real boundary/history operands."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.coupling import boundary_apply as boundary
from pristine_boundary import Oracle
from bench_acoustic import compare


def verify_spec(state, cfg, path, *, lead, dt, nested):
    results = {}
    fields = (('u', state.u, 1), ('v', state.v, 2),
              ('theta', state.theta, 4), ('ph', state.ph_perturbation, 3),
              ('mu', state.mu_perturbation[None], 5))
    if nested:
        fields += (('w', state.w, 7),)
    for name, field, tag in fields:
        records = np.asarray(getattr(state, name+'_bdy'), np.float32)
        rate = (records[1]-records[0])/np.float32(cfg.update_cadence_s)
        oracle = Oracle(path, state.theta.shape, nested=nested, dt=dt, dtbc=lead)
        oracle.boundary('scalar', records[0], rate)
        oracle.run('spec_bdytend', tag=tag)
        got = jax.jit(lambda b: boundary.specified_boundary_tendency(
            b, lead, cfg.update_cadence_s, z_len=field.shape[0],
            y_len=field.shape[1], x_len=field.shape[2], dtype=field.dtype,
            config=cfg))(getattr(state, name+'_bdy'))
        ref = oracle.field('field_tend', field.shape)
        results[name] = compare(ref, got)
    return results


def verify_scalars(state, metrics, cfg, path, *, lead, dt):
    got = jax.jit(lambda s, m: boundary.nested_scalar_boundary_tendencies(
        s, lead, m, dt, cfg))(state, metrics)
    jax.block_until_ready(got)
    results = {}
    for name, output in zip(boundary.NESTED_BOUNDARY_SCALAR_SPECIES, got, strict=True):
        leaf = getattr(state, name+'_bdy', None)
        if leaf is None:
            # The real producer currently exports QV only. Do not invent zero
            # boundary records and report them as a pristine-WRF scalar gate.
            # This checks the retained interface's absent-leaf branch only.
            results[name] = {'passed': bool(np.all(np.asarray(output) == 0)),
                             'oracle_check': False, 'boundary_leaf_present': False,
                             'scope': 'absent producer leaf; zero tendency interface regression'}
            continue
        oracle = Oracle(path, state.theta.shape, nested=True, dt=dt, dtbc=lead)
        for vector in ('c1h', 'c2h', 'c1f', 'c2f'):
            oracle.set(vector, getattr(metrics, vector))
        oracle.set('scalar', getattr(state, name))
        oracle.set('mu', state.mu_total)
        records = np.asarray(getattr(state, name+'_bdy'), np.float32)
        rate = (records[1]-records[0])/np.float32(cfg.update_cadence_s)
        oracle.boundary('scalar', records[0], rate)
        oracle.run('relax_bdy_scalar')
        relax = oracle.field('scalar_tend', output.shape)
        oracle.run('spec_bdytend')
        reference = relax + oracle.field('field_tend', output.shape)
        results[name] = compare(reference, output)
        results[name].update(oracle_check=True, boundary_leaf_present=True)
    return results


def verify_ph_history(domain, cfg, path):
    """Coherently changed MU/MUTS from the same observed WRF history pair.

    This is an operator fixture spanning two history endpoints, not an internal
    acoustic savepoint. Each RK acoustic dt receives its own coherent live mass
    and distinct pre/post-advance ph work. The unchanged Fortran routine reads
    the pre-advance ring; its interior is replaced by the post-advance array,
    reproducing advance_w's boundary exclusion at the caller.
    """
    data_dir = Path('<USER_HOME>/wrf_gpu2_lanes/b-diff/rk_fixtures')
    meta = json.loads((data_dir/(domain+'.json')).read_text())
    fraction = meta['pair']['dt_s']/meta['pair']['snapshot_interval_seconds']
    with np.load(data_dir/(domain+'.npz')) as data:
        def endpoints(name):
            current = data['ref_'+name]
            previous = current-(data['state_'+name]-current)/fraction
            return np.asarray(previous, np.float32), np.asarray(current, np.float32)
        mu_old, mu_new = endpoints('mu_total')
        ph_old, ph_new = endpoints('ph_perturbation')
        c1, c2 = np.asarray(data['metric_c1f'], np.float32), np.asarray(data['metric_c2f'], np.float32)
    assert np.any(mu_new != mu_old), 'Mass-evolution gate must change live mass'
    before = np.float32(fraction)*(ph_new-ph_old)
    advanced = ph_new-ph_old
    assert np.any(before != advanced), 'Interior-preservation gate needs distinct arrays'
    nz = ph_old.shape[0]-1
    results = {}
    for rk, trips in ((1, 1), (2, 2), (3, 4)):
        dts = np.float32(meta['pair']['dt_s']/trips)
        mu_tend = (mu_new-mu_old)/dts
        coherent_old = mu_new-dts*mu_tend
        mass_old = c1[:, None, None]*coherent_old[None]+c2[:, None, None]
        mass_new = c1[:, None, None]*mu_new[None]+c2[:, None, None]
        ph_tend = (mass_new*ph_new-mass_old*ph_old)/dts
        oracle = Oracle(path, (nz, *mu_old.shape), nested=domain=='d02', dts=float(dts))
        for name, value in (('ph', before), ('ph_save', ph_old), ('ph_tendf', ph_tend),
                            ('mu_tend', mu_tend), ('muts', mu_new), ('c1f', c1), ('c2f', c2)):
            oracle.set(name, value)
        oracle.run('spec_bdyupdate_ph', tag=3)
        ref = oracle.field('ph', ph_old.shape)
        ref[:, 1:-1, 1:-1] = advanced[:, 1:-1, 1:-1]
        args = tuple(jnp.asarray(x) for x in (advanced, before, ph_tend, ph_old, mu_tend, mu_new, c1, c2))
        fn = jax.jit(lambda *a: boundary.spec_bdyupdate_ph_tendency_inloop(*a, float(dts), cfg))
        output = fn(*args)
        score = compare(ref, output)
        score['interior_exact'] = bool(np.array_equal(np.asarray(output)[:, 1:-1, 1:-1], advanced[:, 1:-1, 1:-1]))
        score['passed'] = score['passed'] and score['interior_exact']
        score['changed_mass_cells'] = int(np.count_nonzero(mu_new-coherent_old))
        stale = list(args)
        stale[4] = jnp.zeros_like(args[4])
        collapsed = list(args)
        collapsed[0] = args[1]
        score['deleted_mass_walk'] = compare(ref, fn(*stale))
        score['deleted_interior_advance'] = compare(ref, fn(*collapsed))
        score['deletion_sensitive'] = (not score['deleted_mass_walk']['passed']
                                      and not score['deleted_interior_advance']['passed'])
        score['passed'] = score['passed'] and score['deletion_sensitive']
        results[str(rk)] = score
    return {'cases': results, 'pair': meta['pair'],
            'limitation': 'observed history-endpoint operator fixture, not internal acoustic savepoints'}
