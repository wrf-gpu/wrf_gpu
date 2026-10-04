"""Per-entrypoint device time/launches of the whole owned boundary family (legacy vs native).

All entrypoints are compiled and warmed first; the capture then runs each one for
CALLS calls separated by >=50 ms host sleeps, so the parser splits the kernel
timeline at idle gaps (> GAP_MS) into one block per entrypoint, in CASES order.
Standalone component evidence only; no whole-forecast speed claim.
"""
import argparse
import ctypes
import json
import statistics
import sys
import time
from pathlib import Path

source = Path(sys.argv[sys.argv.index('--source')+1])
sys.path.insert(0, str(source/'tests/v025/b_diff'))
import probe_boundary as probe  # resolves --arm before importing gpuwrf/JAX

CALLS = 10
GAP_S = 0.05


def cases(domain):
    jax, jnp, np, bd = probe.jax, probe.jnp, probe.np, probe.boundary
    state, m, cfg, meta = probe.load(domain, 'physical' if domain == 'd01' else 'coupled')
    dt = meta['resolved_dt_s']
    nz, ny, nx = state.theta.shape
    coupled = nested = domain == 'd02'
    lead = jnp.asarray(dt, jnp.float64)
    leafu, leafv, leafph, leafw = [getattr(state, n+'_bdy')[0] for n in ('u', 'v', 'ph', 'w')]
    mu = np.asarray(state.mu_total)
    u_mu = .5*(np.concatenate([mu[:, :1], mu], 1)+np.concatenate([mu, mu[:, -1:]], 1))
    v_mu = .5*(np.concatenate([mu[:1], mu], 0)+np.concatenate([mu, mu[-1:]], 0))
    massu = jnp.asarray(np.asarray(m.c1h)[:, None, None]*u_mu[None]+np.asarray(m.c2h)[:, None, None])
    massv = jnp.asarray(np.asarray(m.c1h)[:, None, None]*v_mu[None]+np.asarray(m.c2h)[:, None, None])
    table = {
        'dry': (lambda s, m, l: probe.dry(s, m, cfg, l, dt, coupled, nested), (state, m, lead)),
        'lateral': (lambda s, m, l: bd.apply_lateral_boundaries(s, l, dt, cfg, m), (state, m, lead)),
        'lateral_spec_only': (lambda s, m, l: bd.apply_lateral_boundaries(s, l, dt, cfg, m, dry_spec_only=True), (state, m, lead)),
        'interpolate': (lambda leaf, l: bd.interpolate_boundary_leaf(leaf, l, cfg.update_cadence_s), (state.u_bdy, lead)),
        'specified': (lambda leaf, l: bd.specified_boundary_tendency(leaf, l, cfg.update_cadence_s, z_len=nz, y_len=ny,
                      x_len=nx+1, dtype=jnp.float64, config=cfg), (state.u_bdy, lead)),
        'normal_u': (lambda leaf, save, mass, mapf: bd.normal_bdy_work_target_u(leaf, save, mass, mass, mapf, config=cfg,
                     coupled_boundary_leaves=coupled), (leafu, state.u, massu, m.msfuy)),
        'normal_v': (lambda leaf, save, mass, mapf: bd.normal_bdy_work_target_v(leaf, save, mass, mass, mapf, config=cfg,
                     coupled_boundary_leaves=coupled), (leafv, state.v, massv, m.msfvx)),
        'tangential_u': (lambda leaf, save, mass, mapf: bd.tangential_bdy_work_target_u(leaf, save, mass, mass, mapf, config=cfg,
                         coupled_boundary_leaves=coupled), (leafu, state.u, massu, m.msfuy)),
        'tangential_v': (lambda leaf, save, mass, mapf: bd.tangential_bdy_work_target_v(leaf, save, mass, mass, mapf, config=cfg,
                         coupled_boundary_leaves=coupled), (leafv, state.v, massv, m.msfvx)),
        'normal_apply': (lambda u, v: bd.apply_normal_bdy_work(u, v, u, v, dt/4, dt, config=cfg), (state.u, state.v)),
        'ph_relax': (lambda ph, leaf, mu, mapf, c1, c2: bd.nested_ph_relax_tendency(ph, leaf, mu, mapf, c1, c2, dt, cfg),
                     (state.ph_perturbation, leafph, state.mu_total, m.msfty, m.c1f, m.c2f)),
        'w_relax': (lambda w, leaf, mu, mapf, c1, c2: bd.nested_w_relax_tendency(w, leaf, mu, mapf, c1, c2, dt, cfg),
                    (state.w, leafw, state.mu_total, m.msfty, m.c1f, m.c2f)),
        'ph_spec_legacy': (lambda ph, mu, c1, c2: bd.spec_bdyupdate_ph_inloop(ph, ph, ph, mu, mu, c1, c2, dt/4, cfg),
                           (state.ph_perturbation, state.mu_total, m.c1f, m.c2f)),
        'ph_spec_live': (lambda ph, mu, c1, c2: bd.spec_bdyupdate_ph_tendency_inloop(ph, ph, ph, ph, mu, mu, c1, c2, dt/4, cfg),
                         (state.ph_perturbation, state.mu_total, m.c1f, m.c2f)),
    }
    if probe.arm == 'native':
        # Fails closed under the flag (non-bundle nest ph relax; not on the PROD path).
        table.pop('ph_relax')
    return {name: (jax.jit(fn), args) for name, (fn, args) in table.items()}, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('native', 'legacy'), required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--domain', choices=('d01', 'd02'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--cpu-smoke', action='store_true')
    args = parser.parse_args()
    jax = probe.jax
    assert jax.devices()[0].platform == ('cpu' if args.cpu_smoke else 'gpu'), jax.devices()
    table, meta = cases(args.domain)
    wall = {}
    for name, (fn, call_args) in table.items():
        for _ in range(3):
            jax.block_until_ready(fn(*call_args))
        samples = []
        for _ in range(CALLS):
            start = time.perf_counter_ns()
            jax.block_until_ready(fn(*call_args))
            samples.append((time.perf_counter_ns()-start)/1000)
        wall[name] = statistics.median(samples)
    record = {'platform': jax.devices()[0].platform, 'device': str(jax.devices()[0]), 'arm': args.arm,
              'domain': args.domain, 'immutable_source': str(args.source), 'fixture_sha256': meta['fixture_sha256'],
              'cases': list(table), 'calls_per_case': CALLS, 'gap_s': GAP_S, 'wall_us_median': wall,
              'scope': 'standalone per-entrypoint boundary family; no whole-forecast speed claim'}
    args.out.write_text(json.dumps(record, indent=2)+'\n')
    if args.profile:
        cudart = ctypes.CDLL('/usr/local/cuda/lib64/libcudart.so')
        assert cudart.cudaProfilerStart() == 0
        for name, (fn, call_args) in table.items():
            time.sleep(GAP_S)
            for _ in range(CALLS):
                jax.block_until_ready(fn(*call_args))
        time.sleep(GAP_S)
        assert cudart.cudaProfilerStop() == 0
        record['profiled'] = True
        args.out.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'cases'}), flush=True)


if __name__ == '__main__':
    main()
