"""Dense ring-stencil scalar relax+spec kernel vs unchanged pristine WRF REAL4.

Operands are literal REAL caller values (WRF mass_weight order, bdy + dtbc*tend
endpoint); WRF relax_bdy_scalar + spec_bdytend own the stencil. Deletion
controls (no spec write, zero gcx) must fail. --mode cpuverify uses the
test-only RN emulation; --mode verify is the asserted-GPU gate.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests/v025/b_core'), str(ROOT/'tests/v025/b_diff')]
os.environ['GPUWRF_BOUNDARY_FP32'] = '1'
os.environ.setdefault('JAX_ENABLE_X64', 'true')
import jax
import jax.numpy as jnp
import numpy as np
import probe_boundary as probe
from pristine_boundary import Oracle, build
from bench_acoustic import compare


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('cpuverify', 'verify'), required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    expected = 'cpu' if a.mode == 'cpuverify' else 'gpu'
    assert jax.devices()[0].platform == expected, jax.devices()
    from gpuwrf.kernels import dyn_boundary_fp32, dyn_ring_relax_fp32 as kernels
    if a.mode == 'cpuverify':
        def cpu_rn(op, x, y, *, interpret):  # test-only RN emulation, never GPU/S2 evidence
            x, y = x.astype(jnp.float64), y.astype(jnp.float64)
            exact = {'add': x+y, 'sub': x-y, 'mul': x*y, 'div': x/y}[op]
            return jax.lax.reduce_precision(exact, exponent_bits=8, mantissa_bits=23).astype(jnp.float32)
        for module in (kernels, dyn_boundary_fp32):
            module._INTERPRET = True
            module._rn = cpu_rn
    bd = probe.boundary
    state, m, cfg, meta = probe.load('d02', 'coupled')
    dt = meta['resolved_dt_s']
    path = build(a.out.parent/'pristine')
    f32 = np.float32
    mass = np.asarray(m.c1h, f32)[:, None, None]*np.asarray(state.mu_total, f32)[None]+np.asarray(m.c2h, f32)[:, None, None]
    coupled = np.asarray(state.qv, f32)*mass
    records = np.asarray(state.qv_bdy, f32)
    rate = (records[1]-records[0])/f32(cfg.update_cadence_s)
    nz = coupled.shape[0]
    rows = {}
    for lead in (0., dt, cfg.update_cadence_s/2):
        value = records[0]+f32(lead)*rate
        oracle = Oracle(path, state.theta.shape, nested=True, dt=dt, dtbc=lead)
        for vector in ('c1h', 'c2h', 'c1f', 'c2f'):
            oracle.set(vector, getattr(m, vector))
        oracle.set('scalar', state.qv)
        oracle.set('mu', state.mu_total)
        oracle.boundary('scalar', records[0], rate)
        oracle.run('relax_bdy_scalar')
        relax = oracle.field('scalar_tend', coupled.shape)
        oracle.run('spec_bdytend')
        reference = relax+oracle.field('field_tend', coupled.shape)
        args = (jnp.asarray(coupled), jnp.asarray(value[:, :, :nz]), jnp.asarray(rate[:, :, :nz]))
        got = jax.jit(lambda c, v, r: bd.native_ring_relax_tendency(c, v, r, dt, cfg))(*args)
        no_spec = jax.jit(lambda c, v: bd.native_ring_relax_tendency(c, v, None, dt, cfg))(*args[:2])
        weights = [bd._wrf_relax_weights(b, dt, cfg) for b in range(cfg.spec_bdy_width)]
        zero_g = jax.jit(lambda c, v, r: kernels.ring_relax_tendency(c, v, r, [f/dt for f, _ in weights],
                         [0.]*len(weights), spec=cfg.spec_zone, relax=cfg.relax_zone))(*args)
        fused = jax.jit(lambda q, v, r, mu, c1, c2: bd.native_ring_relax_tendency(q, v, r, dt, cfg, mass=(mu, c1, c2)))(
            state.qv, args[1], args[2], state.mu_total, m.c1h, m.c2h)
        rows[str(lead)] = {'gate': compare(reference, got), 'gate_fused_mass': compare(reference, fused),
                           'deletion_no_spec': compare(reference, no_spec),
                           'deletion_zero_gcx': compare(reference, zero_g)}
        print(lead, {k: (v['passed'], v['rms'], v['max_abs']) for k, v in rows[str(lead)].items()}, flush=True)
    # Retained (f64) non-bundle nested ph/w strip relax against pristine
    # relax_bdy_dry: a REAL version of this caller order (interpolate physical
    # strip, then couple) misses ph at the 1e9 coupled-value cancellation level,
    # so the native flag keeps it legacy; it is not on the PROD (bundle) path.
    pstate, pm, pcfg, _ = probe.load('d02', 'physical')
    strips = {}
    for lead in (0., dt, cfg.update_cadence_s/2):
        oracle = Oracle(path, pstate.theta.shape, nested=True, dt=dt, dtbc=lead)
        _, msf = probe.inputs_for_oracle(oracle, pstate, pm, pcfg, lead, False)
        oracle.run('relax_bdy_dry')
        row = {}
        try:
            bd.nested_ph_relax_tendency(pstate.ph_perturbation, pstate.ph_bdy[0], pstate.mu_total, pm.msfty,
                                        pm.c1f, pm.c2f, dt, pcfg)
            row['ph_fail_closed'] = {'passed': False}
        except NotImplementedError:
            row['ph_fail_closed'] = {'passed': True}
        for name, leaf, fn, ref_name in (('w', pstate.w_bdy, bd.nested_w_relax_tendency, 'rw_tendf'),):
            field = pstate.ph_perturbation if name == 'ph' else pstate.w
            got = jax.jit(lambda f, l, mu, s_, c1, c2: fn(f, bd.interpolate_boundary_leaf(l, lead, pcfg.update_cadence_s),
                          mu, s_, c1, c2, dt, pcfg))(field, leaf, pstate.mu_total, pm.msfty, pm.c1f, pm.c2f)
            ref = oracle.field(ref_name, got.shape)/msf[name][None]
            row[name] = compare(ref, got)
        strips[str(lead)] = row
        print('strip', lead, {k: (v['passed'], v.get('rms'), v.get('max_abs')) for k, v in row.items()}, flush=True)
    passed = all(r['gate']['passed'] and r['gate_fused_mass']['passed'] for r in rows.values()) and all(
        v['passed'] for row in strips.values() for v in row.values())
    # At lead 0 the fixture field equals its boundary record, so the relax
    # residual is ~0 and only the spec write is observable there.
    controls = (all(not r['deletion_no_spec']['passed'] for r in rows.values())
                and all(not r['deletion_zero_gcx']['passed'] for lead, r in rows.items() if float(lead) > 0))
    result = dict(platform=expected, mode=a.mode, rows=rows, strip_relax=strips, passed=passed and controls, gate_passed=passed,
                  deletion_controls_fail=controls, fixture_sha256=meta['fixture_sha256'], dt=dt,
                  cadence=cfg.update_cadence_s, scope='d02 real QV record; pristine relax_bdy_scalar+spec_bdytend; '
                  'literal REAL caller operands; kernel API for the nest scalar consumer (wired by host-sync)')
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
