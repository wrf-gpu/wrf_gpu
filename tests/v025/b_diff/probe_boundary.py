"""Real PROD boundary operator gate against unchanged pristine WRF REAL4."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests/v025/b_core')]
arm = sys.argv[sys.argv.index('--arm')+1] if '--arm' in sys.argv else 'native'
os.environ['GPUWRF_BOUNDARY_FP32'] = '1' if arm == 'native' else '0'
os.environ.setdefault('JAX_ENABLE_X64', 'true')
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.contracts.state import State
from gpuwrf.contracts.grid import DycoreMetrics
from gpuwrf.coupling import boundary_apply as boundary
from pristine_boundary import Oracle, build
from bench_acoustic import compare

DATA = Path('<USER_HOME>/wrf_gpu2_lanes/b-diff/boundary_fixtures')


def load(domain, variant):
    with np.load(DATA/(domain+'.npz')) as z:
        arrays = {k: jnp.asarray(z[k]) for k in z.files
                  if k.startswith(('state_', 'metric_', variant+'_'))}
    fields = {n: arrays.get('state_'+n, arrays.get(variant+'_'+n)) for n in State.__slots__}
    state = State.tree_unflatten(None, [fields[n] for n in State.__slots__])
    metrics = DycoreMetrics(**{n: arrays['metric_'+n] for n in DycoreMetrics._array_names()})
    meta = json.loads((DATA/(domain+'.json')).read_text())
    assert hashlib.sha256((DATA/(domain+'.npz')).read_bytes()).hexdigest() == meta['fixture_sha256']
    cfg = boundary.BoundaryConfig(update_cadence_s=meta['cadence_s'], force_geopotential=domain=='d01',
        nested_frozen_wrf_boundary_bundle=variant=='coupled', normal_bdy_relax_strength=1.)
    return state, metrics, cfg, meta


def inputs_for_oracle(oracle, state, metrics, cfg, lead, coupled):
    """Independent NumPy REAL caller operands; WRF owns the relaxation itself."""
    a = lambda value: np.asarray(value, np.float32)
    for name in ('c1h', 'c2h', 'c1f', 'c2f'):
        oracle.set(name, getattr(metrics, name))
    mu = a(state.mu_total)
    muu = np.float32(.5)*(np.concatenate((mu[:, :1], mu), axis=1)+np.concatenate((mu, mu[:, -1:]), axis=1))
    muv = np.float32(.5)*(np.concatenate((mu[:1], mu), axis=0)+np.concatenate((mu, mu[-1:]), axis=0))
    layer = lambda c1, c2, col: a(c1)[:, None, None]*col[None]+a(c2)[:, None, None]
    mass = {'u': layer(metrics.c1h, metrics.c2h, muu), 'v': layer(metrics.c1h, metrics.c2h, muv),
            't': layer(metrics.c1h, metrics.c2h, mu), 'ph': layer(metrics.c1f, metrics.c2f, mu),
            'w': layer(metrics.c1f, metrics.c2f, mu)}
    msf = {'u': a(metrics.msfuy), 'v': a(metrics.msfvx), 't': a(metrics.msfty),
           'ph': a(metrics.msfty), 'w': a(metrics.msfty)}
    physical = {'u': a(state.u), 'v': a(state.v), 'w': a(state.w),
                # WRF relax_bdy_dry operand is the perturbation t_2 = theta - T0.
                't': a(state.theta)-np.float32(300),
                'ph': a(state.ph_perturbation), 'mu': a(state.mu_perturbation)[None]}
    oracle.set('ru', mass['u']*physical['u']/msf['u'][None])
    oracle.set('rv', mass['v']*physical['v']/msf['v'][None])
    for name in ('t', 'ph', 'w'):
        oracle.set(name, physical[name])
    oracle.set('mu', state.mu_perturbation)
    oracle.set('mut', state.mu_total)
    for name, field in [('u', 'u'), ('v', 'v'), ('w', 'w'), ('t', 'theta'), ('ph', 'ph'), ('mu', 'mu')]:
        records = a(getattr(state, field+'_bdy'))
        if coupled or name == 'mu':
            values = records[0]
            tend = (records[1]-records[0])/np.float32(cfg.update_cadence_s)
        else:
            # Preserve the released root/physical adapter's frozen reference
            # mass convention. Exact coupled SINT records have a separate gate.
            transformed = []
            for record in records:
                leaf = record.copy()
                nz, ny, nx = physical[name].shape
                for side in range(4):
                    for width in range(5):
                        if side < 2:
                            col = width if side == 0 else nx-1-width
                            factor = mass[name][:, :, col]
                            scale = msf[name][:, col][None]
                            count = ny
                        else:
                            row = width if side == 2 else ny-1-width
                            factor = mass[name][:, row, :]
                            scale = msf[name][row, :][None]
                            count = nx
                        strip = record[side, width, :nz, :count]
                        if name == 't':
                            strip = strip-np.float32(300)  # physical theta record -> t'
                        value = strip*factor
                        if name in ('u', 'v'):
                            value = value/scale
                        leaf[side, width, :nz, :count] = value
                transformed.append(leaf)
            values = transformed[0]
            tend = (transformed[1]-transformed[0])/np.float32(cfg.update_cadence_s)
        oracle.boundary(name, values, tend)
    return mass, msf


def dry(state, metrics, cfg, lead, dt, coupled, nested):
    out = boundary.specified_relax_dry_tendencies(state, lead, metrics, dt, cfg,
        include_nested_w=nested, coupled_boundary_leaves=coupled)
    names = ('ru', 'rv', 't', 'ph', 'mu') + (('w',) if nested else ())
    return tuple(getattr(out, name) for name in names)


# Storage-dtype operands may enter a kernel as f64 and are rounded to REAL at
# load (results may be widened back exactly on store): f64 may appear only in
# pointers, loads/stores, the f64->f32 RN convert and the exact f32->f64 widen.
_TTIR_F64_OPS = ('tt.load', 'tt.store', 'arith.truncf', 'arith.extf', 'tt.splat', 'tt.addptr', 'tt.func',
                 'tt.broadcast', 'tt.expand_dims', 'tt.return')
_PTX_F64_OPS = ('ld.', 'st.', 'cvt.rn.f32.f64', 'cvt.f64.f32', 'mov.')


def ttir_f64_ok(line):
    if 'f64' not in line:
        return True
    tokens = line.split()
    op = tokens[2] if len(tokens) > 2 and tokens[1] == '=' else tokens[0]
    return op.strip('"').startswith(_TTIR_F64_OPS)


def ptx_f64_ok(line):
    code = line.split('//')[0].strip()
    if '.f64' not in code:
        return True
    opcode = code.split()[0] if not code.startswith('@') else code.split()[1]
    return opcode.startswith(_PTX_F64_OPS)


def pallas_eqns(jaxpr):
    """All pallas_call equations, recursing through pjit/closed sub-jaxprs."""
    found = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == 'pallas_call':
            found.append(eqn)
            continue
        for value in eqn.params.values():
            for sub in (value if isinstance(value, (tuple, list)) else (value,)):
                inner = getattr(sub, 'jaxpr', sub)
                if hasattr(inner, 'eqns'):
                    found += pallas_eqns(inner)
    return found


def family_pallas_eqns(domain):
    """Kernels of all 14 family entrypoints (bench cases) plus the ring-relax API."""
    sys.argv += ['--source', str(ROOT)] if '--source' not in sys.argv else []
    import bench_boundary_family as family
    table, _ = family.cases(domain)
    eqns = []
    for fn, call_args in table.values():
        eqns += pallas_eqns(jax.make_jaxpr(fn)(*call_args).jaxpr)
    state, m, cfg, meta = load(domain, 'physical' if domain == 'd01' else 'coupled')
    nz = state.qv.shape[0]
    leaf = state.qv_bdy[0, :, :, :nz]
    ring = lambda c, v, r: boundary.native_ring_relax_tendency(c, v, r, meta['resolved_dt_s'], cfg)
    eqns += pallas_eqns(jax.make_jaxpr(ring)(state.qv, leaf, leaf).jaxpr)
    return eqns


def work_target_jaxpr(state, metrics, cfg, coupled):
    """The four caller-work targets on the fixture's initial leaf/field/mass."""
    def targets(s, m):
        mu = s.mu_total
        muu = .5*(jnp.concatenate((mu[:, :1], mu), 1)+jnp.concatenate((mu, mu[:, -1:]), 1))
        muv = .5*(jnp.concatenate((mu[:1], mu), 0)+jnp.concatenate((mu, mu[-1:]), 0))
        out = []
        for wind, col, scale in (('u', muu, m.msfuy), ('v', muv, m.msfvx)):
            mass = m.c1h[:, None, None]*col[None]+m.c2h[:, None, None]
            field, leaf = getattr(s, wind), getattr(s, wind+'_bdy')[0]
            for kind in ('normal', 'tangential'):
                fn = getattr(boundary, kind+'_bdy_work_target_'+wind)
                out.append(fn(leaf, field, mass, mass, scale, config=cfg, coupled_boundary_leaves=coupled))
        return out
    return jax.make_jaxpr(targets)(state, metrics).jaxpr


def verify_dry(state, metrics, cfg, path, *, lead, dt, coupled, nested):
    oracle = Oracle(path, state.theta.shape, nested=nested, dt=dt, dtbc=lead)
    _, msf = inputs_for_oracle(oracle, state, metrics, cfg, lead, coupled)
    oracle.run('relax_bdy_dry')
    fn = jax.jit(lambda s, m: dry(s, m, cfg, lead, dt, coupled, nested))
    got = fn(state, metrics)
    jax.block_until_ready(got)
    names = ('ru_tendf', 'rv_tendf', 't_tendf', 'ph_tendf', 'mu_tend') + (('rw_tendf',) if nested else ())
    refs = [oracle.field(n, x.shape) for n, x in zip(names, got)]
    for index, field in ((2, 't'), (3, 'ph'), *(([(5, 'w')] if nested else []))):
        refs[index] /= msf[field][None]
    return {n: compare(r, x) for n, r, x in zip(names, refs, got)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=('cpuverify', 'verify', 'hlo', 'lower', 'backend'), required=True)
    p.add_argument('--arm', choices=('native', 'legacy'), default='native')
    p.add_argument('--domain', choices=('d01', 'd02'), default='d01')
    p.add_argument('--variant', choices=('physical', 'coupled'), default='physical')
    p.add_argument('--suite', choices=('dry', 'all'), default='dry')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    expected = 'cpu' if args.mode in ('cpuverify', 'hlo', 'lower', 'backend') else 'gpu'
    assert jax.devices()[0].platform == expected, jax.devices()
    if args.arm == 'native' and args.mode in ('cpuverify', 'hlo'):
        from gpuwrf.kernels import dyn_boundary_fp32 as kernels
        kernels._INTERPRET = True
        def cpu_rn(op, x, y, *, interpret):
            # Test-only emulation of explicit PTX RN boundaries, never S2/GPU evidence.
            x, y = x.astype(jnp.float64), y.astype(jnp.float64)
            value = {'add': lambda: x+y, 'sub': lambda: x-y,
                     'mul': lambda: x*y, 'div': lambda: x/y}[op]()
            # Explicit RN-even: XLA's excess-precision rewrite would otherwise
            # drop the f32 round trip between chained emulated operations.
            return jax.lax.reduce_precision(value, exponent_bits=8, mantissa_bits=23).astype(jnp.float32)
        kernels._rn = cpu_rn
    state, metrics, cfg, meta = load(args.domain, args.variant)
    dt = meta['resolved_dt_s']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = dict(platform=expected, domain=args.domain, variant=args.variant, arm=args.arm,
                  fixture_sha256=meta['fixture_sha256'], shape=list(state.theta.shape),
                  resolved_dt_s=dt, cadence_s=cfg.update_cadence_s,
                  cache={k: os.environ.get(k) for k in ('JAX_COMPILATION_CACHE_DIR', 'GPUWRF_XLA_AUTOTUNE_CACHE_DIR')})
    if args.mode in ('lower', 'backend'):
        from jax._src.pallas.triton import lowering
        jp = jax.make_jaxpr(lambda s, m: dry(s, m, cfg, dt, dt, args.variant=='coupled', args.domain=='d02'))(state, metrics).jaxpr
        eqns = pallas_eqns(jp)
        if args.suite == 'all':
            eqns += pallas_eqns(work_target_jaxpr(state, metrics, cfg, args.variant == 'coupled'))
            eqns += family_pallas_eqns(args.domain)
        ir = [str(lowering.lower_jaxpr_to_triton_module(e.params['jaxpr'], e.params['grid_mapping'], 'cuda', 120).module)
              for e in eqns]
        # Each kernel compiles with its own pallas_call num_warps (Pallas default 4).
        warps = [getattr(e.params.get('compiler_params'), 'num_warps', None) or 4 for e in eqns]
        args.out.with_suffix('.ttir').write_text('\n'.join(ir))
        bad_ir = [line.strip() for text in ir for line in text.splitlines() if not ttir_f64_ok(line)]
        result.update(lowered=len(ir), f64_in_ir=any('f64' in text for text in ir),
                      f64_arith_ir=len(bad_ir), f64_arith_ir_examples=bad_ir[:5])
        assert ir and not bad_ir, bad_ir[:5]
        if args.mode == 'backend':
            import triton
            from triton.backends.compiler import GPUTarget
            kernels = {}
            for text, num_warps in zip(ir, warps):
                key = hashlib.sha256(text.encode()).hexdigest()
                if key in kernels:
                    continue
                ttir = args.out.parent/(key+'.ttir')
                ttir.write_text(text)
                compiled = triton.compile(str(ttir), target=GPUTarget('cuda', 120, 32),
                                          options={'num_warps': num_warps, 'num_stages': 3})
                assert compiled.module is None and compiled.function is None and compiled._run is None
                ptx = compiled.asm['ptx']
                bad_ptx = [line.strip() for line in ptx.splitlines() if not ptx_f64_ok(line)]
                assert not bad_ptx, bad_ptx[:5]
                args.out.parent.joinpath(key+'.ptx').write_text(ptx)
                args.out.parent.joinpath(key+'.cubin').write_bytes(compiled.asm['cubin'])
                kernels[key] = {'shared': compiled.metadata.shared, 'num_warps': num_warps,
                                'ptx_f64_interface_lines': sum('f64' in line for line in ptx.splitlines()),
                                'cubin_sha256': hashlib.sha256(compiled.asm['cubin']).hexdigest()}
            result.update(compiled_unique=len(kernels), kernels=kernels,
                          target='cuda/sm120/32, explicit; no GPU module loaded',
                          ptx_f64_arithmetic=0, GPU_execution=False)
    elif args.mode == 'hlo':
        fn = jax.jit(lambda s, m: dry(s, m, cfg, dt, dt, args.variant=='coupled', args.domain=='d02'))
        text = fn.lower(state, metrics).compiler_ir(dialect='hlo').as_hlo_text()
        args.out.with_suffix('.hlo').write_text(text)
        result.update(f64_hlo_lines=sum('f64[' in line for line in text.splitlines()))
    else:
        path = build(args.out.parent/'pristine')
        cases = {str(lead): verify_dry(state, metrics, cfg, path, lead=lead, dt=dt,
                 coupled=args.variant=='coupled', nested=args.domain=='d02')
                 for lead in (0., dt, cfg.update_cadence_s/2)}
        result.update(cases=cases, passed=all(v['passed'] for row in cases.values() for v in row.values()))
        if args.suite == 'all':
            from boundary_extra_gates import verify_spec, verify_scalars, verify_ph_history
            result['spec'] = {str(lead): verify_spec(state, cfg, path, lead=lead,
                dt=dt, nested=args.domain=='d02') for lead in (0., dt, cfg.update_cadence_s/2)}
            if args.variant == 'coupled':
                result['scalars'] = {str(lead): verify_scalars(state, metrics, cfg, path,
                    lead=lead, dt=dt) for lead in (0., dt, cfg.update_cadence_s/2)}
            result['ph_history'] = verify_ph_history(args.domain, cfg, path)
            extra = [v['passed'] for group in ('spec', 'scalars')
                     for row in result.get(group, {}).values() for v in row.values()]
            extra += [v['passed'] for v in result['ph_history']['cases'].values()]
            result['passed'] = result['passed'] and all(extra)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    if result.get('passed') is False:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
