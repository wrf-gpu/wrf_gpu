"""Real PROD two-domain one-root CPU forecast with the actual history writer and the V1 extract.

Not WRF fidelity or GPU admission. Full real physics/coupling/force-down. Frames:
lead 0 (both domains), d01 own step 1 (54 s), d02 own step 3 (54 s).

--mode on  : every frame is written by an OFF writer (history_off/) and by an
             extract-ON writer (history_on/) from the same carry; the ON writer's
             extract is finalized into extract/.
--mode off : OFF writer only (history_off/); used on a main snapshot for the
             main-vs-branch wrfout identity comparison.
--mode rawoff (M2): two extract writers from the same carry, both with the frozen ALISIOS point CSVs
             (--points): FULL (full history written to history_full/, extract_full/) and RAW-OFF
             (extract only via the reduced fetch set, nothing written; extract_rawoff/). Records per
             frame the device->host payload bytes, writer wall and RSS.
Receipts: receipt.json (+ ZNT carry snapshots for the ON extract parity).
"""
from pathlib import Path
import argparse, hashlib, json, os, resource, sys, time, traceback
from datetime import timedelta

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--case', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--mode', choices=('on', 'off', 'rawoff'), required=True)
parser.add_argument('--support', type=Path, default=None, help='dir with cpu_safe.py (default <source>/proofs/o1_nlbind)')
parser.add_argument('--points', type=Path, nargs='*', default=())
args = parser.parse_args()
tree = args.source; out = args.output; out.mkdir(exist_ok=False)
assert os.environ['JAX_PLATFORMS'] == 'cpu'
assert not Path('/tmp/wrf_gpu2_quiet').exists()
sys.path[:0] = [str(tree / 'src'), str(args.support or tree / 'proofs/o1_nlbind')]
started = time.monotonic()
rec = {'scope': __doc__, 'mode': args.mode, 'phase': 'imports', 'frames': [], 'source': str(tree), 'GPU': False}
rc = 1


def save():
    rec.update(wall_s=time.monotonic() - started, peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (out / 'receipt.json').write_text(json.dumps(rec, indent=2, default=str) + '\n')
    print(rec['phase'], round(rec['wall_s'], 1), flush=True)


try:
    import jax
    import numpy as np
    from jax.experimental import pallas as pl
    from jax._src.pallas import pallas_call as pc
    original = pc.pallas_call
    def interpreted(*a, **k): return original(*a, **{**k, 'interpret': True})
    pc.pallas_call = pl.pallas_call = interpreted
    import cpu_safe
    cpu_safe.install()
    from gpuwrf.contracts import state as sc
    sc._gpu_device = lambda: jax.devices('cpu')[0]
    from gpuwrf.backends import pallas as facade
    facade.cuda_native = lambda: False
    from gpuwrf.integration import nested_pipeline as p
    from gpuwrf.runtime import domain_tree as dt
    assert jax.devices()[0].platform == 'cpu'
    rec['gpuwrf_file'] = p.__file__
    case = args.case; names = ('d01', 'd02')
    cfg = p.NestedPipelineConfig(case, out / 'history_off', out / 'proof', hours=3, max_dom=2, aot_prefetch=False)
    rec['phase'] = 'load'; save()
    hierarchy, bundles, meta, start, dts, carries = p._load_domains(cfg, names)
    rec.update(dt_s=dts, run_start=start.isoformat(), grids={n: list(c.state.theta.shape) for n, c in carries.items()})
    cadence = {'d01': 1, 'd02': 3}
    alarms = {'d01': (1,), 'd02': (3,)}
    lands = {n: jax.device_get(c.noahmp_land) for n, c in carries.items()}

    def make_writer(directory):
        directory.mkdir()
        return p._PerDomainWrfoutWriter(output_dir=directory, input_dir=case, run_start=start, bundles=bundles,
                                         output_cadence_steps=cadence, dt_by_domain=dts, initial_lands=dict(lands))

    writers = {'off': make_writer(out / 'history_off')} if args.mode != 'rawoff' else {}
    if args.mode == 'on':
        from gpuwrf.extract.collector import ExtractCollector, plan_for, engine_identity
        from gpuwrf.extract.request import load_request
        config = out / 'extract_request.json'
        config.write_text(json.dumps({'schema': 'gpuwrf.extract.request.v1', 'run_id': 'prod20260725probe',
                                      'member': 'm000', 'arm': 'probe',
                                      'issue_utc': (start - timedelta(hours=6)).strftime('%Y-%m-%dT%H:%M:%SZ')}))
        request = load_request('v1', config, out / 'extract')
        on = make_writer(out / 'history_on')
        on._extract = ExtractCollector(request, run_start=start, input_dir=case, plans={
            n: plan_for(n, bundles[n].namelist, dts[n], cadence[n] * dts[n] / 60.0, alarms[n], True) for n in names})
        writers['on'] = on
    if args.mode == 'rawoff':
        from gpuwrf.extract.collector import ExtractCollector, plan_for, engine_identity, _rss_kib
        from gpuwrf.extract.request import load_request
        rec['d2h'] = []
        for label, raw in (('full', 'on'), ('rawoff', 'off')):
            config = out / f'extract_request_{label}.json'
            config.write_text(json.dumps({'schema': 'gpuwrf.extract.request.v1', 'run_id': 'prod20260725probe',
                                          'member': 'm000', 'arm': label, 'raw_history': raw,
                                          'points_csv': [str(p.resolve()) for p in args.points],
                                          'issue_utc': (start - timedelta(hours=6)).strftime('%Y-%m-%dT%H:%M:%SZ')}))
            request = load_request('v1', config, out / f'extract_{label}')
            w = make_writer(out / f'history_{label}')
            w._extract = ExtractCollector(request, run_start=start, input_dir=case, plans={
                n: plan_for(n, bundles[n].namelist, dts[n], cadence[n] * dts[n] / 60.0, alarms[n], True,
                            dx_m=bundles[n].grid.projection.dx_m) for n in names})
            # Fetched payload (= this writer's device->host batch) counted at submit, before the extract keeps
            # only its inputs; the full writer also checks grid column metrics == its full payload.
            from gpuwrf.extract.collector import _grid_column_metrics
            rec.setdefault('column_metrics_check', [])
            orig_submit = w._extract.submit  # (not `original`: the pallas interpret patch closes over it)
            def counted(name, own_step, prepared, state, _orig=orig_submit, _label=label, **kw):
                rec['d2h'].append({'writer': _label, 'domain': name, 'step': own_step,
                                   'n_fields': len(prepared.fields),
                                   'bytes': int(sum(np.asarray(v).nbytes for v in prepared.fields.values()))})
                if _label == 'full':
                    cols = _grid_column_metrics(prepared.grid, prepared.dimensions)
                    rec['column_metrics_check'].append({'domain': name, 'step': own_step, 'names': sorted(cols),
                        'payload_equal': all(np.asarray(prepared.fields[k]).astype(np.float32).tobytes() == v.tobytes()
                                             for k, v in cols.items())})
                return _orig(name, own_step, prepared, state, **kw)
            w._extract.submit = counted
            writers[label] = w
    znt = {}

    def emit(name, step, carry):
        for label, writer in writers.items():
            t0 = time.perf_counter()
            result = writer(name, step, carry)
            wall = time.perf_counter() - t0
            if label == 'rawoff':
                assert result['wrfout'] is None and not writer.written[name], 'RAW off wrote a frame'
                rec['frames'].append({'writer': label, 'domain': name, 'step': step, 'writer_wall_s': wall,
                                      'rss_kib': _rss_kib()})
                continue
            path = Path(writer.written[name][-1])
            rec['frames'].append({'writer': label, 'domain': name, 'step': step, 'path': str(path),
                                  'bytes': path.stat().st_size, 'writer_wall_s': wall,
                                  'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        znt[f'{name}_s{step}'] = np.array(carry.state.roughness_m, np.float32)
        save()
    emit.wants_carry = True
    for name in names:
        emit(name, 0, carries[name])
    nested = meta['nesting']
    domain = dt.DomainTree.from_domains(hierarchy, bundles, feedback_enabled=nested['feedback'],
                                        feedback_spec_zone=nested['spec_zone'],
                                        feedback_smooth_option=nested['smooth_option'])
    rec['phase'] = 'real_root'; save()
    result = dt.run_operational_domain_tree(domain, root_steps=1, carries=carries,
                                            initial_own_steps={'d01': 0, 'd02': 0}, output=emit,
                                            output_cadence_steps=cadence, output_alarm_steps=alarms,
                                            max_event_tail=0)
    jax.block_until_ready(result.carries)
    assert result.own_steps == {'d01': 1, 'd02': 3}
    np.savez(out / 'ZNT_carry.npz', **znt)
    if args.mode == 'on':
        rec['phase'] = 'finalize'; save()
        rec['extract'] = writers['on']._extract.finalize(engine=engine_identity())
    if args.mode == 'rawoff':
        rec['phase'] = 'finalize'; save()
        rec['extract'] = {label: w._extract.finalize(engine=engine_identity()) for label, w in writers.items()}
        rec['history_rawoff_entries'] = sorted(p.name for p in (out / 'history_rawoff').iterdir())
    rec['phase'] = 'PASS'; rc = 0
except BaseException as e:
    rec.update(phase='FAIL', exception=repr(e), traceback=traceback.format_exc())
finally:
    save(); sys.stdout.flush(); sys.stderr.flush(); os._exit(rc)
