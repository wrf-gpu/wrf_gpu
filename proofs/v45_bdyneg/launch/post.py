"""Read a verified v0.4.4 checkpoint; retain actual boundary inputs and compare pristine WRF.

No CUDA process exists in this phase. PD intermediate arrays are not invented.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from bdyneg_common import sha, write

SPECIES = ('qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'Ni', 'Nr')


def compare(saved, shape, *, cadence, lead, oracle, library):
    nz, ny, nx = shape
    y, x = np.ogrid[:ny, :nx]
    ring = np.minimum(np.minimum(y, ny - 1 - y), np.minimum(x, nx - 1 - x)) == 0
    results = []
    for name in SPECIES:
        field, records = saved[name], saved[name + '_bdy']
        assert records.shape[0] == 2 and np.isfinite(records).all()
        # Unchanged bdy_interp1 convention, not inferred from history output.
        value = records.astype(np.float32)
        rate = ((value[1] - value[0]).astype(np.float64) * (1.0 / float(cadence))).astype(np.float32)
        o = oracle(library, shape, nested=True, dt=cadence / 3, dtbc=lead, dts=cadence / 9)
        for key, v in (('c1h', saved['c1h']), ('c2h', saved['c2h']),
                       ('mu', saved['mu_total']), ('msf', np.ones((ny, nx), np.float32)),
                       ('field', field)):
            o.set(key, v)
        o.boundary('scalar', value[0], rate)
        o.run('spec_bdy_final', tag=4)
        want = o.field('field', shape)
        assert np.isfinite(want).all() and np.isfinite(field).all()
        got = np.asarray(field, np.float32)[:, ring]
        ref = want[:, ring]
        normal = (ref == 0) | (np.abs(ref) >= np.finfo(np.float32).tiny)
        mismatch = (got.view(np.uint32) != ref.view(np.uint32)) & normal
        negative = int(np.count_nonzero(ref < 0))
        # Semantic mutant: a blanket post-boundary clip disagrees whenever WRF is negative.
        clip_changes = int(np.count_nonzero((np.maximum(ref, 0).view(np.uint32)
                                           != ref.view(np.uint32)) & normal))
        results.append(dict(species=name, normal_mismatches=int(mismatch.sum()),
                            subnormal_reference=int((~normal).sum()),
                            pristine_negative=negative, port_negative=int((got < 0).sum()),
                            pristine_min=float(ref.min()), port_min=float(got.min()),
                            blanket_clip_mutant_disagreements=clip_changes))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seal', required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--mode', choices=('smoke', 'gpu'), required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.seal).read_text())
    # Every retained payload is authenticated before unpickling, using the original reader.
    from gpuwrf.runtime.restart_store import RestartStore
    from gpuwrf.assimilation.letkf.alisios_guard import guarded_session
    from gpuwrf.io.netcdf_lock import Dataset
    generations = sorted((a.run / 'checkpoints').glob('generation-*'))
    assert len(generations) == 1, 'one final checkpoint expected'
    store = RestartStore(a.run / 'checkpoints', 2 * 1024**3, 2, 0)
    snapshot, manifest = store.read(generations[0], device=False)
    expected = dict(zip(('d01', 'd02', 'd03'), (1, 3, 9) if a.mode == 'smoke' else (400, 1200, 3600)))
    assert snapshot['own_steps'] == manifest['own_steps'] == expected
    assert manifest['dt_s'] == {'d01': 27., 'd02': 9., 'd03': 3.}
    assert manifest['identity']['input_dir'] == cfg['input_dir']
    for name in ('namelist.input', 'wrfinput_d01', 'wrfinput_d02', 'wrfinput_d03', 'wrfbdy_d01'):
        assert manifest['identity']['inputs'][name] == cfg['input_sha256'][name]
    a.receipt.mkdir(parents=True, exist_ok=True)
    write(a.receipt / 'CHECKPOINT_MANIFEST.json', manifest)
    library = None
    if a.mode == 'gpu':
        from bdyneg_pristine import build
        library = build(a.run / 'oracle_build')
    paths = {Path(cfg['input_dir']) / ('wrfinput_' + d): ['20260111'] for d in expected}
    rows, artifacts = [], []
    with guarded_session(paths, ['20260110'], a.receipt / 'coefficient_read_guard',
                         purpose='Authorized unchanged C checkpoint coefficient read; no truth scoring'):
        for domain, payload in snapshot['carries'].items():
            state, scratch = payload['state_fields'], payload['scratch_fields']
            saved = {name: np.asarray(state[name]) for name in (*SPECIES, 'mu_total')}
            for name in SPECIES:
                if name + '_bdy' in state:
                    saved[name + '_bdy'] = np.asarray(state[name + '_bdy'])
                elif domain != 'd01':
                    raise ValueError('nested boundary leaf missing: ' + domain + '/' + name)
            saved['muts'] = np.asarray(scratch['muts'])
            with Dataset(Path(cfg['input_dir']) / ('wrfinput_' + domain)) as ds:
                saved['c1h'] = np.asarray(ds.variables['C1H'][0], np.float32)
                saved['c2h'] = np.asarray(ds.variables['C2H'][0], np.float32)
            assert all(np.isfinite(v).all() for v in saved.values())
            target = a.receipt / (domain + '_actual_boundary_operands.npz')
            np.savez_compressed(target, **saved)
            with np.load(target, allow_pickle=False) as z:
                assert set(z.files) == set(saved)
                assert all(z[k].dtype == v.dtype and z[k].shape == v.shape
                           and z[k].tobytes() == v.tobytes() for k, v in saved.items())
            artifacts.append(dict(path=str(target), sha256=sha(target), bytes=target.stat().st_size))
            info = dict(domain=domain, shape=list(saved['qc'].shape), own_step=expected[domain],
                        carried_mass_equals_muts=bool(np.array_equal(saved['mu_total'], saved['muts'])),
                        PD_pre_MP_operands='NOT_RETAINED_BY_NORMAL_CHECKPOINT')
            if domain != 'd01' and library is not None:
                from bdyneg_pristine import Oracle
                cadence = 27. if domain == 'd02' else 9.
                lead = ((expected[domain] - 1) % 3 + 1) * manifest['dt_s'][domain]
                info['pristine_spec_final'] = compare(saved, saved['qc'].shape,
                    cadence=cadence, lead=lead, oracle=Oracle, library=library)
            rows.append(info)
    # GPU-vs-GPU saved-history match is determinism evidence, not the WRF oracle.
    matches = None
    if a.mode == 'gpu':
        fixture = Path(cfg['C_d02_h3_fixture'])
        assert sha(fixture) == cfg['C_d02_h3_sha256']
        state = snapshot['carries']['d02']['state_fields']
        with np.load(fixture, allow_pickle=False) as old:
            matches = {}
            for model, history in (('qc', 'QCLOUD'), ('qr', 'QRAIN'), ('qv', 'QVAPOR')):
                field = state[model]
                strips = (field[..., :8], field[..., -8:], field[..., :8, :], field[..., -8:, :])
                matches[history] = all(np.asarray(v).dtype == old[history + '__' + side].dtype
                    and np.asarray(v).tobytes() == old[history + '__' + side].tobytes()
                    for side, v in zip(('W', 'E', 'S', 'N'), strips))
    report = dict(mode=a.mode, checkpoint_verified=True, own_steps=expected, rows=rows,
                  exact_operand_artifacts=artifacts, all_array_readback='BYTE_EXACT',
                  saved_C_boundary_identity=matches, identity_scope='GPU regression/determinism only',
                  GPU_native_gates='NOT-APPLICABLE' if a.mode == 'smoke' else 'see actual BACKEND + oracle rows',
                  scientific_gate='OPEN: root PD amplitude and unobserved events remain unclassified')
    failures = [r for domain in rows for r in domain.get('pristine_spec_final', [])
                if r['normal_mismatches']]
    report['nested_spec_final_debug_gate'] = ('NOT-APPLICABLE' if a.mode == 'smoke'
                                             else 'FAIL' if failures else 'PASS')
    if a.mode == 'gpu':
        cache = a.run / 'layer/upper/jax'
        if cache.exists():
            e41_env = dict(os.environ, JAX_PLATFORMS='cpu', GPUWRF_JAX_CACHE='0',
                          CUOBJDUMP='/usr/local/cuda/bin/cuobjdump')
            with (a.receipt / 'E41.log').open('w') as stream:
                result = subprocess.run([sys.executable, str(Path(__file__).parent / 'wave_cubin_gate.py'),
                    str(cache), str(a.run / 'e41'), '--run-proof',
                    str(a.run / 'output/proofs/nested_pipeline_run.json')],
                    env=e41_env, stdout=stream, stderr=subprocess.STDOUT, timeout=210)
            verdict = a.run / 'e41/wave_gate.json'
            if not verdict.is_file():
                raise ValueError('E41 scan did not produce its required verdict JSON')
            gate = json.loads(verdict.read_text())
            write(a.receipt / 'E41.json', gate)
            report['E41'] = dict(verdict=gate['e41_gate'], rc=result.returncode,
                                 hard_stack_gate=gate['hard_stack_gate'])
        else:
            report['E41'] = dict(verdict='REUSED_UNCHANGED_BUNDLE_NO_NEW_CACHE',
                                evidence='sealed bundle consumer checker and source hashes')
    write(a.receipt / 'POST.json', report)
    if library is not None:
        write(a.receipt / 'PRISTINE_SOURCE_MANIFEST.json', json.loads((a.run / 'oracle_build/manifest.json').read_text()))
    return 3 if failures or report.get('E41', {}).get('verdict') == 'FAIL' else 0


if __name__ == '__main__':
    raise SystemExit(main())
