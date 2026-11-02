"""Classify the preregistered event using authenticated, full-domain operands.

CPU only. One targeted replay of pristine spec_bdy_final; the other fifteen
completed post comparisons are carried with their source/operand provenance.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def real(value):
    x = np.float32(value)
    return dict(value=float(x), float32_bits=f'{struct.unpack("<I", x.tobytes())[0]:08x}')


def assert_normal_equal(actual, reference):
    normal = (reference == 0) | (np.abs(reference) >= np.finfo(np.float32).tiny)
    mismatch = (actual.view(np.uint32) != reference.view(np.uint32)) & normal
    assert not mismatch.any(), f'pristine normal-byte mismatch count={int(mismatch.sum())}'


def inspect(receipt, deploy, work):
    post = json.loads((receipt / 'POST.json').read_text())
    result = json.loads((receipt / 'RESULT.json').read_text())
    cfg = json.loads((deploy / 'seal.json').read_text())
    assert result['mode'] == 'gpu' and result['model_rc'] == result['post_rc'] == 0
    assert post['checkpoint_verified'] and post['all_array_readback'] == 'BYTE_EXACT'
    assert post['own_steps'] == {'d01': 400, 'd02': 1200, 'd03': 3600}
    for artifact in post['exact_operand_artifacts']:
        assert sha(artifact['path']) == artifact['sha256']
    for name in ('bdyneg_pristine.py', 'post.py', 'model.py'):
        assert sha(deploy / 'tools' / name) == cfg['tools'][name]
    helper = deploy / 'tools/bdyneg_pristine.py'
    spec = importlib.util.spec_from_file_location('captured_h1_pristine', helper)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    library = oracle.build(work / 'oracle')
    manifest = json.loads((work / 'oracle/manifest.json').read_text())
    previous = json.loads((receipt / 'PRISTINE_SOURCE_MANIFEST.json').read_text())
    assert manifest['source_sha256'] == previous['source_sha256']
    assert manifest['routine_sha256'] == previous['routine_sha256']
    with np.load(receipt / 'd02_actual_boundary_operands.npz', allow_pickle=False) as z:
        saved = {k: z[k] for k in ('qc', 'qc_bdy', 'mu_total', 'muts', 'c1h', 'c2h')}
    assert all(np.isfinite(v).all() for v in saved.values())
    shape = saved['qc'].shape
    assert shape == (44, 117, 267)
    y, x = np.ogrid[:shape[1], :shape[2]]
    ring0 = np.minimum(np.minimum(y, shape[1]-1-y), np.minimum(x, shape[2]-1-x)) == 0
    # spec_bdy_final does not read the prior field on the overwritten spec zone.
    # Start that OUTPUT region at zero so an accidental no-op cannot self-pass.
    initial_reference = saved['qc'].copy()
    initial_reference[:, ring0] = np.float32(0)
    records = saved['qc_bdy'].astype(np.float32)
    rate = ((records[1] - records[0]).astype(np.float64) * (1. / 27.)).astype(np.float32)
    o = oracle.Oracle(library, shape, nested=True, dt=9., dtbc=27., dts=3.)
    for key, value in (('field', initial_reference), ('mu', saved['mu_total']),
                       ('c1h', saved['c1h']), ('c2h', saved['c2h']),
                       ('msf', np.ones(shape[1:], np.float32))):
        o.set(key, value)
    o.boundary('scalar', records[0], rate)
    o.run('spec_bdy_final', tag=4)
    reference = o.field('field', shape)
    assert np.isfinite(reference).all()
    got, want = saved['qc'][:, ring0], reference[:, ring0]
    assert_normal_equal(got, want)
    noop_mismatches = int(np.count_nonzero(got.view(np.uint32) != initial_reference[:, ring0].view(np.uint32)))
    assert noop_mismatches > 0
    try:
        assert_normal_equal(got, initial_reference[:, ring0])
    except AssertionError as exc:
        assert str(exc) == f'pristine normal-byte mismatch count={noop_mismatches}'
    else:
        raise AssertionError('missing pristine dispatch was not detected')
    # The real captured reference first passes. A clip mutant must then fail
    # this same numerical comparison with AssertionError, not a setup error.
    try:
        assert_normal_equal(np.maximum(got, np.float32(0)), want)
    except AssertionError as exc:
        assert str(exc) == 'pristine normal-byte mismatch count=1'
        clip_mutant = dict(verdict='REJECTED_BY_ASSERTION', mismatches=1)
    else:
        raise AssertionError('captured negative failed to detect a blanket clip')
    k, j, i = 0, 62, 266
    # Packed side1 is E; distance0; tangent j62. No cropped-domain oracle.
    a, b = records[0, 1, 0, k, j], records[1, 1, 0, k, j]
    diff = np.float32(b-a)
    tendency = np.float32(np.float64(diff) * (1./27.))
    increment = np.float32(np.float32(27.) * tendency)
    numerator = np.float32(a + increment)
    denominator = np.float32(np.float32(saved['c1h'][k] * saved['mu_total'][j, i]) + saved['c2h'][k])
    evaluated = np.float32(numerator / denominator)
    expected = np.float32(-2.4461745563170156e-12)
    assert real(evaluated) == real(reference[k, j, i]) == real(saved['qc'][k, j, i]) == real(expected)
    original = Path(cfg['C_d02_h3_fixture'])
    assert sha(original) == cfg['C_d02_h3_sha256']
    with np.load(original, allow_pickle=False) as z:
        assert real(z['QCLOUD__E'][k, j, -1]) == real(expected)
    assert a == 0 and b < 0 and numerator == b and denominator > 0
    assert real(np.float32(b / denominator)) == real(expected)
    root = {}
    with np.load(receipt / 'd01_actual_boundary_operands.npz', allow_pickle=False) as z:
        for name in ('qv', 'qc', 'qr', 'qi', 'qs', 'qg', 'Ni', 'Nr'):
            field = z[name]
            assert np.isfinite(field).all()
            root[name] = dict(negative_count=int(np.count_nonzero(field < 0)),
                              minimum=float(field.min()))
        root_leaves = sorted(z.files)
    carried = [row for domain in post['rows'] for row in domain.get('pristine_spec_final', [])]
    assert len(carried) == 16 and not any(r['normal_mismatches'] for r in carried)
    return dict(schema='v45-bdyneg.actual-H1.v1', verdict='INHERITED_RECORD',
        declared_H1='REFUTED_AS_CREATION_AT_FINAL_PIN',
        event=dict(domain='d02', frame='2026-01-11_03:00:00', field='QCLOUD',
                   index_kji=[k,j,i], packed_record_index=['record',1,0,k,j],
                   complete_domain_shape=list(shape), original_target=real(expected),
                   actual=real(saved['qc'][k,j,i]), pristine=real(reference[k,j,i])),
        actual_operands=dict(lower=real(a), upper=real(b), difference=real(diff),
            retained_rate=real(tendency), dtbc=27., parent_cadence_s=27.,
            increment=real(increment), numerator=real(numerator), denominator=real(denominator),
            mu_total=real(saved['mu_total'][j,i]), scratch_muts=real(saved['muts'][j,i]),
            c1h=real(saved['c1h'][k]), c2h=real(saved['c2h'][k])),
        causal_scope='Negative already in upper coupled record. Zero lower record and endpoint '
                    'increment exactly equal upper record; final temporal interpolation/division '
                    'propagate the sign. Upstream record construction is not classified.',
        targeted_pristine_replay=dict(species='qc', normal_mismatches=0,
            pristine_negative=int(np.count_nonzero(want<0)),
            subnormal_reference=int(np.count_nonzero((want!=0)&(np.abs(want)<np.finfo(np.float32).tiny))),
            initial_reference_spec_zone='ZERO; actual boundary records/mass/coefficients unchanged',
            no_op_oracle_mutant=dict(verdict='REJECTED_BY_ASSERTION', mismatches=noop_mismatches),
            blanket_clip_mutant=clip_mutant, exact_endpoint_replacement='same negative result'),
        completed_post_reused=dict(species_domain_comparisons=16, normal_mismatches=0,
            source_sha256=manifest['source_sha256'], POST_sha256=sha(receipt/'POST.json'),
            identity_scope='Pristine same-operand operator fidelity; saved-C equality is regression only'),
        root_PD=dict(disposition='OPEN_NO_SAME_STEP_PD_OPERANDS',
            reported_event='d01 QCLOUD 2026-01-11_02:00:09, root step267',
            retained_checkpoint='2026-01-11_03:00:00, root step400',
            retained_leaves=root_leaves, root_h3_final_stats=root,
            missing=['step267 pre/post-PD scalar and tendencies with complete stencil',
                     'step267 acoustic transport ru_m/rv_m and flow-copy source',
                     'step267 pre-microphysics/after-final-boundary water'],
            conclusion='Final history and h3 checkpoint do not reconstruct the reported root step. '
                       'No root-PD attribution or blanket nonnegativity waiver.'),
        original_C_regression=dict(exact_target=True, saved_strip_identity=post['saved_C_boundary_identity']),
        E41=post['E41'], native_qualification='OPEN: E41 INCOMPLETE remains INCOMPLETE',
        production_changes=0, extra_GPU_runs=0,
        source_manifest=manifest, operands=post['exact_operand_artifacts'],
        scientific_gate='Final-pin creation H1 refuted; upstream coupled-record origin and root PD remain OPEN')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--deploy', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    report = inspect(args.receipt, args.deploy, args.work)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(verdict=report['verdict'], H1=report['declared_H1'],
                         target=report['event'], root_PD=report['root_PD']['disposition'])))
