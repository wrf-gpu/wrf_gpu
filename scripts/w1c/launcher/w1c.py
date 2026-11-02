"""W1c-1L: four cold members C vs C+SIC (sharpened-inversion IC variant) x 0219/0530; no retry, no resume.

Reuses the SCREEN4 member path unchanged (screen4.run_member / full_history; run_four semantics in run_rows, screen4_member.py
zero-compile refusal, w1a_* cache/consumer/prime gates, sealed copies). New here: the W1c manager admission
(rows from the ALISIOS ID confirmation, C entries = base cases of the pinned SCREEN6 admission v2, SIC entries =
icvariant staged facades bound to their variant admission + R10 key receipt), and the W1C registry tags.
Registry issue stays the BASE model cycle; the variant id/identity ride in tags/binding.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import icvariant as iv
import w1c_rows
import w1c_cache
import screen4 as s
c = s.c
q = c.q
from w1a_cache import seed, unchanged

SCHEMA = 'wrf_gpu2.w1c.admission.v1'
SCOPE = 'W1C_1L_COLD_NO_RESUME'
CASES = ('20260219_18z', '20260530_18z')
ARMS = ('C', 'SIC_1L')
BASE_ADMISSION_SHA = '37e2de3be3a68e1cb7264c776aa476961a886e306507f20704f1fea26cb5fd8c'   # SCREEN6 admission v2
PRIME_SHA = '1ad806178ef63b5fb867d4049c787198703ed3eaaf1d56f6836c083eec302b17'
# Manager decision (A) 2026-10-31T05:32:36Z: the paid layer keeps 10 entries written by ALP1 C3 (03:54-04:00Z) after the
# prime; W1c freezes on prime (1863, byte-identical) + exactly those 10, pinned here (sealed file, SHA in the source).
CACHE_INVENTORY = Path(__file__).resolve().parent / 'W1C_CACHE_INVENTORY.json'
CACHE_INVENTORY_SHA = '45e11e3c2091c65769f254a127e918e5a3be6f3c7a03e60831d092ccbf734d7d'
CACHE_ADDED_AFTER_PRIME = 10
C_PARAMS_SHA = '5ab3f61665204e7350d291848e9abc9a8335544f4d0717e0b36b0884997664d6'
LANE = Path('<USER_HOME>/wrf_gpu2_lanes/v46-icvariant')
DATA = Path('<DATA_ROOT>/wrf_gpu2/scratch/v46-icvariant')
OUTPUT_ROOT = DATA / 'W1C_1L_20261031'      # pinned layout (LAYOUT.json); raw = <OUTPUT_ROOT>/<our_id>/wrfout
PHASES = ['preflight', 'member1', 'member2', 'member3', 'member4', 'halt']
TAG = 'W1C_1L_V1'
SCREEN4_C_RECEIPTS = {'20260219_18z': '<DATA_ROOT>/wrf_gpu2/scratch/v45-waveops/W1A_SCREEN4_20261030/MEMBER_1_DONE.json',
                      '20260530_18z': '<DATA_ROOT>/wrf_gpu2/scratch/v45-waveops/W1A_SCREEN4_20261030/MEMBER_2_DONE.json'}
from w1c_disk import GPU_SECONDS_TOTAL_CAP, SIC_PAIR_CAP, MEMBER_CAP
from w1c_disk import (MEMBER_RAW_GIB, MARGIN_GIB, MEMBER_FLOOR_GIB, PRIVATE_PEAK_GIB, MEMBER_SETUP_S, WAIT_POLL_S,
                      RAW_JOURNAL, DiskFloorTimeout, wait_for_floor, run_rows)


def _pinned(ref, what):
    q.require(isinstance(ref, dict) and ref.get('path') and q.sha(ref['path']) == ref.get('sha256'), f'{what} changed/missing')
    return q.load(ref['path'])


ORDER = [('20260530_18z', 'C'), ('20260530_18z', 'SIC_1L'), ('20260219_18z', 'C'), ('20260219_18z', 'SIC_1L')]


def identities(ref):
    """Rows from the signed GO (identity fields signed; full rows from its SHA-pinned go_draft) or an ID confirmation."""
    doc = _pinned(ref, 'W1c identity confirmation / signed GO')
    rows = w1c_rows.signed_rows(doc, lambda r: _pinned(r, 'GO draft pinned by the signed GO'))
    q.require(len(rows) == 4 and [r['ordinal'] for r in rows] == [1, 2, 3, 4], 'W1c rows/ordinals changed')
    q.require([(r['base_case_id'], r['arm_id']) for r in rows] == ORDER, 'W1c run order/case/arm set changed')
    for r in rows:
        q.require(re.fullmatch('[0-9a-f]{32}', r['our_id']) and r['registry_run_id'] == 'optimize:' + r['our_id'] and
                  r['row_id'] == f"{TAG}.{r['base_case_id']}.{r['arm_id']}.det0" and r['launcher_policy'] == 'NO_RESUME' and
                  r['model_hours'] == 18, 'W1c identity/policy mismatch')
        q.require(r['parameters'] == {'nt_c': 100e6, 'entrainment': .33}, 'W1c scalar envelope must be C for both arms')
    q.require(len({r['our_id'] for r in rows}) == 4, 'duplicate W1c identities')
    return rows


def variant_case(case_id, ref, base, params):
    """SIC entry: staged facade + its variant admission + R10 key receipt, all re-verified."""
    entry = _pinned(ref['entry'], f'{case_id} variant entry')
    adm = _pinned(ref['admission'], f'{case_id} variant admission')
    q.require(entry['case_admission_sha256'] == ref['admission']['sha256'], 'variant entry/admission mismatch')
    v = entry['variant']
    q.require(v['base_case_id'] == case_id and v['base_case_admission_sha256'] == base['case_admission_sha256'] and
              v['base_inventory_sha256'] == iv.inventory_digest(base['inputs']) and v['policy'] == iv.POLICY_A1,
              'variant not bound to the admitted base')
    q.require(iv.variant_identity(adm, base)['variant_identity_sha256'] == v['variant_identity_sha256'] and
              entry['case_id'] == f"{case_id}~{adm['variant_id']}", 'variant identity mismatch')
    receipt = Path(entry['input_dir']) / 'VARIANT_FACADE_RECEIPT.json'
    q.require(q.sha(receipt) == v['facade_receipt_sha256'], 'variant facade receipt changed')
    iv.verify_inventory(entry['input_dir'], entry['inputs'], 'R10')
    q.require(q.sha(ref['keys']['path']) == ref['keys']['sha256'], 'variant key receipt changed')
    iv.verify_key_receipt(ref['keys']['path'], base, entry, params)
    for k in ('model_init_utc', 'model_end_utc', 'bundle_sha256', 'metadata_sha256'):
        q.require(entry.get(k) == base.get(k), f'variant {k} differs from base')
    return dict(entry, variant_admission_id=adm['variant_id'])


def base_authority(ref, case_id, screen6):
    """Per-case base authority (e.g. the issued 0219 supplement with sizes); names/SHA/clocks must equal the
    paid SCREEN6 v2 entry exactly."""
    base = iv.load_base_case(ref['path'], ref['sha256'], case_id)
    q.require([(x['name'], x['sha256']) for x in base['inputs']] == [(x['name'], x['sha256']) for x in screen6['inputs']] and
              all(base[k] == screen6[k] for k in ('case_admission_sha256', 'model_init_utc', 'model_end_utc', 'input_dir',
                                                  'bundle_sha256', 'metadata_sha256')),
              f'{case_id} base authority differs from the paid SCREEN6 v2 entry')
    return base


def c_identity(case_id, wrfout_files, ref):
    """Evidence only: rerun C history vs the archived SCREEN4 C member (all 54 files by name + SHA)."""
    q.require(q.sha(ref['path']) == ref['sha256'], 'SCREEN4 C reference receipt changed')
    old = {Path(f['path']).name: f['sha256'] for f in q.load(ref['path'])['wrfout_files']}
    new = {Path(f['path']).name: f['sha256'] for f in wrfout_files}
    diff = sorted(n for n in set(old) | set(new) if old.get(n) != new.get(n))
    return {'reference': ref, 'n_reference': len(old), 'n_rerun': len(new), 'identical': not diff and len(old) == 54,
            'differing_files': diff[:60], 'claim': 'IDENTITY_EVIDENCE_ONLY'}


def validate(admission, expected, *, gpu=False):
    q.require(q.sha(admission) == expected, 'W1c admission changed')
    a = q.load(admission)
    q.require(a.get('schema') == SCHEMA and a.get('authorized_by') == 'manager' and a.get('scope') == SCOPE and
              a.get('failure_policy') == 'STOP_ON_FIRST_FAIL' and a.get('NO_RESUME') is True and
              a.get('auto_retry') is False, 'W1c scope/policy changed')
    out = Path(a['output_root'])
    q.require(out == OUTPUT_ROOT, 'W1c output root differs from the sealed layout')
    rows = identities(a['identities'])
    q.require(a['base_admission']['sha256'] == BASE_ADMISSION_SHA, 'base admission pin changed')
    for case_id in CASES:
        q.require(a['C_identity_reference'][case_id]['path'] == SCREEN4_C_RECEIPTS[case_id], 'C identity reference path changed')
    q.require(a['prime']['sha256'] == PRIME_SHA, 'prime pin changed')
    prime = _pinned(a['prime'], 'prime')
    q.require(prime['status'] == 'E41_PASSED_PENDING_NULL_CHANGE_RESUME', 'prime status changed')
    inv = _pinned({'path': str(CACHE_INVENTORY), 'sha256': CACHE_INVENTORY_SHA}, 'W1c cache inventory')
    q.require(inv.get('schema') == 'wrf_gpu2.w1c.cache_inventory.v1' and inv.get('prime_sha256') == PRIME_SHA and
              inv.get('layer') == a['cache_layer'], 'cache inventory not bound to this prime/layer')
    try:
        frozen, added = w1c_cache.frozen_inventory(prime['compiled'], inv['compiled'], prime['programs'])
    except ValueError as exc:
        raise ValueError(f'W1c cache inventory refused: {exc}') from None
    q.require(sorted(added) == sorted(inv.get('added_after_prime', [])) and len(added) == CACHE_ADDED_AFTER_PRIME,
              'cache inventory additions differ from the pinned ALP1 C3 set')
    # from here on every cache guard (seed / unchanged / loaded_artifacts in run_member) uses the frozen superset;
    # loaded programs stay the prime's (prime['programs'] unchanged: d01 bff015d2, fused/d02 aedca60f)
    prime = dict(prime, compiled=frozen, compiled_added_after_prime=added)
    q.require(a['logical_root'] == str(c.LOGICAL) and a['cache_layer'] == str(c.LOGICAL / 'bundle_layer'),
              'approved cache/environment namespace changed')
    q.require(a['parameters']['C']['sha256'] == C_PARAMS_SHA and a['parameters']['SIC_1L'] == a['parameters']['C'],
              'both arms must use the C parameter JSON')
    params = Path(a['parameters']['C']['path'])
    q.require(q.sha(params) == C_PARAMS_SHA and q.load(params) == c.parameters(100e6, .33), 'C parameter operand tree differs')
    cases = {}
    for case_id in CASES:
        screen6 = iv.load_base_case(a['base_admission']['path'], BASE_ADMISSION_SHA, case_id)
        base = base_authority(a['base_authorities'][case_id], case_id, screen6)
        q.require(base['bundle_sha256'] == q.BUNDLE_SHA and base['metadata_sha256'] == q.META_SHA, 'base engine binding changed')
        iv.verify_inventory(base['input_dir'], base['inputs'], 'R2')
        cases[(case_id, 'C')] = base
        cases[(case_id, 'SIC_1L')] = variant_case(case_id, a['variants'][case_id], base, params)
    for r in rows:
        q.require(r.get('raw_wrfout_dir') in (None, str(out / r['our_id'] / 'wrfout')), 'GO raw_wrfout_dir differs from the sealed layout')
        want = cases[(r['base_case_id'], r['arm_id'])]['case_id']
        q.require(r['case_id'] == want, 'row case_id differs from the staged case')
        if r['arm_id'] != 'C':
            q.require(r['variant_id'] == cases[(r['base_case_id'], r['arm_id'])]['variant']['variant_id'] and
                      r['variant_identity_sha256'] == cases[(r['base_case_id'], r['arm_id'])]['variant']['variant_identity_sha256'],
                      'row variant identity differs from the staged variant')
        r['runtime_parameters'] = a['parameters'][r['arm_id']]
        r.setdefault('extract_output_dir', str(out / r['our_id'] / 'extract_not_requested'))
        r['input_case_id'] = want
    if gpu:
        q.require(a['execution_authority'] == 'MANAGER_GATE_AFTER_ALISIOS_SIGNED_GO', 'W1c execution authority mismatch')
    return a, rows, prime, cases


def registry(root, member, case, *, smoke):
    """screen4.registry with W1C purpose/tags; issue = base cycle; variant identity in the binding."""
    from gpuwrf.optimize.registry import Invocation
    binding = {'case_id': member['base_case_id'], 'arm_id': member['arm_id'], 'input_case_id': case['case_id'],
               'model_init_utc': case['model_init_utc'], 'model_end_utc': case['model_end_utc'],
               'row_id': member['row_id'], 'NO_RESUME': True}
    if 'variant' in case:
        binding.update(variant_id=case['variant']['variant_id'],
                       variant_identity_sha256=case['variant']['variant_identity_sha256'])
    meta = q.registry_metadata(root, Path(case['input_dir']), smoke=smoke, issue=member['base_case_id'], binding=binding)
    data = q.load(meta); data['purpose'] = 'W1C-1L cold/no resume; ' + member['row_id']; q.save(meta, data)
    inv = Invocation(meta, Path(case['input_dir']), root)
    inv.run_id = member['registry_run_id']; inv.start['run_id'] = inv.run_id
    tags = [TAG, member['base_case_id'], member['arm_id'], 'det0']
    if 'variant' in case:
        tags.append(case['variant']['variant_id'])
    q.require(all(len(t) <= 40 for t in tags), 'registry tag > 40 chars')
    inv.start['tags'] = tags
    return inv


def preflight_ok(rows, cases, here, admission_sha):
    for r in rows:
        case = cases[(r['base_case_id'], r['arm_id'])]
        pref = q.load(LANE / 'W1C_INPUT_CHECKS' / f"{r['base_case_id']}.{r['arm_id']}" / 'NATIVE_INPUT_PREFLIGHT.json')
        q.require(pref['status'] == 'PASS' and pref['case_id'] == case['case_id'] and
                  pref['admission_sha256'] == admission_sha and pref['checker_sha256'] == q.sha(here / 'screen4_inputs.py') and
                  pref['input_sha256'] == {x['name']: x['sha256'] for x in case['inputs']} and
                  pref['created_unix_s'] >= q.epoch(os.environ['GA_START_UTC']),
                  'W1c current-sequence native input preflight missing/stale')


def main():
    p = argparse.ArgumentParser(allow_abbrev=False); p.add_argument('--seal-sha', required=True)
    p.add_argument('--admission', type=Path, required=True); p.add_argument('--schema-smoke', type=Path)
    x = p.parse_args(); here = Path(__file__).resolve().parent; q.sealed(here, x.seal_sha)
    smoke = os.environ.get('WRF_GPU2_SEQ_SMOKE') == '1' or x.schema_smoke is not None
    a, rows, prime, cases = validate(x.admission, os.environ['W1AQ_ADMISSION_SHA'], gpu=not smoke)
    s.registry = registry   # the only seam in the reused SCREEN4 member path (run_member resolves it at call time)
    if x.schema_smoke:
        x.schema_smoke.mkdir(exist_ok=False)
        for r in rows:
            d = x.schema_smoke / r['our_id']; d.mkdir(); inv = registry(d, r, cases[(r['base_case_id'], r['arm_id'])], smoke=True)
            inv.begin({}); inv.finish(0)
        return 0
    preflight_ok(rows, cases, here, os.environ['W1AQ_ADMISSION_SHA'])
    if smoke:
        root = Path(os.environ['WRF_GPU2_SMOKE_DIR']) / 'work'; root.mkdir(); (root / 'environment').mkdir()
        seed(c.LOGICAL / 'bundle_layer', root / 'bundle_layer', prime['compiled'])
        fds = (); end = time.time() + 1200; deadline = time.monotonic() + 900
    else:
        from gpuwrf.optimize.contrasts import _lock_fds
        fds = _lock_fds(os.environ)
        q.require(fds and os.environ.get('GPUWRF_GPU_LOCK_LABEL') == a['gpu_label'], 'official lane GPU lock required')
        q.require(os.environ.get('GA_GATE_ID') == a['gate_id'], 'foreign CPU lease job')
        q.require(os.environ.get('GA_RW_GATE_ID', '').startswith('rwgate_'), 'manager reviewed gate invocation marker required')
        w = a['window']
        end = min(q.epoch(w['hard_end']), q.epoch(os.environ['GA_END_UTC']))
        start = q.epoch(os.environ['GA_START_UTC'])
        q.require(q.epoch(w['not_before']) <= start <= q.epoch(w['latest_sequence_start']) and
                  time.time() <= q.epoch(w['latest_GPU_start']) and time.time() + 34 * 60 <= end,
                  'W1c window/34min remaining refused')
        q.require(shutil.disk_usage('<DATA_ROOT>').free >= MEMBER_FLOOR_GIB * q.GIB, 'W1c DATA entry floor 437 GiB refused')
        expiry = re.findall(r'^expires-utc:\s*(.+)$', Path(f"/tmp/wrf_gpu2_run_approvals/{a['gpu_label']}.lease").read_text(), re.M)
        q.require(len(expiry) == 1 and q.epoch(expiry[0]) >= end, 'W1c GPU lease expires early')
        root = Path(a['output_root']); root.mkdir(exist_ok=False)
        deadline = min(time.monotonic() + GPU_SECONDS_TOTAL_CAP,
                       q.execution_deadline(time.monotonic(), end - time.time(),
                                            os.environ.get('ALISIOS_OPTIMIZE_WALL_DEADLINE') or None))
    unchanged(c.LOGICAL / 'bundle_layer', prime['compiled'])
    sa = dict(a, parameters={'C': a['parameters']['C'], 'SIC_1L': a['parameters']['C']})
    sic_used = {'s': 0.0}

    def peak():
        q.require(q.own_bytes(root) <= PRIVATE_PEAK_GIB * q.GIB, 'W1c private peak 24 GiB exceeded')

    def execute(member):
        case = cases[(member['base_case_id'], member['arm_id'])]
        if member['arm_id'] != 'C':
            q.require(sic_used['s'] + MEMBER_CAP <= SIC_PAIR_CAP, 'SIC pair GPU budget 900 s exhausted')
        t0 = time.monotonic()
        result = s.run_member(root, member, case, sa, prime, fds, deadline, end, smoke=smoke, check_extra=peak)
        if member['arm_id'] != 'C':
            sic_used['s'] += time.monotonic() - t0
        result.update(input_case_id=case['case_id'], variant=case.get('variant'), source_cycle=member['base_case_id'],
                      base_case_id=member['base_case_id'], raw_deleted=False, raw_policy='KEEP_UNTIL_MANAGER_DISPOSITION')
        if member['arm_id'] == 'C' and result['status'] == 'PASS':
            result['C_identity_vs_SCREEN4'] = c_identity(member['base_case_id'], result['wrfout_files'],
                                                         a['C_identity_reference'][member['base_case_id']])
        q.save(root / f"MEMBER_{member['ordinal']}_DONE.json", result)
        return result
    def before_member(member):
        if smoke:
            return 0.0
        waited = wait_for_floor(lambda: shutil.disk_usage('<DATA_ROOT>').free / q.GIB, MEMBER_FLOOR_GIB,
                                lambda: min(deadline - time.monotonic(), end - 300 - time.time()))
        # window re-check after any wait (v46-seqharden lesson): the clock moved, the member must still fit
        q.require(time.time() + MEMBER_CAP + MEMBER_SETUP_S <= end - 300 and
                  time.monotonic() + MEMBER_CAP + MEMBER_SETUP_S <= deadline, 'W1c window/deadline exhausted after disk wait')
        expiry = re.findall(r'^expires-utc:\s*(.+)$', Path(f"/tmp/wrf_gpu2_run_approvals/{a['gpu_label']}.lease").read_text(), re.M)
        q.require(len(expiry) == 1 and q.epoch(expiry[0]) >= time.time() + MEMBER_CAP + MEMBER_SETUP_S, 'W1c GPU lease expires before member end')
        return waited
    results = run_rows(rows, execute, before_member)
    for member, result in zip(rows, results):
        out = root / member['our_id']; out.mkdir(exist_ok=True)
        for key, value in {'path': str(out), 'wrfout': str(out / 'wrfout'), 'GPU_seconds': None, 'compiled_new': None,
                           'loaded_cheap_keys': {}, 'registry_complete': False}.items():
            result.setdefault(key, value)
        if not (out / 'MEMBER_RECEIPT.json').exists():
            q.save_terminal(out / 'MEMBER_RECEIPT.json', result)
        elif q.load(out / 'MEMBER_RECEIPT.json') != result:
            q.save_terminal(out / 'MEMBER_RECEIPT.v2.json', result)
        q.save(root / f"MEMBER_{member['ordinal']}_DONE.json", result)
    failed = any(r['status'] == 'FAIL' for r in results)
    disk_halt = any(str(r.get('reason', '')).startswith('HALT_DISK_FLOOR') for r in results)
    pre_halt = any(str(r.get('reason', '')).startswith('HALT_PRE_MEMBER') for r in results)
    final = {'status': 'W1C_1L_PASS' if all(r['status'] == 'PASS' for r in results) else
             ('W1C_1L_FAIL' if failed else ('W1C_1L_HALTED_DISK_FLOOR' if disk_halt else
                                            ('W1C_1L_HALTED_PRE_MEMBER' if pre_halt else 'W1C_1L_FAIL'))),
             'global_abort': any(r.get('global_abort') for r in results),
             'members': results, 'halted_after_member4': all(r['attempted'] for r in results),
             'halted_on_first_fail': any(r['status'] == 'FAIL' for r in results), 'member_slots_recorded': 4,
             'auto_continue': False, 'auto_retry': False, 'NO_RESUME': True, 'restart_qualification': 'OPEN',
             'failure_policy': 'STOP_ON_FIRST_FAIL', 'seal_sha256': x.seal_sha,
             'admission_sha256': os.environ['W1AQ_ADMISSION_SHA'],
             'raw_release': 'PER_MEMBER_AFTER_ALISIOS_RELEASE (w1c_raw_release.py, journal ' + RAW_JOURNAL + ')',
             'GPU_entry_cap_s': GPU_SECONDS_TOTAL_CAP, 'member_floor_GiB': MEMBER_FLOOR_GIB,
             'disk_wait_s_total': sum(r.get('disk_wait_s') or 0 for r in results),
             'halted_on_disk_floor': any(str(r.get('reason', '')).startswith('HALT_DISK_FLOOR') for r in results),
             'SIC_pair_seconds': sic_used['s'],
             'device': 'cpu' if smoke else 'gpu', 'real_lock_used': not smoke, 'completed_phases': PHASES,
             'registry_schema_valid': True, 'production_registry_writes': 0 if smoke else
             (2 * sum(r['attempted'] for r in results) if all(not r['attempted'] or r.get('registry_complete') for r in results) else None),
             'native_gpu_gates': 'NOT-APPLICABLE' if smoke else 'PER_MEMBER_RESULT'}
    q.save_terminal(root / 'W1C_1L_RESULT.json', final)
    if smoke:
        q.save(os.environ['WRF_GPU2_SMOKE_RECEIPT'], final)
    return 0 if final['status'] == 'W1C_1L_PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
