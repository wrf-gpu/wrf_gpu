"""SCREEN7: 0530 x {C_new, alp1_low, alp1_high} on the paid ALP1 C3 FULL program; cold, no resume, no retry.

Reuses the W1c path unchanged (screen4.run_member, screen4_member.py zero-compile refusal, w1a_* cache/consumer/prime
gates, w1c_disk.run_rows STOP_ON_FIRST_FAIL + per-member DATA floor with bounded wait, signed-GO rows). New here:
  * rows = the ALISIOS-confirmed SCREEN7 bindings (da1f13b8) [+ the signed GO, identity fields equal, same order];
  * prime = ALP1_NEW_PRIME (FULL selection, 1873 compiled == live paid layer), programs d01 12b0faf2 / fused 043a7e43;
  * the FULL operand check of run_member (seam `c.parameters` -> alp1_contract.parameters(alp1 of the member));
  * a CAP PER MEMBER incl. post (registry, model, loaded-program/cache/history checks): 450 s; the model process is
    killed at the member deadline, every post check refuses past it; the entry itself is hard-capped by s7_budget.py
    on the shared ALP1 wave ledger, and a member is admitted only if its full cap fits the remaining wave budget.
Registry issue = the 0530 base cycle; tags W1A_TECH_ALP1_V1/SCREEN7. Raw KEEP until a per-member ALISIOS directive
(s7_raw_release.py, which exports the decision metrics before the first unlink).
"""
import argparse
import os
from pathlib import Path
import re
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import icvariant as iv
import w1c_rows
import w1c_disk as dk
import alp1_contract as g
import screen4 as s
import s7_keys
c = s.c
q = c.q
from w1a_cache import seed, unchanged

SCHEMA = 'wrf_gpu2.screen7.admission.v1'
SCOPE = 'SCREEN7_0530_COLD_NO_RESUME'
CASE = '20260530_18z'
ARMS = ('C_new', 'alp1_low', 'alp1_high')
ALP1 = {'C_new': .23, 'alp1_low': .184, 'alp1_high': .276}
PARAM_FILE = {'C_new': 'C3_prime', 'alp1_low': 'J3a', 'alp1_high': 'J3b'}
TAG = 'W1A_TECH_ALP1_V1'
BINDINGS_SHA = 'da1f13b8a788ac373df501c76dc946b34f950f76abd0ae811fca972ac488b701'
CONFIRMATION = ('<DATA_ROOT>/alisios/state/manager/cross/TO_WRF_GPU2_FROM_ALISIOS_MGR_SCREEN7_0530_IDS_CONFIRMED_20261031T0031Z.md',
                '71009a0dbf48752a6292dc70b1e5dd81739daeedce0f2fc680494e490aa5904c')
BASE_ADMISSION_SHA = '37e2de3be3a68e1cb7264c776aa476961a886e306507f20704f1fea26cb5fd8c'   # SCREEN6 admission v2 (0530 entry)
PRIME_PATH = '<DATA_ROOT>/wrf_gpu2/scratch/v45-waveops/W1A_ALP1_20261031/ALP1_NEW_PRIME.json'
LANE = Path('<USER_HOME>/wrf_gpu2_lanes/v46-icvariant')
DATA = Path('<DATA_ROOT>/wrf_gpu2/scratch/v46-icvariant')
OUTPUT_ROOT = DATA / 'SCREEN7_20261031'      # pinned layout (LAYOUT.json); raw = <OUTPUT_ROOT>/<our_id>/wrfout
RAW_JOURNAL = 'RAW_DELETE_JOURNAL_SCREEN7.jsonl'
PHASES = ['preflight', 'member1', 'member2', 'member3', 'halt']
MEMBER_CAP_S = 450          # per member incl. registry + model + post (ALISIOS 06:58Z); model alone stays <= 360 (run_member)
RELEASE_S = 120           # == s7_budget.RELEASE_S (lock release reserve of the wave ledger)
FINAL_RESERVE_S = 30        # result/receipt writes after the last member, inside the entry cap
MEMBER_FLOOR_GIB = dk.MEMBER_FLOOR_GIB   # 437 = 430 + one member raw 5.16 + 1 margin, per member, bounded wait
PRIVATE_PEAK_GIB = 24
# w1c_disk.wait_for_floor refuses when the usable time cannot fit MEMBER_CAP + MEMBER_SETUP_S (+ one poll); SCREEN7's
# member cap already includes setup -> the threshold becomes 450 s (+ poll). Module globals are read at call time.
dk.MEMBER_CAP, dk.MEMBER_SETUP_S = MEMBER_CAP_S, 0


def _pinned(ref, what):
    q.require(isinstance(ref, dict) and ref.get('path') and q.sha(ref['path']) == ref.get('sha256'), f'{what} changed/missing')
    return q.load(ref['path'])


def layout_ok(layout):
    """LAYOUT.json is the immutable pointer ALISIOS binds (manager 07:32Z): regular file, mode 0444, same output_root."""
    layout = Path(layout)
    q.require(layout.is_file() and not layout.is_symlink() and (layout.stat().st_mode & 0o777) == 0o444 and
              q.load(layout)['output_root'] == str(OUTPUT_ROOT), 'SCREEN7 LAYOUT.json not the read-only (0444) sealed pointer')


def identities(ids, *, go_required):
    """Rows from the ALISIOS-confirmed bindings; a signed GO (if present/required) must list the same identities."""
    q.require(ids['bindings']['sha256'] == BINDINGS_SHA, 'SCREEN7 bindings pin changed')
    q.require(ids['confirmation'] == {'path': CONFIRMATION[0], 'sha256': CONFIRMATION[1]} and
              q.sha(CONFIRMATION[0]) == CONFIRMATION[1], 'SCREEN7 ALISIOS ID confirmation changed')
    rows = [dict(r) for r in _pinned(ids['bindings'], 'SCREEN7 bindings')['rows']]
    q.require([r['ordinal'] for r in rows] == [1, 2, 3] and [r['arm_id'] for r in rows] == list(ARMS),
              'SCREEN7 rows/order changed')
    for r in rows:
        arm = r['arm_id']; ref = r['runtime_parameters']; pf = s7_keys.PARAMS[PARAM_FILE[arm]]
        q.require(r['case_id'] == CASE and re.fullmatch('[0-9a-f]{32}', r['our_id']) and
                  r['registry_run_id'] == 'optimize:' + r['our_id'] and r['row_id'] == f'{TAG}.{CASE}.{arm}.det0' and
                  r['launcher_policy'] == 'COLD_NO_RESUME' and r['model_hours'] == 18 and r['cold_compile_allowed'] is False and
                  sorted(r['program_selection']) == g.SELECTION, 'SCREEN7 identity/policy/selection mismatch')
        q.require(r['parameters'] == dict(g.DEFAULTS, alp1=ALP1[arm]), 'SCREEN7 parameter envelope changed')
        q.require(ref == {'path': pf[0], 'sha256': pf[1]} and q.sha(ref['path']) == ref['sha256'] and
                  q.load(ref['path']) == g.parameters(ALP1[arm]), 'SCREEN7 exact FULL operand bytes/tree changed')
    q.require(len({r['our_id'] for r in rows}) == 3, 'duplicate SCREEN7 identities')
    if ids.get('go') is not None:
        signed = w1c_rows.signed_rows(_pinned(ids['go'], 'SCREEN7 signed GO'), lambda r: _pinned(r, 'GO draft pinned by the GO'))
        q.require([tuple(m.get(k) for k in w1c_rows.IDENTITY_FIELDS) for m in signed] ==
                  [tuple(r[k] for k in w1c_rows.IDENTITY_FIELDS) for r in rows], 'signed GO identities != confirmed bindings')
        for r, m in zip(rows, signed):
            q.require(m.get('raw_wrfout_dir') in (None, str(OUTPUT_ROOT / r['our_id'] / 'wrfout')),
                      'GO raw_wrfout_dir differs from the sealed layout')
            if m.get('extract_output_dir'):
                r['extract_output_dir'] = m['extract_output_dir']   # ALISIOS extract target (release evidence lives there)
    else:
        q.require(not go_required, 'SCREEN7 execution needs the signed ALISIOS GO')
    return rows


def prime_ok(ref):
    q.require(ref['path'] == PRIME_PATH, 'SCREEN7 foreign prime path')
    prime = _pinned(ref, 'ALP1 C3 FULL prime')
    q.require(prime['status'] == 'ALP1_C3_PRIME_PASS' and prime['bundle_sha256'] == q.BUNDLE_SHA and
              prime['selection'] == g.SELECTION and prime['runtime_parameters'] == g.parameters(.23) and
              prime['device'] == 'gpu' and prime['actual_GPU_E41'] == 'PASS' and
              prime['source_layer'] == str(c.LOGICAL / 'bundle_layer') and
              sorted(prime['programs']['domains']) == ['d01', 'fused/d02'] and len(prime['compiled']) == 1873,
              'SCREEN7 prime is not the qualified FULL C3 prime')
    return prime


def keys_ok(ref, base):
    """R10 receipt: re-check its parts and recombine (pure); bound to this 0530 base inventory."""
    doc = _pinned(ref, 'SCREEN7 R10 key receipt')
    q.require(doc['schema'] == s7_keys.SCHEMA and doc['status'] == 'PASS', 'SCREEN7 R10 key receipt not PASS')
    parts = doc['parts']
    may = _pinned(parts['20260530_18z']['receipt'], 'R10 0530 part'); feb = _pinned(parts['20260219_18z']['receipt'], 'R10 0219 part')
    again = s7_keys.combine(may, feb)
    q.require(again['comparisons'] == doc['comparisons'] and may['inventory_sha256'] == iv.inventory_digest(base['inputs']) and
              may['input_dir'] == base['input_dir'], 'SCREEN7 R10 receipt not reproducible / not this 0530 base')
    return doc


def validate(admission, expected, *, gpu=False):
    q.require(q.sha(admission) == expected, 'SCREEN7 admission changed')
    a = q.load(admission)
    q.require(a.get('schema') == SCHEMA and a.get('authorized_by') == 'manager' and a.get('scope') == SCOPE and
              a.get('failure_policy') == 'STOP_ON_FIRST_FAIL' and a.get('NO_RESUME') is True and
              a.get('auto_retry') is False and a.get('member_cap_s') == MEMBER_CAP_S, 'SCREEN7 scope/policy changed')
    q.require(Path(a['output_root']) == OUTPUT_ROOT, 'SCREEN7 output root differs from the sealed layout')
    layout_ok(Path(__file__).resolve().parent / 'LAYOUT.json')
    rows = identities(a['identities'], go_required=gpu)
    q.require(a['base_admission']['sha256'] == BASE_ADMISSION_SHA, 'base admission pin changed')
    base = iv.load_base_case(a['base_admission']['path'], BASE_ADMISSION_SHA, CASE)
    q.require(base['bundle_sha256'] == q.BUNDLE_SHA and base['metadata_sha256'] == q.META_SHA, 'base engine binding changed')
    iv.verify_inventory(base['input_dir'], base['inputs'], 'R2')
    prime = prime_ok(a['prime'])
    q.require(a['logical_root'] == str(c.LOGICAL) and a['cache_layer'] == str(c.LOGICAL / 'bundle_layer'),
              'approved cache/environment namespace changed')
    a['keys_doc'] = keys_ok(a['keys'], base)
    q.require(isinstance(a.get('budget_ledger'), dict) and
              a['budget_ledger'].get('path') == '<USER_HOME>/wrf_gpu2_lanes/v45-waveops/ALP1_WAVE_GPU_BUDGET_20261031.jsonl' and
              re.fullmatch('[0-9a-f]{64}', a['budget_ledger'].get('sha256', '')), 'SCREEN7 wave ledger pin missing')
    for r in rows:
        r.setdefault('extract_output_dir', str(OUTPUT_ROOT / r['our_id'] / 'extract_not_requested'))
    if gpu:
        q.require(a['execution_authority'] == 'MANAGER_GATE_AFTER_ALISIOS_SIGNED_GO', 'SCREEN7 execution authority mismatch')
    return a, rows, prime, dict(base, case_id=CASE)


def registry(root, member, case, *, smoke):
    """screen4.registry with the ALP1/SCREEN7 purpose and tags; issue = the 0530 base cycle."""
    from gpuwrf.optimize.registry import Invocation
    binding = {'case_id': CASE, 'arm_id': member['arm_id'], 'model_init_utc': case['model_init_utc'],
               'model_end_utc': case['model_end_utc'], 'row_id': member['row_id'], 'NO_RESUME': True,
               'selection': g.SELECTION, 'raw_policy': 'KEEP_UNTIL_SPECIFIC_ALISIOS_DISPOSITION'}
    meta = q.registry_metadata(root, Path(case['input_dir']), smoke=smoke, issue=CASE, binding=binding)
    data = q.load(meta); data['purpose'] = 'SCREEN7 ALP1 FULL selection cold/no resume; ' + member['row_id']; q.save(meta, data)
    inv = Invocation(meta, Path(case['input_dir']), root)
    inv.run_id = member['registry_run_id']; inv.start['run_id'] = inv.run_id
    tags = [TAG, 'SCREEN7', CASE, member['arm_id'], 'det0']
    q.require(all(len(t) <= 40 for t in tags), 'registry tag > 40 chars')
    inv.start['tags'] = tags
    return inv


def full_operand(alp):
    """Seam for run_member's operand gate (c.parameters(nt_c, entrainment)): the member's exact FULL tree."""
    want = g.parameters(alp)

    def parameters(nt_c, entrainment):
        q.require(nt_c == 1e8 and entrainment == .33, 'SCREEN7 fixed nt_c/entrainment changed')
        return want
    return parameters


def preflight_ok(rows, case, here, admission_sha):
    for r in rows:
        pref = q.load(LANE / 'S7_INPUT_CHECKS' / r['arm_id'] / 'NATIVE_INPUT_PREFLIGHT.json')
        q.require(pref['status'] == 'PASS' and pref['case_id'] == CASE and pref['admission_sha256'] == admission_sha and
                  pref['checker_sha256'] == q.sha(here / 'screen4_inputs.py') and
                  pref['input_sha256'] == {x['name']: x['sha256'] for x in case['inputs']} and
                  pref['S7_runtime_parameters'] == r['runtime_parameters'] and
                  pref['created_unix_s'] >= q.epoch(os.environ['GA_START_UTC']),
                  'SCREEN7 current-sequence native input preflight missing/stale')


def admit_member(mono, deadline, wall, end, info, lease_end):
    """A member starts only if its FULL cap (incl. post) fits the entry deadline, the window (end - 300 s release),
    the remaining wave budget (3600 - used - release) and the GPU lease. Raises otherwise (-> HALT_PRE_MEMBER)."""
    q.require(mono + MEMBER_CAP_S <= deadline, 'SCREEN7 member cap does not fit the entry deadline')
    q.require(wall + MEMBER_CAP_S <= end - 300, 'SCREEN7 member cap does not fit the window')
    q.require(info['prior_used_s'] + info['entry_elapsed_s'] + MEMBER_CAP_S <= g.TOTAL_GPU_SECONDS - RELEASE_S,
              'SCREEN7 member cap does not fit the remaining wave GPU budget')
    q.require(lease_end >= wall + MEMBER_CAP_S, 'SCREEN7 GPU lease expires before member end')


def lease_expiry(label):
    expiry = re.findall(r'^expires-utc:\s*(.+)$', Path(f'/tmp/wrf_gpu2_run_approvals/{label}.lease').read_text(), re.M)
    q.require(len(expiry) == 1, 'SCREEN7 GPU lease unreadable')
    return q.epoch(expiry[0])


def _final_budget(read):
    try:
        return read()
    except Exception as exc:   # never lose the result: the ledger END written by s7_budget.py is the charge
        return {'error': f'{type(exc).__name__}: {exc}'}


def main():
    p = argparse.ArgumentParser(allow_abbrev=False); p.add_argument('--seal-sha', required=True)
    p.add_argument('--admission', type=Path, required=True); p.add_argument('--schema-smoke', type=Path)
    x = p.parse_args(); here = Path(__file__).resolve().parent; q.sealed(here, x.seal_sha)
    smoke = os.environ.get('WRF_GPU2_SEQ_SMOKE') == '1' or x.schema_smoke is not None
    a, rows, prime, case = validate(x.admission, os.environ['W1AQ_ADMISSION_SHA'], gpu=not smoke)
    s.registry = registry   # seam 1 of the reused SCREEN4 member path (run_member resolves it at call time)
    if x.schema_smoke:
        x.schema_smoke.mkdir(exist_ok=False)
        for r in rows:
            d = x.schema_smoke / r['our_id']; d.mkdir(); inv = registry(d, r, case, smoke=True)
            inv.begin({}); inv.finish(0)
        return 0
    preflight_ok(rows, case, here, os.environ['W1AQ_ADMISSION_SHA'])
    budget_used = lambda: None
    if smoke:
        root = Path(os.environ['WRF_GPU2_SMOKE_DIR']) / 'work'; root.mkdir(); (root / 'environment').mkdir()
        seed(c.LOGICAL / 'bundle_layer', root / 'bundle_layer', prime['compiled'])
        fds = (); end = time.time() + 1200; deadline = time.monotonic() + 900
    else:
        from gpuwrf.optimize.contrasts import _lock_fds
        import s7_budget
        fds = _lock_fds(os.environ)
        q.require(fds and os.environ.get('GPUWRF_GPU_LOCK_LABEL') == a['gpu_label'], 'official lane GPU lock required')
        q.require(os.environ.get('GA_GATE_ID') == a['gate_id'], 'foreign CPU lease job')
        q.require(os.environ.get('GA_RW_GATE_ID', '').startswith('rwgate_'), 'manager reviewed gate invocation marker required')
        ident = os.environ['W1A_ALP1_BUDGET_ENTRY_ID']
        budget_used = lambda: s7_budget.active(ident, x.seal_sha)
        info = budget_used()
        w = a['window']
        end = min(q.epoch(w['hard_end']), q.epoch(os.environ['GA_END_UTC']))
        start = q.epoch(os.environ['GA_START_UTC'])
        q.require(q.epoch(w['not_before']) <= start <= q.epoch(w['latest_sequence_start']) and
                  time.time() <= q.epoch(w['latest_GPU_start']) and time.time() + MEMBER_CAP_S + 300 <= end,
                  'SCREEN7 window refused')
        q.require(shutil.disk_usage('<DATA_ROOT>').free >= MEMBER_FLOOR_GIB * q.GIB, 'SCREEN7 DATA entry floor 437 GiB refused')
        q.require(lease_expiry(a['gpu_label']) >= end, 'SCREEN7 GPU lease expires early')
        root = Path(a['output_root']); root.mkdir(exist_ok=False)
        deadline = min(time.monotonic() + info['entry_cap_s'] - info['entry_elapsed_s'] - FINAL_RESERVE_S,
                       q.execution_deadline(time.monotonic(), end - time.time(),
                                            os.environ.get('ALISIOS_OPTIMIZE_WALL_DEADLINE') or None))
    unchanged(c.LOGICAL / 'bundle_layer', prime['compiled'])     # 1873 == live paid layer (incl. the 10 C3 additions)
    protected = [x.admission, Path(PRIME_PATH), Path(a['keys']['path'])] + [Path(r['runtime_parameters']['path']) for r in rows]
    stamps = q.stamps(protected)
    sa = dict(a, parameters={r['arm_id']: r['runtime_parameters'] for r in rows})

    def check_extra():
        q.require(q.own_bytes(root) <= PRIVATE_PEAK_GIB * q.GIB, 'SCREEN7 private peak 24 GiB exceeded')
        q.require(q.stamps(protected) == stamps, 'SCREEN7 admission/prime/keys/operand bytes changed')
        budget_used()

    def execute(member):
        t0 = time.monotonic(); member_deadline = min(deadline, t0 + MEMBER_CAP_S)
        original = c.parameters; c.parameters = full_operand(ALP1[member['arm_id']])   # seam 2: FULL operand gate
        try:
            result = s.run_member(root, member, case, sa, prime, fds, member_deadline, end, smoke=smoke, check_extra=check_extra)
        finally:
            c.parameters = original
        if result['status'] == 'PASS' and not smoke:
            proof = q.load(root / member['our_id'] / 'proofs/nested_pipeline_run.json')
            if proof.get('host_suspend', {}).get('suspended_s') != 0:
                result.update(status='FAIL', failure='SCREEN7 member crossed a host suspend')
        wall = time.monotonic() - t0
        if result['status'] == 'PASS' and wall > MEMBER_CAP_S:
            result.update(status='FAIL', failure=f'SCREEN7 member cap {MEMBER_CAP_S} s incl. post exceeded ({wall:.1f} s)')
        result.update(member_cap_s=MEMBER_CAP_S, member_wall_s=wall, source_cycle=CASE, raw_deleted=False,
                      raw_policy='KEEP_UNTIL_SPECIFIC_ALISIOS_DISPOSITION', alp1=ALP1[member['arm_id']])
        q.save(root / f"MEMBER_{member['ordinal']}_DONE.json", result)
        return result

    def before_member(member):
        if smoke:
            return 0.0
        waited = dk.wait_for_floor(lambda: shutil.disk_usage('<DATA_ROOT>').free / q.GIB, MEMBER_FLOOR_GIB,
                                   lambda: min(deadline - time.monotonic(), end - 300 - time.time()))
        # after any wait the clock moved: the member's FULL cap must still fit the entry, the window, the wave
        # budget and the lease (ALP1 rule: admit next only if the remaining fits its full cap)
        admit_member(time.monotonic(), deadline, time.time(), end, budget_used(), lease_expiry(a['gpu_label']))
        return waited
    results = dk.run_rows(rows, execute, before_member)
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
    reasons = [str(r.get('reason', '')) for r in results]
    status = ('SCREEN7_PASS' if all(r['status'] == 'PASS' for r in results) else
              'SCREEN7_FAIL' if any(r['status'] == 'FAIL' for r in results) else
              'SCREEN7_HALTED_DISK_FLOOR' if any(x.startswith('HALT_DISK_FLOOR') for x in reasons) else
              'SCREEN7_HALTED_PRE_MEMBER' if any(x.startswith('HALT_PRE_MEMBER') for x in reasons) else 'SCREEN7_FAIL')
    final = {'status': status, 'global_abort': any(r.get('global_abort') for r in results), 'members': results,
             'halted_after_member3': all(r['attempted'] for r in results),
             'halted_on_first_fail': any(r['status'] == 'FAIL' for r in results), 'member_slots_recorded': 3,
             'auto_continue': False, 'auto_retry': False, 'NO_RESUME': True, 'restart_qualification': 'OPEN',
             'failure_policy': 'STOP_ON_FIRST_FAIL', 'seal_sha256': x.seal_sha, 'admission_sha256': os.environ['W1AQ_ADMISSION_SHA'],
             'raw_release': 'PER_MEMBER_AFTER_ALISIOS_RELEASE (s7_raw_release.py: metrics export before unlink, journal ' + RAW_JOURNAL + ')',
             'member_cap_s': MEMBER_CAP_S, 'member_cap_definition': 'registry begin + native model + all post checks',
             'member_floor_GiB': MEMBER_FLOOR_GIB, 'disk_wait_s_total': sum(r.get('disk_wait_s') or 0 for r in results),
             'budget': None if smoke else _final_budget(budget_used), 'device': 'cpu' if smoke else 'gpu', 'real_lock_used': not smoke,
             'completed_phases': PHASES, 'registry_schema_valid': True,
             'production_registry_writes': 0 if smoke else
             (2 * sum(r['attempted'] for r in results) if all(not r['attempted'] or r.get('registry_complete') for r in results) else None),
             'native_gpu_gates': 'NOT-APPLICABLE' if smoke else 'PER_MEMBER_RESULT'}
    q.save_terminal(root / 'SCREEN7_RESULT.json', final)
    if smoke:
        q.save(os.environ['WRF_GPU2_SMOKE_RECEIPT'], final)
    return 0 if status == 'SCREEN7_PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
