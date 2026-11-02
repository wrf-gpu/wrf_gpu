"""SCREEN7 launcher host logic: per-member raw release with metrics export BEFORE the first unlink, wave-ledger
budget rules, R10 key combine, member admission (cap incl. post), identities. CPU, no JAX.

Pure modules load from scripts/w1c/{screen7,launcher,.}; launcher logic that needs the sealed reused modules
(screen4/alp1_contract/w1a_*) loads from the lane seal dir S7_SEALED_01 and is skipped where that is absent.
Tests that bind the real ALISIOS-confirmed bindings / parameter files / ledger skip when those are absent."""
import hashlib
import importlib
import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
S7 = REPO / 'scripts' / 'w1c' / 'screen7'
SEALED = Path('<USER_HOME>/wrf_gpu2_lanes/v46-icvariant/S7_SEALED_01')
BINDINGS = Path('<USER_HOME>/wrf_gpu2_lanes/v45-waveops/SCREEN7_0530_MEMBER_BINDINGS_PROPOSED_FULL_20261031.json')
LEDGER = Path('<USER_HOME>/wrf_gpu2_lanes/v45-waveops/ALP1_WAVE_GPU_BUDGET_20261031.jsonl')
for p in (S7, REPO / 'scripts' / 'w1c' / 'launcher', REPO / 'scripts' / 'w1c'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
rr = importlib.import_module('s7_raw_release')
keys = importlib.import_module('s7_keys')
need_bindings = pytest.mark.skipif(not BINDINGS.exists(), reason='real SCREEN7 bindings absent')

_INIT = datetime(2026, 5, 30, 18)
NAMES = [f"wrfout_d0{d}_{(_INIT + timedelta(seconds=math.ceil(h * 3600 / dt) * dt)).strftime('%Y-%m-%d_%H:%M:%S')}"
         for d, dt in ((1, 27), (2, 9), (3, 3)) for h in range(1, 19)]


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _w(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True); Path(p).write_text(json.dumps(obj)); return Path(p)


def _fake_doc(frames, row, *, records=18, tamper=False):
    recs = [{'tau_h': i + 1, 'source': {'path': str(p), 'sha256': ('0' * 64 if tamper and i == 0 else _sha(p))}}
            for i, p in enumerate(frames[:records])]
    return {'schema': 'wrf_gpu2.screen7.metrics_before_unlink.v1', 'row_id': row['row_id'], 'our_id': row['our_id'],
            'hours': 18, 'records': recs, 'decision_summary_18h_mean': {'north_sc': {'LWP': 1.0}}, 'truth_reads': 0}


# ------------------------------------------------------------------ release world (real bindings, synthetic raw)
@pytest.fixture
def world(tmp_path, monkeypatch):
    if not BINDINGS.exists():
        pytest.skip('real SCREEN7 bindings absent')
    bound = json.loads(BINDINGS.read_text())['rows']
    root = tmp_path / 'SCREEN7_X'; root.mkdir()
    rows = [{k: b[k] for k in ('ordinal', 'our_id', 'registry_run_id', 'row_id')} |
            {'extract_output_dir': str(tmp_path / 'extracts' / b['our_id']), 'raw_wrfout_dir': str(root / b['our_id'] / 'wrfout')}
            for b in bound]
    go = _w(tmp_path / 'GO.json', {'members': rows})
    adm = _w(tmp_path / 'ADM.json', {'schema': 'wrf_gpu2.screen7.admission.v1', 'authorized_by': 'manager',
                                     'output_root': str(root),
                                     'identities': {'bindings': {'path': str(BINDINGS), 'sha256': rr.BINDINGS_SHA},
                                                    'go': {'path': str(go), 'sha256': _sha(go)}}})
    row = rows[0]; oid = row['our_id']; w = root / oid / 'wrfout'; w.mkdir(parents=True)
    files = []
    for n in NAMES:
        p = w / n; p.write_bytes(os.urandom(32)); files.append({'path': str(p), 'bytes': 32, 'sha256': _sha(p)})
    _w(root / 'MEMBER_1_DONE.json', {'registry_run_id': 'optimize:' + oid, 'status': 'PASS', 'wrfout_files': files})
    standing = _w(tmp_path / 'STANDING.json', {
        'schema': 'alisios.screen7.standing_raw_disposition.v1', 'status': 'ISSUED', 'issued_by': 'alisios_mgr',
        'issued_utc': '2026-01-01T00:00:00Z', 'valid_until_utc': '2099-01-01T00:00:00Z',
        'members': [{'our_id': r['our_id'], 'row_id': r['row_id']} for r in rows]})
    standing.chmod(0o444)
    sref = {'path': str(standing), 'sha256': _sha(standing)}
    x = Path(row['extract_output_dir'])
    for n, o in {'EXTRACT_DONE.json': {'status': 'EXTRACT_DONE'}, 'SHA_READBACK.json': {'status': 'SHA_READBACK_PASS'},
                 'READER_PROOF.json': {'all18new_fields_pass': True}, 'POST_GATE_RESULT.json': {'state': 'PASS_METADATA_ONLY'}}.items():
        _w(x / n, o)
    _w(x / rr.INVENTORY, {'row_id': row['row_id'], 'our_id': oid, 'raw_root': str(w), 'only_files': files,
                          'total_bytes': 32 * 54, 'authorization': sref, 'no_directories_or_metadata_may_be_unlinked': True})
    cross = tmp_path / 'cross'; cross.mkdir()
    calls = []

    def compute(frames, r):
        calls.append(('export', [Path(f).name for f in frames], all(Path(f).exists() for f in frames)))
        return _fake_doc(frames, r)
    monkeypatch.setattr(rr.s7_metrics, 'compute', compute)
    return dict(tmp=tmp_path, root=root, row=row, rows=rows, files=files, cross=cross, standing=standing, sref=sref,
                x=x, adm=adm, go=go, calls=calls, wdir=w)


def _directive(world):
    x = world['x']
    tokens = [world['row']['row_id'], world['row']['our_id'], world['sref']['sha256'], _sha(x / rr.INVENTORY),
              _sha(x / 'EXTRACT_DONE.json'), _sha(x / 'READER_PROOF.json'), _sha(x / 'SHA_READBACK.json')]
    p = world['cross'] / f"{rr.PREFIX}{world['row']['our_id']}.md"
    p.write_text('# SCREEN7 release\n' + '\n'.join(f'- `{t}`' for t in tokens) + '\n'); return p


def _main(world, capsys):
    rc = rr.main(['--admission', str(world['adm']), '--admission-sha', _sha(world['adm']), '--cross', str(world['cross']),
                  '--once', '--standing', str(world['standing']), '--standing-sha', world['sref']['sha256']])
    return rc, capsys.readouterr().out


def _left(world):
    return sorted(p.name for p in world['wdir'].iterdir())


def _metrics(world):
    return world['root'] / world['row']['our_id'] / rr.METRICS


def _publish(path, doc):
    fd = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        return rr.s7_metrics.publish(fd, Path(path).name, rr.s7_metrics.encode(doc))
    finally:
        os.close(fd)


def test_release_exports_metrics_before_first_unlink(world, capsys):
    _directive(world)
    rc, out = _main(world, capsys)
    assert rc == 0 and 'RELEASED' in out and _left(world) == []
    (_, names, all_present), = world['calls']
    assert all_present and names == sorted(n for n in NAMES if n.startswith('wrfout_d03_')) and len(names) == 18
    lines = [json.loads(x) for x in (world['root'] / rr.JOURNAL).read_text().splitlines()]
    assert [ln['action'] for ln in lines] == ['unlink', 'unlink_result'] * 54 + ['member_summary']
    assert {ln['metrics_sha256'] for ln in lines} == {_sha(_metrics(world))}
    assert rr.JOURNAL == 'RAW_DELETE_JOURNAL_SCREEN7.jsonl'


def test_journal_carries_standing_contract_fields(world, capsys):
    d = _directive(world)
    rc, _out = _main(world, capsys)
    lines = [json.loads(x) for x in (world['root'] / rr.JOURNAL).read_text().splitlines()]
    inv = Path(world['row']['extract_output_dir']) / rr.INVENTORY
    for ln in lines:
        assert ln['inventory'] == {'path': str(inv), 'sha256': _sha(inv)} and ln['standing'] == world['sref']
        assert ln['go'] == {'path': str(world['go']), 'sha256': _sha(world['go'])}
        assert ln['directive'] == {'path': str(d), 'sha256': _sha(d)} and ln['utc'].endswith('Z')
    by = {f['path']: f for f in world['files']}
    for ln in (x for x in lines if x['action'] == 'unlink'):
        f = by[ln['path']]
        assert (ln['expected_sha256'], ln['expected_bytes']) == (f['sha256'], f['bytes']) == (ln['actual_sha256'], ln['actual_bytes'])
    assert all(x['ok'] for x in lines if x['action'] == 'unlink_result')
    s = lines[-1]
    assert rc == 0 and s['action'] == 'member_summary' and s['deleted_files'] == 54 and s['deleted_bytes'] == 32 * 54
    assert s['residual_files'] == [] and s['complete'] is True


def test_unlink_failure_journaled_and_stops(world, capsys, monkeypatch):
    real = os.unlink; calls = {'n': 0}

    def unlink(path, *a, **kw):
        if str(path).startswith('wrfout_'):
            calls['n'] += 1
            if calls['n'] == 3:
                raise PermissionError('denied')
        return real(path, *a, **kw)
    monkeypatch.setattr(os, 'unlink', unlink)
    _directive(world)
    _rc, out = _main(world, capsys)
    lines = [json.loads(x) for x in (world['root'] / rr.JOURNAL).read_text().splitlines()]
    res = [x for x in lines if x['action'] == 'unlink_result']
    assert [x['ok'] for x in res] == [True, True, False] and 'PermissionError' in res[-1]['error']
    s = lines[-1]
    assert s['action'] == 'member_summary' and s['deleted_files'] == 2 and len(s['residual_files']) == 52 and s['complete'] is False
    assert 'REFUSE' in out and len(_left(world)) == 52


def test_export_failure_unlinks_nothing(world, capsys, monkeypatch):
    def boom(*_a):
        raise ValueError('geometry drift')
    monkeypatch.setattr(rr.s7_metrics, 'compute', boom)
    _directive(world)
    _rc, out = _main(world, capsys)
    assert 'REFUSE' in out and 'metrics export failed' in out and len(_left(world)) == 54
    assert not (world['root'] / rr.JOURNAL).exists()


@pytest.mark.parametrize('kind', ['17_records', 'foreign_sha', 'other_member', 'no_summary'])
def test_bad_export_refused_before_unlink(world, capsys, monkeypatch, kind):
    def bad(frames, r):
        if kind == '17_records':
            return _fake_doc(frames, r, records=17)
        if kind == 'foreign_sha':
            return _fake_doc(frames, r, tamper=True)
        d = _fake_doc(frames, r)
        if kind == 'other_member':
            d['our_id'] = 'f' * 32
        else:
            d['decision_summary_18h_mean'] = {}
        return d
    monkeypatch.setattr(rr.s7_metrics, 'compute', bad)
    _directive(world)
    _rc, out = _main(world, capsys)
    assert 'REFUSE' in out and len(_left(world)) == 54 and not _metrics(world).exists()


def test_partial_raw_without_metrics_refused(world, capsys):
    (world['wdir'] / NAMES[0]).unlink()
    _directive(world)
    _rc, out = _main(world, capsys)
    assert 'REFUSE' in out and 'metrics export missing' in out and len(_left(world)) == 53
    assert world['calls'] == []


def test_existing_verified_metrics_resume_without_reexport(world, capsys):
    d03 = [world['wdir'] / n for n in NAMES if n.startswith('wrfout_d03_')]
    _publish(_metrics(world), _fake_doc(sorted(map(str, d03)), world['row']))
    (world['wdir'] / NAMES[0]).unlink()            # an earlier attempt unlinked one d01 file after exporting
    _directive(world)
    rc, out = _main(world, capsys)
    assert rc == 0 and 'RELEASED' in out and _left(world) == [] and world['calls'] == []


def test_go_identities_must_equal_bindings(world):
    doc = json.loads(world['go'].read_text()); doc['members'][0], doc['members'][1] = doc['members'][1], doc['members'][0]
    _w(world['go'], doc)
    a = json.loads(world['adm'].read_text()); a['identities']['go']['sha256'] = _sha(world['go']); _w(world['adm'], a)
    with pytest.raises(rr.Refuse, match='signed GO identities'):
        rr.contract(world['adm'], _sha(world['adm']))


def test_w1c_standing_schema_refused(world, capsys):
    s = json.loads(world['standing'].read_text()); s['schema'] = 'alisios.w1c.standing_raw_disposition.v1'
    world['standing'].chmod(0o644); _w(world['standing'], s); world['standing'].chmod(0o444)
    world['sref']['sha256'] = _sha(world['standing'])
    _directive(world)
    rc, out = _main(world, capsys)
    assert rc == 2 and 'REFUSE standing' in out and len(_left(world)) == 54


def test_F1_member_dir_swapped_after_validation_writes_nothing(world, monkeypatch):
    """Critic control (cr-w44-launchers 09:02Z F1): member dir renamed + symlinked after plan(): no metrics write."""
    directive = _directive(world); original = rr.plan
    member = world['root'] / world['row']['our_id']; held = world['tmp'] / 'held_member'

    def changed_plan(*args):
        result = original(*args)
        member.rename(held); member.symlink_to(held, target_is_directory=True)
        return result
    monkeypatch.setattr(rr, 'plan', changed_plan)
    with pytest.raises((rr.Refuse, OSError)):
        rr.release_member(world['root'], world['row'], directive, world['sref'], dry_run=False)
    assert not (held / rr.METRICS).exists() and world['calls'] == []
    assert len(list((held / 'wrfout').iterdir())) == 54 and not list(held.glob('.*tmp'))


@pytest.mark.parametrize('kind', ['symlink', 'hardlink', 'group_writable'])
def test_F1_existing_metrics_name_must_be_private_regular(world, capsys, kind):
    target = world['tmp'] / 'elsewhere.json'; target.write_text('KEEP\n')
    m = _metrics(world)
    if kind == 'symlink':
        m.symlink_to(target)
    elif kind == 'hardlink':        # a valid metrics doc, but a second name exists elsewhere -> not ours alone
        d03 = [world['wdir'] / n for n in NAMES if n.startswith('wrfout_d03_')]
        _publish(m, _fake_doc(sorted(map(str, d03)), world['row'])); os.link(m, world['tmp'] / 'second_name.json')
    else:
        d03 = [world['wdir'] / n for n in NAMES if n.startswith('wrfout_d03_')]
        _publish(m, _fake_doc(sorted(map(str, d03)), world['row'])); m.chmod(0o664)
    _directive(world)
    _rc, out = _main(world, capsys)
    assert 'REFUSE' in out and len(_left(world)) == 54 and target.read_text() == 'KEEP\n'


def test_F2_metrics_fsynced_before_first_unlink(world, capsys, monkeypatch):
    import stat as st_
    events = []; real_fsync, real_unlink = os.fsync, os.unlink

    def fsync(fd):
        events.append('fsync_dir' if st_.S_ISDIR(os.fstat(fd).st_mode) else 'fsync_file'); return real_fsync(fd)

    def unlink(path, *a, **kw):
        if str(path).startswith('wrfout_'):
            events.append('unlink_raw')
        return real_unlink(path, *a, **kw)
    monkeypatch.setattr(os, 'fsync', fsync); monkeypatch.setattr(os, 'unlink', unlink)
    _directive(world)
    rc, _out = _main(world, capsys)
    first = events.index('unlink_raw')
    assert rc == 0 and 'fsync_file' in events[:first] and 'fsync_dir' in events[:first]
    assert events[:first].index('fsync_file') < max(i for i, e in enumerate(events[:first]) if e == 'fsync_dir')


def test_publish_never_overwrites_and_leaves_no_temp(tmp_path):
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        rr.s7_metrics.publish(fd, 'm.json', b'{"a": 1}\n')
        with pytest.raises(FileExistsError):
            rr.s7_metrics.publish(fd, 'm.json', b'{"a": 2}\n')
    finally:
        os.close(fd)
    assert (tmp_path / 'm.json').read_bytes() == b'{"a": 1}\n' and sorted(p.name for p in tmp_path.iterdir()) == ['m.json']
    assert (tmp_path / 'm.json').stat().st_mode & 0o777 == 0o600


# (Independent oracle, lane evidence only — the suite guard refuses unguarded ALISIOS reads: s7_metrics.export of the
# ALISIOS compact frames of the C0530 W1c member equals ALISIOS's own SCREEN6_A_EVAL C_mean 54/54, see lane notes.)


# ------------------------------------------------------------------ wave ledger budget
def _bud():
    if not SEALED.exists():
        pytest.skip('sealed S7 dir absent')
    if str(SEALED) not in sys.path:
        sys.path.insert(0, str(SEALED))
    return importlib.import_module('s7_budget')


def _pair(i, phase='J3', seconds=10.0, **kw):
    return [{'kind': 'START', 'id': i, 'phase': phase}, dict({'kind': 'END', 'id': i, 'phase': phase, 'seconds': seconds,
                                                             'released': True}, **kw)]


def test_budget_sums_resolved_entries():
    b = _bud()
    assert b.budget(_pair('a', 'C3_prime', 100.) + _pair('b', 'J3', 50.5) + _pair('c', b.PHASE, 7.)) == 157.5


@pytest.mark.parametrize('bad', ['unresolved', 'duplicate', 'unsettled_kind', 'unknown_phase', 'nan', 'not_released'])
def test_budget_refusals(bad):
    b = _bud(); rows = _pair('a')
    if bad == 'unresolved':
        rows = rows[:1]
    elif bad == 'duplicate':
        rows = rows + rows[:1]
    elif bad == 'unsettled_kind':
        rows[1]['kind'] = 'UNSETTLED'
    elif bad == 'unknown_phase':
        rows = _pair('a', 'SCREEN8')
    elif bad == 'nan':
        rows[1]['seconds'] = float('nan')
    else:
        rows[1]['released'] = False
    with pytest.raises(Exception):
        b.budget(rows)


def test_entry_cap_never_past_wave_or_window():
    b = _bud()
    assert b.entry_cap(1962.4, 10_000) == 1500
    assert b.entry_cap(2200., 10_000) == pytest.approx(3600 - 2200 - 120)
    assert b.entry_cap(0., 900) == 780
    with pytest.raises(Exception):
        b.entry_cap(3500., 10_000)


@pytest.mark.skipif(not LEDGER.exists(), reason='wave ledger absent')
def test_real_ledger_remainder():
    b = _bud()
    rows = [json.loads(x) for x in LEDGER.read_text().splitlines()]
    used = b.budget(rows)
    assert used >= 1962.398 and b.entry_cap(used, 10_000) <= 1500


# ------------------------------------------------------------------ R10 combine (pure)
def _progs(key='k', fp='dd4b800895f7bbeadfc979ae6e77d05eaefe1f41afe9de8c1967d8dfba1ea817'):
    return {n: {'cheap_key': f'{key}-{n}', 'components': {'source_fingerprint': fp, 'x': 1}} for n in ('d01', 'd02', 'd03', 'fused/d02')}


def _part(case, programs=None, **over):
    iv = keys.iv
    plan = keys.PLAN[case]
    doc = {'schema': keys.SCHEMA_PART, 'case_id': case, 'platform': 'cpu', 'env': dict(iv.EXPECTED_CONTEXT['env']),
           'engine': {'gpuwrf_file': iv.EXPECTED_CONTEXT['engine']['gpuwrf_file']},
           'keys': {n: {'params': {'name': n, 'path': keys.PARAMS[n][0], 'sha256': keys.PARAMS[n][1]},
                        'programs': (programs or {}).get(n, _progs())} for n in plan},
           'input_dir': '/x', 'inventory_sha256': 'i', 'GPUWRF_WRF_ROOT': '/x/tables', 'seconds': 1}
    doc.update(over); return doc


need_params = pytest.mark.skipif(not all(Path(p).exists() for p, _ in keys.PARAMS.values()), reason='ALP1 parameter files absent')


@need_params
def test_combine_pass():
    r = keys.combine(_part('20260530_18z'), _part('20260219_18z'))
    assert r['status'] == 'PASS' and set(r['comparisons']) == {'0530.C3_prime_vs_0219.C3_prime', '0530.J3a_vs_0530.C3_prime',
                                                               '0530.J3b_vs_0530.C3_prime'}


@need_params
@pytest.mark.parametrize('mut', ['case_key', 'alp1_key', 'fingerprint', 'params', 'swapped', 'missing_arm', 'env'])
def test_combine_refusals(mut):
    may, feb = _part('20260530_18z'), _part('20260219_18z')
    if mut == 'case_key':
        feb['keys']['C3_prime']['programs'] = _progs('other')
    elif mut == 'alp1_key':
        may['keys']['J3b']['programs'] = _progs('other')
    elif mut == 'fingerprint':      # consistent everywhere (keys still equal): only the frozen-engine pin can refuse
        for part in (may, feb):
            for row in part['keys'].values():
                row['programs'] = _progs(fp='0' * 64)
    elif mut == 'params':
        may['keys']['J3a']['params']['sha256'] = '0' * 64
    elif mut == 'swapped':
        may, feb = feb, may
    elif mut == 'missing_arm':
        del may['keys']['J3b']
    else:
        may['env'] = dict(may['env'], GPUWRF_JAX_CACHE='1')
    with pytest.raises(Exception):
        keys.combine(may, feb)


# ------------------------------------------------------------------ launcher (sealed reused modules)
def _s7():
    if not SEALED.exists() or not BINDINGS.exists():
        pytest.skip('sealed S7 dir / bindings absent')
    if str(SEALED) not in sys.path:
        sys.path.insert(0, str(SEALED))
    s = importlib.import_module('s7')
    if not getattr(s, '_test_private_confirmation', False):
        # the real confirmation lives in the ALISIOS cross dir (suite-guarded); the pin logic runs on a private copy
        import tempfile
        p = Path(tempfile.mkdtemp(prefix='s7conf-')) / 'CONFIRMED.md'; p.write_text('SCREEN7 IDs confirmed (test copy)\n')
        s.CONFIRMATION = (str(p), _sha(p)); s._test_private_confirmation = True
    return s


def _ids(s, go=None):
    return {'bindings': {'path': str(BINDINGS), 'sha256': s.BINDINGS_SHA},
            'confirmation': {'path': s.CONFIRMATION[0], 'sha256': s.CONFIRMATION[1]}, 'go': go}


def test_member_cap_constants():
    s = _s7()
    assert s.MEMBER_CAP_S == 450 and s.dk.MEMBER_CAP == 450 and s.dk.MEMBER_SETUP_S == 0
    assert s.RELEASE_S == _bud().RELEASE_S and s.MEMBER_FLOOR_GIB == 437


def test_identities_from_confirmed_bindings():
    s = _s7()
    rows = s.identities(_ids(s), go_required=False)
    assert [r['arm_id'] for r in rows] == ['C_new', 'alp1_low', 'alp1_high']
    assert [r['parameters']['alp1'] for r in rows] == [.23, .184, .276]
    with pytest.raises(Exception, match='signed ALISIOS GO'):
        s.identities(_ids(s), go_required=True)


@pytest.mark.parametrize('mut', ['order', 'raw_dir', 'our_id'])
def test_identities_signed_go_must_match(tmp_path, mut):
    s = _s7()
    rows = [{k: r[k] for k in ('ordinal', 'row_id', 'our_id', 'registry_run_id')} for r in json.loads(BINDINGS.read_text())['rows']]
    if mut == 'order':
        rows[0], rows[1] = rows[1], rows[0]
    elif mut == 'raw_dir':
        rows[0]['raw_wrfout_dir'] = '/tmp/elsewhere/wrfout'
    else:
        rows[2]['our_id'] = 'e' * 32
    go = _w(tmp_path / 'GO.json', {'members': rows})
    with pytest.raises(Exception):
        s.identities(_ids(s, {'path': str(go), 'sha256': _sha(go)}), go_required=True)


def test_identities_signed_go_ok(tmp_path):
    s = _s7()
    rows = [{k: r[k] for k in ('ordinal', 'row_id', 'our_id', 'registry_run_id')} |
            {'raw_wrfout_dir': str(s.OUTPUT_ROOT / r['our_id'] / 'wrfout'), 'extract_output_dir': f"/x/{r['our_id']}"}
            for r in json.loads(BINDINGS.read_text())['rows']]
    go = _w(tmp_path / 'GO.json', {'members': rows})
    got = s.identities(_ids(s, {'path': str(go), 'sha256': _sha(go)}), go_required=True)
    assert [r['extract_output_dir'] for r in got] == [f"/x/{r['our_id']}" for r in rows]


def test_full_operand_seam():
    s = _s7()
    f = s.full_operand(.184)
    assert f(1e8, .33) == s.g.parameters(.184)
    with pytest.raises(Exception):
        f(1e8, .30)
    # run_member resolves c.parameters at call time on the same module object (the seam is effective)
    src = (SEALED / 'screen4.py').read_text()
    assert "c.parameters(member['parameters']['nt_c'],member['parameters']['entrainment'])" in src and s.s.c is s.c


@pytest.mark.parametrize('case', ['ok', 'deadline', 'window', 'wave', 'lease'])
def test_admit_member(case):
    s = _s7()
    mono, wall = 1000., 5000.
    args = dict(mono=mono, deadline=mono + 450, wall=wall, end=wall + 450 + 300,
                info={'prior_used_s': 1962.4, 'entry_elapsed_s': 600.}, lease_end=wall + 450)
    if case == 'deadline':
        args['deadline'] -= 1
    elif case == 'window':
        args['end'] -= 1
    elif case == 'wave':
        args['info'] = {'prior_used_s': 1962.4, 'entry_elapsed_s': 3600 - 120 - 450 - 1962.4 + 1}
    elif case == 'lease':
        args['lease_end'] -= 1
    if case == 'ok':
        s.admit_member(**args)
    else:
        with pytest.raises(Exception):
            s.admit_member(**args)


@pytest.mark.parametrize('mode,ok', [(0o444, True), (0o644, False), (0o664, False)])
def test_layout_must_be_read_only_pointer(tmp_path, mode, ok):
    s = _s7()
    p = tmp_path / 'LAYOUT.json'; p.write_text((SEALED / 'LAYOUT.json').read_text()); p.chmod(mode)
    if ok:
        s.layout_ok(p)
    else:
        with pytest.raises(Exception, match='0444'):
            s.layout_ok(p)
    p.chmod(0o644)


def test_layout_foreign_root_refused(tmp_path):
    s = _s7()
    d = json.loads((SEALED / 'LAYOUT.json').read_text()); d['output_root'] = '/tmp/elsewhere'
    p = _w(tmp_path / 'LAYOUT.json', d); p.chmod(0o444)
    with pytest.raises(Exception):
        s.layout_ok(p)
    p.chmod(0o644)


def test_sealed_layout_is_read_only():
    s = _s7()
    s.layout_ok(SEALED / 'LAYOUT.json')


def test_summary_counts_actual_unlinks_when_result_log_fails(world, capsys, monkeypatch):
    """Critic control (09:41Z): a one-time fsync failure of the result row right after a successful unlink must not
    undercount the summary (deleted files/bytes == what is actually gone)."""
    real = os.fsync; state = {'unlinks': 0, 'failed': False}
    real_unlink = os.unlink

    def unlink(path, *a, **kw):
        r = real_unlink(path, *a, **kw)
        if str(path).startswith('wrfout_'):
            state['unlinks'] += 1
        return r

    def fsync(fd):
        if state['unlinks'] == 3 and not state['failed']:
            state['failed'] = True
            raise OSError('injected result-row fsync failure')
        return real(fd)
    monkeypatch.setattr(os, 'unlink', unlink); monkeypatch.setattr(os, 'fsync', fsync)
    _directive(world)
    _rc, out = _main(world, capsys)
    gone = 54 - len(_left(world))
    s = [json.loads(x) for x in (world['root'] / rr.JOURNAL).read_text().splitlines()][-1]
    assert state['failed'] and 'REFUSE' in out and gone == 3
    assert s['action'] == 'member_summary' and s['deleted_files'] == gone and s['deleted_bytes'] == 32 * gone
    assert len(s['residual_files']) == 54 - gone
