"""W1c-1L per-member raw release tool, ALISIOS Addendum10 format (Markdown directive with SCREEN6 prefix, separate
RAW_WRFOUT_RELEASE_INVENTORY.json, gate evidence files, issued standing JSON). Private synthetic fixtures, CPU, no JAX."""
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[3] / 'scripts' / 'w1c' / 'launcher'
_spec = importlib.util.spec_from_file_location('w1c_raw_release', LAUNCHER / 'w1c_raw_release.py')
rr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(rr)

import math
from datetime import datetime, timedelta

_INIT = datetime(2026, 5, 31)
# Native history clock (ALISIOS native-clock fix 04:24Z; screen4.full_history): d01 dt 27 s writes at
# ceil(h*3600/27)*27 s (+18/+9/+0 s past the hour), d02/d03 exactly hourly.
NAMES = [f"wrfout_d0{d}_{(_INIT + timedelta(seconds=math.ceil(h * 3600 / dt) * dt)).strftime('%Y-%m-%d_%H:%M:%S')}"
         for d, dt in ((1, 27), (2, 9), (3, 3)) for h in range(1, 19)]   # 54


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _w(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True); Path(p).write_text(json.dumps(obj)); return Path(p)


@pytest.fixture
def world(tmp_path):
    root = tmp_path / 'W1C_1L_X'; root.mkdir()
    rows = []
    for i in range(1, 5):
        oid = f'{i:032x}'
        rows.append({'ordinal': i, 'our_id': oid, 'registry_run_id': 'optimize:' + oid, 'row_id': f'W1C_1L_V1.c.{i}.det0',
                     'extract_output_dir': str(tmp_path / 'extracts' / oid), 'raw_wrfout_dir': str(root / oid / 'wrfout')})
    ids = _w(tmp_path / 'GO.json', {'members': rows})
    adm = _w(tmp_path / 'ADM.json', {'schema': 'wrf_gpu2.w1c.admission.v1', 'authorized_by': 'manager', 'output_root': str(root),
                                     'identities': {'path': str(ids), 'sha256': _sha(ids)}})
    row = rows[0]; oid = row['our_id']
    w = root / oid / 'wrfout'; w.mkdir(parents=True)
    files = []
    for n in NAMES:
        p = w / n; p.write_bytes(os.urandom(32)); files.append({'path': str(p), 'bytes': 32, 'sha256': _sha(p)})
    _w(root / oid / 'MEMBER_RECEIPT.json', {})
    _w(root / 'MEMBER_1_DONE.json', {'registry_run_id': 'optimize:' + oid, 'status': 'PASS', 'wrfout_files': files})
    standing = _w(tmp_path / 'STANDING.json', {
        'schema': 'alisios.w1c.standing_raw_disposition.v1', 'status': 'ISSUED', 'issued_by': 'alisios_mgr',
        'issued_utc': '2026-01-01T00:00:00Z', 'valid_until_utc': '2099-01-01T00:00:00Z',
        'members': [{'our_id': r['our_id'], 'row_id': r['row_id']} for r in rows]})
    standing.chmod(0o444)
    sref = {'path': str(standing), 'sha256': _sha(standing)}
    x = Path(row['extract_output_dir'])
    ev = {'EXTRACT_DONE.json': {'status': 'EXTRACT_DONE'}, 'SHA_READBACK.json': {'status': 'SHA_READBACK_PASS'},
          'READER_PROOF.json': {'all18new_fields_pass': True}, 'POST_GATE_RESULT.json': {'state': 'PASS_METADATA_ONLY'}}
    for n, o in ev.items():
        _w(x / n, o)
    inv = {'row_id': row['row_id'], 'our_id': oid, 'raw_root': str(w), 'only_files': files, 'total_bytes': 32 * 54,
           'authorization': sref, 'no_directories_or_metadata_may_be_unlinked': True}
    _w(x / rr.INVENTORY, inv)
    cross = tmp_path / 'cross'; cross.mkdir()
    return dict(tmp=tmp_path, root=root, row=row, rows=rows, files=files, cross=cross, standing=standing, sref=sref, x=x,
                adm=adm, inv=inv)


def _directive(world, drop=None):
    x = world['x']
    tokens = [world['row']['row_id'], world['row']['our_id'], world['sref']['sha256'], _sha(x / rr.INVENTORY),
              _sha(x / 'EXTRACT_DONE.json'), _sha(x / 'READER_PROOF.json'), _sha(x / 'SHA_READBACK.json')]
    text = '# W1c-1L release\n' + '\n'.join(f'- `{t}`' for t in tokens if t != drop) + '\n'
    p = world['cross'] / f"{rr.PREFIX}{world['row']['our_id']}.md"; p.write_text(text); return p


def _main(world, capsys, standing=True, extra=()):
    args = ['--admission', str(world['adm']), '--admission-sha', _sha(world['adm']), '--cross', str(world['cross']), '--once', *extra]
    if standing:
        args += ['--standing', str(world['standing']), '--standing-sha', world['sref']['sha256']]
    rc = rr.main(args)
    return rc, capsys.readouterr().out


def _all_exist(world):
    return all(Path(f['path']).exists() for f in world['files'])


def test_release_deletes_exactly_the_inventory_and_journals_first(world, capsys):
    _directive(world)
    rc, out = _main(world, capsys)
    assert rc == 0 and f"RELEASED {world['row']['our_id']} 54" in out
    assert not any(Path(f['path']).exists() for f in world['files'])
    root, oid = world['root'], world['row']['our_id']
    assert (root / 'MEMBER_1_DONE.json').exists() and (root / oid / 'MEMBER_RECEIPT.json').exists() and (root / oid / 'wrfout').is_dir()
    assert all((world['x'] / n).exists() for n in ('EXTRACT_DONE.json', rr.INVENTORY))
    lines = [json.loads(x) for x in (root / rr.JOURNAL).read_text().splitlines()]
    assert len(lines) == 54 and all(x['standing_sha256'] == world['sref']['sha256'] and x['row_id'] == world['row']['row_id'] for x in lines)


def test_draft_mode_checks_but_never_deletes(world, capsys):
    _directive(world)
    rc, out = _main(world, capsys, standing=False)
    assert 'CHECKED' in out and _all_exist(world) and not (world['root'] / rr.JOURNAL).exists()


def test_old_screen6_directives_of_other_ids_ignored(world, capsys):
    (world['cross'] / f'{rr.PREFIX}{"f" * 32}.md').write_text('old SCREEN6')
    rc, out = _main(world, capsys)
    assert out.strip() == '' and _all_exist(world)


@pytest.mark.parametrize('drop', ['row_id', 'standing', 'inventory', 'EXTRACT_DONE', 'READER_PROOF', 'SHA_READBACK'])
def test_directive_must_bind_every_sha(world, capsys, drop):
    x = world['x']
    which = {'row_id': world['row']['row_id'], 'standing': world['sref']['sha256'], 'inventory': _sha(x / rr.INVENTORY),
             'EXTRACT_DONE': _sha(x / 'EXTRACT_DONE.json'), 'READER_PROOF': _sha(x / 'READER_PROOF.json'),
             'SHA_READBACK': _sha(x / 'SHA_READBACK.json')}[drop]
    _directive(world, drop=which)
    rc, out = _main(world, capsys)
    assert 'REFUSE' in out and _all_exist(world)


@pytest.mark.parametrize('name,key,bad', [('EXTRACT_DONE.json', 'status', 'PARTIAL'), ('SHA_READBACK.json', 'status', 'FAIL'),
                                          ('READER_PROOF.json', 'all18new_fields_pass', False),
                                          ('POST_GATE_RESULT.json', 'state', 'FAIL')])
def test_gate_evidence_must_pass(world, capsys, name, key, bad):
    _w(world['x'] / name, {key: bad})
    _directive(world)
    rc, out = _main(world, capsys)
    assert 'REFUSE' in out and _all_exist(world)


@pytest.mark.parametrize('mut', ['raw_root', 'authorization', 'no_dirs_flag', 'count', 'total_bytes', 'file_sha', 'foreign_path', 'our_id'])
def test_inventory_defects_refused(world, capsys, mut):
    inv = json.loads(json.dumps(world['inv']))
    if mut == 'raw_root':
        inv['raw_root'] = str(world['tmp'] / 'elsewhere')
    elif mut == 'authorization':
        inv['authorization'] = {'path': 'x', 'sha256': '0' * 64}
    elif mut == 'no_dirs_flag':
        inv['no_directories_or_metadata_may_be_unlinked'] = False
    elif mut == 'count':
        inv['only_files'] = inv['only_files'][:-1]; inv['total_bytes'] -= 32
    elif mut == 'total_bytes':
        inv['total_bytes'] += 1
    elif mut == 'file_sha':
        inv['only_files'][3]['sha256'] = '1' * 64
    elif mut == 'foreign_path':
        inv['only_files'][0]['path'] = str(world['tmp'] / 'outside' / Path(inv['only_files'][0]['path']).name)
    else:
        inv['our_id'] = f'{2:032x}'
    _w(world['x'] / rr.INVENTORY, inv)
    _directive(world)
    rc, out = _main(world, capsys)
    assert 'REFUSE' in out and _all_exist(world)


def test_member_must_be_pass(world, capsys):
    m = json.loads((world['root'] / 'MEMBER_1_DONE.json').read_text()); m['status'] = 'FAIL'
    _w(world['root'] / 'MEMBER_1_DONE.json', m)
    _directive(world)
    rc, out = _main(world, capsys)
    assert 'not PASS' in out and _all_exist(world)


def test_go_row_raw_dir_must_be_launcher_layout(world, capsys):
    go = json.loads(Path(world['tmp'] / 'GO.json').read_text()); go['members'][0]['raw_wrfout_dir'] = str(world['tmp'] / 'other')
    ids = _w(world['tmp'] / 'GO.json', go)
    adm = json.loads(world['adm'].read_text()); adm['identities']['sha256'] = _sha(ids); _w(world['adm'], adm)
    _directive(world)
    rc, out = _main(world, capsys)
    assert 'raw_wrfout_dir' in out and _all_exist(world)


@pytest.mark.parametrize('mut', ['draft', 'issuer', 'window', 'members', 'member_row_id', 'sha', 'writable'])
def test_standing_disposition_checks(world, capsys, mut):
    s = json.loads(world['standing'].read_text())
    world['standing'].chmod(0o644)
    if mut == 'draft':
        s['status'] = 'DRAFT_UNISSUED'
    elif mut == 'window':
        s['valid_until_utc'] = '2026-01-02T00:00:00Z'
    elif mut == 'members':
        s['members'] = s['members'][1:] + s['members'][:1]
    elif mut == 'member_row_id':
        s['members'][2]['row_id'] = 'W1C_1L_V1.other.det0'
    elif mut == 'issuer':
        s['issued_by'] = 'someone_else'
    if mut not in ('sha', 'writable'):
        world['standing'].write_text(json.dumps(s))
        world['sref']['sha256'] = _sha(world['standing'])
    if mut != 'writable':
        world['standing'].chmod(0o444)
    if mut == 'sha':
        world['sref']['sha256'] = '0' * 64
    _directive(world)
    rc, out = _main(world, capsys)
    assert rc == 2 and 'REFUSE standing' in out and _all_exist(world)


def test_changed_file_bytes_kept(world, capsys):
    _directive(world)
    Path(world['files'][5]['path']).write_bytes(os.urandom(32))
    rc, out = _main(world, capsys)
    assert 'MISMATCH' in out and Path(world['files'][5]['path']).exists()


@pytest.mark.parametrize('kind', ['extra_file', 'symlink_inside'])
def test_unexpected_wrfout_dir_content_refused(world, capsys, kind):
    _directive(world)
    w = Path(world['row']['raw_wrfout_dir'])
    if kind == 'extra_file':
        (w / 'notes.txt').write_text('x')
    else:
        (w / 'link').symlink_to(world['files'][0]['path'])
    rc, out = _main(world, capsys)
    assert 'REFUSE' in out and _all_exist(world)


@pytest.mark.parametrize('alias', ['wrfout_dir', 'member_dir'])
def test_symlinked_parent_never_followed(world, capsys, alias):
    o = world['tmp'] / 'outside'; o.mkdir()
    keep = o / NAMES[0]; keep.write_bytes(Path(world['files'][0]['path']).read_bytes())
    member = world['root'] / world['row']['our_id']
    if alias == 'wrfout_dir':
        shutil.rmtree(member / 'wrfout'); (member / 'wrfout').symlink_to(o)
    else:
        real = world['tmp'] / 'member_real'; shutil.move(str(member), real); member.symlink_to(real)
    _directive(world)
    rc, out = _main(world, capsys)
    assert 'REFUSE' in out and keep.exists() and not (world['root'] / rr.JOURNAL).exists()


def test_root_alias_refused_even_without_wrfout(world, capsys):
    """Fully consistent alias world (admission root, GO row, inventory, receipts all via a symlinked root) and no wrfout
    dir yet: only the root closure can refuse."""
    alias = world['tmp'] / 'alias_root'; alias.symlink_to(world['root'])
    real, al = str(world['root']), str(alias)
    shutil.rmtree(Path(world['row']['raw_wrfout_dir']))
    go = json.loads((world['tmp'] / 'GO.json').read_text())
    for r in go['members']:
        r['raw_wrfout_dir'] = r['raw_wrfout_dir'].replace(real, al)
    ids = _w(world['tmp'] / 'GO.json', go)
    adm = json.loads(world['adm'].read_text()); adm['output_root'] = al; adm['identities']['sha256'] = _sha(ids); _w(world['adm'], adm)
    m = json.loads((world['root'] / 'MEMBER_1_DONE.json').read_text())
    for f in m['wrfout_files']:
        f['path'] = f['path'].replace(real, al)
    _w(world['root'] / 'MEMBER_1_DONE.json', m)
    inv = json.loads(json.dumps(world['inv'])); inv['raw_root'] = inv['raw_root'].replace(real, al)
    for f in inv['only_files']:
        f['path'] = f['path'].replace(real, al)
    _w(world['x'] / rr.INVENTORY, inv)
    _directive(world)
    rc, out = _main(world, capsys)
    assert 'REFUSE' in out and 'RELEASED' not in out


def test_inventory_and_receipt_with_same_foreign_path_refused(world, capsys):
    """Receipt and inventory agree on a path outside <root>/<id>/wrfout: only the lexical path rule refuses."""
    o = world['tmp'] / 'outside'; o.mkdir()
    keep = o / NAMES[0]; keep.write_bytes(Path(world['files'][0]['path']).read_bytes())
    m = json.loads((world['root'] / 'MEMBER_1_DONE.json').read_text()); m['wrfout_files'][0]['path'] = str(keep)
    _w(world['root'] / 'MEMBER_1_DONE.json', m)
    inv = json.loads(json.dumps(world['inv'])); inv['only_files'][0]['path'] = str(keep); _w(world['x'] / rr.INVENTORY, inv)
    _directive(world)
    rc, out = _main(world, capsys)
    assert 'REFUSE' in out and keep.exists() and _all_exist(world)


def test_journal_symlink_refused(world, capsys):
    _directive(world)
    target = world['tmp'] / 'elsewhere.jsonl'; target.write_text(''); target.chmod(0o600)   # only O_NOFOLLOW refuses
    (world['root'] / rr.JOURNAL).symlink_to(target)
    rc, out = _main(world, capsys)
    assert 'REFUSE' in out and target.read_text() == '' and _all_exist(world)


def test_swap_between_hash_and_unlink_kept(world, capsys, monkeypatch):
    _directive(world)
    real_fsync = rr.os.fsync
    victim = Path(world['files'][0]['path']); swapped = []

    def fsync_then_swap(fd):
        real_fsync(fd)
        if victim.exists() and not swapped:
            tmp = victim.with_name('swap.tmp'); tmp.write_bytes(victim.read_bytes()); os.replace(tmp, victim); swapped.append(1)
    monkeypatch.setattr(rr.os, 'fsync', fsync_then_swap)
    rc, out = _main(world, capsys)
    assert 'inode changed' in out and victim.exists()


def test_journal_written_before_unlink(world, capsys, monkeypatch):
    _directive(world)
    seen = []

    def fake_unlink(p, **kw):
        seen.append((Path(p).name, (world['root'] / rr.JOURNAL).read_text().count('\n')))
        raise OSError('simulated crash')
    monkeypatch.setattr(rr.os, 'unlink', fake_unlink)
    rc, out = _main(world, capsys)
    assert 'REFUSE' in out and seen == [(sorted(NAMES)[0], 1)] and _all_exist(world)


def test_foreign_admission_refused(world):
    with pytest.raises(rr.Refuse):
        rr.main(['--admission', str(world['adm']), '--admission-sha', '0' * 64, '--once'])


def test_native_clock_names_match_real_screen4_receipt_and_release():
    """d01 names are NOT on the hour (+18/+9/+0 s); the fixture follows the native rule and equals the real paid 0530 C
    receipt (metadata only) when available; the whole release path above already runs on these names."""
    assert sorted({n[-2:] for n in NAMES if n.startswith('wrfout_d01')}) == ['00', '09', '18']
    assert all(n.endswith(':00:00') for n in NAMES if not n.startswith('wrfout_d01'))
    assert all(rr.WRFOUT.match(n) for n in NAMES)
    real = Path('<DATA_ROOT>/wrf_gpu2/scratch/v45-waveops/W1A_SCREEN4_20261030/MEMBER_2_DONE.json')
    if not real.exists():
        pytest.skip('paid SCREEN4 receipt not on this host')
    names = sorted(Path(f['path']).name for f in json.loads(real.read_text())['wrfout_files'])
    assert names == sorted(NAMES)


# --------------------------------------------------------------- critic F2 residuals (04:26Z): TOCTOU parent swap, journal hardlink
def _outside_copies(world):
    o = world['tmp'] / 'outside_member' / 'wrfout'; o.mkdir(parents=True)
    for f in world['files']:
        (o / Path(f['path']).name).write_bytes(Path(f['path']).read_bytes())
    return o.parent


@pytest.mark.parametrize('swap', ['symlink', 'fresh_dir'])
def test_F2a_parent_swapped_after_validation_never_followed(world, capsys, monkeypatch, swap):
    """Member dir replaced right after validation (critic F2a): by a symlink to same-byte copies outside, or by a new
    real dir with the same bytes. Neither the outside copies nor the originals are deleted."""
    _directive(world)
    outside = _outside_copies(world)
    member = world['root'] / world['row']['our_id']
    moved = world['tmp'] / 'moved_member'
    real_plan = rr.plan

    def plan_then_swap(*a, **k):
        out = real_plan(*a, **k)
        shutil.move(str(member), moved)
        if swap == 'symlink':
            member.symlink_to(outside)
        else:
            shutil.copytree(outside, member)
        return out
    monkeypatch.setattr(rr, 'plan', plan_then_swap)
    rc, out = _main(world, capsys)
    assert 'REFUSE' in out and 'RELEASED' not in out
    assert all((outside / 'wrfout' / Path(f['path']).name).exists() for f in world['files'])
    assert all((moved / 'wrfout' / Path(f['path']).name).exists() for f in world['files'])
    if swap == 'fresh_dir':
        assert all((member / 'wrfout' / Path(f['path']).name).exists() for f in world['files'])


def test_F2a_path_rewalked_before_each_unlink(world, capsys, monkeypatch):
    """After the first unlink the member path is re-pointed through a symlink to the SAME directory: the anchored dirfd
    is unchanged but the path walk no longer reaches it without following a symlink -> stop, remaining files kept."""
    _directive(world)
    member = world['root'] / world['row']['our_id']
    hidden = world['tmp'] / 'hidden_member'
    real_unlink = rr.os.unlink
    calls = []

    def unlink_then_repoint(name, **kw):
        real_unlink(name, **kw); calls.append(name)
        if len(calls) == 1:
            os.rename(member, hidden); member.symlink_to(hidden)
    monkeypatch.setattr(rr.os, 'unlink', unlink_then_repoint)
    rc, out = _main(world, capsys)
    assert 'REFUSE' in out and len(calls) == 1
    assert sum((hidden / 'wrfout' / Path(f['path']).name).exists() for f in world['files']) == 53


@pytest.mark.parametrize('kind', ['hardlink_to_protected', 'group_writable'])
def test_F2b_journal_must_be_private_single_link(world, capsys, kind):
    _directive(world)
    j = world['root'] / rr.JOURNAL
    if kind == 'hardlink_to_protected':
        protected = world['tmp'] / 'protected_metadata.json'; protected.write_text('{"keep": true}\n'); protected.chmod(0o600)
        os.link(protected, j)   # same owner, 0600: only the single-link rule refuses
    else:
        j.write_text(''); j.chmod(0o660)
    rc, out = _main(world, capsys)
    assert 'REFUSE' in out and 'journal not a private single-link' in out and _all_exist(world)
    if kind == 'hardlink_to_protected':
        assert protected.read_text() == '{"keep": true}\n'


def test_F2a_anchored_walk_unit(tmp_path):
    """anchored(): O_NOFOLLOW component walk; refuses a symlink component and a changed inode (each layer on its own)."""
    d = tmp_path / 'a' / 'b'; d.mkdir(parents=True)
    chain = rr.no_symlink_path(d)
    fd = rr.anchored(d, chain); st = os.fstat(fd); os.close(fd)
    assert (st.st_dev, st.st_ino) == chain[-1]
    shutil.move(str(tmp_path / 'a'), tmp_path / 'a_real'); (tmp_path / 'a').symlink_to(tmp_path / 'a_real')
    with pytest.raises(OSError):
        rr.anchored(d, chain)                          # ELOOP: intermediate symlink not followed
    (tmp_path / 'a').unlink(); (tmp_path / 'a' / 'b').mkdir(parents=True)
    with pytest.raises(rr.Refuse):
        rr.anchored(d, chain)                          # same path, different inode


def test_F2a_wrfout_handle_comes_from_anchored_walk(world, capsys, monkeypatch):
    """The deleting dirfd is opened by the anchored component walk of the validated chain, before any unlink."""
    _directive(world)
    order = []
    real_anchored, real_unlink = rr.anchored, rr.os.unlink

    def spy_anchored(path, chain):
        order.append(('anchored', str(path), len(chain))); return real_anchored(path, chain)

    def spy_unlink(name, **kw):
        order.append(('unlink', name)); return real_unlink(name, **kw)
    monkeypatch.setattr(rr, 'anchored', spy_anchored); monkeypatch.setattr(rr.os, 'unlink', spy_unlink)
    rc, out = _main(world, capsys)
    assert 'RELEASED' in out
    wdir = world['row']['raw_wrfout_dir']
    first_unlink = next(i for i, e in enumerate(order) if e[0] == 'unlink')
    # the dirfd open AND the per-unlink re-walk are both anchored walks of wrfout before the first unlink
    assert order[:first_unlink].count(('anchored', wdir, len(Path(wdir).parts))) == 2


# --------------------------------------------------------------- issued GO: identity-only members + SHA-pinned go_draft
_rows_spec = importlib.util.spec_from_file_location('w1c_rows', LAUNCHER / 'w1c_rows.py')
w1c_rows = importlib.util.module_from_spec(_rows_spec); _rows_spec.loader.exec_module(w1c_rows)


def _load_pinned(ref):
    data = Path(ref['path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != ref['sha256']:
        raise ValueError('pinned SHA mismatch')
    return json.loads(data)


def _issued_go(world, mutate=None):
    full = json.loads((world['tmp'] / 'GO.json').read_text())
    draft = _w(world['tmp'] / 'GO_DRAFT.json', full)
    members = [{k: r[k] for k in w1c_rows.IDENTITY_FIELDS} for r in full['members']]
    go = {'members': members, 'go_draft': {'path': str(draft), 'sha256': _sha(draft)}}
    if mutate:
        mutate(go)
    return go


def test_signed_rows_from_pinned_draft(world):
    rows = w1c_rows.signed_rows(_issued_go(world), _load_pinned)
    assert [r['raw_wrfout_dir'] for r in rows] == [r['raw_wrfout_dir'] for r in world['rows']]


@pytest.mark.parametrize('mut', ['swap_order', 'foreign_id', 'draft_sha', 'no_draft'])
def test_signed_rows_refusals(world, mut):
    def m(go):
        if mut == 'swap_order':
            go['members'][0], go['members'][1] = go['members'][1], go['members'][0]
        elif mut == 'foreign_id':
            go['members'][2]['our_id'] = 'e' * 32
        elif mut == 'draft_sha':
            go['go_draft']['sha256'] = '0' * 64
        else:
            go.pop('go_draft')
    with pytest.raises(ValueError):
        w1c_rows.signed_rows(_issued_go(world, m), _load_pinned)


def test_release_tool_with_identity_only_issued_go(world, capsys):
    """The real issued-GO shape: the tool takes raw_wrfout_dir/extract_output_dir from the pinned draft and releases."""
    go = _w(world['tmp'] / 'GO_ISSUED.json', _issued_go(world))
    adm = json.loads(world['adm'].read_text()); adm['identities'] = {'path': str(go), 'sha256': _sha(go)}; _w(world['adm'], adm)
    _directive(world)
    rc, out = _main(world, capsys)
    assert f"RELEASED {world['row']['our_id']} 54" in out


def test_release_tool_refuses_issued_go_draft_mismatch(world):
    go = _w(world['tmp'] / 'GO_ISSUED.json', _issued_go(world, lambda g: g['members'][3].update(row_id='x')))
    adm = json.loads(world['adm'].read_text()); adm['identities'] = {'path': str(go), 'sha256': _sha(go)}; _w(world['adm'], adm)
    with pytest.raises(rr.Refuse):
        rr.main(['--admission', str(world['adm']), '--admission-sha', _sha(world['adm']), '--cross', str(world['cross']), '--once'])
