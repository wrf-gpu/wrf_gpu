#!/usr/bin/env python3
"""SCREEN7 per-member raw wrfout release = the W1c-1L tool (w1c_raw_release.py, sealed fcea4969) with three changes:
(1) contract = the pinned SCREEN7 manager admission: rows = the signed GO rows, identity-equal to the confirmed
    bindings da1f13b8 (3 members); (2) journal RAW_DELETE_JOURNAL_SCREEN7.jsonl, standing schema
    alisios.screen7.standing_raw_disposition.v1; (3) NEW (ALISIOS 2026-10-31T06:58Z): BEFORE the first unlink of a
    member the decision metrics (s7_metrics.export: north LWP/CWP, cloud fraction at CWP >= 10 g/m2, T2/wind means,
    ALISIOS definitions) are exported from the 18 raw d03 frames to <output_root>/<our_id>/S7_METRICS_BEFORE_UNLINK.json
    and read back (18 records, source SHAs == release inventory SHAs); no verified export -> no unlink; every journal
    line carries the metrics SHA. Original W1c description follows.

W1c-1L per-member raw wrfout release, run by the manager as a watcher (SCREEN6 pattern, ALISIOS Addendum10 format).

Authority: the ISSUED ALISIOS standing disposition W1c-1L (JSON `alisios.w1c.standing_raw_disposition.v1`, 0444, SHA pinned
on the command line and in the signed GO; its members == the four rows of the pinned W1c admission; inside its window).
Directive (per member, ALISIOS): `cross/TO_WRF_GPU2_W1A_SCREEN6_RAW_DISPOSITION_<our_id>.md` (SCREEN6 prefix by Addendum10,
content W1c). It must contain the row_id, our_id, standing SHA and the SHAs of EXTRACT_DONE.json, READER_PROOF.json,
SHA_READBACK.json and RAW_WRFOUT_RELEASE_INVENTORY.json of that member's extract_output_dir (row of the signed GO).
Evidence files there must read EXTRACT_DONE / all18new_fields_pass / SHA_READBACK_PASS and POST_GATE_RESULT PASS_METADATA_ONLY.
Inventory {row_id, our_id, raw_root, only_files[{path,bytes,sha256}], total_bytes, authorization, no_directories_...}:
raw_root == row raw_wrfout_dir == <output_root>/<our_id>/wrfout (lexical), authorization == pinned standing ref,
only_files == the PASS member receipt's wrfout_files (54, exact path/bytes/sha256).
Deleted: ONLY those regular files, where the root and every directory down to wrfout are real directories (no symlink
component), through a dirfd of wrfout opened by an O_NOFOLLOW component walk from '/' whose inode chain equals the
validated chain (critic F2a); hash from an O_NOFOLLOW fd; file inode re-checked and the path re-walked to the same anchored
dir right before unlinkat; size + SHA-256 recomputed before each unlink; a journal line fsync'd to <output_root>/RAW_DELETE_JOURNAL_W1C_1L.jsonl
(opened O_NOFOLLOW via the anchored root; must be ours, 0600, single-link: critic F2b) BEFORE each unlink. Never: directories, the root, receipts, logs, inputs, namelists,
ledgers, compacts, extracts, the journal. Without --standing: draft mode, checks everything except the standing
authority and reports only (never deletes).
Usage: w1c_raw_release.py --admission A.json --admission-sha S [--standing PATH --standing-sha S] [--once]
"""
import argparse, glob, hashlib, json, os, re, stat, sys, time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w1c_rows  # noqa: E402
import s7_metrics  # noqa: E402

CROSS = Path('<DATA_ROOT>/alisios/state/manager/cross')
PREFIX = 'TO_WRF_GPU2_W1A_SCREEN6_RAW_DISPOSITION_'     # Addendum10: SCREEN6 watch name, W1c content
JOURNAL = 'RAW_DELETE_JOURNAL_SCREEN7.jsonl'
METRICS = 'S7_METRICS_BEFORE_UNLINK.json'
BINDINGS_SHA = 'da1f13b8a788ac373df501c76dc946b34f950f76abd0ae811fca972ac488b701'
INVENTORY = 'RAW_WRFOUT_RELEASE_INVENTORY.json'
WRFOUT = re.compile(r'^wrfout_d0[1-3]_\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}$')


class Refuse(Exception):
    pass


def need(ok, msg):
    if not ok:
        raise Refuse(msg)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 24), b''):
            h.update(c)
    return h.hexdigest()


def pinned(path, digest, what):
    data = Path(path).read_bytes()
    need(hashlib.sha256(data).hexdigest() == digest, f'{what} bytes differ from pinned SHA')
    return json.loads(data)


def _utc(t):
    return datetime.fromisoformat(str(t).replace('Z', '+00:00')).astimezone(timezone.utc)


def contract(admission, admission_sha):
    """(output_root, {our_id: row}) from the pinned SCREEN7 admission: signed GO rows, identity-equal to the bindings."""
    a = pinned(admission, admission_sha, 'SCREEN7 admission')
    need(a.get('schema') == 'wrf_gpu2.screen7.admission.v1' and a.get('authorized_by') == 'manager', 'not a SCREEN7 manager admission')
    ids = a['identities']
    need(ids['bindings']['sha256'] == BINDINGS_SHA and ids.get('go'), 'SCREEN7 bindings pin / signed GO missing')
    bound = pinned(ids['bindings']['path'], BINDINGS_SHA, 'SCREEN7 bindings')['rows']
    go = pinned(ids['go']['path'], ids['go']['sha256'], 'SCREEN7 signed GO')
    try:
        members = w1c_rows.signed_rows(go, lambda r: pinned(r['path'], r['sha256'], 'GO draft pinned by the signed GO'))
    except ValueError as exc:
        raise Refuse(str(exc)) from None
    need([tuple(m.get(k) for k in w1c_rows.IDENTITY_FIELDS) for m in members] ==
         [tuple(b.get(k) for k in w1c_rows.IDENTITY_FIELDS) for b in bound], 'signed GO identities != confirmed bindings')
    rows = {r['our_id']: dict(r, arm_id=b['arm_id'], case_id=b['case_id'], go_ref={'path': ids['go']['path'], 'sha256': ids['go']['sha256']})
            for r, b in zip(members, bound)}
    need(len(rows) == 3 and all(re.fullmatch('[0-9a-f]{32}', k) for k in rows), 'SCREEN7 identities malformed')
    return Path(a['output_root']), rows


def standing_ok(path, digest, rows, now=None):
    s = pinned(path, digest, 'standing disposition')
    need((os.stat(path).st_mode & 0o222) == 0, 'standing disposition not read-only (0444)')
    need(s.get('schema') == 'alisios.screen7.standing_raw_disposition.v1' and s.get('status') == 'ISSUED' and
         s.get('issued_by') == 'alisios_mgr', 'standing disposition not ISSUED by alisios_mgr')
    try:
        lo, hi = _utc(s['issued_utc']), _utc(s['valid_until_utc'])
    except (KeyError, TypeError, ValueError):
        raise Refuse('standing disposition window missing') from None
    now = now or datetime.now(timezone.utc)
    need(lo <= now <= hi, 'standing disposition outside its window')
    members = s.get('members') or []
    need([m.get('our_id') for m in members] == sorted(rows, key=lambda k: rows[k]['ordinal']) and
         all(rows[m['our_id']]['row_id'] == m.get('row_id') for m in members), 'standing members != SCREEN7 admission rows')
    return {'path': str(path), 'sha256': digest}


def no_symlink_path(path):
    """Absolute, normalised, and no component (root .. leaf) is a symlink; every component an existing directory.
    Returns the (dev, ino) chain of '/', each component ... leaf (lstat at validation time)."""
    path = Path(path)
    need(path.is_absolute() and os.path.normpath(str(path)) == str(path), f'path not absolute/normalised: {path}')
    st = os.lstat('/'); chain = [(st.st_dev, st.st_ino)]
    cur = Path('/')
    for part in path.parts[1:]:
        cur = cur / part
        st = os.lstat(cur)
        need(not stat.S_ISLNK(st.st_mode) and stat.S_ISDIR(st.st_mode), f'symlink or non-directory component: {cur}')
        chain.append((st.st_dev, st.st_ino))
    return chain


def anchored(path, chain):
    """Open `path` by walking every component with O_NOFOLLOW|O_DIRECTORY from '/', each opened directory's (dev, ino)
    equal to the validated chain (critic F2a: no intermediate symlink is ever followed, no swapped inode trusted).
    Returns an fd of the leaf directory (caller closes)."""
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        st = os.fstat(fd); need((st.st_dev, st.st_ino) == chain[0], 'root of the filesystem changed')
        parts = Path(path).parts[1:]
        need(len(parts) + 1 == len(chain), 'anchor chain length mismatch')
        for part, want in zip(parts, chain[1:]):
            nfd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = nfd
            st = os.fstat(fd)
            need((st.st_dev, st.st_ino) == want, f'directory changed since validation: {part}')
        out, fd = fd, -1
        return out
    finally:
        if fd >= 0:
            os.close(fd)


def receipt(root, our_id):
    for f in sorted(root.glob('MEMBER_*_DONE.json')):
        need(not f.is_symlink(), f'symlinked receipt {f}')
        m = json.loads(f.read_text())
        if m.get('registry_run_id') == 'optimize:' + our_id:
            return m
    return None


def _evidence(xdir, name, key, value):
    p = xdir / name
    need(p.is_file() and not p.is_symlink(), f'evidence missing: {name}')
    d = json.loads(p.read_text())
    need(d.get(key) == value, f'evidence {name}: {key} != {value!r}')
    return sha_file(p)


def plan(root, row, directive_text, standing_ref):
    """Validate one member release; return (wrfout_dir, {name: (sha256, bytes)}) to delete (possibly empty).
    standing_ref None = draft mode (standing binding not checked; caller never deletes)."""
    our_id = row['our_id']
    root = Path(root)
    root_chain = no_symlink_path(root)
    xdir = Path(row['extract_output_dir'])
    shas = {'EXTRACT_DONE': _evidence(xdir, 'EXTRACT_DONE.json', 'status', 'EXTRACT_DONE'),
            'SHA_READBACK': _evidence(xdir, 'SHA_READBACK.json', 'status', 'SHA_READBACK_PASS'),
            'READER_PROOF': _evidence(xdir, 'READER_PROOF.json', 'all18new_fields_pass', True)}
    _evidence(xdir, 'POST_GATE_RESULT.json', 'state', 'PASS_METADATA_ONLY')
    inv_path = xdir / INVENTORY
    need(inv_path.is_file() and not inv_path.is_symlink(), 'release inventory missing')
    inv_sha = sha_file(inv_path); inv = json.loads(inv_path.read_text())
    tokens = [row['row_id'], our_id, inv_sha, *shas.values()] + ([standing_ref['sha256']] if standing_ref else [])
    for token in tokens:
        need(token in directive_text, f'directive does not bind {token[:24]}')
    wdir = root / our_id / 'wrfout'
    need(row.get('raw_wrfout_dir') == str(wdir), 'GO row raw_wrfout_dir != <output_root>/<our_id>/wrfout')
    need(inv.get('row_id') == row['row_id'] and inv.get('our_id') == our_id and inv.get('raw_root') == str(wdir) and
         inv.get('no_directories_or_metadata_may_be_unlinked') is True, 'inventory header (row/our_id/raw_root) mismatch')
    if standing_ref:
        need(inv.get('authorization') == standing_ref, 'inventory authorization != pinned standing disposition')
    only = inv.get('only_files') or []
    need(len(only) == 54 and inv.get('total_bytes') == sum(f['bytes'] for f in only), 'inventory count/total_bytes')
    m = receipt(root, our_id)
    need(m is not None, 'no member receipt yet')
    need(m.get('status') == 'PASS', 'member not PASS (no release of FAIL members)')
    key = lambda fs: sorted((f['path'], f['bytes'], f['sha256']) for f in fs)
    need(key(only) == key(m.get('wrfout_files', [])), 'inventory only_files != member receipt wrfout_files')
    rec = {}
    for f in only:
        p = Path(f['path'])
        need(str(p) == str(wdir / p.name) and WRFOUT.match(p.name), f'inventory path not lexically <root>/<id>/wrfout/<wrfout>: {p}')
        rec[p.name] = (f['sha256'], f['bytes'])
    if not os.path.lexists(wdir):
        return wdir, {}, root_chain, None
    wchain = no_symlink_path(wdir)                 # <root>/<our_id> and wrfout are real directories (critic F2)
    need(wchain[:len(root_chain)] == root_chain, 'root changed during validation')
    todo = {}
    for name in sorted(os.listdir(wdir)):
        st = os.lstat(wdir / name)
        need(stat.S_ISREG(st.st_mode), f'refuse non-regular {wdir / name}')
        need(name in rec, f'unexpected file in wrfout dir: {name}')
        todo[name] = rec[name]
    return wdir, todo, root_chain, wchain


def _open_journal(root, root_chain):
    """Journal via the anchored root dirfd, O_NOFOLLOW; must be a regular file owned by us, mode 0600, single link
    (critic F2b: a pre-existing hardlink to foreign/protected metadata is refused, nothing appended)."""
    rfd = anchored(root, root_chain)
    try:
        fd = os.open(JOURNAL, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600, dir_fd=rfd)
    finally:
        os.close(rfd)
    st = os.fstat(fd)
    ok = stat.S_ISREG(st.st_mode) and st.st_uid == os.getuid() and st.st_nlink == 1 and (st.st_mode & 0o077) == 0
    if not ok:
        os.close(fd)
        raise Refuse('journal not a private single-link regular file owned by this user')
    return os.fdopen(fd, 'a')


def _metrics_check(doc, row, d03):
    """The export must be this member's, complete (18 hourly d03 records) and of exactly the inventoried d03 bytes."""
    need(doc.get('schema') == 'wrf_gpu2.screen7.metrics_before_unlink.v1' and doc.get('our_id') == row['our_id'] and
         doc.get('row_id') == row['row_id'] and doc.get('hours') == 18 and len(doc.get('records', [])) == 18,
         'metrics export foreign/incomplete')
    need([(r['source']['path'], r['source']['sha256']) for r in doc['records']] == d03, 'metrics sources != inventoried d03 frames')
    need(doc.get('decision_summary_18h_mean') and doc.get('truth_reads') == 0, 'metrics decision summary missing')


def metrics_before_unlink(root, row, wdir, todo, wchain):
    """ALISIOS 06:58Z: export + durably publish + read back the decision metrics BEFORE the first unlink of this member.
    Critic F1/F2: every metrics read/write goes through a dir fd of <root>/<our_id> obtained by the anchored O_NOFOLLOW walk
    whose inode chain equals the validated one (a swapped/symlinked member dir refuses before any write); publication is
    exclusive temp + link (no overwrite) + file and directory fsync (s7_metrics.publish). If an earlier attempt already
    published (and possibly unlinked part of the member), that file must verify the same way and is fsynced again."""
    rec = _inventory_rec(root, row)
    need(all(rec.get(n) == v for n, v in todo.items()), 'release inventory changed since validation')
    d03 = sorted((str(wdir / n), sha) for n, (sha, _b) in rec.items() if n.startswith('wrfout_d03_'))
    need(len(d03) == 18, 'inventory lacks the 18 d03 frames')
    mdir = root / row['our_id']
    need(wchain is not None and str(mdir) == str(wdir.parent), 'member directory chain missing')
    mfd = anchored(mdir, wchain[:-1])
    try:
        try:
            os.stat(METRICS, dir_fd=mfd, follow_symlinks=False); exists = True
        except FileNotFoundError:
            exists = False
        if not exists:
            need(len(todo) == 54, 'metrics export missing although raw was already partly unlinked')
            try:
                doc = s7_metrics.compute([p for p, _s in d03], row)
                _metrics_check(doc, row, d03)
                digest = s7_metrics.publish(mfd, METRICS, s7_metrics.encode(doc))
            except Refuse:
                raise
            except Exception as exc:
                raise Refuse(f'metrics export failed, nothing unlinked: {type(exc).__name__}: {exc}') from None
        try:
            digest, data = s7_metrics.readback(mfd, METRICS)
        except (OSError, ValueError) as exc:
            raise Refuse(f'metrics file not verifiable: {exc}') from None
        _metrics_check(json.loads(data), row, d03)
    finally:
        os.close(mfd)
    return digest


def _inventory_rec(root, row):
    inv = json.loads((Path(row['extract_output_dir']) / INVENTORY).read_text())
    return {Path(f['path']).name: (f['sha256'], f['bytes']) for f in inv['only_files']}


def release_member(root, row, disp_path, standing_ref, *, dry_run):
    text = Path(disp_path).read_text()
    wdir, todo, root_chain, wchain = plan(root, row, text, None if dry_run else standing_ref)
    dsha = sha_file(disp_path)
    if dry_run or not todo:
        if dry_run:
            print(f"DRY-RUN {row['our_id']}: would delete {len(todo)} files")
        return 0
    inv_path = Path(row['extract_output_dir']) / INVENTORY
    inv_sha = sha_file(inv_path)
    need(inv_sha in text, 'release inventory changed since the directive was validated')
    metrics_sha = metrics_before_unlink(Path(root), row, wdir, todo, wchain)
    # standing journal contract (ALISIOS SCREEN7 standing ca231994 journal_required): every row carries the inventory,
    # standing, GO and directive path+SHA and the metrics SHA; an 'unlink' intent row (expected + actual size/SHA) is
    # fsynced BEFORE each unlink, an 'unlink_result' row after it (ok / error), and one 'member_summary' row at the end
    # (deleted files/bytes, residual files in the wrfout dir).
    ctx = {'our_id': row['our_id'], 'row_id': row['row_id'], 'inventory': {'path': str(inv_path), 'sha256': inv_sha},
           'standing': dict(standing_ref), 'go': row.get('go_ref'), 'directive': {'path': str(disp_path), 'sha256': dsha},
           'authority': Path(disp_path).name, 'authority_sha256': dsha, 'standing_sha256': standing_ref['sha256'],
           'metrics_sha256': metrics_sha}
    utc = lambda: time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

    def log(jf, rec):
        jf.write(json.dumps(dict(ctx, utc=utc(), **rec)) + '\n'); jf.flush(); os.fsync(jf.fileno())
    n = 0; deleted_bytes = 0
    dfd = anchored(wdir, wchain)
    try:
        dst = os.fstat(dfd)
        with _open_journal(Path(root), root_chain) as jf:
          try:
            for name, (want_sha, want_bytes) in todo.items():
                ffd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=dfd)
                try:
                    fst = os.fstat(ffd)
                    need(stat.S_ISREG(fst.st_mode), f'non-regular {name}')
                    h = hashlib.sha256()
                    with os.fdopen(os.dup(ffd), 'rb') as fh:
                        for c in iter(lambda: fh.read(1 << 24), b''):
                            h.update(c)
                    got = h.hexdigest()
                finally:
                    os.close(ffd)
                need(got == want_sha and fst.st_size == want_bytes, f'MISMATCH, kept {wdir / name}')
                log(jf, {'action': 'unlink', 'path': str(wdir / name), 'expected_sha256': want_sha, 'expected_bytes': want_bytes,
                         'sha256': got, 'bytes': fst.st_size, 'actual_sha256': got, 'actual_bytes': fst.st_size, 'inode': fst.st_ino})
                now = os.stat(name, dir_fd=dfd, follow_symlinks=False)
                need((now.st_dev, now.st_ino) == (fst.st_dev, fst.st_ino), f'inode changed between hash and unlink, kept {wdir / name}')
                # the path root -> member -> wrfout must still walk (O_NOFOLLOW, same inode chain) to the anchored dir
                chk = anchored(wdir, wchain)
                try:
                    cst = os.fstat(chk)
                finally:
                    os.close(chk)
                need((cst.st_dev, cst.st_ino) == (dst.st_dev, dst.st_ino), f'wrfout path no longer reaches the anchored dir, kept {wdir / name}')
                try:
                    os.unlink(name, dir_fd=dfd)
                except OSError as exc:
                    log(jf, {'action': 'unlink_result', 'path': str(wdir / name), 'ok': False, 'error': f'{type(exc).__name__}: {exc}'})
                    raise
                n += 1; deleted_bytes += fst.st_size   # counted at the successful unlink, before any journal I/O can fail
                log(jf, {'action': 'unlink_result', 'path': str(wdir / name), 'ok': True})
          finally:
            residual = sorted(os.listdir(dfd))
            log(jf, {'action': 'member_summary', 'deleted_files': n, 'deleted_bytes': deleted_bytes,
                     'residual_files': residual, 'complete': n == len(todo) and not residual})
    finally:
        os.close(dfd)
    return n


def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--admission', required=True); p.add_argument('--admission-sha', required=True)
    p.add_argument('--standing'); p.add_argument('--standing-sha'); p.add_argument('--once', action='store_true')
    p.add_argument('--cross', default=str(CROSS))
    a = p.parse_args(argv)
    root, rows = contract(a.admission, a.admission_sha)
    dry = not (a.standing and a.standing_sha)
    done = set()
    while True:
        standing_ref = None
        if not dry:
            try:
                standing_ref = standing_ok(a.standing, a.standing_sha, rows)
            except (Refuse, OSError) as exc:
                print(time.strftime('%H:%M:%SZ', time.gmtime()), 'REFUSE standing', exc, flush=True)
                if a.once:
                    return 2
                time.sleep(20); continue
        for disp in sorted(glob.glob(os.path.join(a.cross, PREFIX + '*.md'))):
            our_id = Path(disp).name[len(PREFIX):-3]
            if our_id in done or our_id not in rows:      # old SCREEN6 directives (other IDs) are never touched
                continue
            try:
                n = release_member(root, rows[our_id], disp, standing_ref, dry_run=dry)
                print(time.strftime('%H:%M:%SZ', time.gmtime()), 'RELEASED' if not dry else 'CHECKED', our_id, n, flush=True)
                if not dry:
                    done.add(our_id)
            except (Refuse, OSError) as exc:
                print(time.strftime('%H:%M:%SZ', time.gmtime()), 'REFUSE', our_id, exc, flush=True)
        if a.once:
            return 0
        time.sleep(20)


if __name__ == '__main__':
    sys.exit(main())
