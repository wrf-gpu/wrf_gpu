"""W1c IC-variant admission: refusal rules R1-R10 on small synthetic wrfinput-like files (CPU, no JAX).

The synthetic base is built WRF-balanced with icvariant._rebalance, so the balance tests here check the
wiring (increment, closures, refusals); the independent WRF calibration of the relations is the real
real.exe files (EOS rel <= 3.4e-7, THM <= 4.4e-5 K, lane notes v46-icvariant).
"""
import hashlib
import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip('netCDF4')

_SPEC = importlib.util.spec_from_file_location(
    'icvariant', Path(__file__).resolve().parents[3] / 'scripts' / 'w1c' / 'icvariant.py')
iv = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(iv)

NZ, NY, NX, W = 6, 10, 12, 2
NOW = datetime(2026, 10, 31, 12, tzinfo=timezone.utc)


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _write_wrfinput(path, seed=0):
    rng = np.random.default_rng(seed)
    eta_w = np.linspace(1.0, 0.0, NZ + 1)
    eta_h = 0.5 * (eta_w[:-1] + eta_w[1:])
    ptop = 5000.0
    c3f, c4f = eta_w, np.zeros(NZ + 1)
    c3h, c4h = eta_h, np.zeros(NZ)
    c1f, c2f = np.ones(NZ + 1), np.zeros(NZ + 1)
    rdnw = 1.0 / (eta_w[1:] - eta_w[:-1])
    rdn = np.zeros(NZ); rdn[1:] = 1.0 / (eta_h[1:] - eta_h[:-1])
    mub = 95000.0 + 100 * rng.random((NY, NX))
    mu = 50 * rng.standard_normal((NY, NX))
    t_init = np.repeat(np.linspace(0.0, 40.0, NZ)[:, None, None], NY, 1).repeat(NX, 2)
    pb = c3h[:, None, None] * mub[None] + ptop
    t = t_init + rng.standard_normal((NZ, NY, NX))
    qv = 0.01 * np.exp(-np.arange(NZ))[:, None, None] * (1 + 0.1 * rng.random((NZ, NY, NX)))
    qv[-1] = 0.0  # dry top level (exact +0.0)
    land = np.zeros((NY, NX)); land[3:6, 6:9] = 1.0
    f = {'T': t, 'QVAPOR': qv, 'MU': mu, 'MUB': mub, 'PB': pb, 'T_INIT': t_init, 'C1F': c1f, 'C2F': c2f,
         'C3F': c3f, 'C4F': c4f, 'C3H': c3h, 'C4H': c4h, 'RDNW': rdnw, 'RDN': rdn, 'P_TOP': ptop,
         'PHB': np.cumsum(np.ones((NZ + 1, NY, NX)) * 3000.0, axis=0), 'moist': [qv]}
    r = iv._rebalance(f)
    dims3, dimsw = ('Time', 'bottom_top', 'south_north', 'west_east'), ('Time', 'bottom_top_stag', 'south_north', 'west_east')
    with netCDF4.Dataset(path, 'w') as ds:
        for n, s in (('Time', None), ('DateStrLen', 19), ('bottom_top', NZ), ('bottom_top_stag', NZ + 1),
                     ('south_north', NY), ('west_east', NX), ('west_east_stag', NX + 1)):
            ds.createDimension(n, s)
        ds.setncattr('TITLE', 'OUTPUT FROM REAL_EM'); ds.setncattr('START_DATE', '2026-05-31_00:00:00')
        ds.setncattr('USE_THETA_M', 1)
        tv = ds.createVariable('Times', 'S1', ('Time', 'DateStrLen')); tv[0] = list('2026-05-31_00:00:00')
        def put(n, a, dims, dtype='f4'):
            v = ds.createVariable(n, dtype, dims); v.setncattr('units', 'x'); v[...] = np.asarray(a)[None]
        put('T', t, dims3); put('QVAPOR', qv, dims3); put('P', r['P'], dims3); put('PB', pb, dims3)
        put('AL', r['AL'], dims3); put('T_INIT', t_init, dims3); put('PH', r['PH'], dimsw); put('PHB', f['PHB'], dimsw)
        put('THM', r['tm'] - iv.T0, dims3); put('P_HYD', r['P'] + pb, dims3)
        put('MU', mu, ('Time', 'south_north', 'west_east')); put('MUB', mub, ('Time', 'south_north', 'west_east'))
        put('LANDMASK', land, ('Time', 'south_north', 'west_east')); put('HGT', 100 * land, ('Time', 'south_north', 'west_east'))
        put('U', np.zeros((NZ, NY, NX + 1)), ('Time', 'bottom_top', 'south_north', 'west_east_stag'))
        for n, a, d in (('C1F', c1f, 'bottom_top_stag'), ('C2F', c2f, 'bottom_top_stag'), ('C3F', c3f, 'bottom_top_stag'),
                        ('C4F', c4f, 'bottom_top_stag'), ('C3H', c3h, 'bottom_top'), ('C4H', c4h, 'bottom_top'),
                        ('RDNW', rdnw, 'bottom_top'), ('RDN', rdn, 'bottom_top'), ('ZNU', eta_h, 'bottom_top')):
            put(n, a, ('Time', d))
        put('P_TOP', np.array(ptop), ('Time',))


def _balanced_edit(base, out, *, dt=0.5, cells=None, rebalance=True, close_thm=True):
    """Variant: T += dt and qv *= 0.95 on levels 1-3 of `cells` (default interior water), rebalanced."""
    shutil.copyfile(base, out)
    fb = iv._fields(base)
    with netCDF4.Dataset(out, 'r+') as ds:
        land = np.asarray(ds['LANDMASK'][0]) > 0.5
        m = ~land; m[:W] = m[-W:] = False; m[:, :W] = m[:, -W:] = False
        if cells is not None:
            m = cells
        sel = np.zeros((NZ, NY, NX), bool); sel[1:4] = m
        t = np.asarray(ds['T'][0], np.float64); q = np.asarray(ds['QVAPOR'][0], np.float64)
        t[sel] += dt; q[sel] *= 0.95
        ds['T'][0] = t; ds['QVAPOR'][0] = q
    fv = iv._fields(out)
    rb, rv = iv._rebalance(fb), iv._rebalance(fv)
    with netCDF4.Dataset(out, 'r+') as ds:
        if rebalance:
            for k in ('P', 'AL', 'PH'):
                ds[k][0] = fb[k] + (rv[k] - rb[k])
            ds['P_HYD'][0] = np.asarray(ds['P_HYD'][0], np.float64) + (rv['P'] - rb['P'])
        if close_thm:
            thm = np.asarray(ds['THM'][0]); thm[sel] = (rv['tm'] - iv.T0)[sel]; ds['THM'][0] = thm
    return sel


@pytest.fixture
def case(tmp_path):
    base = tmp_path / 'base'; base.mkdir()
    (base / 'namelist.input').write_text('&domains\n spec_bdy_width = 2,\n/\n')
    _write_wrfinput(base / 'wrfinput_d01')
    (base / 'wrflowinp_d01').write_bytes(b'lowinp')
    inputs = [{'name': n, 'sha256': _sha(base / n), 'bytes': (base / n).stat().st_size}
              for n in ('namelist.input', 'wrfinput_d01', 'wrflowinp_d01')]
    entry = {'case_id': '20260530_18z', 'input_dir': str(base), 'inputs': inputs,
             'case_admission_sha256': 'c' * 64, 'model_init_utc': '2026-05-31T00:00:00Z',
             'model_end_utc': '2026-05-31T18:00:00+00:00', 'bundle_sha256': 'b' * 64, 'metadata_sha256': 'e' * 64}
    mgr = tmp_path / 'BASE_ADMISSION.json'
    mgr.write_text(json.dumps({'authorized_by': 'manager', 'cases': {'20260530_18z': entry}}))
    base_case = iv.load_base_case(mgr, _sha(mgr), '20260530_18z')
    deliver = tmp_path / 'alisios'; deliver.mkdir()
    for n in ('prov.json', 'bal.json'):
        (deliver / n).write_text('{"receipt": "%s"}' % n)
    return tmp_path, base, base_case, deliver


def _admission(tmp_path, base_case, deliver, rows, **over):
    a = {'schema': iv.SCHEMA, 'status': 'ISSUED', 'issued_by': 'alisios_mgr', 'variant_id': 'SND1SHARP_v1',
         'policy': iv.POLICY_A1, 'contract_sha256': iv.CONTRACT_SHA256,
         'issued_utc': '2026-10-31T00:00:00Z', 'valid_until_utc': '2026-11-01T00:00:00Z',
         'truth_allowed': False, 'GPU_launch_authorized': False,
         'base_case': {'case_id': base_case['case_id'], 'case_admission': {'path': '/x', 'sha256': base_case['case_admission_sha256']},
                       'model_init_utc': '2026-05-31T00:00:00+00:00', 'model_end_utc': '2026-05-31T18:00:00Z'},
         'files': rows,
         'provenance': {'receipt': {'path': str(deliver / 'prov.json'), 'sha256': _sha(deliver / 'prov.json')}},
         'balance': {'operator': 'x', 'receipt': {'path': str(deliver / 'bal.json'), 'sha256': _sha(deliver / 'bal.json')}}}
    a.update(over)
    p = tmp_path / 'ADMISSION.json'; p.write_text(json.dumps(a))
    return p, _sha(p)


def _row(base_case, path, name='wrfinput_d01', changed=None):
    base_sha = next(x['sha256'] for x in base_case['inputs'] if x['name'] == name)
    if changed is None:
        changed = iv.structural_diff(Path(base_case['input_dir']) / name, path)
    return {'target_relative_name': name, 'path': str(path), 'sha256': _sha(path), 'size_bytes': Path(path).stat().st_size,
            'base_sha256': base_sha, 'changed_variables': changed}


def _variant(case, **edit):
    tmp_path, base, base_case, deliver = case
    v = deliver / 'wrfinput_d01'
    _balanced_edit(base / 'wrfinput_d01', v, **edit)
    return v


def _expect(code, fn, *args, **kw):
    with pytest.raises(iv.VariantRefused) as e:
        fn(*args, **kw)
    assert e.value.code == code, str(e.value)


def test_balanced_variant_passes_and_stages(case):
    tmp_path, base, base_case, deliver = case
    v = _variant(case)
    p, s = _admission(tmp_path, base_case, deliver, [_row(base_case, v)])
    rec = iv.preflight(p, s, base_case, now=NOW)
    assert rec['status'] == 'PASS' and rec['files'][0]['changed_variables'] == ['AL', 'P', 'PH', 'P_HYD', 'QVAPOR', 'T', 'THM']
    entry = iv.stage(p, s, base_case, tmp_path / 'facade', now=NOW)
    assert entry['case_id'] == '20260530_18z~SND1SHARP_v1' and entry['case_admission_sha256'] == s
    staged = {x['name']: x['sha256'] for x in entry['inputs']}
    assert staged['wrfinput_d01'] == _sha(v) and staged['namelist.input'] == base_case['inputs'][0]['sha256']
    assert all(_sha(tmp_path / 'facade' / n) == h for n, h in staged.items())
    assert entry['variant']['variant_identity_sha256'] == rec['variant_identity_sha256']
    _expect('S1', iv.stage, p, s, base_case, tmp_path / 'facade', now=NOW)  # never overwrite a facade


def test_identity_changes_with_file_or_receipt(case):
    tmp_path, base, base_case, deliver = case
    v = _variant(case)
    a = json.loads(_admission(tmp_path, base_case, deliver, [_row(base_case, v)])[0].read_text())
    ident = lambda x, bc=base_case: iv.variant_identity(x, bc)['variant_identity_sha256']
    b = json.loads(json.dumps(a)); b['files'][0]['sha256'] = '0' * 64
    c = json.loads(json.dumps(a)); c['provenance']['receipt']['sha256'] = '1' * 64
    other = dict(base_case, inputs=[dict(x, sha256='9' * 64) if x['name'] == 'wrflowinp_d01' else x for x in base_case['inputs']])
    assert len({ident(a), ident(b), ident(c), ident(a, other)}) == 4


def test_wrong_admission_hash_refused(case):
    tmp_path, base, base_case, deliver = case
    p, s = _admission(tmp_path, base_case, deliver, [_row(base_case, _variant(case))])
    _expect('R1', iv.preflight, p, '0' * 64, base_case, now=NOW)


@pytest.mark.parametrize('over', [{'status': 'DRAFT'}, {'issued_by': 'someone'}, {'variant_id': 'bad id/../x'},
                                  {'valid_until_utc': '2026-10-31T06:00:00Z'}, {'GPU_launch_authorized': True},
                                  {'schema': 'alisios.ic_variant_admission.v1'}, {'policy': 'ANY_KEY_NEUTRAL'},
                                  {'contract_sha256': '0' * 64}, {'issued_utc': None}, {'valid_until_utc': 'soon'},
                                  {'status': 'DRAFT', 'issued_utc': None, 'issued_by': None}])
def test_admission_header_refusals(case, over):
    tmp_path, base, base_case, deliver = case
    p, s = _admission(tmp_path, base_case, deliver, [_row(base_case, _variant(case))], **over)
    _expect('R1', iv.preflight, p, s, base_case, now=NOW)


def test_wrong_delivered_file_hash_refused(case):
    tmp_path, base, base_case, deliver = case
    row = _row(base_case, _variant(case)); row['sha256'] = 'f' * 64
    p, s = _admission(tmp_path, base_case, deliver, [row])
    _expect('R4', iv.preflight, p, s, base_case, now=NOW)


def test_base_mismatch_refused(case):
    tmp_path, base, base_case, deliver = case
    v = _variant(case)
    row = _row(base_case, v); row['base_sha256'] = 'a' * 64
    p, s = _admission(tmp_path, base_case, deliver, [row])
    _expect('R2', iv.preflight, p, s, base_case, now=NOW)
    other = dict(base_case, case_admission_sha256='d' * 64)
    p, s = _admission(tmp_path, base_case, deliver, [_row(base_case, v)])
    _expect('R2', iv.preflight, p, s, other, now=NOW)


def test_namelist_not_replaceable(case):
    tmp_path, base, base_case, deliver = case
    nl = deliver / 'namelist.input'; nl.write_text('&domains\n spec_bdy_width = 3,\n/\n')
    p, s = _admission(tmp_path, base_case, deliver, [_row(base_case, nl, name='namelist.input', changed=['x'])])
    _expect('R3', iv.preflight, p, s, base_case, now=NOW)


def test_identical_file_refused(case):
    tmp_path, base, base_case, deliver = case
    v = deliver / 'wrfinput_d01'; shutil.copyfile(base / 'wrfinput_d01', v)
    p, s = _admission(tmp_path, base_case, deliver, [_row(base_case, v, changed=['T'])])
    _expect('R4', iv.preflight, p, s, base_case, now=NOW)


@pytest.mark.parametrize('var', ['HGT', 'ZNU', 'PB', 'MUB', 'T_INIT', 'LANDMASK'])
def test_static_field_change_refused(case, var):
    """Changing a field outside the GPU-proven key-neutral set (eta, terrain, base state, land) is refused."""
    tmp_path, base, base_case, deliver = case
    v = _variant(case)
    with netCDF4.Dataset(v, 'r+') as ds:
        a = np.asarray(ds[var][...]); a.reshape(-1)[min(5, a.size - 1)] += np.float32(0.25); ds[var][...] = a
    p, s = _admission(tmp_path, base_case, deliver, [_row(base_case, v)])
    with pytest.raises(iv.VariantRefused) as e:
        iv.preflight(p, s, base_case, now=NOW)
    assert e.value.code == 'R6' and var in str(e.value)


def test_declared_set_must_equal_measured(case):
    tmp_path, base, base_case, deliver = case
    v = _variant(case)
    p, s = _admission(tmp_path, base_case, deliver, [_row(base_case, v, changed=['QVAPOR', 'T'])])
    _expect('R6', iv.preflight, p, s, base_case, now=NOW)


def test_global_attribute_change_refused(case):
    tmp_path, base, base_case, deliver = case
    v = _variant(case)
    with netCDF4.Dataset(v, 'r+') as ds:
        ds.setncattr('START_DATE', '2026-05-31_06:00:00')
    p, s = _admission(tmp_path, base_case, deliver, [_row(base_case, v, changed=['AL', 'P', 'PH', 'P_HYD', 'QVAPOR', 'T', 'THM'])])
    _expect('R5', iv.preflight, p, s, base_case, now=NOW)


@pytest.mark.parametrize('where', ['land', 'zone'])
def test_locality_land_and_outer_zone(case, where):
    tmp_path, base, base_case, deliver = case
    cells = np.zeros((NY, NX), bool)
    if where == 'land':
        cells[4, 7] = True
    else:
        cells[0, 5] = True
    v = _variant(case, cells=cells)
    p, s = _admission(tmp_path, base_case, deliver, [_row(base_case, v)])
    _expect('R8', iv.preflight, p, s, base_case, now=NOW)


def test_non_finite_and_negative_qv_refused(case):
    tmp_path, base, base_case, deliver = case
    v = _variant(case)
    with netCDF4.Dataset(v, 'r+') as ds:
        q = np.asarray(ds['QVAPOR'][...]); q[0, 2, 5, 5] = -1e-6; ds['QVAPOR'][...] = q
    p, s = _admission(tmp_path, base_case, deliver, [_row(base_case, v)])
    _expect('R7', iv.preflight, p, s, base_case, now=NOW)


@pytest.mark.parametrize('edit', [dict(rebalance=False, close_thm=True), dict(rebalance=True, close_thm=False),
                                  dict(rebalance=False, close_thm=False)])
def test_unbalanced_variant_refused(case, edit):
    tmp_path, base, base_case, deliver = case
    v = _variant(case, **edit)
    p, s = _admission(tmp_path, base_case, deliver, [_row(base_case, v)])
    _expect('R9', iv.preflight, p, s, base_case, now=NOW)


def test_ph_only_unbalanced_refused(case):
    """P/AL/THM rebalanced but PH left at base: the hydrostatic increment test alone must catch it."""
    tmp_path, base, base_case, deliver = case
    v = _variant(case)
    with netCDF4.Dataset(base / 'wrfinput_d01') as db, netCDF4.Dataset(v, 'r+') as ds:
        ds['PH'][...] = db['PH'][...]
    p, s = _admission(tmp_path, base_case, deliver, [_row(base_case, v)])
    with pytest.raises(iv.VariantRefused) as e:
        iv.preflight(p, s, base_case, now=NOW)
    assert e.value.code == 'R9' and 'PH increment' in str(e.value)


def _programs(**over):
    comps = {c: 'x' for c in ('static_config', 'carry_aval')}
    comps['source_fingerprint'] = iv.EXPECTED_CONTEXT['engine']['source_fingerprint']
    progs = {n: {'cheap_key': 'k' + n, 'components': dict(comps)} for n in iv.EXPECTED_PROGRAMS}
    progs.update(over)
    return progs


def test_compare_programs():
    base = _programs()
    assert all(r['shared'] for r in iv.compare_programs(base, _programs()).values())
    changed = _programs(d01={'cheap_key': 'z', 'components': dict(base['d01']['components'], static_config='y')})
    with pytest.raises(iv.VariantRefused) as e:
        iv.compare_programs(base, changed)
    assert e.value.code == 'R10' and 'static_config' in str(e.value)
    same_key_other_component = _programs(d03={'cheap_key': 'kd03', 'components': dict(base['d03']['components'], static_config='q')})
    _expect('R10', iv.compare_programs, base, same_key_other_component)
    missing = {k: v for k, v in _programs().items() if k != 'd02'}
    _expect('R10', iv.compare_programs, base, missing)
    _expect('R10', iv.compare_programs, base, _programs(**{'fused/d02': {'cheap_key': None, 'components': {}}}))
    _expect('R10', iv.compare_programs, base, dict(_programs(), **{'fused/d01': base['d01']}))


# --------------------------------------------------------------- critic v1 counterexamples (F1-F5) and races
def test_F1_corrupted_base_static_field_refused(case):
    """Base wrfinput edited in place (forbidden HGT) and the variant carrying the same edit: base bytes no longer
    match the admitted inventory -> R2, nothing staged."""
    tmp_path, base, base_case, deliver = case
    v = _variant(case)
    for f in (base / 'wrfinput_d01', v):
        with netCDF4.Dataset(f, 'r+') as ds:
            x = ds['HGT'][...]; x[0, 5, 5] += np.float32(100); ds['HGT'][...] = x
    p, s = _admission(tmp_path, base_case, deliver, [_row(base_case, v, changed=['AL', 'P', 'PH', 'P_HYD', 'QVAPOR', 'T', 'THM'])])
    _expect('R2', iv.preflight, p, s, base_case, now=NOW)
    _expect('R2', iv.stage, p, s, base_case, tmp_path / 'facade', now=NOW)
    assert not (tmp_path / 'facade').exists() and not (tmp_path / 'facade.partial').exists()


def test_F1_base_authority_pinned(case):
    tmp_path, base, base_case, deliver = case
    mgr = tmp_path / 'BASE_ADMISSION.json'
    _expect('R2', iv.load_base_case, mgr, '0' * 64, '20260530_18z')
    _expect('R2', iv.load_base_case, mgr, _sha(mgr), '20260219_18z')


def test_F2_admission_rewritten_after_pinned_read(case, monkeypatch):
    """The admission file is swapped to an unchecked HGT-edited variant right after the pinned read: stage uses
    the pinned object, stages the checked bytes and never the swapped file."""
    tmp_path, base, base_case, deliver = case
    v = _variant(case); good_sha = _sha(v)
    p, s = _admission(tmp_path, base_case, deliver, [_row(base_case, v)])
    bad = deliver / 'uncertified'; shutil.copyfile(v, bad)
    with netCDF4.Dataset(bad, 'r+') as ds:
        x = ds['HGT'][...]; x[0, 5, 5] += np.float32(100); ds['HGT'][...] = x
    real = iv.pinned_json

    def race(path, expected, code):
        obj = real(path, expected, code)
        a = json.loads(Path(path).read_text())
        a['files'][0].update(path=str(bad), sha256=_sha(bad), size_bytes=bad.stat().st_size)
        Path(path).write_text(json.dumps(a))
        return obj
    monkeypatch.setattr(iv, 'pinned_json', race)
    entry = iv.stage(p, s, base_case, tmp_path / 'facade', now=NOW)
    assert _sha(tmp_path / 'facade' / 'wrfinput_d01') == good_sha != _sha(bad)
    assert entry['case_admission_sha256'] == s


def test_F2_delivered_file_swapped_after_copy(case, monkeypatch):
    """The delivered file changes after its verified copy: checks and publication use the copy."""
    tmp_path, base, base_case, deliver = case
    v = _variant(case); good_sha = _sha(v)
    p, s = _admission(tmp_path, base_case, deliver, [_row(base_case, v)])
    real = iv._copy_verified

    def copy_then_mutate(src, dst, want_sha, want_bytes, code):
        real(src, dst, want_sha, want_bytes, code)
        if Path(src) == v:
            with netCDF4.Dataset(v, 'r+') as ds:
                x = ds['HGT'][...]; x[0, 5, 5] += np.float32(7); ds['HGT'][...] = x
    monkeypatch.setattr(iv, '_copy_verified', copy_then_mutate)
    iv.stage(p, s, base_case, tmp_path / 'facade', now=NOW)
    assert _sha(tmp_path / 'facade' / 'wrfinput_d01') == good_sha != _sha(v)


def test_F2_snapshot_mutation_during_checks_refused(case, monkeypatch):
    tmp_path, base, base_case, deliver = case
    v = _variant(case)
    p, s = _admission(tmp_path, base_case, deliver, [_row(base_case, v)])
    real = iv.check_file

    def check_then_mutate(base_file, variant_file, *a, **k):
        out = real(base_file, variant_file, *a, **k)
        Path(variant_file).chmod(0o600)
        with netCDF4.Dataset(variant_file, 'r+') as ds:
            x = ds['HGT'][...]; x[0, 5, 5] += np.float32(3); ds['HGT'][...] = x
        return out
    monkeypatch.setattr(iv, 'check_file', check_then_mutate)
    _expect('S2', iv.stage, p, s, base_case, tmp_path / 'facade', now=NOW)
    assert not (tmp_path / 'facade').exists() and not (tmp_path / 'facade.partial').exists()


def test_F3_unbalanced_P_HYD_refused(case):
    tmp_path, base, base_case, deliver = case
    v = _variant(case)
    with netCDF4.Dataset(v, 'r+') as ds:
        x = ds['P_HYD'][...]; x[0, 2, 5, 5] += np.float32(100000); ds['P_HYD'][...] = x
    p, s = _admission(tmp_path, base_case, deliver, [_row(base_case, v)])
    with pytest.raises(iv.VariantRefused) as e:
        iv.preflight(p, s, base_case, now=NOW)
    assert e.value.code == 'R9' and 'P_HYD' in str(e.value)


def test_F3_small_P_HYD_offset_refused(case):
    """Isolated negative control at the tolerance scale: 1 Pa P_HYD offset (limit 0.5 Pa) is refused."""
    tmp_path, base, base_case, deliver = case
    v = _variant(case)
    with netCDF4.Dataset(v, 'r+') as ds:
        x = ds['P_HYD'][...].astype(np.float64); x[0, 2, 5, 5] += 1.0; ds['P_HYD'][...] = x.astype(np.float32)
    p, s = _admission(tmp_path, base_case, deliver, [_row(base_case, v)])
    _expect('R9', iv.preflight, p, s, base_case, now=NOW)


@pytest.mark.parametrize('var', ['U', 'MU'])
def test_F4_uncontracted_key_neutral_edit_refused(case, var):
    """U and MU are key-neutral (0219 != 0530) but NOT in the A1 contract (MU fixed, other fields preserved)."""
    tmp_path, base, base_case, deliver = case
    v = _variant(case)
    with netCDF4.Dataset(v, 'r+') as ds:
        x = ds[var][...]; x.reshape(-1)[(x.size // 2)] += np.float32(10); ds[var][...] = x
    p, s = _admission(tmp_path, base_case, deliver, [_row(base_case, v)])
    with pytest.raises(iv.VariantRefused) as e:
        iv.preflight(p, s, base_case, now=NOW)
    assert e.value.code == 'R6' and var in str(e.value)


def test_F4_wrfbdy_not_replaceable_under_A1(case):
    tmp_path, base, base_case, deliver = case
    bdy = deliver / 'wrfbdy_d01'; bdy.write_bytes(b'bdy')
    row = {'target_relative_name': 'wrfbdy_d01', 'path': str(bdy), 'sha256': _sha(bdy), 'size_bytes': 3,
           'base_sha256': 'a' * 64, 'changed_variables': ['T_BXS']}
    p, s = _admission(tmp_path, base_case, deliver, [row])
    _expect('R3', iv.preflight, p, s, base_case, now=NOW)


def test_F4_policy_subset_of_key_neutral():
    for pol in iv.POLICIES.values():
        assert pol['editable'] <= iv.KEY_NEUTRAL_WRFINPUT
        assert set(pol['replaceable']) <= {'wrfinput_d01', 'wrfinput_d02', 'wrfinput_d03'}
    assert 'MU' not in iv.POLICIES[iv.POLICY_A1]['editable'] and 'U' not in iv.POLICIES[iv.POLICY_A1]['editable']


def test_F5_signed_zero_detected_and_refused(case):
    tmp_path, base, base_case, deliver = case
    v = _variant(case)
    with netCDF4.Dataset(v, 'r+') as ds:
        x = ds['HGT'][...]; x[0, 5, 5] = np.float32(-0.0); ds['HGT'][...] = x
    assert 'HGT' in iv.structural_diff(base / 'wrfinput_d01', v)
    p, s = _admission(tmp_path, base_case, deliver, [_row(base_case, v)])
    with pytest.raises(iv.VariantRefused) as e:
        iv.preflight(p, s, base_case, now=NOW)
    assert e.value.code == 'R6' and 'HGT' in str(e.value)


def test_F5_signed_zero_in_protected_cell_refused(case):
    """An editable field flipped +0 -> -0 only on a land cell: numerically equal, bitwise changed -> R8."""
    tmp_path, base, base_case, deliver = case
    v = _variant(case)
    with netCDF4.Dataset(base / 'wrfinput_d01') as db:
        j, i = np.argwhere(np.asarray(db['LANDMASK'][0]) > 0.5)[0]
    with netCDF4.Dataset(v, 'r+') as ds:
        x = ds['QVAPOR'][...]; assert x[0, NZ - 1, j, i] == 0 and not np.signbit(x[0, NZ - 1, j, i])
        x[0, NZ - 1, j, i] = np.float32(-0.0); ds['QVAPOR'][...] = x
    p, s = _admission(tmp_path, base_case, deliver, [_row(base_case, v)])
    _expect('R8', iv.preflight, p, s, base_case, now=NOW)


def test_refused_stage_leaves_nothing(case):
    tmp_path, base, base_case, deliver = case
    v = _variant(case, rebalance=False)
    p, s = _admission(tmp_path, base_case, deliver, [_row(base_case, v)])
    _expect('R9', iv.stage, p, s, base_case, tmp_path / 'facade', now=NOW)
    assert not (tmp_path / 'facade').exists() and not (tmp_path / 'facade.partial').exists()


# --------------------------------------------------------------- R10 receipt binding (no JAX: synthetic receipt)
def _staged(case):
    tmp_path, base, base_case, deliver = case
    v = _variant(case)
    p, s = _admission(tmp_path, base_case, deliver, [_row(base_case, v)])
    entry = iv.stage(p, s, base_case, tmp_path / 'facade', now=NOW)
    params = tmp_path / 'C.json'; params.write_text('{"schema": "gpuwrf-runtime-parameters-v1", "domains": {}}')
    r = {'schema': 'wrf_gpu2.w1c.variant_keys.v2', 'status': 'PASS', 'binding': iv.key_receipt_binding(base_case, entry, params),
         'base': _programs(), 'variant': _programs(), 'created_utc': '2026-10-31T12:00:00+00:00',
         **json.loads(json.dumps(iv.EXPECTED_CONTEXT))}
    rp = tmp_path / 'KEYS.json'; rp.write_text(json.dumps(r))
    return base_case, entry, params, rp, r


def test_R10_receipt_bound(case):
    base_case, entry, params, rp, r = _staged(case)
    assert iv.verify_key_receipt(rp, base_case, entry, params, not_before_unix=NOW.timestamp())['status'] == 'PASS'


@pytest.mark.parametrize('mutate', ['params', 'facade', 'base', 'stale', 'programs', 'status'])
def test_R10_receipt_mismatch_refused(case, mutate):
    base_case, entry, params, rp, r = _staged(case)
    nb = None
    if mutate == 'params':
        params.write_text('{"schema": "gpuwrf-runtime-parameters-v1", "domains": {"d01": {}}}')
    elif mutate == 'facade':
        entry = dict(entry, inputs=[dict(x, sha256='0' * 64) if x['name'] == 'wrfinput_d01' else x for x in entry['inputs']])
    elif mutate == 'base':
        base_case = dict(base_case, authority={'path': '/other', 'sha256': '1' * 64})
    elif mutate == 'stale':
        nb = NOW.timestamp() + 3600
    elif mutate == 'programs':
        r['variant'] = {k: v for k, v in r['variant'].items() if k != 'd03'}; rp.write_text(json.dumps(r))
    elif mutate == 'status':
        r['status'] = 'PARTIAL'; rp.write_text(json.dumps(r))
    _expect('R10', iv.verify_key_receipt, rp, base_case, entry, params, not_before_unix=nb)


def test_F1_inventory_layer_refuses_corrupted_base(case):
    """Layer 1 of F1 (pre-copy inventory check) on its own."""
    tmp_path, base, base_case, deliver = case
    (base / 'wrflowinp_d01').write_bytes(b'lowinX')
    _expect('R2', iv.verify_inventory, base, base_case['inputs'], 'R2')
    v = _variant(case)
    p, s = _admission(tmp_path, base_case, deliver, [_row(base_case, v)])
    with pytest.raises(iv.VariantRefused) as e:
        iv.check_admission(json.loads(p.read_text()), base_case, now=NOW) and iv._build(
            json.loads(p.read_text()), base_case, tmp_path / 'snap', now=NOW)
    assert e.value.code == 'R2' and 'inventory' in str(e.value) and not (tmp_path / 'snap').exists()


def test_F1_copy_layer_refuses_changed_bytes(tmp_path):
    """Layer 2 of F1/F2 (verified copy) on its own: bytes that differ from the pin are never accepted."""
    src = tmp_path / 'src'; src.write_bytes(b'abc')
    _expect('R2', iv._copy_verified, src, tmp_path / 'dst', _sha(src)[:-1] + '0', 3, 'R2')
    _expect('R2', iv._copy_verified, src, tmp_path / 'dst2', _sha(src), 4, 'R2')
    link = tmp_path / 'link'; link.symlink_to(src)
    _expect('R4', iv._copy_verified, link, tmp_path / 'dst3', _sha(src), 3, 'R4')


# --------------------------------------------------------------- v3: critic v2 residuals R1 (context) and R2 (attributes)
@pytest.mark.parametrize('mutation', ['missing_context', 'foreign_engine_file', 'foreign_fingerprint', 'foreign_env',
                                      'extra_env_drift', 'wrong_dt', 'not_cpu', 'foreign_component_fingerprint',
                                      'foreign_fingerprint_both_sides'])
def test_v3_R1_receipt_execution_context_refused(case, mutation):
    base_case, entry, params, rp, r = _staged(case)
    if mutation == 'missing_context':
        for k in ('engine', 'env', 'dt_s', 'platform'):
            r.pop(k)
    elif mutation == 'foreign_engine_file':
        r['engine']['gpuwrf_file'] = '/FOREIGN/gpuwrf/__init__.py'
    elif mutation == 'foreign_fingerprint':
        r['engine']['source_fingerprint'] = '0' * 64
    elif mutation == 'foreign_env':
        r['env'] = {'JAX_PLATFORMS': 'gpu', 'GPUWRF_FAST_DEFAULTS': '0'}
    elif mutation == 'extra_env_drift':
        r['env']['PYTHONPATH'] = '<USER_HOME>/src/wrf_gpu2/src'
    elif mutation == 'wrong_dt':
        r['dt_s'] = {'d01': 54.0, 'd02': 18.0, 'd03': 6.0}
    elif mutation == 'not_cpu':
        r['platform'] = 'gpu'
    elif mutation == 'foreign_component_fingerprint':
        r['variant']['d02']['components']['source_fingerprint'] = '1' * 64
    else:  # consistent between base and variant (compare_programs passes) but not the frozen engine
        for side in ('base', 'variant'):
            for prog in r[side].values():
                prog['components']['source_fingerprint'] = '2' * 64
    rp.write_text(json.dumps(r))
    _expect('R10', iv.verify_key_receipt, rp, base_case, entry, params, not_before_unix=NOW.timestamp())


def test_v3_R1_key_proof_refuses_foreign_env_before_loading(case, monkeypatch):
    base_case, entry, params, rp, r = _staged(case)
    for k, v in iv.EXPECTED_CONTEXT['env'].items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv('GPUWRF_FAST_DEFAULTS', '0')
    with pytest.raises(iv.VariantRefused) as e:
        iv.key_proof(base_case, entry, params, Path(rp).parent / 'kp')
    assert e.value.code == 'R10' and 'environment' in str(e.value)


def test_v3_real_0530_receipt_still_verifies():
    """Carry the real 0530 CPU key proof (v2 smoke, no reload): it satisfies the frozen v3 context."""
    lane = Path('<USER_HOME>/wrf_gpu2_lanes/v46-icvariant/smoke/KEYS.json')
    if not lane.exists():
        pytest.skip('lane receipt not on this host')
    r = json.loads(lane.read_text())
    for k in ('platform', 'engine', 'env', 'dt_s'):
        assert r[k] == iv.EXPECTED_CONTEXT[k], k
    assert {p['components']['source_fingerprint'] for p in (*r['base'].values(), *r['variant'].values())} == \
        {iv.EXPECTED_CONTEXT['engine']['source_fingerprint']}


@pytest.mark.parametrize('where', ['global', 'variable'])
def test_v3_R2_one_ulp_attribute_change_refused(case, where):
    tmp_path, base, base_case, deliver = case
    v = _variant(case)
    for f in (base / 'wrfinput_d01', v):
        with netCDF4.Dataset(f, 'r+') as ds:
            (ds if where == 'global' else ds['T']).setncattr('PRIVATE_BITS', np.array([1.0, 2.0], dtype='f8'))
    with netCDF4.Dataset(v, 'r+') as ds:
        obj = ds if where == 'global' else ds['T']
        x = obj.getncattr('PRIVATE_BITS'); x[0] = np.nextafter(x[0], 2.0); obj.setncattr('PRIVATE_BITS', x)
    _expect('R5', iv.structural_diff, base / 'wrfinput_d01', v)


@pytest.mark.parametrize('pair', [(np.float32(1.0), np.float64(1.0)), (np.array([1, 2], 'i4'), np.array([1, 2, 0], 'i4')),
                                  ('abc', 'abc '), (np.float32(0.0), np.float32(-0.0))])
def test_v3_R2_attribute_record_type_shape_text_bits(pair):
    a, b = pair
    assert iv._attr_record(a) != iv._attr_record(b)
    assert iv._attr_record(a) == iv._attr_record(a)


def test_v3_cli_refuses_tmp_tmpdir(monkeypatch, tmp_path):
    monkeypatch.setenv('TMPDIR', '/tmp')
    out = tmp_path / 'o.json'
    assert iv.main(['preflight', '--base-admission', '/x', '--base-admission-sha', '0', '--base-case-id', 'c',
                    '--out', str(out), '--admission', '/y', '--admission-sha', '0']) == 2
    monkeypatch.delenv('TMPDIR')
    assert iv.main(['preflight', '--base-admission', '/x', '--base-admission-sha', '0', '--base-case-id', 'c',
                    '--out', str(out), '--admission', '/y', '--admission-sha', '0']) == 2


def test_v31_base_entry_without_sizes(case, tmp_path):
    """The real 0219 SCREEN6 entry lists name+sha only: accepted, SHA still enforced, staged entry gets sizes."""
    _, base, base_case, deliver = case
    nosize = dict(base_case, inputs=[{'name': x['name'], 'sha256': x['sha256']} for x in base_case['inputs']])
    v = _variant(case)
    p, s = _admission(tmp_path, nosize, deliver, [_row(nosize, v)])
    entry = iv.stage(p, s, nosize, tmp_path / 'facade_ns', now=NOW)
    assert all('bytes' in x for x in entry['inputs'])
    (base / 'wrflowinp_d01').write_bytes(b'lowinX')
    _expect('R2', iv.verify_inventory, base, nosize['inputs'], 'R2')
    _expect('R2', iv._copy_verified, base / 'wrflowinp_d01', tmp_path / 'x', nosize['inputs'][2]['sha256'], None, 'R2')
