"""W1c IC case variants (v2): admission validation, snapshot staging, native balance, bound key proof.

A variant is an admitted W1a base case (GPU-proven on the paid programs) in which ALISIOS replaced
wrfinput_d01..d03 by copies edited with its A1 operator (ALISIOS W1c build contract c466dd34:
MU fixed, local theta/qv, incremental P/PH/AL/P_HYD, THM closed, PSFC diagnosed, all other fields
preserved, land and outer zones bit-equal). Everything else is the base case byte for byte.

v2/v3 trust model (critic cr-w44-launchers 2026-10-31T00:24:10Z F1-F5; 01:20:33Z R1/R2 in v3):
  * the admission is read ONCE as pinned bytes and that parsed object is used throughout (F2);
  * the base case entry comes from a pinned manager admission and every base input byte is verified
    against its inventory before use (F1);
  * all checks run on a private SNAPSHOT (verified copies of the delivered files and of the replaced base
    files); `stage` publishes exactly the checked snapshot by an atomic rename, a refused build leaves
    nothing behind (F1/F2);
  * identity and protected cells use raw bit equality (signed zero, NaN payloads), numeric tolerances only
    for physical residuals (F5);
  * the A1 scientific edit policy is separate from (and a subset of) the key-neutral eligibility set (F4);
  * P_HYD is checked as an increment equal to the P increment with PB frozen (F3);
  * v3: the R10 receipt's execution context (engine file + source fingerprint, exact env, CPU, dt) is frozen and
    verified (R1); NetCDF attributes compare as typed exact text / dtype+shape+raw bits (R2).

Refusal codes (VariantRefused.code), all before any GPU work:
  R1 admission header/identity    R2 base authority/inventory   R3 replaceable files (policy)
  R4 delivered bytes              R5 structure/global attrs/Times bit-identical
  R6 edit scope (A1 policy)       R7 finite / QVAPOR >= 0       R8 land + outer-zone bit equality
  R9 native WRF balance           R10 program keys + receipt binding        S* staging
No weather value is printed or logged (ALISIOS model-input rule).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 'alisios.ic_variant_admission.v2'
POLICY_A1 = 'A1_THETA_QV_REBALANCE_V1'
CONTRACT_SHA256 = 'c466dd3415c42a1761f77a57b2b30065515fab1b8459e2bebfb4e3a54c17887f'
# [M] 0219 vs 0530 W1a facades: the only data variables that differ (identical list d01/d02/d03); both cases
# loaded the same 2 paid executables with 0 compiles in 10/10 GPU members. ELIGIBILITY only (not a permission).
KEY_NEUTRAL_WRFINPUT = frozenset(
    'AL ALBBCK LAI MU P PH PSFC P_HYD Q2 QVAPOR SMOIS SST T T2 TH2 THM TMN TSK TSLB U U10 V V10 VEGFRA'.split())
_BDY_SIDES = ('BXS', 'BXE', 'BYS', 'BYE', 'BTXS', 'BTXE', 'BTYS', 'BTYE')
KEY_NEUTRAL_WRFBDY = frozenset(f'{v}_{s}' for v in ('MU', 'PH', 'QVAPOR', 'T', 'U', 'V') for s in _BDY_SIDES)
# A1 PERMISSION (ALISIOS contract c466dd34): the only variables an A1 variant may change.
POLICIES = {POLICY_A1: {'replaceable': ('wrfinput_d01', 'wrfinput_d02', 'wrfinput_d03'),
                        'editable': frozenset({'T', 'QVAPOR', 'THM', 'P', 'PH', 'AL', 'P_HYD', 'PSFC'})}}
assert POLICIES[POLICY_A1]['editable'] <= KEY_NEUTRAL_WRFINPUT
EXPECTED_PROGRAMS = ('d01', 'd02', 'd03', 'fused/d02')
_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$')
# Native balance thresholds (pre-registered; float32 storage scale; real.exe W1a bases measure EOS rel
# 3.4e-7..5.6e-7 and THM closure <= 4.4e-5 K). Closure limits are max(fixed, 2 x base file).
EOS_REL_MAX = 1.0e-6
THM_CLOSURE_MAX_K = 1.0e-4
HYD_INCREMENT_MAX = {'P': 0.5, 'PH': 0.5, 'AL': 1.0e-5, 'P_HYD': 0.5}   # Pa, m2/s2, m3/kg, Pa
R_D, R_V, P1000, T0 = 287.0, 461.6, 1.0e5, 300.0
CP = 7.0 * R_D / 2.0
CVPM = -(CP - R_D) / CP
MOIST = ('QVAPOR', 'QCLOUD', 'QRAIN', 'QICE', 'QSNOW', 'QGRAUP')
KEY_ENV = ('JAX_PLATFORMS', 'GPUWRF_FAST_DEFAULTS', 'PYTHONPATH', 'GPUWRF_JAX_CACHE', 'JAX_ENABLE_COMPILATION_CACHE')
# Frozen R10 execution context (v3, critic R1): the released engine bundle 32ea753a, its traced-source
# fingerprint [M: KEYS smoke + KP1, identical for all 4 programs], the exact key-relevant environment, CPU, dt.
ENGINE_SRC = '<USER_HOME>/wrf_gpu2_lanes/v45-rc/runtime/bundle/src/src'
EXPECTED_CONTEXT = {
    'platform': 'cpu',
    'engine': {'gpuwrf_file': ENGINE_SRC + '/gpuwrf/__init__.py',
               'source_fingerprint': 'dd4b800895f7bbeadfc979ae6e77d05eaefe1f41afe9de8c1967d8dfba1ea817'},
    'env': {'JAX_PLATFORMS': 'cpu', 'GPUWRF_FAST_DEFAULTS': '1', 'PYTHONPATH': ENGINE_SRC,
            'GPUWRF_JAX_CACHE': '0', 'JAX_ENABLE_COMPILATION_CACHE': 'false'},
    'dt_s': {'d01': 27.0, 'd02': 9.0, 'd03': 3.0},
}


class VariantRefused(ValueError):
    def __init__(self, code, message):
        super().__init__(f'{code}: {message}')
        self.code = code


def refuse_unless(ok, code, message):
    if not ok:
        raise VariantRefused(code, message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _utc(text):
    return datetime.fromisoformat(text.replace('Z', '+00:00')).astimezone(timezone.utc)


def _same_instant(a, b):
    try:
        return _utc(a) == _utc(b)
    except (TypeError, ValueError, AttributeError):
        return False


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))


def pinned_json(path, expected_sha, code):
    """Read the bytes ONCE, check their SHA, parse those same bytes."""
    data = Path(path).read_bytes()
    refuse_unless(hashlib.sha256(data).hexdigest() == expected_sha, code, f'{Path(path).name}: bytes differ from pinned SHA')
    return json.loads(data)


def inventory_digest(inputs):
    return hashlib.sha256(canonical(sorted((x['name'], x['sha256'], x.get('bytes')) for x in inputs)).encode()).hexdigest()


def verify_inventory(directory, inputs, code):
    """Every listed file: regular, not a symlink, size and SHA as inventoried."""
    for x in inputs:
        p = Path(directory) / x['name']
        refuse_unless(p.is_file() and not p.is_symlink(), code, f'{x["name"]}: not a regular file')
        refuse_unless(('bytes' not in x or p.stat().st_size == x['bytes']) and sha(p) == x['sha256'], code,
                      f'{x["name"]}: bytes differ from inventory')


# --------------------------------------------------------------------------- authorities (R1, R2)
def load_base_case(admission_path, admission_sha, case_id):
    """Base case entry from the pinned manager admission (SCREEN4/6 'cases' table)."""
    a = pinned_json(admission_path, admission_sha, 'R2')
    refuse_unless(a.get('authorized_by') == 'manager', 'R2', 'base admission not manager-authorized')
    entry = (a.get('cases') or {}).get(case_id)
    refuse_unless(isinstance(entry, dict) and entry.get('case_id') == case_id, 'R2', f'base case {case_id} not in admission')
    for k in ('input_dir', 'inputs', 'case_admission_sha256', 'model_init_utc', 'model_end_utc'):
        refuse_unless(k in entry, 'R2', f'base case entry lacks {k}')
    refuse_unless(all(isinstance(x.get('sha256'), str) and len(x['sha256']) == 64 for x in entry['inputs']), 'R2',
                  'base inventory lacks SHA-256')  # sizes optional (the 0219 entry of SCREEN6 v2 lists name+sha only)
    return dict(entry, authority={'path': str(admission_path), 'sha256': admission_sha})


def check_admission(a, base_case, *, now=None):
    """R1-R3 on the already pinned+parsed admission object. Returns {name: row}."""
    refuse_unless(a.get('schema') == SCHEMA and a.get('status') == 'ISSUED' and a.get('issued_by') == 'alisios_mgr',
                  'R1', 'schema/status/issuer')
    refuse_unless(isinstance(a.get('variant_id'), str) and bool(_ID_RE.match(a['variant_id'])), 'R1', 'variant_id syntax')
    refuse_unless(a.get('policy') in POLICIES, 'R1', 'unknown edit policy')
    refuse_unless(a.get('contract_sha256') == CONTRACT_SHA256, 'R1', 'edit contract SHA mismatch')
    now = now or datetime.now(timezone.utc)
    try:
        window = (_utc(a['issued_utc']), _utc(a['valid_until_utc']))
    except (KeyError, TypeError, ValueError, AttributeError):
        raise VariantRefused('R1', 'issued_utc/valid_until_utc missing or not a UTC timestamp') from None
    refuse_unless(window[0] <= now <= window[1], 'R1', 'outside validity window')
    for flag in ('truth_allowed', 'GPU_launch_authorized'):
        refuse_unless(a.get(flag) is False, 'R1', f'{flag} must be false (GPU needs a separate GO)')
    for key in ('provenance', 'balance'):
        rec = (a.get(key) or {}).get('receipt') or {}
        refuse_unless(bool(rec.get('path')) and Path(rec['path']).is_file() and sha(rec['path']) == rec.get('sha256'),
                      'R1', f'{key} receipt missing or SHA mismatch')
    b = a.get('base_case') or {}
    refuse_unless(b.get('case_id') == base_case['case_id'] and
                  (b.get('case_admission') or {}).get('sha256') == base_case['case_admission_sha256'] and
                  _same_instant(b.get('model_init_utc'), base_case['model_init_utc']) and
                  _same_instant(b.get('model_end_utc'), base_case['model_end_utc']), 'R2', 'base case binding mismatch')
    policy = POLICIES[a['policy']]
    base_inv = {x['name']: x for x in base_case['inputs']}
    rows = {}
    refuse_unless(bool(a.get('files')), 'R3', 'no replaced file')
    for r in a['files']:
        name = r.get('target_relative_name')
        refuse_unless(name in policy['replaceable'], 'R3', f'{name}: not replaceable under {a["policy"]}')
        refuse_unless(name not in rows, 'R3', f'duplicate file {name}')
        refuse_unless(name in base_inv and r.get('base_sha256') == base_inv[name]['sha256'], 'R2',
                      f'{name}: base_sha256 != admitted base input')
        refuse_unless(isinstance(r.get('sha256'), str) and r['sha256'] != r['base_sha256'], 'R4', f'{name}: identical to base (no-op)')
        refuse_unless(isinstance(r.get('changed_variables'), list) and r['changed_variables'], 'R6', f'{name}: changed_variables missing')
        rows[name] = r
    return rows


def variant_identity(a, base_case):
    payload = {'schema': SCHEMA, 'policy': a['policy'], 'contract_sha256': a['contract_sha256'],
               'base_case_id': base_case['case_id'], 'base_case_admission_sha256': base_case['case_admission_sha256'],
               'base_inventory_sha256': inventory_digest(base_case['inputs']), 'variant_id': a['variant_id'],
               'files': sorted((r['target_relative_name'], r['sha256']) for r in a['files']),
               'provenance_receipt_sha256': a['provenance']['receipt']['sha256'],
               'balance_receipt_sha256': a['balance']['receipt']['sha256']}
    return {'variant_case_id': f"{base_case['case_id']}~{a['variant_id']}",
            'variant_identity_sha256': hashlib.sha256(canonical(payload).encode()).hexdigest(), 'payload': payload}


# --------------------------------------------------------------------------- bitwise structure (R5-R8)
def _bits(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    return a.view(f'u{a.dtype.itemsize}') if a.dtype.kind in 'fiub' and a.dtype.itemsize in (1, 2, 4, 8) else a.view(np.uint8)


def _attr_record(value):
    """Typed exact record of a NetCDF attribute: exact text, or dtype + shape + raw bits (critic R2)."""
    import numpy as np
    if isinstance(value, str):
        return ('str', value)
    if isinstance(value, bytes):
        return ('bytes', value.hex())
    a = np.ascontiguousarray(np.asarray(value))
    if a.dtype.kind in 'US':
        return ('text-array', a.dtype.str, a.shape, [str(x) for x in a.reshape(-1)])
    return ('numeric', a.dtype.str, a.shape, a.tobytes().hex())


def _attrs(obj):
    return [(k, _attr_record(obj.getncattr(k))) for k in obj.ncattrs()]   # ordered: names, order, types, bits


def _raw(var):
    import numpy as np
    var.set_auto_maskandscale(False)
    return np.asarray(var[...])


def structural_diff(base, variant):
    """Sorted variables whose RAW BYTES differ; refuse on any other difference."""
    import netCDF4
    with netCDF4.Dataset(base) as db, netCDF4.Dataset(variant) as dv:
        refuse_unless({k: len(v) for k, v in db.dimensions.items()} == {k: len(v) for k, v in dv.dimensions.items()},
                      'R5', 'dimensions differ')
        refuse_unless(set(db.variables) == set(dv.variables), 'R5', 'variable set differs')
        refuse_unless(_attrs(db) == _attrs(dv), 'R5', 'global attributes differ')
        changed = []
        for n in sorted(db.variables):
            vb, vv = db.variables[n], dv.variables[n]
            refuse_unless(vb.dimensions == vv.dimensions and vb.dtype == vv.dtype and _attrs(vb) == _attrs(vv),
                          'R5', f'{n}: dims/dtype/attributes differ')
            if _raw(vb).tobytes() != _raw(vv).tobytes():
                changed.append(n)
    refuse_unless('Times' not in changed and not any(c.startswith('md___') for c in changed), 'R5', 'Times/bdy clocks differ')
    return changed


def check_values_and_locality(base, variant, changed, spec_bdy_width):
    import netCDF4
    import numpy as np
    with netCDF4.Dataset(base) as db, netCDF4.Dataset(variant) as dv:
        land = _raw(dv['LANDMASK'])[0] > 0.5
        for n in changed:
            x, y = _raw(dv.variables[n]), _raw(db.variables[n])
            if x.dtype.kind == 'f':
                refuse_unless(bool(np.isfinite(x).all()), 'R7', f'{n}: non-finite values')
            if n == 'QVAPOR':
                refuse_unless(bool((x >= 0).all()), 'R7', f'{n}: negative water vapour')
            dims = dv.variables[n].dimensions
            refuse_unless(len(dims) >= 3 and dims[-1].startswith('west_east') and dims[-2].startswith('south_north'),
                          'R6', f'{n}: not a horizontal field')
            ny, nx = x.shape[-2], x.shape[-1]
            w = int(spec_bdy_width)
            mask = np.zeros((ny, nx), bool); mask[:w] = mask[-w:] = True; mask[:, :w] = mask[:, -w:] = True
            lm = land
            if lm.shape != (ny, nx):  # staggered face: land if either neighbour is land
                pad = np.zeros((ny, nx), bool)
                if nx == lm.shape[1] + 1:
                    pad[:, :-1] |= lm; pad[:, 1:] |= lm
                elif ny == lm.shape[0] + 1:
                    pad[:-1, :] |= lm; pad[1:, :] |= lm
                lm = pad
            mask |= lm
            diff = (_bits(x) != _bits(y)).reshape(-1, ny, nx).any(axis=0)
            refuse_unless(not bool((diff & mask).any()), 'R8', f'{n}: changed on land or in the outer {w}-cell zone')


# --------------------------------------------------------------------------- native balance (R9)
def _fields(path):
    import netCDF4
    import numpy as np
    with netCDF4.Dataset(path) as ds:
        g = lambda n: np.asarray(ds[n][0], dtype=np.float64)
        out = {n: g(n) for n in ('T', 'QVAPOR', 'P', 'PB', 'PH', 'PHB', 'AL', 'MU', 'MUB', 'THM', 'T_INIT', 'P_HYD',
                                 'C1F', 'C2F', 'C3F', 'C4F', 'C3H', 'C4H', 'RDNW', 'RDN')}
        out['moist'] = [g(n) for n in MOIST if n in ds.variables]
        out['P_TOP'] = float(np.asarray(ds['P_TOP'][...]).reshape(-1)[0])
    return out


def _rebalance(f):
    """WRF start_em rebalance relations (dyn_em/start_em.F:2368-2460, use_theta_m=1, hypsometric_opt=2)."""
    import numpy as np
    mu, mub, pb, phb = f['MU'], f['MUB'], f['PB'], f['PHB']
    c1f, c2f, c3f, c4f, c3h, c4h, rdnw, rdn, ptop = (f[k] for k in ('C1F', 'C2F', 'C3F', 'C4F', 'C3H', 'C4H', 'RDNW', 'RDN', 'P_TOP'))
    qsum = sum(f['moist'])
    tm = (f['T'] + T0) * (1.0 + (R_V / R_D) * f['QVAPOR'])
    nz = tm.shape[0]
    pr = np.empty_like(tm); alt = np.empty_like(tm)
    kk = nz - 1; k = kk + 1
    qvf2 = 1.0 / (1.0 + qsum[kk]); qvf1 = qsum[kk] * qvf2
    pr[kk] = -0.5 * ((c1f[k] * mu) + qvf1 * (c1f[k] * mub + c2f[k])) / rdnw[kk] / qvf2
    alt[kk] = (R_D / P1000) * tm[kk] * ((pr[kk] + pb[kk]) / P1000) ** CVPM
    for kk in range(nz - 2, -1, -1):
        k = kk + 1
        qtot = 0.5 * (qsum[kk] + qsum[kk + 1]); qvf2 = 1.0 / (1.0 + qtot); qvf1 = qtot * qvf2
        pr[kk] = pr[kk + 1] - ((c1f[k] * mu) + qvf1 * (c1f[k] * mub + c2f[k])) / qvf2 / rdn[kk + 1]
        alt[kk] = (R_D / P1000) * tm[kk] * ((pr[kk] + pb[kk]) / P1000) ** CVPM
    alb = (R_D / P1000) * (f['T_INIT'] + T0) * (pb / P1000) ** CVPM
    ph = np.empty((nz + 1,) + tm.shape[1:]); ph[0] = phb[0]
    for k in range(1, nz + 1):
        pfu = c3f[k] * (mub + mu) + c4f[k] + ptop
        pfd = c3f[k - 1] * (mub + mu) + c4f[k - 1] + ptop
        phm = c3h[k - 1] * (mub + mu) + c4h[k - 1] + ptop
        ph[k] = ph[k - 1] + alt[k - 1] * phm * np.log(pfd / pfu)
    return {'P': pr, 'AL': alt - alb, 'PH': ph - phb, 'alt': alt, 'alb': alb, 'tm': tm}


def balance(base, variant):
    """R9: EOS + THM closure on the variant; (variant - base) of P/AL/PH equal to the increment of WRF's
    rebalance relations; P_HYD increment == P increment (start_em p_hyd = p + pb, PB frozen). Only increments
    are tested: real.exe files are not start_em-rebalance exact in absolute terms (base P up to ~3e2 Pa)."""
    import numpy as np
    fb, fv = _fields(base), _fields(variant)
    rv, rb = _rebalance(fv), _rebalance(fb)

    def closures(f, r):
        alt_file = (R_D / P1000) * r['tm'] * ((f['P'] + f['PB']) / P1000) ** CVPM
        return (float(np.max(np.abs((f['AL'] - (alt_file - r['alb'])) / alt_file))),
                float(np.max(np.abs(f['THM'] - (r['tm'] - T0)))))
    eos_rel, thm = closures(fv, rv)
    eos_base, thm_base = closures(fb, rb)
    eos_lim, thm_lim = max(EOS_REL_MAX, 2.0 * eos_base), max(THM_CLOSURE_MAX_K, 2.0 * thm_base)
    inc = {k: float(np.max(np.abs((fv[k] - fb[k]) - (rv[k] - rb[k])))) for k in ('P', 'PH', 'AL')}
    inc['P_HYD'] = float(np.max(np.abs((fv['P_HYD'] - fb['P_HYD']) - (fv['P'] - fb['P']))))
    out = {'EOS_rel_max': eos_rel, 'THM_closure_max_K': thm, 'hydrostatic_increment_max': inc,
           'base_EOS_rel_max': eos_base, 'base_THM_closure_max_K': thm_base,
           'thresholds': {'EOS_rel_max': eos_lim, 'THM_closure_max_K': thm_lim, 'hydrostatic_increment_max': HYD_INCREMENT_MAX}}
    refuse_unless(eos_rel <= eos_lim, 'R9', 'EOS closure (AL vs T/qv/P) violated')
    refuse_unless(thm <= thm_lim, 'R9', 'THM not closed with T/QVAPOR')
    for k, lim in HYD_INCREMENT_MAX.items():
        refuse_unless(inc[k] <= lim, 'R9', f'{k} increment not hydrostatically balanced')
    return out


def _namelist_int(path, key, default):
    m = re.search(rf'^\s*{key}\s*=\s*(\d+)', Path(path).read_text(), re.M | re.I)
    return int(m.group(1)) if m else default


def check_file(base_file, variant_file, name, declared, policy, spec_bdy_width):
    changed = structural_diff(base_file, variant_file)
    refuse_unless(changed == sorted(declared), 'R6', f'{name}: measured changed variables != declared')
    extra = sorted(set(changed) - POLICIES[policy]['editable'])
    refuse_unless(not extra, 'R6', f'{name}: changes outside {policy}: {extra}')
    check_values_and_locality(base_file, variant_file, changed, spec_bdy_width)
    return {'name': name, 'changed_variables': changed, 'balance': balance(base_file, variant_file)}


# --------------------------------------------------------------------------- snapshot build + stage
def _copy_verified(src, dst, want_sha, want_bytes, code):
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = Path(src)
    refuse_unless(src.is_file() and not src.is_symlink(), code, f'{src.name}: not a regular file')
    shutil.copyfile(src, dst)
    dst.chmod(0o400)
    refuse_unless((want_bytes is None or dst.stat().st_size == want_bytes) and sha(dst) == want_sha, code,
                  f'{dst.name}: copied bytes differ from pin')


def _build(a, base_case, snap, *, now=None):
    """Checks R1-R9 on a private snapshot `snap` (created here). Returns (receipt, inputs)."""
    rows = check_admission(a, base_case, now=now)
    base_dir = Path(base_case['input_dir'])
    verify_inventory(base_dir, base_case['inputs'], 'R2')                      # F1: authenticate the live base
    snap.mkdir(parents=True, mode=0o700)
    inputs = []
    for x in base_case['inputs']:
        if x['name'] in rows:
            r = rows[x['name']]
            _copy_verified(r['path'], snap / x['name'], r['sha256'], r.get('size_bytes'), 'R4')
            _copy_verified(base_dir / x['name'], snap / '_base' / x['name'], x['sha256'], x.get('bytes'), 'R2')
            inputs.append({'name': x['name'], 'sha256': r['sha256'], 'bytes': r['size_bytes'], 'origin': 'variant'})
        else:
            _copy_verified(base_dir / x['name'], snap / x['name'], x['sha256'], x.get('bytes'), 'R2')
            inputs.append(dict(x, bytes=(snap / x['name']).stat().st_size, origin='base'))
    refuse_unless(set(rows) <= {x['name'] for x in base_case['inputs']}, 'R3', 'replaced file not in base inputs')
    width = _namelist_int(snap / 'namelist.input', 'spec_bdy_width', 5)
    checks = [check_file(snap / '_base' / n, snap / n, n, rows[n]['changed_variables'], a['policy'], width)
              for n in sorted(rows)]
    for x in inputs:  # the checked snapshot bytes are still the pinned bytes
        refuse_unless(sha(snap / x['name']) == x['sha256'], 'S2', f'{x["name"]}: snapshot changed during checks')
    if (snap / '_base').exists():
        for p in (snap / '_base').rglob('*'):
            if p.is_file():
                p.chmod(0o600)
        shutil.rmtree(snap / '_base')
    receipt = {'schema': 'wrf_gpu2.w1c.variant_preflight.v2', 'status': 'PASS', **variant_identity(a, base_case),
               'policy': a['policy'], 'base_case_id': base_case['case_id'],
               'base_authority': base_case.get('authority'), 'base_inventory_sha256': inventory_digest(base_case['inputs']),
               'files': checks, 'spec_bdy_width': width, 'checked_unix_s': time.time()}
    return receipt, inputs


def _rmtree(path):
    if Path(path).exists():
        for p in Path(path).rglob('*'):
            try:
                p.chmod(0o700 if p.is_dir() else 0o600)
            except OSError:
                pass
        shutil.rmtree(path, ignore_errors=True)


def preflight(admission_path, expected_sha, base_case, *, now=None):
    """R1-R9 on a private snapshot that is discarded; returns the receipt."""
    import tempfile
    a = pinned_json(admission_path, expected_sha, 'R1')
    tmp = Path(tempfile.mkdtemp(prefix='w1c_preflight_'))
    try:
        receipt, _ = _build(a, base_case, tmp / 'snap', now=now)
    finally:
        _rmtree(tmp)
    return dict(receipt, admission_sha256=expected_sha)


def stage(admission_path, expected_sha, base_case, out_dir, *, now=None):
    """Build + check a snapshot next to out_dir and publish it by atomic rename. Returns the `cases` entry."""
    a = pinned_json(admission_path, expected_sha, 'R1')                       # F2: one pinned read
    out_dir = Path(out_dir)
    refuse_unless(not out_dir.exists(), 'S1', 'variant facade directory already exists')
    partial = out_dir.with_name(out_dir.name + '.partial')
    refuse_unless(not partial.exists(), 'S1', 'stale partial facade exists')
    try:
        receipt, inputs = _build(a, base_case, partial, now=now)
        receipt.update(admission_sha256=expected_sha, input_dir=str(out_dir), inputs=inputs,
                       staged_utc=datetime.now(timezone.utc).isoformat())
        (partial / 'VARIANT_FACADE_RECEIPT.json').write_text(json.dumps(receipt, indent=1) + '\n')
        os.rename(partial, out_dir)
    except BaseException:
        _rmtree(partial)
        raise
    entry = {k: base_case[k] for k in ('model_init_utc', 'model_end_utc', 'bundle_sha256', 'metadata_sha256') if k in base_case}
    entry.update(case_id=receipt['variant_case_id'], input_dir=str(out_dir),
                 inputs=[{k: v for k, v in i.items() if k != 'origin'} for i in inputs],
                 case_admission_sha256=expected_sha,
                 variant={'variant_id': a['variant_id'], 'policy': a['policy'],
                          'variant_identity_sha256': receipt['variant_identity_sha256'],
                          'base_case_id': base_case['case_id'], 'base_case_admission_sha256': base_case['case_admission_sha256'],
                          'base_inventory_sha256': receipt['base_inventory_sha256'],
                          'facade_receipt_sha256': sha(out_dir / 'VARIANT_FACADE_RECEIPT.json')})
    return entry


# --------------------------------------------------------------------------- R10 program keys (CPU prescreen)
def program_keys(input_dir, params, scratch, *, hours=18, max_dom=3):
    """Product cheap keys of every step program (same loader/runtime as `gpuwrf aot-key`, plus the member's
    runtime parameters). CPU only; never compiles or steps. CPU-target keys compare CASES; they are not the
    paid GPU cache keys."""
    refuse_unless(os.environ.get('JAX_PLATFORMS') == 'cpu', 'R10', 'key proof runs on CPU only')
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.runtime.aot_key_probe import case_program_keys
    from gpuwrf.runtime.domain_tree import DomainTree
    cfg = pipeline.NestedPipelineConfig(Path(input_dir), scratch / 'out', scratch / 'proof', hours=hours,
                                        max_dom=max_dom, scratch_dir=scratch, runtime_parameters=params)
    hierarchy, bundles, meta, _s, dt, carries = pipeline._load_domains(cfg, pipeline.domain_names_for(max_dom))
    nesting = meta.get('nesting', {'feedback': bool(cfg.feedback), 'spec_zone': 1, 'smooth_option': 1})
    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=nesting['feedback'],
                                   feedback_spec_zone=nesting['spec_zone'], feedback_smooth_option=nesting['smooth_option'])
    keys = case_program_keys(tree, carries, components=True)
    return ({k: {'cheap_key': v.get('cheap_key'), 'components': v.get('components', {})} for k, v in keys.items()},
            {k: float(v) for k, v in dt.items()})


def compare_programs(base, variant):
    """Exactly the expected program set in both, keys equal, no differing component."""
    refuse_unless(set(base) == set(EXPECTED_PROGRAMS) and set(variant) == set(EXPECTED_PROGRAMS), 'R10',
                  f'program set differs from {EXPECTED_PROGRAMS}')
    rows = {}
    for name in EXPECTED_PROGRAMS:
        a, b = base[name], variant[name]
        if a['cheap_key'] is None or b['cheap_key'] is None:
            rows[name] = {'shared': False, 'differing_components': ['<no key>']}
            continue
        comps = sorted(c for c in set(a['components']) | set(b['components'])
                       if a['components'].get(c) != b['components'].get(c))
        rows[name] = {'shared': a['cheap_key'] == b['cheap_key'] and not comps, 'differing_components': comps}
    refuse_unless(all(r['shared'] for r in rows.values()), 'R10',
                  'variant changes a step program: ' + json.dumps({k: v for k, v in rows.items() if not v['shared']}))
    return rows


def key_receipt_binding(base_case, entry, params_path, *, hours=18, max_dom=3):
    """What a key receipt must bind; recomputed by the launcher before it trusts a receipt."""
    params = Path(params_path)
    return {'base_case_id': base_case['case_id'], 'base_authority': base_case.get('authority'),
            'base_inventory_sha256': inventory_digest(base_case['inputs']),
            'variant_case_id': entry['case_id'], 'variant_admission_sha256': entry['case_admission_sha256'],
            'variant_identity_sha256': entry['variant']['variant_identity_sha256'],
            'facade_inventory_sha256': inventory_digest(entry['inputs']), 'facade_dir': entry['input_dir'],
            'params_sha256': sha(params), 'params_schema': json.loads(params.read_text()).get('schema'),
            'hours': hours, 'max_dom': max_dom, 'expected_programs': list(EXPECTED_PROGRAMS)}


def key_proof(base_case, entry, params_path, scratch):
    """R10: verify both inventories, compute keys base vs staged variant, return the bound receipt."""
    verify_inventory(base_case['input_dir'], base_case['inputs'], 'R2')
    verify_inventory(entry['input_dir'], entry['inputs'], 'R10')
    refuse_unless({k: os.environ.get(k) for k in KEY_ENV} == EXPECTED_CONTEXT['env'], 'R10', 'key-proof environment differs from the frozen context')
    import gpuwrf  # noqa: F401  release defaults before jax backend init
    refuse_unless(gpuwrf.__file__ == EXPECTED_CONTEXT['engine']['gpuwrf_file'], 'R10', 'key proof imported a foreign engine')
    import jax
    refuse_unless(jax.devices()[0].platform == 'cpu', 'R10', 'key proof process not on CPU')
    from gpuwrf.runtime.aot_key_cli import private_compile_caches
    t0 = time.monotonic()
    with private_compile_caches(scratch / 'cache'):
        (scratch / 'b').mkdir(); (scratch / 'v').mkdir()
        kb, dtb = program_keys(base_case['input_dir'], params_path, scratch / 'b')
        kv, dtv = program_keys(entry['input_dir'], params_path, scratch / 'v')
    refuse_unless(dtb == dtv == EXPECTED_CONTEXT['dt_s'], 'R10', 'work/clock (dt) differs')
    rows = compare_programs(kb, kv)
    fps = {p['components'].get('source_fingerprint') for p in (*kb.values(), *kv.values())}
    refuse_unless(fps == {EXPECTED_CONTEXT['engine']['source_fingerprint']}, 'R10', 'traced-source fingerprint differs from the frozen engine')
    verify_inventory(entry['input_dir'], entry['inputs'], 'R10')                # unchanged while probed
    return {'schema': 'wrf_gpu2.w1c.variant_keys.v2', 'status': 'PASS', 'platform': 'cpu',
            'binding': key_receipt_binding(base_case, entry, params_path),
            'engine': {'gpuwrf_file': gpuwrf.__file__, 'source_fingerprint': kb['d01']['components'].get('source_fingerprint')},
            'env': {k: os.environ.get(k) for k in KEY_ENV}, 'dt_s': dtv,
            'base': kb, 'variant': kv, 'comparison': rows, 'seconds': round(time.monotonic() - t0, 1),
            'scope': 'CPU prescreen: CPU-target keys compare cases; GPU zero-compile + paid-key + loaded-bytes gates stay mandatory',
            'created_utc': datetime.now(timezone.utc).isoformat()}


def verify_key_receipt(path, base_case, entry, params_path, *, not_before_unix=None):
    """Launcher gate: a stale/foreign/partial key receipt is refused."""
    r = json.loads(Path(path).read_text())
    refuse_unless(r.get('schema') == 'wrf_gpu2.w1c.variant_keys.v2' and r.get('status') == 'PASS', 'R10', 'key receipt not PASS')
    refuse_unless(r.get('binding') == key_receipt_binding(base_case, entry, params_path), 'R10', 'key receipt binding mismatch')
    for k in ('platform', 'engine', 'env', 'dt_s'):  # critic R1: the execution context is part of the proof
        refuse_unless(r.get(k) == EXPECTED_CONTEXT[k], 'R10', f'key receipt {k} missing or differs from the frozen context')
    compare_programs(r['base'], r['variant'])
    fps = {p.get('components', {}).get('source_fingerprint') for p in (*r['base'].values(), *r['variant'].values())}
    refuse_unless(fps == {EXPECTED_CONTEXT['engine']['source_fingerprint']}, 'R10', 'program components carry a foreign source fingerprint')
    if not_before_unix is not None:
        refuse_unless(_utc(r['created_utc']).timestamp() >= not_before_unix, 'R10', 'stale key receipt')
    return r


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog='icvariant', allow_abbrev=False)
    sub = p.add_subparsers(dest='cmd', required=True)
    for c in ('preflight', 'stage', 'keys'):
        s = sub.add_parser(c, allow_abbrev=False)
        s.add_argument('--base-admission', type=Path, required=True, help='pinned manager admission with the cases table')
        s.add_argument('--base-admission-sha', required=True); s.add_argument('--base-case-id', required=True)
        s.add_argument('--out', type=Path, required=True)
        if c in ('preflight', 'stage'):
            s.add_argument('--admission', type=Path, required=True); s.add_argument('--admission-sha', required=True)
        if c == 'stage':
            s.add_argument('--facade', type=Path, required=True)
        if c == 'keys':
            s.add_argument('--entry', type=Path, required=True, help='STAGE.json written by `stage`')
            s.add_argument('--params', type=Path, required=True)
    a = p.parse_args(argv)
    tmp = os.environ.get('TMPDIR', '')
    if not (tmp and Path(tmp).is_absolute() and Path(tmp).is_dir() and not Path(tmp).resolve().is_relative_to('/tmp')
            and Path(tmp).stat().st_uid == os.getuid()):
        print('REFUSED S0: TMPDIR must be a private existing lane directory (not /tmp, which is RAM tmpfs)')
        return 2
    try:
        base_case = load_base_case(a.base_admission, a.base_admission_sha, a.base_case_id)
        if a.cmd == 'preflight':
            out = preflight(a.admission, a.admission_sha, base_case)
        elif a.cmd == 'stage':
            out = stage(a.admission, a.admission_sha, base_case, a.facade)
        else:
            import tempfile
            entry = json.loads(a.entry.read_text())
            with tempfile.TemporaryDirectory(prefix='w1c_keys_') as tmp:
                out = key_proof(base_case, entry, a.params, Path(tmp))
    except VariantRefused as exc:
        a.out.write_text(json.dumps({'status': 'REFUSED', 'code': exc.code, 'reason': str(exc)}, indent=1) + '\n')
        print('REFUSED', exc)
        return 2
    a.out.write_text(json.dumps(out, indent=1, default=str) + '\n')
    print('PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
