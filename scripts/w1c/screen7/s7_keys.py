"""SCREEN7 R10 (CPU prescreen): the 0530 FULL-selection step programs equal the paid 0219 C3 FULL programs.

The paid ALP1 C3 prime (c3a748e9) compiled the FULL program on 0219 x C3_prime.json. SCREEN7 runs 0530 x
{C3_prime, J3a, J3b}. Product cheap keys (icvariant.program_keys = the `gpuwrf aot-key` loader + runtime
parameters) are computed per case in its OWN CPU process (own GPUWRF_WRF_ROOT tables), then compared:
  0530 x C3_prime == 0219 x C3_prime   (case independence of the FULL program)
  0530 x J3a == 0530 x J3b == 0530 x C3_prime   (alp1 is a traced operand, not a compile constant)
CPU-target keys compare cases/operands; they are NOT the paid GPU keys. The GPU zero-compile refusal,
paid-key and loaded-bytes gates in every member stay mandatory.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))
import icvariant as iv

SCREEN6_V2 = ('<USER_HOME>/wrf_gpu2_lanes/v45-waveops/W1A_SCREEN6_ADMISSION_v2_20261030.json',
              '37e2de3be3a68e1cb7264c776aa476961a886e306507f20704f1fea26cb5fd8c')
PARAMS = {'C3_prime': ('<USER_HOME>/wrf_gpu2_lanes/v45-waveops/ALP1_PARAMETERS_FULL_20261031/C3_prime.json',
                       '8cfe39b8372216406066b2295a8b1b5c78e4ffab3bc6bccc8ae79ebe71d3d5d4'),
          'J3a': ('<USER_HOME>/wrf_gpu2_lanes/v45-waveops/ALP1_PARAMETERS_FULL_20261031/J3a.json',
                  '9cfc929019ea6b805f64258b8ae0a4ff2ae86bb7449f0e02f9ec015d6bada4c0'),
          'J3b': ('<USER_HOME>/wrf_gpu2_lanes/v45-waveops/ALP1_PARAMETERS_FULL_20261031/J3b.json',
                  'c914ab44efeb304cc25d8a907f209c6a6b5e2b65fbf7ce1d4e1d7e3cd173af80')}
PLAN = {'20260530_18z': ('C3_prime', 'J3a', 'J3b'), '20260219_18z': ('C3_prime',)}
SCHEMA_PART = 'wrf_gpu2.screen7.keys_part.v1'
SCHEMA = 'wrf_gpu2.screen7.keys.v1'


def _params(name):
    path, want = PARAMS[name]
    iv.refuse_unless(iv.sha(path) == want, 'R10', f'{name} parameter file changed')
    return {'name': name, 'path': path, 'sha256': want}


def part(case_id, scratch):
    """One case, one process: keys for every parameter file of PLAN[case_id]."""
    case = iv.load_base_case(SCREEN6_V2[0], SCREEN6_V2[1], case_id)
    iv.verify_inventory(case['input_dir'], case['inputs'], 'R2')
    iv.refuse_unless(os.environ.get('GPUWRF_WRF_ROOT') == str(Path(case['input_dir']) / 'tables'), 'R10',
                     'GPUWRF_WRF_ROOT must be this case tables dir')
    iv.refuse_unless({k: os.environ.get(k) for k in iv.KEY_ENV} == iv.EXPECTED_CONTEXT['env'], 'R10',
                     'key-proof environment differs from the frozen context')
    import gpuwrf  # noqa: F401  release defaults before jax backend init
    iv.refuse_unless(gpuwrf.__file__ == iv.EXPECTED_CONTEXT['engine']['gpuwrf_file'], 'R10', 'foreign engine')
    import jax
    iv.refuse_unless(jax.devices()[0].platform == 'cpu', 'R10', 'key proof process not on CPU')
    from gpuwrf.runtime.aot_key_cli import private_compile_caches
    keys = {}
    t0 = time.monotonic()
    with private_compile_caches(scratch / 'cache'):
        for name in PLAN[case_id]:
            ref = _params(name)
            (scratch / name).mkdir()
            k, dt = iv.program_keys(case['input_dir'], Path(ref['path']), scratch / name)
            iv.refuse_unless(dt == iv.EXPECTED_CONTEXT['dt_s'], 'R10', 'work/clock (dt) differs')
            keys[name] = {'params': ref, 'programs': k}
    iv.verify_inventory(case['input_dir'], case['inputs'], 'R2')
    return {'schema': SCHEMA_PART, 'status': 'COMPUTED', 'platform': 'cpu', 'case_id': case_id,
            'base_authority': {'path': SCREEN6_V2[0], 'sha256': SCREEN6_V2[1]}, 'input_dir': case['input_dir'],
            'inventory_sha256': iv.inventory_digest(case['inputs']),
            'engine': {'gpuwrf_file': gpuwrf.__file__}, 'env': {k: os.environ.get(k) for k in iv.KEY_ENV},
            'GPUWRF_WRF_ROOT': os.environ['GPUWRF_WRF_ROOT'], 'keys': keys,
            'seconds': round(time.monotonic() - t0, 1), 'created_utc': datetime.now(timezone.utc).isoformat()}


def combine(may, feb):
    """Pure check of two part receipts -> final R10 receipt (raises on any difference)."""
    iv.refuse_unless(may['schema'] == feb['schema'] == SCHEMA_PART and may['case_id'] == '20260530_18z' and
                     feb['case_id'] == '20260219_18z', 'R10', 'part receipts foreign/swapped')
    for p in (may, feb):
        iv.refuse_unless(sorted(p['keys']) == sorted(PLAN[p['case_id']]) and
                         p['env'] == iv.EXPECTED_CONTEXT['env'] and p['platform'] == 'cpu' and
                         p['engine']['gpuwrf_file'] == iv.EXPECTED_CONTEXT['engine']['gpuwrf_file'], 'R10',
                         'part receipt incomplete/foreign context')
        for name, row in p['keys'].items():
            iv.refuse_unless(row['params'] == _params(name), 'R10', 'part receipt parameter binding differs')
            fps = {v['components'].get('source_fingerprint') for v in row['programs'].values()}
            iv.refuse_unless(fps == {iv.EXPECTED_CONTEXT['engine']['source_fingerprint']}, 'R10',
                             'traced-source fingerprint differs from the frozen engine')
    ref = feb['keys']['C3_prime']['programs']
    rows = {'0530.C3_prime_vs_0219.C3_prime': iv.compare_programs(ref, may['keys']['C3_prime']['programs'])}
    for name in ('J3a', 'J3b'):
        rows[f'0530.{name}_vs_0530.C3_prime'] = iv.compare_programs(may['keys']['C3_prime']['programs'],
                                                                    may['keys'][name]['programs'])
    return {'schema': SCHEMA, 'status': 'PASS', 'platform': 'cpu', 'comparisons': rows,
            'cpu_keys': {n: v['cheap_key'] for n, v in ref.items()},
            'parts': {'20260530_18z': {k: may[k] for k in ('input_dir', 'inventory_sha256', 'GPUWRF_WRF_ROOT', 'seconds')},
                      '20260219_18z': {k: feb[k] for k in ('input_dir', 'inventory_sha256', 'GPUWRF_WRF_ROOT', 'seconds')}},
            'scope': 'CPU prescreen: CPU-target keys compare cases/operands; GPU zero-compile + paid-key + loaded-bytes gates stay mandatory',
            'created_utc': datetime.now(timezone.utc).isoformat()}


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    sub = p.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('part'); a.add_argument('--case-id', choices=sorted(PLAN), required=True)
    a.add_argument('--scratch', type=Path, required=True); a.add_argument('--out', type=Path, required=True)
    b = sub.add_parser('combine'); b.add_argument('--may', type=Path, required=True)
    b.add_argument('--feb', type=Path, required=True); b.add_argument('--out', type=Path, required=True)
    x = p.parse_args()
    if x.cmd == 'part':
        x.scratch.mkdir(exist_ok=False)
        doc = part(x.case_id, x.scratch)
    else:
        doc = combine(json.loads(x.may.read_text()), json.loads(x.feb.read_text()))
        doc['parts']['20260530_18z']['receipt'] = {'path': str(x.may), 'sha256': iv.sha(x.may)}
        doc['parts']['20260219_18z']['receipt'] = {'path': str(x.feb), 'sha256': iv.sha(x.feb)}
    tmp = x.out.with_name('.' + x.out.name + '.tmp')
    tmp.write_text(json.dumps(doc, indent=1) + '\n'); tmp.replace(x.out)
    print(doc['status'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
