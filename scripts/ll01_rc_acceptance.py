"""Standing LL01 RC wind gate registered by manager 2026-10-05 21:51:30Z.

Score frames first with ll01_projection_score.py. Compare RC and OFF from the
same ORIGINAL CPU-WRF tau36 history, using the saved projection-plus-pair bound.
This is a history-projection diagnostic; thermal fields remain confounded and
the full 72-hour CPU-WRF release gates are still required.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

assert os.environ.get('JAX_PLATFORMS') == 'cpu'
REGION_SHA = '72d8a76b4f7ddceb9cf49f726db6c4f5a8e04fb8d45f6d36cc1e119df70c72d8'
GATE_REGIONS = ('MEDANO_5KM', 'TENERIFE_LAND', 'COAST_LE5KM')
KEYS = {(d, h) for d in ('d01', 'd02', 'd03') for h in range(37, 49)}
BOUND_RAW_SHA = 'c975acf007dfa62a6fc4507f676cd9941f27ab0430d32108b039878fd5d41cb5'
BOUND_CANON_SHA = '619d4547b6bc8a062e48e0cda474719690ee836d3eca37506866d3e1df44548d'
OFF_RAW_SHA = 'de67e9478fd782d404982cfceb80457544f777a8d7a93b3cb1a95f494a87d1a0'
OFF_CANON_SHA = '74bd5953653df124e7d6b0ad74c8a4bd4362549105cd144de00d7ab5ef859dd2'
OFF_REINIT_SHA = 'aea94c788ce1ea4a86032f95bcb4028ecf498e73883d1c82a07094dd7c67d802'
OFF_COUNTS = dict(zip(GATE_REGIONS, (5, 7, 8)))


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def require_file_sha(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Frozen registration artifact differs: {path}')


def load_json(path):
    return json.loads(path.read_text())


def read_score(directory):
    receipt = load_json(directory / 'receipt.json')
    if receipt['region_operator_sha'] != REGION_SHA or receipt['n_rows'] != 36:
        raise ValueError('Score must cover the frozen regions and all 36 domain-hours')
    rows = [json.loads(line) for line in (directory / 'projection_differences.jsonl').read_text().splitlines()]
    keyed = {(r['domain'], r['lead']): r for r in rows}
    if len(rows) != len(keyed) or set(keyed) != KEYS:
        raise ValueError('Missing, duplicate, or foreign domain-hour')
    return keyed


def rms(cell):
    if cell is None:
        return None
    value = cell['rms']
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Nonfinite or invalid RMS cannot pass the gate')
    return float(value)


def evaluate(off_score, candidate_score, off_reinit, candidate_reinit, bounds, require_on=()):
    if set(off_score) != KEYS or set(candidate_score) != KEYS:
        raise ValueError('Missing, duplicate, or foreign domain-hour')
    # Bind the ORIGINAL artifacts, not merely the supplied arithmetic. These
    # API-level pins also reject callers that bypass main's raw-file checks.
    if canonical_sha(bounds) != BOUND_CANON_SHA:
        raise ValueError('Frozen registered wind bound differs')
    if canonical_sha([off_score[k] for k in sorted(KEYS)]) != OFF_CANON_SHA:
        raise ValueError('Frozen measured OFF score differs')
    if bounds['gate_source'] != 'manager MAILBOX2026-10-05T18:43:28Z':
        raise ValueError('Use the original registered projection-plus-pair bound')
    for manifest in (off_reinit, candidate_reinit):
        if manifest.get('history_role') != 'cpu_wrf' or manifest['lead'] != 36 or manifest['device_platform'] != 'gpu':
            raise ValueError('Gate requires an actual GPU continuation from original CPU tau36 history')
    for key in ('input_dir', 'domains', 'prefix_receipts', 'own_steps', 'dt_s'):
        if off_reinit[key] != candidate_reinit[key]:
            raise ValueError(f'CPU-state provenance or clocks differ: {key}')
    enabled = candidate_reinit.get('resolved_boolean_flags', {})
    if any(enabled.get(flag, '').lower() not in ('1', 'true', 'yes', 'on') for flag in require_on):
        raise ValueError('A required RC flag was not recorded enabled in the measured process')
    lookup = {(r['domain'], r['lead'], r['region']): r for r in bounds['rows']}
    if len(lookup) != len(bounds['rows']):
        raise ValueError('Duplicate bound row')
    details, region_results = [], []
    for domain, lead in sorted(KEYS):
        a, b = off_score[domain, lead], candidate_score[domain, lead]
        if (a['xtime_min'], a['time']) != (b['xtime_min'], b['time']):
            raise ValueError('OFF and RC frame clocks differ')
        if a['regions'].keys() != b['regions'].keys():
            raise ValueError('Region inventory differs')
        for region in a['regions']:
            aa = a['regions'][region]['wind_vector10']['projected_vs_cpu_wrf']
            bb = b['regions'][region]['wind_vector10']['projected_vs_cpu_wrf']
            av, bv = rms(aa), rms(bb)
            if av is None or bv is None:
                if av != bv:
                    raise ValueError('Region eligibility differs')
                continue
            bound = lookup[domain, lead, region]
            ceiling = rms({'rms': bound['bound_rms']})
            parts = rms({'rms': bound['projection_rms']}) + rms({'rms': bound['gpu_pair_rms']})
            if ceiling != parts:
                raise ValueError('Bound differs from its registered projection-plus-pair sum')
            if aa['n'] != bb['n'] or aa['n'] != bound['n']:
                raise ValueError('Region pixel count differs')
            details.append({'domain': domain, 'lead': lead, 'region': region, 'n': aa['n'],
                'off_rms': av, 'rc_rms': bv, 'bound_rms': ceiling,
                'off_inside': av <= ceiling, 'rc_inside': bv <= ceiling})
    for region in GATE_REGIONS:
        selected = [r for r in details if r['domain'] == 'd03' and r['region'] == region]
        if len(selected) != 12 or selected[0]['n'] < 30:
            raise ValueError('Gate region lacks twelve eligible hourly frames')
        off_count = sum(r['off_inside'] for r in selected)
        rc_count = sum(r['rc_inside'] for r in selected)
        if off_count != OFF_COUNTS[region]:
            raise ValueError('OFF count differs from the original registered 5/7/8 baseline')
        region_results.append({'domain': 'd03', 'region': region, 'n': selected[0]['n'],
            'off_hours_inside': off_count, 'rc_hours_inside': rc_count,
            'pass': rc_count >= off_count,
            'off_mean_rms': sum(r['off_rms'] for r in selected) / 12,
            'rc_mean_rms': sum(r['rc_rms'] for r in selected) / 12})
    return {'schema': 'LL01RCWindAcceptance', 'registration': 'manager MAILBOX2026-10-05T21:51:30Z',
        'pass': all(r['pass'] for r in region_results), 'regions': region_results,
        'wind_by_domain_region_lead': details,
        'root_roi_report': [r for r in details if r['domain'] == 'd01' and r['region'] == 'TENERIFE_LAND'],
        'model_sha_off': off_reinit['model_sha'], 'model_sha_rc': candidate_reinit['model_sha'],
        'required_enabled_flags': list(require_on),
        'scope': 'History-projection WIND acceptance only; thermal/QKE/PBLH confounded; full 72h CPU-WRF gates still required'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--off-score', required=True, type=Path)
    p.add_argument('--candidate-score', required=True, type=Path)
    p.add_argument('--off-reinit', required=True, type=Path)
    p.add_argument('--candidate-reinit', required=True, type=Path)
    p.add_argument('--bounds', required=True, type=Path)
    p.add_argument('--require-on', action='append', default=[])
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    # Authenticate frozen inputs BEFORE any candidate evaluation. The output
    # hashes below are provenance receipts, not substitutes for these pins.
    require_file_sha(args.bounds, BOUND_RAW_SHA)
    require_file_sha(args.off_score / 'projection_differences.jsonl', OFF_RAW_SHA)
    require_file_sha(args.off_reinit, OFF_REINIT_SHA)
    result = evaluate(read_score(args.off_score), read_score(args.candidate_score),
        load_json(args.off_reinit), load_json(args.candidate_reinit), load_json(args.bounds), args.require_on)
    files = [args.off_reinit, args.candidate_reinit, args.bounds,
             *(directory / name for directory in (args.off_score, args.candidate_score)
               for name in ('receipt.json', 'projection_differences.jsonl'))]
    result['input_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('LL01_RC_WIND_PASS' if result['pass'] else 'LL01_RC_WIND_FAIL', result['regions'])
    raise SystemExit(0 if result['pass'] else 1)


if __name__ == '__main__':
    main()
