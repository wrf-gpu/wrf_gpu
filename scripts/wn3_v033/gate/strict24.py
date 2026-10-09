"""Strict first-24h D6 from frozen all-frame reports; no floor annex or new limits.

Reuse each recorded frame's unchanged comparator verdict and the frozen policy
function. Require the complete parent report and exactly leads 0..24 per field.
Usage: strict24.py D6_DIR OUT_JSON [--domains d01,d02,d03]
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

FROZEN = Path('<USER_HOME>/wrf_gpu2_lanes/wn3/V33/gate/tools_src/scripts/wn3_score.py')
FROZEN_SHA = 'ba1b5972f191edfdac825341bc39d855085a34d083f30740f9a114a6c3dff24d'


def strict24(d6_dir, domains):
    if hashlib.sha256(FROZEN.read_bytes()).hexdigest() != FROZEN_SHA:
        raise ValueError('frozen scorer SHA drift')
    spec = importlib.util.spec_from_file_location('frozen_wn3_score', FROZEN)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    root = Path(d6_dir)
    summary = json.loads((root / 'summary.json').read_text())
    result = {'hours': 24, 'annex': 'NONE (strict)', 'source': str(root.resolve()),
              'scorer_sha256': FROZEN_SHA, 'domains': {}}
    source_hashes = {}
    expected = list(range(25))
    for domain in domains:
        failures, incomplete = [], []
        try:
            path = root / f'{domain}.json'
            raw = path.read_bytes()
            source_hashes[domain] = hashlib.sha256(raw).hexdigest()
            report = json.loads(raw)
            parent = summary['domains'][domain]
            if report['inputs']['domain'] != domain or report['pairing']['domain'] != domain:
                incomplete.append('requested domain differs from report/input domain')
            input_roots = {side: Path(report['inputs'][f'{side}_dir']).resolve()
                           for side in ('cpu', 'gpu')}
            pairs72 = report['pairing']['pairs']
            leads72 = [p['lead_h'] for p in pairs72]
            if report['pairing']['paired_file_count'] != 73 or leads72 != list(range(73)):
                incomplete.append('parent report does not contain exactly leads 0..72')
            pairs = [p for p in pairs72 if 0 <= p['lead_h'] <= 24]
            if [p['lead_h'] for p in pairs] != expected:
                incomplete.append('subset pairing does not contain exactly leads 0..24')
            for pair in pairs:
                for side, input_root in input_roots.items():
                    file = Path(pair[f'{side}_file'])
                    if not file.name.startswith(f'wrfout_{domain}_') or file.parent.resolve() != input_root:
                        incomplete.append(f'{side} frame domain/input root mismatch at lead {pair["lead_h"]}')
            selected = {Path(p['gpu_file']).name for p in pairs}
            for name, field in report['field_summaries'].items():
                if name == 'Times':
                    field['checks'] = [x for x in field['checks'] if 0 <= x['lead_h'] <= 24]
                    field['compared_lead_count'] = len(field['checks'])
                    field['all_equal'] = all(x['equal'] for x in field['checks'])
                    if [x['lead_h'] for x in field['checks']] != expected:
                        incomplete.append('Times subset leads missing/duplicated')
                elif 'by_lead' in field:
                    field['by_lead'] = [x for x in field['by_lead'] if 0 <= x['lead_h'] <= 24]
                    field['compared_lead_count'] = len(field['by_lead'])
                    for key in ('missing_leads', 'incompatible_leads'):
                        field[key] = [x for x in field[key] if 0 <= x <= 24]
                    if [x['lead_h'] for x in field['by_lead']] != expected:
                        incomplete.append(f'{name} subset leads missing/duplicated')
                    for frame in field['by_lead']:
                        if frame.get('finite_cpu') != frame['n'] or (frame.get('finite_gpu') == frame['n'] and frame.get('finite_pair') != frame['n']):
                            incomplete.append(f'{name} reference/paired values nonfinite at lead {frame["lead_h"]}')
            failures = tool.report_gate_failures(report, 25)
            incomplete += [f"{x['kind']}: {x.get('field')}" for x in failures
                           if x['kind'] in ('unscored_numeric_field', 'incomplete_field')]
            failures += [{'kind': 'missing_required_fields', 'file': f, 'fields': fields}
                         for f, fields in parent['missing_required_fields'].items() if f in selected]
            failures += [copy.deepcopy(x) | {'kind': 'gpu_only_nonfinite'}
                         for x in parent['gpu_only_nonfinite'] if Path(x['file']).name in selected]
            status = 'INCOMPLETE' if incomplete else ('FAIL' if failures else 'PASS')
            result['domains'][domain] = {'status': status, 'pass': status == 'PASS',
                'paired_frames': len(pairs), 'frame_failures': failures, 'incomplete': incomplete}
        except (OSError, KeyError, TypeError, ValueError) as exc:
            result['domains'][domain] = {'status': 'INCOMPLETE', 'pass': False, 'error': str(exc)}
    result['complete'] = all(d['status'] != 'INCOMPLETE' for d in result['domains'].values())
    result['status'] = ('INCOMPLETE' if not result['complete'] else
                        ('PASS' if all(d['pass'] for d in result['domains'].values()) else 'FAIL'))
    result['pass'] = result['status'] == 'PASS'
    result['source_report_sha256'] = source_hashes
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('d6_dir')
    ap.add_argument('out', type=Path)
    ap.add_argument('--domains', default='d01,d02,d03')
    args = ap.parse_args()
    result = strict24(args.d6_dir, tuple(args.domains.split(',')))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_suffix('.json.tmp')
    temp.write_text(json.dumps(result, indent=1) + '\n')
    os.replace(temp, args.out)
    print(json.dumps({k: result[k] for k in ('status', 'complete', 'hours', 'annex')}))
