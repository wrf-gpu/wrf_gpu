"""Receipt inventory for W3; missing evidence remains INCOMPLETE, W2 is context only."""
import datetime
import hashlib
import json
from pathlib import Path

G = Path('<USER_HOME>/wrf_gpu2_lanes/wn3/V33/gate')
I = Path('<USER_HOME>/wrf_gpu2_lanes/integrate')
M = Path('<USER_HOME>/wrf_gpu2_lanes/mass-opus/MO13')


def receipt(path):
    p = Path(path)
    if not p.is_file():
        return {'path': str(p), 'status': 'INCOMPLETE', 'reason': 'MISSING'}
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def context(label, root, run=None, hours=72, domains=None):
    row = {'label': label, 'scope': 'prior-source context; never substitutes for W3' if run is None else 'final W3 scoring receipts', 'evidence': {}}
    for kind, filename in [(f'd6_{hours}h', f'd6_{hours}h_verdict.json'), ('strict24', 'd6_24h_strict_verdict.json')]:
        p = root / filename
        ref = receipt(p)
        if p.exists():
            v = json.loads(p.read_text())
            ref.update(status=v['status'], complete=v['complete'],
                       domains={d: x['status'] for d, x in v['domains'].items()})
        row['evidence'][kind] = ref
    a = root / 'integrity_all'
    p = a / f'integrity_all_{hours}h.json'
    ref = receipt(p)
    if p.exists():
        v = json.loads(p.read_text())
        checked_domains = v['domains']
        expected = set(domains or (('d01', 'd02') if label == 'MON33W2' else ('d01', 'd02', 'd03')))
        complete = set(checked_domains) == expected and v.get('degeneracy_frames') == 'all' and all(
            d.get('pairing_ok') and d.get('frames_paired') == hours + 1 and
            len(set(d.get('frames_checked', []))) == hours + 1 and
            d.get('global_attrs', {}).get('frames_checked') == hours + 1 for d in checked_domains.values())
        ref.update(complete=complete, status='COMPLETE' if complete else 'INCOMPLETE',
                   frames={d: x.get('frames_paired') for d, x in checked_domains.items()})
    row['evidence']['integrity_all'] = ref
    row['evidence']['integrity_classes'] = receipt(a / f'classes_all_{hours}h.json')
    p = a / 'annex_postpass.txt'
    ref = receipt(p)
    if p.exists():
        ref['raw_verdict'] = next((s for s in p.read_text().splitlines() if s.startswith('ANNEX ')), 'INCOMPLETE')
    row['evidence']['integrity_annex'] = ref
    row['evidence']['twin_manifest'] = receipt(root / 'twin_manifest_20260227_18z_72h.json')
    row['evidence']['R32_cells'] = receipt(root / 'cells.txt')
    paths = {
        'V0227': 'FINAL33V/runs/V0227/20260227_18z_a1',
        'V0227M': 'FINAL33V/runs/V0227M/20260227_18z_a1_icp20261005',
        'V0408': 'FINAL33V/runs/V0408/20260408_18z_a1',
        'V0115': 'FINAL33V/runs/V0115/20260115_18z_a1',
        'W2bR': 'W2BPV/runs/W2bR/20260227_18z_a1',
        'W0408': 'FINAL33W2/runs/W0408/20260408_18z_a1',
        'W0115': 'FINAL33W2/runs/W0115/20260115_18z_a1',
        'Z0115_ZP': 'Z0115/runs/ZP/20260115_18z_a1_icp20261005',
        'Z0115_ZR': 'Z0115/runs/ZR/20260115_18z_a1',
        'MON33W2': 'MON33W2/forecast/real',
    }
    run = Path(run) if run is not None else I / paths[label]
    p = run / 'receipt.json'
    ref = receipt(p)
    if p.exists():
        v = json.loads(p.read_text())
        ref.update({k: v.get(k) for k in ('source', 'rc', 'cli_summary', 'wall_s', 'vram_peak_mib')})
        ref['output_receipts'] = len(v.get('outputs', []))
        ref['scope'] = 'existing measured-process attestation; env values excluded'
    row['evidence']['process_receipt'] = ref
    p = run / 'compress_receipt.jsonl'
    ref = receipt(p)
    if p.exists():
        rows = [json.loads(s) for s in p.read_text().splitlines() if s]
        ref.update(rows=len(rows), unique_files=len({x['file'] for x in rows}),
                   value_identical_count=sum(x.get('value_identical') is True for x in rows))
    row['evidence']['compression_receipt'] = ref
    return row


def main():
    arms = [('W3R', '0227 IC R', 72, 3), ('W3P', '0227 IC P', 72, 3),
            ('W3_0408', '0408 primary', 72, 3), ('W3_0115', '0115 IC R primary', 72, 3),
            ('W3P_0115', '0115 IC P replicate', 72, 3), ('W3Rf_0115', '0115 IC R fresh-cache replicate', 72, 3),
            ('MON33W3', 'Monica', 72, 2), ('SWISS33W3', 'Swiss', 24, 1)]
    required = [{'arm': name, 'role': role, 'hours': hours, 'expected_domain_frames': (hours + 1) * ndom,
                 'status': 'INCOMPLETE', 'reason': 'no final composed-source READY/receipts yet',
                 'needed': ['exact source + resolved keys + original input receipts + E200 finite/frame manifest',
                            'frozen D6 every frame/domain + strict first24 (no floor annex)',
                            'all-frame integrity + unchanged registry/annex + completeness',
                            'exact compiled-program E41 + runtime/VRAM ledger'],
                 'reuse_condition': 'Monica/Swiss: transfer long numeric evidence only if final REFTRA paired-REAL branch inactive and exact whole-program byte proof; otherwise final REFTRA-ON run'
                                   if name in ('MON33W3', 'SWISS33W3') else 'final REFTRA-ON trajectory needed'} for name, role, hours, ndom in arms]
    cases = {'W3R': '20260227_18z_a1', 'W3P': '20260227_18z_a1_icp20261005',
             'W3_0408': '20260408_18z_a1', 'W3_0115': '20260115_18z_a1',
             'W3P_0115': '20260115_18z_a1_icp20261005', 'W3Rf_0115': '20260115_18z_a1'}
    for row in required:
        name = row['arm']
        root = G / 'arms0227' / name if name in cases else G.parent / 'mon33' / name
        run = I / 'FINAL33W3/runs' / name / cases[name] if name in cases else I / name / 'forecast/real'
        domains = ('d01', 'd02', 'd03') if name in cases else (('d01', 'd02') if name == 'MON33W3' else ('d01',))
        evidence = context(name, root, run, row['hours'], domains)['evidence']
        row['scoring_receipts'] = evidence
        own_scoring = [evidence[f"d6_{row['hours']}h"], evidence['strict24'], evidence['integrity_all']]
        if all(x.get('complete') for x in own_scoring):
            row['status'] = 'SCORING_COMPLETE'
            row['reason'] = 'raw verdicts recorded; frozen owner annex/other release gates remain separate'
    matrix = [
        ('REFTRA source review/oracles/OFF controls', 'REUSE', '5ab SOURCE MERGE, reviewed source unchanged; no repeat review/tests'),
        ('E41 barrier numeric/oracle proofs', 'REUSE CONDITIONAL', 'carry forward after minimal whole-program same-input byte proof; no new long run solely for barriers'),
        ('E41 compiled programs', 'NEEDED', 'STACK <=256 every final W3 executable (Monica + WN3); exact-source/cubin receipts'),
        ('P0 A5 original-WRF SC/LANDNIGHT', 'OWNER MATCH', 'b-core + mass-opus bind existing proof to final inputs; repeat only if REFTRA changes tested operands/results'),
        ('REFTRA ON KEY-IN-EFFECT', 'NEEDED', 'A9 W3R d02/d03 all3days; frozen sw_clear, TOA bound unchanged'),
        ('D6 strict24 and72 + A3 floor', 'NEEDED', 'REFTRA ON changes forecast results; final composed W3 only; derive strict24 from same frozen72 reports'),
        ('all-frame integrity/R4/classes/annex', 'NEEDED', 'REFTRA ON affects final frames; no sampled/top-level substitute; original caps unchanged'),
        ('R32 twin / paired station72h', 'NEEDED', 'final W3R/W3P + 0115 primary/replicates; alisios + mass-opus frozen rules'),
        ('G-PD / F1pp / G-OVG / URBAN / BD92C', 'NEEDED', 'affected final W3 output, mass-opus/integrate owners; A7/A8 inputs/limits preserved'),
        ('W3 runtime/VRAM', 'NEEDED', 'measure necessary composed canary/timing; do not infer speed from bitwise values'),
        ('all72h paired plots + release docs', 'NEEDED', 'release-docs uses final W3 extraction; W2 113PNG remain prior-source context'),
    ]
    previous = [context(name, G / 'arms0227' / name) for name in
                ('V0227', 'V0227M', 'V0408', 'V0115', 'W2bR', 'W0408', 'W0115', 'Z0115_ZP', 'Z0115_ZR')]
    previous.append(context('MON33W2', G.parent / 'mon33/MON33W2'))
    pins = {str(p): receipt(p) for p in [M/'medano/GATES_FROZEN_FINAL33V.md',
            G/'tools_src/scripts/wn3_score.py', G/'tools_src/scripts/wn3_fast_compare.py',
            G/'tools_src/proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json',
            G/'d6_verdict.py', G/'integrity_all/REGISTRY_ALLFRAMES.md',
            G/'integrity_all/scripts/wn3_integrity.py', G/'integrity_all/scripts/wn3_score.py',
            G/'integ_classes.py', G/'r4_eval.py', G/'integrity_all/annex_postpass.py']}
    report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'release_status': 'INCOMPLETE', 'frozen_rules': pins, 'W3_required': required,
              'W3_final_composition': receipt(I/'FINAL33W3/COMPOSITION_FINAL.json'),
              'reuse_vs_needed': [{'gate': gate, 'action': action, 'basis': basis} for gate, action, basis in matrix],
              'completed_context': previous,
              'Z0115_owner_reading': receipt(M/'final33v/Z0115/Z0115_READING.json'),
              'retained_input_audit': receipt(M/'w3/TAKEOVER_INPUT_AUDIT.json'),
              'scorer_slots': (G/'d6_slots.conf').read_text(),
              'cache': 'NONE (CPU JSON receipts only)',
              'owned_GPU_waiters': [], 'canceled_obsolete_jobs': [],
              'ETA': {'scoring_measured_prior': 'WN3 ZP D6 18m03s; W0115 D6 32m35s; Monica W2 D6+integrity 26m39s',
                      'scoring_W3': 'I: 20-40min per arm after READY; full queue depends on stagger/QUIET/wrf contention',
                      'release': 'U: final source + E41 + GPU schedule/REFTRA forecast + owner gates pending'}}
    out = G/'RELEASE_GATE_INVENTORY.json'
    tmp = out.with_suffix('.json.tmp'); tmp.write_text(json.dumps(report, indent=1) + '\n'); tmp.replace(out)
    pending = sum(x['status'] == 'INCOMPLETE' for x in required)
    print(f'{out}: release INCOMPLETE; {pending}/{len(required)} W3 arms scoring incomplete; {len(previous)} prior-source contexts; {len(matrix)} gate rows')


if __name__ == '__main__':
    main()
