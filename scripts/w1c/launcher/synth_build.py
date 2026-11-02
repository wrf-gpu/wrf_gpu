"""SYNTHETIC W1c fixtures (NOT ALISIOS authority, NOT a key proof): balanced variants for 0219/0530, staged,
fabricated key receipts with the frozen context, fake ID confirmation and W1c admission -> logic tests only."""
import hashlib, json, sys, subprocess
from pathlib import Path
W = Path('<USER_HOME>/wrf_gpu2_lanes/v46-icvariant/W1C_SEALED_01'); sys.path.insert(0, str(W))
import icvariant as iv
L = Path('<USER_HOME>/wrf_gpu2_lanes/v46-icvariant/w1csynth')
MGR = Path('<USER_HOME>/wrf_gpu2_lanes/v45-waveops/W1A_SCREEN6_ADMISSION_v2_20261030.json')
C = Path('<USER_HOME>/wrf_gpu2_lanes/v45-waveops/SCREEN4_PARAMETERS_20261030/C.json')
sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ref = lambda p: {'path': str(p), 'sha256': sh(p)}
variants = {}
for case in ('20260219_18z', '20260530_18z'):
    base = iv.load_base_case(MGR, '37e2de3be3a68e1cb7264c776aa476961a886e306507f20704f1fea26cb5fd8c', case)
    src = L / f'bal_{case}'
    if not src.exists():
        subprocess.run([sys.executable, '-P', '<USER_HOME>/wrf_gpu2_lanes/v46-icvariant/tools/make_balanced.py', base['input_dir'], str(src)], check=True)
    for n in ('prov', 'bal'):
        (L / f'{n}_{case}.json').write_text(json.dumps({'SYNTHETIC': True, 'kind': n, 'case': case}))
    rows = [{'target_relative_name': f'wrfinput_{d}', 'path': str(src / f'wrfinput_{d}'), 'sha256': sh(src / f'wrfinput_{d}'),
             'size_bytes': (src / f'wrfinput_{d}').stat().st_size,
             'base_sha256': next(x['sha256'] for x in base['inputs'] if x['name'] == f'wrfinput_{d}'),
             'changed_variables': iv.structural_diff(Path(base['input_dir']) / f'wrfinput_{d}', src / f'wrfinput_{d}')}
            for d in ('d01', 'd02', 'd03')]
    a = {'schema': iv.SCHEMA, 'status': 'ISSUED', 'issued_by': 'alisios_mgr', 'policy': iv.POLICY_A1,
         'contract_sha256': iv.CONTRACT_SHA256, 'variant_id': 'SYNTH_SIC', 'note': 'SYNTHETIC, NOT ALISIOS',
         'issued_utc': '2026-10-30T23:00:00Z', 'valid_until_utc': '2026-11-01T00:00:00Z', 'truth_allowed': False,
         'GPU_launch_authorized': False,
         'base_case': {'case_id': case, 'case_admission': {'path': '-', 'sha256': base['case_admission_sha256']},
                       'model_init_utc': base['model_init_utc'], 'model_end_utc': base['model_end_utc']},
         'files': rows, 'provenance': {'receipt': ref(L / f'prov_{case}.json')},
         'balance': {'operator': 'synthetic', 'receipt': ref(L / f'bal_{case}.json')}}
    ap = L / f'ADM_{case}.json'; ap.write_text(json.dumps(a, indent=1))
    fac = L / f'facade_{case}'
    if not fac.exists():
        entry = iv.stage(ap, sh(ap), base, fac)
        (L / f'STAGE_{case}.json').write_text(json.dumps(entry, indent=1))
    entry = json.loads((L / f'STAGE_{case}.json').read_text())
    progs = {n: {'cheap_key': 'SYNTHETIC', 'components': {'source_fingerprint': iv.EXPECTED_CONTEXT['engine']['source_fingerprint']}}
             for n in iv.EXPECTED_PROGRAMS}
    keys = {'schema': 'wrf_gpu2.w1c.variant_keys.v2', 'status': 'PASS', 'SYNTHETIC_NOT_A_KEY_PROOF': True,
            'binding': iv.key_receipt_binding(base, entry, C), 'base': progs, 'variant': progs,
            'created_utc': '2026-10-31T02:00:00+00:00', **json.loads(json.dumps(iv.EXPECTED_CONTEXT))}
    kp = L / f'KEYS_{case}.json'; kp.write_text(json.dumps(keys, indent=1))
    variants[case] = {'entry': ref(L / f'STAGE_{case}.json'), 'admission': ref(ap), 'keys': ref(kp)}
rows = []
for i, (cs, arm) in enumerate([('20260530_18z', 'C'), ('20260530_18z', 'SIC_1L'), ('20260219_18z', 'C'), ('20260219_18z', 'SIC_1L')]):
    oid = hashlib.md5(f'{cs}{arm}'.encode()).hexdigest()
    st = json.loads((L / f'STAGE_{cs}.json').read_text())
    rows.append({'ordinal': i + 1, 'row_id': f'W1C_1L_V1.{cs}.{arm}.det0', 'our_id': oid, 'registry_run_id': 'optimize:' + oid,
                 'base_case_id': cs, 'case_id': cs if arm == 'C' else st['case_id'], 'arm_id': arm,
                 'variant_id': None if arm == 'C' else st['variant']['variant_id'],
                 'variant_identity_sha256': None if arm == 'C' else st['variant']['variant_identity_sha256'],
                 'launcher_policy': 'NO_RESUME', 'model_hours': 18, 'parameters': {'nt_c': 100e6, 'entrainment': .33},
                 'extract_output_dir': str(L / 'synth_extract' / oid)})
(L / 'IDS.json').write_text(json.dumps({'SYNTHETIC': True, 'members': rows}, indent=1))
W1 = {'schema': 'wrf_gpu2.w1c.admission.v1', 'authorized_by': 'manager', 'note': 'SYNTHETIC', 'scope': 'W1C_1L_COLD_NO_RESUME',
      'failure_policy': 'STOP_ON_FIRST_FAIL', 'NO_RESUME': True, 'auto_retry': False,
      'execution_authority': 'MANAGER_GATE_AFTER_ALISIOS_SIGNED_GO', 'gpu_label': 'v45-waveops', 'gate_id': 'v46-icvariant-w1c',
      'output_root': '<DATA_ROOT>/wrf_gpu2/scratch/v46-icvariant/W1C_1L_SYNTH',
      'identities': ref(L / 'IDS.json'), 'base_admission': ref(MGR),
      'base_authorities': {c: ref(MGR) for c in ('20260219_18z', '20260530_18z')},
      'C_identity_reference': {'20260219_18z': ref('<DATA_ROOT>/wrf_gpu2/scratch/v45-waveops/W1A_SCREEN4_20261030/MEMBER_1_DONE.json'),
                               '20260530_18z': ref('<DATA_ROOT>/wrf_gpu2/scratch/v45-waveops/W1A_SCREEN4_20261030/MEMBER_2_DONE.json')},
      'prime': ref('<DATA_ROOT>/wrf_gpu2/scratch/v45-waveops/w1a32_v7_35afd9707970.R11_STOPPED_20261030T073852Z/QUALIFIED_PRIME.json'),
      'logical_root': '<DATA_ROOT>/wrf_gpu2/scratch/v45-waveops/w1a32_v7_35afd9707970',
      'cache_layer': '<DATA_ROOT>/wrf_gpu2/scratch/v45-waveops/w1a32_v7_35afd9707970/bundle_layer',
      'parameters': {'C': ref(C), 'SIC_1L': ref(C)}, 'variants': variants,
      'window': {'not_before': '2026-10-31T00:00:00Z', 'latest_sequence_start': '2026-10-31T23:00:00Z',
                 'latest_GPU_start': '2026-10-31T23:10:00Z', 'hard_end': '2026-10-31T23:59:00Z'},
      'budget': {'GPU_entry_cap_min': 32, 'CPU_admission_min': 45}}
(L / 'W1C_ADMISSION_SYNTH.json').write_text(json.dumps(W1, indent=1))
print('admission', sh(L / 'W1C_ADMISSION_SYNTH.json'))
