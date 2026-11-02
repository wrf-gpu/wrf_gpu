import copy, hashlib, json, sys
from pathlib import Path
W = Path('<USER_HOME>/wrf_gpu2_lanes/v46-icvariant/W1C_SEALED_01'); sys.path.insert(0, str(W))
import w1c
S = Path('<USER_HOME>/wrf_gpu2_lanes/v46-icvariant/w1csynth'); A = json.loads((S / 'W1C_ADMISSION_SYNTH.json').read_text())
sh = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ids = json.loads((S / 'IDS.json').read_text())
def with_ids(mut):
    d = copy.deepcopy(ids); mut(d); p = S / 'IDS_mut.json'; p.write_text(json.dumps(d)); return {'path': str(p), 'sha256': sh(p)}
def keys_mut(case, mut):
    k = json.loads((S / f'KEYS_{case}.json').read_text()); mut(k); p = S / f'KEYS_mut_{case}.json'; p.write_text(json.dumps(k))
    return {'path': str(p), 'sha256': sh(p)}
M = {
 'order_swapped': lambda a: a.update(identities=with_ids(lambda d: d['members'].__setitem__(slice(0, 2), d['members'][1::-1]))),
 'sic_identity_mismatch': lambda a: a.update(identities=with_ids(lambda d: d['members'][1].update(variant_identity_sha256='0' * 64))),
 'sic_params_not_C': lambda a: a['parameters'].update(SIC_1L={'path': a['parameters']['C']['path'], 'sha256': '1' * 64}),
 'base_authority_foreign': lambda a: a['base_authorities'].update({'20260219_18z': {'path': str(S / 'W1C_ADMISSION_SYNTH.json'), 'sha256': sh(S / 'W1C_ADMISSION_SYNTH.json')}}),
 'keys_foreign_env': lambda a: a['variants']['20260530_18z'].update(keys=keys_mut('20260530_18z', lambda k: k['env'].update(GPUWRF_FAST_DEFAULTS='0'))),
 'keys_hash_stale': lambda a: a['variants']['20260219_18z']['keys'].update(sha256='2' * 64),
 'entry_wrong_admission': lambda a: a['variants']['20260530_18z'].update(admission=a['variants']['20260219_18z']['admission']),
 'output_root_foreign': lambda a: a.update(output_root='<DATA_ROOT>/wrf_gpu2/scratch/v45-waveops/W1C_1L_X'),
 'prime_foreign': lambda a: a['prime'].update(sha256='3' * 64),
 'retry_on': lambda a: a.update(auto_retry=True),
 'C_ref_path_foreign': lambda a: a['C_identity_reference'].update({'20260530_18z': a['C_identity_reference']['20260219_18z']}),
}
bad = 0
for name, mut in M.items():
    a = copy.deepcopy(A); mut(a); p = S / 'ADM_mut.json'; p.write_text(json.dumps(a))
    try:
        w1c.validate(p, sh(p)); print('ACCEPTED (BAD)', name); bad += 1
    except Exception as e:
        print('refused', name, '|', type(e).__name__, str(e)[:110])
print('false accepts:', bad)
