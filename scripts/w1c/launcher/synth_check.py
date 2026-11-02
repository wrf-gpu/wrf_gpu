import hashlib, json, os, sys
from pathlib import Path
W = Path('<USER_HOME>/wrf_gpu2_lanes/v46-icvariant/W1C_SEALED_01'); sys.path.insert(0, str(W))
import w1c
p = Path('<USER_HOME>/wrf_gpu2_lanes/v46-icvariant/w1csynth/W1C_ADMISSION_SYNTH.json')
a, rows, prime, cases = w1c.validate(p, hashlib.sha256(p.read_bytes()).hexdigest())
print('VALIDATE PASS', [(r['ordinal'], r['row_id'], r['input_case_id']) for r in rows])
ref = a['C_identity_reference']['20260530_18z']
old = json.load(open(ref['path']))['wrfout_files']
print('C identity self-check', w1c.c_identity('20260530_18z', old, ref)['identical'])
print('C identity mutant', w1c.c_identity('20260530_18z', [dict(old[0], sha256='0'*64)] + old[1:], ref)['identical'])
