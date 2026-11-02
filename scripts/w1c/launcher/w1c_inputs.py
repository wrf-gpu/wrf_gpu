"""In-window CPU inputs step for one W1c facade (case x arm): re-validate the admission (incl. inventory
re-hash and, for SIC, the bound R10 key receipt) and run the released engine's native cold-input preflight
(screen4_inputs.validate, unchanged) with the member's C parameters. No forecast step, no GPU."""
import argparse
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import w1c
from screen4_inputs import validate

q = w1c.q


def main():
    p = argparse.ArgumentParser(allow_abbrev=False); p.add_argument('--seal-sha', required=True)
    p.add_argument('--admission', type=Path, required=True)
    p.add_argument('--case', choices=w1c.CASES, required=True); p.add_argument('--arm', choices=w1c.ARMS, required=True)
    x = p.parse_args(); here = Path(__file__).resolve().parent; q.sealed(here, x.seal_sha)
    a, rows, prime, cases = w1c.validate(x.admission, os.environ['W1AQ_ADMISSION_SHA'])
    case = cases[(x.case, x.arm)]
    out = w1c.LANE / 'W1C_INPUT_CHECKS' / f'{x.case}.{x.arm}'
    (out / 'NATIVE_INPUT_PREFLIGHT.json').unlink(missing_ok=True)
    t0 = time.monotonic()
    result = validate(dict(case, bundle_sha256=q.BUNDLE_SHA), out, os.environ['W1AQ_ADMISSION_SHA'],
                      parameters=w1c.c.parameters(100e6, .33))
    result.update(W1C_seal=x.seal_sha, W1C_arm=x.arm, W1C_base_case=x.case, W1C_variant=case.get('variant'),
                  W1C_seconds=time.monotonic() - t0)
    q.save(out / 'NATIVE_INPUT_PREFLIGHT.json', result)
    if os.environ.get('WRF_GPU2_SEQ_SMOKE') == '1' and x.case == w1c.CASES[-1] and x.arm == w1c.ARMS[-1]:
        q.save(os.environ['WRF_GPU2_SMOKE_RECEIPT'], dict(device='cpu', real_lock_used=False,
               completed_phases=[f'inputs{k[4:8]}{arm}' for k in w1c.CASES for arm in w1c.ARMS],
               registry_schema_valid=True, production_registry_writes=0, native_gpu_gates='NOT-APPLICABLE'))


if __name__ == '__main__':
    main()
