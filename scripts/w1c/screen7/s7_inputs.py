"""In-window CPU inputs step for one SCREEN7 arm: re-validate the admission (bindings/operands/prime/R10/0530
inventory) and run the released engine's native cold-input preflight (screen4_inputs.validate, unchanged) on the
0530 facade with the arm's exact FULL parameter tree. No forecast step, no GPU."""
import argparse
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s7
from screen4_inputs import validate

q = s7.q


def main():
    p = argparse.ArgumentParser(allow_abbrev=False); p.add_argument('--seal-sha', required=True)
    p.add_argument('--admission', type=Path, required=True); p.add_argument('--arm', choices=s7.ARMS, required=True)
    x = p.parse_args(); here = Path(__file__).resolve().parent; q.sealed(here, x.seal_sha)
    a, rows, prime, case = s7.validate(x.admission, os.environ['W1AQ_ADMISSION_SHA'])
    row = next(r for r in rows if r['arm_id'] == x.arm)
    out = s7.LANE / 'S7_INPUT_CHECKS' / x.arm
    (out / 'NATIVE_INPUT_PREFLIGHT.json').unlink(missing_ok=True)
    t0 = time.monotonic()
    result = validate(dict(case, bundle_sha256=q.BUNDLE_SHA), out, os.environ['W1AQ_ADMISSION_SHA'],
                      parameters=q.load(row['runtime_parameters']['path']))
    result.update(S7_seal=x.seal_sha, S7_arm=x.arm, S7_runtime_parameters=row['runtime_parameters'],
                  S7_seconds=time.monotonic() - t0)
    q.save(out / 'NATIVE_INPUT_PREFLIGHT.json', result)
    if os.environ.get('WRF_GPU2_SEQ_SMOKE') == '1' and x.arm == s7.ARMS[-1]:
        q.save(os.environ['WRF_GPU2_SMOKE_RECEIPT'], dict(device='cpu', real_lock_used=False,
               completed_phases=[f'inputs{arm}' for arm in s7.ARMS],
               registry_schema_valid=True, production_registry_writes=0, native_gpu_gates='NOT-APPLICABLE'))


if __name__ == '__main__':
    main()
