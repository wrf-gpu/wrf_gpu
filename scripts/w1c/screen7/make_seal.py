"""Seal S7_SEALED_01 for one pinned SCREEN7 manager admission: SEAL.json, entry.sh, inputs.sh, PLAN.json.

Usage: python -P make_seal.py --admission A.json --source-commit <sha>
Refuses if any reused module differs from its sealed origin (SCREEN6_SEALED_03, ALP1_J3_REFERENCE_SEALED_01,
W1C_SEALED_01). Files become read-only.
"""
import argparse, hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINS = {
    '<USER_HOME>/wrf_gpu2_lanes/v45-waveops/SCREEN6_SEALED_03/SEAL.json':
        ('screen4.py', 'screen4_member.py', 'screen4_inputs.py', 'w1a_cache.py', 'w1a_change.py', 'w1a_inputs.py',
         'w1a_meta.py', 'w1a_qualify.py', 'w1a_verify.py', 'gpu_gate.py', 'night_seq_v10.sh'),
    '<USER_HOME>/wrf_gpu2_lanes/v45-waveops/ALP1_J3_REFERENCE_SEALED_01/SEAL.json': ('alp1_contract.py',),
    '<USER_HOME>/wrf_gpu2_lanes/v46-icvariant/W1C_SEALED_01/SEAL.json': ('w1c_rows.py', 'w1c_disk.py', 'icvariant.py'),
}
REUSED = tuple(f for files in ORIGINS.values() for f in files)
NEW = ('s7.py', 's7_budget.py', 's7_inputs.py', 's7_keys.py', 's7_metrics.py', 's7_raw_release.py', 'LAYOUT.json')
PY = '<USER_HOME>/miniconda3/bin/python'
BUNDLE_SRC = '<USER_HOME>/wrf_gpu2_lanes/v45-rc/runtime/bundle/src/src'
LOCK = '<USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


ENTRY = r'''#!/usr/bin/env bash
set -euo pipefail
umask 077
ulimit -c 0
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
py=@PY@
base=(HOME=<USER_HOME> PATH=<USER_HOME>/miniconda3/bin:/usr/local/cuda/bin:/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8
      JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=@SRC@ TMPDIR=<USER_HOME>/wrf_gpu2_lanes/v46-icvariant/tmp)
if [[ ${WRF_GPU2_SEQ_SMOKE:-0} == 1 ]]; then
  exec env -i "${base[@]}" WRF_GPU2_SEQ_SMOKE=1 W1AQ_ADMISSION_SHA=@ASHA@ \
    WRF_GPU2_SMOKE_DIR="${WRF_GPU2_SMOKE_DIR:?}" WRF_GPU2_SMOKE_RECEIPT="${WRF_GPU2_SMOKE_RECEIPT:?}" \
    GA_START_UTC="${GA_START_UTC:?}" GA_END_UTC="${GA_END_UTC:?}" \
    "$py" -P "$here/s7.py" --seal-sha @SSHA@ --admission @ADM@
fi
context=(GA_START_UTC="${GA_START_UTC:?}" GA_END_UTC="${GA_END_UTC:?}" GA_GATE_ID="${GA_GATE_ID:?}"
         GA_RW_GATE_ID="${GA_RW_GATE_ID:?}" W1AQ_ADMISSION_SHA="${W1AQ_ADMISSION_SHA:?}"
         W1A_ALP1_BUDGET_SHA="${W1A_ALP1_BUDGET_SHA:?}" ALISIOS_OPTIMIZE_WALL_DEADLINE="${ALISIOS_OPTIMIZE_WALL_DEADLINE:-}")
if [[ ${1:-} != --locked ]]; then
  # wave ledger START before the official lock wrapper, END after it (s7_budget.py, ALP1 rule); entry hard cap there
  exec systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 \
    env -i "${base[@]}" "${context[@]}" \
    taskset -c @CORES@ nice -n 19 "$py" -P "$here/s7_budget.py" --seal-sha @SSHA@ --admission @ADM@ -- \
    @LOCK@ --label @LABEL@ -- bash "$0" --locked
fi
exec env -i "${base[@]}" "${context[@]}" W1A_ALP1_BUDGET_ENTRY_ID="${W1A_ALP1_BUDGET_ENTRY_ID:?}" \
  GPUWRF_GPU_LOCK_FD="${GPUWRF_GPU_LOCK_FD:?}" GPUWRF_GPU_LOCK_HELD="${GPUWRF_GPU_LOCK_HELD:?}" \
  GPUWRF_GPU_LOCK_LABEL="${GPUWRF_GPU_LOCK_LABEL:?}" GPUWRF_GPU_LOCK_TOKEN="${GPUWRF_GPU_LOCK_TOKEN:?}" \
  GPUWRF_GPU_LOCK_FILE="${GPUWRF_GPU_LOCK_FILE:?}" GPUWRF_GPU_LOCK_HOLDER_FILE="${GPUWRF_GPU_LOCK_HOLDER_FILE:?}" \
  "$py" -P "$here/s7.py" --seal-sha @SSHA@ --admission @ADM@
'''

INPUTS = r'''#!/usr/bin/env bash
set -euo pipefail
umask 077
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for arm in C_new alp1_low alp1_high; do
  systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 taskset -c @CORES@ nice -n 19 timeout 900 \
    env -i HOME=<USER_HOME> PATH=<USER_HOME>/miniconda3/bin:/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    JAX_PLATFORMS=cpu GPUWRF_FAST_DEFAULTS=1 GPUWRF_JAX_CACHE=0 JAX_ENABLE_COMPILATION_CACHE=false \
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=@SRC@ TMPDIR=<USER_HOME>/wrf_gpu2_lanes/v46-icvariant/tmp \
    W1AQ_ADMISSION_SHA=@ASHA@ WRF_GPU2_SEQ_SMOKE="${WRF_GPU2_SEQ_SMOKE:-0}" \
    WRF_GPU2_SMOKE_RECEIPT="${WRF_GPU2_SMOKE_RECEIPT:-}" GA_START_UTC="${GA_START_UTC:?}" GA_END_UTC="${GA_END_UTC:?}" \
    @PY@ -P "$here/s7_inputs.py" --seal-sha @SSHA@ --admission @ADM@ --arm "$arm"
done
'''


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--admission', type=Path, required=True); p.add_argument('--source-commit', required=True)
    p.add_argument('--cores', default='24,25,28')
    x = p.parse_args()
    reused_from = {}
    for origin, files in ORIGINS.items():
        ref = json.loads(Path(origin).read_text())
        for f in files:
            assert sha(HERE / f) == ref['files'][f], f'reused module differs from {origin}: {f}'
        reused_from[origin] = {'sha256': sha(origin), 'files': list(files)}
    bundle_files = json.loads(Path(next(iter(ORIGINS))).read_text())['bundle_files']
    adm = json.loads(x.admission.read_text()); asha = sha(x.admission)
    assert (HERE / 'LAYOUT.json').stat().st_mode & 0o777 == 0o444, 'LAYOUT.json must already be 0444 (ALISIOS-bound pointer)'
    layout = json.loads((HERE / 'LAYOUT.json').read_text())
    assert adm['output_root'] == layout['output_root'], 'admission output_root != LAYOUT.json'
    seal = {'schema': 'wrf_gpu2.screen7.seal.v1', 'source_commit': x.source_commit,
            'files': {f: sha(HERE / f) for f in (*REUSED, *NEW)}, 'bundle_files': bundle_files,
            'reused_from': reused_from, 'admission': {'path': str(x.admission.resolve()), 'sha256': asha},
            'authority': 'ONLY manager gate after distinct signed ALISIOS GO'}
    (HERE / 'SEAL.json').write_text(json.dumps(seal, indent=2) + '\n')
    ssha = sha(HERE / 'SEAL.json')
    sub = lambda t: (t.replace('@PY@', PY).replace('@SRC@', BUNDLE_SRC).replace('@ASHA@', asha).replace('@SSHA@', ssha)
                     .replace('@ADM@', str(x.admission.resolve())).replace('@LABEL@', adm['gpu_label'])
                     .replace('@CORES@', x.cores).replace('@LOCK@', LOCK))
    for name, t in (('entry.sh', ENTRY), ('inputs.sh', INPUTS)):
        (HERE / name).write_text(sub(t)); (HERE / name).chmod(0o555)
    env = {'PATH': '<USER_HOME>/miniconda3/bin:/usr/local/cuda/bin:/usr/bin:/bin', 'PYTHONPATH': BUNDLE_SRC,
           'GA_GATE_ID': adm['gate_id'], 'W1AQ_ADMISSION_SHA': asha, 'W1A_ALP1_BUDGET_SHA': adm['budget_ledger']['sha256']}
    smoke = lambda d: [PY, '-P', str(HERE / 's7.py'), '--seal-sha', ssha, '--admission', str(x.admission.resolve()),
                       '--schema-smoke', '{smoke_dir}/' + d]
    w = adm['window']; b = adm['budget']
    plan = {'schema': 'wrf_gpu2.gpu.sequence.v1', 'source_commit': x.source_commit, 'purpose': 'SCREEN7_0530',
            'steps': [{'lane': adm['gate_id'] + '-inputs', 'path': str(HERE / 'inputs.sh'), 'sha256': sha(HERE / 'inputs.sh'),
                       'cap_min': b['CPU_admission_min'], 'args': [], 'env': env, 'required_modules': ['gpuwrf.optimize.registry'],
                       'phases': ['inputsC_new', 'inputsalp1_low', 'inputsalp1_high'], 'registry_smoke': smoke('inputs')},
                      {'lane': adm['gate_id'], 'path': str(HERE / 'entry.sh'), 'sha256': sha(HERE / 'entry.sh'),
                       'cap_min': b['GPU_entry_cap_min'], 'args': [], 'env': env, 'required_modules': ['gpuwrf.optimize.registry'],
                       'phases': ['preflight', 'member1', 'member2', 'member3', 'halt'], 'registry_smoke': smoke('models')}],
            'layout': layout, 'window': w, 'budget': b,
            'total_admission_minutes': b['CPU_admission_min'] + b['GPU_entry_cap_min'],
            'execution': 'distinct manager gate / ALISIOS signed GO'}
    (HERE / 'PLAN.json').write_text(json.dumps(plan, indent=2) + '\n')
    for f in (*REUSED, *NEW, 'SEAL.json', 'PLAN.json'):
        (HERE / f).chmod(0o444)
    print(json.dumps({'seal_sha256': ssha, 'plan_sha256': sha(HERE / 'PLAN.json'), 'admission_sha256': asha}))


if __name__ == '__main__':
    main()
