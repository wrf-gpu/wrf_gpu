"""Optional real-model CPU integration; normal unit collection remains bounded."""
from pathlib import Path
import os
import subprocess
import sys
import json
import pytest


def test_real_nested_znt_two_domains_two_frames(tmp_path):
    if os.environ.get('RUN_ZNT_REAL_NESTED')!='1':
        pytest.skip('real CPU nested forecast (20-minute cap): opt in with RUN_ZNT_REAL_NESTED=1')
    case=Path(os.environ.get('ZNT_REAL_CASE','<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725'))
    if not all((case/name).is_file() for name in ('wrfinput_d01','wrfinput_d02','wrfbdy_d01','namelist.input')):
        pytest.skip('real PROD two-domain input fixture unavailable')
    source=Path(__file__).resolve().parents[2]
    if not (source/'proofs/o1_nlbind/cpu_safe.py').is_file():
        pytest.skip('CPU tridiagonal harness unavailable in this sparse source export')
    env={**os.environ,'JAX_PLATFORMS':'cpu','GPUWRF_FAST_DEFAULTS':'1','GPUWRF_CENSUS':'0',
         'GPUWRF_JAX_CACHE':'0','GPUWRF_NESTED_AOT':'0','GPUWRF_NESTED_FUSE':'0','WRFOUT_ZNT':'1',
         'JAX_ENABLE_X64':'true','JAX_ENABLE_COMPILATION_CACHE':'false'}
    target=tmp_path/'real'
    command=[sys.executable,str(Path(__file__).with_name('real_nested_probe.py')),
             '--source',str(source),'--case',str(case),'--output',str(target)]
    try:result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=1200)
    except subprocess.TimeoutExpired:pytest.fail('real CPU nested forecast did not finish within 20 minutes; no completion claim')
    assert result.returncode==0,result.stdout[-4000:]+result.stderr[-2000:]
    receipt=json.loads((target/'receipt.json').read_text())
    assert receipt['phase']=='PASS' and len(receipt['frames'])==4
    assert all(f['all_values_match_current_carry'] for f in receipt['frames'])
