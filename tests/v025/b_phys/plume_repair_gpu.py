"""Direct asserted-GPU plume WRF gates, including an active limiter."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import jax
import numpy as np
from gpuwrf.kernels.phys_mynn_plume import dmp_mf_columns_native
from gpuwrf.physics import mynn_edmf as E

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',required=True,type=Path)
args=parser.parse_args()
assert os.environ.get('GPUWRF_GPU_LOCK_HELD')=='1'
assert jax.devices()[0].platform=='gpu'
path=Path(__file__).parent/'fixtures/limiter_active_wrf.json'
fixture=json.loads(path.read_text())
result=dmp_mf_columns_native(**{key:np.asarray(value) for key,value in fixture['inputs'].items()},interpret=False)
assert float(result['active'][0])==1.0
errors={}
for name,values in fixture['expected'].items():
 actual=np.asarray(result[name][0]);expected=np.asarray(values)
 assert actual.dtype==np.float32 and np.isfinite(actual).all()
 error=float(np.max(np.abs(actual-expected))/max(1e-30,np.max(np.abs(expected))))
 assert error<=fixture['tolerance'],(name,error)
 errors[name]=error
path=Path(__file__).resolve().parents[2]/'test_mynn_edmf_oracle.py'
spec=importlib.util.spec_from_file_location('historical_d03_oracle',path)
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
E.dmp_mf_columns=lambda *a,**kw:dmp_mf_columns_native(*a,interpret=False,**kw)
column=json.loads(Path(oracle.COL).read_text())
oracle.test_dmp_mf_matches_wrf_oracle(column)
receipt=dict(device_platform=jax.devices()[0].platform,backend=jax.default_backend(),
             historical_d03_wrf_pass=True,limiter_stress_wrf_pass=True,
             limiter_adjustment=fixture['adjustment'],limiter_relative_errors=errors,
             tolerance=fixture['tolerance'],scope=fixture['scope'])
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
