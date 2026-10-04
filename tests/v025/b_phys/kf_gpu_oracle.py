"""Run the unmodified frozen WRF KF gate on the GPU, without pytest hooks."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',required=True,type=Path)
args=parser.parse_args()
if os.environ.get('GPUWRF_GPU_LOCK_HELD')!='1':
    raise RuntimeError('requires GPU lock')
os.environ['JAX_PLATFORMS']='cuda'
os.environ['GPUWRF_KF_RESIDENT_TABLES']='1'
os.environ['GPUWRF_WRITE_PROOFS']='1'
import jax
assert jax.devices()[0].platform == 'gpu'
if jax.default_backend() not in ('gpu','cuda'):
    raise RuntimeError('GPU oracle must not fall back to CPU')
path=Path(__file__).resolve().parents[2]/'test_kf_cumulus_oracle.py'
spec=importlib.util.spec_from_file_location('b_phys_gpu_kf_oracle',path)
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
args.output.parent.mkdir(parents=True,exist_ok=True)
oracle.JAX_PROOF=str(args.output)
oracle.test_jax_vs_oracle()
proof=json.loads(args.output.read_text())
proof['actual_backend']=jax.default_backend()
proof['actual_device']=str(jax.devices()[0])
args.output.write_text(json.dumps(proof,indent=2)+'\n')
print(proof['verdict'],proof['actual_backend'],len(proof['cases']))
