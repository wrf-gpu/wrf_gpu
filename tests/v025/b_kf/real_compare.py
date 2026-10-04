"""Compare column outputs with pristine WRF real day/night truth.

The binary fixture layout and source checksums are in real_fixture.yaml.
"""
import argparse,json,sys,time,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--period',choices=('day','night'),required=True);p.add_argument('--interpret',action='store_true');p.add_argument('--outputs',type=Path);args=p.parse_args()
sys.path.insert(0,str(args.source/'src'))
import jax,jax.numpy as jnp,numpy as np
from gpuwrf._x64_config import configure_jax_x64
configure_jax_x64()
if not args.outputs and not args.interpret:
    assert jax.devices()[0].platform == 'gpu'
    assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'
root=Path('<USER_HOME>/wrf_gpu2_lanes/b-kf/real_oracle')
expected=np.fromfile(root/f'{args.period}_outputs.bin',dtype='<f4').reshape(-1,271)
fields=('RTHCUTEN','RQVCUTEN','RQCCUTEN','RQRCUTEN','RQICUTEN','RQSCUTEN')
scalars=('RAINCV','PRATEC','NCA','CUTOP','CUBOT','ISHALL','TIMEC')
start=time.perf_counter()
if args.outputs:
    data=np.load(args.outputs)
    actual={key:data[key.lower()] for key in fields+scalars}
else:
    from gpuwrf.kernels.phys_kf_column import kf_column
    inputs=np.fromfile(root/f'{args.period}_inputs.bin',dtype='<f4').reshape(-1,8,44)
    one=lambda *a:kf_column(*a,54.,9000.,44,interpret=args.interpret)
    fn=jax.jit(jax.vmap(one))
    actual={key:np.asarray(value) for key,value in jax.block_until_ready(fn(*(jnp.asarray(inputs[:,i,:]) for i in range(8)))).items()}
record=dict(period=args.period,backend=jax.default_backend(),interpret=args.interpret,columns=len(expected),fields={},scalars={},failures=[],elapsed_s=time.perf_counter()-start)
for i,key in enumerate(fields):
    target=expected[:,i*44:(i+1)*44].astype(np.float64)
    error=np.max(np.abs(actual[key]-target),axis=1)
    scale=np.maximum(np.max(np.abs(target),axis=1),1e-9)
    fail=(error>1e-9)&(error>0.002*scale)
    record['fields'][key]=dict(max_abs=float(np.max(error)),max_rel=float(np.max(error/scale)),failed_columns=np.flatnonzero(fail).tolist())
    if fail.any():record['failures'].append(key)
for i,key in enumerate(scalars):
    target=expected[:,264+i].astype(np.float64);error=np.abs(actual[key]-target)
    if key=='RAINCV': fail=error>np.maximum(.0001,.003*np.abs(target))
    elif key in ('CUTOP','CUBOT'): fail=error>=.5
    elif key=='ISHALL': fail=error!=0
    elif key in ('NCA','TIMEC'): fail=error>1e-6
    else: fail=np.zeros(len(error),bool) # diagnostics disclosed, no invented tolerance
    record['scalars'][key]=dict(max_abs=float(np.max(error)),failed_columns=np.flatnonzero(fail).tolist())
    if fail.any():record['failures'].append(key)
record['nonfinite']=int(sum(np.count_nonzero(~np.isfinite(a)) for a in actual.values()))
record['verdict']='PASS' if not record['failures'] and record['nonfinite']==0 else 'FAIL'
args.output.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
sys.exit(0 if record['verdict']=='PASS' else 1)
