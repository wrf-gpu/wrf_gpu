"""Real day/night plume regression coverage; NOT WRF fidelity evidence."""
import inspect
import json
from pathlib import Path
import os

os.environ['JAX_PLATFORMS']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['GPUWRF_EDMF_FUSED_PLUME']='0'
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.physics import mynn_edmf as E
from gpuwrf.kernels.phys_mynn_plume import dmp_mf_columns_native

root=Path('<USER_HOME>/wrf_gpu2_lanes/b-phys')
records={}
for period in ('night','day'):
 for domain in ('d01','d02'):
  with np.load(root/f'columns_p0_{period}/{domain}.npz') as data:
   values={k:np.asarray(data[k]) for k in data.files}
  keys=list(inspect.signature(E._dmp_setup).parameters)
  columns=values['thl'].shape[0]
  setup_args={k:jnp.asarray(values[k]) for k in keys if k not in ('dx','psig_shcu')}
  setup_args['psig_shcu']=jnp.ones((columns,),jnp.float64)
  setup=jax.jit(jax.vmap(lambda *a:E._dmp_setup(*a,dx=float(values['dx']))))(
      *[setup_args[k] for k in keys if k!='dx'])
  active=np.asarray(setup['active'])
  idx=np.unique(np.concatenate((np.flatnonzero(active)[:12],np.flatnonzero(~active)[:12],[columns-1])))
  args={k:jnp.asarray(v[idx] if v.ndim else v) for k,v in values.items()}
  with jax.enable_x64(False):
   reference=E.dmp_mf_columns(**{k:v.astype(jnp.float32) for k,v in args.items()})
  # Source WRF:6742 marks dry plume maxmf negative in the native path.
  reference['maxmf']=jnp.where((reference['active']>0) & (jnp.max(reference['edmf_qc'],axis=-1)<1e-8),
                              -reference['maxmf'],reference['maxmf'])
  candidate=dmp_mf_columns_native(**args)
  errors={}
  for name in reference:
   a,b=np.asarray(reference[name]),np.asarray(candidate[name])
   assert np.isfinite(b).all(),(period,domain,name)
   errors[name]=float(np.max(np.abs(a-b))/max(1.0,float(np.max(np.abs(a)))))
  records[f'{period}_{domain}']=dict(columns=len(idx),source_active=int(active[idx].sum()),
      reference_active=int(np.asarray(reference['active']).sum()),
      native_active=int(np.asarray(candidate['active']).sum()),normalized_errors=errors,
      dtypes={k:str(v.dtype) for k,v in candidate.items()})
  # Predeclared fp32 arithmetic regression envelope; independent WRF gate is separate.
  assert max(errors.values())<=1e-4,(period,domain,errors)
path=root/'BP19/regression.json'
path.write_text(json.dumps(dict(scope='native-fp32 JAX reference regression only, not fidelity',
                               envelope=1e-4,records=records),indent=2)+'\n')
print(json.dumps({k:dict(columns=v['columns'],active=v['native_active'],max_error=max(v['normalized_errors'].values())) for k,v in records.items()}))
