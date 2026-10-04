"""Lower the GPU column IR on CPU; no CUDA backend or GPU execution.

Uses JAX's private Triton lowering for installed JAX 0.10. This compiler
preflight catches lowering failures before spending a shared GPU lease.
"""
import argparse
import importlib.util
from pathlib import Path
import json
import jax
import jax.numpy as jnp
import numpy as np
from jax._src.pallas.triton import lowering
from gpuwrf.physics import thompson_column as tc


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--out',type=Path,required=True)
 parser.add_argument('--events',action='store_true')
 parser.add_argument('--mutants',action='store_true')
 args=parser.parse_args()
 assert jax.devices()[0].platform=='cpu',jax.devices()
 x=np.load('<USER_HOME>/wrf_gpu2_lanes/b-thompson/columns_p0/day_d01.npz')
 state=tc.ThompsonColumnState(**{k:jnp.asarray(x[k].reshape(-1,44)[:1]) for k in tc.ThompsonColumnState.__slots__})
 if args.events:
  from gpuwrf.kernels.phys_thompson_full import full_column
  call=lambda s:full_column(s,54.,return_events=True)
 else:
  call=lambda s:tc._step_thompson_column_full_impl(s,54.,False)
 args.out.parent.mkdir(parents=True,exist_ok=True)
 cases=[('canonical',None)]
 if args.mutants:
  from gpuwrf.coupling import physics_couplers as pc
  path=Path(__file__).parent/'oracle_reference/p0c_ocol_runner.py'
  spec=importlib.util.spec_from_file_location('frozen_mutant_compile',path)
  runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
  for name in ('gate_off','rev0_ungated','all_lqc','post_branch','no_ksed','post_rho'):
   cases.append((name,runner.patch_for(name,pc,tc,jnp)))
 results=[]
 for name,patch in cases:
  if patch:
   module,attribute,fn=patch;original=getattr(module,attribute);setattr(module,attribute,fn)
  try:
   jax.clear_caches()
   jp=jax.make_jaxpr(call)(state).jaxpr
   params=next(e.params for e in jp.eqns if e.primitive.name=='pallas_call')
   result=lowering.lower_jaxpr_to_triton_module(params['jaxpr'],params['grid_mapping'],'cuda',120)
   asm=str(result.module)
   if name=='canonical':args.out.with_suffix('.ttir').write_text(asm)
   results.append({'case':name,'lowering':'PASS','IR_characters':len(asm)})
   print('CPU Triton lowering PASS',name,len(asm),flush=True)
  finally:
   if patch:setattr(module,attribute,original)
 args.out.write_text(json.dumps({'backend':'cpu','GPU_execution':False,'target':'cuda/sm120','results':results})+'\n')


if __name__=='__main__':
 main()
