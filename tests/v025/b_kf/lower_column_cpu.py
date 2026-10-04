"""CPU-only frontend lowering; no GPU backend lookup, no device execution."""
from pathlib import Path
import argparse,json,sys,time,traceback
p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
p.add_argument('--warm-rain',action='store_true');p.add_argument('--no-qi',action='store_true');p.add_argument('--no-qs',action='store_true');args=p.parse_args()
sys.path.insert(0,str(args.source/'src'))
import jax,jax.numpy as jnp
from gpuwrf._x64_config import configure_jax_x64
from gpuwrf.kernels.phys_kf_column import kf_column
from jax._src.pallas.triton import lowering
configure_jax_x64()
assert {d.platform for d in jax.devices()}=={'cpu'}
root=args.output;root.mkdir(parents=True,exist_ok=True)
data=[json.loads((args.source/f'proofs/v060/savepoints/kf_case_{cid}.json').read_text()) for cid in range(1,6)]
a=[jnp.asarray([d['columns'][key] for d in data],jnp.float32) for key in ('T','QV','P','DZ','RHO','W0AVG','U','V')]
options=dict(warm_rain=args.warm_rain,f_qi=not args.no_qi,f_qs=not args.no_qs)
closed=jax.make_jaxpr(jax.vmap(lambda *a:kf_column(*a,54.,9000.,40,**options)))(*a)
params=next(e.params for e in closed.jaxpr.eqns if str(e.primitive)=='pallas_call')
start=time.perf_counter();record=dict(source=str(args.source),platform=jax.default_backend(),target_sm=120,gpu_execution=False,options=options,scope='CPU-only Triton IR frontend lowering; not a GPU compile or runtime receipt')
try:
    result=lowering.lower_jaxpr_to_triton_module(params['jaxpr'],params['grid_mapping'],'cuda',120)
    (root/'kernel.ttir').write_text(result.module.operation.get_asm())
    record['verdict']='PASS';record['grid']=result.grid
except Exception:
    record['verdict']='ERROR';record['error']=traceback.format_exc();print(record['error'])
record['elapsed_s']=time.perf_counter()-start
(root/'ttir_receipt.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='error'}),flush=True)
sys.exit(0 if record['verdict']=='PASS' else 1)
