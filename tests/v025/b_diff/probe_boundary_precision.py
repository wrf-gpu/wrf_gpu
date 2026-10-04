import argparse,sys,json,hashlib,os,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--arm',choices=['native','legacy'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.source/'tests/v025/b_diff'))
import probe_boundary as probe
import jax,jax.numpy as jnp,numpy as np,jaxlib._jax as x
from jax.experimental import topologies
assert jax.devices()[0].platform=='cpu'
target=Path('<USER_HOME>/wrf_gpu2_lanes/compile/phase2/R3_joint/default_import/captured_rtx_target.textproto').read_text()
topo=topologies.get_topology_desc(platform='cuda',target_config=target,topology='1x1x1');assert isinstance(topo.devices[0].client,x.CompileOnlyPyClient)
shard=jax.sharding.SingleDeviceSharding(topo.devices[0])
def abstract(v):return jax.ShapeDtypeStruct(v.shape,v.dtype,weak_type=getattr(v,'weak_type',False),sharding=shard) if hasattr(v,'shape') and hasattr(v,'dtype') else v
bd=probe.boundary
math={'add','subtract','multiply','divide','power','exponential','log','sqrt','rsqrt','maximum','minimum','abs','negate','floor','ceil','round-nearest-even','round-nearest-afz','remainder','reduce','dot'}
result={'arm':a.arm,'source':str(a.source),'boundary_sha256':hashlib.sha256(Path(bd.__file__).read_bytes()).hexdigest(),'runtime':'CPU CUDAhidden +CompileOnlyCUDA target; no compile/execute','scope':'isolated owned boundary entrypoint HLO f64-result arithmetic; Pallas interiors separate BD24 TTIR/PTX zero; unowned scalarconsumer excluded','domains':{}}
a.out.mkdir(exist_ok=True)
for domain in ['d01','d02']:
 state,m,cfg,meta=probe.load(domain,'physical' if domain=='d01' else 'coupled');dt=meta['resolved_dt_s'];nz,ny,nx=state.theta.shape
 lead=jnp.asarray(dt,jnp.float64)
 leafu,leafv,leafph,leafw=[getattr(state,n+'_bdy')[0] for n in ['u','v','ph','w']]
 mu=np.asarray(state.mu_total);u_mu=.5*(np.concatenate([mu[:,:1],mu],1)+np.concatenate([mu,mu[:,-1:]],1));v_mu=.5*(np.concatenate([mu[:1],mu],0)+np.concatenate([mu,mu[-1:]],0))
 massu=jnp.asarray(np.asarray(m.c1h)[:,None,None]*u_mu[None]+np.asarray(m.c2h)[:,None,None]);massv=jnp.asarray(np.asarray(m.c1h)[:,None,None]*v_mu[None]+np.asarray(m.c2h)[:,None,None])
 cases={
 'dry':(lambda s,m,l:probe.dry(s,m,cfg,l,dt,domain=='d02',domain=='d02'),(state,m,lead)),
 'lateral':(lambda s,m,l:bd.apply_lateral_boundaries(s,l,dt,cfg,m),(state,m,lead)),
 'lateral_spec_only':(lambda s,m,l:bd.apply_lateral_boundaries(s,l,dt,cfg,m,dry_spec_only=True),(state,m,lead)),
 'interpolate':(lambda leaf,l:bd.interpolate_boundary_leaf(leaf,l,cfg.update_cadence_s),(state.u_bdy,lead)),
 'specified':(lambda leaf,l:bd.specified_boundary_tendency(leaf,l,cfg.update_cadence_s,z_len=nz,y_len=ny,x_len=nx+1,dtype=jnp.float64,config=cfg),(state.u_bdy,lead)),
 'normal_u':(lambda leaf,save,mass,mapf:bd.normal_bdy_work_target_u(leaf,save,mass,mass,mapf,config=cfg,coupled_boundary_leaves=domain=='d02'),(leafu,state.u,massu,m.msfuy)),
 'normal_v':(lambda leaf,save,mass,mapf:bd.normal_bdy_work_target_v(leaf,save,mass,mass,mapf,config=cfg,coupled_boundary_leaves=domain=='d02'),(leafv,state.v,massv,m.msfvx)),
 'tangential_u':(lambda leaf,save,mass,mapf:bd.tangential_bdy_work_target_u(leaf,save,mass,mass,mapf,config=cfg,coupled_boundary_leaves=domain=='d02'),(leafu,state.u,massu,m.msfuy)),
 'tangential_v':(lambda leaf,save,mass,mapf:bd.tangential_bdy_work_target_v(leaf,save,mass,mass,mapf,config=cfg,coupled_boundary_leaves=domain=='d02'),(leafv,state.v,massv,m.msfvx)),
 'normal_apply':(lambda u,v:bd.apply_normal_bdy_work(u,v,u,v,dt/4,dt,config=cfg),(state.u,state.v)),
 'ph_relax':(lambda ph,leaf,mu,mapf,c1,c2:bd.nested_ph_relax_tendency(ph,leaf,mu,mapf,c1,c2,dt,cfg),(state.ph_perturbation,leafph,state.mu_total,m.msfty,m.c1f,m.c2f)),
 'w_relax':(lambda w,leaf,mu,mapf,c1,c2:bd.nested_w_relax_tendency(w,leaf,mu,mapf,c1,c2,dt,cfg),(state.w,leafw,state.mu_total,m.msfty,m.c1f,m.c2f)),
 'ph_spec_legacy':(lambda ph,mu,c1,c2:bd.spec_bdyupdate_ph_inloop(ph,ph,ph,mu,mu,c1,c2,dt/4,cfg),(state.ph_perturbation,state.mu_total,m.c1f,m.c2f)),
 'ph_spec_live':(lambda ph,mu,c1,c2:bd.spec_bdyupdate_ph_tendency_inloop(ph,ph,ph,ph,mu,mu,c1,c2,dt/4,cfg),(state.ph_perturbation,state.mu_total,m.c1f,m.c2f))}
 if a.arm=='native':cases.pop('ph_relax')  # fails closed under the flag (non-bundle nest, not PROD)
 rows={};result['domains'][domain]=rows
 for name,(fn,args) in cases.items():
  hlo=jax.jit(fn).lower(*jax.tree.map(abstract,args)).compiler_ir('hlo').as_hlo_text();(a.out/(domain+'_'+name+'.hlo')).write_text(hlo)
  bad=[]
  for line in hlo.splitlines():
   match=re.search(r'=\s*f64\[[^\]]*\](?:\{[^}]*\})?\s+([\w-]+)\(',line)
   if match and match[1] in math:bad.append(line)
  rows[name]={'f64_result_math':len(bad),'hlo_sha256':hashlib.sha256(hlo.encode()).hexdigest(),'examples':bad[:3]}
  print(domain,name,len(bad),flush=True)
 (a.out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');jax.clear_caches()
