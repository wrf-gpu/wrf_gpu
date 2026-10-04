"""Measure the precision hypothesis with unchanged production equations."""
from pathlib import Path
import sys,json,dataclasses
import numpy as np

gate=Path('<USER_HOME>/wrf_gpu2_lanes/host-sync/B36/gate.py').read_text()
scope={'__name__':'__main__'}
exec(compile(gate.split('\ndef error(')[0],'<B36-frozen-fixture-loader>','exec'),scope)
jax,jnp=scope['jax'],scope['jnp']
def real(value):
    return value.astype(jnp.float32) if hasattr(value,'dtype') and np.issubdtype(value.dtype,np.floating) else value
case=scope['cases'][-1]
child=scope['initialize_child_scalar_boundaries'](case['child'])
child=child.replace(_cast=False,**{n:real(getattr(child,n)) for n in child.active_field_names()})
parent=case['parent'].replace(_cast=False,**{n:real(getattr(case['parent'],n)) for n in case['parent'].active_field_names()})
cm,pm=jax.tree.map(real,case['cm']),jax.tree.map(real,case['pm'])
forced=scope['build_child_boundary_package'](child,parent,case['weights'],bdy_width=5,parent_metrics=pm,child_metrics=cm,coupled_forcedown=True,parent_grid_ratio=3,_compiled_producers=True)
cfg=dataclasses.replace(case['namelist'].boundary_config,update_cadence_s=case['cadence'],force_geopotential=False,nested_frozen_wrf_boundary_bundle=True)
nz,ny,nx=child.qv.shape
oracle=scope['Oracle'](scope['consumer_so'],(nz,ny,nx),nested=True,dt=case['dt'],dtbc=0.,dts=case['dt'])
oracle.set('mu',child.mu_total)
for n in ['c1h','c2h']:oracle.set(n,getattr(cm,n))
results=[]
for lead in [0.,case['dt'],case['cadence']/2]:
    actual=scope['nested_scalar_boundary_tendencies'](forced,lead,cm,case['dt'],cfg)
    for i,n in enumerate(scope['SPECIES'],1):
        records=np.asarray(getattr(forced,n+'_bdy'),np.float32)
        oracle.set('scalar',getattr(forced,n));oracle.set('scalar_tend',np.zeros((nz,ny,nx),np.float32))
        oracle.boundary('scalar',records[0],(records[1]-records[0])/np.float32(case['cadence']))
        oracle.s[1]=np.float32(lead)
        oracle.run('relax_bdy_scalar')
        oracle.set('field_tend',oracle.field('scalar_tend',(nz,ny,nx)))
        oracle.run('spec_bdytend',tag=6)
        ref=oracle.field('field_tend',(nz,ny,nx)).astype(np.float64)
        got=np.asarray(actual[i],np.float64)
        diff=got-ref
        rms=float(np.sqrt(np.mean(diff*diff)));mx=float(np.max(np.abs(diff)))
        r=float(np.sqrt(np.mean(ref*ref)));m=float(np.max(np.abs(ref)))
        results.append(dict(species=n,lead=lead,rms=rms,max_abs=mx,rms_ref=r,max_ref=m,passed=rms<=2e-5*max(1.,r) and mx<=2e-4*max(1.,m)))
Path(scope['a'].out).write_text(json.dumps(dict(source=str(scope['a'].source),work='WN3 h18, diagnostic matched REAL4 operands ONLY; production equations unchanged',results=results),indent=2)+'\n')
print(json.dumps(results),flush=True)
