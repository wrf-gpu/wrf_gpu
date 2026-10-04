"""Actual PROD/WN3 operands, independent SINT and pristine REAL4 boundary gates."""
import argparse
import dataclasses
import hashlib
import json
import pickle
from pathlib import Path
import sys
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--skip-restart',action='store_true',help='GPU arm: CPU RestartStore gate already covers record bytes')
a=p.parse_args()
sys.path[:0]=[str(a.source/'src'),str(a.source/'scripts'),str(a.source/'tests/v025/b_diff')]
import gpuwrf
import jax
import jax.numpy as jnp
from netCDF4 import Dataset
from gpuwrf.nesting.boundary_construction import (initialize_child_scalar_boundaries,
    build_child_boundary_package,build_nest_force_weights)
from gpuwrf.coupling.boundary_apply import nested_scalar_boundary_tendencies,BoundaryConfig
from gpuwrf.runtime.domain_tree import DomainTree
from gpuwrf.runtime.checkpoint import write_checkpoint,read_checkpoint_with_runtime_state
from gpuwrf.runtime.operational_mode import _initial_carry_for_run
from v0234_wrf_sint_source_oracle import sint_full,wrf_sides
from v0234_wrf_scalar_boundary_oracle import scalar_boundary_tendency
from pristine_boundary import Oracle,build as build_consumer
from pristine_forcedown import forcedown,build as build_force
from real4_source_oracle import tendency as real4_tendency
from gpuwrf.runtime.restart_store import RestartStore
assert jax.devices()[0].platform=='cpu'
SPECIES=('qc','qr','qi','qs','qg','Ni','Nr')
FIELDS=dict(qc='QCLOUD',qr='QRAIN',qi='QICE',qs='QSNOW',qg='QGRAUP',Ni='QNICE',Nr='QNRAIN')
ART=Path('<USER_HOME>/wrf_gpu2_lanes/host-sync/B36')
force_so=build_force(ART/'oracle/force')
consumer_so=build_consumer(ART/'oracle/consumer')
cases=[]
field_inputs=[]


def weather_state(state,file):
    with Dataset(file) as nc:
        raw={s:np.asarray(nc.variables[v][0]) for s,v in FIELDS.items()}
        raw.update(mu_perturbation=np.asarray(nc.variables['MU'][0]),mub=np.asarray(nc.variables['MUB'][0]))
    field_inputs.append(dict(path=str(file),fields={n:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for n,v in raw.items()}))
    updates={n:jnp.asarray(raw[n],dtype=getattr(state,n).dtype) for n in SPECIES}
    mu=jnp.asarray(raw['mu_perturbation'],dtype=state.mu_total.dtype)
    updates.update(mu_total=mu+jnp.asarray(raw['mub'],dtype=mu.dtype),mu_perturbation=mu)
    return state.replace(**updates)
inputs=Path('<USER_HOME>/wrf_gpu2_lanes/integrate/diagnose/common_inputs.pkl')
with inputs.open('rb') as f:h,b,_,_,dts,_=pickle.load(f)
edge=DomainTree.from_domains(h,b).edges['d01'][0]
cpu=Path('<DATA_ROOT>/alisios/runs/20260725_18z/wn2_wrf/alisios_operational_d01_121x71_d02_268x118_oldgrid_v1/attempt_001/cpu_case')
states=[];input_paths=[inputs]
for name in ['d01','d02']:
    valid='2026-07-26_01:00:18' if name=='d01' else '2026-07-26_01:00:00'
    file=cpu/f'wrfout_{name}_{valid}'
    assert file.is_file(),file
    input_paths.append(file)
    states.append(weather_state(b[name].state,file))
cases.append(dict(name='PROD-d02',parent=states[0],child=states[1],pm=b['d01'].namelist.metrics,
    cm=b['d02'].namelist.metrics,weights=edge.weights,start=(edge.spec.i_parent_start,edge.spec.j_parent_start),ratio=3,
    dt=dts['d02'],cadence=dts['d01'],grid=b['d02'].grid,namelist=b['d02'].namelist))
paths=[Path('<USER_HOME>/wrf_gpu2_lanes/wn3/W3/ni_probe/rain01')/f'd0{n}_first_exceed.pkl' for n in [2,3]]
payload=[]
for file in paths:
    with file.open('rb') as f:payload.append(pickle.load(f))
input_paths+=paths
pw=build_nest_force_weights(parent_grid_ratio=3,i_parent_start=92,j_parent_start=36,parent_grid=payload[0]['grid'],child_grid=payload[1]['grid'])
cases.append(dict(name='WN3-d03',parent=payload[0]['before'].state,child=payload[1]['before'].state,
    pm=payload[0]['namelist'].metrics,cm=payload[1]['namelist'].metrics,weights=pw,start=(92,36),ratio=3,
    dt=payload[1]['namelist'].dt_s,cadence=payload[0]['namelist'].dt_s,grid=payload[1]['grid'],namelist=payload[1]['namelist']))
prod12=dict(cases[0],name='PROD-d02-CPU-h12')
for role,name in [('parent','d01'),('child','d02')]:
    file=cpu/f'wrfout_{name}_2026-07-26_12:00:00'
    prod12[role]=weather_state(b[name].state,file)
    input_paths.append(file)
cases.append(prod12)
wn3cpu=Path('<USER_HOME>/wrf_gpu2_lanes/wn3/cases/20260227_18z_a1/wrfinput_d03').resolve().parent
winter=dict(cases[1],name='WN3-d03-CPU-h18')
for role,name,template in [('parent','d02',payload[0]),('child','d03',payload[1])]:
    file=wn3cpu/f'wrfout_{name}_2026-02-28_18:00:00'
    winter[role]=weather_state(template['before'].state,file)
    input_paths.append(file)
cases.append(winter)


def error(ref,got):
    ref=np.asarray(ref,dtype=np.float64);got=np.asarray(got,dtype=np.float64)
    diff=got-ref
    rms=float(np.sqrt(np.mean(diff*diff)));rms_ref=float(np.sqrt(np.mean(ref*ref)))
    mx=float(np.max(np.abs(diff)));mx_ref=float(np.max(np.abs(ref)))
    return dict(rms=rms,max_abs=mx,rms_ref=rms_ref,max_ref=mx_ref,
        passed=bool(np.isfinite(got).all() and rms<=2e-5*max(1.,rms_ref) and mx<=2e-4*max(1.,mx_ref)))


results=[]
for case in cases:
    child=initialize_child_scalar_boundaries(case['child'])
    assert all(getattr(child,n+'_bdy').shape[0]==2 for n in SPECIES)
    assert initialize_child_scalar_boundaries(child) is child
    # Match _load_domains: precision enforcement occurs before first forcedown.
    child=_initial_carry_for_run(child,case['namelist']).state
    parent_namelist=(b['d01'].namelist if case['name'].startswith('PROD')
                     else payload[0]['namelist'])
    case['parent']=_initial_carry_for_run(case['parent'],parent_namelist).state
    cm,pm=case['cm'],case['pm']
    expected_mass=lambda state,m: np.asarray(m.c1h,np.float32)[:,None,None]*np.asarray(state.mu_total,np.float32)[None]+np.asarray(m.c2h,np.float32)[:,None,None]
    pmass=expected_mass(case['parent'],pm);cmass=expected_mass(child,cm)
    forced=build_child_boundary_package(child,case['parent'],case['weights'],
        bdy_width=5,parent_metrics=pm,child_metrics=cm,coupled_forcedown=True,parent_grid_ratio=3,_compiled_producers=True)
    jax.block_until_ready(forced)
    nz,ny,nx=child.qv.shape;side_len=forced.qv_bdy.shape[-1]
    cfg=dataclasses.replace(case['namelist'].boundary_config,update_cadence_s=float(case['cadence']),
        force_geopotential=False,nested_frozen_wrf_boundary_bundle=True)
    result=dict(name=case['name'],shape=[nz,ny,nx],start=case['start'],dt=case['dt'],cadence=case['cadence'],
        spec_zone=cfg.spec_zone,relax_zone=cfg.relax_zone,producer=[],consumer=[],restart=[],mutations=[])
    for n in SPECIES:
        coarse=np.asarray(getattr(case['parent'],n),np.float32)*pmass
        fine=sint_full(coarse,ratio=3,i_parent_start=case['start'][0],j_parent_start=case['start'][1],child_ny=ny,child_nx=nx)
        records=np.asarray(getattr(forced,n+'_bdy'))
        expected=np.stack([wrf_sides(np.asarray(getattr(child,n),np.float32)*cmass,width=5,side_len=side_len),wrf_sides(fine,width=5,side_len=side_len)])
        row={'species':n,'source_literal_exact':bool(np.allclose(records,expected,rtol=2e-7,atol=2e-7)),
             'parent_active_cells':int(np.count_nonzero(coarse)),'child_active_cells':int(np.count_nonzero(getattr(child,n)))}
        old,rate=forcedown(force_so,coarse,np.asarray(getattr(child,n),np.float32)*cmass,start=case['start'],ratio=3,cadence=case['cadence'],side_len=side_len)
        row['pristine_value']=error(old,records[0])
        difference=records[1].astype(np.float32)-records[0].astype(np.float32)
        row['pristine_rate']=error(rate,(difference.astype(np.float64)*(1.0/float(case['cadence']))).astype(np.float32))
        result['producer'].append(row)
    oracle=Oracle(consumer_so,(nz,ny,nx),nested=True,dt=case['dt'],dtbc=0.,dts=case['dt'])
    # relax_bdy_scalar's formal mu receives grid%mut at the solve_em call site.
    oracle.set('mu',child.mu_total)
    for name in ['c1h','c2h']:oracle.set(name,getattr(cm,name))
    for lead in [0.,float(case['dt']),float(case['cadence'])/2]:
        actual=nested_scalar_boundary_tendencies(forced,lead,cm,float(case['dt']),cfg)
        for index,n in enumerate(SPECIES,1):
            records=np.asarray(getattr(forced,n+'_bdy'))
            expected=real4_tendency(np.asarray(getattr(forced,n)),np.asarray(forced.mu_total),np.asarray(cm.c1h),np.asarray(cm.c2h),records,
                lead_seconds=lead,cadence_s=case['cadence'],dt_full=case['dt'],spec_zone=cfg.spec_zone,relax_zone=cfg.relax_zone,spec_exp=cfg.spec_exp)
            oracle.set('scalar',getattr(forced,n));oracle.set('scalar_tend',np.zeros((nz,ny,nx),np.float32))
            rec32=records.astype(np.float32)
            rate32=((rec32[1]-rec32[0]).astype(np.float64)*(1.0/float(case['cadence']))).astype(np.float32)
            oracle.boundary('scalar',rec32[0],rate32)
            oracle.s[1]=np.float32(lead)
            oracle.run('relax_bdy_scalar')
            # solve_em passes the SAME moist/scalar tendency to both routines.
            # The generic extracted adapter names the spec argument field_tend.
            oracle.set('field_tend',oracle.field('scalar_tend',(nz,ny,nx)))
            oracle.run('spec_bdytend',tag=6)
            ref=oracle.field('field_tend',(nz,ny,nx))
            row=dict(species=n,lead=lead,source_literal_exact=bool(np.allclose(actual[index],expected,rtol=2e-7,atol=2e-7)),pristine=error(ref,actual[index]))
            result['consumer'].append(row)
        if lead==0.:
            for index,n in enumerate(SPECIES,1):
                maximum=float(np.max(np.abs(np.asarray(actual[index]))))
                missing=forced.replace(**{n+'_bdy':None})
                mutant=nested_scalar_boundary_tendencies(missing,lead,cm,float(case['dt']),cfg)[index]
                result['mutations'].append(dict(species=n,active_tendency=maximum>0.,deletion_changes_result=bool(np.any(np.asarray(mutant)!=np.asarray(actual[index])))))
    results.append(result)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(source=str(a.source),backend='cpu',inputs=[str(p) for p in input_paths],field_inputs=field_inputs,results=results),indent=2)+'\n')
    if a.skip_restart:
        print(json.dumps(dict(name=result['name'],producer_pass=sum(r['source_literal_exact'] and r['pristine_value']['passed'] and r['pristine_rate']['passed'] for r in result['producer']),consumer_pass=sum(r['source_literal_exact'] and r['pristine']['passed'] for r in result['consumer']),restart='skipped',mutations=result['mutations'])),flush=True)
        continue
    results.pop()
    carry=_initial_carry_for_run(forced,case['namelist'])
    store=RestartStore(ART/('restart-'+case['name']),2_000_000_000,1,0)
    identity=dict(case=case['name'],purpose='B36 actual operational carry oracle')
    generation,receipt=store.save({'child':carry},{'child':0},{'child':float(case['dt'])},identity)
    restored,_=store.read(generation,expected_identity=identity)
    restored_carry=restored['carries']['child']
    for n in SPECIES:
        x=np.asarray(getattr(carry.state,n+'_bdy'))
        z=np.asarray(getattr(restored_carry.state,n+'_bdy'))
        result['restart'].append(dict(species=n,exact=x.shape==z.shape and x.dtype==z.dtype and x.tobytes()==z.tobytes()))
    result['restart_receipt']=receipt
    results.append(result)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(source=str(a.source),backend='cpu',inputs=[str(p) for p in input_paths],field_inputs=field_inputs,results=results),indent=2)+'\n')
    print(json.dumps(dict(name=result['name'],producer_pass=sum(r['source_literal_exact'] and r['pristine_value']['passed'] and r['pristine_rate']['passed'] for r in result['producer']),consumer_pass=sum(r['source_literal_exact'] and r['pristine']['passed'] for r in result['consumer']),restart_pass=sum(r['exact'] for r in result['restart']),mutations=result['mutations'])),flush=True)
assert all(x['source_literal_exact'] and x['pristine_value']['passed'] and x['pristine_rate']['passed'] for r in results for x in r['producer'])
assert all(x['source_literal_exact'] and x['pristine']['passed'] for r in results for x in r['consumer'])
assert all(x['exact'] for r in results for x in r['restart'])
assert all(x['deletion_changes_result'] for r in results for x in r['mutations'] if x['active_tendency'])
