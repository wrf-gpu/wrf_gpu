"""Actual GPU B36 oracle plus matched force-down component cost."""
from pathlib import Path
import hashlib,json,os,time,statistics

HERE=Path(__file__).resolve().parent
gate=(HERE/'gate.py').read_text()
gate=gate.replace("assert jax.devices()[0].platform=='cpu'", "assert os.environ.get('GPUWRF_GPU_LOCK_HELD') == '1'\nassert jax.devices()[0].platform=='gpu'")
gate=gate.replace("backend='cpu'", "backend='gpu'")
# The same independent Fortran/source gates run in the measured GPU process.
scope={'__name__':'__main__','os':os}
gate_error=None
try:
    exec(compile(gate,str(HERE/'gate.py'),'exec'),scope)
except AssertionError as exc:  # verdict JSON is already written; still measure cost
    gate_error=repr(exc)
jax=scope['jax']
assert jax.devices()[0].platform=='gpu'
cost=[]
for case in scope['cases']:
    child=scope['initialize_child_scalar_boundaries'](case['child'])
    child=scope['_initial_carry_for_run'](child,case['namelist']).state
    parent=case['parent']
    old=child.replace(**{n+'_bdy':None for n in scope['SPECIES']})
    kwargs=dict(bdy_width=5,parent_metrics=case['pm'],child_metrics=case['cm'],coupled_forcedown=True,parent_grid_ratio=3,_compiled_producers=True)
    call=lambda state:scope['build_child_boundary_package'](state,parent,case['weights'],**kwargs)
    jax.block_until_ready(call(old));jax.block_until_ready(call(child))
    before=[];after=[]
    for i in range(10):
        order=[(old,before),(child,after)] if i%2==0 else [(child,after),(old,before)]
        for state,values in order:
            t=time.perf_counter();out=call(state);jax.block_until_ready(out);values.append(time.perf_counter()-t)
    cost.append(dict(case=case['name'],before_median_s=statistics.median(before),after_median_s=statistics.median(after),before_samples=before,after_samples=after,work_difference='adds7activeWRFscalarrecordfamilies; originaldry/QVworksame'))
result=dict(gate_error=gate_error,platform=jax.devices()[0].platform,affinity=sorted(os.sched_getaffinity(0)),omp=os.environ.get('OMP_NUM_THREADS'),cost=cost,oracle=str(scope['a'].out),source=str(scope['a'].source),harness_sha256=hashlib.sha256(gate.encode()).hexdigest())
(scope['a'].out.parent/'force_cost.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
assert gate_error is None,gate_error
