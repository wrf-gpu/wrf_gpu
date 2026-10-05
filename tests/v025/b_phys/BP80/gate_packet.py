"""Matched pristine Swiss parity, original-source no-snow bytes and kind census."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

root=Path('<USER_HOME>/wrf_gpu2_lanes/b-phys/BP80')
snap=root/'snap'
sys.path.insert(0,str(snap/'src'))
sys.path.insert(0,str(snap/'tests/v025/b_phys'))
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.physics import surface_layer as SL
from gpuwrf.physics.fp32 import surface_layer_real as REAL
from precision_inventory import inventory
import test_sfclay_snow_andreas as T
assert jax.devices()[0].platform=='cpu'
assert SL.__file__.startswith(str(snap))
spec=importlib.util.spec_from_file_location('gpuwrf.physics.bp80_original_sf',root/'baseline_surface_layer.py')
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
rows,dx=T.cases();state=T.view(rows)._replace(dx_m=dx)
snow=jnp.asarray([r['snowh'] for r in rows]).reshape(len(rows),1)
expected=T.oracle(Path('<USER_HOME>/wrf_gpu2_lanes/b-phys/pytest/andreas_oracle0/mynn_oracle'),rows,dx)
packet=dict(schema='BP80-gate-v1',source_commit='13fdd001b',baseline_commit='f807f59e1',
            scope='CPU component pristine fidelity, regression bytes and native kind; no coupled/GPU forecast claim',
            backend='cpu',thresholds=T.THRESHOLDS,
            source_sha256={str(p.relative_to(snap)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                snap/'src/gpuwrf/physics/surface_layer.py',snap/'src/gpuwrf/physics/fp32/surface_layer_real.py',
                snap/'src/gpuwrf/physics/noahmp_coupler.py']},swiss={},canary={},native_kind={})
for native in [False,True]:
    REAL._NATIVE_REAL=native
    dtype=jnp.float32 if native else jnp.float64
    diag=SL.surface_layer_with_diagnostics(state,snowh=snow)
    actual=T.diagnostics_arrays(diag)
    before=T.diagnostics_arrays(old._surface_layer_impl(state,False,dtype))
    metrics={}
    for name,limit in T.THRESHOLDS.items():
        delta=np.abs(actual[name]-expected[name]);rel=delta/np.maximum(np.abs(expected[name]),1.e-12)
        bdelta=np.abs(before[name]-expected[name]);brel=bdelta/np.maximum(np.abs(expected[name]),1.e-12)
        bad=(rel>limit)&(delta>=1.e-6)
        metrics[name]=dict(max_abs=float(delta.max()),relative_max=float(rel.max()),failed_cells=int(bad.sum()),
                           old_failed_cells=int(((brel>limit)&(bdelta>=1.e-6)).sum()))
    assert all(m['failed_cells']==0 for m in metrics.values())
    packet['swiss']['native' if native else 'legacy']=dict(n_columns=len(rows),
        snow_columns=sum(r['snowh']>=.1 for r in rows),bare_thin_columns=sum(r['snowh']<.1 for r in rows),
        nonfinite=sum(int((~np.isfinite(np.asarray(x))).sum()) for x in jax.tree.leaves(diag)),metrics=metrics)
    packet['canary']['native' if native else 'legacy']={}
    arms=json.loads((snap/'tests/v025/b_phys/fixtures/sfclay_snow_canary.json').read_text())['arms']
    for domain,arm in arms.items():
        st=T.view(arm['columns'])._replace(dx_m=arm['dx_m'])
        depth=jnp.zeros_like(st.xland)
        oldfn=jax.jit(lambda x:old._surface_layer_impl(x,False,dtype))
        newfn=jax.jit(lambda x,z:SL.surface_layer_with_diagnostics(x,snowh=z))
        modes={}
        for mode in ['eager','jit']:
            prev=old._surface_layer_impl(st,False,dtype) if mode=='eager' else oldfn(st)
            new=SL.surface_layer_with_diagnostics(st,snowh=depth) if mode=='eager' else newfn(st,depth)
            aa=[np.asarray(x) for x in jax.tree.leaves(prev)]
            bb=[np.asarray(x) for x in jax.tree.leaves(new)]
            same=[a.dtype==b.dtype and a.tobytes()==b.tobytes() for a,b in zip(aa,bb)]
            modes[mode]=dict(leaves=len(aa),byte_exact=sum(same),pass_all=all(same),
                old_output_sha=hashlib.sha256(b''.join(x.tobytes() for x in aa)).hexdigest(),
                new_output_sha=hashlib.sha256(b''.join(x.tobytes() for x in bb)).hexdigest())
            assert all(same),(native,domain,mode,[i for i,v in enumerate(same) if not v])
        packet['canary']['native' if native else 'legacy'][domain]=dict(columns=len(arm['columns']),modes=modes)
        jax.clear_caches()
REAL._NATIVE_REAL=True
for name,fn in [('old',lambda x,z:old._surface_layer_impl(x,False,jnp.float32)),
                ('snow_candidate',lambda x,z:SL.surface_layer_with_diagnostics(x,snowh=z))]:
    hlo=jax.jit(fn).lower(state,snow).compiler_ir('hlo').as_hlo_text()
    _,nodes=inventory(hlo,snap)
    compute=[n for n in nodes if n['category']=='compute']
    packet['native_kind'][name]=dict(f64_compute=len(compute),hlo_sha=hashlib.sha256(hlo.encode()).hexdigest())
    assert not compute
causal=Path('<USER_HOME>/wrf_gpu2_lanes/mass-opus/MO08/andreas_compare.json')
packet['causal_corroboration']=dict(path=str(causal),sha256=hashlib.sha256(causal.read_bytes()).hexdigest(),
    hour1=json.loads(causal.read_text())['1'],authority='mass-opus MO10b pristine-WRF branch-off experiment; independently owned')
packet['status']='CPU_COMPONENT_GREEN'
(root/'gate_packet.json').write_text(json.dumps(packet,indent=2)+'\n')
print(json.dumps(dict(status=packet['status'],swiss={k:dict(columns=v['n_columns'],snow=v['snow_columns'],
    old_failed_fields=[n for n,m in v['metrics'].items() if m['old_failed_cells']]) for k,v in packet['swiss'].items()},
    canary_byte_exact={k:{d:{m:v['pass_all'] for m,v in a['modes'].items()} for d,a in ds.items()} for k,ds in packet['canary'].items()},
    native_kind=packet['native_kind'])),flush=True)
