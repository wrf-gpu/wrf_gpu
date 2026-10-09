"""Actual native KF adapter plus actual production cadence/held-rate consumer.

Only the Pallas execution mode is changed to CPU interpret. Each refresh is
computed by the real adapter, then supplied verbatim to the production cadence
function to observe its pre-clear rates without repeating the expensive kernel.
Thermodynamic State is imposed from the same original snapshot each own step;
KF carry alone evolves. This is a seam replay, not a coupled forecast.
"""
from pathlib import Path
import dataclasses, json, os, sys, time
from types import SimpleNamespace
import argparse
p=argparse.ArgumentParser()
p.add_argument('--fixture',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--lead',type=int,required=True)
args=p.parse_args()
ROOT=args.fixture
SOURCE=Path(__file__).resolve().parents[3]
args.output.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(SOURCE/'src'))
sys.path.insert(0,str(SOURCE/'scripts/v025'))
os.environ.update(JAX_PLATFORMS='cpu',GPUWRF_FAST_DEFAULTS='0',GPUWRF_JAX_CACHE='0',
    GPUWRF_KF_COLUMN_FP32='1',GPUWRF_KF_RESIDENT_TABLES='1',GPUWRF_CARRY_REAL_ALL='1',GPUWRF_DYN_REAL_ALL='1')
from real_state import load_real_snapshot
import jax,jax.numpy as jnp,numpy as np
from gpuwrf.contracts import state as S
from gpuwrf.kernels import phys_kf_column as K
from gpuwrf.coupling import scan_adapters as A
from gpuwrf.runtime import operational_mode as op
from gpuwrf.diagnostics import census
from gpuwrf.kernels.dyn_carry_fp32 import real_state
assert jax.devices()[0].platform=='cpu'
S._gpu_device=lambda:jax.devices('cpu')[0]
original_kernel=K.kf_column
K.kf_column=lambda *a,**kw:original_kernel(*a,**kw,interpret=True)
original_adapter=A.kf_adapter
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Carry:
    cumulus_carry:object
    cumulus_tendencies:object
    census:object
    def replace(self,**kw):return dataclasses.replace(self,**kw)

manifest=json.loads((ROOT/'MANIFEST.json').read_text())
fields=('RTHCUTEN','RQVCUTEN','RQCCUTEN','RQRCUTEN','RQICUTEN','RQSCUTEN')
cols=lambda a:np.moveaxis(np.asarray(a),0,-1).reshape(-1,np.asarray(a).shape[0])
for row in [r for r in manifest['cases'] if r['lead']==args.lead]:
    if Path('/tmp/wrf_gpu2_quiet').exists():raise SystemExit('QUIET: defer CPU replay')
    lead=row['lead'];warm=np.load(ROOT/f'tau{lead}.npz')
    stamp=Path(row['restart']).name.removeprefix('wrfrst_d01_')
    snapshot=load_real_snapshot(Path(row['restart']).parent,domain='d01',wrfout_name='wrfout_d01_'+stamp)
    state=real_state(snapshot.state)
    # Prefer original current restart prognostics over the coincident history.
    with __import__('netCDF4').Dataset(row['restart']) as ds:
        ds.set_auto_mask(False)
        read=lambda n:jnp.asarray(ds[n][0],jnp.float32)
        state=state.replace(_cast=False,theta=read('THM_2')+jnp.float32(300),p=read('P')+read('PB'),
            p_perturbation=read('P'),ph=read('PH_2')+read('PHB'),u=read('U_2'),v=read('V_2'),w=read('W_2'),
            qv=read('QVAPOR'),qc=read('QCLOUD'),qr=read('QRAIN'),qi=read('QICE'),qs=read('QSNOW'))
    grid=SimpleNamespace(metrics=snapshot.metrics,projection=SimpleNamespace(dx_m=row['dx']))
    nml=SimpleNamespace(dt_s=row['dt'],cumulus_cadence_steps=row['stepcu'],cudt_minutes=row['cudt_minutes'],grid=grid)
    back=lambda x:jnp.asarray(np.moveaxis(np.asarray(x).reshape(70,120,-1),-1,0),jnp.float32)
    carry=Carry((back(warm['W0AVG']),jnp.asarray(warm['NCA'].reshape(70,120))),
        tuple(back(warm[f]) for f in fields)+(jnp.asarray(warm['PRATEC'].reshape(70,120)),),census.initial_census())
    prepare=dict(T=cols(A._temperature_from_theta(A._dry_theta_view(state),state.p)),
        QV=cols(state.qv),P=cols(state.p),DZ=cols(jnp.maximum(state.ph[1:]/A.GRAVITY_M_S2-state.ph[:-1]/A.GRAVITY_M_S2,1.)),
        RHO=cols(A._wrf_phy_prep_rho_from_state(state,grid.metrics,output_dtype=jnp.float32)),
        W=cols(state.w),U=cols(A._u_mass(state)),V=cols(A._v_mass(state)))
    prep_diff={n:float(np.max(np.abs(a-warm[n]))) for n,a in prepare.items()}
    (args.output/f'tau{lead}_caller.json').write_text(json.dumps(prep_diff,indent=2)+'\n')
    print('CALLER',lead,prep_diff,flush=True)
    adapter=jax.jit(lambda st,w,n,held:original_adapter(st,row['dt'],w,n,grid=grid,
        stepcu=row['stepcu'],cudt=row['cudt_minutes'],held_tendencies=held,return_tendencies=True))
    outputs=[];calls=[];started=time.monotonic()
    for index in range(1,9):
        if Path('/tmp/wrf_gpu2_quiet').exists():raise SystemExit('QUIET: defer CPU replay')
        step=row['step']+index;run=step==1 or step%row['stepcu']==0
        if run:
            values=jax.block_until_ready(adapter(state,*carry.cumulus_carry,carry.cumulus_tendencies))
            used,wa,nn=values
            op.kf_adapter=lambda *args,_values=values,**kw:_values
            calls.append(step)
        else:
            used=carry.cumulus_tendencies;wa,nn=carry.cumulus_carry
            # The unselected traced refresh remains shape/dtype safe.
            op.kf_adapter=lambda *args,_used=used,_wa=wa,_nn=nn,**kw:(_used,_wa,_nn)
        apply=jax.jit(lambda st,cy:op._kf_cadence_step(st,cy,nml,jnp.asarray(step)))
        updated,newcarry=jax.block_until_ready(apply(state,carry))
        clear=(np.asarray(nn)>0)&(np.floor(np.asarray(nn)/row['dt']+.5)<=1)
        zeros=np.zeros((8400,),np.float32)
        record=np.concatenate([*(cols(a) for a in used[:6]),cols(newcarry.cumulus_carry[0]),
            np.asarray(newcarry.cumulus_carry[1]).reshape(-1,1),
            (np.asarray(used[6])*np.float32(row['dt'])).reshape(-1,1),np.asarray(used[6]).reshape(-1,1),
            *([zeros[:,None]]*4),clear.reshape(-1,1).astype(np.float32),np.full((8400,1),float(run),np.float32)],axis=1)
        assert record.shape==(8400,317),record.shape
        assert np.isfinite(record).all()
        # Verify actual returned carry, including clear timing, against rates used.
        for consumed,held in zip(used[:6],newcarry.cumulus_tendencies[:6]):
            assert np.array_equal(np.asarray(held),np.where(clear[None],0,np.asarray(consumed)))
        outputs.append(record.astype(np.float32));carry=newcarry
        print('STEP',lead,step,'refresh',run,'clear',int(clear.sum()),flush=True)
    np.savez(args.output/f'tau{lead}_port.npz',records=np.stack(outputs,axis=1))
    info=dict(lead=lead,backend='cpu',source=str(SOURCE),calls=calls,steps=8,
        native='actual kf_adapter -> step_kf_column -> kf_column (CPU interpreter)',
        cadence='actual operational_mode._kf_cadence_step; refresh response precomputed once by actual adapter and supplied verbatim',
        scope='frozen original thermodynamic state imposed each step; warm KF carry evolves; not coupled trajectory',
        all_finite=True,elapsed_s=time.monotonic()-started,timing_claim=False)
    (args.output/f'tau{lead}_port_info.json').write_text(json.dumps(info,indent=2)+'\n')
    print('DONE',lead,flush=True)
