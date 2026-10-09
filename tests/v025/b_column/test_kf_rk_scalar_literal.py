"""KF moist_tend RK scalar consumers on original real d01 operands.

WRF's unmodified rk_update_scalar/rk_update_scalar_pd are the reference.
The actual KF source is coupled and add_a2a-masked before those consumers.
"""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import os

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling import kf_rk as K
from gpuwrf.runtime import operational_mode as op

ROOT=Path(__file__).resolve().parents[3]
SPECIES={'qv':('QVAPOR','RQVCUTEN'),'qc':('QCLOUD','RQCCUTEN'),'qr':('QRAIN','RQRCUTEN'),
         'qi':('QICE','RQICUTEN'),'qs':('QSNOW','RQSCUTEN')}

class State(SimpleNamespace):
    def replace(self,**kw):return State(**(vars(self)|kw))

@pytest.fixture(scope='module')
def original(tmp_path_factory):
    folder=os.environ.get('GPUWRF_KF_RK_REAL_CASE')
    if folder is None:pytest.skip('requires original P0227 KF source operands')
    path=Path(folder)/'wrfrst_d01_2026-03-01_00:00:00'
    from netCDF4 import Dataset
    with Dataset(path) as d:
        d.set_auto_mask(False)
        read=lambda n:np.asarray(d[n][0],np.float32)
        crop=lambda a:a[:9,20:32,60:73] if a.ndim==3 else a[20:32,60:73] if a.ndim==2 else a[:9]
        o={n:crop(read(n)) for n in ['MU_1','MU_2','MUB','C1H','C2H','MAPFAC_MX','MAPFAC_MY']}
        for name,(q,r) in SPECIES.items():o[name]=crop(read(q));o['R'+name]=crop(read(r))
    assert all(np.isfinite(a).all() for a in o.values())
    assert np.any(o['Rqv']!=0) and np.any(o['MU_1']!=o['MU_2'])
    spec=importlib.util.spec_from_file_location('kf_pristine_scalar',ROOT/'tests/v025/b_diff/pristine_scalar_rk.py')
    w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
    return o,w,w.build(tmp_path_factory.mktemp('pristine_kf_scalar'))

@pytest.mark.parametrize('name',list(SPECIES))
@pytest.mark.parametrize('pd',[False,True])
def test_kf_real_scalar_chain_original(original,name,pd):
    o,w,lib=original;dt=54.;q=o[name];raw=o['R'+name]
    ns=SimpleNamespace(precision='fp32',c1h=jnp.asarray(o['C1H']),c2h=jnp.asarray(o['C2H']))
    mu=o['MU_1'];mub=o['MUB'];mut=mu+mub
    mus=[(mu+np.float32(f)*(o['MU_2']-mu)).astype(np.float32) for f in [1/3,.5,1.]]
    # add_a2a is the source mask, distinct from the advection rectangle.
    masked=np.asarray(K.kf_add_a2a(jnp.asarray(raw),specified_or_nested=True))
    mass=ns.c1h[:,None,None]*jnp.asarray(mut)[None]+ns.c2h[:,None,None]
    phys={name:K.kf_add_a2a(mass*jnp.asarray(raw),specified_or_nested=True)}
    sc=np.zeros_like(q);adv=[np.zeros_like(q) for _ in range(3)]
    origin=State(mu_total=jnp.asarray(mut),**{name:jnp.asarray(q),name+'_bdy':None})
    for i,dtr in enumerate([dt/3,dt/2,dt]):
        pd_species=(name,) if pd and i==2 else ()
        reference=op._rk_update_scalar_pd(origin,pd_species,(phys,{name:sc}),dtr,ns) if pd_species else origin
        tends=op._root_scalar_stage_tendencies((jnp.asarray(adv[i]),),(name,),origin,
            SimpleNamespace(spec_zone=5),jnp.asarray(o['MAPFAC_MY']),bounded=True,sc_tendencies={name:sc})
        tends=op._with_physics_sc((name,),tends,phys,pd_species)
        stage=State(mu_total=jnp.asarray(mus[i]+mub),**{name:jnp.asarray(q)})
        got=op._apply_moisture_large_step(stage,reference,q_tendencies=tends,dt_rk=dtr,metrics=ns,species=(name,))
    reference=w.chain(lib,q=q,phys=raw,sc_other=sc,adv=adv,mu1=mu,mus=mus,mub=mub,mut=mut,
        c1=o['C1H'],c2=o['C2H'],msftx=o['MAPFAC_MX'],msfty=o['MAPFAC_MY'],dt=dt,
        spec_zone=5,nested=False,specified=True,pd=pd,owned=False,route=0)
    actual=np.asarray(getattr(got,name));assert np.isfinite(actual).all() and np.isfinite(reference).all()
    finalmass=o['C1H'][:,None,None]*(mus[-1]+mub)[None]+o['C2H'][:,None,None]
    addend=np.maximum(np.abs(q),dt*np.abs(np.asarray(phys[name]))/finalmass).astype(np.float64)
    bound=8*np.finfo(np.float32).eps*np.maximum(addend,np.abs(reference))+1e-36
    assert np.all(np.abs(actual.astype(np.float64)-reference)<=bound)
