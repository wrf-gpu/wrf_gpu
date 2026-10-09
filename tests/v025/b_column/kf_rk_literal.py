"""REAL4 literal source/lifetime oracle independent of JAX.

The six assignments below are calculate_phy_tend's KF block, followed by
advance_ppt's KFETASCHEME and phy_prep_part2's KF division. Original source
hashes are recorded at build time; bounds are supplied by the test, not fitted.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

WRF=Path('<USER_HOME>/src/wrf_pristine/WRF')
FORTRAN='''subroutine lifetime(ncol,nz,dt,c1,c2,mut,muts,r,prate,nca,coupled,held,nout,rain) bind(c)
use iso_c_binding
implicit none
integer(c_int),value :: ncol,nz
real(c_float),value :: dt
real(c_float) :: c1(nz),c2(nz),mut(ncol),muts(ncol),r(ncol,nz,6),prate(ncol),nca(ncol)
real(c_float) :: coupled(ncol,nz,6),held(ncol,nz,6),nout(ncol),rain(ncol)
integer :: i,k,s
do s=1,6
do k=1,nz
do i=1,ncol
 coupled(i,k,s)=(c1(k)*mut(i)+c2(k))*r(i,k,s)
enddo
enddo
enddo
held=coupled
do i=1,ncol
 rain(i)=prate(i)*dt
 nout(i)=nca(i)
 if(nca(i)>0)then
  if(nint(nca(i)/dt)<=1)held(i,:,:)=0.
  nout(i)=nca(i)-dt
 endif
enddo
do s=1,6
do k=1,nz
do i=1,ncol
 held(i,k,s)=held(i,k,s)/(c1(k)*muts(i)+c2(k))
enddo
enddo
enddo
end subroutine lifetime
'''

def build(path):
    path.mkdir(parents=True,exist_ok=True)
    (path/'oracle.f90').write_text(FORTRAN)
    source_paths=[WRF/'dyn_em/module_em.F',WRF/'phys/module_physics_addtendc.F',WRF/'dyn_em/module_big_step_utilities_em.F']
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    # Guard the literal's source expressions against a different WRF checkout.
    assert 'RTHCUTEN(I,K,J)=(c1(k)*mut(I,J)+c2(k))*RTHCUTEN(I,K,J)' in source_paths[0].read_text()
    assert 'NINT(NCA(I,J) / DT) .LE. 1' in source_paths[1].read_text()
    assert 'RTHCUTEN(I,K,J)=RTHCUTEN(I,K,J)/(c1(k)*MUT(I,J)+c2(k))' in source_paths[2].read_text()
    fc=os.environ.get('FC','<DATA_ROOT>/canairy_meteo/artifacts/envs/wrf-build/bin/x86_64-conda-linux-gnu-gfortran')
    result=subprocess.run([fc,'-O2','-shared','-fPIC','-ffree-line-length-none',str(path/'oracle.f90'),'-o',str(path/'oracle.so')],capture_output=True,text=True,timeout=60)
    result.check_returncode()
    (path/'SOURCE.json').write_text(json.dumps({'source_sha256':sources,'compiler':fc,'flags':['-O2','-shared','-fPIC'],'REAL_bytes':4,'literal_SHA256':hashlib.sha256(FORTRAN.encode()).hexdigest()},indent=2)+'\n')
    lib=ctypes.CDLL(str(path/'oracle.so'));fn=lib.lifetime
    fn.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_float]+[ctypes.c_void_p]*11
    return fn

def run(fn,rates,c1,c2,mut,muts,nca,dt):
    ncol=int(np.asarray(mut).size);nz=int(np.asarray(c1).size)
    arrays=[np.asfortranarray(a,np.float32) for a in
            [c1,c2,np.asarray(mut).reshape(-1),np.asarray(muts).reshape(-1)]]
    r=np.asfortranarray(np.stack([np.asarray(v).reshape(nz,ncol).T for v in rates[:6]],axis=-1),np.float32)
    pr=np.asfortranarray(np.asarray(rates[6]).reshape(ncol),np.float32)
    nc=np.asfortranarray(np.asarray(nca).reshape(ncol),np.float32)
    coupled=np.zeros_like(r,order='F');held=np.zeros_like(r,order='F')
    no=np.zeros(ncol,np.float32);rain=np.zeros(ncol,np.float32)
    fn(ncol,nz,float(dt),*[a.ctypes.data for a in (*arrays,r,pr,nc,coupled,held,no,rain)])
    shape=np.asarray(rates[0]).shape
    profiles=lambda a:tuple(a[:,:,i].T.reshape(shape) for i in range(6))
    return profiles(coupled),profiles(held)+(np.asarray(rates[6]),),no.reshape(np.asarray(nca).shape),rain.reshape(np.asarray(nca).shape)
