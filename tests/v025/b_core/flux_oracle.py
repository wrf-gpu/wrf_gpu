"""Pristine WRF statement functions with an explicit specified-face caller."""
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

SOURCE = Path("<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_advect_em.F")


def build(directory, contraction="off"):
    directory.mkdir(parents=True,exist_ok=True)
    text=SOURCE.read_text()
    # The original REAL statement functions, including SIGN(time_step), copied
    # verbatim. The positive operational time step is part of this caller.
    functions=re.search(r"(?ms)^   flux4\(.*?(?=\n\n\n   LOGICAL :: specified)",text).group(0)
    head="""subroutine faces(nx,ny,nz,axis,upstream,q,v,f) bind(c)
use iso_c_binding
implicit none
integer(c_int),value :: nx,ny,nz,axis,upstream
real(c_float),intent(in) :: q(nx,nz,ny),v(nx,nz,ny)
real(c_float),intent(out) :: f(nx,nz,ny)
integer :: i,j,k,m,n,time_step
real :: flux3,flux4,flux5,flux6,q_im3,q_im2,q_im1,q_i,q_ip1,q_ip2,ua
"""
    body="""
time_step=1
f=0.
n=merge(nx,ny,axis==2)
do j=1,ny
do k=1,nz
do i=1,nx
m=merge(i,j,axis==2)
ua=v(i,k,j)
if(m>=4.and.m<=n-2)then
 if(axis==2)then
  f(i,k,j)=ua*flux5(q(i-3,k,j),q(i-2,k,j),q(i-1,k,j),q(i,k,j),q(i+1,k,j),q(i+2,k,j),ua)
 else
  f(i,k,j)=ua*flux5(q(i,k,j-3),q(i,k,j-2),q(i,k,j-1),q(i,k,j),q(i,k,j+1),q(i,k,j+2),ua)
 endif
elseif(m==3.or.m==n-1)then
 if(axis==2)then
  f(i,k,j)=ua*flux3(q(i-2,k,j),q(i-1,k,j),q(i,k,j),q(i+1,k,j),ua)
 else
  f(i,k,j)=ua*flux3(q(i,k,j-2),q(i,k,j-1),q(i,k,j),q(i,k,j+1),ua)
 endif
elseif(m==2.or.m==n)then
 if(axis==2)then
  q_i=q(i,k,j);q_im1=q(i-1,k,j)
 else
  q_i=q(i,k,j);q_im1=q(i,k,j-1)
 endif
 if(upstream/=0)then
  if(m==2.and.q_i<0.)q_im1=q_i
  if(m==n.and.q_im1>0.)q_i=q_im1
 endif
 f(i,k,j)=ua*.5*(q_i+q_im1)
endif
enddo
enddo
enddo
end subroutine
"""
    source=directory/"faces.f90";source.write_text(head+functions+body)
    flags=["-O2","-march=native","-ffp-contract="+contraction,"-fPIC","-shared","-fcheck=bounds","-ffree-line-length-none"]
    cmd=["<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran",*flags,str(source),"-o",str(directory/"faces.so")]
    subprocess.run(cmd,check=True,capture_output=True,text=True)
    (directory/"manifest.json").write_text(json.dumps(dict(source=str(SOURCE),statement_functions_sha256=hashlib.sha256(functions.encode()).hexdigest(),compile=cmd,scope="WRF order5 specified face tiers; positive step; optional normal upstream"),indent=2)+"\n")
    return directory/"faces.so"


def evaluate(library, field, velocity, axis, upstream=False):
    q=np.asfortranarray(np.asarray(field,np.float32).transpose(2,0,1))
    v=np.asfortranarray(np.asarray(velocity,np.float32).transpose(2,0,1))
    out=np.empty_like(q)
    fn=ctypes.CDLL(str(library)).faces
    fn.argtypes=[ctypes.c_int]*5+[ctypes.c_void_p]*3
    nx,nz,ny=q.shape
    fn(nx,ny,nz,axis,int(upstream),q.ctypes.data,v.ctypes.data,out.ctypes.data)
    return out.transpose(1,2,0).copy()
