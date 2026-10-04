"""Verbatim pristine advect_scalar with full-domain specified/nested caller."""
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

SOURCE=Path("<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_advect_em.F")


def build(directory,contraction="off"):
    directory.mkdir(parents=True,exist_ok=True)
    text=SOURCE.read_text()
    routine=re.search(r"(?ims)^SUBROUTINE advect_scalar\s*\(.*?^END SUBROUTINE advect_scalar\s*$",text).group(0)
    flags=sorted(set(re.findall(r"config_flags%([a-z_]+)",routine,re.I)))
    logical=[f for f in flags if f not in ("h_sca_adv_order","v_sca_adv_order")]
    preamble="module scalar_original\nuse iso_c_binding\nimplicit none\ntype grid_config_rec_type\n"
    preamble+="\n".join("logical :: "+f+"=.false." for f in logical)
    preamble+="\ninteger :: h_sca_adv_order=5,v_sca_adv_order=3\nend type\ncharacter(512) :: wrf_err_message\ncontains\n"
    wrapper="""
subroutine scalar(nx,ny,nz,nested,q,ru,rv,rom,mu,map,fzm,fzp,rdzw,c1,c2,rdx,rdy,tend) bind(c)
integer(c_int),value :: nx,ny,nz,nested
real(c_float),value :: rdx,rdy
real(c_float),intent(in) :: q(nx+1,nz+1,ny+1),ru(nx+1,nz+1,ny+1)
real(c_float),intent(in) :: rv(nx+1,nz+1,ny+1),rom(nx+1,nz+1,ny+1)
real(c_float),intent(in) :: mu(nx+1,ny+1),map(nx+1,ny+1)
real(c_float),intent(in) :: fzm(nz+1),fzp(nz+1),rdzw(nz+1),c1(nz+1),c2(nz+1)
real(c_float),intent(out) :: tend(nx+1,nz+1,ny+1)
type(grid_config_rec_type) :: cfg
cfg%nested=nested/=0
cfg%specified=nested==0
tend=0.
call advect_scalar(q,q,tend,ru,rv,rom,c1,c2,mu,1,cfg, &
 map,map,map,map,map,map,fzm,fzp,rdx,rdy,rdzw, &
 1,nx+1,1,ny+1,1,nz+1, &
 1,nx+1,1,ny+1,1,nz+1, &
 1,nx,1,ny,1,nz+1)
end subroutine
subroutine wrf_error_fatal(message)
character(*),intent(in) :: message
print *,message
error stop 99
end subroutine
end module
"""
    source=directory/"scalar_original.f90";source.write_text(preamble+routine+wrapper)
    compiler="<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
    options=["-O2","-march=native","-ffp-contract="+contraction,"-fPIC","-shared","-fcheck=bounds","-ffree-line-length-none","-J",str(directory)]
    cmd=[compiler,*options,str(source),"-o",str(directory/"scalar.so")]
    subprocess.run(cmd,check=True,capture_output=True,text=True)
    (directory/"manifest.json").write_text(json.dumps(dict(source=str(SOURCE),routine_sha256=hashlib.sha256(routine.encode()).hexdigest(),compile=cmd,
        scope="entire original advect_scalar; specified(d01)/nested(d02), h5/v3,positive time_step1,noopen/periodic/symmetric/polar; fulltile kte=kde=nz+1 and mass ktf=nz",
        caller="ids=ims=its=1;ide=ime=nx+1,ite=nx; analogousy; kds=kms=kts=1,kde=kme=kte=nz+1; allREAL binary32; tendencyinitial0"),indent=2)+"\n")
    return directory/"scalar.so"


def evaluate(library,arrays,rdx,nested=False):
    q=arrays["q"];nz,ny,nx=q.shape
    def volume(v):
        arr=np.pad(np.asarray(v,np.float32),((0,max(0,nz+1-v.shape[0])),(0,max(0,ny+1-v.shape[1])),(0,max(0,nx+1-v.shape[2]))),mode="edge")
        return np.asfortranarray(arr.transpose(2,0,1))
    def plane(v):return np.asfortranarray(np.pad(np.asarray(v,np.float32),((0,1),(0,1)),mode="edge").T)
    def vector(v):return np.pad(np.asarray(v,np.float32),(0,nz+1-len(v)),mode="edge")
    values=[volume(arrays[k]) for k in ("q","ru","rv","rom")]+[plane(arrays[k]) for k in ("mu","map")]+[vector(arrays[k]) for k in ("fzm","fzp","rdzw","c1","c2")]
    out=np.empty((nx+1,nz+1,ny+1),np.float32,order="F")
    fn=ctypes.CDLL(str(library)).scalar
    fn.argtypes=[ctypes.c_int]*4+[ctypes.c_void_p]*11+[ctypes.c_float]*2+[ctypes.c_void_p]
    fn(nx,ny,nz,int(nested),*[v.ctypes.data for v in values],float(rdx),float(rdx),out.ctypes.data)
    return out[:nx,:nz,:ny].transpose(1,2,0).copy()
