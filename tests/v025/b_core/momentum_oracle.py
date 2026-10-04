"""Full pristine WRF momentum advection; explicit C-grid tile bounds."""
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np
from scalar_oracle import SOURCE


def build(directory,contraction="off"):
    directory.mkdir(parents=True,exist_ok=True)
    text=SOURCE.read_text();names=("advect_u","advect_v","advect_w")
    bodies={n:re.search(r"(?ims)^SUBROUTINE "+n+r"\s*\(.*?^END SUBROUTINE "+n+r"\s*$",text).group(0) for n in names}
    flags=sorted(set(re.findall(r"config_flags%([a-z_]+)","\n".join(bodies.values()),re.I)))
    ints=[n for n in flags if n.endswith("adv_order")]
    pre="module momentum_original\nuse iso_c_binding\nimplicit none\ntype grid_config_rec_type\n"
    pre+="\n".join("logical :: "+n+"=.false." for n in flags if n not in ints)
    pre+="\n"+"\n".join("integer :: "+n+"="+("5" if n.startswith("h_") else "3") for n in ints)
    constants=SOURCE.parents[1]/"share/module_model_constants.F"
    cb=re.search(r"(?im)^.*PARAMETER :: cb[^\n]*",constants.read_text()).group(0)
    pre+="\nend type\n"+cb+"\ncharacter(512) :: wrf_err_message\ncontains\n"
    wrapper="""
subroutine momentum(nx,ny,nz,kind,nested,q,ru,rv,rom,mu,mux,muy,mvx,mvy,map, &
 fzm,fzp,rdz,c1,c2,rdx,rdy,tend) bind(c)
integer(c_int),value :: nx,ny,nz,kind,nested
real(c_float),value :: rdx,rdy
real(c_float),intent(in) :: q(nx+1,nz+1,ny+1),ru(nx+1,nz+1,ny+1)
real(c_float),intent(in) :: rv(nx+1,nz+1,ny+1),rom(nx+1,nz+1,ny+1)
real(c_float),intent(in) :: mu(nx+1,ny+1),mux(nx+1,ny+1),muy(nx+1,ny+1)
real(c_float),intent(in) :: mvx(nx+1,ny+1),mvy(nx+1,ny+1),map(nx+1,ny+1)
real(c_float),intent(in) :: fzm(nz+1),fzp(nz+1),rdz(nz+1),c1(nz+1),c2(nz+1)
real(c_float),intent(out) :: tend(nx+1,nz+1,ny+1)
type(grid_config_rec_type) :: cfg
integer :: itile,jtile
cfg%nested=nested/=0;cfg%specified=nested==0
tend=0.
itile=nx;jtile=ny
if(kind==0)itile=nx+1
if(kind==1)jtile=ny+1
select case(kind)
"""
    for i,n in enumerate(names):
        wrapper+=f"case({i})\ncall {n}(q,q,tend,ru,rv,rom,c1,c2,mu,1,cfg, &\n mux,muy,mvx,mvy,map,map,fzm,fzp,rdx,rdy,rdz, &\n 1,nx+1,1,ny+1,1,nz+1, &\n 1,nx+1,1,ny+1,1,nz+1, &\n 1,itile,1,jtile,1,nz+1)\n"
    wrapper+="end select\nend subroutine\nsubroutine wrf_error_fatal(message)\ncharacter(*),intent(in) :: message\nprint *,message\nerror stop 99\nend subroutine\nend module\n"
    src=directory/"momentum_original.f90";src.write_text(pre+"\n".join(bodies.values())+wrapper)
    compiler="<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"
    cmd=[compiler,"-O2","-march=native","-ffp-contract="+contraction,"-fPIC","-shared","-fcheck=bounds","-ffree-line-length-none","-J",str(directory),str(src),"-o",str(directory/"momentum.so")]
    subprocess.run(cmd,check=True,capture_output=True,text=True)
    (directory/"manifest.json").write_text(json.dumps(dict(source=str(SOURCE),routine_sha256={n:hashlib.sha256(b.encode()).hexdigest() for n,b in bodies.items()},compile=cmd,
        cb_declaration=cb.strip(),scope="three entire pristine momentum routines, full u/v/w Cgrid tiles, specified/nested, h5/v3,positive step,noopen/periodic/polar; kde=kte=nz+1"),indent=2)+"\n")


def evaluate(library,kind,arrays,field,rdx,nested=False):
    nz,ny,nx=arrays["q"].shape
    def volume(v):return np.asfortranarray(np.pad(np.asarray(v,np.float32),((0,nz+1-v.shape[0]),(0,ny+1-v.shape[1]),(0,nx+1-v.shape[2])),mode="edge").transpose(2,0,1))
    def plane(v):return np.asfortranarray(np.pad(np.asarray(v,np.float32),((0,ny+1-v.shape[0]),(0,nx+1-v.shape[1])),mode="edge").T)
    def vector(v):return np.pad(np.asarray(v,np.float32),(0,nz+1-len(v)),mode="edge")
    values=[volume(field)]+[volume(arrays[k]) for k in ("ru_full","rv_full","rom")]+[plane(arrays[k]) for k in ("mu","mux","muy","mvx","mvy","map")]+[vector(arrays[k]) for k in ("fzm","fzp","rdz","c1","c2")]
    out=np.empty((nx+1,nz+1,ny+1),np.float32,order="F")
    fn=ctypes.CDLL(str(library)).momentum
    fn.argtypes=[ctypes.c_int]*5+[ctypes.c_void_p]*15+[ctypes.c_float]*2+[ctypes.c_void_p]
    fn(nx,ny,nz,kind,int(nested),*[v.ctypes.data for v in values],rdx,rdx,out.ctypes.data)
    return out[:field.shape[2],:field.shape[0],:field.shape[1]].transpose(1,2,0).copy()
