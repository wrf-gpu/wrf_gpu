"""Verbatim pristine advect_scalar_pd, REAL4 caller with explicit WRF masses."""
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

SOURCE = Path("<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_advect_em.F")


def build(directory, contraction="off"):
    directory.mkdir(parents=True, exist_ok=True)
    routine = re.search(r"(?ims)^SUBROUTINE advect_scalar_pd\s*\(.*?^END SUBROUTINE advect_scalar_pd\s*$", SOURCE.read_text()).group(0)
    flags = sorted(set(re.findall(r"config_flags%(\w+)", routine, re.I)))
    logical = [n for n in flags if n not in ("h_sca_adv_order", "v_sca_adv_order")]
    preamble = "module pd_original\nuse iso_c_binding\nimplicit none\ntype grid_config_rec_type\n"
    preamble += "\n".join("logical :: " + n + "=.false." for n in logical)
    preamble += "\ninteger :: h_sca_adv_order=5,v_sca_adv_order=3\nend type\ncharacter(512) :: wrf_err_message\ncontains\n"
    wrapper = """
subroutine pd(nx,ny,nz,q,qold,ru,rv,rom,mut,mub,muold,mx,my,ux,uy,vx,vy,fzm,fzp,rdzw,c1,c2,rdx,rdy,dt,tend) bind(c)
integer(c_int),value :: nx,ny,nz
real(c_float),value :: rdx,rdy,dt
real(c_float),intent(in) :: q(-3:nx+4,1:nz+1,-3:ny+4),qold(-3:nx+4,1:nz+1,-3:ny+4)
real(c_float),intent(in) :: ru(-3:nx+4,1:nz+1,-3:ny+4),rv(-3:nx+4,1:nz+1,-3:ny+4),rom(-3:nx+4,1:nz+1,-3:ny+4)
real(c_float),intent(in) :: mut(-3:nx+4,-3:ny+4),mub(-3:nx+4,-3:ny+4),muold(-3:nx+4,-3:ny+4)
real(c_float),intent(in) :: mx(-3:nx+4,-3:ny+4),my(-3:nx+4,-3:ny+4)
real(c_float),intent(in) :: ux(-3:nx+4,-3:ny+4),uy(-3:nx+4,-3:ny+4),vx(-3:nx+4,-3:ny+4),vy(-3:nx+4,-3:ny+4)
real(c_float),intent(in) :: fzm(nz+1),fzp(nz+1),rdzw(nz+1),c1(nz+1),c2(nz+1)
real(c_float),intent(out) :: tend(-3:nx+4,1:nz+1,-3:ny+4)
real(c_float) :: ht(-3:nx+4,1:nz+1,-3:ny+4),zt(-3:nx+4,1:nz+1,-3:ny+4)
type(grid_config_rec_type) :: cfg
cfg%specified=.true.
tend=0.;ht=0.;zt=0.
call advect_scalar_pd(q,qold,tend,ht,zt,ru,rv,rom,c1,c2,mut,mub,muold,1,cfg,.false., &
 ux,uy,vx,vy,mx,my,fzm,fzp,rdx,rdy,rdzw,dt, &
 1,nx+1,1,ny+1,1,nz+1,-3,nx+4,-3,ny+4,1,nz+1,1,nx,1,ny,1,nz+1)
end subroutine
subroutine wrf_error_fatal(message)
character(*),intent(in) :: message
print *,message
error stop 99
end subroutine
end module
"""
    source = directory / "pd_original.f90"
    source.write_text(preamble + routine + wrapper)
    command = ["<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran", "-O2", "-march=native",
               "-ffp-contract=" + contraction, "-fPIC", "-shared", "-fcheck=bounds",
               "-ffree-line-length-none", "-J", str(directory), str(source), "-o", str(directory / "pd.so")]
    subprocess.run(command, check=True, capture_output=True, text=True)
    (directory / "manifest.json").write_text(json.dumps(dict(source=str(SOURCE),
        routine_sha256=hashlib.sha256(routine.encode()).hexdigest(), compile=command,
        scope="pristine full advect_scalar_pd; root specified, h5/v3, time_step1, tenddec false; all REAL4",
        caller="mut=full current dry mass; mub=base; muold=old perturbation; memories include 4 edge-repeated halo cells; full physical tile",
        dimensions="WRF i,k,j; ids/its1,ide nx+1,ite nx; analog y; k1..nz+1, ktf nz"), indent=2)+"\n")
    return directory / "pd.so"


def evaluate(library, arrays, rdx, rdy, dt):
    nz, ny, nx = arrays["q"].shape
    def volume(a):
        a = np.asarray(a, np.float32)
        a = np.pad(a, ((0,nz+1-a.shape[0]),(4,ny+4-a.shape[1]),(4,nx+4-a.shape[2])), mode="edge")
        return np.asfortranarray(a.transpose(2,0,1))
    def plane(a):
        a = np.asarray(a,np.float32)
        return np.asfortranarray(np.pad(a,((4,ny+4-a.shape[0]),(4,nx+4-a.shape[1])),mode="edge").T)
    def vector(a):
        a = np.asarray(a,np.float32)
        return np.pad(a,(0,nz+1-len(a)),mode="edge")
    values = ([volume(arrays[n]) for n in ("q","qold","ru","rv","rom")]
              +[plane(arrays[n]) for n in ("mut","mub","muold","mx","my","ux","uy","vx","vy")]
              +[vector(arrays[n]) for n in ("fzm","fzp","rdzw","c1","c2")])
    out = np.empty((nx+8,nz+1,ny+8),np.float32,order="F")
    fn = ctypes.CDLL(str(library)).pd
    fn.argtypes = [ctypes.c_int]*3+[ctypes.c_void_p]*19+[ctypes.c_float]*3+[ctypes.c_void_p]
    fn(nx,ny,nz,*[a.ctypes.data for a in values],rdx,rdy,dt,out.ctypes.data)
    return out[4:4+nx,:nz,4:4+ny].transpose(1,2,0).copy()
