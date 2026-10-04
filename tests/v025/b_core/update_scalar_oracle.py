"""Verbatim pristine rk_update_scalar, REAL4 root (specified) caller, sc_tend supplied."""
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np

SOURCE = Path("<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_em.F")


def build(directory, contraction="off"):
    directory.mkdir(parents=True, exist_ok=True)
    routine = re.search(r"(?ims)^SUBROUTINE rk_update_scalar\s*\(.*?^END SUBROUTINE rk_update_scalar\s*$",
                        SOURCE.read_text()).group(0)
    preamble = ("module update_original\nuse iso_c_binding\nimplicit none\ntype grid_config_rec_type\n"
                "logical :: nested=.false.,specified=.true.,periodic_x=.false.\ninteger :: rk_ord=3\n"
                "end type\ncontains\n")
    wrapper = """
subroutine update(nx,ny,nz,rk_step,spec_zone,dt,s1,tend,sct,mx,my,c1,c2,muold,munew,mub,s2) bind(c)
integer(c_int),value :: nx,ny,nz,rk_step,spec_zone
real(c_float),value :: dt
real(c_float),intent(in) :: s1(nx+1,nz+1,ny+1),tend(nx+1,nz+1,ny+1),sct(nx+1,nz+1,ny+1)
real(c_float),intent(in) :: mx(nx+1,ny+1),my(nx+1,ny+1),muold(nx+1,ny+1),munew(nx+1,ny+1),mub(nx+1,ny+1)
real(c_float),intent(in) :: c1(nz+1),c2(nz+1)
real(c_float),intent(out) :: s2(nx+1,nz+1,ny+1)
real(c_float) :: a1(nx+1,nz+1,ny+1,1),a2(nx+1,nz+1,ny+1,1),sc(nx+1,nz+1,ny+1,1),zero(nx+1,nz+1,ny+1)
type(grid_config_rec_type) :: cfg
a1(:,:,:,1)=s1;a2(:,:,:,1)=s1;sc(:,:,:,1)=sct;zero=0.
call rk_update_scalar(scs=1,sce=1,scalar_1=a1,scalar_2=a2,sc_tend=sc,advect_tend=tend, &
 h_tendency=zero,z_tendency=zero,msftx=mx,msfty=my,c1=c1,c2=c2,mu_old=muold,mu_new=munew,mu_base=mub, &
 rk_step=rk_step,dt=dt,spec_zone=spec_zone,config_flags=cfg,tenddec=.false., &
 ids=1,ide=nx+1,jds=1,jde=ny+1,kds=1,kde=nz+1,ims=1,ime=nx+1,jms=1,jme=ny+1,kms=1,kme=nz+1, &
 its=1,ite=nx,jts=1,jte=ny,kts=1,kte=nz+1)
s2=a2(:,:,:,1)
end subroutine
end module
"""
    source = directory / "update_original.f90"
    source.write_text(preamble + routine + wrapper)
    command = ["<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran", "-O2", "-march=native",
               "-ffp-contract=" + contraction, "-fPIC", "-shared", "-fcheck=bounds",
               "-ffree-line-length-none", "-J", str(directory), str(source), "-o", str(directory / "update.so")]
    subprocess.run(command, check=True, capture_output=True, text=True)
    (directory / "manifest.json").write_text(json.dumps(dict(
        source=str(SOURCE), routine_sha256=hashlib.sha256(routine.encode()).hexdigest(), compile=command,
        scope="pristine full rk_update_scalar; specified root, rk_ord 3, tenddec false, all REAL4",
        caller="solve_em other_scalar_advance: scalar_1=old, mu_old=mu_1, mu_new=mu_2 (perturbations), mu_base=mub"),
        indent=2) + "\n")
    return directory / "update.so"


def evaluate(library, arrays, rk_step, dt, spec_zone):
    nz, ny, nx = arrays["s1"].shape

    def volume(a):
        a = np.asarray(a, np.float32)
        return np.asfortranarray(np.pad(a, ((0, 1), (0, 1), (0, 1)), mode="edge").transpose(2, 0, 1))

    def plane(a):
        return np.asfortranarray(np.pad(np.asarray(a, np.float32), ((0, 1), (0, 1)), mode="edge").T)

    def vector(a):
        a = np.asarray(a, np.float32)
        return np.pad(a, (0, nz + 1 - len(a)), mode="edge")

    values = ([volume(arrays[n]) for n in ("s1", "tend", "sct")]
              + [plane(arrays[n]) for n in ("mx", "my")]
              + [vector(arrays[n]) for n in ("c1", "c2")]
              + [plane(arrays[n]) for n in ("muold", "munew", "mub")])
    out = np.empty((nx + 1, nz + 1, ny + 1), np.float32, order="F")
    fn = ctypes.CDLL(str(library)).update
    fn.argtypes = [ctypes.c_int] * 5 + [ctypes.c_float] + [ctypes.c_void_p] * 11
    fn(nx, ny, nz, int(rk_step), int(spec_zone), float(dt), *[v.ctypes.data for v in values], out.ctypes.data)
    return out[:nx, :nz, :ny].transpose(1, 2, 0).copy()
