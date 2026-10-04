"""Unchanged WRF sixth_order_diffusion, with rk_tendency caller arguments."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

WRF = Path("<USER_HOME>/src/wrf_pristine/WRF")
MAPS = ("msftx", "msfty", "msfux", "msfuy", "msfvx", "msfvy")


def build(directory):
    directory.mkdir(parents=True, exist_ok=True)
    source = (WRF / "dyn_em/module_big_step_utilities_em.F").read_text()
    routines = {n: re.search(r"(?ims)^\s*SUBROUTINE " + n + r"\b.*?^\s*END SUBROUTINE " + n + r"\b", source).group(0)
                for n in ("sixth_order_diffusion", "horizontal_diffusion", "horizontal_diffusion_3dmp")}
    body = "\n".join(routines.values())
    flags = sorted(set(re.findall(r"config_flags%(\w+)", body)))
    config = []
    for flag in flags:
        typ, default = ("integer", "0") if flag == "diff_6th_slopeopt" else ("real", "0.") if flag == "diff_6th_thresh" else ("logical", ".false.")
        config.append(f"{typ} :: {flag}={default}")
    header = "module wrf_diff_oracle\nuse iso_c_binding\nimplicit none\ntype grid_config_rec_type\n" + "\n".join(config) + "\nend type\ncontains\n"
    wrapper = '''
subroutine oracle(nx,ny,nz,which,opt,f3,f2,c,dt,factor) bind(c)
integer(c_int),value :: nx,ny,nz,which,opt
real(c_float) :: f3(nx+3,nz+1,ny+3,5),f2(nx+3,ny+3,7),c(nz+1,2)
real(c_float),value :: dt,factor
character(len=1),parameter :: names(4) = ['u','v','w','m']
type(grid_config_rec_type) :: cfg
cfg%specified=.true.
call sixth_order_diffusion(names(which),f3(:,:,:,1),f3(:,:,:,2),f2(:,:,1),dt, &
 cfg,c(:,1),c(:,2),opt,factor,f3(:,:,:,3),f3(:,:,:,4),1.,1., &
 f2(:,:,2),f2(:,:,3),f2(:,:,4),f2(:,:,5),f2(:,:,6),f2(:,:,7), &
 1,nx+1,1,ny+1,1,nz+1,0,nx+2,0,ny+2,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
end subroutine
'''
    wrapper += '''
subroutine oracle_h(nx,ny,nz,which,opt,f3,f2,c,rdx,rdy) bind(c)
integer(c_int),value :: nx,ny,nz,which,opt
real(c_float) :: f3(nx+3,nz+1,ny+3,5),f2(nx+3,ny+3,7),c(nz+1,2)
real(c_float),value :: rdx,rdy
character(len=1),parameter :: names(4) = ['u','v','w','m']
type(grid_config_rec_type) :: cfg
cfg%specified=.true.
if(which==4)then
call horizontal_diffusion_3dmp(names(which),f3(:,:,:,1),f3(:,:,:,2),f2(:,:,1),c(:,1),c(:,2), &
 cfg,f3(:,:,:,3),f2(:,:,4),f2(:,:,5),f2(:,:,6),1./f2(:,:,6),f2(:,:,7),f2(:,:,2),f2(:,:,3), &
 0.,f3(:,:,:,5),rdx,rdy, &
 1,nx+1,1,ny+1,1,nz+1,0,nx+2,0,ny+2,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
else
call horizontal_diffusion(names(which),f3(:,:,:,1),f3(:,:,:,2),f2(:,:,1),c(:,1),c(:,2), &
 cfg,f2(:,:,4),f2(:,:,5),f2(:,:,6),1./f2(:,:,6),f2(:,:,7),f2(:,:,2),f2(:,:,3), &
 0.,f3(:,:,:,5),rdx,rdy, &
 1,nx+1,1,ny+1,1,nz+1,0,nx+2,0,ny+2,1,nz+1,1,nx+1,1,ny+1,1,nz+1)
endif
end subroutine
end module
'''
    path = directory / "oracle.f90"
    path.write_text(header + body + wrapper)
    command = [os.environ.get("FC", "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"), "-O2", "-fPIC", "-shared", "-ffree-line-length-none", "-ffp-contract=off", "-fcheck=bounds", "-J", str(directory), str(path), "-o", str(directory / "oracle.so")]
    subprocess.run(command, check=True, capture_output=True, text=True)
    caller = (WRF / "dyn_em/module_em.F").read_text()
    calls = re.findall(r"(?is)CALL sixth_order_diffusion\( '[uvwm]'.*?\)", caller)
    manifest = dict(source=str(WRF / "dyn_em/module_big_step_utilities_em.F"),
                    source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    routine_sha256={n:hashlib.sha256(b.encode()).hexdigest() for n,b in routines.items()},
                    caller=str(WRF / "dyn_em/module_em.F"), caller_sha256=hashlib.sha256(caller.encode()).hexdigest(),
                    caller_calls=calls[:4], command=command, real_kind_bytes=4,
                    limitations="Real-derived b-core stage fields, not internal WRF step savepoints; slopeopt=0, specified/nested.")
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return directory / "oracle.so"


def run(path, field, mut, c1, c2, maps, *, name, dt, factor, monotonic,
        kh=None, base=None):
    nz = len(c1) - int(name == "w")
    ny, nx = mut.shape
    f3 = np.zeros((nx+3, nz+1, ny+3, 5), np.float32, order="F")
    f2 = np.ones((nx+3, ny+3, 7), np.float32, order="F")
    k, y, x = field.shape
    f3[1:x+1, :k, 1:y+1, 0] = np.asarray(field, np.float32).transpose(2, 0, 1)
    for channel, value in ((2,base),(4,kh)):
        if value is not None:
            kk,yy,xx=value.shape
            f3[1:xx+1,:kk,1:yy+1,channel]=np.asarray(value,np.float32).transpose(2,0,1)
    for i, value in enumerate((mut, *[maps[m] for m in MAPS])):
        yy, xx = value.shape
        f2[1:xx+1, 1:yy+1, i] = np.asarray(value, np.float32).T
    c = np.empty((nz+1, 2), np.float32, order="F")
    for i, value in enumerate((c1, c2)):
        c[:, i] = np.pad(np.asarray(value, np.float32), (0, nz+1-len(value)), mode="edge")
    library = ctypes.CDLL(str(path))
    fn = library.oracle if kh is None else library.oracle_h
    fn.argtypes = [ctypes.c_int]*5 + [ctypes.c_void_p]*3 + [ctypes.c_float]*2
    fn.restype = None
    fn(nx, ny, nz, ("u", "v", "w", "m").index(name)+1, 2 if monotonic else 1,
       f3.ctypes.data, f2.ctypes.data, c.ctypes.data, dt, factor)
    result = f3[1:x+1, :k, 1:y+1, 1].transpose(1, 2, 0).copy()
    if kh is not None:
        return result
    if name in ("u", "w"):
        result /= maps["msfuy" if name == "u" else "msfty"][None]
    elif name == "v":
        result *= (np.float32(1) / maps["msfvx"])[None]
    return result
