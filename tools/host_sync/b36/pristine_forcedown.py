"""Extract unchanged WRF REAL4 bdy_interp1/SINTB with a C adapter."""
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np

WRF = Path('<USER_HOME>/src/wrf_pristine/WRF')


def build(directory):
    directory.mkdir(parents=True, exist_ok=True)
    sources = {name: (WRF/name).read_text() for name in
               ('share/interp_fcn.F', 'share/sint.F')}
    routines = {}
    for name, source in [('bdy_interp1', sources['share/interp_fcn.F']),
                         ('SINTB', sources['share/sint.F'])]:
        match = re.search(r'(?ims)^\s*SUBROUTINE '+name+r'\b.*?^\s*END(?:\s+SUBROUTINE(?:\s+'+name+r')?)?\s*$', source)
        if match is None:
            raise ValueError(name)
        routines[name] = match.group(0)
    wrapper = '''
module module_state_description
end module
module b36_adapter
use iso_c_binding
implicit none
contains
subroutine forcedown(px,py,nx,ny,nz,ipos,jpos,ratio,cdt,parent,child,bx,by) bind(c)
integer(c_int),value :: px,py,nx,ny,nz,ipos,jpos,ratio
real(c_float),value :: cdt
real(c_float) :: parent(px+1,nz,py+1),child(nx+1,nz,ny+1)
real(c_float) :: bx(ny+1,nz,5,2,2),by(nx+1,nz,5,2,2)
integer :: imask(nx+1,ny+1)
real :: unused(1)
imask=1
unused=0.
call bdy_interp1(parent, &
 1,px+1,1,nz+1,1,py+1, 1,px+1,1,nz,1,py+1, 1,px+1,1,nz,1,py+1, &
 child,1,max(nx+1,ny+1),5, &
 1,nx+1,1,nz+1,1,ny+1, 1,nx+1,1,nz,1,ny+1, 1,nx+1,1,nz,1,ny+1, &
 0,imask,.false.,.false.,ipos,jpos,ratio,ratio, &
 unused,bx(:,:,:,1,1),unused,bx(:,:,:,2,1), &
 unused,by(:,:,:,1,1),unused,by(:,:,:,2,1), &
 unused,bx(:,:,:,1,2),unused,bx(:,:,:,2,2), &
 unused,by(:,:,:,1,2),unused,by(:,:,:,2,2),cdt,cdt/ratio)
end subroutine
end module
subroutine nl_get_spec_zone(dom,value)
integer :: dom,value
value=1
end subroutine
subroutine nl_get_relax_zone(dom,value)
integer :: dom,value
value=4
end subroutine
'''
    path = directory/'forcedown.f90'
    path.write_text(wrapper+'\n'+'\n'.join(routines.values()))
    command = ['<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran', '-O2',
               '-shared', '-fPIC', '-ffree-line-length-none', '-ffp-contract=off',
               '-fcheck=bounds', '-J', str(directory), str(path),
               '-o', str(directory/'forcedown.so')]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(done.stderr)
    (directory/'forcedown_manifest.json').write_text(json.dumps(dict(
        source_sha256={n: hashlib.sha256(v.encode()).hexdigest() for n,v in sources.items()},
        routine_sha256={n: hashlib.sha256(v.encode()).hexdigest() for n,v in routines.items()},
        command=command, REAL_bytes=4, spec_zone=1, relax_zone=4,
        adapter='mass-grid nonstaggered, actual nest starts/ratio/parentdt; memory halo extends one cell, edge padded',
    ), indent=2)+'\n')
    return directory/'forcedown.so'


def forcedown(path, parent, child, *, start, ratio, cadence, side_len):
    nz,py,px = parent.shape
    cz,ny,nx = child.shape
    assert cz == nz
    p = np.asfortranarray(np.pad(np.asarray(parent,np.float32),((0,0),(0,1),(0,1)),mode='edge').transpose(2,0,1))
    q = np.asfortranarray(np.pad(np.asarray(child,np.float32),((0,0),(0,1),(0,1)),mode='edge').transpose(2,0,1))
    bx = np.zeros((ny+1,nz,5,2,2),np.float32,order='F')
    by = np.zeros((nx+1,nz,5,2,2),np.float32,order='F')
    library = ctypes.CDLL(str(path))
    fn = library.forcedown
    fn.argtypes = [ctypes.c_int]*8+[ctypes.c_float]+[ctypes.c_void_p]*4
    fn.restype = None
    fn(px,py,nx,ny,nz,*start,ratio,cadence,*[a.ctypes.data for a in (p,q,bx,by)])
    old = np.zeros((4,5,nz,side_len),np.float32)
    rate = np.zeros_like(old)
    for side in range(4):
        a = bx if side<2 else by
        count = ny if side<2 else nx
        old[side,:,:,:count] = a[:count,:,:,side%2,0].transpose(2,1,0)
        rate[side,:,:,:count] = a[:count,:,:,side%2,1].transpose(2,1,0)
    return old, rate
