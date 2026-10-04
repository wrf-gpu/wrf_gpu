"""Compile verbatim pristine-WRF REAL32 BouLac and tridiag2 column oracles.

Only a C ABI wrapper and module constants are supplied. Routine bodies are
extracted unchanged from the pinned WRF source, with hashes in the receipt.
No tolerance or candidate implementation lives in this oracle builder.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

WRF = Path('<USER_HOME>/src/wrf_pristine/WRF')


def build(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = WRF / 'phys/module_bl_mynnedmf.F'
    text = source.read_text()
    bodies = {}
    for name in ('boulac_length', 'tridiag2'):
        match = re.search(r'^ *subroutine ' + name + r'\(.*?^ *end subroutine ' + name + r'\s*$',
                          text, re.I | re.M | re.S)
        if match is None:
            raise RuntimeError(f'pristine WRF routine missing: {name}')
        bodies[name] = match.group(0)
    constants = 'integer, parameter :: kind_phys=4\nreal(kind_phys), parameter :: grav=9.81, tref=300.0, gtr=grav/tref\n'
    wrapper = """
subroutine bp_boulac(nz,nb,zw,dz,qtke,theta,lb1,lb2) bind(C)
use iso_c_binding
integer(c_int),value :: nz,nb
real(c_float),intent(in) :: zw(nz+1,nb),dz(nz,nb),qtke(nz,nb),theta(nz,nb)
real(c_float),intent(out) :: lb1(nz,nb),lb2(nz,nb)
integer :: col
do col=1,nb
 call boulac_length(1,nz,zw(:,col),dz(:,col),qtke(:,col),theta(:,col),lb1(:,col),lb2(:,col))
enddo
end subroutine
subroutine bp_tridiag(nz,nb,a,b,c,d,x) bind(C)
use iso_c_binding
integer(c_int),value :: nz,nb
real(c_float),intent(in) :: a(nz,nb),b(nz,nb),c(nz,nb),d(nz,nb)
real(c_float),intent(out) :: x(nz,nb)
integer :: col
do col=1,nb
 call tridiag2(nz,a(:,col),b(:,col),c(:,col),d(:,col),x(:,col))
enddo
end subroutine
"""
    generated = 'module bp_wrf\nimplicit none\n' + constants + 'contains\n' + '\n'.join(bodies.values()) + wrapper + 'end module\n'
    (output / 'oracle.f90').write_text(generated)
    compiler = os.environ.get('FC', '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran')
    flags = ['-O2', '-ffp-contract=off', '-fPIC', '-shared', '-fcheck=bounds']
    subprocess.run(['timeout', '60', compiler, *flags, 'oracle.f90', '-o', 'oracle.so'],
                   cwd=output, check=True, capture_output=True)
    receipt = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   routine_sha256={name: hashlib.sha256(body.encode()).hexdigest() for name, body in bodies.items()},
                   compiler=subprocess.check_output([compiler, '--version'], text=True).splitlines()[0],
                   flags=flags, kind_phys=4, constants=constants,
                   library_sha256=hashlib.sha256((output / 'oracle.so').read_bytes()).hexdigest())
    (output / 'provenance.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return Oracle(output / 'oracle.so')


class Oracle:
    def __init__(self, library):
        self.lib = ctypes.CDLL(str(library))
        self.ptr = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
        self.lib.bp_boulac.argtypes = [ctypes.c_int, ctypes.c_int] + [self.ptr] * 6
        self.lib.bp_boulac.restype = None
        self.lib.bp_tridiag.argtypes = [ctypes.c_int, ctypes.c_int] + [self.ptr] * 5
        self.lib.bp_tridiag.restype = None

    def boulac(self, zw, dz, qtke, theta):
        dz, qtke, theta, zw = [np.ascontiguousarray(v, dtype=np.float32) for v in (dz, qtke, theta, zw)]
        nb, nz = dz.shape
        lb1, lb2 = np.empty_like(dz), np.empty_like(dz)
        self.lib.bp_boulac(nz, nb, zw, dz, qtke, theta, lb1, lb2)
        return lb1, lb2

    def tridiag(self, a, b, c, d):
        a, b, c, d = [np.ascontiguousarray(v, dtype=np.float32) for v in (a, b, c, d)]
        nb, nz = d.shape
        result = np.empty_like(d)
        self.lib.bp_tridiag(nz, nb, a, b, c, d, result)
        return result
