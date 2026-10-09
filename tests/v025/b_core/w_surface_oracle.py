"""Pristine WRF ``module_bc_em.F::set_w_surface`` behind a C-callable binary32 adapter.

The subroutine body is copied verbatim from the pristine tree and hashed; only ``grid_config_rec_type``
(periodic_x / periodic_y) is stubbed. Called like ``solve_em.F:4818-4834`` (fill_w_flag=.false.).
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

WRF = Path("<USER_HOME>/src/wrf_pristine/WRF")
SOURCE = WRF / "dyn_em/module_bc_em.F"


def build(directory, *, contraction="off"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = SOURCE.read_text()
    routine = re.search(r"(?ims)^\s*SUBROUTINE set_w_surface\b.*?^\s*END SUBROUTINE set_w_surface\b", source).group(0)
    text = (
        "module b_core_wsurf\nuse iso_c_binding\nimplicit none\n"
        "type grid_config_rec_type\nlogical :: periodic_x=.false.\nlogical :: periodic_y=.false.\nend type\n"
        "contains\n" + routine + "\n"
        "subroutine wsurf_oracle(nx,ny,nz,periodic,w,ht,u,v,msftx,msfty,s) bind(c)\n"
        "integer(c_int),value :: nx,ny,nz,periodic\n"
        "real(c_float),intent(inout) :: w(0:nx+2,1:nz+1,0:ny+2)\n"
        "real(c_float),intent(in) :: u(0:nx+2,1:nz+1,0:ny+2),v(0:nx+2,1:nz+1,0:ny+2)\n"
        "real(c_float),intent(in) :: ht(0:nx+2,0:ny+2),msftx(0:nx+2,0:ny+2),msfty(0:nx+2,0:ny+2),s(5)\n"
        "real :: znw(1:nz+1)\ntype(grid_config_rec_type) :: config_flags\nlogical :: fill_w_flag\n"
        "config_flags%periodic_x=iand(periodic,1)/=0\nconfig_flags%periodic_y=iand(periodic,2)/=0\nfill_w_flag=.false.\nznw=0.\n"
        "call set_w_surface(config_flags,znw,fill_w_flag,w,ht,u,v,s(3),s(4),s(5),s(1),s(2),msftx,msfty, &\n"
        "  1,nx+1,1,ny+1,1,nz+1, 0,nx+2,0,ny+2,1,nz+1, 1,nx+1,1,ny+1,1,nz+1)\n"
        "end subroutine\nend module\n"
    )
    (directory / "wsurf.f90").write_text(text)
    cmd = [os.environ.get("FC", "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"), "-O2", "-fPIC", "-shared",
           "-ffree-line-length-none", "-fcheck=bounds", "-ffp-contract=" + contraction,
           "-J", str(directory), str(directory / "wsurf.f90"), "-o", str(directory / "wsurf.so")]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    meta = dict(source=str(SOURCE), source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                routine_sha256=hashlib.sha256(routine.encode()).hexdigest(), compile=cmd, real_kind_bytes=4)
    (directory / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    return directory / "wsurf.so"


def evaluate(library, u, v, ht, msftx, msfty, cf, *, dx, dy, periodic_x=False, periodic_y=False):
    """WRF w(k=1) (ny, nx) for ``u`` (nz, ny, nx+1), ``v`` (nz, ny+1, nx), ``ht``/``msft*`` (ny, nx)."""
    nz, ny, nx = u.shape[0], u.shape[1], u.shape[2] - 1
    shape3 = (nx + 3, nz + 1, ny + 3)

    def f3(field):
        out = np.zeros(shape3, dtype=np.float32, order="F")
        a = np.asarray(field, dtype=np.float32)
        out[1:1 + a.shape[2], :a.shape[0], 1:1 + a.shape[1]] = a.transpose(2, 0, 1)
        return out

    def f2(field):
        out = np.zeros((nx + 3, ny + 3), dtype=np.float32, order="F")
        out[1:nx + 1, 1:ny + 1] = np.asarray(field, dtype=np.float32).T
        if periodic_x:  # WRF periodic halo exchange: i = ids-1 -> ide-1, i = ide -> ids
            out[0, :] = out[nx, :]
            out[nx + 1, :] = out[1, :]
        if periodic_y:  # same in j
            out[:, 0] = out[:, ny]
            out[:, ny + 1] = out[:, 1]
        return out

    w = np.zeros(shape3, dtype=np.float32, order="F")
    one = np.float32(1.0)
    s = np.asarray([one / np.float32(dx), one / np.float32(dy), *[np.float32(c) for c in cf]], dtype=np.float32)
    args = [f2(ht), f3(u), f3(v), f2(msftx), f2(msfty)]
    lib = ctypes.CDLL(str(library))
    ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)
    lib.wsurf_oracle.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p] * 7
    lib.wsurf_oracle(nx, ny, nz, int(bool(periodic_x)) + 2 * int(bool(periodic_y)), ptr(w), *map(ptr, args), ptr(s))
    return np.ascontiguousarray(w[1:nx + 1, 0, 1:ny + 1].T)
