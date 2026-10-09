"""Pristine WRF ``dyn_em/module_big_step_utilities_em.F::w_damp`` behind a C-callable binary32 adapter.

The subroutine body is copied verbatim and hashed; only its environment is stubbed: the host module supplies
``w_alpha``/``w_beta`` (share/module_model_constants.F:88-89), ``degrad`` (the ``USE module_llxy`` line is the one
textual removal), ``wrf_err_message`` and no-op ``wrf_debug``/``get_current_*`` calls, and ``grid_config_rec_type``
(w_crit_cfl, zadvect_implicit, polar, fft_filter_lat, w_damping). Compiled with ``-cpp`` (OPTIMIZE_CFL_TEST unset,
like the pristine build). Called like ``module_em.F:738`` (rk_tendency).
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
SOURCE = WRF / "dyn_em/module_big_step_utilities_em.F"
CONSTANTS = WRF / "share/module_model_constants.F"


def build(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = SOURCE.read_text()
    routine = re.search(r"(?ims)^SUBROUTINE w_damp\b.*?^END SUBROUTINE w_damp\b", source).group(0)
    body = re.sub(r"(?im)^\s*USE module_llxy\s*$", "", routine, count=1)
    constants = CONSTANTS.read_text()
    w_alpha = re.search(r"(?im)w_alpha\s*=\s*([0-9.]+)", constants).group(1)
    w_beta = re.search(r"(?im)w_beta\s*=\s*([0-9.]+)", constants).group(1)
    text = (
        "module b_core_wdamp\nuse iso_c_binding\nimplicit none\n"
        f"real, parameter :: w_alpha = {w_alpha}\nreal, parameter :: w_beta = {w_beta}\n"
        "real, parameter :: degrad = 3.1415926/180.\ncharacter(len=256) :: wrf_err_message\n"
        "type grid_config_rec_type\nreal :: w_crit_cfl=1.0\ninteger :: zadvect_implicit=0\nlogical :: polar=.false.\n"
        "real :: fft_filter_lat=45.\ninteger :: w_damping=1\nend type\ncontains\n"
        "subroutine wrf_debug(level, msg)\ninteger :: level\ncharacter(len=*) :: msg\nend subroutine\n"
        "subroutine wrf_error_fatal(msg)\ncharacter(len=*) :: msg\nstop 1\nend subroutine\n"
        "subroutine get_current_time_string(s)\ncharacter(len=*) :: s\ns=''\nend subroutine\n"
        "subroutine get_current_grid_name(s)\ncharacter(len=*) :: s\ns=''\nend subroutine\n"
        + body + "\n"
        "subroutine wdamp_oracle(nx,ny,nz,rw_tend,u,v,ww,w,mut,c1f,c2f,rdnw,msfux,msfuy,msfvx,msfvy,s,cfl) bind(c)\n"
        "integer(c_int),value :: nx,ny,nz\n"
        "real(c_float),intent(inout) :: rw_tend(1:nx+1,1:nz+1,1:ny+1)\n"
        "real(c_float),intent(in) :: u(1:nx+1,1:nz+1,1:ny+1),v(1:nx+1,1:nz+1,1:ny+1),ww(1:nx+1,1:nz+1,1:ny+1)\n"
        "real(c_float),intent(in) :: w(1:nx+1,1:nz+1,1:ny+1),mut(1:nx+1,1:ny+1)\n"
        "real(c_float),intent(in) :: msfux(1:nx+1,1:ny+1),msfuy(1:nx+1,1:ny+1),msfvx(1:nx+1,1:ny+1),msfvy(1:nx+1,1:ny+1)\n"
        "real(c_float),intent(in) :: c1f(1:nz+1),c2f(1:nz+1),rdnw(1:nz+1),s(4)\n"
        "real(c_float),intent(out) :: cfl(2)\n"
        "type(grid_config_rec_type) :: config_flags\n"
        "config_flags%w_crit_cfl=s(4)\n"
        "call w_damp(rw_tend,cfl(1),cfl(2),u,v,ww,w,mut,c1f,c2f,rdnw,s(1),s(2),msfux,msfuy,msfvx,msfvy,s(3), &\n"
        "  config_flags, 1,nx+1,1,ny+1,1,nz+1, 1,nx+1,1,ny+1,1,nz+1, 1,nx+1,1,ny+1,1,nz+1)\n"
        "end subroutine\nend module\n"
    )
    (directory / "w_damp.F90").write_text(text)
    cmd = [os.environ.get("FC", "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"), "-cpp", "-O2", "-fPIC", "-shared",
           "-ffree-line-length-none", "-fcheck=bounds", "-ffp-contract=off", "-J", str(directory),
           str(directory / "w_damp.F90"), "-o", str(directory / "w_damp.so")]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    meta = dict(source=str(SOURCE), source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                routine_sha256=hashlib.sha256(routine.encode()).hexdigest(), w_alpha=w_alpha, w_beta=w_beta,
                compile=cmd, real_kind_bytes=4)
    (directory / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    return directory / "w_damp.so"


def evaluate(library, rw_tend, ww, w, mut, c1f, c2f, rdnw, *, dt, w_crit_cfl=1.0, dx=1000.0):
    """WRF w_damp on w-face fields (nz+1, ny, nx) and mut (ny, nx); returns the updated rw_tend (same layout).

    u/v enter only the horizontal-CFL diagnostic (not the tendency) and are passed as zeros."""
    nkw, ny, nx = rw_tend.shape
    nz = nkw - 1

    def f3(a):
        out = np.zeros((nx + 1, nz + 1, ny + 1), dtype=np.float32, order="F")
        out[:nx, :, :ny] = np.asarray(a, np.float32).transpose(2, 0, 1)
        return out

    def f2(a, fill=1.0):
        out = np.full((nx + 1, ny + 1), fill, dtype=np.float32, order="F")
        if a is not None:
            out[:nx, :ny] = np.asarray(a, np.float32).T
        return out

    def f1(a):
        out = np.zeros(nz + 1, dtype=np.float32)
        out[:len(a)] = np.asarray(a, np.float32)
        return out

    rw = f3(rw_tend)
    zeros = f3(np.zeros_like(rw_tend))
    one = np.float32(1.0)
    s = np.asarray([one / np.float32(dx), one / np.float32(dx), np.float32(dt), np.float32(w_crit_cfl)], np.float32)
    cfl = np.zeros(2, np.float32)
    args = [zeros, zeros, f3(ww), f3(w), f2(mut), f1(c1f), f1(c2f), f1(rdnw), f2(None), f2(None), f2(None), f2(None)]
    lib = ctypes.CDLL(str(library))
    ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)
    lib.wdamp_oracle.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 15
    lib.wdamp_oracle(nx, ny, nz, ptr(rw), *map(ptr, args), ptr(s), ptr(cfl))
    return np.ascontiguousarray(rw[:nx, :, :ny].transpose(1, 2, 0)), float(cfl[0])
