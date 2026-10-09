"""Pristine WRF ``share/module_bc.F::zero_grad_bdy`` behind a C-callable binary32 adapter.

The subroutine body is copied verbatim from the pristine tree and hashed; only ``grid_config_rec_type``
(periodic_x) is stubbed. Called like ``solve_em.F:1599`` for specified domains: ``zero_grad_bdy(grid%w_2, 'w', ...)``.
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
SOURCE = WRF / "share/module_bc.F"


def build(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = SOURCE.read_text()
    routine = re.search(r"(?ims)^\s*SUBROUTINE zero_grad_bdy\b.*?^\s*END SUBROUTINE zero_grad_bdy\b", source).group(0)
    text = (
        "module b_core_zero_grad\nuse iso_c_binding\nimplicit none\n"
        "type grid_config_rec_type\nlogical :: periodic_x=.false.\nend type\n"
        "contains\n" + routine + "\n"
        "subroutine zero_grad_oracle(nx,ny,nkw,spec_zone,periodic,field) bind(c)\n"
        "integer(c_int),value :: nx,ny,nkw,spec_zone,periodic\n"
        "real(c_float),intent(inout) :: field(1:nx,1:nkw,1:ny)\n"
        "type(grid_config_rec_type) :: config_flags\n"
        "config_flags%periodic_x=periodic/=0\n"
        "call zero_grad_bdy(field,'w',config_flags,spec_zone, &\n"
        "  1,nx+1,1,ny+1,1,nkw, 1,nx,1,ny,1,nkw, 1,nx,1,ny,1,nkw, 1,nx+1,1,ny+1,1,nkw)\n"
        "end subroutine\nend module\n"
    )
    (directory / "zero_grad.f90").write_text(text)
    cmd = [os.environ.get("FC", "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"), "-O2", "-fPIC", "-shared",
           "-ffree-line-length-none", "-fcheck=bounds", "-J", str(directory), str(directory / "zero_grad.f90"),
           "-o", str(directory / "zero_grad.so")]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    meta = dict(source=str(SOURCE), source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                routine_sha256=hashlib.sha256(routine.encode()).hexdigest(), compile=cmd, real_kind_bytes=4)
    (directory / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    return directory / "zero_grad.so"


def evaluate(library, w, *, spec_zone, periodic_x=False):
    """WRF zero_grad_bdy of a w-type field ``w`` (nz+1, ny, nx); returns the same layout."""
    nkw, ny, nx = w.shape
    field = np.asfortranarray(np.asarray(w, dtype=np.float32).transpose(2, 0, 1))
    lib = ctypes.CDLL(str(library))
    lib.zero_grad_oracle.argtypes = [ctypes.c_int] * 5 + [ctypes.c_void_p]
    lib.zero_grad_oracle(nx, ny, nkw, int(spec_zone), int(bool(periodic_x)), field.ctypes.data_as(ctypes.c_void_p))
    return np.ascontiguousarray(field.transpose(1, 2, 0))
