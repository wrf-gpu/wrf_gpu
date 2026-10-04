"""Unmodified pristine WRF rhs_ph / pg_buoy_w / calc_cq / calc_ww_cp, C-callable REAL4 savepoints.

Bodies are copied verbatim from module_big_step_utilities_em.F and hashed; the wrapper binds the
literal module_em.F caller arguments (calc_ww_cp :163, calc_cq :171, rhs_ph :668 with ph_old = ph and
wwE = ww (zadvect_implicit=0), pg_buoy_w :730 with mu = grid%mu_2). Config: specified, PROD
h_sca_adv_order 5 (registry default), phi_adv_z 1, non_hydrostatic.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

WRF = Path("<USER_HOME>/src/wrf_pristine/WRF")
ROUTINES = ("calc_ww_cp", "calc_cq", "rhs_ph", "pg_buoy_w")
FIELDS3 = ("u", "v", "w", "ww", "ph", "phb", "ph_tend", "rw_tend", "p", "cqu", "cqv", "cqw",
           "q0", "qv", "qc", "qr", "qi", "qs", "qg")
FIELDS2 = ("mut", "muu", "muv", "mu", "mub", "msftx", "msfty", "msfux", "msfuy", "msfvx", "msfvx_inv", "msfvy")
VECTORS = ("c1h", "c2h", "c1f", "c2f", "fnm", "fnp", "rdnw", "rdn", "dnw")
SCALARS = ("rdx", "rdy", "cfn", "cfn1", "g")
CALLS = {  # literal module_em.F actuals (formal -> oracle array)
    "calc_ww_cp": {"mup": "mu"},
    "calc_cq": {},
    "rhs_ph": {"ph_old": "ph", "muuf": "muu", "muvf": "muv"},
    "pg_buoy_w": {"muf": "mu", "mubf": "mub"},
}


def build(directory):
    directory.mkdir(parents=True, exist_ok=True)
    source = (WRF / "dyn_em/module_big_step_utilities_em.F").read_text()
    routines = {n: re.search(r"(?ims)^\s*SUBROUTINE " + n + r"\b.*?^\s*END SUBROUTINE " + n + r"\b", source).group(0)
                for n in ROUTINES}
    flags = sorted(set(re.findall(r"config_flags%(\w+)", "\n".join(routines.values()))))
    ints = {"h_sca_adv_order": 5, "phi_adv_z": 1}
    config = "\n".join(f"integer :: {f}={ints[f]}" if f in ints else
                       f"logical :: {f}={'.true.' if f == 'specified' else '.false.'}" for f in flags)
    macros = source.split("MODULE module_big_step_utilities_em")[0]
    header = (macros + "module wrf_dyn_real_oracle\nuse iso_c_binding\nimplicit none\n"
              "integer,parameter :: PARAM_FIRST_SCALAR=2\nreal,parameter :: g=9.81  ! module_model_constants\ntype grid_config_rec_type\n" + config + "\nend type\ncontains\n")
    wrapper = f'''subroutine oracle(nx,ny,nz,phase,f3,f2,v,s) bind(c)
integer(c_int),value :: nx,ny,nz,phase
real(c_float) :: f3(nx+3,nz+1,ny+3,{len(FIELDS3)}),f2(nx+3,ny+3,{len(FIELDS2)})
real(c_float) :: v(nz+1,{len(VECTORS)}),s({len(SCALARS)})
type(grid_config_rec_type) :: config_flags
select case(phase)
'''
    bounds = dict(ids="1", ide="nx+1", jds="1", jde="ny+1", kds="1", kde="nz+1", ims="0", ime="nx+2",
                  jms="0", jme="ny+2", kms="1", kme="nz+1", its="1", ite="nx+1", jts="1", jte="ny+1",
                  kts="1", kte="nz+1", config_flags="config_flags", non_hydrostatic=".true.", n_moist="7",
                  moist=f"f3(:,:,:,{FIELDS3.index('q0') + 1}:{FIELDS3.index('qg') + 1})")
    for phase, name in enumerate(ROUTINES, 1):
        signature = re.sub(r"![^\n]*", "", routines[name][:routines[name].index(")") + 1]).replace("&", "")
        actual = []
        for arg in [x.strip().lower() for x in signature[signature.index("(") + 1:-1].split(",")]:
            arg = CALLS[name].get(arg, arg)
            if arg in FIELDS3: actual.append(f"f3(:,:,:,{FIELDS3.index(arg) + 1})")
            elif arg in FIELDS2: actual.append(f"f2(:,:,{FIELDS2.index(arg) + 1})")
            elif arg in VECTORS: actual.append(f"v(:,{VECTORS.index(arg) + 1})")
            elif arg in SCALARS: actual.append(f"s({SCALARS.index(arg) + 1})")
            else: actual.append(bounds[arg])
        wrapper += f"case({phase})\ncall {name}( &\n" + ", &\n".join(actual) + ")\n"
    wrapper += "end select\nend subroutine\nend module\n"
    path = directory / "oracle.f90"; path.write_text(header + "\n".join(routines.values()) + "\n" + wrapper)
    command = [os.environ.get("FC", "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran"), "-O2", "-cpp", "-shared", "-fPIC",
               "-ffree-line-length-none", "-ffp-contract=off", "-fcheck=bounds", "-J", str(directory), str(path),
               "-o", str(directory / "oracle.so")]
    cp = subprocess.run(command, capture_output=True, text=True)
    if cp.returncode: raise RuntimeError(cp.stderr)
    manifest = dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    routine_sha256={n: hashlib.sha256(s.encode()).hexdigest() for n, s in routines.items()},
                    caller=str(WRF / "dyn_em/module_em.F"), calls=CALLS, config=config, REAL_bytes=4, command=command)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return directory / "oracle.so"


class Oracle:
    def __init__(self, path, shape):
        self.nz, self.ny, self.nx = nz, ny, nx = shape
        self.f3 = np.zeros((nx + 3, nz + 1, ny + 3, len(FIELDS3)), np.float32, order="F")
        self.f2 = np.ones((nx + 3, ny + 3, len(FIELDS2)), np.float32, order="F")
        self.v = np.zeros((nz + 1, len(VECTORS)), np.float32, order="F")
        self.s = np.zeros(len(SCALARS), np.float32)
        self.fn = ctypes.CDLL(str(path)).oracle
        self.fn.argtypes = [ctypes.c_int] * 4 + [ctypes.c_void_p] * 4; self.fn.restype = None

    def set(self, name, value):
        value = np.asarray(value, np.float32); nz, ny, nx = self.nz, self.ny, self.nx
        if name in FIELDS3:
            k, y, x = value.shape
            self.f3[:, :, :, FIELDS3.index(name)] = np.pad(value, ((0, nz + 1 - k), (1, ny + 2 - y), (1, nx + 2 - x)), mode="edge").transpose(2, 0, 1)
        elif name in FIELDS2:
            y, x = value.shape
            self.f2[:, :, FIELDS2.index(name)] = np.pad(value, ((1, ny + 2 - y), (1, nx + 2 - x)), mode="edge").T
        elif name in VECTORS:
            self.v[:, VECTORS.index(name)] = np.pad(value.reshape(-1), (0, nz + 1 - value.size), mode="edge")
        else: self.s[SCALARS.index(name)] = value

    def field(self, name, shape):
        if name in FIELDS3:
            k, y, x = shape
            return self.f3[1:x + 1, :k, 1:y + 1, FIELDS3.index(name)].transpose(1, 2, 0).copy()
        y, x = shape
        return self.f2[1:x + 1, 1:y + 1, FIELDS2.index(name)].T.copy()

    def run(self, name):
        self.fn(self.nx, self.ny, self.nz, ROUTINES.index(name) + 1, *[a.ctypes.data for a in (self.f3, self.f2, self.v, self.s)])
