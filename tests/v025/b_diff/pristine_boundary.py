"""Unchanged pristine WRF lateral-boundary routines with a REAL4 C adapter."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

WRF = Path('<USER_HOME>/src/wrf_pristine/WRF')
EM_ROUTINES = ('relax_bdy_dry', 'relax_bdy_scalar', 'spec_bdy_dry',
               'spec_bdyupdate_ph', 'mass_weight', 'lbc_fcx_gcx')
SHARE_ROUTINES = ('relax_bdytend', 'relax_bdytend_tile', 'relax_bdytend_core',
                  'spec_bdytend', 'spec_bdyupdate', 'flow_dep_bdy')
# New phases are appended so existing phase indices stay stable.
PHASES = ('relax_bdy_dry', 'relax_bdy_scalar', 'spec_bdy_dry',
          'spec_bdyupdate_ph', 'spec_bdytend', 'spec_bdyupdate', 'relax_bdytend',
          'flow_dep_bdy')
FIELDS3 = ('ru', 'rv', 'ph', 't', 'w', 'ru_tendf', 'rv_tendf', 'ph_tendf',
           't_tendf', 'rw_tendf', 'scalar', 'scalar_tend', 'ph_save', 'field', 'field_tend')
FIELDS2 = ('mu', 'mut', 'mu_tend', 'muts')
VECTORS = ('c1h', 'c2h', 'c1f', 'c2f')
VARIABLES = ('u', 'v', 'ph', 't', 'w', 'mu', 'scalar')


def build(directory):
    directory.mkdir(parents=True, exist_ok=True)
    sources = {n: (WRF/n).read_text() for n in ('dyn_em/module_bc_em.F', 'share/module_bc.F')}
    routines = {}
    for source, names in ((sources['dyn_em/module_bc_em.F'], EM_ROUTINES),
                          (sources['share/module_bc.F'], SHARE_ROUTINES)):
        for name in names:
            routines[name] = re.search(r'(?ims)^\s*SUBROUTINE '+name+r'\b.*?^\s*END SUBROUTINE '+name+r'\b', source).group(0)
    header = '''module bdiff_boundary_oracle
use iso_c_binding
implicit none
type grid_config_rec_type
logical :: nested=.false.,periodic_x=.false.
end type
contains
'''
    wrapper = f'''subroutine oracle(nx,ny,nz,phase,nested,tag,f3,f2,v,bx,by,s) bind(c)
integer(c_int),value :: nx,ny,nz,phase,nested,tag
real(c_float) :: f3(nx+3,nz+1,ny+3,{len(FIELDS3)}),f2(nx+3,ny+3,{len(FIELDS2)})
real(c_float) :: v(nz+1,4),bx(ny+3,nz+1,5,2,7,2),by(nx+3,nz+1,5,2,7,2),s(3)
real :: fcx(5),gcx(5)
character :: variable
type(grid_config_rec_type) :: config_flags
integer :: ktop
config_flags%nested=nested/=0
ktop=nz+1
if(tag==5.and.phase>=5)ktop=1
variable='q'
if(tag==1)variable='u'
if(tag==2)variable='v'
if(tag==3)variable='h'
if(tag==4)variable='t'
if(tag==5)variable='m'
if(tag==7)variable='w'
fcx=0.;gcx=0.
call lbc_fcx_gcx(fcx,gcx,5,1,4,s(1),0.,.not.config_flags%nested,config_flags%nested)
select case(phase)
'''
    bounds = dict(ids='1', ide='nx+1', jds='1', jde='ny+1', kds='1', kde='ktop',
                  ims='0', ime='nx+2', jms='0', jme='ny+2', kms='1', kme='nz+1',
                  ips='1', ipe='nx+1', jps='1', jpe='ny+1', kps='1', kpe='nz+1',
                  its='1', ite='nx+1', jts='1', jte='ny+1', kts='1', kte='ktop',
                  config_flags='config_flags', spec_bdy_width='5', spec_zone='1', relax_zone='4',
                  dtbc='s(2)', dt='s(3)', dts='s(3)', fcx='fcx', gcx='gcx',
                  variable_in='variable', c1='v(:,3)', c2='v(:,4)')
    for phase, name in enumerate(PHASES, 1):
        signature = re.sub(r'![^\n]*', '', routines[name][:routines[name].index(')')+1]).replace('&', '')
        args = [n.strip().lower() for n in signature[signature.index('(')+1:-1].split(',')]
        actual = []
        for arg in args:
            arg = {'field_saved': 'ph_save'}.get(arg, arg)
            arg = arg.replace('field_bdy_tend_', 'field_bt').replace('field_bdy_', 'field_b')
            if name == 'spec_bdy_dry':
                arg = {n: n+'f' for n in ('ru_tend', 'rv_tend', 'ph_tend', 't_tend', 'rw_tend')}.get(arg, arg)
            if name == 'spec_bdyupdate_ph':
                arg = {'field': 'ph', 'field_tend': 'ph_tendf'}.get(arg, arg)
            if name == 'flow_dep_bdy':
                # WRF passes ru_m/rv_m: the staggered momentum slots.
                arg = {'u': 'ru', 'v': 'rv'}.get(arg, arg)
            match = re.fullmatch(r'(u|v|ph|t|w|mu|scalar|field)_b(t?)(xs|xe|ys|ye)', arg)
            if match:
                var, tend, side = match.groups()
                idx = VARIABLES.index(var)+1 if var != 'field' else 7
                axis = 'bx' if side.startswith('x') else 'by'
                side_idx = '1' if side.endswith('s') else '2'
                level = '1:1' if var == 'mu' else '1:ktop' if var == 'field' else ':'
                actual.append(f'{axis}(:,{level},:,{side_idx},{idx},{2 if tend else 1})')
            elif arg in FIELDS3:
                actual.append(f'f3(:,:,:,{FIELDS3.index(arg)+1})')
            elif arg in FIELDS2:
                actual.append(f'f2(:,:,{FIELDS2.index(arg)+1})')
            elif arg in VECTORS:
                actual.append(f'v(:,{VECTORS.index(arg)+1})')
            elif arg in bounds:
                actual.append(bounds[arg])
            else:
                raise KeyError((name, arg))
        wrapper += f'case({phase})\ncall {name}( &\n'+', &\n'.join(actual)+')\n'
    wrapper += 'end select\nend subroutine\nend module\n'
    path = directory/'oracle.f90'
    path.write_text(header+'\n'.join(routines.values())+'\n'+wrapper)
    cmd = [os.environ.get('FC', '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'),
           '-O2', '-cpp', '-shared', '-fPIC', '-ffree-line-length-none', '-ffp-contract=off',
           '-fcheck=bounds', '-J', str(directory), str(path), '-o', str(directory/'oracle.so')]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    manifest = dict(source_sha256={n: hashlib.sha256(s.encode()).hexdigest() for n, s in sources.items()},
                    routine_sha256={n: hashlib.sha256(s.encode()).hexdigest() for n, s in routines.items()},
                    caller_sha256=hashlib.sha256((WRF/'dyn_em/solve_em.F').read_bytes()).hexdigest(),
                    REAL_bytes=4, command=cmd, variables=VARIABLES,
                    mapping='separate Fortran BX/BY leading strides; boundary two axes value/tendency; shared wrapper widths5/spec1/relax4',
                    limitations='specified/nested nonperiodic operators; actual operand conventions recorded in probe')
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return directory/'oracle.so'


class Oracle:
    def __init__(self, path, shape, *, nested=False, dt=54., dtbc=0., dts=13.5):
        self.nz, self.ny, self.nx = shape
        nz, ny, nx = shape
        self.f3 = np.zeros((nx+3, nz+1, ny+3, len(FIELDS3)), np.float32, order='F')
        self.f2 = np.zeros((nx+3, ny+3, len(FIELDS2)), np.float32, order='F')
        self.v = np.zeros((nz+1, 4), np.float32, order='F')
        self.bx = np.zeros((ny+3, nz+1, 5, 2, 7, 2), np.float32, order='F')
        self.by = np.zeros((nx+3, nz+1, 5, 2, 7, 2), np.float32, order='F')
        self.s = np.asarray([dt, dtbc, dts], np.float32)
        self.nested = int(nested)
        self.library = ctypes.CDLL(str(path))
        self.fn = self.library.oracle
        self.fn.argtypes = [ctypes.c_int]*6+[ctypes.c_void_p]*6
        self.fn.restype = None

    def set(self, name, value):
        a = np.asarray(value, np.float32)
        nz, ny, nx = self.nz, self.ny, self.nx
        if name in FIELDS3:
            k, y, x = a.shape
            self.f3[:, :, :, FIELDS3.index(name)] = np.pad(a, ((0, nz+1-k), (1, ny+2-y), (1, nx+2-x)), mode='edge').transpose(2, 0, 1)
        elif name in FIELDS2:
            y, x = a.shape
            self.f2[:, :, FIELDS2.index(name)] = np.pad(a, ((1, ny+2-y), (1, nx+2-x)), mode='edge').T
        else:
            self.v[:, VECTORS.index(name)] = np.pad(a, (0, nz+1-a.size), mode='edge')

    def boundary(self, name, values, tendencies):
        index = VARIABLES.index(name)
        for record, a in enumerate((values, tendencies)):
            a = np.asarray(a, np.float32)
            for side in range(4):
                target = self.bx if side < 2 else self.by
                n = self.ny+1 if side < 2 else self.nx+1
                source = a[side, :5, :, :n].transpose(2, 1, 0)
                target[:, :, :, side % 2, index, record] = np.pad(source, ((1, target.shape[0]-source.shape[0]-1),
                    (0, self.nz+1-source.shape[1]), (0, 5-source.shape[2])), mode='edge')

    def field(self, name, shape):
        if name in FIELDS3:
            k, y, x = shape
            return self.f3[1:x+1, :k, 1:y+1, FIELDS3.index(name)].transpose(1, 2, 0).copy()
        y, x = shape
        return self.f2[1:x+1, 1:y+1, FIELDS2.index(name)].T.copy()

    def run(self, phase, tag=6):
        self.fn(self.nx, self.ny, self.nz, PHASES.index(phase)+1, self.nested, tag,
                *[a.ctypes.data for a in (self.f3, self.f2, self.v, self.bx, self.by, self.s)])
