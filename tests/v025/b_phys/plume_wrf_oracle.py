"""Pristine-object DMP oracle with the retained column argument interface."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
WRF = Path('<USER_HOME>/src/wrf_pristine/WRF')
FC = '<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran'


def build(output, *, isolated=True, instrument=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = (ROOT / 'proofs/mynn_edmf/fortran_oracle/oracle.f90').read_text()
    if isolated:
        source = source.replace('do mode = 1, 0, -1', 'do mode = 1, 1, -1')
        start = source.index('       sqv(k)    =')
        stop = source.index('       qnc(k)    =', start)
        source = source[:start] + """       sqv(k) = qv(k)
       sqc(k) = qc(k)
       sqi(k) = 0.0_kind_phys
       sqs(k) = 0.0_kind_phys
       sqw(k) = qi(k)
       thl(k) = th(k)
       thv(k) = tk(k)
""" + source[stop:]
        source = source.replace('bl_mynn_edmf_mom = 0', 'bl_mynn_edmf_mom = 1')
        start = source.index('    call mynn_tendencies(')
        stop = source.index('    ! ---- dump', start)
        source = source[:start] + source[stop:]
        source = source.replace('    ! ---- dump', """    call wr_arr(u_out, 'edmf_s_awthl',s_awthl,kte+1)
    call wr_arr(u_out, 'edmf_s_awu',s_awu,kte+1)
    call wr_arr(u_out, 'edmf_s_awv',s_awv,kte+1)
    call wr_arr(u_out, 'edmf_edmf_qt',edmf_qt,kte)
    call wr_arr(u_out, 'edmf_edmf_qc',edmf_qc,kte)
    call wr_arr(u_out, 'edmf_edmf_thl',edmf_thl,kte)
    ! ---- dump""")
    (output / 'oracle.f90').write_text(source)
    phys = WRF / 'phys'
    physics_object = phys / 'module_bl_mynnedmf.o'
    if instrument:
        text = (phys / 'module_bl_mynnedmf.F').read_text()
        assert text.count('      UPA = UPA*adjustment') == 1
        text = text.replace('      UPA = UPA*adjustment',
                            "      write(*,'(A,ES24.16)') 'BP_LIMITER_ADJUSTMENT=', adjustment\n      UPA = UPA*adjustment")
        (output / 'instrumented.F').write_text(text)
        subprocess.run(['timeout','120',FC,'-cpp','-ffree-form','-ffree-line-length-none','-O2',
                        '-I'+str(phys),'-I'+str(WRF/'share'),'-c','instrumented.F','-o','instrumented.o'],
                       cwd=output, check=True, capture_output=True)
        physics_object = output / 'instrumented.o'
    objects = [physics_object, phys/'module_bl_mynnedmf_common.o',
               WRF/'share/module_model_constants.o', phys/'ccpp_kind_types.o']
    cmd = [FC,'-I'+str(output),'-I'+str(phys),'-ffree-line-length-none','-fcheck=bounds',
           'oracle.f90',*[str(p) for p in objects],'-o','oracle']
    result = subprocess.run(['timeout','60',*cmd],cwd=output,check=True,capture_output=True,text=True)
    (output/'build.log').write_text(result.stdout+result.stderr)
    provenance = dict(isolated_dmp=isolated, logging_only_instrumentation=instrument,
                      source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                      wrf_source_sha256=hashlib.sha256((phys/'module_bl_mynnedmf.F').read_bytes()).hexdigest(),
                      objects={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in objects},
                      command=cmd, surface_temperature_convention='TSK/exner(kts)')
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return output/'oracle'


def run(executable, directory, kwargs):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    a = {k:np.asarray(v) for k,v in kwargs.items()}
    scalar = lambda name:float(a[name].reshape(-1)[0])
    values = dict(nz=a['thl'].shape[-1],delt=scalar('dt'),dx=scalar('dx'),ust=scalar('ust'),
                  pblh=scalar('pblh'),xland=scalar('xland'),ts=scalar('ts'),th_sfc=scalar('ts'),
                  psfc=float(a['p'].reshape(-1)[0]),ps=float(a['p'].reshape(-1)[0]),
                  wspd=float(np.hypot(a['u'].reshape(-1)[0],a['v'].reshape(-1)[0])),
                  flt=scalar('flt'),fltv=scalar('fltv'),flq=scalar('flq'),flqv=scalar('flqv'))
    for label,key in [('u','u'),('v','v'),('w','w'),('th','thl'),('tk','thv'),('qv','sqv'),
                      ('qc','sqc'),('qi','sqw'),('p','p'),('exner','exner'),('rho','rho'),('dz','dz'),('qke','qke')]:
        values[label] = a[key].reshape(-1).tolist()
    for name in ('e_edmf','e_mom','e_mixs','e_cmix'):values[name] = 1
    for name in ('e_tke','e_mixqt'):values[name] = 0
    flat = '\n'.join(key+'='+(','.join(f'{x:.9e}' for x in v) if isinstance(v,list) else str(v)) for key,v in values.items())+'\n'
    (directory/'column_d03_12z.flat').write_text(flat)
    result = subprocess.run(['timeout','20',str(executable)],cwd=directory,check=True,capture_output=True,text=True)
    (directory/'run.log').write_text(result.stdout+result.stderr)
    outputs = {}
    for line in (directory/'oracle_out.txt').read_text().splitlines():
        if '=' in line:
            key,value=line.split('=',1)
            outputs[key] = np.array([float(x) for x in value.split(',')])
    return outputs,result.stdout
