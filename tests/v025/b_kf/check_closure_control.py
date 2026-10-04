"""Compare native closure decisions with the literal pristine WRF control block.

This is an independent control oracle, not a meteorological speed fixture.
The input cases cover rejection, retry, AINCMX, convergence, and iteration ten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(a.source / 'src'))
import jax
import jax.numpy as jnp
import numpy as np
from gpuwrf.kernels.phys_kf_column import _closure_control
assert {d.platform for d in jax.devices()} == {'cpu'}
truth = Path('<USER_HOME>/src/wrf_pristine/WRF/phys/module_cu_kfeta.F')
source = truth.read_text()
start = source.index('          IF(NOITR.EQ.1)THEN')
end = source.index('        ENDDO iter', start)
block = '\n'.join(line for line in source[start:end].splitlines()
                  if line.strip() and not line.lstrip().startswith('!'))
block = block.replace('EXIT iter', 'done=1; EXIT iter')
block = '\n'.join('            done=1; EXIT iter' if line.strip() == 'EXIT' else line
                  for line in block.splitlines())
block = block.replace('            TDER=TDER2*AINC', '            rescaled=1\n            TDER=TDER2*AINC')
prefix = """program control_oracle
implicit none
real :: x(10),y(7)
integer :: ios
open(11,file='control_inputs.bin',access='stream',form='unformatted',status='old')
open(12,file='control_expected.bin',access='stream',form='unformatted',status='replace')
do
 read(11,iostat=ios)x
 if(ios<0)exit
 call decide(x,y)
 write(12)y
enddo
contains
subroutine decide(x,y)
real,intent(in)::x(10)
real,intent(out)::y(7)
integer::done,rescaled,ishall,noitr,ncount,nk,ltop,initial_count
real::ainc,aincold,fabeold,abe,abeg,aincmx,fabe,stab,dabe,dfda,tder,tder2,pptflx,pptfl2
real::umf(1),dmf(1),detlq(1),detic(1),udr(1),uer(1),der(1),ddr(1)
real::umf2(1),dmf2(1),detlq2(1),detic2(1),udr2(1),uer2(1),der2(1),ddr2(1)
initial_count=nint(x(1));ishall=nint(x(2));noitr=nint(x(3))
ainc=x(4);aincold=x(5);fabeold=x(6);abe=x(7);abeg=x(8);aincmx=x(9)
stab=.95;fabe=1.;done=0;rescaled=0;ltop=1;tder2=1.;pptfl2=1.
umf2=1.;dmf2=1.;detlq2=1.;detic2=1.;udr2=1.;uer2=1.;der2=1.;ddr2=1.
if(ishall==1)then
 done=1
else
call iteration
endif
y=[ainc,aincold,fabeold,real(noitr),real(done),real(merge(1,0,ishall==2)),real(rescaled)]
contains
"""
# Fortran cannot nest contained subprograms here; run the exact block in its own
# internal subroutine with output intent variables, then pack its results.
prefix = prefix[:prefix.index('if(ishall==1)then')] + """
if(ishall==1)then
 done=1
else
iter: do ncount=initial_count,initial_count
"""
suffix = """
enddo iter
endif
y=[ainc,aincold,fabeold,real(noitr),real(done),real(merge(1,0,ishall==2)),real(rescaled)]
end subroutine
end program
"""
# WRF RETURN exits decide before packing; pack its observable control variables
# directly at each RETURN, preserving the source's branch ordering.
pack = 'y=[ainc,aincold,fabeold,real(noitr),real(done),real(merge(1,0,ishall==2)),real(rescaled)]'
block = block.replace('RETURN', 'done=1; ' + pack + '; RETURN')
driver = prefix + block + suffix
(a.output / 'control_oracle.f90').write_text(driver)
cases = [
 ('rescale', 1,0,0,1.,1.,1.,1000.,300.,1.5),
 ('reject_cape', 1,0,0,1.,1.,1.,1000.,1100.,2.),
 # A negative remaining-CAPE control state reaches the small-AINC RETURN.
 # This exercises source ordering only; it is not a meteorological fixture.
 ('reject_small', 1,0,0,.06,.06,1.,1000.,-200.,2.),
 ('nochange_retry', 2,0,0,.5,.50005,.3,1000.,200.,2.),
 ('dfda_retry', 2,0,0,.8,.6,.2,1000.,300.,2.),
 ('noitr_exit', 3,0,1,.6,.6,.2,1000.,300.,2.),
 ('aincmx_exit', 1,0,0,1.,1.,1.,1000.,300.,1.),
 ('converged', 1,0,0,.7,.7,1.,1000.,50.,2.),
 ('final_iteration', 10,0,0,.7,.6,.4,1000.,300.,2.),
 ('last_retry', 10,0,0,.8,.6,.2,1000.,300.,2.),
 ('tiny_dabe_retry', 1,0,0,.7,.7,1.,.0005,.00045,2.),
 ('zero_cape_converged', 2,0,0,.8,.7,.4,1000.,0.,2.),
 ('shallow_exit', 1,1,0,.7,.7,1.,1000.,300.,2.),
]
x = np.asarray([list(c[1:]) + [0.] for c in cases], '<f4')
x.tofile(a.output / 'control_inputs.bin')
fc = '<USER_HOME>/miniconda3/envs/wrfbuild/bin/x86_64-conda-linux-gnu-gfortran'
subprocess.run([fc,'-O2','-ffree-line-length-none','control_oracle.f90','-o','control_oracle'],
               cwd=a.output, check=True, timeout=30)
subprocess.run([str(a.output.resolve()/'control_oracle')],cwd=a.output,check=True,timeout=30)
expected = np.fromfile(a.output / 'control_expected.bin',dtype='<f4').reshape(-1,7)
actual = []
for row in x:
    ncount,shallow,noitr,ainc,old,fabold,abe,abeg,aincmx = row[:9]
    fab = np.float32(abeg/abe)
    dabe = np.maximum(np.float32(abe-abeg),np.float32(.1)*abe)
    with jax.enable_x64(False):
        c = _closure_control(jnp.int32(ncount),jnp.bool_(shallow),jnp.int32(noitr),
                             *map(jnp.float32,(ainc,old,fabold,fab,abe,dabe,aincmx)))
    actual.append([float(c[key]) for key in ('AINC','AINCOLD','FABEOLD','NOITR','done','rejected','rescale')])
actual=np.asarray(actual)
checks=np.max(np.abs(actual-expected),axis=1)<=1e-6
actions = [(0,0,0,1), (0,1,1,0), (0,1,1,0), (1,0,0,0),
           (1,0,0,0), (1,1,0,0), (0,1,0,0), (0,1,0,0),
           (0,1,0,0), (1,0,0,0), (1,0,0,0), (0,1,0,0), (0,1,0,0)]
# NOITR/done/rejected/rescale must reach the named branch, not merely agree.
checks &= np.all(actual[:,3:] == np.asarray(actions),axis=1)
record=dict(verdict='PASS' if checks.all() else 'FAIL',
            source_sha256=hashlib.sha256(truth.read_bytes()).hexdigest(),
            block_sha256=hashlib.sha256(source[start:end].encode()).hexdigest(),
            rows=[dict(case=c[0],actual=row.tolist(),wrf=t.tolist(),pass_=bool(ok))
                  for c,row,t,ok in zip(cases,actual,expected,checks)])
(a.output/'control_receipt.json').write_text(json.dumps(record,indent=2)+'\n')
print(record['verdict'],int(checks.sum()),len(checks))
sys.exit(0 if checks.all() else 1)
