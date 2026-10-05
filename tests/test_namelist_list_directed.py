"""List-directed WRF input semantics against an independent Fortran reader."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from gpuwrf.io.gen2_accessor import Gen2Run, parse_namelist


FORTRAN = '''program reader
implicit none
logical :: top_lid(3) = .false.
integer :: e_we(3) = [101,102,103], mp_physics(3) = [8,8,8]
real(8) :: dx(3) = [9000.d0,3000.d0,1000.d0]
character(64) :: input_outname(3) = [character(64) :: 'old1','old2','old3']
character(1024) :: path
integer :: i, status
namelist /test_fields/ top_lid, e_we, dx, mp_physics, input_outname
call get_command_argument(1,path)
open(10,file=trim(path),status='old',action='read')
read(10,nml=test_fields,iostat=status)
if (status /= 0) stop 2
close(10)
write(*,'(3L2)') top_lid
write(*,*) e_we
write(*,'(3ES25.17E3)') dx
write(*,*) mp_physics
do i=1,3
write(*,'(A)') trim(input_outname(i))
end do
end program
'''


@pytest.fixture(scope='module')
def reader(tmp_path_factory):
    compiler = os.environ.get('FC') or shutil.which('gfortran')
    assert compiler and Path(compiler).exists(), 'Fortran namelist oracle compiler required'
    root = tmp_path_factory.mktemp('namelist-fortran')
    source = root/'reader.f90'
    source.write_text(FORTRAN)
    exe = root/'reader'
    subprocess.run([compiler,str(source),'-o',str(exe)],check=True,capture_output=True,text=True,timeout=30)
    return exe


DEFAULTS = {'top_lid':[False]*3,'e_we':[101,102,103],
            'dx':[9000.,3000.,1000.],'mp_physics':[8]*3,
            'input_outname':['old1','old2','old3']}


def resolved(nml, key):
    raw = nml.get('test_fields',{}).get(key, [])
    vals = raw if isinstance(raw,list) else [raw]
    return [vals[i] if i<len(vals) and vals[i] is not None else DEFAULTS[key][i]
            for i in range(3)]


@pytest.mark.parametrize('body', [
    'top_lid=.true.,,.true.',
    'top_lid=3*.false.',
    'top_lid=.false.,2*.true.',
    'top_lid=2*,.true.',
    'e_we=,2*181',
    'e_we=129,2*',
    'dx=2*3000.d0,1.d3',
    'dx=,6000,',
    'mp_physics=3*8',
    'mp_physics=,2*4',
    "input_outname=2*'wrf!out,a', 'it''s a file'",
    "input_outname=1*,'x',",
    'e_we=1,2,3, e_we=2*,4',
    "input_outname='a!b',\n 'quote''x',\n 'three' ! actual comment",
    'e_we=1 2 3, dx=9000 3000 1000',
])
def test_lists_match_fortran(reader,tmp_path,body):
    path=tmp_path/'namelist.input'
    path.write_text('&test_fields\n'+body+'\n/\n')
    result=subprocess.run([str(reader),str(path)],check=True,capture_output=True,text=True,timeout=5)
    lines=result.stdout.splitlines()
    expected={'top_lid':[v=='T' for v in lines[0].split()],
              'e_we':[int(v) for v in lines[1].split()],
              'dx':[float(v) for v in lines[2].split()],
              'mp_physics':[int(v) for v in lines[3].split()],
              'input_outname':lines[4:7]}
    actual=parse_namelist(path)
    for key in DEFAULTS:
        assert resolved(actual,key)==expected[key], (body,key,actual,expected)
        # Exercise the real grid accessor's null/default handling too.
        consumed = [Gen2Run._nml_list_value(actual['test_fields'], key, i, DEFAULTS[key][i])
                    for i in range(3)]
        assert consumed == expected[key], (body,key,consumed,expected)


@pytest.mark.parametrize('body', [
    'top_lid=3*.flase.', 'e_we=0*10', 'e_we=-2*10', 'e_we=12abc',
    "input_outname='unterminated", 'e_we(2)=10', 'broken statement',
])
def test_malformed_or_unsupported_input_is_loud(tmp_path,body):
    path=tmp_path/'namelist.input'
    path.write_text('&test_fields\n'+body+'\n/\n')
    with pytest.raises(ValueError):
        parse_namelist(path)


def test_nulls_preserve_positions_and_prior_assignments(tmp_path):
    path=tmp_path/'namelist.input'
    path.write_text('&domains e_we=,,181, dx=3*, e_we=129,2*, /\n')
    assert parse_namelist(path)=={'domains':{'e_we':[129,None,181],'dx':[None,None,None]}}


def test_standard_inputs_keep_exact_existing_values(tmp_path):
    path=tmp_path/'namelist.input'
    path.write_text("&domains\n e_we=43,\n dx=3000,3000,\n/\n"
                    "&physics mp_physics=8, icloud=.true., names='one','two', /\n")
    assert parse_namelist(path)=={'domains':{'e_we':43,'dx':[3000,3000]},
                                 'physics':{'mp_physics':8,'icloud':True,'names':['one','two']}}
