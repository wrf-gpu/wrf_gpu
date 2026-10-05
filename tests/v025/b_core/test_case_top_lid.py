"""Case top-lid wiring follows the pristine WRF Registry logical defaults.

These are configuration/dispatch gates; physical acceptance is against the
original CPU-WRF Swiss and WN3 forecasts, including top-face W stability.
"""
import ast
from datetime import datetime, timezone
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import subprocess

from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.io.gen2_accessor import parse_namelist
from gpuwrf.runtime.aot_cheap_key import static_config_hash
from tests.dynamics.test_diffopt1_smagorinsky_integration import _build_grid, _namelist

ROOT = Path(__file__).resolve().parents[3]


def _build(**options):
    grid = _build_grid(ny=4, nx=5, nz=4, dx=3000.)
    return pipeline._make_namelist(
        grid=grid, tendencies=_namelist(grid).tendencies, metrics=grid.metrics,
        dt_s=18., parent_dt_s=None, run_start=datetime(2023, 1, 15, tzinfo=timezone.utc),
        radiation_static=None, cu_physics=0, **options,
    )


def test_registry_and_real_swiss_default_are_open():
    registry = ROOT / 'data/wrf_pristine/WRF/Registry/Registry.EM_COMMON'
    line = next(line for line in registry.read_text().splitlines()
                if line.split()[:3] == ['rconfig', 'logical', 'top_lid'])
    assert line.split()[3:6] == ['namelist,dynamics', 'max_domains', '.false.']
    run = SimpleNamespace(namelist=parse_namelist(ROOT / 'examples/switzerland_d01/namelist.input'))
    assert 'top_lid' not in run.namelist['dynamics']
    assert pipeline._domain_top_lid(run, 'd01') is False
    assert _build().top_lid is False


@pytest.mark.parametrize('selection', (False, True))
def test_explicit_lid_value_reaches_operational_program(selection):
    assert _build(top_lid=selection).top_lid is selection


def test_parser_preserves_per_domain_logicals_and_omitted_defaults(tmp_path):
    path = tmp_path / 'namelist.input'
    path.write_text('&dynamics\n top_lid = .true., .false., .true.,\n/\n')
    run = SimpleNamespace(namelist=parse_namelist(path))
    assert [pipeline._domain_top_lid(run, f'd{i:02d}') for i in (1,2,3,4)] == [True,False,True,False]
    path.write_text('&dynamics\n top_lid = .true.,\n/\n')
    run = SimpleNamespace(namelist=parse_namelist(path))
    assert [pipeline._domain_top_lid(run, f'd{i:02d}') for i in (1,2,3)] == [True,False,False]


@pytest.fixture(scope='module')
def lid_reader(tmp_path_factory):
    source = '''program reader
implicit none
logical :: top_lid(3) = .false.
character(1024) :: path
namelist /dynamics/ top_lid
call get_command_argument(1,path)
open(10,file=trim(path),status='old',action='read')
read(10,nml=dynamics)
close(10)
write(*,'(3L2)') top_lid
end program
'''
    root=tmp_path_factory.mktemp('lid-fortran')
    (root/'reader.f90').write_text(source)
    compiler=os.environ['FC']
    exe=root/'reader'
    subprocess.run([compiler,str(root/'reader.f90'),'-o',str(exe)],check=True,
                   capture_output=True,text=True,timeout=30)
    return exe


@pytest.mark.parametrize('value', ('.true.,,.true.', '3*.false.', '.false.,2*.true.', '2*,.true.'))
def test_actual_lid_loader_matches_fortran_nulls_and_repeats(lid_reader,tmp_path,value):
    path=tmp_path/'namelist.input'
    path.write_text('&dynamics\n top_lid='+value+',\n/\n')
    p=subprocess.run([str(lid_reader),str(path)],capture_output=True,text=True,check=True,timeout=5)
    expected=[v=='T' for v in p.stdout.split()]
    run=SimpleNamespace(namelist=parse_namelist(path))
    assert [pipeline._domain_top_lid(run,f'd{i:02d}') for i in (1,2,3)]==expected


@pytest.mark.parametrize('value', (1,0,'false','3*.false.','.flase.'))
def test_lid_logical_type_errors_are_loud(value):
    run=SimpleNamespace(namelist={'dynamics':{'top_lid':value}})
    with pytest.raises(ValueError, match='Fortran logical'):
        pipeline._domain_top_lid(run,'d01')


def test_lid_is_part_of_compiled_static_configuration():
    assert static_config_hash(_build(top_lid=False)) != static_config_hash(_build(top_lid=True))


def test_actual_loader_reads_case_lid_and_records_resolved_value():
    # Deleting the threading hunk must fail even if _make_namelist's default
    # happens to equal this particular case. Explicit true is separately tested.
    tree = ast.parse(inspect.getsource(pipeline._load_domains))
    calls = [node for node in ast.walk(tree) if isinstance(node,ast.Call)
             and isinstance(node.func,ast.Name) and node.func.id == '_make_namelist']
    assert len(calls) == 1
    keywords = {k.arg:ast.unparse(k.value) for k in calls[0].keywords}
    assert keywords['top_lid'] == '_domain_top_lid(run, name)'
    metadata = [node for node in ast.walk(tree) if isinstance(node,ast.Dict)
                for key,value in zip(node.keys,node.values)
                if isinstance(key,ast.Constant) and key.value == 'top_lid'
                and ast.unparse(value) == 'bool(namelist.top_lid)']
    assert len(metadata) == 1
