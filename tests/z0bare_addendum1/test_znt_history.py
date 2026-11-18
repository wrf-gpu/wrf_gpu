"""Opt-in carried ZNT: real Swiss writer identity, publication and step isolation."""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import hashlib
import os
import sys
import subprocess

import numpy as np
import pytest
from gpuwrf.io import wrfout_writer as base, znt_history as znt
from gpuwrf.io.netcdf_lock import Dataset
from gpuwrf.io.gen2_accessor import Gen2Run

REAL = Path(os.environ.get('ZNT_TEST_REAL_CASE', '<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu'))


def real_payload():
    frames=sorted(REAL.glob('wrfout_d01_*'))
    from gpuwrf.config.paths import wrf_root
    if len(frames)<2 or not (REAL/'wrfinput_d01').is_file():
        pytest.skip('optional real Swiss CPU-WRF case unavailable; set test REAL case to a mounted copy')
    if not (wrf_root()/'Registry/Registry.EM_COMMON').is_file():
        pytest.skip('optional pristine WRF Registry unavailable for metadata oracle')
    frame=frames[1]
    run = Gen2Run(REAL);source = run.grid('d01');grid = source.as_grid_spec()
    with Dataset(frame) as ds:
        ds.set_auto_mask(False)
        raw = {n:np.asarray(v[0] if v.dimensions and v.dimensions[0]=='Time' else v[:]).copy()
               for n,v in ds.variables.items() if n not in ('Times','XTIME')}
    from gpuwrf.io.noahmp_land_init import build_noahmp_land_state
    initial,_,_=build_noahmp_land_state(REAL,'d01')
    roughness=np.asarray(initial.znt).copy()
    import shlex
    registry=wrf_root()/'Registry/Registry.EM_COMMON'
    row=next(shlex.split(line) for line in registry.read_text().splitlines()
             if line.split()[:3]==['state','real','ZNT'])
    metadata={'units':row[10],'description':row[9],
              'MemoryOrder':''.join({'i':'X','j':'Y','k':'Z'}[x] for x in row[3]).ljust(3),
              'stagger':'' if row[6]=='-' else row[6], 'FieldType':104}
    state = {**raw, 'theta':raw.get('THM',raw['T'])+np.float32(300),
             'p_total':raw['P']+raw['PB'], 'ph_total':raw['PH']+raw['PHB'],
             'mu_total':raw['MU']+raw['MUB'], 'roughness_m':roughness}
    kw = dict(domain='d01',domain_authority=base.bind_wrfout_domain_authority('d01',source,grid),
              run_start=datetime(2023,1,15),valid_time=datetime(2023,1,15,1),lead_hours=1.,
              diagnostics=raw,full_variable_set=True)
    return state,grid,SimpleNamespace(use_theta_m=1,sf_surface_physics=4,bl_pbl_physics=5),kw,metadata


def inventory(path):
    with Dataset(path) as ds:
        ds.set_auto_mask(False)
        return {n:(v.dtype.str,v.dimensions,np.asarray(v[:]).tobytes(),
                   {k:v.getncattr(k) for k in v.ncattrs()}) for n,v in ds.variables.items()}


def test_real_writer_off_identity_on_adds_only_znt(tmp_path,monkeypatch):
    # The reference writer source itself is exactly v0.4.6, not a copied implementation.
    root=Path(__file__).resolve().parents[2]
    if subprocess.run(['git','-C',str(root),'cat-file','-e','v0.4.6:src/gpuwrf/io/wrfout_writer.py'],capture_output=True).returncode:
        pytest.skip('optional v0.4.6 writer Git object unavailable for release comparison')
    original=subprocess.check_output(['git','-C',str(root),'show','v0.4.6:src/gpuwrf/io/wrfout_writer.py'])
    assert hashlib.sha256(original).digest()==hashlib.sha256(Path(base.__file__).read_bytes()).digest()
    state,grid,nl,kw,metadata=real_payload()
    baseline=base.write_wrfout_netcdf(state,grid,nl,tmp_path/'base.nc',**kw)
    monkeypatch.setenv('WRFOUT_ZNT','0')
    off=znt.write_wrfout_netcdf(state,grid,nl,tmp_path/'off.nc',**kw)
    monkeypatch.setenv('WRFOUT_ZNT','1')
    on=znt.write_wrfout_netcdf(state,grid,nl,tmp_path/'on.nc',**kw)
    a,b,c=map(inventory,(baseline,off,on))
    assert a.keys()==b.keys() and c.keys()==a.keys()|{'ZNT'}
    for name in a:
        assert a[name][:3]==b[name][:3]==c[name][:3],name
        for attr in a[name][3]:
            np.testing.assert_array_equal(a[name][3][attr],b[name][3][attr])
            np.testing.assert_array_equal(a[name][3][attr],c[name][3][attr])
    with Dataset(on) as ds:
        assert np.asarray(ds['ZNT'][0]).tobytes()==state['roughness_m'].astype(np.float32).tobytes()
        for key in ('units','description','MemoryOrder','stagger','FieldType'):
            np.testing.assert_array_equal(ds['ZNT'].getncattr(key),metadata[key])
        assert ds['ZNT'].dimensions==('Time','south_north','west_east')
    # Missing flag defaults OFF, including when no control file exists.
    monkeypatch.delenv('WRFOUT_ZNT')
    assert not znt.enabled()


def test_journal_composition_async_and_atomic_failure(tmp_path,monkeypatch):
    from gpuwrf.io.async_wrfout import AsyncWrfoutWriter
    state,grid,nl,kw,_=real_payload();monkeypatch.setenv('WRFOUT_ZNT','1')
    prepared=znt.prepare_wrfout_payload(state,grid,nl,tmp_path/'async.nc',**kw)
    captured=[]
    def journal(temporary,target):
        assert not target.exists()
        with Dataset(temporary) as ds:assert 'ZNT' in ds.variables
        captured.append('complete')
        base._publish_wrfout_noreplace(temporary,target)
    prepared=znt.attach_publication(prepared,journal)
    with AsyncWrfoutWriter() as writer:
        writer.submit(prepared,expected_domain='d01',expected_domain_authority=kw['domain_authority'])
    assert captured==['complete'] and 'ZNT' in inventory(tmp_path/'async.nc')
    with pytest.raises(FileExistsError):
        base.write_prepared_wrfout(prepared,expected_domain='d01',expected_domain_authority=kw['domain_authority'])
    bad=znt.prepare_wrfout_payload(state,grid,nl,tmp_path/'fail.nc',**kw)
    original=base._write_float_variable
    def fail(ds,spec,*args):
        if spec.name=='ZNT':raise RuntimeError('append failure')
        return original(ds,spec,*args)
    monkeypatch.setattr(base,'_write_float_variable',fail)
    with pytest.raises(RuntimeError):
        base.write_prepared_wrfout(bad,expected_domain='d01',expected_domain_authority=kw['domain_authority'])
    assert not (tmp_path/'fail.nc').exists()


def test_current_noah_land_and_water_carried_roughness():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from test_noahmp_coupler import _build, HAVE_TABLES
    if not HAVE_TABLES:pytest.skip("optional pristine Noah-MP tables unavailable")
    from gpuwrf.physics.noahmp_coupler import noahmp_surface_adapter
    from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics
    state,land,static,rad,clock=_build()
    before=np.asarray(state.roughness_m).copy()
    diag=surface_layer_with_diagnostics(state)
    out,ls,_=noahmp_surface_adapter(state,land,static,radiation=rad,clock=clock,dt=90.)
    mask=np.asarray(state.xland)<1.5
    assert np.asarray(out.roughness_m)[mask].tobytes()==np.asarray(ls.znt)[mask].tobytes()
    np.testing.assert_array_equal(np.asarray(out.roughness_m)[~mask],np.asarray(diag.znt)[~mask])
    assert np.any(np.asarray(out.roughness_m)[mask]!=before[mask])


def test_output_option_keeps_full_step_proto_and_aot_key(monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from test_aot_cheap_key import _build_call
    from gpuwrf.runtime import aot_cheap_key as ck
    from gpuwrf.runtime.operational_mode import _advance_chunk_fori
    import jax
    from gpuwrf.contracts import state as state_contract
    monkeypatch.setattr(state_contract,'_gpu_device',lambda:jax.devices('cpu')[0])
    carry,nl,clock=_build_call()
    start=jax.numpy.asarray(1,jax.numpy.int32);kwargs=dict(n_steps=1,cadence=1)
    keys=[];protos=[]
    for flag in ('0','1'):
        monkeypatch.setenv('WRFOUT_ZNT',flag)
        jax.clear_caches()  # observe a real retrace, not a cached lowering after an env flip
        keys.append(ck.cheap_key(_advance_chunk_fori,(carry,nl,start,clock),kwargs,nl))
        computation=_advance_chunk_fori.lower(carry,nl,start,clock,**kwargs).compiler_ir('hlo')
        protos.append(hashlib.sha256(computation.as_hlo_text(print_large_constants=True).encode()).hexdigest())
    assert keys[0] is not None and keys[0]==keys[1] and protos[0]==protos[1]
    scope=ck._trace_reachable_source_files(ck._gpuwrf_package_root())
    assert Path(znt.__file__).resolve() not in scope
    # The comparison is self-contained: no lane receipt is a test input.
    assert len(scope)>0
    monkeypatch.setenv('WRFOUT_ZNT','0'); before=ck.source_fingerprint_hash()
    monkeypatch.setenv('WRFOUT_ZNT','1'); assert ck.source_fingerprint_hash()==before


def test_cli_flag_is_output_only_and_independent_of_controls(monkeypatch,capsys):
    from gpuwrf import cli
    monkeypatch.setenv('WRFOUT_ZNT','0')
    # Stop at the first pre-I/O argument guard; exercise the real selector wiring.
    common=['run','--input-dir','unused','--output-dir','unused',
            '--max-dom','2','--domains-from-namelist']
    args=cli.build_parser().parse_args(common+['--history-znt'])
    assert args.surface_controls is None and args.history_znt
    assert cli._cmd_run(args)==2 and znt.enabled()
    args=cli.build_parser().parse_args(common)
    assert not args.history_znt
    assert cli._cmd_run(args)==2 and not znt.enabled()
