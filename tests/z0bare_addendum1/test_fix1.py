"""ZNT output policy and subset contracts; structural resume plus real IO.

Sentinel stepping below is transport evidence. Real nested physics/output is
checked separately by the bounded real-case harness, not inferred from it.
"""
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from gpuwrf.io import znt_history as znt,wrfout_writer as base
from gpuwrf.io.netcdf_lock import Dataset
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_m7_netcdf_writer import synthetic_case,writer_authority


def payload(tmp_path):
    state,grid,nl=synthetic_case()
    state.roughness_m=(np.arange(grid.ny*grid.nx).reshape(grid.ny,grid.nx)/1000+.002).astype(np.float32)
    kw=dict(domain='d02',domain_authority=writer_authority(grid),run_start=datetime(2026,1,1),
            valid_time=datetime(2026,1,1,1),lead_hours=1.)
    return state,grid,nl,kw


@pytest.mark.parametrize('subset',[None,('T2',),('T2','ZNT'),('ZNT',)])
def test_subset_contract(tmp_path,monkeypatch,subset):
    monkeypatch.setenv('WRFOUT_ZNT','1')
    state,grid,nl,kw=payload(tmp_path)
    path=znt.write_wrfout_netcdf(state,grid,nl,tmp_path/'history.nc',variable_subset=subset,**kw)
    with Dataset(path) as ds:
        wanted=subset is None or 'ZNT' in subset
        assert ('ZNT' in ds.variables)==wanted
        if wanted:np.testing.assert_array_equal(ds['ZNT'][0],state.roughness_m)
        if subset is not None:assert set(ds.variables)<={'Times','XTIME',*subset}


def test_shared_async_full_payload_respects_each_write_subset(tmp_path,monkeypatch):
    from gpuwrf.io.async_wrfout import AsyncWrfoutWriter
    monkeypatch.setenv('WRFOUT_ZNT','1')
    state,grid,nl,kw=payload(tmp_path)
    p=znt.prepare_wrfout_payload(state,grid,nl,tmp_path/'full.nc',**kw)
    callback=p.restart_publication
    with AsyncWrfoutWriter() as writer:
        for filename,subset in (('exclude.nc',('T2',)),('include.nc',('T2','ZNT'))):
            writer.submit_subset(p,expected_domain='d02',expected_domain_authority=kw['domain_authority'],
                                 variable_subset=subset,target=tmp_path/filename)
        writer.submit(p,expected_domain='d02',expected_domain_authority=kw['domain_authority'])
    assert p.restart_publication is callback
    for filename,wanted in (('exclude.nc',False),('include.nc',True),('full.nc',True)):
        with Dataset(tmp_path/filename) as ds:
            assert ('ZNT' in ds.variables)==wanted
            if wanted:np.testing.assert_array_equal(ds['ZNT'][0],state.roughness_m)


def test_batch_refused_through_cli_and_real_driver_before_load(tmp_path,monkeypatch,capsys):
    from gpuwrf import cli
    from gpuwrf.integration import nested_pipeline as pipeline
    monkeypatch.setenv('GPUWRF_BATCH_ENSEMBLE','2')
    monkeypatch.setenv('WRFOUT_ZNT','1')
    monkeypatch.setattr(pipeline,'_load_batched_domains',lambda *a,**k:pytest.fail('model initialization reached'))
    cfg=pipeline.NestedPipelineConfig(tmp_path/'absent',tmp_path/'out',tmp_path/'proof',1.,2,aot_prefetch=False)
    with pytest.raises(ValueError,match='history-znt.*B=1'):
        pipeline.execute_nested_pipeline(cfg)
    args=cli.build_parser().parse_args(['run','--input-dir','absent','--output-dir','absent','--history-znt'])
    assert cli._cmd_run(args)==2
    assert 'history-znt' in capsys.readouterr().err
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('initial_on',[False,True])
def test_actual_checkpoint_resume_refuses_output_flip_and_preserves_history(tmp_path,monkeypatch,initial_on):
    from tests.v025.restart import test_single_domain_restart as fixture
    from gpuwrf.runtime import restart_store as storage
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.contracts import state as state_contract
    monkeypatch.setattr(state_contract,'_gpu_device',lambda:jax.devices('cpu')[0])
    seed=fixture.seed.__wrapped__()
    actual_identity=storage.nested_identity
    finals=[]
    pipeline,storage,writers=fixture._install_driver(monkeypatch,seed,finals)
    # Keep the real identity/output-policy implementation, not the fixture's digest stub.
    monkeypatch.setattr(storage,'nested_identity',actual_identity)
    monkeypatch.setattr(pipeline,'Gen2Run',Gen2Run,raising=False)
    minimal_prepare=pipeline.prepare_wrfout_payload
    monkeypatch.setattr(base,'prepare_wrfout_payload',minimal_prepare)
    monkeypatch.setattr(pipeline,'prepare_wrfout_payload',znt.prepare_wrfout_payload)
    folder=tmp_path/'in';folder.mkdir()
    (folder/'namelist.input').write_text('&time_control\n history_interval=60,\n/\n')
    monkeypatch.setenv('WRFOUT_ZNT','1' if initial_on else '0')
    monkeypatch.delenv('GPUWRF_BATCH_ENSEMBLE',raising=False)
    cfg=pipeline.NestedPipelineConfig(folder,tmp_path/'out',tmp_path/'proof',3.,1,
        emit_initial_history=True,checkpoint_dir=tmp_path/'store',checkpoint_interval_steps=100,
        checkpoint_max_bytes=10**8,checkpoint_reserve_bytes=0,aot_prefetch=False)
    result=pipeline.execute_nested_pipeline(cfg)
    assert result['verdict']=='PIPELINE_GREEN'
    store=storage.RestartStore(cfg.checkpoint_dir,10**8,2,0);generation=store.latest()
    snapshot,manifest=store.read(generation,device=False)
    assert znt.recorded_policy(manifest['identity'])==initial_on
    assert ('history_znt' in manifest['identity']['output_policy']['history'])==initial_on
    before={p.name:p.read_bytes() for p in cfg.output_dir.glob('wrfout_*')}
    assert len(before)==4
    for name in before:
        with Dataset(cfg.output_dir/name) as ds:assert ('ZNT' in ds.variables)==initial_on
    # Exercise the real checkpoint reader, journal reopening and completed-run resume.
    resumed=pipeline.execute_nested_pipeline(replace(cfg,resume_checkpoint=generation))
    assert resumed['verdict']=='PIPELINE_GREEN'
    assert before=={p.name:p.read_bytes() for p in cfg.output_dir.glob('wrfout_*')}
    monkeypatch.setenv('WRFOUT_ZNT','0' if initial_on else '1')
    monkeypatch.setattr(pipeline,'_load_domains',lambda *a,**k:pytest.fail('model initialization reached'))
    with pytest.raises(ValueError,match='history-znt output policy differs'):
        pipeline.execute_nested_pipeline(replace(cfg,resume_checkpoint=generation))
    assert before=={p.name:p.read_bytes() for p in cfg.output_dir.glob('wrfout_*')}
    assert bool(znt.recorded_policy({})) is False  # pre-feature manifests mean OFF


@pytest.mark.parametrize('fsno',[0.,.4,1.])
def test_snow_blend_through_carried_writer_source(tmp_path,monkeypatch,fsno):
    import jax.numpy as jnp
    import test_noahmp_energy_canopy as inputs
    from gpuwrf.physics.noahmp.energy import noahmp_energy_canopy
    from gpuwrf.physics.noahmp.energy_radiation import radiation_twostream
    ls=inputs._land_state();forcing=inputs._midday_forcing();static=inputs._static()
    phen=inputs._phenology()._replace(fveg=inputs._f(0),elai=inputs._f(0),esai=inputs._f(0))
    ep=inputs._energy_params()._replace(z0bare=inputs._f(.02))
    rad,extras=radiation_twostream(ls,forcing,static,phen,inputs._rad_params(),60.)
    rad=rad._replace(fsno=inputs._f(fsno))
    current,flux,_=noahmp_energy_canopy(ls,forcing,static,rad,60.,phen=phen,params=ep,rad_extras=extras)
    state,grid,nl,kw=payload(tmp_path)
    state.roughness_m=np.broadcast_to(np.asarray(current.znt),(grid.ny,grid.nx)).copy()
    expected=np.float32(.02*(1-fsno)+float(ep.z0sno[0,0])*fsno)
    monkeypatch.setenv('WRFOUT_ZNT','1')
    path=znt.write_wrfout_netcdf(state,grid,nl,tmp_path/'snow.nc',variable_subset=('ZNT',),**kw)
    with Dataset(path) as ds:np.testing.assert_allclose(ds['ZNT'][0],expected,rtol=2e-7,atol=1e-9)
