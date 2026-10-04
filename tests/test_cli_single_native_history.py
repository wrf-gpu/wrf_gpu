"""Public single-domain routing keeps WRF initialization and resident history."""
from pathlib import Path
import json
import pytest
from gpuwrf import cli
from gpuwrf.integration import nested_pipeline as shared


@pytest.mark.parametrize('hours', [1, 24])
def test_swiss_public_cli_uses_single_root_and_emits_initial_history(tmp_path, monkeypatch, capsys, hours):
    from gpuwrf.integration import daily_pipeline as daily
    source = Path(__file__).resolve().parents[1]/'examples/switzerland_d01'
    configs = []
    def execute(config):
        configs.append(config)
        return {'verdict':'PIPELINE_GREEN','wrfout_files':['t0','h1'],'metadata':{}}
    monkeypatch.setattr(shared, 'execute_nested_pipeline', execute)
    monkeypatch.setattr(daily, 'execute_daily_pipeline', lambda config: pytest.fail('native fell back to daily'))
    monkeypatch.setenv('XLA_PYTHON_CLIENT_ALLOCATOR', 'cuda_async')
    assert cli.main(['run','--input-dir',str(source),'--output-dir',str(tmp_path/'out'),
                     '--domain','d01','--hours',str(hours),'--scratch-dir',str(tmp_path/'scratch')]) == 0
    assert len(configs) == 1
    config = configs[0]
    assert config.max_dom == 1 and config.hours == hours
    assert config.emit_initial_history is True
    assert config.input_dir == source and config.output_dir == tmp_path/'out'
    assert json.loads(capsys.readouterr().out)['effective_max_dom'] == 1


def test_cpu_replay_single_domain_keeps_daily_compatibility(tmp_path, monkeypatch, capsys):
    from gpuwrf.integration import daily_pipeline as daily
    source = Path(__file__).resolve().parents[1]/'examples/switzerland_d01'
    (tmp_path/'namelist.input').write_bytes((source/'namelist.input').read_bytes())
    for name in ('wrfinput_d01','wrfbdy_d01'):
        (tmp_path/name).symlink_to(source/name)
    # Detection reads filenames only; replay execution itself is stubbed.
    for hour in (0,1):
        (tmp_path/f'wrfout_d01_2023-01-15_{hour:02d}:00:00').touch()
    configs = []
    def execute(config):
        configs.append(config)
        return {'verdict':'PIPELINE_GREEN','wrfout_files':[]}
    monkeypatch.setattr(daily, 'execute_daily_pipeline', execute)
    monkeypatch.setattr(shared, 'execute_nested_pipeline', lambda config: pytest.fail('replay selected native'))
    assert cli.main(['run','--input-dir',str(tmp_path),'--output-dir',str(tmp_path/'out'),
                     '--domain','d01','--hours','1','--scratch-dir',str(tmp_path/'scratch')]) == 0
    assert len(configs) == 1
    assert json.loads(capsys.readouterr().out)['init_mode'] == 'cpu_wrf_replay'


def test_swiss_dry_run_reports_the_actual_initial_history_default(tmp_path, capsys):
    source = Path(__file__).resolve().parents[1]/'examples/switzerland_d01'
    assert cli.main(['run','--input-dir',str(source),'--output-dir',str(tmp_path/'out'),
                     '--hours','1','--dry-run']) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['effective_max_dom'] == 1 and plan['emit_initial_history'] is True


def test_shared_swiss_header_and_initialized_surface_match_original_cpu_wrf():
    from types import SimpleNamespace
    import numpy as np
    from gpuwrf.io.netcdf_lock import Dataset
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.io.wrf_history_metadata import wrf_history_global_attributes
    source = Path(__file__).resolve().parents[1]/'examples/switzerland_d01'
    truth = Path('<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu/wrfout_d01_2023-01-15_00:00:00')
    if not truth.exists():
        pytest.skip('original Swiss CPU-WRF reference not mounted')
    attrs = wrf_history_global_attributes(source/'wrfinput_d01', Gen2Run(source).namelist, 'd01', 18)
    writer = object.__new__(shared._PerDomainWrfoutWriter)
    writer.input_dir = source
    with Dataset(source/'wrfinput_d01') as initial:
        grid = SimpleNamespace(nx=len(initial.dimensions['west_east']),
                               ny=len(initial.dimensions['south_north']), nz=len(initial.dimensions['bottom_top']))
    writer.bundles = {'d01':SimpleNamespace(grid=grid)}
    fields = writer._initial_surface_fields('d01')
    with Dataset(truth) as cpu:
        assert set(cpu.ncattrs()) <= set(attrs)
        for name in cpu.ncattrs():
            if name not in {'TITLE','NTASKS_X','NTASKS_Y','NTASKS_TOTAL'}:
                np.testing.assert_array_equal(attrs[name], cpu.getncattr(name), err_msg=name)
        for name, value in fields.items():
            if name in cpu.variables:
                expected = np.asarray(cpu[name][0])
                np.testing.assert_array_equal(np.asarray(value,dtype=expected.dtype), expected, err_msg=name)
