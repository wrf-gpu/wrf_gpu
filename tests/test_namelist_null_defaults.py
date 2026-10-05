"""Legal Fortran nulls retain each forecast accessor's supplied default.

These exercise configuration consumers, not forecast fidelity. The parser's
independent compiled Fortran oracle lives in test_namelist_list_directed.py.
"""
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwrf import cli
from gpuwrf.integration import daily_pipeline as daily
from gpuwrf.integration import d02_replay as replay
from gpuwrf.integration import nested_pipeline as nested
from gpuwrf.io.gen2_accessor import Gen2Run, parse_namelist
from gpuwrf.io.lower_boundary import _domain_value, load_lower_boundary
from gpuwrf.io.validation import lead_time_slice
from gpuwrf.io.wrf_history_metadata import wrf_history_global_attributes


CASE = Path('<USER_HOME>/wrf_gpu2_lanes/wn3/cases/20260227_18z_a1')


def case(tmp_path, body):
    path = tmp_path / 'namelist.input'
    path.write_text(body)
    return SimpleNamespace(namelist=parse_namelist(path))


@pytest.mark.parametrize('reader', ('nested', 'daily', 'gen2', 'lower_boundary'))
@pytest.mark.parametrize('literal, explicit, default', (
    ('.false.', False, True),
    ('0', 0, 6),
    ('0.d0', 0.0, 30.0),
    ("'chosen'", 'chosen', 'registry'),
))
def test_null_slots_use_typed_default_without_changing_explicit_value(
    tmp_path, reader, literal, explicit, default,
):
    run = case(tmp_path, f'&physics control=1*,{literal},1*, /\n')
    group = run.namelist['physics']

    def read(index):
        if reader == 'nested':
            return nested._domain_list_value(group['control'], index, default)
        if reader == 'daily':
            return daily._domain_namelist_value(run, 'physics', 'control', f'd{index+1:02d}', default)
        if reader == 'gen2':
            return Gen2Run._nml_list_value(group, 'control', index, default)
        return _domain_value(group, 'control', index, default)

    assert [read(i) for i in range(4)] == [default, explicit, default, default]
    # A scalar null has the same meaning as an omitted control.
    group['control'] = None
    assert read(0) == default


@pytest.mark.parametrize('reader', ('nested', 'daily', 'gen2', 'lower_boundary'))
def test_existing_scalar_and_short_list_policy_is_preserved(tmp_path, reader):
    run = case(tmp_path, '&physics control=6,8, /\n')
    group = run.namelist['physics']

    def read(index):
        if reader == 'nested':
            return nested._domain_list_value(group['control'], index, 99)
        if reader == 'daily':
            return daily._domain_namelist_value(run, 'physics', 'control', f'd{index+1:02d}', 99)
        if reader == 'gen2':
            return Gen2Run._nml_list_value(group, 'control', index, 99)
        return _domain_value(group, 'control', index, 99)

    assert [read(i) for i in range(3)] == [6, 8, 99 if reader == 'gen2' else 8]
    group['control'] = 0
    assert read(2) == 0


@pytest.mark.parametrize('reader, group, key, default, expected', (
    (nested._domain_int, 'dynamics', 'time_step_sound', 6, 0),
    (nested._domain_float, 'physics', 'radt', 30.0, 0.0),
))
def test_numeric_conversions_happen_after_null_resolution(tmp_path, reader, group, key, default, expected):
    run = case(tmp_path, f'&{group} {key}=1*,0, /\n')
    assert reader(run, group, key, 'd01', default) == default
    assert reader(run, group, key, 'd02', default) == expected
    assert reader(run, group, key, 'd03', default) == expected
    run.namelist[group][key] = None
    assert reader(run, group, key, 'd01', default) == default
    run.namelist[group][key] = 'invalid'
    with pytest.raises(ValueError):
        reader(run, group, key, 'd01', default)


def test_history_interval_and_validation_use_hourly_null_default(tmp_path):
    run = case(tmp_path, '&time_control history_interval=,30, /\n')
    assert nested._history_interval_minutes_by_domain(run, ('d01', 'd02')) == {'d01': 60., 'd02': 30.}
    assert lead_time_slice(run, 3) == 3


def test_required_root_time_step_null_keeps_missing_control_error(tmp_path):
    run = case(tmp_path, '&domains time_step=1*,18, /\n')
    with pytest.raises(ValueError, match='namelist has no .*time_step'):
        nested._dt_by_domain(run, ('d01',))
    run.namelist['time_control'] = {'time_step': 18}
    assert nested._dt_by_domain(run, ('d01',)) == {'d01': 18.}


def test_nest_integer_null_uses_default_and_preserves_group_precedence(tmp_path):
    run = case(tmp_path, '&domains spec_bdy_width=,3, /\n')
    assert replay._namelist_int_any(run, 'spec_bdy_width', 5, domain='d01') == 5
    assert replay._namelist_int_any(run, 'spec_bdy_width', 5, domain='d02') == 3
    run.namelist['bdy_control'] = {'spec_bdy_width': 7}
    assert replay._namelist_int_any(run, 'spec_bdy_width', 5, domain='d01') == 7


def test_nest_start_null_has_same_open_time_as_omitted_control(tmp_path):
    run = case(tmp_path, '&time_control start_hour=,0, /\n')
    assert replay.wrf_nest_opens_with_parent_start(run, 'd01', 'd02')
    run.namelist['time_control']['start_hour'][1] = 6
    assert not replay.wrf_nest_opens_with_parent_start(run, 'd01', 'd02')


def test_logical_and_theta_m_null_defaults(tmp_path):
    run = case(tmp_path, '&bdy_control have_bcs_moist=1*, /\n&dynamics use_theta_m=1*, /\n')
    assert nested._root_bdy_flag(run.namelist, 'have_bcs_moist') is False
    assert replay._wrf_use_theta_m(run, 'd01') == 1


def test_null_lower_boundary_control_does_not_open_aux4(tmp_path):
    run = case(tmp_path, '&physics sst_update=1*, /\n')
    assert load_lower_boundary(tmp_path / 'absent', run.namelist, 'd01',
                               run_start=datetime(2026, 2, 27), dt_s=54., shape=(2, 3)) is None


def test_cli_null_duration_and_scheme_use_omitted_defaults(tmp_path):
    run = case(tmp_path, '&time_control run_days=1*, run_hours=2, /\n&physics mp_physics=,8, /\n')
    assert cli._namelist_forecast_hours(tmp_path / 'namelist.input') == 2
    assert cli._namelist_scheme_codes(run.namelist['physics'], 'mp_physics') == {8}


def test_gen2_real_input_top_pressure_scalar_null_default(tmp_path):
    parsed = case(tmp_path, '&domains p_top_requested=1*, /\n').namelist
    run = Gen2Run(CASE)
    run._namelist = parsed
    grid = run.grid('d01')
    assert grid.top_pressure_pa == 5000.0


def test_history_metadata_null_integer_real_and_scalar_defaults(tmp_path):
    run = case(tmp_path, '&physics aer_aod550_opt=,2, radt=1*,10, bucket_mm=1*, /\n')
    attrs = wrf_history_global_attributes(CASE / 'wrfinput_d01', run.namelist, 'd01', 54.)
    np.testing.assert_array_equal(attrs['AER_AOD550_OPT'], np.int32(1))
    np.testing.assert_array_equal(attrs['RADT'], np.float32(0.))
    np.testing.assert_array_equal(attrs['BUCKET_MM'], np.float32(-1.))
    attrs = wrf_history_global_attributes(CASE / 'wrfinput_d02', run.namelist, 'd02', 18.)
    assert attrs['AER_AOD550_OPT'] == 2 and attrs['RADT'] == 10.
