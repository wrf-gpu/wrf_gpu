"""Attribution and accounting contracts on a tiny node-traced capture."""
import importlib.util
from pathlib import Path
import sqlite3

import pytest

spec = importlib.util.spec_from_file_location('b_phys_census', Path(__file__).with_name('family_census.py'))
census = importlib.util.module_from_spec(spec)
spec.loader.exec_module(census)


def capture(path):
    conn = sqlite3.connect(path)
    conn.executescript('''
    create table StringIds(id integer, value text);
    create table NVTX_EVENTS(start integer,end integer,globalTid integer,text text,textId integer);
    create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,correlationId integer,globalTid integer);
    create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,correlationId integer,graphNodeId integer);
    create table CUPTI_ACTIVITY_KIND_MEMCPY(start integer,end integer,correlationId integer,copyKind integer);
    create table CUPTI_ACTIVITY_KIND_MEMSET(start integer,end integer,correlationId integer);
    insert into NVTX_EVENTS values(0,100,1,'hlo_op=x#name=jit(step_mynn_pbl_column)/jit(_where)#',null);
    insert into CUPTI_ACTIVITY_KIND_RUNTIME values(50,1,1);
    insert into CUPTI_ACTIVITY_KIND_KERNEL values(120,140,1,7);
    insert into CUPTI_ACTIVITY_KIND_KERNEL values(140,170,99,null);
    insert into CUPTI_ACTIVITY_KIND_MEMCPY values(170,180,1,8);
    insert into CUPTI_ACTIVITY_KIND_MEMSET values(180,185,99);
    ''')
    conn.commit()
    conn.close()


def test_source_scope_and_graph_rows_conserve_device_work(tmp_path):
    path = tmp_path / 'capture.sqlite'
    capture(path)
    result = census.reduce_window(path)
    assert result['totals']['kernels'] == 2
    assert result['totals']['ops'] == 4
    assert result['totals']['device_ns'] == 65
    assert result['families']['MYNN']['graph_kernels'] == 1
    assert result['families']['MYNN']['kernel_ns'] == 20
    assert result['families']['MYNN']['copy_kind_8'] == 1
    assert result['families']['UNKNOWN']['ops'] == 2


def test_bad_duration_cannot_become_speed_evidence(tmp_path):
    path = tmp_path / 'capture.sqlite'
    capture(path)
    conn = sqlite3.connect(path)
    conn.execute('update CUPTI_ACTIVITY_KIND_KERNEL set end=start-1')
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match='inverted'):
        census.reduce_window(path)


def test_exact_hlo_symbol_recovers_unscoped_graph_work(tmp_path):
    path = tmp_path / 'capture.sqlite'
    capture(path)
    conn = sqlite3.connect(path)
    conn.execute('alter table CUPTI_ACTIVITY_KIND_KERNEL add column shortName integer')
    conn.execute("insert into StringIds values(7,'loop_native_fusion_3')")
    conn.execute('update CUPTI_ACTIVITY_KIND_KERNEL set shortName=7 where correlationId=99')
    conn.commit()
    conn.close()
    result = census.reduce_window(path, {'loop_native_fusion_3': 'THOMPSON'})
    assert result['families']['THOMPSON']['kernel_ns'] == 30
    assert result['hlo_mapped_kernels']['THOMPSON'] == 1
    assert result['families']['UNKNOWN']['ops'] == 1  # unmatched memset retained
    assert result['totals']['device_ns'] == 65


def test_reused_symbol_without_call_scope_stays_ambiguous(tmp_path):
    path = tmp_path / 'capture.sqlite'
    capture(path)
    conn = sqlite3.connect(path)
    conn.execute('alter table CUPTI_ACTIVITY_KIND_KERNEL add column shortName integer')
    conn.execute("insert into StringIds values(7,'loop_reused_fusion_3')")
    conn.execute('update CUPTI_ACTIVITY_KIND_KERNEL set shortName=7')
    conn.commit()
    conn.close()
    mapping = {'loop_reused_fusion_3': 'THOMPSON'}
    result = census.reduce_window(path, mapping)
    aliases = census.mark_shared_symbols({'d01_ordinary': result}, {'d01': mapping})
    assert aliases['d01']['loop_reused_fusion_3'] == ['MYNN', 'THOMPSON']
    assert result['families']['MYNN']['kernel_ns'] == 20
    assert result['families']['AMBIGUOUS']['kernel_ns'] == 30
    assert result['totals']['device_ns'] == 65
