"""Ensure compile/outside transfers do not leak into warm per-call counts."""

import importlib.util
from pathlib import Path
import sqlite3

import pytest

spec = importlib.util.spec_from_file_location('rad_nsys', Path(__file__).with_name('parse_nsys.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_trace(path):
    db = sqlite3.connect(path)
    db.executescript('''
        CREATE TABLE StringIds (id INTEGER, value TEXT);
        CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, textId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER, copyKind INTEGER, bytes INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET (start INTEGER, end INTEGER);
        INSERT INTO StringIds VALUES (1, 'BR_full_d02_lw_jump_native');
        INSERT INTO NVTX_EVENTS VALUES (100, 1000, NULL, 1);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1, 20), (110, 150), (210, 250), (310, 350), (1001, 1100);
        INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (10, 20, 1, 8000), (160, 180, 8, 1024), (260, 280, 8, 1024), (360, 380, 8, 1024), (1010, 1020, 2, 8000);
        INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES (180, 190), (280, 290), (380, 390);
    ''')
    db.commit()
    db.close()


def test_per_call_counts_exclude_compile_and_fetch(tmp_path):
    path = tmp_path / 'trace.sqlite'
    make_trace(path)
    row = module.parse(path, 3)['ranges'][0]
    assert row['kernels'] == 1
    assert row['ops'] == 3
    assert row['device_ms'] == pytest.approx(0.00007)
    assert row['transfers'] == {'8': {'count': 1, 'bytes': 1024}}


def test_absent_ranges_fail_closed(tmp_path):
    path = tmp_path / 'trace.sqlite'
    make_trace(path)
    with sqlite3.connect(path) as db:
        db.execute('DELETE FROM NVTX_EVENTS')
    with pytest.raises(ValueError, match='No warm'):
        module.parse(path, 3)
