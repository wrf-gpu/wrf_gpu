"""W1c-1L launcher host logic: per-member disk gate (decision (b)) and per-member raw release tool. CPU, no JAX."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[3] / 'scripts' / 'w1c' / 'launcher'


def _load(name):
    spec = importlib.util.spec_from_file_location(name, LAUNCHER / f'{name}.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


disk = _load('w1c_disk')


# ------------------------------------------------------------------ disk gate
def test_floor_constant_is_rule():
    assert disk.MEMBER_FLOOR_GIB == 437 and disk.MEMBER_FLOOR_GIB >= 430 + 5.16 + 1.0


def test_wait_returns_when_space_appears():
    free = iter([435.0, 436.5, 437.2])
    sleeps = []
    waited = disk.wait_for_floor(lambda: next(free), 437, lambda: 10_000, sleep=sleeps.append, log=lambda *_: None)
    assert waited == 2 * disk.WAIT_POLL_S and len(sleeps) == 2


def test_wait_refuses_when_member_no_longer_fits():
    clock = {'left': disk.MEMBER_CAP + disk.MEMBER_SETUP_S + disk.WAIT_POLL_S + 30}

    def sleep(s):
        clock['left'] -= s
        clock['n'] = clock.get('n', 0) + 1
        assert clock['n'] < 100, 'wait loop not bounded by the remaining time'
    with pytest.raises(disk.DiskFloorTimeout):
        disk.wait_for_floor(lambda: 400.0, 437, lambda: clock['left'], sleep=sleep, log=lambda *_: None)
    assert clock['left'] < disk.MEMBER_CAP + disk.MEMBER_SETUP_S + disk.WAIT_POLL_S


def _rows():
    return [{'ordinal': i, 'row_id': f'r{i}', 'registry_run_id': f'optimize:{i}', 'case_id': 'c', 'arm_id': 'C'} for i in (1, 2, 3, 4)]


def test_run_rows_disk_halt_records_not_attempted():
    calls = []

    def before(m):
        if m['ordinal'] == 3:
            raise disk.DiskFloorTimeout('no space')
        return 5.0
    res = disk.run_rows(_rows(), lambda m: calls.append(m['ordinal']) or {'status': 'PASS', **m}, before)
    assert calls == [1, 2]
    assert [r['status'] for r in res] == ['PASS', 'PASS', 'NOT_ATTEMPTED', 'NOT_ATTEMPTED']
    assert res[2]['reason'].startswith('HALT_DISK_FLOOR ordinal=3') and res[3]['attempted'] is False
    assert res[0]['disk_wait_s'] == 5.0 and res[2]['registry_started'] is False


def test_run_rows_stop_on_first_fail_skips_disk_gate():
    gated = []
    res = disk.run_rows(_rows(), lambda m: {'status': 'FAIL' if m['ordinal'] == 2 else 'PASS', **m},
                        lambda m: gated.append(m['ordinal']) or 0.0)
    assert gated == [1, 2] and [r['status'] for r in res] == ['PASS', 'FAIL', 'NOT_ATTEMPTED', 'NOT_ATTEMPTED']
    assert res[2]['reason'].startswith('STOP_ON_FIRST_FAIL')


def test_run_rows_exception_is_fail():
    def boom(m):
        raise RuntimeError('x')
    res = disk.run_rows(_rows(), boom, lambda m: 0.0)
    assert res[0]['status'] == 'FAIL' and res[1]['status'] == 'NOT_ATTEMPTED'


# ------------------------------------------------------------------ critic v1 launcher findings F1-F3
def test_F1_inputs_arm_labels_match_w1c_arms():
    import ast, re
    ms = (LAUNCHER / 'make_seal.py').read_text()
    inputs = ast.literal_eval(next(n.value for n in ast.parse(ms).body if isinstance(n, ast.Assign) and n.targets[0].id == 'INPUTS'))
    arms = set(re.findall(r'\d{8}_18z:(\w+)', inputs))
    w1c_arms = next(ast.literal_eval(n.value) for n in ast.parse((LAUNCHER / 'w1c.py').read_text()).body
                    if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'ARMS')
    assert arms == set(w1c_arms) == {'C', 'SIC_1L'}


def test_F3_pre_member_refusal_recorded_not_escaped():
    def before(m):
        if m['ordinal'] == 2:
            raise ValueError('W1c window/deadline exhausted after disk wait')
        return 0.0
    res = disk.run_rows(_rows(), lambda m: {'status': 'PASS', **m}, before)
    assert len(res) == 4 and res[0]['status'] == 'PASS'
    assert [r['status'] for r in res[1:]] == ['NOT_ATTEMPTED'] * 3
    assert res[1]['reason'].startswith('HALT_PRE_MEMBER ordinal=2: ValueError') and res[1]['registry_started'] is False
    assert res[1].get('global_abort') is False


def test_F3_signal_during_gate_recorded_as_global_abort():
    def before(m):
        raise KeyboardInterrupt()
    res = disk.run_rows(_rows(), lambda m: {'status': 'PASS', **m}, before)
    assert [r['status'] for r in res] == ['NOT_ATTEMPTED'] * 4 and res[0]['global_abort'] is True


def test_layout_pins_output_root_everywhere():
    """LAYOUT.json (SHA-bound pointer /output_root for ALISIOS) == w1c.OUTPUT_ROOT; raw dirs follow the hard rule."""
    import ast
    lay = json.loads((LAUNCHER / 'LAYOUT.json').read_text())
    src = (LAUNCHER / 'w1c.py').read_text()
    assert "OUTPUT_ROOT = DATA / 'W1C_1L_20261031'" in src and "DATA = Path('<DATA_ROOT>/wrf_gpu2/scratch/v46-icvariant')" in src
    assert lay['output_root'] == '<DATA_ROOT>/wrf_gpu2/scratch/v46-icvariant/W1C_1L_20261031'
    assert lay['journal'] == lay['output_root'] + '/RAW_DELETE_JOURNAL_W1C_1L.jsonl'
    assert all(m['raw_wrfout_dir'] == f"{lay['output_root']}/{m['our_id']}/wrfout" for m in lay['members']) and len(lay['members']) == 4
    assert "'admission output_root != LAYOUT.json'" in (LAUNCHER / 'make_seal.py').read_text()
