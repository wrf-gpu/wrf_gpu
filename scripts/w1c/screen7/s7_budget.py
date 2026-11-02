"""SCREEN7 elapsed accounting on the shared ALP1 wave ledger (fork of the sealed alp1_budget.py, df5da186).

Same ledger file, lock, record kinds and charge rule as the ALP1 wave (C3 prime + J3 already recorded): START is
appended under the ledger lock BEFORE the official GPU lock wrapper, END after the wrapper exits, charged with the
whole entry elapsed (lock wait, every member, disk waits, post checks, release). The wave total (3600 s) includes
SCREEN7; an unresolved/duplicate/unknown record blocks. New here: phase SCREEN7_0530 (one entry for the three
members), and `budget()` accepts the ALP1 phase names plus this one. Members are admitted inside the entry by
s7.py via active() (each member needs its full cap incl. post within the remaining wave budget).
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alp1_contract as g
q = g.q
LEDGER = q.ROOT / 'ALP1_WAVE_GPU_BUDGET_20261031.jsonl'
PHASE = 'SCREEN7_0530'
ENTRY_CAP_S, RELEASE_S = 1500, 120
KNOWN_PHASES = set(g.CAPS) | {PHASE}


def records():
    return [json.loads(x) for x in LEDGER.read_text().splitlines()] if LEDGER.exists() else []


def append(value):
    with LEDGER.open('a') as f:
        json.dump(value, f); f.write('\n'); f.flush(); os.fsync(f.fileno())


def budget(rows):
    """alp1_contract.budget rules over every record; returns used seconds of the wave."""
    starts = {}; ends = {}
    for row in rows:
        q.require(row['kind'] in ('START', 'END'), 'wave ledger: unknown budget record')
        target = starts if row['kind'] == 'START' else ends
        q.require(row['id'] not in target, 'wave ledger: duplicate budget event'); target[row['id']] = row
    q.require(set(starts) == set(ends), 'wave ledger: unresolved GPU entry blocks SCREEN7')
    used = 0.
    for ident, end in ends.items():
        seconds = end['seconds']; start = starts[ident]
        q.require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0 and
                  end['released'] is True and end['phase'] == start['phase'] and start['phase'] in KNOWN_PHASES,
                  'wave ledger: unknown/invalid GPU budget charge')
        used += seconds
    return used


def entry_cap(used, calendar_s):
    """Whole-entry hard cap: never past the wave remainder minus release, never past the window."""
    cap = min(ENTRY_CAP_S, g.TOTAL_GPU_SECONDS - used - RELEASE_S, calendar_s - RELEASE_S)
    q.require(cap > 0, 'SCREEN7 wave remainder/window cannot fit an entry')
    return cap


def active(ident, seal):
    """Inside the entry: this START is the only open one; returns prior wave use and entry elapsed."""
    rows = records(); starts = [r for r in rows if r['kind'] == 'START']; ends = [r for r in rows if r['kind'] == 'END']
    opened = {r['id'] for r in starts} - {r['id'] for r in ends}
    q.require(opened == {ident}, 'SCREEN7 active budget reservation missing/foreign')
    q.require(sum(r['id'] == ident for r in starts) == 1, 'SCREEN7 duplicate active reservation')
    start = next(r for r in starts if r['id'] == ident)
    q.require(start['phase'] == PHASE and start['seal_sha256'] == seal and
              start['admission_sha256'] == os.environ['W1AQ_ADMISSION_SHA'], 'SCREEN7 active budget context changed')
    used = budget([r for r in rows if r['id'] != ident])
    elapsed = time.clock_gettime(time.CLOCK_BOOTTIME) - start['boottime_s']
    q.require(0 <= elapsed < start['cap_s'] and used + elapsed < g.TOTAL_GPU_SECONDS - RELEASE_S,
              'SCREEN7 whole-wave or entry GPU budget exhausted')
    return {'prior_used_s': used, 'entry_elapsed_s': elapsed, 'entry_cap_s': start['cap_s']}


def main():
    p = argparse.ArgumentParser(allow_abbrev=False); p.add_argument('--seal-sha', required=True)
    p.add_argument('--admission', type=Path, required=True)
    p.add_argument('command', nargs=argparse.REMAINDER); x = p.parse_args()
    here = Path(__file__).resolve().parent; q.sealed(here, x.seal_sha)
    import s7
    a = s7.validate(x.admission, os.environ['W1AQ_ADMISSION_SHA'], gpu=True)[0]
    command = x.command[1:] if x.command[:1] == ['--'] else x.command
    q.require(command and command[0] == '<USER_HOME>/src/wrf_gpu2/scripts/with_gpu_lock.sh',
              'SCREEN7 budget must surround the official GPU lock')
    end = min(q.epoch(a['window']['hard_end']), q.epoch(os.environ['GA_END_UTC']))
    lock = LEDGER.with_suffix('.lock'); fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        expected = os.environ['W1A_ALP1_BUDGET_SHA']
        q.require((q.sha(LEDGER) if LEDGER.exists() else 'NONE') == expected and expected == a['budget_ledger']['sha256'],
                  'SCREEN7 prior ledger pin mismatch')
        used = budget(records()); cap = entry_cap(used, end - time.time())
        ident = uuid.uuid4().hex; start = time.clock_gettime(time.CLOCK_BOOTTIME)
        append({'kind': 'START', 'id': ident, 'phase': PHASE, 'boottime_s': start, 'prior_used_s': used,
                'cap_s': cap, 'release_s': RELEASE_S, 'seal_sha256': x.seal_sha,
                'admission_sha256': os.environ['W1AQ_ADMISSION_SHA'], 'prior_ledger_sha256': expected})
    finally:
        os.close(fd)
    names = ('HOME', 'PATH', 'LANG', 'LC_ALL', 'JAX_PLATFORMS', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'TMPDIR',
             'GA_START_UTC', 'GA_END_UTC', 'GA_GATE_ID', 'GA_RW_GATE_ID', 'W1AQ_ADMISSION_SHA', 'W1A_ALP1_BUDGET_SHA',
             'ALISIOS_OPTIMIZE_WALL_DEADLINE')
    env = {k: os.environ[k] for k in names if k in os.environ}; env['W1A_ALP1_BUDGET_ENTRY_ID'] = ident
    proc = None; stopped = False; rc = 1; released = False

    def stop(sig, frame):
        nonlocal stopped
        if not stopped and proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGTERM); stopped = True
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
    try:
        proc = subprocess.Popen(command, env=env)
        while proc.poll() is None:
            elapsed = time.clock_gettime(time.CLOCK_BOOTTIME) - start
            if elapsed >= cap or used + elapsed >= g.TOTAL_GPU_SECONDS - RELEASE_S:
                stop(signal.SIGTERM, None)
            time.sleep(.2)
        rc = proc.wait(); released = rc == 0   # as alp1_budget: a non-zero entry stays UNSETTLED (manual settlement)
    finally:
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGTERM); rc = proc.wait(); released = rc == 0
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        seconds = time.clock_gettime(time.CLOCK_BOOTTIME) - start
        terminal = {'kind': 'END' if released else 'UNSETTLED', 'id': ident, 'phase': PHASE, 'seconds': seconds,
                    'rc': rc, 'released': released, 'charge': 'CONSERVATIVE_ENTRY_ELAPSED_INCLUDING_WAIT_POST_RELEASE'}
        if released:
            fd = os.open(lock, os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX); append(terminal)
            finally:
                os.close(fd)
        q.save_terminal(s7.LANE / f'SCREEN7_GPU_ENTRY_{ident}.json', terminal)
    return rc if rc >= 0 else 128 - rc


if __name__ == '__main__':
    raise SystemExit(main())
