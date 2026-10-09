"""Pause owned CPU scorers during QUIET or a wrf.exe CPU overlap; preserve prior STOPs."""
import datetime
import os
from pathlib import Path
import signal
import time

G = Path('<USER_HOME>/wrf_gpu2_lanes/wn3/V33/gate')
paused = {}  # PID -> starttime; never resume another actor's STOP or a reused PID
launches = {}  # validated owned launch: PID -> (starttime, exact executable/script argv)
end = datetime.datetime(2026, 10, 8, tzinfo=datetime.timezone.utc).timestamp()
PROGRAMS = {str(G / name).encode() for name in (
    'wn3_score_nolimit.py', 'd6_verdict.py', 'strict24.py', 'r4_eval.py',
    'tools_src/scripts/wn3_fast_compare.py', 'tools_src/scripts/compare_wrfout_grid.py',
    'integrity_all/scripts/wn3_integrity.py', 'integrity_all/annex_postpass.py',
)} | {str(G.parent / 'mon33' / name).encode() for name in ('d6_domains.py', 'r4_eval_domains.py')}
DRIVERS = {str(G / name).encode() for name in (
    'd6_arm.sh', 'manual_arm.sh', 'arm_watch4.sh', 'score_arm_0227.sh',
    'integrity_all/integ_arm.sh', 'integrity_all/classify_all.sh',
)} | {str(G.parent / 'mon33/mon33_score.sh').encode()}


def owned_launch(args, parent_pid):
    # A data argument under wn3 is never ownership. Match the executed script,
    # or a verified scorer-driver ancestor for stdin/externally located helpers.
    if len(args) > 1 and args[1] in PROGRAMS:
        return True
    seen = set()
    while parent_pid > 1 and parent_pid not in seen:
        seen.add(parent_pid)
        try:
            p = Path('/proc') / str(parent_pid)
            ancestor_args = (p / 'cmdline').read_bytes().split(b'\0')
            stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            if len(ancestor_args) > 1 and ancestor_args[1] in DRIVERS | PROGRAMS:
                return True
            parent_pid = int(stat[1])
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            break
    return False


def processes():
    wrf, owned = set(), {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            comm = (p / 'comm').read_text().strip()
            pid = int(p.name)
            if pid == os.getpid():
                continue
            cores = os.sched_getaffinity(pid)
            if comm == 'wrf.exe':
                wrf.update(cores)
            args = (p / 'cmdline').read_bytes().split(b'\0')
            if comm.startswith('python'):
                stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
                identity = (stat[19], tuple(args[:2]))
                if launches.get(pid) == identity or owned_launch(args, int(stat[1])):
                    launches[pid] = identity
                    owned[pid] = (stat[19], stat[0], cores)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return wrf, owned


def log(message):
    with (G / 'scorer_governor.log').open('a') as f:
        f.write(datetime.datetime.now(datetime.timezone.utc).isoformat() + ' ' + message + '\n')


def main():
    global paused, launches
    log(f'start pid {os.getpid()}')
    while time.time() < end and not (G / 'STOP_scorer_governor').exists():
        wrf, owned = processes()
        quiet = Path('/tmp/wrf_gpu2_quiet').exists()
        for pid, (stamp, state, cores) in owned.items():
            blocked = quiet or bool(cores & wrf)
            try:
                if blocked and state not in ('T', 't') and pid not in paused:
                    os.kill(pid, signal.SIGSTOP)
                    paused[pid] = stamp
                    log(f'STOP {pid} quiet={quiet} wrf_overlap={sorted(cores & wrf)}')
                elif not blocked and paused.get(pid) == stamp:
                    os.kill(pid, signal.SIGCONT)
                    del paused[pid]
                    log(f'CONT {pid}')
            except ProcessLookupError:
                pass
        paused = {pid: stamp for pid, stamp in paused.items() if pid in owned and owned[pid][0] == stamp}
        launches = {pid: identity for pid, identity in launches.items() if pid in owned}
        time.sleep(5)
    for pid, stamp in paused.items():
        _, owned = processes()
        if pid in owned and owned[pid][0] == stamp:
            os.kill(pid, signal.SIGCONT)
    log('end')


if __name__ == '__main__':
    main()
