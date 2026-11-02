"""Own exactly one setsid(2) group for the whole W4 arm, including RAM wait.

No JAX, CUDA, NVML, or official-lock acquisition here. TERM/INT/HUP handlers
and unconditional finally/EXIT cleanup terminate, escalate and reap only this
arm. A Linux subreaper also catches an unexpectedly escaping descendant.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time


def boot():
    return time.clock_gettime(time.CLOCK_BOOTTIME)


def process_rows():
    rows = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            v = path.read_text().rsplit(')',1)[1].split()
            rows[int(path.parent.name)] = dict(state=v[0], parent=int(v[1]), group=int(v[2]),
                                              session=int(v[3]), born=int(v[19]))
        except (OSError,ValueError,IndexError):
            pass
    return rows


def owned_rows(leader):
    rows = process_rows()
    owned = {leader}
    while True:
        more = {pid for pid,row in rows.items() if row['parent'] == os.getpid()
                or row['parent'] in owned or row['session'] == leader}
        if more <= owned: break
        owned |= more
    return {pid:row for pid,row in rows.items() if pid in owned}


def lock_holders(rows, lock_file):
    try: inode = lock_file.stat()
    except FileNotFoundError: return []
    holders = []
    for pid,row in rows.items():
        for path in Path(f'/proc/{pid}/fd').glob('*'):
            try:
                value = path.stat()
                if (value.st_dev,value.st_ino) == (inode.st_dev,inode.st_ino):
                    holders.append(dict(pid=pid,born=row['born'],fd=int(path.name)))
            except (OSError,ValueError):
                pass
    return holders


def reap(process):
    # Let Popen collect its own direct child's status before reaping adopted
    # grandchildren, so normal failure/exit codes are preserved.
    process.poll()
    if process.returncode is None: return
    while True:
        try:
            if os.waitpid(-1,os.WNOHANG)[0] == 0: break
        except ChildProcessError:
            break


def cleanup(process, lock_file, term_grace, kill_grace):
    seen, groups, observed_holders = set(), set(), []
    for sig,seconds in ((signal.SIGTERM,term_grace),(signal.SIGKILL,kill_grace)):
        deadline = boot()+seconds
        first = True
        while True:
            rows = owned_rows(process.pid)
            seen.update(rows)
            live = {pid:row for pid,row in rows.items() if row['state'] != 'Z'}
            observed_holders.extend(lock_holders(rows,lock_file))
            if first:
                # The sole group created by this controller. Parent/supervisor
                # remains outside it and can always wait/reap after signalling.
                if process.pid == os.getpgrp(): raise RuntimeError('refuse to signal supervisor group')
                try: os.killpg(process.pid,sig)
                except ProcessLookupError: pass
                first = False
            for pid,row in live.items():
                groups.add(row['group'])
                # Individual descendant signals cover a helper that moved to
                # another group/session; never signal a foreign group wholesale.
                try: os.kill(pid,sig)
                except ProcessLookupError: pass
            reap(process)
            remaining = owned_rows(process.pid)
            if not remaining: break
            if boot() >= deadline: break
            time.sleep(.025)
        if not owned_rows(process.pid): break
    try: process.wait(timeout=.2)
    except subprocess.TimeoutExpired: pass
    reap(process)
    remaining = owned_rows(process.pid)
    holders = lock_holders(remaining,lock_file)
    return dict(signalled_pids=sorted(seen), groups_seen=sorted(groups),
                remaining_pids=sorted(remaining), live_remaining=[pid for pid,row in remaining.items() if row['state']!='Z'],
                lock_file=str(lock_file), observed_owned_lock_fds=observed_holders,
                owned_lock_fds_after=holders, all_children_reaped=not remaining)


def supervise(command, *, seconds, receipt, group_file, lock_file, term_grace=2., kill_grace=5.):
    if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0) != 0:
        raise OSError(ctypes.get_errno(),'cannot install child subreaper')
    stopped = [0]
    previous = {}
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        previous[sig] = signal.signal(sig,lambda number,_frame:stopped.__setitem__(0,number))
    start = boot(); process = None; record = dict(supervisor_pid=os.getpid(),setsid_calls=0,cap_s=seconds)
    rc = 125
    def save():
        receipt.parent.mkdir(parents=True,exist_ok=True)
        receipt.write_text(json.dumps(record,indent=2)+'\n')
    try:
        env = dict(os.environ,W4_SUPERVISOR_PID=str(os.getpid()),W4_ARM_GROUP_FILE=str(group_file))
        # The ONLY setsid call in the launcher: no GNU timeouts/new groups in
        # RAM/lock/payload/scoring descendants. One whole-arm clock owns expiry.
        process = subprocess.Popen(command,env=env,start_new_session=True)
        record.update(arm_pid=process.pid,pgid=process.pid,sid=process.pid,setsid_calls=1)
        group_file.parent.mkdir(parents=True,exist_ok=True)
        group_file.write_text(json.dumps(record)+'\n')
        save()
        while process.poll() is None and not stopped[0] and boot()-start < seconds:
            time.sleep(.025)
        original = process.poll()
        timed_out = original is None and not stopped[0]
        rc = 128+stopped[0] if stopped[0] else 124 if timed_out else 128-original if original < 0 else original
        record.update(original_rc=original,timed_out=timed_out,signal=stopped[0])
    finally:
        if process is not None:
            record['cleanup'] = cleanup(process,lock_file,term_grace,kill_grace)
            if not record['cleanup']['all_children_reaped'] or record['cleanup']['owned_lock_fds_after']:
                rc = 125
            elif rc == 0 and record['cleanup']['signalled_pids']:
                rc = 125  # a successful parent may not leave background work
        record.update(rc=rc,elapsed_s=boot()-start,supervisor_peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save()
        for sig,handler in previous.items(): signal.signal(sig,handler)
    return rc


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds',type=float,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--group-file',type=Path,required=True)
    p.add_argument('--lock-file',type=Path,default=Path('/tmp/wrf_gpu2_gpu.lock'))
    p.add_argument('--term-grace',type=float,default=2.)
    p.add_argument('--kill-grace',type=float,default=5.)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args(); command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command or a.seconds<=0: p.error('positive whole-arm budget and command required')
    raise SystemExit(supervise(command,seconds=a.seconds,receipt=a.receipt,group_file=a.group_file,
                              lock_file=a.lock_file,term_grace=a.term_grace,kill_grace=a.kill_grace))
