#!/usr/bin/env python3
"""Run the staged shipped example with four CPU-WRF MPI ranks and retain clocks.

The run directory must already contain immutable inputs, wrf.exe and WRF tables.
No WRF installation or case outside that directory is written.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def session_processes(session_id):
    """PRTE gives each MPI rank its own process group inside the new session."""
    members = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
            if os.getsid(pid) == session_id:
                members[pid] = (os.getpgid(pid), (entry / "comm").read_text().strip())
        except (ProcessLookupError, FileNotFoundError):
            continue
    return members


def signal_session(session_id, signum):
    for group in {group for group, _ in session_processes(session_id).values()}:
        try:
            os.killpg(group, signum)
        except ProcessLookupError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--mpirun", required=True)
    parser.add_argument("--timeout-s", type=int, default=3600)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if list(root.glob("wrfout_d01_*")) or list(root.glob("rsl.*")):
        raise ValueError("use a fresh output directory")
    if Path("/tmp/wrf_gpu2_quiet").exists():
        print("QUIET", flush=True)
        return 3
    cpus = {8, 9, 12, 13}
    os.environ.update(JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1")
    command = ["taskset", "-c", "8,9,12,13", "nice", "-n", "19", args.mpirun,
               "--use-hwthread-cpus", "--bind-to", "none", "-np", "4", "./wrf.exe"]
    start, mono = time.time(), time.monotonic()
    paused_at, pauses, seen = None, [], {}
    last_progress, affinity = mono, {}
    status = "running"
    with (root.parent / "mpi.log").open("wb") as log:
        proc = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, env=os.environ.copy())

        def stop(signum, _frame):
            raise KeyboardInterrupt(f"signal {signum}")

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        (root.parent / "process.json").write_text(json.dumps({"wrapper_pid": os.getpid(), "mpi_pid": proc.pid,
            "mpi_pgid": proc.pid, "argv": command, "run_dir": str(root)}, indent=2) + "\n")
        try:
            while proc.poll() is None:
                now = time.monotonic()
                quiet = Path("/tmp/wrf_gpu2_quiet").exists()
                if quiet and paused_at is None:
                    signal_session(proc.pid, signal.SIGSTOP)
                    paused_at = now
                    print("QUIET: CPU-WRF process group paused; timing will be marked invalid", flush=True)
                elif not quiet and paused_at is not None:
                    signal_session(proc.pid, signal.SIGCONT)
                    pauses.append({"start_offset_s": paused_at - mono, "duration_s": now - paused_at})
                    paused_at = None
                    print("QUIET cleared: CPU-WRF resumed", flush=True)
                if now - mono > args.timeout_s:
                    status = "timeout"
                    break
                for f in root.glob("wrfout_d01_*"):
                    seen.setdefault(f.name, time.time() - start)
                if now - last_progress >= 30 or len(affinity) < 4:
                    for pid, (_, command_name) in session_processes(proc.pid).items():
                        if command_name != "wrf.exe":
                            continue
                        try:
                            mask = os.sched_getaffinity(pid)
                            affinity[str(pid)] = sorted(mask)
                            if not mask <= cpus:
                                raise RuntimeError(f"WRF rank escaped CPU grant: {pid}, {sorted(mask)}")
                        except (ProcessLookupError, FileNotFoundError):
                            continue
                    print(f"CPU-WRF elapsed={now-mono:.1f}s frames_seen={len(seen)} ranks={len(affinity)}", flush=True)
                    last_progress = now
                time.sleep(0.5)
            if status == "running":
                status = "finished"
        finally:
            if proc.poll() is None:
                signal_session(proc.pid, signal.SIGCONT)
                signal_session(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    signal_session(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=10)
            end = time.time()
            frames = [{"file": f.name, "bytes": f.stat().st_size, "last_write_s": f.stat().st_mtime - start,
                       "first_seen_s": seen.get(f.name)} for f in sorted(root.glob("wrfout_d01_*"))]
            rsl = root / "rsl.out.0000"
            success = rsl.exists() and "SUCCESS COMPLETE WRF" in rsl.read_text(errors="replace")
            complete = (proc.returncode == 0 and success and len(affinity) == 4 and len(frames) == 25
                        and frames[-1]["file"] == "wrfout_d01_2023-01-16_00:00:00")
            receipt = {"start_utc": dt.datetime.fromtimestamp(start, dt.timezone.utc).isoformat(),
                "end_utc": dt.datetime.fromtimestamp(end, dt.timezone.utc).isoformat(), "argv": command,
                "rc": proc.returncode, "status": status, "wrf_success": success, "complete_24h": complete,
                "wall_s": end - start, "last_file_wall_s": max((f["last_write_s"] for f in frames), default=None),
                "forecast_hours": 24, "frames": frames, "rank_affinity": affinity, "quiet_pauses": pauses,
                "timing_valid": complete and not pauses and paused_at is None,
                "power": "not measured; CPU energy needs separately labelled evidence"}
            (root.parent / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps({k: v for k, v in receipt.items() if k not in ("frames", "argv")}), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
