"""Keep the GPU lease until nsys and its recorded target have exited."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def alive(record):
    try:
        stat = Path(f"/proc/{record['pid']}/stat").read_text().rsplit(")",1)[1].split()
        return stat[19] == record["starttime"] and stat[0] != "Z"
    except FileNotFoundError:
        return False


def main():
    if Path("/tmp/wrf_gpu2_quiet").exists():
        print("quiet window active; release GPU without starting profiler", flush=True)
        return 125
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-pid-file", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.app_pid_file.exists():
        raise SystemExit("requires a command and a fresh target PID file")
    child = subprocess.Popen(command, start_new_session=True)
    interrupted = False

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        if args.app_pid_file.exists():
            record = json.loads(args.app_pid_file.read_text())
            if alive(record):
                os.kill(record["pid"], signal.SIGTERM)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while child.poll() is None:
        if interrupted:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait()
        else:
            time.sleep(.2)
    rc = child.returncode
    record = json.loads(args.app_pid_file.read_text()) if args.app_pid_file.exists() else None
    if record and alive(record):
        print("nsys returned while target still alive; waiting inside GPU lock",flush=True)
        until = time.monotonic()+120
        while alive(record) and not interrupted and time.monotonic()<until:
            time.sleep(1)
        if alive(record):
            os.kill(record["pid"], signal.SIGTERM)
            until = time.monotonic()+10
            while alive(record) and time.monotonic()<until:
                time.sleep(.2)
            if alive(record):
                os.kill(record["pid"], signal.SIGKILL)
                while alive(record):
                    time.sleep(.2)
            rc = rc or 1
    print("profiler and recorded target exited; GPU lease can return",flush=True)
    return 124 if interrupted else rc


if __name__ == "__main__":
    raise SystemExit(main())
