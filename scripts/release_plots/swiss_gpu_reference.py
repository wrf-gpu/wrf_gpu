#!/usr/bin/env python3
"""Measure the shipped single-domain CLI in a fresh arm under the dev GPU lock.

This parent uses only the standard library and never creates a CUDA pool.
The child runs the public CLI and asserts its actual JAX backend before exit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


CHILD = """
import json, os, sys
from pathlib import Path
import gpuwrf
assert Path(gpuwrf.__file__).resolve().is_relative_to(Path(os.environ['SWISS_SRC_ROOT']).resolve()), gpuwrf.__file__
from gpuwrf.cli import main
rc = main(sys.argv[1:])
import jax
device = jax.devices()[0]
assert device.platform == 'gpu', device
Path(os.environ['SWISS_DEVICE_PROOF']).write_text(json.dumps({
    'platform': device.platform, 'device': str(device), 'device_kind': device.device_kind,
    'gpuwrf_path': str(Path(gpuwrf.__file__).resolve()),
    'cli_rc': rc, 'effective_environment': {
        k: v for k, v in os.environ.items() if k.startswith(('GPUWRF_', 'JAX_', 'XLA_', 'OMP_', 'CUDA_'))
        and not any(s in k for s in ('LOCK', 'TOKEN', 'LEASE', 'APPROVAL'))
    }}, indent=2) + '\\n')
raise SystemExit(rc)
"""


def stop_group(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--arm-dir", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("cold", "cached"))
    parser.add_argument("--python", required=True)
    parser.add_argument("--timeout-s", type=int, default=1800)
    args = parser.parse_args()
    if os.environ.get("GPUWRF_GPU_LOCK_HELD") != "1":
        raise RuntimeError("run via scripts/with_gpu_lock.sh --label release-docs")
    lock_fd = int(os.environ["GPUWRF_GPU_LOCK_FD"])
    os.fstat(lock_fd)
    if not os.path.samefile(f"/proc/self/fd/{lock_fd}", os.environ["GPUWRF_GPU_LOCK_FILE"]):
        raise RuntimeError("inherited GPU lock descriptor does not match the lock file")
    repo, arm = args.repo.resolve(), args.arm_dir.resolve()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        raise RuntimeError("model snapshot must be clean")
    arm.mkdir(parents=True, exist_ok=False)
    case = repo / "examples/switzerland_d01"
    env = dict(os.environ, JAX_PLATFORMS="cuda", PYTHONPATH=str(repo / "src"),
               SWISS_DEVICE_PROOF=str(arm / "device.json"), SWISS_SRC_ROOT=str(repo / "src"))
    command = [args.python, "-u", "-c", CHILD, "run", "--input-dir", str(case),
        "--output-dir", str(arm / "wrfout"), "--proof-dir", str(arm / "proofs"),
        "--scratch-dir", str(arm / "scratch"), "--domain", "d01", "--hours", "24"]
    start, end, status = None, None, "running"
    with (arm / "gpu_dmon.log").open("wb") as power, (arm / "cli.log").open("wb") as log:
        monitor = subprocess.Popen(["nvidia-smi", "dmon", "-s", "pucm", "-d", "1", "-o", "T"],
                                   stdout=power, stderr=subprocess.STDOUT, start_new_session=True)
        start = time.time()
        try:
            proc = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True, pass_fds=(lock_fd,))
        except BaseException:
            stop_group(monitor)
            raise

        def interrupted(signum, _frame):
            raise KeyboardInterrupt(f"signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            try:
                proc.wait(timeout=args.timeout_s)
                status = "finished"
            except subprocess.TimeoutExpired:
                status = "timeout"
            except KeyboardInterrupt:
                status = "interrupted"
        finally:
            stop_group(proc)
            end = time.time()
            stop_group(monitor)
    frames = [{"file": f.name, "bytes": f.stat().st_size, "last_write_s": f.stat().st_mtime - start}
              for f in sorted((arm / "wrfout").glob("wrfout_d01_*"))]
    proof = json.loads((arm / "device.json").read_text()) if (arm / "device.json").exists() else {}
    expected = [f"wrfout_d01_{(dt.datetime(2023, 1, 15)+dt.timedelta(hours=h)).strftime('%Y-%m-%d_%H:%M:%S')}"
                for h in range(25)]
    complete = (proc.returncode == 0 and proof.get("platform") == "gpu"
                and [f["file"] for f in frames] == expected and all(f["bytes"] > 0 for f in frames)
                and all(0 <= f["last_write_s"] <= end - start + 0.1 for f in frames))
    receipt = {"phase": args.phase, "start_utc": dt.datetime.fromtimestamp(start, dt.timezone.utc).isoformat(),
        "end_utc": dt.datetime.fromtimestamp(end, dt.timezone.utc).isoformat(), "rc": proc.returncode,
        "status": status, "forecast_hours": 24, "complete_24h": complete, "device": proof,
        "wall_s": end - start, "last_file_wall_s": max((f["last_write_s"] for f in frames), default=None),
        "frames": frames, "cli_args": command[4:],
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "src_tree": subprocess.check_output(["git", "rev-parse", "HEAD:src/gpuwrf"], cwd=repo, text=True).strip(),
        "inputs_sha256": {n: hashlib.sha256((case/n).read_bytes()).hexdigest()
                          for n in ("namelist.input", "wrfinput_d01", "wrfbdy_d01")},
        "timing_method": "CLI process creation through original file writes; power samples cover the CLI process"}
    (arm / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("frames", "cli_args", "device")}), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
