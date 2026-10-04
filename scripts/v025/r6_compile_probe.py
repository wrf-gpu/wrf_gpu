#!/usr/bin/env python3
"""R6 compile-recovery probe — PARENT (measurement orchestration only).

One cold-compile / cached-load / warm measurement of ONE forecast entry:

  * creates a unique EMPTY persistent-cache dir (cold) or reuses a populated
    one (--reuse-cache, cached-load),
  * binds the R5-frozen child environment (autotune-0 XLA flags, cache on,
    WRF source authority roots; published in probe.json),
  * launches the GPU child (sprint-folder r6_compile_probe_child.py — it owns
    the jax imports; see its docstring for why it lives outside scripts/v025),
  * timestamps every child log line, samples child RSS + nvidia-smi GPU bytes,
    enforces a wall cap, and writes probe.json with SHA-256 of every artifact.

Refuses to run outside scripts/with_gpu_lock.sh (GPUWRF_GPU_LOCK_HELD).

Roles: this file implements ONLY the ``run`` (parent) role; the child role
lives in .agent/sprints/2026-09-17-v0250-r6-compile-recovery/r6_compile_probe_child.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
CHILD = (REPO / ".agent/sprints/2026-09-17-v0250-r6-compile-recovery/"
         "r6_compile_probe_child.py").resolve()
FAST_RUN_DIR = Path("<DATA_ROOT>/wrf_gpu2/v025/m0/cpu_arms/fastbind_r1")
DEFAULT_OUT_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v025/r6")
LOCK_FILE = "/tmp/wrf_gpu2_gpu.lock"
STRIPPED_ENV = ("JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES", "XLA_FLAGS", "JAX_PLATFORM_NAME")

# R5-frozen child environment (scripts/v025/m0_window_parent.py::_base_child_environment)
# minus the cache dir, which is per run.
R5_FROZEN_ENV = {
    "XLA_FLAGS": "--xla_gpu_autotune_level=0",
    "GPUWRF_JAX_CACHE": "1",
    "GPUWRF_JAX_CACHE_LOCK": "0",
    "JAX_COMPILATION_CACHE_MAX_SIZE": "-1",
    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none",
    "GPUWRF_XLA_AUTOTUNE_CACHE": "0",
    "GPUWRF_WRF_ROOT": "<USER_HOME>/src/wrf_pristine/WRF",
    "GPUWRF_WRF_SRC": "<USER_HOME>/src/wrf_pristine/WRF",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def _cache_inventory(cache_dir: Path) -> dict[str, Any]:
    entries = [
        {"name": p.name, "bytes": p.stat().st_size}
        for p in sorted(cache_dir.rglob("*")) if p.is_file()
    ]
    large = [e for e in entries if e["bytes"] >= (1 << 20)]
    return {
        "entry_count": len(entries),
        "large_entry_count_ge_1MiB": len(large),
        "large_entries": sorted(large, key=lambda e: -e["bytes"])[:12],
        "total_bytes": sum(e["bytes"] for e in entries),
    }


def _lock_rows() -> list[str]:
    try:
        st = os.stat(LOCK_FILE)
    except OSError:
        return []
    # /proc/locks prints MAJOR:MINOR:INODE (major/minor hex, inode DECIMAL); match the
    # FULL key -- an inode-only match can hit an unrelated file on another device.
    key = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
    return [
        line for line in Path("/proc/locks").read_text().splitlines()
        if len(line.split()) >= 6 and line.split()[5] == key
    ]


def _nvidia_smi(*query: str) -> str:
    try:
        return subprocess.run(["nvidia-smi", *query], capture_output=True, text=True, timeout=20).stdout
    except Exception as exc:  # noqa: BLE001
        return f"nvidia-smi failed: {exc}"


def _gpu_mem_used_by_pid(pid: int) -> int | None:
    out = _nvidia_smi("--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits")
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
            try:
                return int(parts[1]) * 1024 * 1024
            except ValueError:
                return None
    return None


def _rss(pid: int) -> tuple[int | None, int | None]:
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None, None
    rss = hwm = None
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) * 1024
        elif line.startswith("VmHWM:"):
            hwm = int(line.split()[1]) * 1024
    return rss, hwm


def run_main(args: argparse.Namespace) -> int:
    if os.environ.get("GPUWRF_GPU_LOCK_HELD") != "1":
        print("r6_compile_probe: refuse -- run under scripts/with_gpu_lock.sh", file=sys.stderr)
        return 2
    if not CHILD.is_file():
        print(f"r6_compile_probe: child script missing: {CHILD}", file=sys.stderr)
        return 2
    out_dir = Path(args.out_root) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    if args.reuse_cache:
        cache_dir = Path(args.reuse_cache)
        if not cache_dir.is_dir():
            print(f"r6_compile_probe: --reuse-cache {cache_dir} missing", file=sys.stderr)
            return 2
        cache_mode = "REUSED (cached-load pass)"
    else:
        cache_dir = out_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=False)
        cache_mode = "UNIQUE EMPTY (cold pass)"
    cache_before = _cache_inventory(cache_dir)

    env = {k: v for k, v in os.environ.items()
           if k not in STRIPPED_ENV and not (k.startswith("GPUWRF_") and not k.startswith("GPUWRF_GPU_LOCK"))}
    env.update(R5_FROZEN_ENV)
    env["GPUWRF_JAX_CACHE_DIR"] = str(cache_dir.resolve())
    if args.loop_mode:
        env["GPUWRF_ADVANCE_CHUNK_LOOP"] = args.loop_mode
    for extra in args.env or []:
        key, _, value = extra.partition("=")
        env[key] = value

    child_log = out_dir / "child.log"
    result_path = out_dir / "result.json"
    samples_path = out_dir / "samples.jsonl"
    hlo_path = out_dir / "hlo_optimized.txt.gz"
    final_path = out_dir / "final_state.npz"
    child_cmd = [
        sys.executable, str(CHILD), "child",
        "--entry", args.entry, "--hours", str(args.hours), "--run-dir", str(args.run_dir),
        "--invoke", str(args.invoke), "--result", str(result_path),
    ]
    if args.segment_steps:
        child_cmd += ["--segment-steps", str(args.segment_steps)]
    if args.save_final:
        child_cmd += ["--save-final", str(final_path)]
    if args.hlo_text:
        child_cmd += ["--hlo-text", str(hlo_path)]

    probe: dict[str, Any] = {
        "schema": "wrf_gpu2.v025.r6.compile_probe.v1",
        "run_id": args.run_id,
        "label": args.label,
        "entry": args.entry,
        "segment_steps": args.segment_steps,
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                   capture_output=True, text=True).stdout.strip(),
        "worktree": str(REPO),
        "child_script": str(CHILD),
        "cache_dir": str(cache_dir),
        "cache_mode": cache_mode,
        "cache_before": cache_before,
        "wall_cap_s": args.wall_cap,
        "child_command": child_cmd,
        "child_env_bound": {k: env[k] for k in sorted(set(R5_FROZEN_ENV) | {"GPUWRF_JAX_CACHE_DIR"} | ({"GPUWRF_ADVANCE_CHUNK_LOOP"} if args.loop_mode else set()))},
        "child_env_stripped": [k for k in STRIPPED_ENV if k in os.environ],
        "lock": {"holder_file": Path(LOCK_FILE + ".holder").read_text().strip() if Path(LOCK_FILE + ".holder").exists() else "",
                 "proc_locks_rows_before": _lock_rows(), "label": os.environ.get("GPUWRF_GPU_LOCK_LABEL")},
        "nvidia_smi_before": _nvidia_smi(),
        "loadavg_before": Path("/proc/loadavg").read_text().strip(),
        "started_utc": _utc_now(),
    }
    _write_json(out_dir / "probe.json", probe)

    launch_ns = time.monotonic_ns()
    env["GPUWRF_R6_PARENT_LAUNCH_NS"] = str(launch_ns)
    log_handle = child_log.open("w", encoding="utf-8")
    proc = subprocess.Popen(child_cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            since = (time.monotonic_ns() - launch_ns) / 1e9
            log_handle.write(f"[+{since:9.3f}s] {line}")
            log_handle.flush()

    pumper = threading.Thread(target=pump, daemon=True)
    pumper.start()

    peak_rss = peak_hwm = peak_gpu = 0
    killed = False
    with samples_path.open("w") as samples:
        while proc.poll() is None:
            since = (time.monotonic_ns() - launch_ns) / 1e9
            rss, hwm = _rss(proc.pid)
            gpu = _gpu_mem_used_by_pid(proc.pid)
            peak_rss = max(peak_rss, rss or 0)
            peak_hwm = max(peak_hwm, hwm or 0)
            peak_gpu = max(peak_gpu, gpu or 0)
            samples.write(json.dumps({"t": round(since, 3), "rss": rss, "hwm": hwm, "gpu_used": gpu}) + "\n")
            samples.flush()
            if since > args.wall_cap:
                killed = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                break
            time.sleep(args.sample_period)
    rc = proc.wait()
    pumper.join(timeout=30)
    log_handle.close()
    elapsed = (time.monotonic_ns() - launch_ns) / 1e9

    child = json.loads(result_path.read_text()) if result_path.exists() else {"status": "NO_RESULT_FILE"}
    artifacts = {}
    for p in (child_log, result_path, samples_path, hlo_path, final_path):
        if p.exists():
            artifacts[p.name] = {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
    probe.update({
        "finished_utc": _utc_now(),
        "child_returncode": rc,
        "child_elapsed_since_launch_s": elapsed,
        "wall_cap_killed": killed,
        "peak_child_rss_bytes": peak_rss,
        "peak_child_rss_hwm_bytes": peak_hwm,
        "peak_child_gpu_used_bytes": peak_gpu,
        "child_result": child,
        "cache_after": _cache_inventory(cache_dir),
        "nvidia_smi_after": _nvidia_smi(),
        "loadavg_after": Path("/proc/loadavg").read_text().strip(),
        "proc_locks_rows_after": _lock_rows(),
        "artifacts": artifacts,
        "status": ("WALL_CAP_KILLED" if killed else child.get("status", "UNKNOWN")) if rc != 0 or killed else child.get("status"),
    })
    _write_json(out_dir / "probe.json", probe)
    summary = {k: child.get(k) for k in ("status", "steps", "cadence", "segment_steps", "lower_seconds",
                                          "compile_seconds", "executable_ready_since_parent_launch_s",
                                          "warm_median_s", "warm_s_per_fc_h")}
    summary.update({"rc": rc, "elapsed_s": round(elapsed, 1), "killed": killed,
                    "peak_rss_GiB": round(peak_hwm / 2**30, 2), "peak_gpu_GiB": round(peak_gpu / 2**30, 2)})
    print("R6_PROBE_SUMMARY " + json.dumps(summary, sort_keys=True))
    return 0 if (rc == 0 and not killed) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    run = parser.add_subparsers(dest="role", required=True).add_parser("run")
    run.add_argument("--entry", choices=("mono", "segmented", "single"), required=True)
    run.add_argument("--segment-steps", type=int, default=None)
    run.add_argument("--hours", type=float, default=1.0)
    run.add_argument("--run-dir", type=Path, default=FAST_RUN_DIR)
    run.add_argument("--invoke", type=int, default=0,
                     help="public-entry invocations (index 0 = first warm execution in-process)")
    run.add_argument("--run-id", required=True)
    run.add_argument("--label", default="r6-compile-recovery")
    run.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    run.add_argument("--reuse-cache", type=Path, default=None,
                     help="cached-load pass: existing populated cache dir")
    run.add_argument("--wall-cap", type=float, default=3600.0)
    run.add_argument("--sample-period", type=float, default=2.0)
    run.add_argument("--save-final", action="store_true")
    run.add_argument("--hlo-text", action="store_true")
    run.add_argument("--loop-mode", default=None,
                     help="GPUWRF_ADVANCE_CHUNK_LOOP override (segmented only)")
    run.add_argument("--env", action="append", default=[],
                     help="extra KEY=VALUE for the child (disclosed in probe.json)")
    args = parser.parse_args()
    return run_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
