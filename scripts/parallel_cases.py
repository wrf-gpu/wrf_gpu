"""Run several WRF cases on one GPU at once: VRAM/host-RAM admission, one product CLI process per case.

Entry point: scripts/run_parallel_cases.sh (runs this file with JAX_PLATFORMS=cpu; the cases get the caller's
environment back). Each case runs ``python -m gpuwrf.cli run --input-dir CASE --output-dir OUT/<name>/wrfout ARGS``.

Per-case GPU need = the product C-auto plan of that case (gpu_allocator: pool budget + outside-pool headroom),
which the CLI records after a successful run of the same geometry/settings. A case without a plan runs ALONE
first (the CLI's demand allocation) and records it; later cases with the same plan then run in parallel.
--pool-gib P instead pins every case to a P GiB preallocated cuda_async pool (need = P + 1 GiB).
Successful cases record their peaks next to the plan (<plan>.measured.json): host need = 1.1 x the largest VmHWM, else
--host-gb-per-case; VRAM need >= 1.1 x the largest per-PID nvidia-smi peak (and that alone admits a case whose plan is
missing, e.g. under the vmm allocator, whose JAX memory_stats() are empty). JAX memory stats are never read.
Admission (FIFO): sum(need) <= free VRAM at launch - margin; sum(host need) <= MemAvailable + RSS of the
running cases - reserve; at most --max-parallel. Receipts: OUT/<name>/receipt.json (rc, wall, peak RSS, per-PID VRAM
peak from nvidia-smi every 5 s, outputs[] with publication times), OUT/parallel_run.json, OUT/gpu_dmon.log.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

GIB = 1024**3
REPO = Path(__file__).resolve().parents[1]
EXPLICIT_ALLOCATOR_ENV = ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_MEM_FRACTION")
# Case options after '--' are matched EXACTLY against this allowlist (no prefixes: the product's argparse would expand
# an abbreviation such as '--output-d' to --output-dir). Value = number of values. Everything else is refused.
CLI_ALLOWED = {"--domain": 1, "--max-dom": 1, "--hours": 1, "--domains-from-namelist": 0, "--emit-initial-history": 0,
               "--aot-prefetch": 0, "--no-aot-prefetch": 0, "--feedback": 0, "--score": 0, "--force-gpu-run": 0}
_PER_CASE = "per-case path; one shared value would mix the cases"
_NO_RESTART = "checkpoint/resume is not supported by the launcher (run such a case alone with the CLI)"
CLI_REFUSED = {"--input-dir": "set per case by the launcher", "--output-dir": "set per case by the launcher",
               "--namelist": "each case uses its own namelist.input", "--scratch-dir": _PER_CASE, "--proof-dir": _PER_CASE,
               "--compare-cpu-dir": _PER_CASE, "--dry-run": "use the launcher's own --dry-run",
               "--checkpoint-dir": _NO_RESTART, "--checkpoint-interval-steps": _NO_RESTART,
               "--checkpoint-max-bytes": _NO_RESTART, "--checkpoint-max-generations": _NO_RESTART,
               "--checkpoint-reserve-bytes": _NO_RESTART, "--resume-checkpoint": _NO_RESTART,
               "-h": "not a case option", "--help": "not a case option"}


def check_cli_args(tokens: list[str]) -> list[str]:
    """Validate the case options against CLI_ALLOWED by exact name (also --opt=VALUE); raise ValueError otherwise."""
    out, i = [], 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("-"):
            raise ValueError(f"unexpected value {tok!r}: case options must be exact long options")
        name, eq, value = tok.partition("=")
        if name in CLI_REFUSED:
            raise ValueError(f"{name}: {CLI_REFUSED[name]}")
        if name not in CLI_ALLOWED:
            raise ValueError(f"{tok!r} is not an allowed exact case option (no abbreviations); allowed: {', '.join(CLI_ALLOWED)}")
        if CLI_ALLOWED[name] == 0:
            if eq:
                raise ValueError(f"{name} takes no value")
            out.append(name)
            i += 1
        elif eq:
            if not value:
                raise ValueError(f"{name}= needs a value")
            out.append(tok)
            i += 1
        else:
            if i + 1 >= len(tokens) or tokens[i + 1].startswith("-"):
                raise ValueError(f"{name} needs a value")
            out += [name, tokens[i + 1]]
            i += 2
    return out
DEV_LOCK_FILE = Path("/tmp/wrf_gpu2_gpu.lock")  # dev GPU-lock infrastructure (gpu_preflight.LOCK_FILE)


def utc(t: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if t is None else t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def case_env() -> dict[str, str]:
    """The caller's environment for the cases (the wrapper forced JAX_PLATFORMS=cpu only for this process)."""
    env = dict(os.environ)
    saved = env.pop("_PARALLEL_CASES_JAX_PLATFORMS", None)
    if saved is not None:
        if saved == "<unset>":
            env.pop("JAX_PLATFORMS", None)
        else:
            env["JAX_PLATFORMS"] = saved
    return env


def case_names(dirs: list[Path]) -> list[str]:
    """Shortest unique '_'-joined path tails (alisios run dirs all end in run/run)."""
    parts = [[p for p in d.resolve().parts if p != "/"] for d in dirs]
    for k in range(1, max(len(p) for p in parts) + 1):
        names = ["_".join(p[-k:]) for p in parts]
        if len(set(names)) == len(names):
            return names
    raise ValueError("the same case directory is listed twice")


def _size_here(case_dirs: list[str], cli_args: list[str]) -> list[dict]:
    """Runs in a child with the case's environment: same import, same key path as `python -m gpuwrf.cli run`."""
    from gpuwrf import cli  # import-time env defaults (e.g. autotune XLA_FLAGS) apply exactly as in a case process
    from gpuwrf.runtime import gpu_allocator as ga
    out = []
    for case_dir in case_dirs:
        args = cli.build_parser().parse_args(["run", "--input-dir", case_dir, "--output-dir", "unused", *cli_args])
        if cli._effective_max_dom(args) <= 1:
            out.append({"error": f"{case_dir}: single-domain run has no C-auto plan; pass --pool-gib"})
            continue
        probe = dict(os.environ)
        probe["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # stop right after the plan key: no device query, no message
        with contextlib.redirect_stderr(io.StringIO()):
            ga.configure_cli_pool(args, probe)
        session = ga._SESSION
        if session is None:
            out.append({"source": "sizing-run", "need_bytes": None, "plan_path": None, "note": "case memory metadata unavailable"})
            continue
        rec = {"plan_key": session["key"], "plan_path": str(session["path"]), "domains": session["domains"]}
        try:
            plan = ga.read_plan(session["path"], session["key"], session["domains"])
        except (OSError, ValueError, KeyError, TypeError):
            out.append(rec | {"source": "sizing-run", "need_bytes": None})
            continue
        headroom = ga.outside_headroom(plan)
        out.append(rec | {"source": "c-auto-plan", "budget_bytes": int(plan["budget_bytes"]), "headroom_bytes": headroom,
                          "need_bytes": int(plan["budget_bytes"]) + headroom})
    bridge = sys.modules.get("jax._src.xla_bridge")
    if getattr(bridge, "_backends", None):
        raise SystemExit("sizing initialised a JAX backend")
    return out


def product_sizing(case_dirs: list[Path], cli_args: list[str], env: dict[str, str]) -> list[dict]:
    """Per-case need from the product C-auto plans, keyed exactly as each case's own CLI process keys it."""
    guard = {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}  # belt and braces: the import never initialises a backend
    res = subprocess.run([sys.executable, __file__, "--size-cases-json", json.dumps([[str(d) for d in case_dirs], cli_args])],
                         env=env | guard, capture_output=True, text=True, timeout=600)
    if res.returncode != 0:
        raise RuntimeError(f"case sizing failed: {res.stderr.strip()[-2000:]}")
    sizes = json.loads(res.stdout.strip().splitlines()[-1])
    for size in sizes:
        if "error" in size:
            raise ValueError(size["error"])
    return sizes


def measured(sizing: dict) -> dict:
    """Peaks of earlier successful launcher cases with the same plan key (sidecar <plan>.measured.json)."""
    if not sizing.get("plan_path"):
        return {}
    try:
        return json.loads(Path(sizing["plan_path"]).with_suffix(".measured.json").read_text())
    except (OSError, ValueError):
        return {}


def host_need_kb(sizing: dict, default_gb: float) -> tuple[int, str]:
    peak = int(measured(sizing).get("vmhwm_kb_max", 0))
    if peak > 0:
        return int(peak * 1.10), "1.1 x measured VmHWM"
    return int(default_gb * 1e6), "--host-gb-per-case"


def apply_vram_floor(sizing: dict) -> dict:
    """Need >= 1.1 x the measured per-PID device memory (nvidia-smi). Under the vmm allocator JAX memory_stats() is
    empty, so a C-auto plan may be missing or footprint-only; a measured peak then admits the case on its own."""
    fb = int(measured(sizing).get("vram_peak_mib_max", 0))
    if fb <= 0:
        return sizing
    floor = int(fb * 1.10 * 1024**2)
    if sizing.get("need_bytes") is None:  # no reserved pool: the case's own preflight must see its whole need free
        return sizing | {"source": "measured-fb", "need_bytes": floor, "measured_vram_peak_mib": fb,
                         "child_env": {"GPUWRF_MIN_FREE_VRAM_GIB": f"{floor / GIB:.2f}"}}
    return sizing | {"need_bytes": max(int(sizing["need_bytes"]), floor), "measured_vram_peak_mib": fb}


def record_peaks(sizing: dict, vmhwm_kb: int, vram_peak_mib: int) -> None:
    if not sizing.get("plan_path"):
        return
    old = measured(sizing)
    new = {"vmhwm_kb_max": max(int(old.get("vmhwm_kb_max", 0)), vmhwm_kb),
           "vram_peak_mib_max": max(int(old.get("vram_peak_mib_max", 0)), vram_peak_mib)}
    side = Path(sizing["plan_path"]).with_suffix(".measured.json")
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps(new) + "\n")


def gpu_memory() -> tuple[int, int]:
    """(CUDA-visible total, free) bytes via nvidia-smi; no CUDA context."""
    from gpuwrf.runtime.gpu_allocator import _pool_device_memory
    return _pool_device_memory()


def vram_by_pid() -> dict[int, int]:
    """Per-process device memory (MiB) from nvidia-smi; {} when unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return {int(p): int(m) for p, m in (l.split(",") for l in out.splitlines() if "," in l)}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


def mem_available_kb() -> int:
    return next(int(line.split()[1]) for line in open("/proc/meminfo") if line.startswith("MemAvailable"))


def proc_kb(pid: int, field: str) -> int:
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith(field + ":"):
                return int(line.split()[1])
    except OSError:
        pass
    return 0


TEARDOWN_GRACE_S = 60.0  # TERM -> KILL escalation for the cases' process groups
HELPER_GRACE_S = 10.0  # same for helper groups (dmon, compressor + its pool workers)


def signal_groups(cases: list, sig: int) -> None:
    for c in cases:
        if c.proc is not None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(c.proc.pid, sig)


def stop_process_groups(procs: list, grace_s: float) -> None:
    """TERM each process group (cases and helpers run in their own session), KILL the groups after grace_s, reap.

    A group signal also reaches descendants (compressor pool workers, dmon), which a plain terminate() of the
    parent would leave running."""
    for proc in procs:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGTERM)
    deadline = time.time() + grace_s
    while time.time() < deadline and any(proc.poll() is None for proc in procs):
        time.sleep(0.2)
    for proc in procs:  # the leader may be gone while descendants live on: always KILL the whole group
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
    for proc in procs:
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)


def teardown(cases: list) -> None:
    """TERM every started case's process group, KILL what is left after TEARDOWN_GRACE_S, reap."""
    stop_process_groups([c.proc for c in cases if c.proc is not None], TEARDOWN_GRACE_S)


def lock_fds(env: dict[str, str]) -> tuple[int, ...]:
    fd = env.get("GPUWRF_GPU_LOCK_FD", "")
    return (int(fd),) if fd.isdigit() and os.path.exists(f"/proc/self/fd/{fd}") else ()


class Case:
    def __init__(self, case_dir: Path, name: str, out: Path):
        self.case_dir, self.name, self.out = case_dir, name, out
        self.sizing: dict = {}
        self.host_kb, self.host_src = 0, ""
        self.proc: subprocess.Popen | None = None
        self.t0 = self.t1 = 0.0
        self.vmhwm_kb = 0
        self.concurrent_at_start = 0
        self.rc: int | None = None
        self.note = ""
        self.argv: list[str] = []
        self.env: list[str] = []  # GPUWRF_/JAX_/XLA_/... env of the case process (receipt evidence)
        self.published: dict[str, float] = {}  # history file -> first seen (s after start): survives later compression
        self.vram_peak_mib = 0

    def watch_outputs(self) -> None:
        for p in (self.out / "wrfout").glob("wrfout_d*"):
            if not p.name.endswith(".tmp") and p.name not in self.published:
                self.published[p.name] = round(time.time() - self.t0, 3)

    @property
    def need(self) -> int | None:
        return self.sizing.get("need_bytes")


def fits(running: list[Case], case: Case, vram_capacity: int, host_capacity_kb: int, max_parallel: int) -> bool:
    """FIFO admission of one more case next to the running ones (case.need None = sizing run, exclusive)."""
    if case.need is None or any(r.need is None for r in running):
        return not running
    return (len(running) < max_parallel
            and sum(r.need for r in running) + case.need <= vram_capacity
            and sum(r.host_kb for r in running) + case.host_kb <= host_capacity_kb)


def outputs(c: "Case") -> list[dict]:
    """History files in publication order: t_published_s = first seen by the 1 s poll (file mtimes change on compression)."""
    return [{"d": name[7:10], "file": name, "t_published_s": t, "bytes": (c.out / "wrfout" / name).stat().st_size}
            for name, t in sorted(c.published.items(), key=lambda kv: (kv[1], kv[0]))
            if (c.out / "wrfout" / name).exists()]


def cli_summary(path: Path) -> dict:
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return {k: d.get(k) for k in ("effective_hours", "effective_max_dom", "all_domains_finite", "all_outputs_present")}


def allocator_lines(log: Path) -> list[str]:
    try:
        return [l.strip() for l in log.read_text(errors="replace").splitlines()
                if l.startswith("gpuwrf: C-auto") or "re-exec with XLA_PYTHON_CLIENT_ALLOCATOR" in l][:4]
    except OSError:
        return []


def parse(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    own, cli_args = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    ap = argparse.ArgumentParser(prog="run_parallel_cases.sh", description=__doc__.split("\n")[0])
    ap.add_argument("cases", nargs="+", type=Path, help="case input directories (wrfinput_d0*, wrfbdy_d01, namelist.input)")
    ap.add_argument("--out-root", type=Path, required=True, help="one sub-directory per case is created here")
    ap.add_argument("--max-parallel", type=int, default=64)
    ap.add_argument("--pool-gib", type=float, default=None, help="fixed per-case pool instead of the C-auto plans")
    ap.add_argument("--host-gb-per-case", type=float, default=16.0, help="host RSS per case when none was measured yet")
    ap.add_argument("--host-reserve-gb", type=float, default=8.0)
    ap.add_argument("--min-mem-available-gb", type=float, default=6.0, help="host watchdog: stop all cases below this")
    ap.add_argument("--vram-margin-gib", type=float, default=0.0)
    ap.add_argument("--gpu-log-interval", type=int, default=5, help="nvidia-smi dmon -s pucm period in s (0 = off)")
    ap.add_argument("--vram-sample-interval", type=float, default=5.0, help="per-case VRAM sampling period in s")
    ap.add_argument("--compress", action="store_true", help="lossless rolling deflate of finished frames (nccopy -d1 -s)")
    ap.add_argument("--compress-workers", type=int, default=2)
    ap.add_argument("--compress-cpus", default="", help="taskset CPU list for the compressor")
    ap.add_argument("--dry-run", action="store_true", help="print the admission plan as JSON and exit")
    ap.add_argument("--cli-json", default=None, help=argparse.SUPPRESS)  # tests: replace the product CLI command
    a = ap.parse_args(own)
    try:
        cli_args = check_cli_args(cli_args)
    except ValueError as exc:
        ap.error(str(exc))
    return a, cli_args


def main(argv: list[str] | None = None, sizer=None, gpu=None, vram=None) -> int:
    sizer, gpu, vram = sizer or product_sizing, gpu or gpu_memory, vram or vram_by_pid
    a, cli_args = parse(sys.argv[1:] if argv is None else argv)
    env = case_env()
    forced = "--force-gpu-run" in cli_args or env.get("GPUWRF_FORCE_GPU_RUN") == "1"
    try:
        from gpuwrf.runtime.gpu_preflight import gpu_lock_required  # the cases' own nested-preflight lock policy
    except ImportError:  # products before the public run path always demanded the lock proof
        def gpu_lock_required(_env, lock_file):
            return True, "this gpuwrf version always requires the GPU lock"
    lock_required, lock_policy = gpu_lock_required(env, lock_file=DEV_LOCK_FILE)
    if not a.dry_run and lock_required and env.get("GPUWRF_GPU_LOCK_HELD") != "1" and not forced:
        print(f"run_parallel_cases: this machine requires the GPU lock ({lock_policy}); hold it for the whole batch:\n"
              "  scripts/with_gpu_lock.sh --label <name> -- scripts/run_parallel_cases.sh ...\n"
              "or append --force-gpu-run after '--' when no other process uses this GPU.", file=sys.stderr)
        return 2
    if a.pool_gib is None and any(env.get(k) for k in EXPLICIT_ALLOCATOR_ENV):
        print("run_parallel_cases: explicit XLA preallocation/fraction env would override the per-case sizing; "
              "unset it or use --pool-gib", file=sys.stderr)
        return 2
    names = case_names(a.cases)
    cases = [Case(d.resolve(), n, a.out_root.resolve() / n) for d, n in zip(a.cases, names)]
    if not a.dry_run:
        busy = [str(c.out) for c in cases if c.out.exists()]
        if busy:
            print(f"run_parallel_cases: output directories exist (history files are never overwritten): {busy}", file=sys.stderr)
            return 2
    total, free = gpu()
    vram_capacity = free - int(a.vram_margin_gib * GIB)
    pool_env: dict[str, str] = {}
    if a.pool_gib is not None:
        fraction = math.ceil(a.pool_gib * GIB / total * 1e6) / 1e6
        # The pool is reserved before the case's preflight reads free VRAM: it needs only the outside headroom.
        pool_env = {"XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async", "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
                    "XLA_CLIENT_MEM_FRACTION": f"{fraction:.6f}", "GPUWRF_MIN_FREE_VRAM_GIB": "1.0"}

    def size(batch: list[Case]) -> None:
        unsized = [c for c in batch if a.pool_gib is None and c.need is None]
        if a.pool_gib is not None:
            for c in batch:
                c.sizing = {"source": "pool-gib", "need_bytes": int((a.pool_gib + 1.0) * GIB), "child_env": pool_env}
        elif unsized:
            for c, sizing in zip(unsized, sizer([c.case_dir for c in unsized], cli_args, env)):
                c.sizing = sizing
        for c in batch:
            if a.pool_gib is None:
                c.sizing = apply_vram_floor(c.sizing)
            c.host_kb, c.host_src = host_need_kb(c.sizing, a.host_gb_per_case)

    try:
        size(cases)
    except (ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"run_parallel_cases: {exc}", file=sys.stderr)
        return 2
    host_capacity = lambda running: (mem_available_kb() + sum(proc_kb(r.proc.pid, "VmRSS") for r in running)  # noqa: E731
                                     - int(a.host_reserve_gb * 1e6))
    if a.dry_run:
        start_now: list[Case] = []
        for c in cases:
            if not fits(start_now, c, vram_capacity, host_capacity([]), a.max_parallel):
                break
            start_now.append(c)
        print(json.dumps({"vram_total_bytes": total, "vram_free_bytes": free, "vram_capacity_bytes": vram_capacity,
                          "host_capacity_kb": host_capacity([]), "would_start_now": len(start_now),
                          "cases": [{"name": c.name, "case_dir": str(c.case_dir), **{k: v for k, v in c.sizing.items() if k != "child_env"},
                                     "host_need_kb": c.host_kb, "host_need_source": c.host_src} for c in cases]}, indent=1))
        return 0

    root = a.out_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    run = {"schema": "gpuwrf-parallel-cases-v1", "start_utc": utc(), "argv": sys.argv if argv is None else argv,
           "cli_args": cli_args, "vram_total_bytes": total, "vram_free_bytes_at_start": free,
           "vram_capacity_bytes": vram_capacity, "cpu_affinity": sorted(os.sched_getaffinity(0)), "events": [],
           "gpu_lock": {"required": lock_required, "policy": lock_policy, "held": env.get("GPUWRF_GPU_LOCK_HELD") == "1"}}
    t_start = time.time()
    cli = json.loads(a.cli_json) if a.cli_json else [sys.executable, "-m", "gpuwrf.cli", "run"]
    stop = {"reason": None, "t": 0.0}
    pending, running, done = list(cases), [], []
    helpers: list[subprocess.Popen] = []
    compressor = None
    peak, t_vram = 0, 0.0

    def on_signal(signum, _frame):
        if stop["reason"] is None:
            stop["reason"], stop["t"] = f"signal {signum}", time.time()
            signal_groups(running, signal.SIGTERM)

    old_handlers = {s: signal.signal(s, on_signal) for s in (signal.SIGTERM, signal.SIGINT)}
    error: BaseException | None = None
    try:
        if a.gpu_log_interval > 0:
            with contextlib.suppress(OSError):
                helpers.append(subprocess.Popen(["nvidia-smi", "dmon", "-s", "pucm", "-d", str(a.gpu_log_interval), "-o", "T"],
                                                stdout=open(root / "gpu_dmon.log", "w"), stderr=subprocess.STDOUT,
                                                start_new_session=True))
        if a.compress:
            cmd = [sys.executable, str(REPO / "scripts/compress_wrfout.py"), str(root), "--workers", str(a.compress_workers)]
            if a.compress_cpus:
                cmd = ["taskset", "-c", a.compress_cpus, *cmd]
            compressor = subprocess.Popen(["nice", "-n", "19", *cmd], stdout=open(root / "compress.log", "w"),
                                          stderr=subprocess.STDOUT, env=env | {"JAX_PLATFORMS": "cpu"},
                                          start_new_session=True)
        while pending or running:
            for r in [r for r in running if r.proc.poll() is not None]:
                r.t1, r.rc = time.time(), r.proc.returncode
                r.watch_outputs()
                running.remove(r)
                done.append(r)
                if r.rc == 0:
                    record_peaks(r.sizing, r.vmhwm_kb, r.vram_peak_mib)
                    with contextlib.suppress(ValueError, RuntimeError, subprocess.TimeoutExpired):
                        size(pending)  # a sizing run may have recorded the plan (and host peak) the others need
                write_receipt(r, cli)
            while pending and stop["reason"] is None:
                c = pending[0]
                if not fits(running, c, vram_capacity, host_capacity(running), a.max_parallel):
                    if not running:  # does not fit even alone
                        c.note = "not admitted: needs more VRAM or host RAM than available"
                        c.rc = 75
                        done.append(pending.pop(0))
                        write_receipt(c, cli)
                        continue
                    break
                pending.pop(0)
                c.out.mkdir(parents=True)
                c.concurrent_at_start = len(running)
                child_env = env | c.sizing.get("child_env", {})
                c.env = sorted(f"{k}={v}" for k, v in child_env.items()
                               if k.startswith(("GPUWRF_", "JAX_", "XLA_", "OMP_", "CUDA_", "TF_")) and "TOKEN" not in k)
                c.t0 = time.time()
                c.argv = [*cli, "--input-dir", str(c.case_dir), "--output-dir", str(c.out / "wrfout"), *cli_args]
                try:
                    c.proc = subprocess.Popen(c.argv,
                                              stdout=open(c.out / "cli_stdout.json", "w"), stderr=open(c.out / "run.log", "w"),
                                              env=child_env, start_new_session=True,
                                              pass_fds=lock_fds(env))
                except BaseException as exc:
                    c.note, c.rc = f"launch failed: {type(exc).__name__}: {exc}", None
                    done.append(c)
                    raise
                running.append(c)
                run["events"].append({"t_s": round(c.t0 - t_start, 1), "start": c.name, "running": len(running)})
            if stop["reason"] is not None and pending:
                for c in pending:
                    c.note, c.rc = f"not started: {stop['reason']}", None
                    done.append(c)
                    write_receipt(c, cli)
                pending.clear()
            peak = max(peak, len(running))
            for r in running:
                r.vmhwm_kb = max(r.vmhwm_kb, proc_kb(r.proc.pid, "VmHWM"))
                r.watch_outputs()
            if running and time.time() - t_vram >= a.vram_sample_interval:
                t_vram, by_pid = time.time(), vram()
                for r in running:
                    r.vram_peak_mib = max(r.vram_peak_mib, by_pid.get(r.proc.pid, 0))
            if running and stop["reason"] is None and mem_available_kb() < a.min_mem_available_gb * 1e6:
                on_signal(0, None)
                stop["reason"] = f"host watchdog: MemAvailable < {a.min_mem_available_gb} GB"
                run["events"].append({"t_s": round(time.time() - t_start, 1), "watchdog": stop["reason"]})
            if stop["reason"] is not None and time.time() - stop["t"] > TEARDOWN_GRACE_S:
                signal_groups(running, signal.SIGKILL)
            time.sleep(1.0)
    except BaseException as exc:  # never leave a started case or helper behind; the original error is re-raised below
        error = exc
        stop["reason"] = stop["reason"] or f"launcher error: {type(exc).__name__}: {exc}"
        teardown(running)
        for r in list(running):
            r.t1, r.rc = time.time(), r.proc.returncode
            r.note = r.note or f"stopped: {stop['reason']}"
            running.remove(r)
            done.append(r)
        for c in pending:
            c.note, c.rc = f"not started: {stop['reason']}", None
            done.append(c)
        pending.clear()
        for c in done:
            with contextlib.suppress(Exception):
                write_receipt(c, cli)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        abnormal = error is not None or stop["reason"] is not None
        if compressor is not None and not abnormal:  # normal end: let the compressor finish its final pass
            with contextlib.suppress(OSError):
                (root / ".cases_done").write_text(utc() + "\n")
            compressor.wait()
        stop_process_groups(helpers + ([compressor] if compressor is not None and abnormal else []), HELPER_GRACE_S)
    t_end = time.time()
    ok = [c for c in done if c.rc == 0]
    hours = [cli_summary(c.out / "cli_stdout.json").get("effective_hours") for c in ok]
    run.update(end_utc=utc(t_end), wall_s=round(t_end - t_start, 3), peak_concurrency=peak, stop_reason=stop["reason"],
               cases={c.name: {"rc": c.rc, "note": c.note, "wall_s": round(c.t1 - c.t0, 3) if c.proc else None,
                               "vram_peak_mib": c.vram_peak_mib, "vmhwm_kb": c.vmhwm_kb} for c in done},
               case_hours=sum(h for h in hours if h) if all(hours) else None)
    if error is not None:
        run["launcher_error"] = f"{type(error).__name__}: {error}"
        with contextlib.suppress(Exception):
            (root / "parallel_run.json").write_text(json.dumps(run, indent=1) + "\n")
        raise error
    if ok and len(ok) == len(cases) and run["case_hours"]:
        run["wall_s_per_case_hour"] = round(run["wall_s"] / run["case_hours"], 3)
    if compressor is not None:
        run["compress_rc"] = compressor.returncode
    (root / "parallel_run.json").write_text(json.dumps(run, indent=1) + "\n")
    print(json.dumps({k: run[k] for k in ("wall_s", "peak_concurrency", "cases", "stop_reason")} |
                     {"wall_s_per_case_hour": run.get("wall_s_per_case_hour")}))
    return 0 if len(ok) == len(cases) else 1


_SOURCE: dict = {}


def source_ref() -> dict:
    """The source the cases import (computed once): gpuwrf package location and, for a git checkout, commit/src tree/clean."""
    if not _SOURCE:
        import importlib.util
        spec = importlib.util.find_spec("gpuwrf")
        pkg = Path(spec.origin).resolve().parent if spec and spec.origin else None
        git = lambda *a: subprocess.run(["git", "-C", str(pkg or REPO), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
        _SOURCE.update({"gpuwrf_path": str(pkg) if pkg else None, "git_head": git("rev-parse", "HEAD") or None,
                        "src_tree": git("rev-parse", "HEAD:src/gpuwrf") or None,
                        "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))})
    return dict(_SOURCE)


def write_receipt(c: Case, cli: list[str]) -> None:
    c.out.mkdir(parents=True, exist_ok=True)
    rec = {"schema": "gpuwrf-parallel-case-v1", "name": c.name, "case_dir": str(c.case_dir), "output_dir": str(c.out / "wrfout"),
           "source": source_ref(),
           "command": c.argv or [*cli, "--input-dir", str(c.case_dir), "--output-dir", str(c.out / "wrfout")],
           "sizing": {k: v for k, v in c.sizing.items()}, "host_need_kb": c.host_kb, "host_need_source": c.host_src,
           "rc": c.rc, "note": c.note, "concurrent_at_start": c.concurrent_at_start, "env": c.env}
    if c.proc is not None:
        rec |= {"pid": c.proc.pid, "start_utc": utc(c.t0), "end_utc": utc(c.t1), "wall_s": round(c.t1 - c.t0, 3),
                "vmhwm_kb": c.vmhwm_kb, "allocator": allocator_lines(c.out / "run.log"),
                "vram_peak_mib": c.vram_peak_mib, "cli_summary": cli_summary(c.out / "cli_stdout.json"), "outputs": outputs(c)}
    (c.out / "receipt.json").write_text(json.dumps(rec, indent=1) + "\n")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--size-cases-json"]:
        print(json.dumps(_size_here(*json.loads(sys.argv[2]))))
        raise SystemExit(0)
    raise SystemExit(main())
