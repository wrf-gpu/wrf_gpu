"""Run several WRF cases on one GPU at once: VRAM/host-RAM admission, one product CLI process per case.

Entry point: scripts/run_parallel_cases.sh (runs this file with JAX_PLATFORMS=cpu; the cases get the caller's
environment back). Each case runs ``python -m gpuwrf.cli run --input-dir CASE --output-dir OUT/<name>/wrfout ARGS``.

Per-case GPU need = the product C-auto plan of that case (gpu_allocator: pool budget + outside-pool headroom),
which the CLI records after a successful run of the same geometry/settings. A case without a plan runs ALONE
first (the CLI's demand allocation) and records it; later cases with the same plan then run in parallel.
--pool-gib P instead pins every case to a P GiB preallocated cuda_async pool (need = P + 1 GiB).
Successful cases record their peaks next to the plan (<plan>.measured.json), COLD (compiling, mixed or unknown) and WARM
(every required executable loaded from a persistent AOT blob + .meta, per the case proof) separately: host need = 1.1 x the
warm VmHWM while every artifact of the last warm run still exists, else 1.1 x the cold VmHWM, else --host-gb-per-case;
VRAM need >= 1.1 x the largest per-PID nvidia-smi peak (and that alone admits a case whose plan is missing, e.g. under the
vmm allocator, whose JAX memory_stats() are empty). JAX memory stats are never read.
Admission (FIFO): sum(need) <= free VRAM at launch - margin; sum(host need) + one-compile headroom (largest cold - admitted
need) <= MemAvailable + RSS of the running cases - reserve; at most --max-parallel. Receipts: OUT/<name>/receipt.json (rc, wall, peak RSS, per-PID VRAM
peak from nvidia-smi every 5 s, outputs[] with publication times), OUT/parallel_run.json, OUT/gpu_dmon.log.

Several GPUs (--gpus all | 0,1,...; ensemble mode): one process per case, each bound to ONE GPU via CUDA_VISIBLE_DEVICES=<index>
+ CUDA_DEVICE_ORDER=PCI_BUS_ID (CUDA index == nvidia-smi index). Sizing, the C-auto plan (keyed by CUDA_VISIBLE_DEVICES) and
VRAM admission are per GPU; host-RAM admission stays global. A case starts on the admitting GPU with the fewest running cases
(then most free VRAM, then lowest index); a sizing run is exclusive on its GPU only. Without --gpus nothing changes (one GPU,
the caller's CUDA_VISIBLE_DEVICES). One failed case never stops the others; --checkpoint-interval-steps N gives every case its
own checkpoint store OUT/<name>/checkpoint, and --retry-failed K re-runs a failed case (rc not 0/2/75, launcher not stopping) up to
K times: resumed from its newest untried verified-looking generation (the CLI verifies it), else from scratch after moving the
failed attempt's history to OUT/<name>/failed-attempt-<k>/.
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
OPTIMIZER_REGISTRY_HOOKS = 2  # includes terminal-completion receipts; checked before optimizer members launch
REPO = Path(__file__).resolve().parents[1]
EXPLICIT_ALLOCATOR_ENV = ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_MEM_FRACTION")
# Case options after '--' are matched EXACTLY against this allowlist (no prefixes: the product's argparse would expand
# an abbreviation such as '--output-d' to --output-dir). Value = number of values. Everything else is refused.
CLI_ALLOWED = {"--domain": 1, "--max-dom": 1, "--hours": 1, "--domains-from-namelist": 0, "--emit-initial-history": 0,
               "--aot-prefetch": 0, "--no-aot-prefetch": 0, "--feedback": 0, "--score": 0, "--force-gpu-run": 0,
               # One parameter file forwarded unchanged to every case (parameter studies under one AOT key, manager
               # 2026-10-24 21:19Z); its qualification is the family status in docs/runtime-parameters.md.
               "--runtime-parameters": 1}
_PER_CASE = "per-case path; one shared value would mix the cases"
_NO_RESTART = "checkpoint/resume is not supported by the launcher (run such a case alone with the CLI)"
_SINGLE_CASE = ("single-case window/checkpoint-branch feature; run such a case directly with "
                "`gpuwrf run` (docs/checkpoint-branching.md)")
CLI_REFUSED = {"--input-dir": "set per case by the launcher", "--output-dir": "set per case by the launcher",
               "--namelist": "each case uses its own namelist.input", "--scratch-dir": _PER_CASE, "--proof-dir": _PER_CASE,
               "--compare-cpu-dir": _PER_CASE, "--dry-run": "use the launcher's own --dry-run",
               "--checkpoint-dir": _NO_RESTART, "--checkpoint-interval-steps": _NO_RESTART,
               "--checkpoint-max-bytes": _NO_RESTART, "--checkpoint-max-generations": _NO_RESTART,
               "--checkpoint-reserve-bytes": _NO_RESTART, "--resume-checkpoint": _NO_RESTART,
               "--extend-run": _NO_RESTART,
               "--decompose": "domain decomposition spans several GPUs per case; run it alone with the CLI",
               "--surface-controls": "per-case control file; use the launcher's --case-surface-controls NAME",
               "--history-znt": "EXPERIMENTAL z0bare ZNT history output is a single-run research option (resume-bound, B>1 refused)",
               "--minutes": _SINGLE_CASE, "--end-time": _SINGLE_CASE,
               "--branch-from": _SINGLE_CASE, "--branch-manifest": _SINGLE_CASE,
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


def _blob_and_meta(entry: dict) -> list[str] | None:
    """The persistent AOT artifact an executable was loaded from: [blob, meta] (both required), else None."""
    blob, meta = entry.get("blob_path"), entry.get("meta_path")
    return [str(blob), str(meta)] if blob and meta else None


def cache_state(out: Path) -> dict:
    """Was this finished case WARM? Read from the case's own proof (wrfout/proofs/nested_pipeline_run.json): EVERY
    executable the run needed (metadata.nested_aot.domains: the root plus every fused/nested program) must have been loaded
    from a persistent AOT artifact in this process — source "aot_blob" with its blob + .meta, or "aot_memory_cache" with
    load_count >= 1 served by the startup prefetch (aot_prefetch.domains[name], source "aot_blob"; its cheap key may be an
    alias of the runtime key: the prefetch registers a loaded executable under every alias with the same HLO, E149) — and
    none may have been compiled here (compiled_memory_cache / fallback / aot_written / load_count 0 / not loaded). The
    product's own source label is the evidence: an executable compiled in-process is labelled compiled_memory_cache. Fail closed:
    anything unknown is cold. Returns {"warm": True|False|None, "executables": {name: [blob, meta]} (warm only),
    "cold_executables": [names]}; no proof or no nested AOT record -> warm None (recorded as cold)."""
    try:
        meta = json.loads((out / "wrfout" / "proofs" / "nested_pipeline_run.json").read_text()).get("metadata") or {}
        doms = (meta.get("nested_aot") or {}).get("domains") or {}
        pre = (meta.get("aot_prefetch") or {}).get("domains") or {}
    except (OSError, ValueError, AttributeError):
        return {"warm": None, "executables": {}, "cold_executables": []}
    if not isinstance(doms, dict) or not doms:
        return {"warm": None, "executables": {}, "cold_executables": []}
    exes, cold = {}, []
    for name, d in sorted(doms.items()):
        art = None
        if isinstance(d, dict) and d.get("loaded") is True and not d.get("aot_written") and not d.get("error"):
            if d.get("source") == "aot_blob":
                art = _blob_and_meta(d)
            elif d.get("source") == "aot_memory_cache" and int(d.get("load_count") or 0) >= 1:
                p = pre.get(name) if isinstance(pre, dict) else None
                if isinstance(p, dict) and p.get("source") == "aot_blob" and p.get("loaded") is True and not p.get("error"):
                    art = _blob_and_meta(p)
        if art:
            exes[name] = art
        else:
            cold.append(name)
    return {"warm": not cold, "executables": exes if not cold else {}, "cold_executables": cold}


def cache_warm(sizing: dict) -> bool:
    """Warm admission for this plan key: a warm peak was recorded AND every artifact (blob + .meta of every required
    executable) that warm run loaded still exists (same plan key = same source, flags and geometry, so the next case loads
    exactly those executables instead of compiling)."""
    m = measured(sizing)
    exes = m.get("warm_executables")
    return (int(m.get("vmhwm_kb_warm_max", 0)) > 0 and isinstance(exes, dict) and bool(exes)
            and all(isinstance(v, list) and len(v) == 2 and all(Path(f).is_file() for f in v) for v in exes.values()))


def host_cold_kb(sizing: dict, default_gb: float) -> tuple[int, str]:
    """Host need if this case COMPILES: 1.1 x the cold (compiling) VmHWM; a v0.3.0 sidecar without the cold/warm split
    -> 1.1 x its VmHWM (a max over all runs, i.e. cold); else never less than --host-gb-per-case."""
    m = measured(sizing)
    cold = int(m.get("vmhwm_kb_cold_max", 0))
    if cold > 0:
        return int(cold * 1.10), "1.1 x measured cold VmHWM"
    legacy = int(m.get("vmhwm_kb_max", 0))
    if legacy > 0 and "vmhwm_kb_warm_max" not in m:
        return int(legacy * 1.10), "1.1 x measured VmHWM"
    return int(default_gb * 1e6), "--host-gb-per-case"


def host_need_kb(sizing: dict, default_gb: float) -> tuple[int, str]:
    """Host need = 1.1 x the WARM VmHWM while the AOT cache is warm for this plan key (cache_warm), else the cold need.
    A cold compile needs ~2.4x the host memory of a warm run (WN3: 14.8 vs 6.2 GB), so one max() over both would admit
    warm production runs as if every case compiled."""
    if cache_warm(sizing):
        return int(int(measured(sizing)["vmhwm_kb_warm_max"]) * 1.10), "1.1 x measured warm VmHWM (AOT cache warm)"
    return host_cold_kb(sizing, default_gb)


def apply_vram_floor(sizing: dict) -> dict:
    """Need >= 1.1 x the measured per-PID device memory (nvidia-smi). Under the vmm allocator JAX memory_stats() is
    empty, so a C-auto plan may be missing or footprint-only; a measured peak then admits the case on its own."""
    m = measured(sizing)
    fb = int(m.get("vram_peak_mib_max", 0))  # max over cold AND warm: warm runs peak higher in VRAM (prefetch residency)
    if fb <= 0:
        return sizing
    floor = int(fb * 1.10 * 1024**2)
    report = {"measured_vram_peak_mib": fb, "measured_vram_peak_mib_cold": int(m.get("vram_peak_mib_cold_max", 0)) or None,
              "measured_vram_peak_mib_warm": int(m.get("vram_peak_mib_warm_max", 0)) or None,
              "plan_need_bytes": sizing.get("need_bytes")}  # plan vs measured: a C-auto pool reserves its plan regardless
    if sizing.get("need_bytes") is None:  # no reserved pool: the case's own preflight must see its whole need free
        return sizing | report | {"source": "measured-fb", "need_bytes": floor,
                                  "child_env": {"GPUWRF_MIN_FREE_VRAM_GIB": f"{floor / GIB:.2f}"}}
    return sizing | report | {"need_bytes": max(int(sizing["need_bytes"]), floor)}


def record_peaks(sizing: dict, vmhwm_kb: int, vram_peak_mib: int, state: dict | None = None) -> None:
    """Peaks per plan key; cold (compiling, mixed or unknown) and warm (cache_state warm) recorded separately. A warm run
    also records the artifacts it loaded (warm_executables, replaced by the latest warm run) for cache_warm()."""
    if not sizing.get("plan_path"):
        return
    old = measured(sizing)
    warm = (state or {}).get("warm") is True
    tag = "warm" if warm else "cold"
    new = old | {"vmhwm_kb_max": max(int(old.get("vmhwm_kb_max", 0)), vmhwm_kb),
                 "vram_peak_mib_max": max(int(old.get("vram_peak_mib_max", 0)), vram_peak_mib),
                 f"vmhwm_kb_{tag}_max": max(int(old.get(f"vmhwm_kb_{tag}_max", 0)), vmhwm_kb),
                 f"vram_peak_mib_{tag}_max": max(int(old.get(f"vram_peak_mib_{tag}_max", 0)), vram_peak_mib)}
    if warm:
        new["warm_executables"] = state["executables"]
    side = Path(sizing["plan_path"]).with_suffix(".measured.json")
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps(new) + "\n")


def gpu_memory() -> tuple[int, int]:
    """(CUDA-visible total, free) bytes via nvidia-smi; no CUDA context."""
    from gpuwrf.runtime.gpu_allocator import _pool_device_memory
    return _pool_device_memory()


def gpu_devices() -> list[dict]:
    """Every GPU nvidia-smi lists (PCI bus order): index, uuid, name, compute capability, CUDA-visible total and free bytes.
    No CUDA context. total = memory.total - memory.reserved (the denominator JAX applies a pool fraction to)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.reserved,memory.free,compute_cap,name",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=True).stdout
    devs = []
    for line in out.strip().splitlines():
        idx, uuid, total, reserved, free, cap, name = (p.strip() for p in line.split(",", 6))
        devs.append({"index": idx, "uuid": uuid, "name": name, "compute_cap": cap,
                     "total_bytes": (int(total) - int(reserved)) * 1024**2, "free_bytes": int(free) * 1024**2})
    return devs


def select_devices(spec: str, devices: list[dict]) -> list[dict]:
    """--gpus all | comma-separated nvidia-smi indices -> the chosen device records (refuses unknown/duplicate indices and a
    mix of compute capabilities: the AOT/JIT cache directory is keyed by the FIRST GPU's capability, compile_cache._cuda_tag)."""
    if spec.strip().lower() == "all":
        chosen = list(devices)
    else:
        tokens = [t.strip() for t in spec.split(",") if t.strip()]
        if len(set(tokens)) != len(tokens):
            raise ValueError(f"--gpus lists a GPU twice: {spec!r}")
        by_index = {d["index"]: d for d in devices}
        unknown = [t for t in tokens if t not in by_index]
        if unknown:
            raise ValueError(f"--gpus: no GPU with nvidia-smi index {unknown}; present: {sorted(by_index)}")
        chosen = [by_index[t] for t in tokens]
    if not chosen:
        raise ValueError("--gpus selects no GPU")
    caps = sorted({d["compute_cap"] for d in chosen})
    if len(caps) > 1:
        raise ValueError(f"--gpus mixes compute capabilities {caps}: run one batch per GPU model")
    return chosen


def device_env(dev: dict | None) -> dict[str, str]:
    """Child env binding a case to one GPU (empty without --gpus: the caller's CUDA_VISIBLE_DEVICES is kept)."""
    if dev is None or dev.get("index") is None:
        return {}
    return {"CUDA_VISIBLE_DEVICES": str(dev["index"]), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


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
        self.host_cold_kb = 0  # host need if this case compiles after all (one-compile headroom, see fits)
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
        self.cache: dict = {}  # cache_state() of the finished case (warm/cold evidence)
        self.device: dict | None = None  # GPU this case runs on (--gpus), None = the caller's default device
        self.per_dev: dict = {}  # device key -> (sizing, host_kb, host_src, host_cold_kb): plans are keyed per GPU
        self.attempt = 0  # retry number (--retry-failed); earlier attempts' receipts: OUT/<name>/attempt-<k>.receipt.json
        self.resume_from: str | None = None  # checkpoint generation this attempt resumes from
        self.tried_generations: set[str] = set()
        self.registry_invocation = None  # optimize-only; one start/end per actual attempt

    def use_device(self, key) -> None:
        self.sizing, self.host_kb, self.host_src, self.host_cold_kb = self.per_dev[key]

    def watch_outputs(self) -> None:
        for p in (self.out / "wrfout").glob("wrfout_d*"):
            if not p.name.endswith(".tmp") and p.name not in self.published:
                self.published[p.name] = round(time.time() - self.t0, 3)

    @property
    def need(self) -> int | None:
        return self.sizing.get("need_bytes")

    @property
    def device_key(self):
        return None if self.device is None else self.device["index"]


def fits(running: list[Case], case: Case, vram_capacity: int, host_capacity_kb: int, max_parallel: int) -> bool:
    """FIFO admission of one more case next to the running ones (case.need None = sizing run, exclusive). Host: the sum of
    the needs plus ONE-COMPILE HEADROOM — the largest (cold - admitted) need among them — so that one case admitted at its
    warm need that compiles after all (e.g. an evicted blob, an env change) still fits instead of tripping the host-RAM
    watchdog for every running case."""
    if case.need is None or any(r.need is None for r in running):
        return not running
    group = [*running, case]
    headroom = max(max(0, c.host_cold_kb - c.host_kb) for c in group)
    return (len(running) < max_parallel
            and sum(r.need for r in running) + case.need <= vram_capacity
            and sum(c.host_kb for c in group) + headroom <= host_capacity_kb)


def fits_device(running: list[Case], case: Case, dev_key, vram_capacity: int, host_capacity_kb: int, max_parallel: int,
                max_per_gpu: int) -> bool:
    """fits() for one GPU of several: VRAM need and the exclusive sizing run count per GPU (running cases with the same
    device key), host RAM and --max-parallel over ALL running cases. With one GPU this is exactly fits()."""
    on_dev = [r for r in running if r.device_key == dev_key]
    if len(running) >= max_parallel or len(on_dev) >= max_per_gpu:
        return False
    if case.need is None or any(r.need is None for r in on_dev):
        if on_dev:
            return False
        if not running:
            return True  # alone on the machine, whatever the budget (as fits)
    elif sum(r.need for r in on_dev) + case.need > vram_capacity:
        return False
    group = [*running, case]
    headroom = max(max(0, c.host_cold_kb - c.host_kb) for c in group)
    return sum(c.host_kb for c in group) + headroom <= host_capacity_kb


def latest_untried_generation(store: Path, tried: set[str]) -> str | None:
    """Newest checkpoint generation (manifest created_ns) not tried before. Only the manifest is read here; the CLI
    verifies size/hash/identity on resume and fails the attempt otherwise (the next retry then takes the next older one)."""
    gens = []
    for man in store.glob("generation-*/manifest.json"):
        with contextlib.suppress(OSError, ValueError, TypeError, KeyError, AttributeError):
            gens.append((int(json.loads(man.read_text())["created_ns"]), str(man.parent)))
    for _, gen in sorted(gens, reverse=True):
        if gen not in tried:
            return gen
    return None


NO_RETRY_RC = (0, 2, 75)  # success, CLI refusal (deterministic), not admitted


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
    ap.add_argument("--gpus", default=None, help="'all' or comma-separated nvidia-smi indices: one process per case, each bound "
                    "to one of these GPUs (ensemble mode); default: one GPU, the caller's CUDA_VISIBLE_DEVICES")
    ap.add_argument("--max-per-gpu", type=int, default=None, help="at most this many cases per GPU (default: VRAM decides)")
    ap.add_argument("--checkpoint-interval-steps", type=int, default=0,
                    help="every case checkpoints into its own OUT/<name>/checkpoint every N root steps (CLI restart store)")
    ap.add_argument("--retry-failed", type=int, default=0, help="re-run a failed case up to K times (resume from its own "
                    "checkpoint when one exists); other cases are never stopped by a failure")
    ap.add_argument("--case-surface-controls", default=None, metavar="NAME",
                    help="Track P: every case directory holds its own control file NAME, passed as that case's "
                    "--surface-controls (gpuwrf optimize continuous members); a case without it is refused")
    ap.add_argument("--case-runtime-parameters", default=None, metavar="NAME",
                    help="every case holds its own runtime parameter JSON NAME, passed as --runtime-parameters")
    ap.add_argument("--dry-run", action="store_true", help="print the admission plan as JSON and exit")
    ap.add_argument("--cli-json", default=None, help=argparse.SUPPRESS)  # tests: replace the product CLI command
    a = ap.parse_args(own)
    if a.checkpoint_interval_steps < 0 or a.retry_failed < 0 or (a.max_per_gpu is not None and a.max_per_gpu < 1):
        ap.error("--checkpoint-interval-steps/--retry-failed must be >= 0 and --max-per-gpu >= 1")
    try:
        cli_args = check_cli_args(cli_args)
    except ValueError as exc:
        ap.error(str(exc))
    name = a.case_surface_controls
    if name is not None:
        if name in ("", ".", "..") or name != Path(name).name:
            ap.error("--case-surface-controls takes a plain file name inside each case directory")
        missing = [str(d) for d in a.cases if not (d / name).is_file()]
        if missing:
            ap.error(f"--case-surface-controls {name}: no such file in {missing}")
    name = a.case_runtime_parameters
    if name is not None:
        if name in ("", ".", "..") or name != Path(name).name:
            ap.error("--case-runtime-parameters takes a plain file name inside each case directory")
        if any(token.split("=", 1)[0] == "--runtime-parameters" for token in cli_args):
            ap.error("per-case runtime parameters cannot be combined with a shared --runtime-parameters")
        missing = [str(d) for d in a.cases if not (d / name).is_file()]
        if missing:
            ap.error(f"--case-runtime-parameters {name}: no such file in {missing}")
    return a, cli_args


def main(argv: list[str] | None = None, sizer=None, gpu=None, vram=None, devices=None) -> int:
    sizer, gpu, vram, devices = sizer or product_sizing, gpu or gpu_memory, vram or vram_by_pid, devices or gpu_devices
    a, cli_args = parse(sys.argv[1:] if argv is None else argv)
    env = case_env()
    deadline_text = env.get("ALISIOS_OPTIMIZE_WALL_DEADLINE")
    deadline = None if deadline_text is None else float(deadline_text)
    if deadline is not None and not math.isfinite(deadline):
        raise ValueError("optimizer wallclock deadline must be finite")
    forced = "--force-gpu-run" in cli_args or env.get("GPUWRF_FORCE_GPU_RUN") == "1"
    try:
        from gpuwrf.runtime.gpu_preflight import gpu_lock_required  # the cases' own nested-preflight lock policy
    except ImportError:  # products before the public run path always demanded the lock proof
        def gpu_lock_required(_env, lock_file):
            return True, "this gpuwrf version always requires the GPU lock"
    lock_required, lock_policy = gpu_lock_required(env, lock_file=DEV_LOCK_FILE)
    cpu_only = env.get("JAX_PLATFORMS", "").strip().lower() == "cpu"  # CPU-only cases never create a CUDA client
    if cpu_only:
        lock_required, lock_policy = False, "CPU-only cases (JAX_PLATFORMS=cpu)"
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
    if a.gpus is not None:
        if env.get("CUDA_VISIBLE_DEVICES") is not None:
            print("run_parallel_cases: --gpus selects physical nvidia-smi indices; unset CUDA_VISIBLE_DEVICES", file=sys.stderr)
            return 2
        try:
            devs = select_devices(a.gpus, devices())
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            print(f"run_parallel_cases: {exc}", file=sys.stderr)
            return 2
        dev_items = [(d["index"], d) for d in devs]
        totals = {d["index"]: (d["total_bytes"], d["free_bytes"]) for d in devs}
    else:
        dev_items = [(None, None)]
        totals = {None: gpu()}
    total, free = totals[dev_items[0][0]]
    capacity = {k: f - int(a.vram_margin_gib * GIB) for k, (_t, f) in totals.items()}
    if cpu_only:  # no GPU memory is held: only host RAM and --max-parallel admit CPU cases
        capacity = {k: 1 << 62 for k in capacity}
    vram_capacity = capacity[dev_items[0][0]]
    pool_env: dict = {}
    if a.pool_gib is not None:
        for k, (t, _f) in totals.items():
            fraction = math.ceil(a.pool_gib * GIB / t * 1e6) / 1e6
            # The pool is reserved before the case's preflight reads free VRAM: it needs only the outside headroom.
            pool_env[k] = {"XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async", "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
                           "XLA_CLIENT_MEM_FRACTION": f"{fraction:.6f}", "GPUWRF_MIN_FREE_VRAM_GIB": "1.0"}

    def size(batch: list[Case]) -> None:
        """Per GPU (the C-auto plan key contains CUDA_VISIBLE_DEVICES): size the unsized cases with that GPU's env, then
        the VRAM floor and host needs; the case's current sizing = its assigned GPU's, else the first GPU's."""
        for key, dev in dev_items:
            if a.pool_gib is not None:
                for c in batch:
                    c.per_dev[key] = ({"source": "pool-gib", "need_bytes": int((a.pool_gib + 1.0) * GIB),
                                       "child_env": pool_env[key]}, 0, "", 0)
            else:
                unsized = [c for c in batch if key not in c.per_dev or c.per_dev[key][0].get("need_bytes") is None]
                if unsized:
                    for c, sizing in zip(unsized, sizer([c.case_dir for c in unsized], cli_args, env | device_env(dev))):
                        c.per_dev[key] = (sizing, 0, "", 0)
            for c in batch:
                sizing = c.per_dev[key][0]
                if a.pool_gib is None:
                    sizing = apply_vram_floor(sizing)
                host_kb, host_src = host_need_kb(sizing, a.host_gb_per_case)
                c.per_dev[key] = (sizing, host_kb, host_src, max(host_kb, host_cold_kb(sizing, a.host_gb_per_case)[0]))
        for c in batch:
            c.use_device(c.device_key if c.device is not None else dev_items[0][0])

    max_per_gpu = a.max_per_gpu or 1 << 30
    order = {k: i for i, (k, _d) in enumerate(dev_items)}

    def place(running: list[Case], c: Case, host_cap: int) -> bool:
        """Admit c on the admitting GPU with the fewest running cases, then most free planned VRAM, then lowest index."""
        choices = []
        for key, dev in dev_items:
            c.use_device(key)
            if fits_device(running, c, key, capacity[key], host_cap, a.max_parallel, max_per_gpu):
                on_dev = [r for r in running if r.device_key == key]
                choices.append((len(on_dev), -(capacity[key] - sum(r.need or 0 for r in on_dev)), order[key], key, dev))
        if not choices:
            c.use_device(c.device_key if c.device is not None else dev_items[0][0])
            return False
        *_rank, key, dev = min(choices, key=lambda t: t[:3])
        c.use_device(key)
        c.device = dev
        return True

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
            if not place(start_now, c, host_capacity([])):
                break
            start_now.append(c)
        print(json.dumps({"vram_total_bytes": total, "vram_free_bytes": free, "vram_capacity_bytes": vram_capacity,
                          "host_capacity_kb": host_capacity([]), "would_start_now": len(start_now),
                          **({"devices": [d | {"capacity_bytes": capacity[k]} for k, d in dev_items]} if a.gpus else {}),
                          "cases": [{"name": c.name, "case_dir": str(c.case_dir), **{k: v for k, v in c.sizing.items() if k != "child_env"},
                                     **({"gpu": c.device_key} if a.gpus else {}),
                                     "host_need_kb": c.host_kb, "host_need_source": c.host_src, "host_need_cold_kb": c.host_cold_kb,
                                     "aot_cache_warm": cache_warm(c.sizing),
                                     "measured_peaks": {k: v for k, v in measured(c.sizing).items() if k != "warm_executables"}}
                                    for c in cases]}, indent=1))
        return 0

    root = a.out_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    run = {"schema": "gpuwrf-parallel-cases-v1", "start_utc": utc(), "argv": sys.argv if argv is None else argv,
           "cli_args": cli_args, "vram_total_bytes": total, "vram_free_bytes_at_start": free,
           "vram_capacity_bytes": vram_capacity, "cpu_affinity": sorted(os.sched_getaffinity(0)), "events": [],
           **({"devices": [d | {"capacity_bytes": capacity[k]} for k, d in dev_items]} if a.gpus else {}),
           "checkpoint_interval_steps": a.checkpoint_interval_steps, "retry_failed": a.retry_failed,
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
            if deadline is not None and time.monotonic() >= deadline and stop["reason"] is None:
                on_signal(0, None)
                stop["reason"] = "optimizer hard wallclock budget exhausted"
            for r in [r for r in running if r.proc.poll() is not None]:
                r.t1, r.rc = time.time(), r.proc.returncode
                if r.registry_invocation is not None:
                    r.registry_invocation.finish(r.rc, aborted=stop["reason"] is not None)
                r.watch_outputs()
                running.remove(r)
                if r.rc == 0:
                    r.cache = cache_state(r.out)
                    record_peaks(r.sizing, r.vmhwm_kb, r.vram_peak_mib, r.cache)
                    with contextlib.suppress(ValueError, RuntimeError, subprocess.TimeoutExpired):
                        size(pending)  # a sizing run may have recorded the plan (and host peak) the others need
                write_receipt(r, cli)
                if (r.rc not in NO_RETRY_RC and r.rc is not None and r.attempt < a.retry_failed
                        and stop["reason"] is None):
                    prepare_retry(r, a.checkpoint_interval_steps > 0)
                    pending.insert(0, r)  # a retry goes first: the other cases keep running meanwhile
                    run["events"].append({"t_s": round(time.time() - t_start, 1), "retry": r.name, "attempt": r.attempt,
                                          "resume_from": r.resume_from})
                    continue
                done.append(r)
            while pending and stop["reason"] is None:
                if deadline is not None and time.monotonic() >= deadline:
                    on_signal(0, None)
                    stop["reason"] = "optimizer hard wallclock budget exhausted"
                    break
                c = pending[0]
                if not place(running, c, host_capacity(running)):
                    if not running:  # does not fit even alone
                        c.note = "not admitted: needs more VRAM or host RAM than available"
                        c.rc = 75
                        done.append(pending.pop(0))
                        write_receipt(c, cli)
                        continue
                    break
                pending.pop(0)
                c.out.mkdir(parents=True, exist_ok=c.attempt > 0)
                c.concurrent_at_start = len(running)
                child_env = env | c.sizing.get("child_env", {}) | device_env(c.device)
                c.env = sorted(f"{k}={v}" for k, v in child_env.items()
                               if k.startswith(("GPUWRF_", "JAX_", "XLA_", "OMP_", "CUDA_", "TF_")) and "TOKEN" not in k)
                c.t0 = time.time()
                c.argv = [*cli, "--input-dir", str(c.case_dir), "--output-dir", str(c.out / "wrfout"), *cli_args]
                if a.case_surface_controls is not None:
                    c.argv += ["--surface-controls", str(c.case_dir / a.case_surface_controls)]
                if a.case_runtime_parameters is not None:
                    c.argv += ["--runtime-parameters", str(c.case_dir / a.case_runtime_parameters)]
                if a.checkpoint_interval_steps > 0:
                    c.argv += ["--checkpoint-dir", str(c.out / "checkpoint"),
                               "--checkpoint-interval-steps", str(a.checkpoint_interval_steps)]
                if c.resume_from:
                    c.argv += ["--resume-checkpoint", c.resume_from]
                try:
                    if env.get("ALISIOS_OPTIMIZE_REGISTRY"):
                        from gpuwrf.optimize.registry import Invocation
                        c.registry_invocation = Invocation(env["ALISIOS_OPTIMIZE_REGISTRY"], c.case_dir, c.out)
                        child_env = c.registry_invocation.begin(child_env)
                    c.proc = subprocess.Popen(c.argv,
                                              stdout=open(c.out / "cli_stdout.json", "w"), stderr=open(c.out / "run.log", "w"),
                                              env=child_env, start_new_session=True,
                                              pass_fds=lock_fds(env))
                except BaseException as exc:
                    if c.registry_invocation is not None:
                        c.registry_invocation.finish(1, aborted=isinstance(exc, (KeyboardInterrupt, SystemExit)))
                    c.note, c.rc = f"launch failed: {type(exc).__name__}: {exc}", None
                    done.append(c)
                    raise
                running.append(c)
                run["events"].append({"t_s": round(c.t0 - t_start, 1), "start": c.name, "running": len(running),
                                      **({"gpu": c.device_key} if c.device is not None else {})})
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
        registry_errors = []
        for c in cases:
            if c.registry_invocation is not None:
                try:
                    c.registry_invocation.finish(c.rc, aborted=error is not None or stop["reason"] is not None)
                except Exception as exc:
                    registry_errors.append(exc)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        abnormal = error is not None or stop["reason"] is not None
        if compressor is not None and not abnormal:  # normal end: let the compressor finish its final pass
            with contextlib.suppress(OSError):
                (root / ".cases_done").write_text(utc() + "\n")
            compressor.wait()
        stop_process_groups(helpers + ([compressor] if compressor is not None and abnormal else []), HELPER_GRACE_S)
        if registry_errors:
            raise registry_errors[0]  # never publish success after a missing terminal event
    t_end = time.time()
    ok = [c for c in done if c.rc == 0]
    hours = [cli_summary(c.out / "cli_stdout.json").get("effective_hours") for c in ok]
    run.update(end_utc=utc(t_end), wall_s=round(t_end - t_start, 3), peak_concurrency=peak, stop_reason=stop["reason"],
               cases={c.name: {"rc": c.rc, "note": c.note, "wall_s": round(c.t1 - c.t0, 3) if c.proc else None,
                               "vram_peak_mib": c.vram_peak_mib, "vmhwm_kb": c.vmhwm_kb,
                               **({"gpu": c.device_key} if c.device is not None else {}),
                               **({"attempts": c.attempt + 1, "resumed_from": c.resume_from} if a.retry_failed else {})}
                      for c in done},
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


def prepare_retry(c: Case, checkpointing: bool) -> None:
    """Set a failed case up for its next attempt: keep the failed attempt's receipt/logs as OUT/<name>/attempt-<k>.*, then
    resume from the newest untried checkpoint generation of the case's OWN store, else start from scratch after moving the
    failed history (never overwritten, never mixed) and store to OUT/<name>/failed-attempt-<k>/."""
    k = c.attempt
    for name in ("receipt.json", "run.log", "cli_stdout.json"):
        with contextlib.suppress(OSError):
            (c.out / name).rename(c.out / f"attempt-{k}.{name}")
    gen = latest_untried_generation(c.out / "checkpoint", c.tried_generations) if checkpointing else None
    if gen is not None:
        c.tried_generations.add(gen)
    else:
        aside = c.out / f"failed-attempt-{k}"
        for name in ("wrfout", "checkpoint"):
            if (c.out / name).exists():
                aside.mkdir(exist_ok=True)
                (c.out / name).rename(aside / name)
    c.resume_from = gen
    c.attempt = k + 1
    c.proc, c.rc, c.note, c.device = None, None, "", None
    c.t0 = c.t1 = 0.0
    c.vmhwm_kb = c.vram_peak_mib = 0
    c.published, c.cache, c.argv, c.env = {}, {}, [], []


def write_receipt(c: Case, cli: list[str]) -> None:
    c.out.mkdir(parents=True, exist_ok=True)
    rec = {"schema": "gpuwrf-parallel-case-v1", "name": c.name, "case_dir": str(c.case_dir), "output_dir": str(c.out / "wrfout"),
           "source": source_ref(),
           "command": c.argv or [*cli, "--input-dir", str(c.case_dir), "--output-dir", str(c.out / "wrfout")],
           "sizing": {k: v for k, v in c.sizing.items()}, "host_need_kb": c.host_kb, "host_need_source": c.host_src,
           "host_need_cold_kb": c.host_cold_kb,
           "rc": c.rc, "note": c.note, "concurrent_at_start": c.concurrent_at_start, "env": c.env,
           "attempt": c.attempt, "resumed_from": c.resume_from}
    if c.registry_invocation is not None:
        rec["registry_run_id"] = c.registry_invocation.run_id
        rec["registry_complete"] = c.registry_invocation.complete
    if c.device is not None:
        rec["gpu"] = {k: c.device.get(k) for k in ("index", "uuid", "name", "compute_cap")}
    if c.proc is not None:
        rec |= {"pid": c.proc.pid, "start_utc": utc(c.t0), "end_utc": utc(c.t1), "wall_s": round(c.t1 - c.t0, 3),
                "vmhwm_kb": c.vmhwm_kb, "allocator": allocator_lines(c.out / "run.log"),
                "vram_peak_mib": c.vram_peak_mib, "aot_cache_warm": c.cache.get("warm"),
                "aot_cold_executables": c.cache.get("cold_executables"),
                "cli_summary": cli_summary(c.out / "cli_stdout.json"), "outputs": outputs(c)}
    (c.out / "receipt.json").write_text(json.dumps(rec, indent=1) + "\n")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--size-cases-json"]:
        print(json.dumps(_size_here(*json.loads(sys.argv[2]))))
        raise SystemExit(0)
    raise SystemExit(main())
