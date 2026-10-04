"""S0 G1 harness: production nested run with timing receipts.
Bench-owned copy of .agent/sprints/2026-09-23-v0250-endgame-frontrunner/tools/s0_nested_harness.py;
the only change is REPO now honours $BENCH_SRC_REPO so one harness can measure any --src tree.

Runs the real `gpuwrf.cli run` nested path in-process and adds observation only:
  * root-segment periods (wrapper around nested_pipeline.run_operational_domain_tree);
  * per-domain advance / force host-dispatch records (wrapper around the callables
    handed to domain_tree.run_domain_tree_callbacks);
  * output-callback wall per domain (class-level wrapper of _PerDomainWrfoutWriter.__call__);
  * every JAX trace / lowering / backend-compile / cache event (jax.monitoring);
  * jax.profiler.TraceAnnotation ranges ("S0:*") that nsys records as NVTX "TSL:" ranges;
  * 1 s per-process VRAM samples (nvidia-smi --query-compute-apps) when a GPU is present.
Numerics, executables and scheduling are untouched: every wrapper forwards arguments
and return values unchanged. XLA_PYTHON_CLIENT_ALLOCATOR must be set by the caller
(production value cuda_async) so the CLI does not re-exec and drop the wrappers.

Usage (GPU, inside scripts/with_gpu_lock.sh):
  python s0_nested_harness.py --input-dir DIR --out RUNDIR --hours 3 --tag TAG
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# bench: source tree is parameterized (scripts/bench_prod.sh --src); BENCH_SRC_REPO selects it
REPO = Path(os.environ.get("BENCH_SRC_REPO", "<USER_HOME>/src/wrf_gpu2")).resolve()
sys.path.insert(0, str(REPO / "src"))

T0_NS = time.monotonic_ns()
T0_WALL = time.time()


def _t() -> float:
    return (time.monotonic_ns() - T0_NS) / 1e9


REC: dict = {
    "schema": "wrf_gpu2.v025.s0.nested_harness.v1",
    "segments": [],
    "outputs": [],
    "advance": [],
    "force": [],
    "compile_events": [],
    "cache_events": [],
    "vram_samples": [],
    "errors": [],
}
_LOCK = threading.Lock()


def _add(key: str, item: dict) -> None:
    with _LOCK:
        REC[key].append(item)


def _install_monitoring(jax) -> None:
    def dur(event: str, duration: float, **kw) -> None:
        if any(s in event for s in ("compile", "trace", "mlir", "cache")):
            _add("compile_events", {"t": round(_t(), 4), "event": event, "s": float(duration),
                                    "fun": kw.get("fun_name")})

    def ev(event: str, **kw) -> None:
        if "cache" in event or "compil" in event:
            _add("cache_events", {"t": round(_t(), 4), "event": event})

    jax.monitoring.register_event_duration_secs_listener(dur)
    jax.monitoring.register_event_listener(ev)


PROFILE_SEGMENT = {"k": 0, "active": False}


def _cuda_profiler(fn_name: str) -> None:
    import ctypes

    for lib in ("libcudart.so.12", "libcudart.so"):
        try:
            rc = getattr(ctypes.CDLL(lib), fn_name)()
            _add("errors" if rc else "cache_events", {"t": round(_t(), 3), "event": f"{fn_name} rc={rc} via {lib}"})
            return
        except OSError:
            continue
    _add("errors", {"t": round(_t(), 3), "profiler": f"{fn_name}: libcudart not loadable"})


def _install_wrappers(jax) -> None:
    import gpuwrf.integration.nested_pipeline as npl
    import gpuwrf.runtime.domain_tree as dt

    ann = jax.profiler.TraceAnnotation

    orig_seg = npl.run_operational_domain_tree

    def seg_wrapper(*args, **kwargs):
        k = len(REC["segments"]) + 1
        if PROFILE_SEGMENT["active"] and k == PROFILE_SEGMENT["k"] + 1:
            _cuda_profiler("cudaProfilerStop")  # previous segment incl. its post-call block
            PROFILE_SEGMENT["active"] = False
        if PROFILE_SEGMENT["k"] and k == PROFILE_SEGMENT["k"]:
            _cuda_profiler("cudaProfilerStart")
            PROFILE_SEGMENT["active"] = True
        rec = {"k": k, "root_steps": int(kwargs.get("root_steps", -1)), "t_start": round(_t(), 4)}
        with ann(f"S0:segment:{k}"):
            out = orig_seg(*args, **kwargs)
        rec["t_return"] = round(_t(), 4)
        _add("segments", rec)
        return out

    npl.run_operational_domain_tree = seg_wrapper

    orig_cb = dt.run_domain_tree_callbacks

    def cb_wrapper(hierarchy, carries, *, advance, force=None, **kwargs):
        def adv(name, carry, start_step, n_steps):
            t0 = _t()
            with ann(f"S0:adv:{name}:{int(start_step)}:{int(n_steps)}"):
                out = advance(name, carry, start_step, n_steps)
            _add("advance", {"d": name, "s0": int(start_step), "n": int(n_steps),
                             "t": round(t0, 4), "host_s": round(_t() - t0, 6)})
            return out

        frc = None
        if force is not None:
            def frc(edge, parent, child):
                t0 = _t()
                with ann("S0:force"):
                    out = force(edge, parent, child)
                _add("force", {"t": round(t0, 4), "host_s": round(_t() - t0, 6)})
                return out

        return orig_cb(hierarchy, carries, advance=adv, force=frc, **kwargs)

    dt.run_domain_tree_callbacks = cb_wrapper

    writer_cls = npl._PerDomainWrfoutWriter
    orig_call = writer_cls.__call__

    def call_wrapper(self, name, own_step, carry):
        t0 = _t()
        with ann(f"S0:output:{name}:{int(own_step)}"):
            out = orig_call(self, name, own_step, carry)
        _add("outputs", {"d": name, "own_step": int(own_step), "t_start": round(t0, 4),
                         "s": round(_t() - t0, 4)})
        return out

    writer_cls.__call__ = call_wrapper


def _vram_sampler(stop: threading.Event, period: float) -> None:
    pid = str(os.getpid())
    while not stop.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            mib = None
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2 and parts[0] == pid:
                    mib = int(parts[1])
            _add("vram_samples", {"t": round(_t(), 2), "mib": mib})
        except Exception as exc:  # observation only
            _add("errors", {"t": round(_t(), 2), "vram_sampler": repr(exc)[:200]})
        stop.wait(period)


def _derive(rec: dict) -> dict:
    segs = sorted(rec["segments"], key=lambda s: s["k"])
    for i, s in enumerate(segs):
        nxt = segs[i + 1]["t_start"] if i + 1 < len(segs) else None
        s["period_to_next_s"] = None if nxt is None else round(nxt - s["t_start"], 4)
        s["call_s"] = round(s["t_return"] - s["t_start"], 4)
        s["compile_s_in_call"] = round(sum(e["s"] for e in rec["compile_events"]
                                           if "backend_compile" in e["event"]
                                           and s["t_start"] <= e["t"] <= s["t_return"]), 3)
    out = {"segments": segs}
    # rate uses the period to the next segment start (includes post-call block/finite
    # asserts); the last segment falls back to its call duration.
    rates = []
    for s in segs:
        p = s["period_to_next_s"] if s["period_to_next_s"] is not None else s["call_s"]
        steps = s.get("root_steps", 67) or 67
        rates.append(round(p * 3600.0 / (steps * 54.0), 3))
    out["segment_rate_s_per_fc_h"] = rates
    mib = [v["mib"] for v in rec["vram_samples"] if v.get("mib") is not None]
    out["vram_peak_mib"] = max(mib) if mib else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=int, default=3)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--vram-period", type=float, default=1.0)
    ap.add_argument("--no-vram", action="store_true")
    ap.add_argument("--profile-segment", type=int, default=0,
                    help="bracket root segment K (and its post-call block) with cudaProfilerStart/Stop")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    PROFILE_SEGMENT["k"] = int(args.profile_segment)
    if "XLA_PYTHON_CLIENT_ALLOCATOR" not in os.environ:
        print("s0_nested_harness: refuse -- set XLA_PYTHON_CLIENT_ALLOCATOR (production: cuda_async)",
              file=sys.stderr)
        return 2
    REC.update({
        "tag": args.tag, "argv": sys.argv, "pid": os.getpid(), "t0_wall_utc": T0_WALL,
        "env": {k: v for k, v in os.environ.items()
                if k.startswith(("GPUWRF_", "JAX_", "XLA_", "CUDA_VISIBLE", "TF_"))},
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                   text=True).stdout.strip(),
        "src_tree": subprocess.run(["git", "rev-parse", "HEAD:src/gpuwrf"], cwd=REPO,
                                   capture_output=True, text=True).stdout.strip(),
    })

    def dump(reason: str) -> None:
        with _LOCK:
            snap = json.loads(json.dumps(REC, default=str))
        snap["end_reason"] = reason
        snap["t_end"] = round(_t(), 3)
        snap["derived"] = _derive(snap)
        try:
            snap["host_vmhwm_kb"] = int([l for l in open("/proc/self/status")
                                         if l.startswith("VmHWM")][0].split()[1])
        except Exception:
            pass
        (out / "receipt.json").write_text(json.dumps(snap, indent=1, default=str))

    def on_term(signum, frame):
        dump(f"signal:{signum}")
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    import jax  # noqa: E402

    REC["jax_version"] = jax.__version__
    # hard assert the run is on the GPU (never silently benchmark a CPU fallback); record the device
    device = jax.devices()[0]
    REC["device"] = {"platform": device.platform,
                     "kind": getattr(device, "device_kind", None),
                     "id": getattr(device, "id", None),
                     "backend": jax.default_backend()}
    if device.platform != "gpu":
        print(f"s0_nested_harness: refuse -- jax.devices()[0].platform is {device.platform!r}, not 'gpu' "
              f"(device_kind={getattr(device, 'device_kind', None)!r}); this harness measures GPU runs only",
              file=sys.stderr)
        return 3
    REC["t_jax_import"] = round(_t(), 3)
    _install_monitoring(jax)
    _install_wrappers(jax)
    REC["t_wrappers"] = round(_t(), 3)

    stop = threading.Event()
    sampler = None
    if not args.no_vram and os.environ.get("JAX_PLATFORMS", "") != "cpu":
        sampler = threading.Thread(target=_vram_sampler, args=(stop, args.vram_period), daemon=True)
        sampler.start()

    from gpuwrf import cli  # noqa: E402

    cli_argv = ["run", "--input-dir", args.input_dir, "--output-dir", str(out / "wrfout"),
                "--scratch-dir", str(out / "scratch"), "--max-dom", "2", "--hours", str(args.hours)]
    REC["cli_argv"] = cli_argv
    rc = None
    try:
        with open(out / "cli_stdout.json", "w") as fh, contextlib.redirect_stdout(fh):
            rc = cli.main(cli_argv)
    except Exception as exc:
        _add("errors", {"t": round(_t(), 2), "cli": repr(exc)[:2000]})
        rc = -1
    finally:
        if PROFILE_SEGMENT["active"]:
            _cuda_profiler("cudaProfilerStop")
            PROFILE_SEGMENT["active"] = False
        stop.set()
        if sampler is not None:
            sampler.join(timeout=15)
        REC["rc"] = rc
        try:
            ms = jax.devices()[0].memory_stats() or {}
            REC["jax_memory_stats"] = {k: ms.get(k) for k in
                                       ("peak_bytes_in_use", "bytes_in_use", "bytes_limit",
                                        "largest_alloc_size", "pool_bytes", "peak_pool_bytes")}
        except Exception as exc:
            REC["jax_memory_stats"] = {"error": repr(exc)[:300]}
        dump("completed")
    print(json.dumps({"rc": rc, "receipt": str(out / "receipt.json")}))
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
