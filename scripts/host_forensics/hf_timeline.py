"""HOST-TIMELINE FORENSICS (2026-09-18, worker/glm/hostforensic).

Measures the HOST side of the 77.4% launch gap on the warm FAST loop
(matched-dt54, 67 steps, segment_steps=34 -- the fused_ab_diag.py A/B pattern).

Stages
------
timeline   warm instrumented reps: per-segment dispatch vs block_until_ready,
           once-per-run carry/clock costs.  -> timeline.json
cprofile   cProfile of one warm run; pstats + ranked self-time table split by
           call count (once-per-run / per-segment).  -> cprofile.pstats, cprofile_top.txt
xlatrace   jax.profiler chrome trace of ONE warm segment; parsed into
           host-busy / device-busy / launch-count table.  -> xlatrace.json + raw dir
loopchild  build case, one warm-up, then loop warm runs until STOP file appears
           (py-spy target).  Writes READY when warm.

All stages take the SAME code path as the production entry
(run_forecast_operational_segmented) with timing wrappers patched around
module-level helpers; the wrappers call the originals unchanged, so numbers are
attributable but the computation is untouched.
"""
import argparse
import cProfile
import dataclasses
import json
import os
import pstats
import statistics
import sys
import time
from pathlib import Path

WORKTREE_SRC = "<USER_HOME>/src/wrf_gpu2_wt/hostforensic/src"
ART = Path("<USER_HOME>/src/wrf_gpu2/.agent/sprints/2026-09-18-v0250-host-forensics/artifacts")
sys.path.insert(0, WORKTREE_SRC)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gpuwrf.integration import daily_pipeline as daily  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402

RUN_DIR = Path("<DATA_ROOT>/wrf_gpu2/v025/m0/cpu_arms/fastbind_r1")
SEG = 34
STEPS = 67
DT = 54.0


def build_case():
    config = daily.DailyPipelineConfig(run_id=str(RUN_DIR), run_root=RUN_DIR.parent, hours=1, domain="d01")
    case, resolved = daily._build_real_case(config)
    state = case.state
    capture = daily._capture_boundary_leaves(state, case.namelist)
    cadence = daily._boundary_window_cadence_s(case.namelist)
    record_cadence = float((case.metadata.get("boundary") or {}).get("interval_seconds") or cadence)
    if capture:
        state = daily._rewindow_boundary_leaves(state, capture, segment_start_s=0.0,
                                                record_cadence_s=record_cadence, window_s=cadence)
    nl = dataclasses.replace(case.namelist, dt_s=DT)
    hours = STEPS * DT / 3600.0
    return state, nl, hours


class Timer:
    """Wall-clock accumulator for patched callables."""

    def __init__(self):
        self.calls = 0
        self.total = 0.0
        self.events = []  # (t, dur, tag)

    def wrap(self, fn, tag):
        def wrapper(*a, **kw):
            t0 = time.perf_counter()
            r = fn(*a, **kw)
            dur = time.perf_counter() - t0
            self.calls += 1
            self.total += dur
            self.events.append((t0, dur, tag))
            return r
        return wrapper


def patch_helpers():
    """Patch om helpers + jax.block_until_ready with timers; return restore fn."""
    timers = {
        "advance_chunk": Timer(),          # per-segment dispatch (host side incl. XLA launch)
        "block_theta": Timer(),            # per-segment device drain/sync
        "initial_carry": Timer(),          # once per run
        "clock_base": Timer(),             # once per run
    }
    orig = {
        "advance_chunk": om._advance_chunk,
        "initial_carry": om._committed_initial_carry_for_run,
        "clock_base": om.build_clock_base,
        "block": jax.block_until_ready,
    }
    om._advance_chunk = timers["advance_chunk"].wrap(orig["advance_chunk"], "advance_chunk")
    om._committed_initial_carry_for_run = timers["initial_carry"].wrap(orig["initial_carry"], "initial_carry")
    om.build_clock_base = timers["clock_base"].wrap(orig["clock_base"], "clock_base")
    jax.block_until_ready = timers["block_theta"].wrap(orig["block"], "block_theta")

    def restore():
        om._advance_chunk = orig["advance_chunk"]
        om._committed_initial_carry_for_run = orig["initial_carry"]
        om.build_clock_base = orig["clock_base"]
        jax.block_until_ready = orig["block"]
    return timers, restore


def summarize_timers(timers):
    out = {}
    for name, t in timers.items():
        out[name] = {
            "calls": t.calls,
            "total_s": t.total,
            "events": [{"t0": e0, "dur": d} for e0, d, _ in t.events],
        }
    return out


def run_segmented(state, nl, hours):
    return om.run_forecast_operational_segmented(state, nl, hours, segment_steps=SEG)


def stage_timeline(state, nl, hours, reps=3):
    ART.mkdir(parents=True, exist_ok=True)
    # Warm-up (compile / autotune / cache load excluded from medians).
    t0 = time.perf_counter()
    out = run_segmented(state, nl, hours)
    jax.block_until_ready(out.theta)
    warmup_s = time.perf_counter() - t0

    records = []
    for r in range(reps):
        timers, restore = patch_helpers()
        t0 = time.perf_counter()
        out = run_segmented(state, nl, hours)
        jax.block_until_ready(out.theta)
        wall = time.perf_counter() - t0
        restore()
        rec = {"rep": r, "wall_s": wall, "timers": summarize_timers(timers)}
        ac = rec["timers"]["advance_chunk"]
        bt = rec["timers"]["block_theta"]
        rec["dispatch_sum_s"] = ac["total_s"]
        rec["block_sum_s"] = bt["total_s"]
        rec["per_run_host_s"] = (rec["timers"]["initial_carry"]["total_s"]
                                 + rec["timers"]["clock_base"]["total_s"])
        rec["unaccounted_s"] = wall - ac["total_s"] - bt["total_s"] - rec["per_run_host_s"]
        records.append(rec)
        print(f"rep{r}: wall={wall:.3f}s dispatch={ac['total_s']:.3f}s "
              f"block={bt['total_s']:.3f}s per_run_host={rec['per_run_host_s']:.4f}s "
              f"unaccounted={rec['unaccounted_s']:.3f}s", flush=True)

    med = statistics.median(r["wall_s"] for r in records)
    res = {
        "warmup_s": warmup_s,
        "reps": records,
        "wall_median_s": med,
        "wall_step_ms": 1000.0 * med / STEPS,
        "dispatch_median_s": statistics.median(r["dispatch_sum_s"] for r in records),
        "block_median_s": statistics.median(r["block_sum_s"] for r in records),
        "env": {
            "GPUWRF_FUSED_VERTICAL": os.environ.get("GPUWRF_FUSED_VERTICAL"),
            "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
            "GPUWRF_ADVANCE_CHUNK_LOOP": os.environ.get("GPUWRF_ADVANCE_CHUNK_LOOP"),
            "jax": jax.__version__,
        },
    }
    (ART / "timeline.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "reps"}, indent=1), flush=True)


def stage_cprofile(state, nl, hours):
    ART.mkdir(parents=True, exist_ok=True)
    out = run_segmented(state, nl, hours)
    jax.block_until_ready(out.theta)
    prof = cProfile.Profile()
    prof.enable()
    out = run_segmented(state, nl, hours)
    jax.block_until_ready(out.theta)
    prof.disable()
    prof.dump_stats(str(ART / "cprofile.pstats"))
    st = pstats.Stats(prof)
    st.sort_stats("tottime")
    import io
    buf = io.StringIO()
    st.stream = buf
    st.print_stats(60)
    (ART / "cprofile_top.txt").write_text(buf.getvalue())
    print(buf.getvalue(), flush=True)


def stage_xlatrace(state, nl, hours):
    from jax.profiler import trace
    ART.mkdir(parents=True, exist_ok=True)
    out = run_segmented(state, nl, hours)
    jax.block_until_ready(out.theta)
    trdir = ART / "xlatrace"
    with trace(str(trdir)):
        carry = om._committed_initial_carry_for_run(state, nl)
        cb = om.build_clock_base(nl)
        carry = om._advance_chunk(carry, nl, jnp.asarray(1, dtype=jnp.int32), cb,
                                  n_steps=SEG, cadence=int(nl.radiation_cadence_steps))
        jax.block_until_ready(carry.state.theta)
    parse_chrome_trace(trdir)


def parse_chrome_trace(trdir: Path):
    import gzip
    cands = list(trdir.rglob("trace.json.gz")) + list(trdir.rglob("trace.json"))
    if not cands:
        print("NO TRACE FILE FOUND", flush=True)
        return
    tf = cands[0]
    data = json.loads(gzip.open(tf, "rt").read() if tf.suffix == ".gz" else tf.read_text())
    events = data.get("traceEvents", [])
    kernels = []
    host_by_cat = {}
    host_by_name = {}
    threads = {}
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        name = e.get("name", "")
        dur = float(e.get("dur", 0.0))
        ts = float(e.get("ts", 0.0))
        tid = e.get("tid")
        if cat == "Kernel":
            kernels.append((ts, dur, name))
        else:
            host_by_cat[cat] = host_by_cat.get(cat, 0.0) + dur
            host_by_name[name] = host_by_name.get(name, (0, 0.0))
            n, d = host_by_name[name]
            host_by_name[name] = (n + 1, d + dur)
            threads.setdefault(str(tid), [0, 0.0])
            threads[str(tid)][0] += 1
            threads[str(tid)][1] += dur
    tmin = min([k[0] for k in kernels] + [e.get("ts", 0) for e in events if e.get("ph") == "X"] or [0])
    tmax = max([(k[0] + k[1]) for k in kernels] +
               [e.get("ts", 0) + e.get("dur", 0) for e in events if e.get("ph") == "X"] or [0])
    wall_us = tmax - tmin
    dev_busy_us = sum(d for _, d, _ in kernels)
    # union of kernel intervals per device (all kernels share stream rows here;
    # union across rows avoids double-count)
    iv = sorted((ts, ts + d) for ts, d, _ in kernels)
    union = 0.0
    cs, ce = None, None
    for s, e2 in iv:
        if cs is None:
            cs, ce = s, e2
        elif s <= ce:
            ce = max(ce, e2)
        else:
            union += ce - cs
            cs, ce = s, e2
    if cs is not None:
        union += ce - cs
    res = {
        "trace_file": str(tf),
        "wall_us": wall_us,
        "wall_s": wall_us / 1e6,
        "kernel_launches": len(kernels),
        "device_busy_union_us": union,
        "device_busy_frac_of_wall": union / wall_us if wall_us else None,
        "avg_kernel_us": (sum(d for _, d, _ in kernels) / len(kernels)) if kernels else None,
        "host_by_cat_us": dict(sorted(host_by_cat.items(), key=lambda kv: -kv[1])[:20]),
        "threads": {k: {"events": v[0], "sum_us": v[1]} for k, v in
                    sorted(threads.items(), key=lambda kv: -kv[1][1])[:12]},
        "top_host_ops": [
            {"name": n[:160], "count": c, "sum_us": d}
            for n, (c, d) in sorted(host_by_name.items(), key=lambda kv: -kv[1][1])[:30]
        ],
    }
    (ART / "xlatrace_summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:4000], flush=True)


def stage_loopchild(state, nl, hours):
    ready = Path("/tmp/hf_pyspy_ready")
    stop = Path("/tmp/hf_pyspy_stop")
    ready.write_text(str(os.getpid()))
    stop.unlink(missing_ok=True)
    n = 0
    while not stop.exists():
        out = run_segmented(state, nl, hours)
        jax.block_until_ready(out.theta)
        n += 1
        print(f"WARM RUN {n} done {time.perf_counter():.1f}", flush=True)
    ready.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="timeline",
                    choices=["timeline", "cprofile", "xlatrace", "loopchild"])
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    t0 = time.perf_counter()
    state, nl, hours = build_case()
    print(f"case built in {time.perf_counter() - t0:.1f}s "
          f"(steps={om._steps_for_hours(hours, DT)})", flush=True)
    if args.stage == "timeline":
        stage_timeline(state, nl, hours, reps=args.reps)
    elif args.stage == "cprofile":
        stage_cprofile(state, nl, hours)
    elif args.stage == "xlatrace":
        stage_xlatrace(state, nl, hours)
    elif args.stage == "loopchild":
        stage_loopchild(state, nl, hours)


if __name__ == "__main__":
    main()
