#!/usr/bin/env python
"""v0.15 kernel probe — parse the perfetto trace of one steady forecast hour.

Computes: GPU busy vs span (launch/gap-bound signature), kernel launches/step,
per-kernel-name aggregation (count, total ms, mean us), memcpy totals, and a
phase attribution via HLO metadata where available.

Usage: python analyze_trace.py <trace.json.gz> [--peek] [--steps 200]
Artifact: proofs/perf/v015/trace_kernel_summary.json
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(path: Path):
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt") as f:
        return json.load(f)


def main() -> int:
    path = Path(sys.argv[1])
    peek = "--peek" in sys.argv
    steps = 200
    if "--steps" in sys.argv:
        steps = int(sys.argv[sys.argv.index("--steps") + 1])
    data = load(path)
    events = data["traceEvents"] if isinstance(data, dict) else data

    pid_names: dict[int, str] = {}
    tid_names: dict[tuple[int, int], str] = {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            pid_names[e["pid"]] = e["args"].get("name", "")
        if e.get("ph") == "M" and e.get("name") == "thread_name":
            tid_names[(e["pid"], e["tid"])] = e["args"].get("name", "")

    gpu_pids = {p for p, n in pid_names.items() if "GPU" in n.upper() or "DEVICE" in n.upper()}

    if peek:
        print("processes:", json.dumps(pid_names, indent=2)[:2000])
        print("threads sample:", json.dumps({f"{k}": v for k, v in list(tid_names.items())[:40]}, indent=2))
        shown = 0
        for e in events:
            if e.get("ph") == "X" and e.get("pid") in gpu_pids and shown < 6:
                print(json.dumps(e)[:600])
                shown += 1
        return 0

    # device complete events
    kern = defaultdict(lambda: [0, 0.0])  # name -> [count, total_us]
    intervals = []
    memcpy_us = 0.0
    memcpy_n = 0
    total_n = 0
    for e in events:
        if e.get("ph") != "X" or e.get("pid") not in gpu_pids:
            continue
        name = e.get("name", "?")
        tname = tid_names.get((e["pid"], e["tid"]), "")
        dur = float(e.get("dur", 0.0))
        ts = float(e.get("ts", 0.0))
        total_n += 1
        kern[name][0] += 1
        kern[name][1] += dur
        intervals.append((ts, ts + dur))
        low = name.lower()
        if "memcpy" in low or "memset" in low or "copy" in low.split(".")[0]:
            memcpy_us += dur
            memcpy_n += 1

    intervals.sort()
    busy = 0.0
    cur_s, cur_e = None, None
    for s, e_ in intervals:
        if cur_s is None:
            cur_s, cur_e = s, e_
        elif s <= cur_e:
            cur_e = max(cur_e, e_)
        else:
            busy += cur_e - cur_s
            cur_s, cur_e = s, e_
    if cur_s is not None:
        busy += cur_e - cur_s
    span = intervals[-1][1] - intervals[0][0] if intervals else 0.0

    top = sorted(kern.items(), key=lambda kv: -kv[1][1])[:80]
    out = {
        "schema": "V015TraceKernelSummary",
        "trace": str(path),
        "steps_assumed": steps,
        "device_event_count": total_n,
        "launches_per_step": round(total_n / steps, 1),
        "span_ms": round(span / 1000.0, 2),
        "busy_ms": round(busy / 1000.0, 2),
        "busy_fraction_of_span": round(busy / span, 4) if span else None,
        "gap_ms": round((span - busy) / 1000.0, 2),
        "memcpy": {"count": memcpy_n, "total_ms": round(memcpy_us / 1000.0, 2)},
        "distinct_kernel_names": len(kern),
        "top_kernels": [
            {
                "name": n[:120],
                "count": c,
                "total_ms": round(t / 1000.0, 2),
                "mean_us": round(t / c, 1),
                "pct_busy": round(t / busy * 100.0, 2) if busy else None,
            }
            for n, (c, t) in top
        ],
    }
    out_path = HERE / "trace_kernel_summary.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    hdr = {k: v for k, v in out.items() if k != "top_kernels"}
    print(json.dumps(hdr, indent=2))
    for row in out["top_kernels"][:30]:
        print(f"{row['total_ms']:>10.2f} ms  n={row['count']:>7}  {row['mean_us']:>8.1f} us  {row['pct_busy']:>5.2f}%  {row['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
