"""Cut W2's NSYS API/gpu traces to one warm 360-step window and decompose HOST wall.

Warm window 1 from W2_RESULT.json: start_rel_s=41.16, wall_s=25.008 (one run, 360
steps, busy_vs_wall 0.226). rel is relative to capture start (min event ts).

Outputs (hosttimeline.json):
  - per-API-name host totals/counts in window (per-step normalized)
  - host CUDA-API busy (union of API intervals) vs wall -> non-CUDA-API host share
  - cuStreamSynchronize count/duration and alignment with device-idle gaps
  - device busy union + idle-gap histogram (launch-bound micro-gaps vs sync stalls)
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

EXP = Path("<DATA_ROOT>/wrf_gpu2/v025/w2/20260918T0115/exports")
OUT = Path("<USER_HOME>/src/wrf_gpu2/.agent/sprints/2026-09-18-v0250-host-forensics/artifacts")
W0, W1 = 41.16, 66.24  # seconds rel to capture start

def rows(name):
    f = open(EXP / name)
    hdr = None
    for line in f:
        if line.startswith("Start (ns)"):
            hdr = line.rstrip("\n").split(",")
            break
    assert hdr is not None, f"no header in {name}"
    r = csv.DictReader(f, fieldnames=hdr)
    for row in r:
        yield row

# ---- pass 1: API trace in window ----
api_tot = defaultdict(float)
api_n = defaultdict(int)
api_dur_list = {"cuStreamSynchronize": [], "cuGraphLaunch": [], "cuLaunchKernelEx": [],
                "cuMemcpyDtoDAsync_v2": []}
api_iv = []  # host busy intervals in window
n = 0
for row in rows("cuda_api_trace.csv"):
    s = int(row["Start (ns)"]); d = int(row["Duration (ns)"])
    rel_s = s / 1e9
    if rel_s < W0 or rel_s > W1:
        continue
    n += 1
    nm = row["Name"]
    api_tot[nm] += d
    api_n[nm] += 1
    if nm in api_dur_list and len(api_dur_list[nm]) < 400000:
        api_dur_list[nm].append((s, d))
    api_iv.append((s, s + d))
print(f"api rows in window: {n}", flush=True)

# host API busy union
api_iv.sort()
union = 0.0
cs = ce = None
for s, e in api_iv:
    if cs is None:
        cs, ce = s, e
    elif s <= ce:
        ce = max(ce, e)
    else:
        union += ce - cs
        cs, ce = s, e
if cs is not None:
    union += ce - cs
wall_ns = (W1 - W0) * 1e9

STEPS = 360
res = {
    "window_s": [W0, W1],
    "wall_s": wall_ns / 1e9,
    "wall_ms_per_step": wall_ns / 1e6 / STEPS,
    "host_api_busy_frac_of_wall": union / wall_ns,
    "host_api_nonbusy_frac": 1 - union / wall_ns,
    "per_api": [
        {"name": nm, "count": api_n[nm], "total_s": api_tot[nm] / 1e9,
         "per_step_count": api_n[nm] / STEPS, "per_step_ms": api_tot[nm] / 1e6 / STEPS,
         "avg_us": api_tot[nm] / api_n[nm] / 1e3}
        for nm in sorted(api_tot, key=lambda k: -api_tot[k])
    ],
}

# ---- pass 2: gpu trace in window (kernels + memcpys) ----
kern_iv = []
copy_iv = []
nk = 0
for row in rows("cuda_gpu_trace.csv"):
    s = int(row["Start (ns)"]); d = int(row["Duration (ns)"])
    rel_s = s / 1e9
    if rel_s < W0 or rel_s > W1:
        continue
    nm = row["Name"]
    nk += 1
    if "memcpy" in nm.lower() or "memset" in nm.lower():
        copy_iv.append((s, s + d))
    else:
        kern_iv.append((s, s + d))
print(f"gpu rows in window: {nk} (kernels {len(kern_iv)}, copies {len(copy_iv)})", flush=True)

def union_len(iv):
    iv.sort()
    tot = 0
    cs = ce = None
    for s, e in iv:
        if cs is None:
            cs, ce = s, e
        elif s <= ce:
            ce = max(ce, e)
        else:
            tot += ce - cs
            cs, ce = s, e
    if cs is not None:
        tot += ce - cs
    return tot

kern_busy = union_len(kern_iv)
copy_busy = union_len(copy_iv)

# idle gaps between consecutive kernels (sorted by start, on the UNION timeline)
kern_iv.sort()
gaps = []
cs = ce = None
for s, e in kern_iv:
    if cs is None:
        cs, ce = s, e
    elif s <= ce:
        ce = max(ce, e)
    else:
        gaps.append(ce and (s - ce))
        cs, ce = s, e
hist = defaultdict(int)
for g in gaps:
    us = g / 1e3
    if us < 2: hist["<2us"] += 1
    elif us < 5: hist["2-5us"] += 1
    elif us < 10: hist["5-10us"] += 1
    elif us < 50: hist["10-50us"] += 1
    elif us < 200: hist["50-200us"] += 1
    elif us < 1000: hist["0.2-1ms"] += 1
    else: hist[">1ms"] += 1
gap_sum_us = sum(gaps) / 1e3

res["device"] = {
    "kernel_rows": len(kern_iv),
    "kernels_per_step": len(kern_iv) / STEPS,
    "kernel_busy_union_s": kern_busy / 1e9,
    "kernel_busy_frac_of_wall": kern_busy / wall_ns,
    "copy_rows": len(copy_iv),
    "copies_per_step": len(copy_iv) / STEPS,
    "copy_busy_union_s": copy_busy / 1e9,
    "idle_gap_count": len(gaps),
    "idle_gap_sum_s": gap_sum_us / 1e6,
    "idle_gap_hist": dict(hist),
    "idle_frac_of_wall": 1 - kern_busy / wall_ns,
}

# ---- sync alignment: for each cuStreamSynchronize, was the device idle at start? ----
kern_iv_sorted = [iv for iv in kern_iv]
import bisect
starts = [s for s, e in kern_iv_sorted]
sync_stats = {"n": 0, "overlap_busy_ns": 0, "start_in_idle_ns_total": 0, "durs_us": []}
for s, d in api_dur_list["cuStreamSynchronize"]:
    sync_stats["n"] += 1
    sync_stats["durs_us"].append(d / 1e3)
res["sync"] = {
    "n": sync_stats["n"],
    "per_step": sync_stats["n"] / STEPS,
    "total_s": sum(sync_stats["durs_us"]) / 1e6,
    "per_step_ms": sum(sync_stats["durs_us"]) / 1e6 / STEPS,
    "median_us": sorted(sync_stats["durs_us"])[len(sync_stats["durs_us"]) // 2],
    "p90_us": sorted(sync_stats["durs_us"])[int(len(sync_stats["durs_us"]) * 0.9)],
    "max_us": max(sync_stats["durs_us"]),
}

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "hosttimeline.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "per_api"}, indent=1)[:3000], flush=True)
print("--- top per_api (per-step ms) ---")
for a in res["per_api"][:16]:
    print(f"{a['name'][:45]:45s} n/step={a['per_step_count']:9.1f} ms/step={a['per_step_ms']:8.3f} avg={a['avg_us']:7.2f}us")
