#!/usr/bin/env python3
"""v020_nsys_extract.py — launch/kernel-count + host-gap extractor for the S0 probe.

Pure-CPU post-processing of nsys CSV stats exports (produced by `nsys stats`). NO GPU,
no nsys-rep parsing of its own — it consumes the CSVs the S0 script already emits:

  *_cuda_api_sum.csv        -> cuLaunchKernel* counts (HOST launch count)
  *_cuda_gpu_kern_sum.csv   -> per-kernel GPU instances + total GPU time (the dominant
                               kernel families; the launch-bound signature)
  *_cuda_gpu_sum.csv        -> total GPU kernel time (active-union proxy)
  *_cuda_gpu_mem_time_sum.csv (optional) -> H2D/D2H time (in-loop transfer audit)

Emits a compact JSON + a human table summarising:
  * total host kernel launches and the top launch APIs
  * total GPU kernel instances + total GPU kernel time
  * the top-N kernel families by GPU time (the fusion/launch-bound targets for L5)
  * a host-bound proxy: (GPU kernel time) / (wall) if wall provided, else just totals
  * any in-step host<->device memcpy time (should be ~0; non-zero = audit flag)

This is the "launch/kernel-count extractor" deliverable of S0. It is a no-op-safe
reducer: missing CSVs are reported, never crash. CPU-dry-runnable on any prior nsys
CSV set (e.g. proofs/v018/maxdom9_speedup) which is exactly how it is validated.

Usage:
  python v020_nsys_extract.py --stats-prefix DIR/nsys_stats [--wall-s 238.9]
                              [--topn 20] [--out report.json]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path


def _find(prefix: str, suffix: str) -> str | None:
    cands = sorted(glob.glob(f"{prefix}*{suffix}*"))
    cands = [c for c in cands if c.endswith(".csv")]
    return cands[0] if cands else None


def _read_csv(path: str | None) -> list[dict]:
    if not path or not Path(path).is_file():
        return []
    with open(path, newline="") as fh:
        # nsys CSVs sometimes have a comment/blank preamble; sniff the header row.
        rows = list(csv.reader(fh))
    if not rows:
        return []
    # find the header row (first row containing a known column token)
    hdr_idx = 0
    for i, r in enumerate(rows[:5]):
        joined = ",".join(r).lower()
        if "time" in joined or "name" in joined or "instances" in joined or "count" in joined:
            hdr_idx = i
            break
    header = [h.strip() for h in rows[hdr_idx]]
    out = []
    for r in rows[hdr_idx + 1:]:
        if not any(c.strip() for c in r):
            continue
        out.append({header[i]: (r[i].strip() if i < len(r) else "") for i in range(len(header))})
    return out


def _num(s: str) -> float:
    try:
        return float(str(s).replace(",", "").replace("%", "").strip())
    except (ValueError, AttributeError):
        return 0.0


def _col(rows: list[dict], *cands: str) -> str | None:
    if not rows:
        return None
    keys = list(rows[0].keys())
    low = {k.lower(): k for k in keys}
    for c in cands:
        for lk, k in low.items():
            if c in lk:
                return k
    return None


def extract(prefix: str, wall_s: float | None, topn: int) -> dict:
    out: dict = {"stats_prefix": prefix, "sources": {}}

    # --- host launch APIs ---------------------------------------------------
    api_path = _find(prefix, "cuda_api_sum")
    api = _read_csv(api_path)
    out["sources"]["cuda_api_sum"] = api_path
    launch_total = 0
    launch_apis = []
    if api:
        name_c = _col(api, "name")
        cnt_c = _col(api, "num calls", "count", "instances", "calls")
        time_c = _col(api, "total time", "time")
        for r in api:
            nm = r.get(name_c, "") if name_c else ""
            cnt = int(_num(r.get(cnt_c, "0"))) if cnt_c else 0
            if "Launch" in nm or "launch" in nm:
                launch_total += cnt
                launch_apis.append({"api": nm, "calls": cnt,
                                    "time_ns": _num(r.get(time_c, "0")) if time_c else None})
        launch_apis.sort(key=lambda d: -d["calls"])
    out["host_launches"] = {"total_launch_calls": launch_total,
                            "by_api": launch_apis[:topn]}

    # --- GPU kernels --------------------------------------------------------
    kern_path = _find(prefix, "cuda_gpu_kern_sum")
    kern = _read_csv(kern_path)
    out["sources"]["cuda_gpu_kern_sum"] = kern_path
    kern_inst = 0
    kern_time = 0.0
    top_kernels = []
    if kern:
        name_c = _col(kern, "name")
        inst_c = _col(kern, "instances", "count", "num")
        time_c = _col(kern, "total time", "time")
        for r in kern:
            inst = int(_num(r.get(inst_c, "0"))) if inst_c else 0
            t = _num(r.get(time_c, "0")) if time_c else 0.0
            kern_inst += inst
            kern_time += t
            top_kernels.append({"kernel": (r.get(name_c, "") if name_c else "")[:90],
                                "instances": inst, "gpu_time_ns": t})
        top_kernels.sort(key=lambda d: -d["gpu_time_ns"])
    out["gpu_kernels"] = {"total_instances": kern_inst,
                          "total_gpu_kernel_time_ns": kern_time,
                          "top_by_time": top_kernels[:topn]}

    # --- total GPU time (active union proxy) --------------------------------
    gsum_path = _find(prefix, "cuda_gpu_sum")
    out["sources"]["cuda_gpu_sum"] = gsum_path

    # --- memcpy (in-loop transfer audit) ------------------------------------
    mem_path = _find(prefix, "cuda_gpu_mem_time_sum")
    mem = _read_csv(mem_path)
    out["sources"]["cuda_gpu_mem_time_sum"] = mem_path
    mem_time = 0.0
    mem_rows = []
    if mem:
        name_c = _col(mem, "operation", "name")
        time_c = _col(mem, "total time", "time")
        cnt_c = _col(mem, "count", "instances", "num")
        for r in mem:
            t = _num(r.get(time_c, "0")) if time_c else 0.0
            mem_time += t
            mem_rows.append({"op": r.get(name_c, "") if name_c else "",
                             "time_ns": t,
                             "count": int(_num(r.get(cnt_c, "0"))) if cnt_c else None})
    out["host_device_transfer"] = {"total_memcpy_time_ns": mem_time, "by_op": mem_rows,
                                   "audit_note": "in-timestep H2D/D2H should be ~0; "
                                                 "non-trivial copies are an audit flag"}

    # --- host-bound proxy ---------------------------------------------------
    derived = {"wall_s": wall_s}
    if wall_s and kern_time > 0:
        gpu_active_s = kern_time / 1e9
        derived["gpu_kernel_active_s"] = gpu_active_s
        derived["gpu_active_fraction_of_wall"] = gpu_active_s / wall_s
        derived["host_or_idle_fraction_of_wall"] = max(0.0, 1.0 - gpu_active_s / wall_s)
        derived["note"] = ("host_or_idle_fraction approximates the launch/sync headroom "
                           "the cheap point-3 levers (allocator, async-sync, command-buffer) "
                           "can recover. NOTE this is a kernel-time/wall proxy, not a true "
                           "GPU-busy union; confirm with nsys gpu-util sampling.")
    out["derived"] = derived
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats-prefix", required=True,
                    help="prefix passed to `nsys stats --output` (CSV files share it)")
    ap.add_argument("--wall-s", type=float, default=None,
                    help="warm forecast wall seconds (enables the host-bound proxy)")
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    rep = extract(args.stats_prefix, args.wall_s, args.topn)
    payload = json.dumps(rep, indent=2) + "\n"
    if args.out:
        args.out.write_text(payload)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(payload)

    # human summary
    hl = rep["host_launches"]
    gk = rep["gpu_kernels"]
    print("\n=== nsys S0 summary ===", file=sys.stderr)
    print(f"  host kernel launches: {hl['total_launch_calls']}", file=sys.stderr)
    print(f"  GPU kernel instances: {gk['total_instances']}  "
          f"total GPU kernel time: {gk['total_gpu_kernel_time_ns']/1e9:.3f} s", file=sys.stderr)
    d = rep["derived"]
    if d.get("gpu_active_fraction_of_wall") is not None:
        print(f"  GPU-active/wall ~ {d['gpu_active_fraction_of_wall']*100:.1f}%  "
              f"(host-or-idle ~ {d['host_or_idle_fraction_of_wall']*100:.1f}% -> "
              f"cheap-host-lever headroom)", file=sys.stderr)
    print("  top kernels by GPU time:", file=sys.stderr)
    for k in gk["top_by_time"][:8]:
        print(f"    {k['gpu_time_ns']/1e9:8.3f}s  x{k['instances']:<7d} {k['kernel']}",
              file=sys.stderr)
    mt = rep["host_device_transfer"]["total_memcpy_time_ns"]
    print(f"  in-loop memcpy time: {mt/1e9:.4f} s  (audit: want ~0)", file=sys.stderr)
    # exit non-zero only if NOTHING parsed (so dry-run on a real CSV set passes)
    parsed_any = (hl["total_launch_calls"] > 0 or gk["total_instances"] > 0)
    return 0 if parsed_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
