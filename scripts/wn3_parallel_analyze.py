"""W4 parallel-capacity analysis for one arm dir (CPU only). Usage: analyze.py ARM_DIR [solo_warm_s_per_h]

Throughput definitions (per case-sim-hour, lower is better):
  whole_run = arm wall (first start -> last exit) / (N cases x hours)  [incl. init/load/output, 3 h amortizes poorly]
  stepping  = 1 / sum_i(1 / r_i), r_i = case i's mean warm-segment rate (s per sim-hour) while all N ran
CPU reference (W1, clean 0227/0502, 4 ranks/case, 3 cases on 12 cores): 384.70 / 355.61 s per case-hour each
  -> 3-parallel throughput 123.385 s per case-sim-hour [I: mean/3].
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

CPU_THROUGHPUT = (384.70172625 + 355.60807708) / 2 / 3
CPU_SINGLE = (384.70172625 + 355.60807708) / 2


def dmon(path: Path) -> dict:
    cols, rows = None, []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            if cols is None and "gpu" in line:
                cols = line.lstrip("#").split()
            continue
        parts = line.split()
        if cols and len(parts) == len(cols):
            rows.append(dict(zip(cols, parts)))
    def col(name):
        vals = [float(r[name]) for r in rows if r.get(name, "-") not in ("-", "")]
        return np.array(vals) if vals else np.array([np.nan])
    fb, sm = col("fb"), col("sm")
    return dict(samples=len(rows), fb_peak_mib=float(np.nanmax(fb)), sm_mean=float(np.nanmean(sm)),
                sm_p50=float(np.nanmedian(sm)), sm_busy_frac=float(np.mean(sm > 0)) if sm.size else None)


def main():
    arm = Path(sys.argv[1])
    solo = float(sys.argv[2]) if len(sys.argv) > 2 else None
    start = json.loads((arm / "start.json").read_text())
    end = json.loads((arm / "end.json").read_text())
    hours = int(start["hours"])
    cases = start["cases"].split()
    per = {}
    for c in cases:
        r = json.loads((arm / c / "receipt.json").read_text())
        seg = r["derived"]["segments"]
        rates = r["derived"]["segment_rate_s_per_fc_h"]
        warm = [rates[i] for i, s in enumerate(seg) if not s.get("compile_s_in_call") and i > 0]
        per[c] = dict(rc=r.get("rc"), wall_s=r.get("t_end"), pre_step_s=seg[0]["t_start"] if seg else None,
                      compile_s=sum(s.get("compile_s_in_call") or 0 for s in seg), segment_rates=rates,
                      warm_mean_s_per_h=float(np.mean(warm)) if warm else None,
                      vram_peak_mib=r["derived"].get("vram_peak_mib"),
                      jax_peak_gb=((r.get("jax_memory_stats") or {}).get("peak_bytes_in_use") or 0) / 1e9,
                      host_vmhwm_gb=(r.get("host_vmhwm_kb") or 0) / 1e6)
    n = len(cases)
    wall = float(end["wall_s"])
    warm = [v["warm_mean_s_per_h"] for v in per.values() if v["warm_mean_s_per_h"]]
    stepping = 1.0 / sum(1.0 / w for w in warm) if len(warm) == n else None
    mem = [l.split() for l in (arm / "hostmem.log").read_text().splitlines() if l[:1].isdigit()] if (arm / "hostmem.log").exists() else []
    out = dict(arm=str(arm), n_cases=n, hours=hours, cpus=start.get("cpus"), rcs=end.get("rcs"), arm_wall_s=wall,
               per_case=per,
               throughput_whole_run_s_per_case_h=wall / (n * hours),
               throughput_stepping_s_per_case_h=stepping,
               vram_sum_of_pid_peaks_mib=sum(v["vram_peak_mib"] or 0 for v in per.values()),
               dmon=dmon(arm / "dmon.log") if (arm / "dmon.log").exists() else None,
               host_mem=dict(min_avail_gb=min(int(m[1]) for m in mem) / 2**20, max_own_rss_gb=max(int(m[2]) for m in mem) / 2**20,
                             watchdog_fired=any("WATCHDOG" in l for l in (arm / "hostmem.log").read_text().splitlines())) if mem else None,
               cpu_reference=dict(single_case_4core_s_per_h=CPU_SINGLE, three_parallel_12core_s_per_case_h=CPU_THROUGHPUT),
               solo_gpu_warm_s_per_h=solo)
    out["speedup_vs_cpu_throughput_stepping"] = CPU_THROUGHPUT / stepping if stepping else None
    out["speedup_vs_cpu_throughput_whole_run"] = CPU_THROUGHPUT / out["throughput_whole_run_s_per_case_h"]
    if solo and stepping:
        out["parallel_gain_vs_solo"] = solo / stepping
    (arm / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("n_cases", "rcs", "arm_wall_s", "throughput_whole_run_s_per_case_h",
                                           "throughput_stepping_s_per_case_h", "speedup_vs_cpu_throughput_stepping",
                                           "speedup_vs_cpu_throughput_whole_run", "vram_sum_of_pid_peaks_mib", "dmon", "host_mem")}
                     | {"warm_per_case": {c: v["warm_mean_s_per_h"] for c, v in per.items()},
                        "compile_per_case": {c: v["compile_s"] for c, v in per.items()}}, indent=1))


if __name__ == "__main__":
    main()
