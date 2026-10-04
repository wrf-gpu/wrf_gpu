#!/usr/bin/env python3
"""R5: CPU-WRF energy per simulated hour WITHOUT a new CPU run — sample RAPL while ALISIOS's wrf.exe ranks run.

Needs one of (granted by the user, root once):
  sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj        -> package energy [M]
  sudo sysctl -w kernel.perf_event_paranoid=0                       -> + per-core energy via perf power_core [M]
Records every second: package energy, per-core energy (if perf allowed), /proc/stat busy jiffies per CPU, and the
wrf.exe processes with their CPU (psr) + run dir (cwd). Attribution:
  per-core [M]: energy of the cores that ran wrf.exe ranks (+ their SMT siblings);
  package share [I]: package energy x (wrf.exe busy core-time / all busy core-time).
CPU J per simulated hour = attributed W x (wall s per simulated hour of those runs, from their rsl timing).

  python scripts/release_plots/rapl_probe.py --seconds 900 --out <USER_HOME>/wrf_gpu2_lanes/release-docs/R5/rapl.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import time

PKG = pathlib.Path("/sys/class/powercap/intel-rapl:0")


def read_pkg_uj():
    try:
        return int((PKG / "energy_uj").read_text())
    except (PermissionError, FileNotFoundError):
        return None


def proc_stat():
    out = {}
    for line in pathlib.Path("/proc/stat").read_text().splitlines():
        if line.startswith("cpu") and line[3:4].isdigit():
            f = line.split()
            vals = list(map(int, f[1:]))
            idle = vals[3] + vals[4]
            out[int(f[0][3:])] = (sum(vals) - idle, sum(vals))
    return out


def wrf_ranks():
    ranks = []
    for p in pathlib.Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            if (p / "comm").read_text().strip() != "wrf.exe":
                continue
            st = (p / "stat").read_text().rsplit(")", 1)[1].split()
            ranks.append({"pid": int(p.name), "psr": int(st[36]), "utime": int(st[11]), "stime": int(st[12]),
                          "cwd": os.readlink(p / "cwd")})
        except (FileNotFoundError, PermissionError, ProcessLookupError, IndexError):
            continue
    return ranks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=900)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    perf = None
    if int(pathlib.Path("/proc/sys/kernel/perf_event_paranoid").read_text()) <= 0:
        perf = subprocess.Popen(["perf", "stat", "-a", "-A", "-x", ",", "-I", "1000", "-e", "power/energy-pkg/",
                                 "-e", "power_core/energy-core/", "-o", str(out.with_suffix(".perf.csv"))],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if read_pkg_uj() is None and perf is None:
        raise SystemExit("RAPL not readable (energy_uj root-only, perf_event_paranoid > 0): needs the one-time root step")
    maxr = int((PKG / "max_energy_range_uj").read_text()) if read_pkg_uj() is not None else None
    samples = []
    t_end = time.time() + a.seconds
    prev = read_pkg_uj()
    acc = 0
    while time.time() < t_end:
        time.sleep(1.0)
        cur = read_pkg_uj()
        if cur is not None and prev is not None:
            acc += (cur - prev) % maxr if maxr else cur - prev
        prev = cur
        samples.append({"t": time.time(), "pkg_uj_acc": acc, "stat": proc_stat(), "wrf": wrf_ranks()})
    if perf:
        perf.terminate()
        perf.wait(10)
    first, last = samples[0], samples[-1]
    dt = last["t"] - first["t"]
    busy = {c: last["stat"][c][0] - first["stat"][c][0] for c in last["stat"]}
    wrf_cpus = sorted({r["psr"] for s in samples for r in s["wrf"]})
    wrf_jiffies = 0
    by_pid_first = {r["pid"]: r for r in first["wrf"]}
    for r in last["wrf"]:
        f0 = by_pid_first.get(r["pid"])
        if f0:
            wrf_jiffies += (r["utime"] + r["stime"]) - (f0["utime"] + f0["stime"])
    pkg_w = (last["pkg_uj_acc"] - first["pkg_uj_acc"]) / 1e6 / dt if last["pkg_uj_acc"] else None
    share = wrf_jiffies / max(1, sum(busy.values()))
    res = {
        "seconds": dt, "pkg_w_mean": pkg_w, "label_pkg": "M",
        "wrf_ranks": len(last["wrf"]), "wrf_cpus": wrf_cpus,
        "wrf_run_dirs": sorted({r["cwd"] for r in last["wrf"]}),
        "wrf_busy_share_of_all_busy": share,
        "wrf_attributed_w_pkg_share": (pkg_w * share) if pkg_w else None, "label_attr": "I (package x busy share)",
        "perf_csv": str(out.with_suffix(".perf.csv")) if perf else None,
        "note": "per-core [M] attribution from perf_csv (power_core/energy-core per CPU) when present",
    }
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
