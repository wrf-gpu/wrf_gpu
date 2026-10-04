"""Per-case admission for N concurrent WN3 cases on one GPU (VR20 / E97 rule). CPU only.

Usage: admission.py RECEIPT_JSON ROOT [--requested N] [--rss-gb 13.6] [--reserve-gb 8]
f      = (1.2 * (JAX live peak + 0.4 GB code) + 0.5 GiB) / 33.66 GB          (XLA_CLIENT_MEM_FRACTION per process)
VRAM   : N * (f * 31.35 GiB + 1 GiB) + desktop_now <= 31.35 GiB              (desktop = memory.used right now)
host   : N * rss_gb <= MemAvailable - reserve_gb
Writes ROOT/mem_fraction and ROOT/admission.json; prints N_max.
"""
import argparse
import json
from pathlib import Path
import subprocess

GIB = 1024 ** 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("receipt", type=Path)
    ap.add_argument("root", type=Path)
    ap.add_argument("--requested", type=int, default=6)
    ap.add_argument("--rss-gb", type=float, default=None, help="per-process host RSS; default = receipt VmHWM")
    ap.add_argument("--reserve-gb", type=float, default=8.0)
    a = ap.parse_args()
    r = json.loads(a.receipt.read_text())
    live = float(r["jax_memory_stats"]["peak_bytes_in_use"]) / 1e9
    f = (1.2 * (live + 0.4) + 0.5 * GIB / 1e9) / 33.66
    per_case_gib = f * 31.35 + 1.0
    used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True).stdout.split()
    desktop_gib = float(used[0]) / 1024.0
    n_vram = int((31.35 - desktop_gib) // per_case_gib)
    avail_gb = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable")) / 1e6
    rss = a.rss_gb if a.rss_gb is not None else float(r.get("host_vmhwm_kb", 13.6e6)) / 1e6
    n_host = int((avail_gb - a.reserve_gb) // rss)
    n_max = max(0, min(a.requested, n_vram, n_host))
    out = {"receipt": str(a.receipt), "jax_live_peak_gb": live, "mem_fraction": round(f, 4), "per_case_gib": per_case_gib,
           "desktop_gib_now": desktop_gib, "n_vram": n_vram, "mem_available_gb": avail_gb, "rss_gb_per_case": rss,
           "reserve_gb": a.reserve_gb, "n_host": n_host, "requested": a.requested, "n_max": n_max}
    a.root.mkdir(parents=True, exist_ok=True)
    (a.root / "mem_fraction").write_text(f"{f:.4f}\n")
    (a.root / "admission.json").write_text(json.dumps(out, indent=1) + "\n")
    print(n_max)


if __name__ == "__main__":
    main()
