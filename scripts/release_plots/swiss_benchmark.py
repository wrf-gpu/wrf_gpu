#!/usr/bin/env python3
"""Plot the shipped Swiss case using original CPU and cached GPU receipts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench_plots import read_dmon
import style
from style import plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-receipt", required=True, type=Path)
    parser.add_argument("--gpu-receipt", required=True, type=Path)
    parser.add_argument("--dmon", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--theme", choices=("light", "dark"), default="light")
    args = parser.parse_args()
    cpu = json.loads(args.cpu_receipt.read_text())
    gpu = json.loads(args.gpu_receipt.read_text())
    if not cpu.get("timing_valid") or not cpu.get("wrf_success"):
        raise ValueError("original CPU-WRF receipt must verify timing and completion")
    if (gpu.get("phase") != "cached" or not gpu.get("complete_24h")
            or gpu.get("device", {}).get("platform") != "gpu"):
        raise ValueError("benchmark needs a complete cached run on the actual GPU")
    if cpu["forecast_hours"] != 24 or gpu["forecast_hours"] != 24:
        raise ValueError("both receipts must cover the same 24 h window")
    if gpu["inputs_sha256"] != {k: v["sha256"] for k, v in cpu["inputs"].items()}:
        raise ValueError("CPU and GPU inputs differ")
    cpu_rate, gpu_rate = cpu["last_file_wall_s"] / 24, gpu["last_file_wall_s"] / 24
    if not 0 < gpu_rate <= gpu["wall_s"] / 24:
        raise ValueError("invalid GPU publication endpoint")
    power = read_dmon(args.dmon)
    summary = {"cpu_s_per_frame": cpu_rate, "gpu_s_per_frame": gpu_rate,
               "speedup": cpu_rate / gpu_rate, "timing_label": "M",
               "model_commit": gpu["git_head"], "cpu_energy_label": "U",
               "cpu_receipt": str(args.cpu_receipt), "gpu_receipt": str(args.gpu_receipt),
               "power": power, "power_source": str(args.dmon)}
    if ("energy_j" in power and power["span_s"] >= gpu["wall_s"] - 3
            and power.get("integration_method", "").startswith("trapezoidal")):
        summary["gpu_board_kj_per_frame"] = power["energy_j"] / 24000
        summary["gpu_energy_label"] = "M"
    palette = style.apply(args.theme)
    fig, axis = plt.subplots(figsize=(8, 3.6))
    rates = [cpu_rate, gpu_rate]
    bars = axis.barh([1, 0], rates,
                     color=[palette["series"][style.CPU], palette["series"][style.GPU]], height=0.5)
    axis.set_yticks([1, 0], ["CPU-WRF v4 · 4 cores", "wrf_gpu · RTX 5090"])
    axis.set_xlim(0, max(rates) * 1.38)
    for bar, rate in zip(bars, rates):
        axis.text(bar.get_width() + max(rates) * 0.02, bar.get_y() + bar.get_height() / 2,
                  f"{rate:.2f} s [M]", va="center", fontsize=10)
    axis.set_xlabel("Seconds per hourly forecast frame · whole cached run / 24")
    axis.set_title(f"Switzerland 3 km · {summary['speedup']:.1f}× measured speedup")
    footer = (f"Tree {gpu['git_head'][:9]}; original CPU-WRF; identical inputs; init + stepping + output. "
              "CPU energy unmeasured [U].")
    if "gpu_board_kj_per_frame" in summary:
        footer += (f"\nGPU board {summary['gpu_board_kj_per_frame']:.3f} kJ/frame [M]; "
                   f"{power['span_s']:.0f} s sampled; unlogged tails excluded; host CPU not measured.")
    style.footer(fig, footer, palette)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    style.save(fig, str(args.out))
    args.json.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
