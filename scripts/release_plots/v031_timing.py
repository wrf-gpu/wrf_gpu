"""Render v0.3.1 timing figures from the selected release receipt.

Run from the repository root: python scripts/release_plots/v031_timing.py
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/release/evidence/v031"
OUTPUT = ROOT / "docs/release/img"


def main():
    receipt = json.loads((EVIDENCE / "prod24_receipt_selected.json").read_text())
    summary = json.loads((EVIDENCE / "summary.json").read_text())
    timing = summary["timing_PROD_RC2_M"]
    current = receipt["t_end"] / 24
    cpu = timing["cpu_wrf_12rank_s_per_fc_h"]
    previous = timing["baseline_FINALb_v030"]["24h"]
    reduction = 100 * (1 - current / previous)
    for dark in (False, True):
        background = "#111827" if dark else "#ffffff"
        foreground = "#edf2f7" if dark else "#17243b"
        muted = "#aebbd0" if dark else "#50627b"
        colors = ["#97a7bc", "#507cc0", "#18a58b"]
        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                             "text.color": foreground, "axes.labelcolor": foreground,
                             "xtick.color": foreground, "ytick.color": foreground}):
            fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7), dpi=170)
            fig.set_facecolor(background)
            for ax in axes:
                ax.set_facecolor(background)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.tick_params(length=0)
                ax.set_axisbelow(True)
                ax.grid(axis="y", color=muted, alpha=0.18)
            ax = axes[0]
            ax.bar(["CPU-WRF\n12 cores", "v0.3.0\nRTX 5090", "v0.3.1\nRTX 5090"],
                   [cpu, previous, current], color=colors, width=0.56)
            ax.set_yscale("log")
            ax.set_ylim(1, 200)
            ax.set_yticks([1, 10, 100], ["1", "10", "100"])
            ax.set_ylabel("Seconds per forecast hour [M] · log scale")
            ax.set_title("Canary 9/3 km · 24 h", loc="left", color=foreground, pad=22)
            for x, value in enumerate([cpu, previous, current]):
                ax.text(x, value * 1.15, f"{value:.3f}" if x else f"{value:g}",
                        ha="center", fontweight="bold")
            ax.text(0.98, 0.93, f"{cpu/current:.1f}×\nreference rate ratio",
                    transform=ax.transAxes, ha="right", va="top", color=colors[2])
            ax = axes[1]
            ax.bar(["v0.3.0", "v0.3.1"], [previous, current],
                   color=colors[1:], width=0.5)
            ax.set_ylim(0, 5.2)
            ax.set_ylabel("GPU seconds per forecast hour [M]")
            ax.set_title("Complete GPU process wall", loc="left", color=foreground, pad=22)
            for x, value in enumerate([previous, current]):
                ax.text(x, value + 0.1, f"{value:.3f}", ha="center", fontweight="bold")
            ax.text(0.98, 0.93, f"{reduction:.1f}% less wall time",
                    transform=ax.transAxes, ha="right", va="top", color=colors[2])
            fig.subplots_adjust(left=0.075, right=0.98, top=0.83, bottom=0.27, wspace=0.34)
            fig.suptitle("Measured cached forecasts on RTX 5090 · Ryzen 9 9950X host",
                         x=0.075, ha="left", fontsize=15, fontweight="bold", y=0.97)
            fig.text(0.075, 0.085,
                     "GPU clocks include initialization and history output; CPU reference is its main loop.\n"
                     "v0.3.1 timing: RC2, 86.507 s / 24 h. RC3 changes glacier output only; forecast fields are unchanged.",
                     color=muted, fontsize=9.5, linespacing=1.7)
            suffix = "_dark" if dark else ""
            fig.savefig(OUTPUT / f"v031_timing{suffix}.png", facecolor=background)
            plt.close(fig)


if __name__ == "__main__":
    main()
