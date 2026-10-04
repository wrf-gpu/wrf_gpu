"""Shared look for the v0.3 release plots (static PNG for the GitHub README).

Palette = dataviz reference instance (validated categorical order, light + dark steps).
Roles: GPU (wrf_gpu) = slot 1 blue, CPU-WRF = slot 2 orange, third entity = slot 3 aqua.
Text never wears a series colour; limits/thresholds are neutral dashed ink.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "text2": "#52514e",
        "muted": "#8a8984",
        "grid": "#e4e3df",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
        "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
        "div_mid": "#f0efec",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "text2": "#c3c2b7",
        "muted": "#8f8e86",
        "grid": "#33332f",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
        "seq": ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
        "div_mid": "#383835",
    },
}

GPU, CPU, THIRD = 0, 1, 2


def apply(theme: str = "light") -> dict:
    t = THEMES[theme]
    plt.rcParams.update(
        {
            "figure.facecolor": t["surface"],
            "axes.facecolor": t["surface"],
            "savefig.facecolor": t["surface"],
            "axes.edgecolor": t["grid"],
            "axes.labelcolor": t["text2"],
            "axes.titlecolor": t["text"],
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": t["grid"],
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": t["text2"],
            "ytick.color": t["text2"],
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "text.color": t["text"],
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "font.family": "DejaVu Sans",
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
        }
    )
    return t


def footer(fig, text: str, t: dict) -> None:
    """Provenance line at the bottom of every figure."""
    fig.text(0.01, 0.005, text, fontsize=7, color=t["muted"], ha="left", va="bottom")


def save(fig, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
