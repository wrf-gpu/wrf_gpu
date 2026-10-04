#!/usr/bin/env python3
"""One-glance fidelity: share of the frozen D6 RMSE limit used, per gate field × domain (rows) and lead hour (columns).

Cell = worst case (max over cases) of RMSE(GPU − CPU-WRF) / limit. Sequential single-hue ramp 0 → 100 %; a cell
> 100 % (a gate failure) is outlined. Reads the same scorer JSONs as identity_curves.py.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import style  # noqa: E402
from identity_curves import FIELDS, load, stack  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append", required=True)
    ap.add_argument("--domains", nargs="+", default=["d01", "d02", "d03"])
    ap.add_argument("--title", default="Share of the release tolerance used, GPU vs CPU-WRF")
    ap.add_argument("--footer", default="")
    ap.add_argument("--theme", default="light", choices=["light", "dark"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cases = {s.partition("=")[0]: pathlib.Path(s.partition("=")[2]) for s in a.case}
    data, limits = load(cases, a.domains)
    rows, labels, leads = [], [], None
    for name, *_ in FIELDS:
        for dom in a.domains:
            if dom not in data or name not in data[dom] or name not in limits:
                continue
            ld, rm, _ = stack(data[dom][name])
            leads = ld if leads is None or len(ld) > len(leads) else leads
            rows.append((ld, np.nanmax(rm, axis=0) / limits[name]))
            labels.append(f"{name} {dom}")
    grid = np.full((len(rows), len(leads)), np.nan)
    for r, (ld, v) in enumerate(rows):
        grid[r, np.searchsorted(leads, ld)] = v
    t = style.apply(a.theme)
    ramp = t["seq"]
    cmap = LinearSegmentedColormap.from_list("share", [t["surface"]] + ramp)
    fig, ax = plt.subplots(figsize=(min(16, max(10, 4 + 0.13 * len(leads))), 0.22 * len(rows) + 1.6))
    im = ax.imshow(np.clip(grid * 100, 0, 100), aspect="auto", cmap=cmap, vmin=0, vmax=100, interpolation="nearest",
                   extent=(leads[0] - 0.5, leads[-1] + 0.5, len(rows) - 0.5, -0.5))
    bad = np.argwhere(grid > 1.0)
    for r, c in bad:
        ax.add_patch(plt.Rectangle((leads[c] - 0.5, r - 0.5), 1, 1, fill=False, ec=t["series"][7], lw=1.5))
    ax.set_yticks(range(len(rows)), labels, fontsize=7.5)
    ax.set_xlabel("forecast lead [h]")
    ax.grid(False)
    step = 24 if leads[-1] > 96 else 12 if leads[-1] > 36 else 6
    ax.set_xticks(np.arange(0, leads[-1] + 1, step))
    for y in np.arange(len(a.domains), len(rows), len(a.domains)) - 0.5:
        ax.axhline(y, color=t["surface"], lw=2)
    cb = fig.colorbar(im, ax=ax, pad=0.01, aspect=30)
    cb.set_label("% of D6 RMSE limit", fontsize=8.5)
    cb.outline.set_visible(False)
    worst = float(np.nanmax(grid)) * 100
    ax.set_title(f"{a.title} — largest share {worst:.0f} %", loc="left", fontsize=11)
    fig.tight_layout()
    if a.footer:
        style.footer(fig, a.footer.replace("dashed = release-gate RMSE limit",
                                           "colour = share of frozen RMSE limit"), t)
    style.save(fig, a.out)
    print(a.out, "worst", round(worst, 1), "fails", len(bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
