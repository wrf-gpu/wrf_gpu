#!/usr/bin/env python3
"""T2 and 10 m wind speed at named places, CPU-WRF vs GPU, over the whole forecast (nearest grid cell).

  JAX_PLATFORMS=cpu python scripts/release_plots/point_series.py --cpu-dir CPU --gpu-dir GPU --domain d03 --out X.png
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import netCDF4
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import style  # noqa: E402
import wrfout_pairs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402

PLACES = [
    ("Tenerife Norte airport", 28.4827, -16.3415),
    ("Izaña (2,371 m)", 28.3090, -16.4994),
    ("Tenerife Sur airport", 28.0445, -16.5725),
    ("Gran Canaria airport", 27.9319, -15.3866),
]


def nearest(lat, lon, plat, plon):
    d = (lat - plat) ** 2 + ((lon - plon) * np.cos(np.deg2rad(plat))) ** 2
    j, i = np.unravel_index(np.argmin(d), d.shape)
    inside = 0 < j < lat.shape[0] - 1 and 0 < i < lat.shape[1] - 1
    return (j, i) if inside else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-dir", required=True)
    ap.add_argument("--gpu-dir", required=True)
    ap.add_argument("--domain", default="d03")
    ap.add_argument("--max-places", type=int, default=3)
    ap.add_argument("--title", default=None)
    ap.add_argument("--footer", default="")
    ap.add_argument("--theme", default="light", choices=["light", "dark"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pr = wrfout_pairs.pairs(pathlib.Path(a.cpu_dir), pathlib.Path(a.gpu_dir), a.domain)
    with netCDF4.Dataset(pr[0][1]) as d:
        lat, lon, hgt = (np.asarray(d.variables[k][0], float) for k in ("XLAT", "XLONG", "HGT"))
    pts = []
    for name, plat, plon in PLACES:
        ji = nearest(lat, lon, plat, plon)
        if ji is not None:
            pts.append((name, ji))
    pts = pts[: a.max_places]
    times = [p[0] for p in pr]
    series = {k: np.zeros((2, len(pr), len(pts))) for k in ("T2", "WS")}
    for n, (_, cp, gp) in enumerate(pr):
        for s, path in enumerate((cp, gp)):
            with netCDF4.Dataset(path) as d:
                t2 = np.asarray(d.variables["T2"][0], float) - 273.15
                ws = np.hypot(np.asarray(d.variables["U10"][0], float), np.asarray(d.variables["V10"][0], float))
            for k, (_, (j, i)) in enumerate(pts):
                series["T2"][s, n, k] = t2[j, i]
                series["WS"][s, n, k] = ws[j, i]
    t = style.apply(a.theme)
    fig, axes = plt.subplots(2, len(pts), figsize=(4.6 * len(pts), 5.4), sharex=True, squeeze=False)
    for k, (name, (j, i)) in enumerate(pts):
        for r, (key, lab) in enumerate([("T2", "2 m temperature [°C]"), ("WS", "10 m wind speed [m/s]")]):
            ax = axes[r, k]
            ax.plot(times, series[key][0, :, k], color=t["series"][style.CPU], lw=2, label="CPU-WRF")
            ax.plot(times, series[key][1, :, k], color=t["series"][style.GPU], lw=1.6, label="wrf_gpu (GPU)")
            if k == 0:
                ax.set_ylabel(lab)
            if r == 0:
                ax.set_title(f"{name}\n(grid cell {hgt[j, i]:.0f} m)", loc="left", fontsize=9.5)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %Hh"))
            ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
            ax.tick_params(axis="x", labelrotation=0, labelsize=7.5)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2)
    lead = int((times[-1] - times[0]).total_seconds() // 3600)
    fig.suptitle(a.title or f"Forecast at single places, {a.domain}, 0–{lead} h: CPU-WRF and GPU", x=0.01, ha="left",
                 fontsize=13, fontweight="semibold", color=t["text"])
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    if a.footer:
        style.footer(fig, a.footer, t)
    style.save(fig, a.out)
    print(a.out, len(pr), "frames", [p[0] for p in pts])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
