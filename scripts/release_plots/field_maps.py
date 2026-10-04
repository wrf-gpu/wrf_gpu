#!/usr/bin/env python3
"""CPU-WRF | GPU | GPU − CPU maps of surface fields at one lead (default: last common frame).

Rows: T2, 10 m wind speed, accumulated grid-scale rain (RAINNC). Coastline = LANDMASK 0.5 contour.
Reads only the needed variables of two wrfout files (numpy + netCDF4; no JAX).

  JAX_PLATFORMS=cpu python scripts/release_plots/field_maps.py --cpu-dir CPU --gpu-dir GPU --domain d03 \
      --out docs/release/img/maps_d03.png
"""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys

import netCDF4
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import style  # noqa: E402
import wrfout_pairs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

ORANGE = ["#fde7dc", "#f8b99b", "#f08a5f", "#eb6834", "#c4501f", "#923a14", "#5e250c"]


def read(path: pathlib.Path) -> dict:
    with netCDF4.Dataset(path) as d:
        v = d.variables
        u10, v10 = np.asarray(v["U10"][0], float), np.asarray(v["V10"][0], float)
        return {
            "T2": np.asarray(v["T2"][0], float) - 273.15,
            "WSPD10": np.hypot(u10, v10),
            "RAINNC": np.asarray(v["RAINNC"][0], float),
            "LANDMASK": np.asarray(v["LANDMASK"][0], float),
            "XLAT": np.asarray(v["XLAT"][0], float),
            "XLONG": np.asarray(v["XLONG"][0], float),
        }


ROWS = [
    ("T2", "2 m temperature", "°C", "orange"),
    ("WSPD10", "10 m wind speed", "m/s", "blue"),
    ("RAINNC", "rain since start", "mm", "blue"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-dir", required=True)
    ap.add_argument("--gpu-dir", required=True)
    ap.add_argument("--domain", default="d03")
    ap.add_argument("--init", default=None, help="UTC ISO initialization time; otherwise first common frame")
    ap.add_argument("--lead", type=int, default=None, help="hours after initialization (default: last common frame)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--footer", default="")
    ap.add_argument("--theme", default="light", choices=["light", "dark"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pr = wrfout_pairs.pairs(pathlib.Path(a.cpu_dir), pathlib.Path(a.gpu_dir), a.domain)
    if not pr:
        raise SystemExit("no paired frames")
    t0 = dt.datetime.fromisoformat(a.init) if a.init else pr[0][0]
    if t0.tzinfo:
        t0 = t0.astimezone(dt.timezone.utc).replace(tzinfo=None)
    if a.lead is None:
        valid, cpath, gpath = pr[-1]
    else:
        sel = [p for p in pr if (p[0] - t0).total_seconds() == a.lead * 3600]
        if not sel:
            raise SystemExit(f"lead {a.lead} h not paired")
        valid, cpath, gpath = sel[0]
    lead_h = int((valid - t0).total_seconds() // 3600)
    c, g = read(cpath), read(gpath)

    t = style.apply(a.theme)
    seq = {"blue": LinearSegmentedColormap.from_list("b", t["seq"] if a.theme == "light" else t["seq"][::-1]),
           "orange": LinearSegmentedColormap.from_list("o", ORANGE)}
    div = LinearSegmentedColormap.from_list("d", ["#184f95", "#6da7ec", t["div_mid"], "#ef8a8a", "#a32020"])
    lon, lat = c["XLONG"], c["XLAT"]
    ny, nx = lat.shape
    aspect = (ny / nx) / np.cos(np.deg2rad(lat.mean()))
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 3 * 4.0 * aspect + 1.0), layout="constrained")
    for r, (key, label, unit, ramp) in enumerate(ROWS):
        cv, gv = c[key], g[key]
        lo = float(min(cv.min(), gv.min()))
        hi = float(max(cv.max(), gv.max()))
        if key == "RAINNC":
            lo = 0.0
        d = gv - cv
        lim = float(np.nanpercentile(np.abs(d), 99.5)) or 1e-6
        rmse = float(np.sqrt(np.mean(d * d)))
        for k, (arr, name) in enumerate([(cv, "CPU-WRF"), (gv, "GPU (wrf_gpu)"), (d, "GPU − CPU")]):
            ax = axes[r, k]
            if k < 2:
                im = ax.pcolormesh(lon, lat, arr, cmap=seq[ramp], vmin=lo, vmax=hi, shading="auto", rasterized=True)
            else:
                im = ax.pcolormesh(lon, lat, arr, cmap=div, norm=TwoSlopeNorm(0.0, -lim, lim), shading="auto",
                                   rasterized=True)
            if c["LANDMASK"].min() < 0.5 < c["LANDMASK"].max():
                ax.contour(lon, lat, c["LANDMASK"], levels=[0.5], colors=t["text"], linewidths=0.6)
            ax.set_aspect(1.0 / np.cos(np.deg2rad(lat.mean())))
            ax.grid(False)
            ax.tick_params(labelsize=7)
            ttl = f"{name}" if k < 2 else f"GPU − CPU  (RMSE {rmse:.2f} {unit})"
            ax.set_title(ttl, fontsize=9.5, loc="left")
            if k != 1:
                cb = fig.colorbar(im, ax=ax if k == 2 else axes[r, :2].tolist(), shrink=0.9, pad=0.01, aspect=25)
                cb.set_label(f"{label} [{unit}]" if k == 0 else f"Δ [{unit}]", fontsize=8)
                cb.ax.tick_params(labelsize=7)
                cb.outline.set_visible(False)
    title = a.title or f"CPU-WRF and GPU — {a.domain} at +{lead_h} h ({valid:%Y-%m-%d %H} UTC)"
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="semibold", color=t["text"])
    if a.footer:
        style.footer(fig, a.footer + "\nDifference colors use the 99.5th percentile; RMSE uses every cell.", t)
        fig.get_layout_engine().set(rect=(0, 0.075, 1, 0.925))
    style.save(fig, a.out)
    print(a.out, "lead", lead_h, cpath.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
