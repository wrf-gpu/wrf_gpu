#!/usr/bin/env python3
"""GPU-vs-CPU-WRF RMSE over lead time for the 10 hard-gate D6 fields, from wrfout-grid-comparison-v1 JSONs.

Input: one or more case scorer dirs (each holds d01.json, d02.json, ... written by scripts/wn3_score.py /
compare_wrfout_grid.py). No wrfout is read here. Lines = domains (ordinal blue ramp: coarse -> fine),
band = min..max over cases, line = median over cases, dashed = frozen D6 RMSE limit.

  JAX_PLATFORMS=cpu python scripts/release_plots/identity_curves.py \
      --case 0227=/path/d6_72h --case 0502=/path/d6_72h --out docs/release/img/identity_curves.png
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

FIELDS = [
    ("T2", "2 m temperature", "K", 1.0),
    ("U10", "10 m wind U", "m/s", 1.0),
    ("V10", "10 m wind V", "m/s", 1.0),
    ("PSFC", "surface pressure", "hPa", 0.01),
    ("RAINNC", "accumulated rain", "mm", 1.0),
    ("T", "potential temp. (3-D)", "K", 1.0),
    ("U", "wind U (3-D)", "m/s", 1.0),
    ("V", "wind V (3-D)", "m/s", 1.0),
    ("W", "vertical wind (3-D)", "m/s", 1.0),
    ("QVAPOR", "water vapour (3-D)", "g/kg", 1000.0),
]
DOMAIN_LABEL = {"d01": "d01", "d02": "d02", "d03": "d03"}


def load(case_dirs: dict[str, pathlib.Path], domains: list[str]):
    """-> data[domain][field] = (leads[h], rmse[case, lead], bias[case, lead]); limits[field]."""
    data: dict = {}
    limits: dict = {}
    for dom in domains:
        per_field: dict = {}
        for label, d in case_dirs.items():
            path = d / f"{dom}.json"
            if not path.exists():
                raise ValueError(f"missing identity domain: {label} {path}")
            js = json.loads(path.read_text())
            for name, *_ in FIELDS:
                fs = js["field_summaries"].get(name)
                if fs is None:
                    raise ValueError(f"missing identity field: {label} {dom} {name}")
                rows = sorted(fs["by_lead"], key=lambda r: r["lead_h"])
                if not rows or fs.get("missing_leads") or fs.get("incompatible_leads"):
                    raise ValueError(f"incomplete identity field: {label} {dom} {name}")
                leads = np.array([r["lead_h"] for r in rows], float)
                rmse = np.array([np.nan if r["rmse"] is None else r["rmse"] for r in rows], float)
                bias = np.array([np.nan if r["bias"] is None else r["bias"] for r in rows], float)
                if not np.isfinite(rmse).all() or not np.isfinite(bias).all():
                    raise ValueError(f"nonfinite identity metrics: {label} {dom} {name}")
                per_field.setdefault(name, []).append((label, leads, rmse, bias))
                for r in rows:
                    spec = (r.get("tolerance_result") or {}).get("spec") or {}
                    if "rmse" in spec:
                        limit = float(spec["rmse"])
                        if name in limits and limits[name] != limit:
                            raise ValueError(f"inconsistent frozen limit: {name}")
                        limits[name] = limit
        if per_field:
            data[dom] = per_field
    return data, limits


def stack(entries):
    """Require complete hourly leads; an intersection would hide omissions."""
    common = entries[0][1].tolist()
    if common != list(range(int(common[0]), int(common[-1]) + 1)) or any(e[1].tolist() != common for e in entries):
        raise ValueError("identity cases must have the same complete hourly lead set")
    leads = np.array(common)
    rm = np.vstack([e[2][np.isin(e[1], leads)] for e in entries])
    bi = np.vstack([e[3][np.isin(e[1], leads)] for e in entries])
    return leads, rm, bi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append", required=True, help="LABEL=DIR with dNN.json scorer outputs")
    ap.add_argument("--domains", nargs="+", default=["d01", "d02", "d03"])
    ap.add_argument("--title", default="GPU vs CPU-WRF: RMSE over forecast lead")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--footer", default="")
    ap.add_argument("--theme", default="light", choices=["light", "dark"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary-json", default=None, help="write the plotted numbers (worst lead per field/domain)")
    ap.add_argument("--metric", choices=["rmse", "bias"], default="rmse")
    a = ap.parse_args()

    case_dirs = {}
    for spec in a.case:
        label, _, d = spec.partition("=")
        case_dirs[label] = pathlib.Path(d)
    data, limits = load(case_dirs, a.domains)
    t = style.apply(a.theme)
    # ordinal blue ramp, coarse -> fine; steps chosen far apart so three nests stay distinguishable
    dom_colors = ({"d01": "#86b6ef", "d02": "#2a78d6", "d03": "#0d366b"} if a.theme == "light"
                  else {"d01": "#3987e5", "d02": "#5598e7", "d03": "#cde2fb"})

    fig, axes = plt.subplots(2, 5, figsize=(15, 6.4), sharex=True)
    summary: dict = {}
    for ax, (name, title, unit, scale) in zip(axes.flat, FIELDS):
        ymax = 0.0
        for dom in a.domains:
            if dom not in data or name not in data[dom]:
                continue
            leads, rm, bi = stack(data[dom][name])
            bias = bi * scale
            rm = rm * scale
            med = np.nanmedian(rm, axis=0)
            lo, hi = np.nanmin(rm, axis=0), np.nanmax(rm, axis=0)
            c = dom_colors[dom]
            if rm.shape[0] > 1 and a.metric == "rmse":
                ax.fill_between(leads, lo, hi, color=c, alpha=0.18, lw=0)
            if a.metric == "rmse":
                ax.plot(leads, med, color=c, lw=1.8, label=DOMAIN_LABEL[dom])
            else:
                if rm.shape[0] > 1:
                    ax.fill_between(leads, bias.min(axis=0), bias.max(axis=0), color=c, alpha=0.18, lw=0)
                ax.plot(leads, np.median(bias, axis=0), color=c, lw=1.8, label=DOMAIN_LABEL[dom])
            ymax = max(ymax, float(np.nanmax(hi)))
            k = int(np.nanargmax(hi))
            summary.setdefault(name, {})[dom] = {
                "worst_lead_h": float(leads[k]), "worst_rmse": float(hi[k] / scale),
                "median_rmse_last_lead": float(med[-1] / scale), "last_lead_h": float(leads[-1]),
                "cases": rm.shape[0],
                "worst_abs_bias": float(np.max(np.abs(bi))),
                "median_bias_last_lead": float(np.median(bi[:, -1])),
            }
        lim = limits.get(name)
        if lim is not None and a.metric == "rmse":
            ax.axhline(lim * scale, color=t["text2"], ls=(0, (4, 3)), lw=1.1)
            ax.text(0.98, lim * scale, "D6 limit", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    fontsize=7.5, color=t["text2"])
            top = max(lim * scale, ymax) * 1.12
            ax.set_ylim(0, top)
            frac = ymax / (lim * scale) if lim else float("nan")
            for dom in summary.get(name, {}):
                summary[name][dom]["limit"] = lim
            ax.text(0.03, 0.80, f"max {100 * frac:.0f} % of limit", transform=ax.transAxes, fontsize=7.5,
                    color=t["text2"])
        if a.metric == "bias":
            ax.axhline(0, color=t["text2"], lw=1, ls=(0, (4, 3)))
        ax.set_title(f"{name} — {title}", fontsize=9.5, loc="left")
        ax.set_ylabel(f"{'RMSE' if a.metric == 'rmse' else 'bias (GPU − CPU)'} [{unit}]")
        lead_max = max((stack(data[d][name])[0][-1] for d in data if name in data[d]), default=24)
        step = 24 if lead_max > 96 else 12 if lead_max > 36 else 6
        ax.set_xticks(np.arange(0, lead_max + 1, step))
        ax.set_xlim(0, lead_max)
    for ax in axes[1]:
        ax.set_xlabel("forecast lead [h]")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), bbox_to_anchor=(0.995, 1.0))
    fig.suptitle(a.title, x=0.01, ha="left", y=1.0, fontsize=13, fontweight="semibold", color=t["text"])
    if a.subtitle:
        fig.text(0.01, 0.955, a.subtitle, fontsize=9, color=t["text2"], ha="left")
    fig.tight_layout(rect=(0, 0.02, 1, 0.955), h_pad=1.2)
    if a.footer:
        style.footer(fig, a.footer, t)
    style.save(fig, a.out)
    if a.summary_json:
        pathlib.Path(a.summary_json).write_text(json.dumps(
            {"cases": {k: str(v) for k, v in case_dirs.items()}, "fields": summary}, indent=1))
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
