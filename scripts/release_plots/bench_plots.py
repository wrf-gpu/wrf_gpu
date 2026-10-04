#!/usr/bin/env python3
"""Release benchmark figures from docs/release/showcase_inputs.json (+ the arm dirs it points to).

Figures:
  speed.png     per-case speed: PROD s per forecast hour and WN3 s per simulated hour, CPU-WRF vs GPU
  parallel.png  N WN3 cases in parallel on one GPU: throughput (case-hours per wall-hour), VRAM, energy per case-hour
  history.png   S1 proxy across the v0.25 iterations
  numbers.json  every plotted number with label + ref (README tables read this)

  JAX_PLATFORMS=cpu python scripts/release_plots/bench_plots.py --inputs docs/release/showcase_inputs.json --out-dir DIR
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import style  # noqa: E402
from run_data import arm_rates, prod_summary  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def read_dmon(path: pathlib.Path) -> dict:
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
        v = [float(r[name]) for r in rows if r.get(name, "-") not in ("-", "")]
        return np.array(v)

    pwr = col("pwr")
    out = {"samples": len(rows), "has_power": bool(pwr.size)}
    if pwr.size:
        power_rows = [r for r in rows if r.get("pwr", "-") not in ("-", "")]
        # Public launcher uses dmon -o T. Its sampling loop can miss seconds;
        # sample count alone then understates energy. Integrate its actual clock.
        if len(power_rows) >= 2 and all("Time" in r for r in power_rows):
            times, day, previous = [], 0.0, None
            for r in power_rows:
                hh, mm, ss = map(float, r["Time"].split(":"))
                sec = hh * 3600 + mm * 60 + ss
                if previous is not None and sec < previous:
                    if previous - sec < 43200:
                        raise ValueError(f"{path}: dmon clock moved backwards")
                    day += 86400  # midnight, not an elapsed-time discontinuity
                times.append(day + sec)
                previous = sec
            gaps = np.diff(times)
            span = times[-1] - times[0]
            if span <= 0:
                raise ValueError(f"{path}: nonpositive power sample span")
            energy = float(np.sum((pwr[:-1] + pwr[1:]) * 0.5 * gaps))
            out.update(energy_j=energy, mean_w=energy / span, span_s=span, max_gap_s=float(gaps.max()),
                       integration_method="trapezoidal integral between timestamped power samples; unlogged tails excluded")
        else:
            # Historical bench logs omit clocks. Retain their declared -d 1
            # rectangular estimate, explicitly identified in the evidence.
            out.update(energy_j=float(pwr.sum()), mean_w=float(pwr.mean()), span_s=float(pwr.size),
                       integration_method="nominal 1 s rectangular samples (log has no usable timestamp span)")
        out.update(median_w=float(np.median(pwr)), valid_power_samples=int(pwr.size))
    fb = col("fb")
    if fb.size:
        out["fb_peak_mib"] = float(fb.max())
    return out


def arm_summary(arm: dict) -> dict:
    d = pathlib.Path(arm["dir"])
    an = arm_rates(d)
    n, hours = int(an["n_cases"]), float(an["hours"])
    s = {
        "dir": str(d), "tree": arm.get("tree"), "n": n, "hours": hours,
        "wall_s": float(an["arm_wall_s"]),
        "whole_s_per_case_h": float(an["throughput_whole_run_s_per_case_h"]),
        "stepping_s_per_case_h": float(an["throughput_stepping_s_per_case_h"]),
        "vram_sum_pid_peaks_mib": float(an.get("vram_sum_of_pid_peaks_mib") or np.nan),
        "fb_peak_mib": float((an.get("dmon") or {}).get("fb_peak_mib") or np.nan),
        "timing_method": an["timing_method"],
        "label": "M", "ref": an["ref"],
    }
    dm = d / "dmon.log"
    if dm.exists():
        info = read_dmon(dm)
        if info["has_power"]:
            s["gpu_energy_j"] = info["energy_j"]
            s["gpu_kj_per_case_h"] = info["energy_j"] / 1000.0 / (n * hours)
            s["gpu_mean_w"] = info["mean_w"]
    return s


def fmt(v, nd=1):
    return f"{v:.{nd}f}"


def fig_speed(cfg, t, out):
    cpu = cfg["cpu"]
    prod_gpu = prod_summary(cfg["prod"])["s_per_fch"]
    prod_cpu = cpu["prod_12rank_s_per_fch"]["value"]
    solo = [s for s in (arm_summary(a) for a in cfg["wn3_arms"]) if s["n"] == 1]
    wn3_gpu = solo[0]["stepping_s_per_case_h"] if solo else cfg["wn3_solo"]["warm_s_per_simh"]["value"]
    wn3_cpu = cpu["wn3_4rank_s_per_simh"]["value"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.0))
    panels = [
        (axes[0], "PROD Canary 9/3 km, 2 domains — seconds per forecast hour",
         [("CPU-WRF, 12 cores", prod_cpu, style.CPU), ("wrf_gpu, 1 × RTX 5090", prod_gpu, style.GPU)]),
        (axes[1], "WN3 Tenerife 9/3/1 km, 3 domains — seconds per simulated hour",
         [("CPU-WRF, 4 cores", wn3_cpu, style.CPU), ("wrf_gpu, 1 × RTX 5090", wn3_gpu, style.GPU)]),
    ]
    for ax, title, bars in panels:
        ys = np.arange(len(bars))[::-1]
        for y, (lab, v, slot) in zip(ys, bars):
            ax.barh(y, v, height=0.55, color=t["series"][slot], edgecolor=t["surface"], linewidth=2)
            ax.text(v + max(b[1] for b in bars) * 0.01, y, f"{v:.1f} s", va="center", fontsize=9, color=t["text"])
        ax.set_yticks(ys, [b[0] for b in bars])
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(b[1] for b in bars) * 1.18)
        ax.set_xlabel("wall seconds (lower is faster)")
        sp = bars[0][1] / bars[1][1]
        ax.text(0.98, 0.10, f"{sp:.1f}× faster", transform=ax.transAxes, ha="right", fontsize=12,
                fontweight="semibold", color=t["text"])
    fig.tight_layout()
    style.footer(fig, f"GPU: PROD {cfg['prod']['whole_run_hours']:g} h whole run incl. start-up; WN3 steady rate, one case alone. "
                      f"{cfg['tree_label']}.", t)
    style.save(fig, out)
    return {"prod_speedup": prod_cpu / prod_gpu, "wn3_solo_speedup": wn3_cpu / wn3_gpu}


def fig_parallel(cfg, arms, t, out):
    cpu = cfg["cpu"]
    arms = sorted(arms, key=lambda a: a["n"])
    n = np.array([a["n"] for a in arms])
    whole = np.array([3600.0 / a["whole_s_per_case_h"] for a in arms])
    step = np.array([3600.0 / a["stepping_s_per_case_h"] for a in arms])
    cpu_tp = 3600.0 / cpu["wn3_3x4_s_per_case_h"]["value"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    ax = axes[0]
    # R3 (72 h) and R4 (4 h) have different startup amortization. Do not
    # connect duplicate N values as if they were one equal-work sweep.
    for hours in sorted({a["hours"] for a in arms}):
        group = [a for a in arms if a["hours"] == hours]
        gn = [a["n"] for a in group]
        gs = [3600 / a["stepping_s_per_case_h"] for a in group]
        gw = [3600 / a["whole_s_per_case_h"] for a in group]
        steady_label = "steady output" if any("polling" in a["timing_method"] for a in group) else "stepping"
        ax.plot(gn, gs, color=t["series"][style.GPU], lw=2, marker="o", ms=7,
                label=f"wrf_gpu, {hours:g} h/case, {steady_label}")
        ax.plot(gn, gw, color=t["series"][style.GPU], lw=2, ls=(0, (4, 2)), marker="o", ms=7,
                mfc=t["surface"], label=f"wrf_gpu, {hours:g} h/case, whole run")
    ax.axhline(cpu_tp, color=t["series"][style.CPU], lw=2, label="CPU-WRF, 3 cases × 4 cores (12 cores)")
    for x, y in zip(n, whole):
        ax.annotate(f"{y / cpu_tp:.1f}× CPU", (x, y), textcoords="offset points", xytext=(10, -4), ha="left",
                    fontsize=8.5, color=t["text2"])
    ax.text(n.max() + 0.25, cpu_tp, f"CPU: {cpu_tp:.0f}", va="bottom", ha="right", fontsize=8.5, color=t["text2"])
    ax.set_xlim(n.min() - 0.4, n.max() + 0.4)
    ax.set_xticks(sorted(set(n)))
    ax.set_xlabel("WN3 cases running in parallel on one RTX 5090")
    ax.set_ylabel("simulated case-hours per wall-hour")
    ax.set_ylim(0, max(step.max(), whole.max()) * 1.15)
    ax.set_title("Throughput (higher is better)", loc="left")
    ax.legend(loc="upper left")

    ax = axes[1]
    memory = {int(x): max(a["vram_sum_pid_peaks_mib"] / 1024 for a in arms if a["n"] == x) for x in set(n)}
    ax.bar(list(memory), list(memory.values()), width=0.6, color=t["series"][style.GPU], edgecolor=t["surface"], linewidth=2)
    ax.axhline(32.0, color=t["text2"], ls=(0, (4, 3)), lw=1.1)
    ax.text(n.min() - 0.3, 32.0, "32 GB card", va="bottom", fontsize=8, color=t["text2"])
    for x, y in sorted(memory.items()):
        ax.text(x, y + 0.4, f"{y:.1f}", ha="center", fontsize=8.5, color=t["text"])
    ax.set_xticks(sorted(set(n)))
    ax.set_ylim(0, 35)
    ax.set_xlabel("cases in parallel")
    ax.set_ylabel("GPU memory, sum of process peaks [GiB]")
    ax.set_title("Memory", loc="left")
    ax.grid(axis="x", visible=False)

    ax = axes[2]
    has_e = [a for a in arms if "gpu_kj_per_case_h" in a]
    lo = cfg["cpu"]["package_w_12core"]["lo"] * cfg["cpu"]["wn3_3x4_s_per_case_h"]["value"] / 1000
    hi = cfg["cpu"]["package_w_12core"]["hi"] * cfg["cpu"]["wn3_3x4_s_per_case_h"]["value"] / 1000
    ax.axhspan(lo, hi, color=t["series"][style.CPU], alpha=0.25, lw=0)
    cpu_power = cfg["cpu"]["package_w_12core"]
    cpu_label = "CPU-WRF, 12 cores (maintainer measurement ≈200 W)" if cpu_power["label"] == "M" else "CPU-WRF, 12 cores [estimate]"
    if lo == hi:
        ax.axhline(lo, color=t["series"][style.CPU], lw=1.5)
    ax.text(0.98, hi, cpu_label, transform=ax.get_yaxis_transform(), va="bottom", ha="right",
            fontsize=8, color=t["text2"])
    if has_e:
        ne = np.array([a["n"] for a in has_e])
        ke = np.array([a["gpu_kj_per_case_h"] for a in has_e])
        for hours in sorted({a["hours"] for a in has_e}):
            group = [a for a in has_e if a["hours"] == hours]
            ax.plot([a["n"] for a in group], [a["gpu_kj_per_case_h"] for a in group], marker="o",
                    color=t["series"][style.GPU], label=f"GPU, {hours:g} h/case")
        ax.legend(loc="lower right")
        for x, y in zip(ne, ke):
            ax.text(x, y + 0.3, f"{y:.1f}", ha="center", fontsize=8.5, color=t["text"])
    else:
        ax.text(0.5, 0.35, "GPU power not logged in\nthis prototype arm\n(R3/R4 log it)", transform=ax.transAxes,
                ha="center", fontsize=9, color=t["muted"])
    ax.set_xticks(sorted(set(n)))
    ax.set_xlim(n.min() - 0.6, n.max() + 0.6)
    ax.set_ylim(0, hi * 1.3)
    ax.set_xlabel("cases in parallel")
    ax.set_ylabel("energy per simulated case-hour [kJ]")
    ax.set_title("Energy (lower is better)", loc="left")
    ax.grid(axis="x", visible=False)
    fig.suptitle("Several forecasts at once on one RTX 5090 vs CPU-WRF on 12 cores", x=0.01, ha="left",
                 fontsize=13, fontweight="semibold", color=t["text"])
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    style.footer(fig, f"WN3 Tenerife 3-nest, tree {cfg['tree_label']}; GPU energy = nvidia-smi board power, 1 s; "
                      f"CPU package = {cpu_power['ref']}; GPU host CPU outside this comparison.", t)
    style.save(fig, out)


def fig_history(cfg, t, out):
    base = cfg["history_baseline"]
    hist = cfg["history_s1_proxy"]
    labels = [base["label"]] + [h["label"] for h in hist]
    vals = [base["value"]] + [h["value"] for h in hist]
    cpu = cfg["cpu"]["prod_12rank_s_per_fch"]["value"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(vals))
    ax.bar(x, vals, width=0.62, color=t["series"][style.GPU], edgecolor=t["surface"], linewidth=2)
    ax.axhline(cpu, color=t["series"][style.CPU], lw=2)
    ax.text(len(vals) - 0.5, cpu, f"CPU-WRF 12 cores: {cpu} s", ha="right", va="bottom", fontsize=8.5,
            color=t["text2"])
    for xi, v in zip(x, vals):
        ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=8.5, color=t["text"])
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("s per forecast hour (warm, PROD)")
    ax.set_title("v0.25 performance work on the same PROD case (lower is faster)", loc="left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    style.footer(fig, "bench S1 proxy (3 h run, segments 2-3) per iteration tree; refs in numbers.json", t)
    style.save(fig, out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--theme", default="light", choices=["light", "dark"])
    a = ap.parse_args()
    cfg = json.loads(pathlib.Path(a.inputs).read_text())
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t = style.apply(a.theme)
    sfx = "" if a.theme == "light" else "_dark"
    nums = {"inputs": a.inputs, "tree": cfg["tree_label"]}
    nums["prod_endpoint"] = prod_summary(cfg["prod"])["endpoint"]
    nums["speed"] = fig_speed(cfg, t, str(out / f"speed{sfx}.png"))
    arms = [arm_summary(x) for x in cfg["wn3_arms"]]
    nums["wn3_arms"] = arms
    fig_parallel(cfg, arms, t, str(out / f"parallel{sfx}.png"))
    fig_history(cfg, t, str(out / f"history{sfx}.png"))
    dm = cfg["prod"].get("dmon")
    if dm and pathlib.Path(dm).exists():
        info = read_dmon(pathlib.Path(dm))
        if info["has_power"]:
            h = cfg["prod"]["whole_run_hours"]
            nums["prod_energy"] = {"gpu_board_kj_per_fch": info["energy_j"] / 1000 / h, "mean_w": info["mean_w"],
                                   "label": "M", "ref": dm,
                                   "cpu_kj_per_fch_lo": cfg["cpu"]["package_w_12core"]["lo"] * cfg["cpu"]["prod_12rank_s_per_fch"]["value"] / 1000,
                                   "cpu_kj_per_fch_hi": cfg["cpu"]["package_w_12core"]["hi"] * cfg["cpu"]["prod_12rank_s_per_fch"]["value"] / 1000,
                                   "cpu_label": "I"}
    (out / "numbers.json").write_text(json.dumps(nums, indent=1))
    print(json.dumps(nums, indent=1)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
