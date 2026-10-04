#!/usr/bin/env python3
"""B200 extrapolation for today's code, anchored on the OLD-version B200 + RTX 5090 measurements.

Every output number is labelled "extrapolated from B200 runs on old version" ([I]).

Sources (old version, v0.20 era, measured 2026-06-23, fp64, single-domain Swiss base state tiled 128^2..1024^2 x 44 lev,
boundary/GWD/Noah-MP off):
  B200  : git 43379efb5:publication/paper.tex tab:perf (L522-540) = PAPER_DATA_INVENTORY.md L99-126 (raw dir gone)
  5090  : proofs/v020/benchmark/T2T3_REPORT.md L28-36 (same version, same ladder)
Method:
  r = B200/5090 speed ratio measured on the old code (same grid): 2.69 (128^2), 3.15 (256^2), 3.61 (384^2); R_inf 3.53.
  The old code ran fp32 ~ fp64 on the 5090 (no fp64-FLOP limit) -> r is a memory-bandwidth/latency-class ratio; today's
  fp32 code is bandwidth-bound too (A54: ~77 % of the HBM floor) -> apply r to today's SATURATED 5090 throughput.
  Upper cap = HBM bandwidth ratio 8.0 / 1.792 TB/s = 4.46. Low end = 2.69 (small-grid ratio).
  Cases per GPU = conservative 180 decimal GB budget / (per-case peak in bytes + 1 GiB growth margin).
  The process peak already includes context/modules; the extra GiB is a portability margin.
  Cases to saturate = N_sat(5090) x r (assumed, not demonstrated on today's B200 code).
  Energy = old-ladder B200 board power at saturation (653-685 W) x extrapolated s per case-hour.
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

OLD = {
    "b200_s_per_fch": {"128": 12.12, "256": 31.95, "384": 65.84, "512": 112.84, "768": 250.62, "896": 356.94, "1024": 475.97},
    "b200_cell_updates_s": {"128": 2.14e7, "256": 3.25e7, "384": 3.55e7, "512": 3.68e7, "768": 3.73e7, "896": 3.56e7, "1024": 3.49e7},
    "b200_board_w": {"128": 453, "256": 582, "384": 617, "512": 653, "768": 679, "896": 685, "1024": 678},
    "b200_r_inf": 3.74e7,
    "rtx5090_fp64_s_per_fch": {"128": 32.6, "256": 100.8, "384": 237.5},
    "rtx5090_r_inf_fp64": 1.06e7,
    "ref_b200": "git 43379efb5:publication/paper.tex tab:perf L522-540 (2026-07-09; ladder measured 2026-06-23, v0.20 era)",
    "ref_5090": "proofs/v020/benchmark/T2T3_REPORT.md L28-36",
}
HBM_TBS = {"rtx5090": 1.792, "b200": 8.0}
B200_MEMORY_BYTES = 180 * 10**9  # NVIDIA 180 GB/GPU; conservative decimal interpretation

# WN3 Tenerife 3-nest: (nx-1)*(ny-1)*(nz-1) mass cells x steps per hour (dt 54/18/6 s)
WN3_CELLS = {"d01": 120 * 70 * 44, "d02": 267 * 117 * 44, "d03": 111 * 93 * 44}
WN3_STEPS_PER_H = {"d01": 3600 / 54, "d02": 3600 / 18, "d03": 3600 / 6}


def wn3_cell_updates_per_case_h() -> float:
    return sum(WN3_CELLS[d] * WN3_STEPS_PER_H[d] for d in WN3_CELLS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sat-stepping-s-per-case-h", type=float, required=True,
                    help="today's saturated RTX 5090 WN3 throughput (stepping), s per case-hour [M]")
    ap.add_argument("--sat-whole-s-per-case-h", type=float, required=True, help="same arm, whole run incl. start-up [M]")
    ap.add_argument("--sat-n", type=int, default=3, help="N at which the 5090 saturates [M]")
    ap.add_argument("--rtx-board-w", type=float, required=True, help="5090 board power at saturation [M or I]")
    ap.add_argument("--rtx-board-w-label", default="M")
    ap.add_argument("--case-vram-gib", type=float, required=True, help="per-case peak GPU memory (per PID) [M]")
    ap.add_argument("--cpu-s-per-case-h", type=float, default=123.38, help="CPU-WRF 3x4-core throughput [M]")
    ap.add_argument("--cpu-w-lo", type=float, default=200.0)
    ap.add_argument("--cpu-w-hi", type=float, default=200.0)
    ap.add_argument("--cpu-power-label", default="M: maintainer measurement")
    ap.add_argument("--timing-method", default="stepping timers")
    ap.add_argument("--tree", default="LW9")
    ap.add_argument("--theme", default="light", choices=["light", "dark"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()

    cu_case_h = wn3_cell_updates_per_case_h()
    rtx_cu_s = cu_case_h / a.sat_stepping_s_per_case_h
    r = {"lo": OLD["rtx5090_fp64_s_per_fch"]["128"] / OLD["b200_s_per_fch"]["128"],
         "mid": OLD["b200_r_inf"] / OLD["rtx5090_r_inf_fp64"],
         "hi_measured": OLD["rtx5090_fp64_s_per_fch"]["384"] / OLD["b200_s_per_fch"]["384"],
         "cap_bandwidth": HBM_TBS["b200"] / HBM_TBS["rtx5090"]}
    sat_w = float(np.mean([OLD["b200_board_w"][k] for k in ("512", "768", "896", "1024")]))
    # startup overhead per case-hour at the measured arm (whole - stepping) is host-side: keep it unscaled
    startup = a.sat_whole_s_per_case_h - a.sat_stepping_s_per_case_h

    def b200(rr):
        step = a.sat_stepping_s_per_case_h / rr
        return {"r": rr, "stepping_s_per_case_h": step, "whole_s_per_case_h_unscaled_startup": step + startup,
                "case_h_per_h_stepping": 3600 / step, "cell_updates_s": rtx_cu_s * rr,
                "kj_per_case_h": sat_w * step / 1000}
    ext = {k: b200(v) for k, v in r.items()}
    cases_fit = int(B200_MEMORY_BYTES // ((a.case_vram_gib + 1.0) * 2**30))
    out = {
        "label": "extrapolated from B200 runs on old version",
        "method": __doc__,
        "inputs": vars(a),
        "old_version": OLD,
        "hbm_tb_s": HBM_TBS,
        "capacity_assumptions": {"budget_bytes": B200_MEMORY_BYTES, "gb_is_decimal": True,
                                 "per_case_process_peak_gib": a.case_vram_gib, "extra_growth_margin_gib": 1.0,
                                 "note": "conservative capacity scenario; B200 allocator peak unmeasured"},
        "wn3_cell_updates_per_case_h": cu_case_h,
        "rtx5090_today": {"cell_updates_s_stepping": rtx_cu_s, "case_h_per_h_stepping": 3600 / a.sat_stepping_s_per_case_h,
                          "case_h_per_h_whole": 3600 / a.sat_whole_s_per_case_h,
                          "kj_per_case_h": a.rtx_board_w * a.sat_stepping_s_per_case_h / 1000,
                          "speedup_vs_old_version_r_inf": rtx_cu_s / OLD["rtx5090_r_inf_fp64"],
                          "throughput_label": "M", "energy_label": "M" if a.rtx_board_w_label == "M" else "I",
                          "timing_method": a.timing_method},
        "b200": ext,
        "b200_board_w_saturated_old": sat_w,
        "b200_cases_fit_vram": cases_fit,
        "b200_cases_to_saturate": {k: a.sat_n * v for k, v in r.items()},
        "cpu": {"s_per_case_h": a.cpu_s_per_case_h, "case_h_per_h": 3600 / a.cpu_s_per_case_h,
                "kj_per_case_h_lo": a.cpu_w_lo * a.cpu_s_per_case_h / 1000,
                "kj_per_case_h_hi": a.cpu_w_hi * a.cpu_s_per_case_h / 1000, "energy_label": a.cpu_power_label},
    }
    pathlib.Path(a.json).write_text(json.dumps(out, indent=1))

    t = style.apply(a.theme)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    names = ["CPU-WRF\n12 cores", "RTX 5090\n(measured)", "B200\n(extrapolated)"]
    tp = [out["cpu"]["case_h_per_h"], out["rtx5090_today"]["case_h_per_h_stepping"], ext["mid"]["case_h_per_h_stepping"]]
    lo = ext["lo"]["case_h_per_h_stepping"]
    hi = ext["cap_bandwidth"]["case_h_per_h_stepping"]
    cols = [t["series"][style.CPU], t["series"][style.GPU], t["series"][style.GPU]]
    ax = axes[0]
    x = np.arange(3)
    bars = ax.bar(x, tp, width=0.6, color=cols, edgecolor=t["surface"], linewidth=2)
    bars[2].set_hatch("//")
    bars[2].set_facecolor(t["surface"])
    bars[2].set_edgecolor(t["series"][style.GPU])
    ax.errorbar([2], [tp[2]], yerr=[[tp[2] - lo], [hi - tp[2]]], color=t["text2"], capsize=6, lw=1.3)
    for xi, v in zip(x, tp):
        ax.text(xi + 0.33, v, f"{v:,.0f}", va="center", fontsize=9, color=t["text"])
    ax.set_xticks(x, names)
    ax.set_ylabel("WN3 case-hours simulated per wall-hour")
    ax.set_title("Throughput per device (steady rate)", loc="left")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    en = [None, out["rtx5090_today"]["kj_per_case_h"], ext["mid"]["kj_per_case_h"]]
    ax.bar([0], [out["cpu"]["kj_per_case_h_hi"]], width=0.6, color=t["series"][style.CPU], alpha=0.35,
           edgecolor=t["surface"], linewidth=2)
    ax.bar([0], [out["cpu"]["kj_per_case_h_lo"]], width=0.6, color=t["series"][style.CPU], edgecolor=t["surface"],
           linewidth=2)
    cpu_scope = "measured" if a.cpu_power_label == "M" else "estimate"
    ax.text(0.33, out["cpu"]["kj_per_case_h_hi"], f"{out['cpu']['kj_per_case_h_lo']:.0f}–{out['cpu']['kj_per_case_h_hi']:.0f}\n[{cpu_scope}]",
            va="center", fontsize=8.5, color=t["text"])
    b1 = ax.bar([1], [en[1]], width=0.6, color=t["series"][style.GPU], edgecolor=t["surface"], linewidth=2)
    b2 = ax.bar([2], [en[2]], width=0.6, color=t["surface"], edgecolor=t["series"][style.GPU], hatch="//", linewidth=2)
    elo, ehi = ext["cap_bandwidth"]["kj_per_case_h"], ext["lo"]["kj_per_case_h"]
    ax.errorbar([2], [en[2]], yerr=[[en[2] - elo], [ehi - en[2]]], color=t["text2"], capsize=6, lw=1.3)
    for xi, v in [(1, en[1]), (2, en[2])]:
        label = out["rtx5090_today"]["energy_label"] if xi == 1 else "I"
        ax.text(xi + 0.33, v, f"{v:.1f} [{label}]", va="center", fontsize=9, color=t["text"])
    ax.set_xticks(x, names)
    ax.set_ylabel("GPU board / CPU package energy per case-hour [kJ]")
    ax.set_title("Energy per simulated case-hour", loc="left")
    ax.grid(axis="x", visible=False)
    fig.suptitle("Scaling to data-centre GPUs — B200 bars are extrapolated from B200 runs on old version",
                 x=0.01, ha="left", fontsize=12.5, fontweight="semibold", color=t["text"])
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    style.footer(fig, (f"B200 = today's RTX 5090 throughput ({a.tree}, N={a.sat_n} saturated) × old-version measured B200/5090 "
                       f"ratio {r['mid']:.2f} (bar; whisker {r['lo']:.2f}–{r['cap_bandwidth']:.2f} = small-grid ratio … HBM "
                       f"bandwidth cap); {cases_fit} cases fit 180 GB; B200 power {sat_w:.0f} W from the old ladder. "
                       f"RTX 5090 power {a.rtx_board_w:.0f} W [{a.rtx_board_w_label}]."), t)
    style.save(fig, a.out)
    print(json.dumps({k: out[k] for k in ("rtx5090_today", "b200_cases_fit_vram", "b200_cases_to_saturate")}, indent=1))
    print({k: round(v["case_h_per_h_stepping"]) for k, v in ext.items()}, {k: round(v["kj_per_case_h"], 2) for k, v in ext.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
