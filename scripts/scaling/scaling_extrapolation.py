#!/usr/bin/env python3
"""Generate the inferred H100/B200/B300 scenarios in docs/release/SCALING_METHOD.md.
Central scales device work and preserves the fixed-cost proxy; the optimistic
upper scenario scales the whole run at the historical central efficiency.
RTX board energy is timestamp-measured; all host shares/combined energy are [I].
B200 was measured on earlier wrf_gpu versions; H100/B200/B300 values are
extrapolated to v0.3.1, not measured."""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

# --- shared look of the v0.3 release plots (palette from release-docs scripts/release_plots/style.py) ---
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "muted": "#8a8984",
              "grid": "#e4e3df", "host": "#cde2fb",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "text2": "#c3c2b7", "muted": "#8f8e86",
             "grid": "#33332f", "host": "#184f95",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]},
}
GPU, CPU = 0, 1

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def apply_theme(theme: str, base: float = 15.0) -> dict:
    t = THEMES[theme]
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "axes.edgecolor": t["grid"], "axes.labelcolor": t["text2"], "axes.titlecolor": t["text"],
        "axes.titlesize": base + 1.0, "axes.titleweight": "semibold", "axes.labelsize": base,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": t["grid"], "grid.linewidth": 0.9,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": t["text2"], "ytick.color": t["text2"],
        "xtick.labelsize": base - 1.0, "ytick.labelsize": base - 1.0,
        "text.color": t["text"], "legend.frameon": False, "legend.fontsize": base - 2.0,
        "font.family": "DejaVu Sans", "lines.linewidth": 2.2, "lines.solid_capstyle": "round",
    })
    return t


# ------------------------------------------------------------------ model inputs (with sources)
VENDOR = {
    "H100": dict(hbm_gb=80.0, bw_tbs=3.35, tdp_w=700.0, mem="80 GB HBM3"),
    "B200": dict(hbm_gb=192.0, bw_tbs=8.0, tdp_w=1000.0, mem="192 GB HBM3e"),
    "B300": dict(hbm_gb=288.0, bw_tbs=8.0, tdp_w=1400.0, mem="288 GB HBM3e"),
}
RTX = dict(hbm_gb=32.0, bw_tbs=1.792, tdp_w=575.0, mem="32 GB GDDR7")

ANCHOR = dict(
    n4_whole_s_per_case_h=7.774, n4_steady_s_per_case_h=4.6076, n4_power_w=322.0,
    n3_whole_s_per_case_h=8.201, n1_whole_s_per_case_h=13.668, power_w=[206.0, 305.0, 322.0],
    per_case_vram_mib=5354.0, host_ram_gb_per_case=6.1,
    cpu_s_per_case_h=123.4, cpu_load_w=200.0,
)
OLD_B200_5090_R = dict(lo_small_grid=2.69, r_inf=3.53, hi=3.61, bw_cap=8.0 / 1.792)
EFF = dict(lo=OLD_B200_5090_R["lo_small_grid"] / OLD_B200_5090_R["bw_cap"],
           mid=OLD_B200_5090_R["r_inf"] / OLD_B200_5090_R["bw_cap"], hi=1.0)
PWR_FRAC = dict(lo=0.56, mid=0.625, hi=0.69)
# host share: the 4 CPU cores driving the GPU, taken at the owner-reported 200 W/12-core load rate
HOST_W = ANCHOR["cpu_load_w"] * 4.0 / 12.0     # ~= 66.7 W
MARGIN_GIB = 1.0


def build(args) -> dict:
    a = ANCHOR
    t_whole = args.n4_whole or a["n4_whole_s_per_case_h"]
    t_steady = args.n4_steady or a["n4_steady_s_per_case_h"]
    t_fixed = t_whole - t_steady
    case_gib = args.case_vram_mib / 1024.0

    def rho(gpu): return VENDOR[gpu]["bw_tbs"] / RTX["bw_tbs"]
    # Central preserves measured fixed cost; optimistic edge scales the whole
    # run at the historical central efficiency (manager 2026-10-05 10:15Z).
    def t_opt(gpu): return t_whole / (rho(gpu) * EFF["mid"])
    def t_mid(gpu): return t_fixed + t_steady / (rho(gpu) * EFF["mid"])
    def t_cons(gpu): return t_fixed + t_steady / (rho(gpu) * EFF["lo"])

    def cases_fit(gpu):
        cap = (VENDOR[gpu]["hbm_gb"] if gpu in VENDOR else RTX["hbm_gb"]) * 1e9
        return max(int(cap * 0.90 // ((case_gib + MARGIN_GIB) * 2**30)), 1), max(int(cap // ((case_gib + MARGIN_GIB) * 2**30)), 1)

    def host_kj(t): return HOST_W * t / 1000.0
    cpu_kj = a["cpu_load_w"] * a["cpu_s_per_case_h"] / 1000.0

    cf5090 = [int(RTX["hbm_gb"] * 1e9 * 0.90 // ((case_gib + MARGIN_GIB) * 2**30)),
              int(RTX["hbm_gb"] * 1e9 // (case_gib * 2**30))]
    energy_path = pathlib.Path(__file__).resolve().parents[2] / "docs/release/evidence/v031/energy_samples.json"
    observed_energy = json.loads(energy_path.read_text())["phases"]["n4"]
    board5090 = observed_energy["board_j_per_case_forecast_hour"] / 1000.0
    out = {
        "label": "extrapolated",
        "caption": ("B200 was measured on earlier wrf_gpu versions; H100/B200/B300 values are "
                    "extrapolated to v0.3.1, not measured."),
        "method": __doc__,
        "anchors": a | dict(rtx_5090=RTX, fixed_s_per_case_h=round(t_fixed, 4)),
        "old_version": OLD_B200_5090_R, "efficiency": EFF, "power_fraction": PWR_FRAC,
        "host_share": {"w": round(HOST_W, 1), "assumption": "4 of 12 CPU cores at the owner-reported 200 W/12-core load rate"},
        "vendor": VENDOR, "rtx_5090": RTX,
        "capacity": {"per_case_gib": round(case_gib, 4), "extra_growth_margin_gib": MARGIN_GIB,
                     "usable_fraction_band": [0.90, 1.0], "note": "per-case = measured per-PID peak; "
                     "host RAM " + str(a["host_ram_gb_per_case"]) + " GB per warm case"},
        "cpu": {"s_per_case_h": a["cpu_s_per_case_h"], "load_w": a["cpu_load_w"],
                "case_h_per_wall_h": 3600.0 / a["cpu_s_per_case_h"], "kj_per_case_h": cpu_kj,
                "timing_label": "M", "energy_label": "I", "label": "I",
                "power_source": "owner-reported power for the 12-core CPU-WRF run"},
        "rtx5090_measured": {
            "case_h_per_wall_h_whole": 3600.0 / t_whole, "s_per_case_h_whole": t_whole,
            "power_w": {"board": a["n4_power_w"], "host": round(HOST_W, 1), "total": round(a["n4_power_w"] + HOST_W, 1)},
            "kj_per_case_h": {"central": round(board5090 + host_kj(t_whole), 3), "board": round(board5090, 3),
                              "host": round(host_kj(t_whole), 3), "band": [round(board5090 + host_kj(t_whole), 3)] * 2},
            "cases_fit": cf5090, "cases_fit_note": "host RAM {} GB".format(round(cf5090[0] * a["host_ram_gb_per_case"])),
            "speedup_vs_cpu": (3600.0 / t_whole) / (3600.0 / a["cpu_s_per_case_h"]),
            "energy_vs_cpu": cpu_kj / (board5090 + host_kj(t_whole)), "label": "M",
            "energy_board_label": "M", "energy_host_label": "I", "energy_total_label": "I",
            "energy_board_source": "docs/release/evidence/v031/energy_samples.json (E158 timestamps)"},
        "gpus": {},
    }
    for gpu in ("H100", "B200", "B300"):
        t_o, t_m, t_c = t_opt(gpu), t_mid(gpu), t_cons(gpu)
        tp_c, tp_m, tp_o = 3600 / t_c, 3600 / t_m, 3600 / t_o
        b_lo = PWR_FRAC["lo"] * VENDOR[gpu]["tdp_w"] * t_o / 1000.0
        b_mid = PWR_FRAC["mid"] * VENDOR[gpu]["tdp_w"] * t_m / 1000.0
        b_hi = PWR_FRAC["hi"] * VENDOR[gpu]["tdp_w"] * t_c / 1000.0
        e_lo, e_hi = b_lo + host_kj(t_o), b_hi + host_kj(t_c)
        e_mid = b_mid + host_kj(t_m)
        cf_lo, cf_hi = cases_fit(gpu)
        out["gpus"][gpu] = {
            "label": "I", "bw_ratio_rho": round(rho(gpu), 3), "cases_fit_band": [cf_lo, cf_hi], "cases_fit": cf_lo,
            "physical_capacity_gb_scenario": VENDOR[gpu]["hbm_gb"],
            "packing_budget_gb": VENDOR[gpu]["hbm_gb"] * 0.90,
            "cases_fit_note": "host RAM {} GB".format(round(cf_lo * a["host_ram_gb_per_case"])),
            "throughput_case_h_per_wall_h": {"central": round(tp_m, 1), "band": [round(tp_c, 1), round(tp_o, 1)]},
            "s_per_case_h_whole": {"central": round(t_m, 3), "band": [round(t_o, 3), round(t_c, 3)]},
            "speedup_vs_cpu": round(tp_m / (3600.0 / a["cpu_s_per_case_h"]), 1),
            "kj_per_case_h": {"central": round(e_mid, 3), "board": round(b_mid, 3), "host": round(host_kj(t_m), 3),
                              "band": [round(e_lo, 3), round(e_hi, 3)]},
            "power_w": {"board_central": round(PWR_FRAC["mid"] * VENDOR[gpu]["tdp_w"], 1),
                        "board_band": [round(PWR_FRAC["lo"] * VENDOR[gpu]["tdp_w"], 1),
                                       round(PWR_FRAC["hi"] * VENDOR[gpu]["tdp_w"], 1)],
                        "host": round(HOST_W, 1)},
            "energy_vs_cpu": round(cpu_kj / e_mid, 1)}
    def wh(value):
        if isinstance(value, dict): return {k: wh(v) for k, v in value.items()}
        if isinstance(value, list): return [wh(v) for v in value]
        return value / 3.6
    out['cpu']['wh_per_case_h'] = out['cpu']['kj_per_case_h'] / 3.6
    for point in [out['rtx5090_measured'], *out['gpus'].values()]:
        point['wh_per_case_h'] = wh(point['kj_per_case_h'])
    return out


# ----- per-device values -----
NAMES = lambda d: ["CPU-WRF\n12 cores\n(3 x 4 ranks)", "RTX 5090\n(measured)\n4 cases",
                   f"H100 SXM\n{d['gpus']['H100']['cases_fit']} cases fit",
                   f"B200\n{d['gpus']['B200']['cases_fit']} cases fit",
                   f"B300\n{d['gpus']['B300']['cases_fit']} cases fit"]
DEV = ["cpu", "rtx5090_measured", "H100", "B200", "B300"]


def _panel(ax, d, kind, t, base):
    x = np.arange(len(DEV)); x.setflags(write=True)
    cpu_col = t["series"][CPU]
    for xi, k in enumerate(DEV):
        ext = k in ("H100", "B200", "B300")
        col = cpu_col if k == "cpu" else t["series"][GPU]
        if k == "cpu":
            v = d["cpu"]["case_h_per_wall_h"] if kind == "throughput" else d["cpu"]["wh_per_case_h"]
            ax.bar(xi, v, width=0.64, color=col, edgecolor=t["surface"], linewidth=2.0, zorder=3)
            lo = hi = v
        elif k == "rtx5090_measured":
            m = d["rtx5090_measured"]
            if kind == "throughput":
                v = lo = hi = m["case_h_per_wall_h_whole"]
                ax.bar(xi, v, width=0.64, color=col, edgecolor=t["surface"], linewidth=2.0, zorder=3)
            else:
                v = lo = hi = m["wh_per_case_h"]["central"]
                b, h = m["wh_per_case_h"]["board"], m["wh_per_case_h"]["host"]
                ax.bar(xi, b, width=0.64, color=col, edgecolor=t["surface"], linewidth=2.0, zorder=3)
                ax.bar(xi, h, width=0.64, bottom=b, color=t["host"], edgecolor=t["text2"], hatch="//", linewidth=1.4, zorder=3)
        else:
            g = d["gpus"][k]
            if kind == "throughput":
                v = g["throughput_case_h_per_wall_h"]["central"]; lo, hi = g["throughput_case_h_per_wall_h"]["band"]
                ax.bar(xi, v, width=0.64, facecolor=t["surface"], edgecolor=col, hatch="//", linewidth=2.0, zorder=3)
            else:
                v = g["wh_per_case_h"]["central"]; lo, hi = g["wh_per_case_h"]["band"]
                b, h = g["wh_per_case_h"]["board"], g["wh_per_case_h"]["host"]
                ax.bar(xi, b, width=0.64, facecolor=t["surface"], edgecolor=col, hatch="//", linewidth=2.0, zorder=3)
                ax.bar(xi, h, width=0.64, bottom=b, color=t["host"], edgecolor=col, hatch="//", linewidth=1.2, zorder=3)
        if hi != lo:
            ax.errorbar([xi], [v], yerr=[[v - lo], [hi - v]], color=t["text2"], capsize=5, lw=1.4, zorder=4)
        ax.text(xi, max(hi, v) * 1.09, f"{v:,.0f}" if kind == "throughput" else f"{v:.2f}",
                ha="center", va="bottom", fontsize=base, color=t["text"])
    ref = d["cpu"]["case_h_per_wall_h"] if kind == "throughput" else d["cpu"]["wh_per_case_h"]
    ax.axhline(ref, color=cpu_col, lw=1.1, ls=(0, (4, 3)), alpha=0.65, zorder=1)
    ax.set_yscale("log"); ax.set_xticks(x, NAMES(d)); ax.grid(axis="x", visible=False)
    if kind == "throughput":
        top = max(d["gpus"][g]["throughput_case_h_per_wall_h"]["band"][1] for g in ("H100", "B200", "B300")) * 1.7
        ax.set_ylim(15, top)
    else:
        ax.set_ylim(0.62 / 3.6, d["cpu"]["wh_per_case_h"] * 1.6)


CAPTION = ("Caption: B200 was measured on earlier wrf_gpu versions;\n"
           "H100/B200/B300 values are extrapolated to v0.3.1, not measured.\n")
FOOT_TP = (CAPTION +
           "Central keeps fixed 3.17 s/case-h; upper band scales the whole run optimistically. [I]\n"
           "B200 packing budget 172.8 GB; 38 cases need ~230 GB host RAM.\n"
           "Method: SCALING_METHOD.md; observed anchors and inferred scenarios remain separate.")
FOOT_EN = (CAPTION +
           "5090 board [M] + ~67 W host share [I] (4/12 of 200 W); combined totals are [I].\n"
           "CPU: owner-reported 200 W / 12-core run. No package or whole-node energy claim.")


def _standalone(d, kind, theme, path, size, base):
    t = apply_theme(theme, base)
    fig = plt.figure(figsize=(size[0] / 150.0, size[1] / 150.0), dpi=150)
    ax = fig.add_axes([0.115, 0.44, 0.868, 0.39])
    _panel(ax, d, kind, t, base)
    if kind == "throughput":
        ax.set_ylabel("case-forecast-hours\nper wall-hour", fontsize=base - 1.0)
        title = f"Throughput per GPU \u2014 B200/B300 \u2248 {d['gpus']['B200']['speedup_vs_cpu']:.1f}x CPU [I]"
        sub = (f"RTX 5090 = {d['rtx5090_measured']['speedup_vs_cpu']:.1f}x a 12-core node [M]   \u00b7   "
               f"[M] measured, [I] extrapolated (hatched)")
        foot = FOOT_TP
    else:
        ax.set_ylabel("Wh / case-forecast-hour\n(board + host share)", fontsize=base - 1.0)
        fig.legend(handles=_legend(t, base), loc="center", bbox_to_anchor=(0.56, 0.255), ncol=3, fontsize=14.0, frameon=False)
        title = f"GPU board + host share \u2014 B200 \u2248 {d['gpus']['B200']['energy_vs_cpu']:.1f}x less energy [I]"
        sub = (f"RTX 5090 board [M] + host [I] = {d['rtx5090_measured']['energy_vs_cpu']:.1f}x less [I]\n"
               f"GPU = board + host share; CPU = owner-reported 12-core load power")
        foot = FOOT_EN
    fig.text(0.012, 0.968, title, fontsize=base + 2.0, fontweight="semibold", color=t["text"], va="top")
    fig.text(0.012, 0.912, sub, fontsize=base - 1.0, color=t["text2"], va="top")
    fig.text(0.012, 0.012, foot, fontsize=14.0, color=t["muted"], va="bottom", linespacing=1.25)
    fig.savefig(path, dpi=150)
    return fig, t


def _legend(t, base):
    from matplotlib.patches import Patch
    return [Patch(facecolor=t["series"][GPU], edgecolor=t["surface"], label="GPU board"),
            Patch(facecolor=t["host"], edgecolor=t["text2"], hatch="//", linewidth=0.8, label=f"Host share [I]"),
            Patch(facecolor=t["series"][CPU], edgecolor=t["surface"], label="CPU energy [I]")]


def draw_throughput(d, theme, path, size=(1400, 800), base=15.0):
    fig, _ = _standalone(d, "throughput", theme, path, size, base); plt.close(fig)


def draw_energy(d, theme, path, size=(1400, 800), base=15.0):
    fig, _ = _standalone(d, "energy", theme, path, size, base); plt.close(fig)


def draw_combined(d, theme, path, size=(2300, 760), base=11.5):
    t = apply_theme(theme, base)
    fig = plt.figure(figsize=(size[0] / 150.0, size[1] / 150.0), dpi=150)
    ax1 = fig.add_axes([0.075, 0.30, 0.40, 0.50]); ax2 = fig.add_axes([0.585, 0.30, 0.40, 0.50])
    _panel(ax1, d, "throughput", t, base - 1.5); ax1.set_ylabel("case-forecast-hours per wall-hour")
    _panel(ax2, d, "energy", t, base - 1.5); ax2.set_ylabel("Wh per case-hour (board + host)")
    ax2.legend(handles=_legend(t, base), loc="upper right", fontsize=base - 3.0)
    fig.text(0.008, 0.966, "wrf_gpu v0.3.1 scaling to data-centre GPUs \u2014 B200 was measured on earlier versions; "
             "H100/B200/B300 are extrapolated to v0.3.1, not measured", fontsize=base + 1.5, fontweight="semibold", color=t["text"], va="top")
    fig.text(0.008, 0.905, f"Throughput: B200/B300 \u2248 {d['gpus']['B200']['speedup_vs_cpu']:.0f}x a 12-core node [I] "
             f"(RTX 5090 {d['rtx5090_measured']['speedup_vs_cpu']:.1f}x [M]).   Energy: B200 \u2248 {d['gpus']['B200']['energy_vs_cpu']:.0f}x less [I] "
             f"(RTX 5090 {d['rtx5090_measured']['energy_vs_cpu']:.1f}x less [M]); GPU = board + host share.",
             fontsize=base - 1.0, color=t["text2"], va="top")
    fig.text(0.008, 0.015, "Bars [M] measured / [I] extrapolated (hatched); whisker = [I] band.  GPU energy = board + host share (4 of 12 CPU cores ~67 W); CPU = 12-core package ~200 W.\n"
             "Cases fit = usable VRAM / per-case (5354 MiB); host RAM ~6.1 GB per warm case \u2192 38 cases ~ 230 GB.  Anchors [M]: wn3 V31 sweep N=4 7.77 s/case-h, 322 W; CPU 123.4 s/h @200 W.  Numbers: scaling_extrapolation.json.",
             fontsize=base - 3.0, color=t["muted"], va="bottom", linespacing=1.5)
    fig.savefig(path, dpi=150); plt.close(fig)


def check_overlaps(d, theme="light", size=(1400, 800), base=15.0):
    """Build the real standalone figures and report out-of-canvas text or collisions."""
    rep = {}
    for kind in ("throughput", "energy"):
        fig, t = _standalone(d, kind, theme, "/tmp/_scale_plot_check.png", size, base)
        fig.canvas.draw(); r = fig.canvas.get_renderer(); W, H = fig.canvas.get_width_height()
        ax = fig.axes[0]
        issues = []
        foot = fig.texts[2] if len(fig.texts) > 2 else fig.texts[-1]
        fb = foot.get_window_extent(r)
        ylo, yhi = ax.get_ylim()
        yticks = [l for l in ax.get_yticklabels() if l.get_text() and ylo <= l.get_position()[1] <= yhi]
        for lab in list(ax.get_xticklabels()) + yticks + [ax.yaxis.label]:
            b = lab.get_window_extent(r)
            if b.x0 < -1 or b.y0 < -1 or b.x1 > W + 1 or b.y1 > H + 1:
                issues.append(f"'{lab.get_text()[:26]}' out of canvas ({b.x0:.0f},{b.y0:.0f},{b.x1:.0f},{b.y1:.0f}) vs {W}x{H}")
            if fb.overlaps(b):
                issues.append(f"footnote overlaps '{lab.get_text()[:24]}'")
        for ft in fig.texts:
            b = ft.get_window_extent(r)
            if b.x0 < -1 or b.y0 < -1 or b.x1 > W + 1 or b.y1 > H + 1:
                issues.append(f"fig-text '{ft.get_text()[:34]}' out of canvas x1={b.x1:.0f} vs {W}")
        leg = ax.get_legend()
        for tx in ax.texts:
            b = tx.get_window_extent(r)
            if leg is not None and leg.get_window_extent(r).overlaps(b):
                issues.append(f"legend overlaps value '{tx.get_text()}'")
        rep[kind] = issues or ["OK"]
        plt.close(fig)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--n4-whole", type=float, default=None)
    ap.add_argument("--n4-steady", type=float, default=None)
    ap.add_argument("--case-vram-mib", type=float, default=None)
    a = ap.parse_args()
    if a.case_vram_mib is None:
        a.case_vram_mib = ANCHOR["per_case_vram_mib"]
    d = build(a)
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for theme, sfx in (("light", ""), ("dark", "_dark")):
        draw_throughput(d, theme, str(out / f"v031_scaling_throughput{sfx}.png"))
        draw_energy(d, theme, str(out / f"v031_scaling_energy{sfx}.png"))
        draw_combined(d, theme, str(out / f"v031_scaling_extrapolation{sfx}.png"))
    (pathlib.Path(a.json) if a.json else out / "scaling_extrapolation.json").write_text(json.dumps(d, indent=1))
    c = d["cpu"]; m = d["rtx5090_measured"]
    print(f"CPU 12c : {c['case_h_per_wall_h']:.1f} case-h/h   {c['kj_per_case_h']:.2f} kJ/case-h [M]")
    print(f"RTX 5090: {m['case_h_per_wall_h_whole']:.1f} case-h/h   {m['kj_per_case_h']['central']:.2f} kJ/case-h "
          f"[M]  ({m['speedup_vs_cpu']:.1f}x CPU / {m['energy_vs_cpu']:.1f}x less energy)")
    for g, v in d["gpus"].items():
        print(f"{g:8s}: {v['throughput_case_h_per_wall_h']['central']:7.1f} case-h/h "
              f"[{v['throughput_case_h_per_wall_h']['band'][0]:.0f}-{v['throughput_case_h_per_wall_h']['band'][1]:.0f}]  "
              f"{v['kj_per_case_h']['central']:.2f} kJ/case-h [{v['kj_per_case_h']['band'][0]:.2f}-{v['kj_per_case_h']['band'][1]:.2f}]  "
              f"rho={v['bw_ratio_rho']} fit={v['cases_fit']}  {v['speedup_vs_cpu']:.0f}x CPU / {v['energy_vs_cpu']:.1f}x less energy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
