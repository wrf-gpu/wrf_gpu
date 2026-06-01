#!/usr/bin/env python3
"""Render the v0.1.0 paper PNG figures (task #59).

PURE RENDERING — no new science. Each figure is built from the cited evidence
table / proof JSON / idealized PPM. Where an underlying number is not present in
the committed evidence, the structure is rendered with the values that ARE
present and a gap note is added on the figure itself (rather than inventing data).

Run CPU-only:  taskset -c 0-3 python3 publish/figures/render_paper_figures.py
Outputs:       publish/figures/*.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
IDEAL = os.path.join(HERE, "idealized")

PASS_G = "#1b7837"   # green
WARN_O = "#d8a000"   # amber
FAIL_R = "#b2182b"   # red
INK = "#1a1a1a"
MUTE = "#555555"

plt.rcParams.update({
    "font.size": 10,
    "font.family": "DejaVu Sans",
    "axes.edgecolor": INK,
    "savefig.dpi": 150,
})


def save(fig, name):
    out = os.path.join(HERE, name)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[ok] {name}")


# ---------------------------------------------------------------------------
# 1. model_role_timeline.png  (§3.2, from ai_process_ledger.md / effort_accounting.md)
# ---------------------------------------------------------------------------
def fig_model_role_timeline():
    # Stages on x-axis; each model gets a horizontal band; the cell color/label
    # encodes the role that model played in that stage (ledger §3.5 + ai_process_ledger).
    stages = [
        "(a) Foundations\nM0-M7\n~195 sprints",
        "(b) F7 dycore\nrewrite\n~26",
        "(c) Phase-B\nphysics M8-M17\n~19",
        "(d) M19\nskill/wind\n~12",
        "(e) Perf\n~15",
        "(f) v0.1.0\nfinish",
    ]
    models = ["GPT-5.5 Pro", "Opus 4.7", "Opus 4.8", "GPT-5.5 (codex)", "Gemini 3.5 (agy)"]
    # role per (model, stage); "" = not active.  M=manager, F=frontrunner,
    # V=verifier, C=critic, T=tiebreak, S=scaffold/foundations
    role_color = {
        "M": ("Manager", "#2166ac"),
        "F": ("Frontrunner", "#1b7837"),
        "V": ("Verifier", "#8073ac"),
        "C": ("Critic", "#d6604d"),
        "T": ("Tiebreak", "#d8a000"),
        "S": ("Scaffold", "#404040"),
    }
    grid = {
        "GPT-5.5 Pro":     ["S", "", "", "", "", ""],
        "Opus 4.7":        ["M", "", "", "", "", ""],
        "Opus 4.8":        ["", "M\nF", "M", "M\nF", "M", "M\nF"],
        "GPT-5.5 (codex)": ["F\nV", "C", "F\nV", "C", "C", "C"],
        "Gemini 3.5 (agy)":["T", "T", "", "T", "T", "T"],
    }

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    nx, ny = len(stages), len(models)
    for j, m in enumerate(models):
        y = ny - 1 - j
        for i in range(nx):
            cell = grid[m][i]
            if not cell:
                ax.add_patch(Rectangle((i, y), 1, 1, facecolor="#f2f2f2",
                                       edgecolor="white", lw=2))
                continue
            roles = cell.split("\n")
            # base color = the most senior role in the cell
            order = ["M", "F", "V", "C", "T", "S"]
            primary = sorted(roles, key=lambda r: order.index(r))[0]
            color = role_color[primary][1]
            ax.add_patch(Rectangle((i, y), 1, 1, facecolor=color,
                                   edgecolor="white", lw=2, alpha=0.92))
            label = " / ".join(role_color[r][0] for r in roles)
            ax.text(i + 0.5, y + 0.5, label, ha="center", va="center",
                    color="white", fontsize=8.2, weight="bold")
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_xticks(np.arange(nx) + 0.5)
    ax.set_xticklabels(stages, fontsize=8.3)
    ax.set_yticks(np.arange(ny) + 0.5)
    ax.set_yticklabels(models[::-1], fontsize=9.5)
    ax.tick_params(length=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title("Model-role timeline across the v0.1.0 build stages\n"
                 "(role each model played per stage; source: ai_process_ledger.md / effort_accounting.md)",
                 fontsize=11, weight="bold")
    # legend
    handles = [Rectangle((0, 0), 1, 1, color=c) for _, c in role_color.values()]
    labels = [n for n, _ in role_color.values()]
    ax.legend(handles, labels, ncol=6, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=8.5)
    ax.text(0.0, -0.30,
            "Verifier cadence: every-sprint (a) → every-milestone (c–f), to conserve tokens once foundations were trustworthy.  "
            "GPT-5.5 / Gemini reactive after stage (a).",
            transform=ax.transAxes, fontsize=7.6, color=MUTE)
    save(fig, "model_role_timeline.png")


# ---------------------------------------------------------------------------
# 2. validation_pyramid.png  (§5, from validation_pyramid.md + §5 v0.1.0 status)
# ---------------------------------------------------------------------------
def fig_validation_pyramid():
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    # 4 trapezoid tiers; widest at top (Tier 4) per the spec's "widest" note,
    # but operational-trust increases upward, so we draw a layered stack with
    # Tier 1 at the base (foundational) and Tier 4 at the top (operational).
    tiers = [
        # (title, sub, status_segments[(label,color)])
        ("Tier 1 — WRF savepoint / operator parity",
         "analytic fixtures, tridiagonal solve, acoustic recurrence,\nadvection order, mass semantics",
         [("PASS", PASS_G)]),
        ("Tier 2 — invariants / conservation / guards",
         "finiteness, bounds, dry-mass & water budget, positivity;\nwarm bubble 6/6 guards-off, d02 finite guards-off",
         [("PASS", PASS_G)]),
        ("Tier 3 — idealized analytic benchmarks",
         "Skamarock warm bubble 6/6, Straka density current 6/6\n(through operational entry)",
         [("PASS", PASS_G)]),
        ("Tier 4 — real-case skill + persistence baseline",
         "d02 3 km: D02_VALIDATED (3 cases, stable 72 h, U10/V10 beat persistence)\n"
         "d03 1 km: D03_1KM_VALIDATED (24 h, T2 RMSE 1.92 K < 3.0 gate, beats persistence; secondary — empirical-partial HFX repair)\n"
         "TOST: n=3 MAM GPU paired-delta, underpowered single-season descriptive check (NOT 'equivalence PASS')",
         [("d02 PASS", PASS_G), ("d03 PASS (secondary)", PASS_G), ("TOST underpowered", WARN_O)]),
    ]
    n = len(tiers)
    H = 1.35           # height per tier band
    gap = 0.14
    base_y = 0.0
    # widths grow upward toward Tier 4 (visually widest = most corpus/runtime cost)
    widths = [6.6, 7.4, 8.2, 9.8]
    cx = 5.5
    for idx, (title, sub, segs) in enumerate(tiers):
        y0 = base_y + idx * (H + gap)
        w = widths[idx]
        x0 = cx - w / 2
        # split the status-strip header into segments; body text spans full width
        strip_h = 0.42
        seg_w = w / len(segs)
        for k, (lab, col) in enumerate(segs):
            ax.add_patch(FancyBboxPatch((x0 + k * seg_w, y0 + H - strip_h), seg_w, strip_h,
                                        boxstyle="square,pad=0.0",
                                        facecolor=col, edgecolor="white", lw=1.5, alpha=0.95))
            ax.text(x0 + (k + 0.5) * seg_w, y0 + H - strip_h / 2, lab, ha="center",
                    va="center", color="white", fontsize=8.2, weight="bold")
        # body box (uses the dominant/first status color, muted)
        body_col = segs[0][1]
        ax.add_patch(FancyBboxPatch((x0, y0), w, H - strip_h,
                                    boxstyle="square,pad=0.0",
                                    facecolor=body_col, edgecolor="white", lw=1.5, alpha=0.78))
        ax.text(cx, y0 + (H - strip_h) * 0.66, title, ha="center", va="center",
                color="white", fontsize=9.8, weight="bold")
        ax.text(cx, y0 + (H - strip_h) * 0.26, sub, ha="center", va="center",
                color="white", fontsize=6.9, style="italic")
    top = base_y + n * (H + gap)
    # vertical "operational trust" arrow
    ax.annotate("", xy=(0.0, top - 0.1), xytext=(0.0, base_y + 0.1),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
    ax.text(-0.35, top / 2, "operational trust →", rotation=90, va="center",
            ha="center", fontsize=9, color=INK)
    ax.text(cx, -0.6, "horizontal width ≈ corpus / runtime / data-management cost  "
            "(Tier 4 widest)", ha="center", fontsize=8, color=MUTE)
    ax.set_xlim(-0.8, 11.0)
    ax.set_ylim(-1.0, top + 0.3)
    ax.axis("off")
    ax.set_title("Validation stack — v0.1.0 status\n(green = PASS, amber = bounded-fail / underpowered)",
                 fontsize=12, weight="bold")
    save(fig, "validation_pyramid.png")


# ---------------------------------------------------------------------------
# PPM helper
# ---------------------------------------------------------------------------
def load_ppm(path):
    return np.asarray(Image.open(path).convert("RGB"))


# ---------------------------------------------------------------------------
# 3. warm_bubble_panel.png  (§6.1, from idealized PPMs)
# ---------------------------------------------------------------------------
def fig_warm_bubble_panel():
    times = [100, 250, 500]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3))
    for ax, t in zip(axes, times):
        img = load_ppm(os.path.join(IDEAL, f"warm_bubble_theta_prime_{t}s.ppm"))
        ax.imshow(img, interpolation="nearest", aspect="auto")
        ax.set_title(f"t = {t} s", fontsize=11, weight="bold")
        ax.set_xlabel("x (20 km domain)", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].set_ylabel("z (10 km)", fontsize=8.5)
    fig.suptitle("Skamarock & Wicker (1998) rising warm bubble — θ′ evolution  (PASS 6/6)",
                 fontsize=12.5, weight="bold", y=1.02)
    fig.text(0.5, -0.04,
             "Through the operational entry point, dt = 0.1 s.  @500 s: θ′ max ≈ 1.920 K "
             "(target 0.5–2.5),  max|w| ≈ 11.68 m/s (target 1–30),  thermal rise ≈ 1924 m "
             "(≥ 500),\nhorizontal drift ≈ 1.8e−12 m (symmetric),  dry-mass drift ≈ 0,  all "
             "snapshots finite.   Color = θ′ (warm anomaly bright).  Source: "
             "proofs/f7n/skamarock_bubble_diagnostics.json.",
             ha="center", fontsize=7.8, color=MUTE)
    save(fig, "warm_bubble_panel.png")


# ---------------------------------------------------------------------------
# 4. straka_density_current_panel.png  (§6.1, from idealized PPM)
# ---------------------------------------------------------------------------
def fig_straka_panel():
    img = load_ppm(os.path.join(IDEAL, "density_current_theta_prime_900s.ppm"))
    h, w, _ = img.shape
    fig, ax = plt.subplots(figsize=(11.5, 3.4))
    # domain is +/- 25.6 km half-width (51.2 km), 6.4 km tall (Straka std config)
    ax.imshow(img, interpolation="nearest", aspect="auto",
              extent=[0, 51.2, 0, 6.4])
    # front position ~14150 m from center; Straka domain centered at x=0 with
    # cold bubble at center -> front at |x-15000|. Mark the measured front band.
    ax.axvline(25.6 + 14.150, color="white", lw=1.4, ls="--")
    ax.text(25.6 + 14.150, 6.0, "front ≈ 14 150 m\n(|x−15000| ≤ 2000)", color="white",
            fontsize=8, ha="center", va="top", weight="bold")
    ax.set_xlabel("x (km)", fontsize=9)
    ax.set_ylabel("z (km)", fontsize=9)
    ax.set_title("Straka et al. (1993) density current — θ′ at t = 900 s  (PASS 6/6)",
                 fontsize=12.5, weight="bold")
    fig.text(0.5, -0.10,
             "Cold front + Kelvin-Helmholtz rotor structure (rotor-count proxy = 4, target 2–4).  "
             "θ′ min ≈ −9.971 K (target −25…−5),  max|w| ≈ 14.575 m/s (target 1–50),  "
             "dry-mass drift ≈ 2.25e−9,  all snapshots finite.\nColor = θ′ (cold anomaly dark).  "
             "Source: proofs/f7n/straka_density_current_diagnostics.json.",
             ha="center", fontsize=7.8, color=MUTE)
    save(fig, "straka_density_current_panel.png")


# ---------------------------------------------------------------------------
# 5. roofline_dycore.png  (§6.4, from roofline_costonly.json + phase_breakdown)
# ---------------------------------------------------------------------------
def fig_roofline():
    rj = json.load(open(os.path.join(ROOT, "proofs/perf/roofline_costonly.json")))
    peak = rj["peak_specs"]
    dy = rj["dycore_only_step"]["roofline"]
    cp = rj["coupled_step"]["roofline"]
    fp32 = peak["fp32_tflops"] * 1e3   # GFLOP/s
    fp64 = peak["fp64_tflops"] * 1e3
    bw = peak["hbm_tbytes_s"] * 1e3    # GB/s
    AI_dy = dy["arithmetic_intensity_flop_per_byte"]
    AI_cp = cp["arithmetic_intensity_flop_per_byte"]
    ridge_fp64 = dy["ridge_AI_fp64"]
    ridge_fp32 = dy["ridge_AI_fp32"]
    achieved_dy = dy["achieved_tflops"] * 1e3  # GFLOP/s
    floor_ms = dy["hbm_bound_floor_ms"]
    wall_ms = dy["warmed_per_step_ms"]
    tax = dy["launch_overhead_factor_vs_hbm_floor"]

    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    ai = np.logspace(-2, 2.2, 400)
    # rooflines: perf = min(peak_flops, bw*AI)
    roof_fp64 = np.minimum(fp64, bw * ai)
    roof_fp32 = np.minimum(fp32, bw * ai)
    ax.loglog(ai, roof_fp64, color="#2166ac", lw=2.2, label=f"fp64 roofline ({fp64/1e3:.2f} TFLOP/s)")
    ax.loglog(ai, roof_fp32, color="#7fbf7b", lw=1.6, ls="--",
              label=f"fp32 roofline ({fp32/1e3:.1f} TFLOP/s)")
    ax.loglog(ai, bw * ai, color="#999999", lw=1.2, ls=":",
              label=f"HBM bandwidth ({bw/1e3:.3f} TB/s)")
    # ridges
    for rg, lab, col in [(ridge_fp64, "fp64 ridge 0.914", "#2166ac"),
                         (ridge_fp32, "fp32 ridge 58.5", "#5aae61")]:
        ax.axvline(rg, color=col, ls="-.", lw=0.9, alpha=0.5)
        ax.text(rg, fp32 * 1.25, lab, rotation=90, fontsize=7, color=col,
                ha="right", va="top")
    # memory-bound region shading (AI < fp64 ridge)
    ax.axvspan(ai[0], ridge_fp64, color="#2166ac", alpha=0.05)
    ax.text(ai[0] * 1.4, fp64 * 0.012, "memory-bound\nregion", fontsize=8,
            color="#2166ac", alpha=0.8)

    # dycore achieved point
    ax.plot(AI_dy, achieved_dy, "o", color=FAIL_R, ms=11, zorder=5)
    ax.annotate(f"dycore step\nAI = {AI_dy:.2f} FLOP/byte\nachieved {achieved_dy/1e3:.3f} TFLOP/s "
                f"({dy['pct_fp64_peak']:.1f}% fp64 peak,\n{dy['pct_hbm_peak']:.1f}% HBM peak)",
                xy=(AI_dy, achieved_dy), xytext=(AI_dy * 1.5, achieved_dy * 0.05),
                fontsize=8, color=FAIL_R,
                arrowprops=dict(arrowstyle="->", color=FAIL_R, lw=1.2))
    # the HBM-floor point at same AI (where the step *could* sit, bandwidth-bound)
    floor_perf = bw * AI_dy  # GFLOP/s on the bandwidth line at this AI
    ax.plot(AI_dy, floor_perf, "^", color=PASS_G, ms=10, zorder=5)
    ax.annotate(f"HBM-bandwidth floor\n{floor_ms:.2f} ms/step",
                xy=(AI_dy, floor_perf), xytext=(AI_dy * 0.12, floor_perf * 1.4),
                fontsize=8, color=PASS_G,
                arrowprops=dict(arrowstyle="->", color=PASS_G, lw=1.1))
    # tax bracket annotation between floor and achieved
    ax.annotate("", xy=(AI_dy * 0.86, floor_perf), xytext=(AI_dy * 0.86, achieved_dy),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
    ax.text(AI_dy * 0.78, np.sqrt(floor_perf * achieved_dy),
            f"{tax:.1f}× launch/\nserialization tax\n({wall_ms:.1f} ms vs {floor_ms:.2f} ms)",
            fontsize=8, ha="right", va="center", weight="bold")

    ax.set_xlabel("Arithmetic intensity (FLOP / byte)", fontsize=10)
    ax.set_ylabel("Performance (GFLOP/s)", fontsize=10)
    ax.set_title("Dycore roofline — RTX 5090 (GB202), fp64, d02 3 km\n"
                 "memory-bandwidth-bound at ~19% HBM peak; NOT fp64-compute-bound",
                 fontsize=11.5, weight="bold")
    ax.set_xlim(0.01, 160)
    ax.set_ylim(50, fp32 * 2.2)
    ax.grid(True, which="both", alpha=0.18)
    ax.legend(loc="lower right", fontsize=8)
    fig.text(0.5, -0.02,
             "Speedup context: 5.29× (clean) / 7.84× (realistic incl. radiation+IO) vs 28-rank CPU-WRF on the same workstation, d02-vs-d02.  "
             "The 5.3× gap to the bandwidth floor is pure kernel-launch/latency overhead.\n"
             "Source: proofs/perf/roofline_costonly.json, phase_breakdown.json, performance_current.md.",
             ha="center", fontsize=7.5, color=MUTE)
    save(fig, "roofline_dycore.png")


# ---------------------------------------------------------------------------
# 6. workflow_loop.png  (§3.6, from the mermaid/ASCII control loop)
# ---------------------------------------------------------------------------
def fig_workflow_loop():
    fig, ax = plt.subplots(figsize=(11.0, 7.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(cx, cy, w, h, text, fc, tc="white", fs=9.5):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor=fc, edgecolor=INK, lw=1.3))
        ax.text(cx, cy, text, ha="center", va="center", color=tc,
                fontsize=fs, weight="bold")
        return (cx, cy, w, h)

    def arrow(p1, p2, label="", style="-|>", color=INK, dashed=False, rad=0.0,
              loff=(0, 0.0)):
        ls = (0, (4, 3)) if dashed else "solid"
        a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=16,
                            color=color, lw=1.5, linestyle=ls,
                            connectionstyle=f"arc3,rad={rad}")
        ax.add_patch(a)
        if label:
            mx, my = (p1[0] + p2[0]) / 2 + loff[0], (p1[1] + p2[1]) / 2 + loff[1]
            ax.text(mx, my, label, fontsize=7.4, color=MUTE, ha="center",
                    va="center", style="italic",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    H = box(6, 9.1, 4.6, 1.0, "Human principal\ninitiator + ~daily top-level steering", "#404040")
    M = box(6, 6.7, 4.6, 1.15, "MANAGER / FRONTRUNNER  (Opus 4.7 → 4.8)\n"
            "scopes contract, freezes file ownership,\nreviews diff, runs gates, decides, merges", "#2166ac")
    W = box(2.0, 3.9, 3.0, 1.2, "WORKER\n(GPT-5.5 / Opus 4.8)\nedits owned files,\nwrites proof objects", "#1b7837", fs=8.8)
    V = box(6.0, 3.9, 3.0, 1.2, "VERIFIER\n(cross-model)\nreruns gates,\ninspects proofs", "#8073ac", fs=8.8)
    C = box(9.9, 3.9, 3.0, 1.2, "CRITIC\n(GPT-5.5, adversarial)\nopposing case before\nmilestone / major plan", "#d6604d", fs=8.5)
    D = box(6, 1.6, 4.4, 0.95, "Claim backed by a proof object on disk?", "#f0f0f0", tc=INK, fs=9.5)
    T = box(11.0, 6.7, 1.9, 0.95, "Tiebreak\nGemini 3.5\n(reactive)", WARN_O, fs=8.0)

    # human <-> manager
    arrow((H[0] - 1.0, H[1] - 0.5), (M[0] - 1.0, M[1] + 0.58), "brief / milestone steering", loff=(-0.2, 0))
    arrow((M[0] + 1.0, M[1] + 0.58), (H[0] + 1.0, H[1] - 0.5), "status", color=MUTE, loff=(0.2, 0))
    # manager -> worker (dispatch)
    arrow((M[0] - 1.6, M[1] - 0.58), (W[0], W[1] + 0.62), "sprint contract +\nfrozen file ownership", rad=-0.15, loff=(-1.3, 0.3))
    # manager -> verifier
    arrow((M[0], M[1] - 0.58), (V[0], V[1] + 0.62), "dispatch gates", loff=(0.9, 0))
    # manager -> critic
    arrow((M[0] + 1.6, M[1] - 0.58), (C[0], C[1] + 0.62), "before milestone /\nmajor plan", rad=0.15, loff=(1.5, 0.3))
    # worker/verifier/critic -> decision
    arrow((W[0], W[1] - 0.62), (D[0] - 1.6, D[1] + 0.5), "diff + proof objects", rad=-0.1, loff=(-0.6, -0.2))
    arrow((V[0], V[1] - 0.62), (D[0], D[1] + 0.5), "pass / REFUSE", loff=(0.9, 0))
    arrow((C[0], C[1] - 0.62), (D[0] + 1.6, D[1] + 0.5), "opposing case", rad=0.1, loff=(0.8, -0.2))
    # tiebreak (dashed, both directions)
    arrow((M[0] + 2.3, M[1]), (T[0] - 0.95, T[1]), "both failed", dashed=True, color=WARN_O)
    arrow((T[0] - 0.95, T[1] - 0.3), (M[0] + 2.3, M[1] - 0.3), "", dashed=True, color=WARN_O)
    # decision -> merge (yes) and back to worker (no)
    box(10.4, 1.6, 2.6, 0.8, "YES → merge + closeout note", PASS_G, fs=8.5)
    arrow((D[0] + 2.2, D[1]), (10.4 - 1.3, 1.6), "")
    arrow((D[0] - 2.2, D[1] + 0.1), (W[0] - 0.2, W[1] - 0.65), "NO → back to worker", rad=0.3, color=FAIL_R, loff=(-1.6, 0.4))
    # merge -> human status
    arrow((10.4, 1.6 + 0.4), (H[0] + 1.6, H[1] - 0.5), "", color=MUTE, rad=-0.35, dashed=True)

    ax.set_title("Governed control loop — repeated ≈500–700× across the build (§3.6)\n"
                 "merge only if a proof object on disk backs the claim",
                 fontsize=12, weight="bold")
    save(fig, "workflow_loop.png")


# ---------------------------------------------------------------------------
# 7. self_correction_timeline.png  (optional, §6.5)
# ---------------------------------------------------------------------------
def fig_self_correction_timeline():
    # (label, kind)  kind: 'overclaim' (red), 'fix' (green), 'status' (blue)
    events = [
        ("v0.0.1 over-claim:\n“bitwise WRF parity”", "overclaim"),
        ("Self-compare retraction\nJAX-vs-JAX, ~7 ops missing", "fix"),
        ("Dycore F7 rebuild close\nidealized PASS vs pristine WRF", "fix"),
        ("Speedup-denominator fix\n22.26× → 5.3× / 7.8×, d02-vs-d02", "fix"),
        ("Persistence baseline\nexposes wind gap (V10<persist)", "overclaim"),
        ("Missing-Coriolis fix\nV10 −0.13 → +0.17", "fix"),
        ("d03 boundary-pump fix\n+6.8 K → d02-quality", "fix"),
        ("HFX/MYNN thermal-rough fix\nHFX 4.22×→2.30×, T2 +3.6→+1.2 K", "fix"),
        ("v0.1.0: d02 validated;\nd03 24 h validated (secondary)", "status"),
    ]
    col = {"overclaim": FAIL_R, "fix": PASS_G, "status": "#2166ac"}

    fig, ax = plt.subplots(figsize=(16.5, 4.6))
    n = len(events)
    xs = np.arange(n)
    ax.plot(xs, np.zeros(n), color="#999999", lw=2, zorder=1)
    bw = 1.02          # box width (data units; spacing between events is 1.0)
    for i, (lab, kind) in enumerate(events):
        c = col[kind]
        up = (i % 2 == 0)
        y = 0.95 if up else -0.95
        ax.plot([i, i], [0, y * 0.52], color=c, lw=1.4, zorder=1)
        ax.plot(i, 0, "o", color=c, ms=12, zorder=3)
        ax.add_patch(FancyBboxPatch((i - bw / 2, y - 0.34), bw, 0.68,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=c, edgecolor="white", lw=1.5, alpha=0.92))
        ax.text(i, y, lab, ha="center", va="center", color="white",
                fontsize=6.3, weight="bold")
    ax.set_xlim(-0.8, n - 0.2)
    ax.set_ylim(-1.8, 1.8)
    ax.axis("off")
    # legend
    leg = [("over-claim caught", FAIL_R), ("self-correction / fix", PASS_G),
           ("current status", "#2166ac")]
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          markersize=11) for _, c in leg]
    ax.legend(handles, [l for l, _ in leg], loc="upper center",
              bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False, fontsize=8.5)
    ax.set_title("AI self-correction timeline (§6.5) — over-claims generated and then caught by the process",
                 fontsize=11.5, weight="bold", y=1.22)
    ax.annotate("", xy=(n - 0.4, -1.35), xytext=(-0.4, -1.35),
                arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=1.2))
    ax.text(n / 2 - 0.5, -1.5, "build chronology →", ha="center", fontsize=8, color=MUTE)
    save(fig, "self_correction_timeline.png")


if __name__ == "__main__":
    fig_model_role_timeline()
    fig_validation_pyramid()
    fig_warm_bubble_panel()
    fig_straka_panel()
    fig_roofline()
    fig_workflow_loop()
    fig_self_correction_timeline()
    print("done")
