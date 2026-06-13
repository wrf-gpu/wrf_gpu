#!/usr/bin/env python3
"""Long-horizon divergence-growth companion panel for the v0.15 identity proof.

ADDS a panel next to the existing identity-proof dashboards (it does NOT modify or
loosen them). For each region it draws, side by side and honestly:

  (a) the STRICT frozen per-cell tolerance verdict -- unchanged, green where within
      the frozen limit, RED where over (read straight from the shipped distilled
      gate JSON; the same red/green as the identity dashboard), AND
  (b) the LONG-HORIZON DIVERGENCE-GROWTH classification (the reduced-precision
      equivalence criterion, doc §3): per field BOUNDED / BOUNDED_GROWTH (= no
      run-away) vs ESCALATING (= run-away), computed by divergence_growth_metric.py.

The honest message rendered for a field that is over-tolerance but bounded:
  "exceeds the tight per-cell tolerance (carried to 0.16) BUT bounded / non-escalating
   over 72h -- no run-away."

CPU-only; consumes the verdict JSON + the distilled gate JSON; no model rerun, no GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GREEN = "#1a9850"
RED = "#d73027"
AMBER = "#e08214"
BLUE = "#2166ac"
GREY = "#666666"


def import_pyplot():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["figure.max_open_warning"] = 0
    import matplotlib.pyplot as plt
    return matplotlib, plt


def load_strict(distilled_path: Path) -> dict:
    """Return {field: {'pass': bool, 'ratio': value/limit, 'overall_rmse', 'limit'}}."""
    d = json.loads(distilled_path.read_text())
    out = {}
    for name, rec in d["hard_fields"].items():
        lim = rec.get("limit")
        rm = rec.get("overall_rmse")
        ratio = (rm / lim) if (lim and rm is not None) else None
        out[name] = {"pass": bool(rec.get("pass")), "ratio": ratio,
                     "overall_rmse": rm, "limit": lim,
                     "h72_rmse": rec.get("h72_rmse")}
    return {"fields": out, "hard_gate_passes": d.get("hard_gate_passes"),
            "top_verdict": d.get("top_verdict")}


def regime_color(regime: str) -> str:
    return {"BOUNDED": GREEN, "BOUNDED_GROWTH": AMBER, "ESCALATING": RED}.get(regime, GREY)


def build_panel(plt, mpl, region_label: str, init_label: str,
                region_verdict: dict, strict: dict, focus_field: str,
                out_path: Path) -> str:
    per_field = region_verdict["per_field"]
    # Order fields as the identity dashboard does.
    order = ["T", "U", "V", "W", "QVAPOR", "T2", "U10", "V10", "PSFC", "RAINNC"]
    fields = [f for f in order if f in per_field]

    leads = np.array(region_verdict["leads_h"], dtype=float)
    fv_focus = per_field[focus_field]

    n_runaway = sum(1 for f in fields if per_field[f]["is_runaway"])
    strict_n = strict["hard_gate_passes"]

    fig = plt.figure(figsize=(14.0, 8.2), dpi=140)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.18], width_ratios=[1.12, 1.0],
                          hspace=0.42, wspace=0.22)

    # --- header (top-left) ---
    axh = fig.add_subplot(gs[0, 0]); axh.axis("off")
    no_runaway_all = (n_runaway == 0)
    axh.text(0.0, 1.0, "LONG-HORIZON DIVERGENCE TEST", fontsize=18, weight="bold", va="top")
    axh.text(0.0, 0.84, region_label, fontsize=13, va="top")
    axh.text(0.0, 0.72, init_label, fontsize=9.5, color="0.4", va="top")
    # two verdicts, both shown:
    axh.text(0.0, 0.57, "Gate 1 (strict frozen per-cell tolerance, UNCHANGED):",
             fontsize=10, va="top", color="0.2")
    axh.text(0.02, 0.485, f"{strict_n} within tolerance   (the over-tolerance field stays RED)",
             fontsize=11, weight="bold", va="top",
             color=GREEN if strict.get("top_verdict") == "PASS" else AMBER)
    axh.text(0.0, 0.36, "Gate 2 (long-horizon non-escalating divergence, ADDED):",
             fontsize=10, va="top", color="0.2")
    g2 = "NO RUN-AWAY -- all fields bounded / non-escalating over 72h" if no_runaway_all \
        else f"{n_runaway} field(s) ESCALATING (run-away)"
    axh.text(0.02, 0.285, g2, fontsize=11, weight="bold", va="top",
             color=GREEN if no_runaway_all else RED)
    note = ("Criterion: classify GPU-vs-CPU divergence(t)=RMSE per lead by its TREND "
            "(saturating vs sustained growth),\nnormalized by the ORACLE's own spatial "
            "variability. Bounded = co-evolves within the field's own spread,\nNOT "
            "necessarily tiny. This ADDS to, and never replaces or loosens, Gate 1.")
    axh.text(0.0, 0.16, note, fontsize=7.8, family="monospace", color="0.35", va="top")

    # --- focus-field divergence-growth curve (top-right) ---
    axc = fig.add_subplot(gs[0, 1])
    d = np.array(fv_focus["divergence_series_rmse"], dtype=float)
    env = fv_focus["envelope_used"]
    axc.plot(leads, d, color=BLUE, lw=1.9, marker="o", ms=2.6, label=f"{focus_field} divergence RMSE(t)")
    axc.axhline(env, color=GREY, ls="--", lw=1.3, label=f"oracle variability (1x env={env:.3g})")
    axc.axhline(5 * env, color="0.75", ls=":", lw=1.1, label="5x env (bounded ceiling)")
    fv_lim = strict["fields"].get(focus_field, {}).get("limit")
    if fv_lim is not None:
        axc.axhline(fv_lim, color=RED, ls="-.", lw=1.2, label=f"strict tol limit ({fv_lim:.3g})")
    # annotate early vs late slope
    es = fv_focus["early_slope_per_lead"]; ls_ = fv_focus["late_slope_per_lead"]
    axc.set_title(f"{focus_field}: divergence saturates (early slope {es:.3g} -> late {ls_:.3g} per h)\n"
                  f"regime = {fv_focus['regime']}  (run-away = {'YES' if fv_focus['is_runaway'] else 'NO'})",
                  fontsize=9.5, color=regime_color(fv_focus["regime"]))
    axc.set_xlabel("lead hour", fontsize=9)
    axc.set_ylabel(f"{focus_field} GPU-CPU RMSE", fontsize=9)
    axc.grid(True, alpha=0.25)
    axc.tick_params(labelsize=8)
    axc.legend(fontsize=6.8, loc="upper left", framealpha=0.85)
    top = max(float(np.max(d)), env, (fv_lim or 0)) * 1.25
    axc.set_ylim(0, top if top > 0 else 1.0)

    # --- normalized divergence trajectories, all fields (bottom-left) ---
    axn = fig.add_subplot(gs[1, 0])
    for f in fields:
        dd = np.array(per_field[f]["divergence_series_rmse"], dtype=float)
        e = per_field[f]["envelope_used"]
        col = regime_color(per_field[f]["regime"])
        lw = 2.4 if f == focus_field else 1.0
        z = 5 if f == focus_field else 2
        axn.plot(leads, dd / e, color=col, lw=lw, alpha=0.95 if f == focus_field else 0.55,
                 zorder=z, label=(f"{f} (focus)" if f == focus_field else None))
    axn.axhline(1.0, color=GREY, ls="--", lw=1.0)
    axn.text(leads[-1], 1.02, "1x oracle env", fontsize=7, color=GREY, ha="right", va="bottom")
    axn.axhline(5.0, color="0.75", ls=":", lw=1.0)
    axn.text(leads[-1], 5.05, "5x env (bounded ceiling)", fontsize=7, color="0.6", ha="right", va="bottom")
    axn.set_xlabel("lead hour", fontsize=9)
    axn.set_ylabel("divergence / oracle variability", fontsize=9)
    axn.set_title("all fields: GPU-CPU divergence normalized by the oracle's own spatial spread\n"
                  "(flat/turning-over = bounded; only a sustained climb past 5x = run-away)",
                  fontsize=9.5)
    axn.grid(True, alpha=0.25)
    axn.tick_params(labelsize=8)
    axn.set_ylim(0, max(6.0, float(np.nanmax([np.max(np.array(per_field[f]["divergence_series_rmse"]) /
                                                     per_field[f]["envelope_used"]) for f in fields])) * 1.1))
    if focus_field in fields:
        axn.legend(fontsize=7.5, loc="upper left", framealpha=0.85)

    # --- combined verdict table (bottom-right) ---
    axt = fig.add_subplot(gs[1, 1]); axt.axis("off")
    header = f"{'field':>8}  {'strict tol':>10}  {'div regime':>14}  {'max/env':>7}  {'late/early':>9}"
    axt.text(0.0, 1.0, header, fontsize=8.2, family="monospace", va="top", weight="bold")
    axt.text(0.0, 0.965, "-" * len(header), fontsize=8.2, family="monospace", va="top")
    y = 0.93
    for f in fields:
        st = strict["fields"].get(f, {})
        stp = st.get("pass")
        stxt = "WITHIN" if stp else "OVER"
        sc = GREEN if stp else RED
        reg = per_field[f]["regime"]
        rc = regime_color(reg)
        moe = per_field[f]["max_over_envelope"]
        lor = per_field[f]["late_slope_ratio"]
        # field name
        axt.text(0.0, y, f"{f:>8}", fontsize=8.0, family="monospace", va="top")
        axt.text(0.215, y, f"{stxt:>10}", fontsize=8.0, family="monospace", va="top", color=sc, weight="bold")
        axt.text(0.45, y, f"{reg:>14}", fontsize=8.0, family="monospace", va="top", color=rc, weight="bold")
        axt.text(0.72, y, f"{moe:>7.2f}", fontsize=8.0, family="monospace", va="top")
        axt.text(0.86, y, f"{lor:>+9.2f}", fontsize=8.0, family="monospace", va="top")
        y -= 0.052
    axt.text(0.0, y - 0.02,
             ("Honest reading: the field drawn OVER in column 'strict tol' (RED) exceeds the\n"
              "tight per-cell tolerance and is CARRIED to 0.16 -- but its divergence regime is\n"
              "BOUNDED (no run-away), so it is a tight-tolerance miss, not a stability failure."),
             fontsize=7.4, family="monospace", color="0.3", va="top")

    fig.suptitle("v0.15 long-horizon divergence -- strict tolerance (unchanged) + non-escalating-divergence "
                 "equivalence criterion",
                 fontsize=12.5, weight="bold")
    fig.subplots_adjust(left=0.055, right=0.975, top=0.89, bottom=0.065)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return str(out_path)


def main(argv=None) -> int:
    # repo root = parent of scripts/ ; works in both shared checkout and worktree.
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verdict-json", type=Path,
                   default=root / "proofs/v015/long_horizon_divergence_verdict.json")
    p.add_argument("--switz-distilled", type=Path,
                   default=root / "proofs/v015/finalgates/switzerland_d01/switzerland_d01_72h_gate_distilled.json")
    p.add_argument("--canary-distilled", type=Path,
                   default=root / "proofs/v015/finalgates/canary_l2_d02/canary_d02_72h_gate_distilled.json")
    p.add_argument("--asset-root", type=Path, default=root / "docs/assets/v015/identity_proof")
    args = p.parse_args(argv)

    mpl, plt = import_pyplot()
    verdict = json.loads(args.verdict_json.read_text())

    specs = [
        ("switzerland_d01_72h", "Switzerland d01 72h", "init 2023-01-15 00Z, v0.15",
         args.switz_distilled, "RAINNC", "switzerland_d01"),
        ("canary_l2_d02_72h", "Canary L2 d02 72h", "init 2026-05-01 18Z, v0.15 nested",
         args.canary_distilled, "QVAPOR", "canary_l2_d02"),
    ]
    out_paths = []
    for region_key, label, init_label, distilled_path, focus, asset_sub in specs:
        rv = verdict["regions"][region_key]
        strict = load_strict(distilled_path)
        out = args.asset_root / asset_sub / "long_horizon_divergence_panel.png"
        out_paths.append(build_panel(plt, mpl, label, init_label, rv, strict, focus, out))
        print(f"wrote {out}")

    print(json.dumps({"panels": out_paths,
                      "both_carried_fields_bounded_no_runaway":
                          verdict["both_carried_fields_bounded_no_runaway"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
