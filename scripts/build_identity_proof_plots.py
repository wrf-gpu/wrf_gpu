#!/usr/bin/env python3
"""Build the v0.14 GPU-vs-CPU *Identity-Proof* visualization suite.

This is a sibling/consumer of ``scripts/build_grid_delta_atlas.py``. It reuses that
tool's pairing, NetCDF parsing, tolerance loading, and streaming-statistics code, and
adds a publication-quality, README-embeddable visual proof that the WRF-GPU port is
true to CPU-WRF v4 across **all cells, all leads, and all core internal variables**
for one region's 72 h GPU-vs-CPU run.

It runs **offline on existing wrfout NetCDF only**. It does not run WRF, JAX, CUDA, or
any model kernel. CPU-only; no GPU is touched.

Deliverables per region (one ``--proof-dir`` / ``--asset-dir``):
  1. Per-variable RMSE *and* bias time series across all leads, with the tolerance
     limit drawn (curves should sit at/under the bound).
  2. A variable x lead "scoreboard" heatmap of normalized error (value / limit),
     green where within tolerance.
  3. GPU-vs-CPU cell-value 1:1 identity scatter panels per variable (subsampled,
     pooled over leads) -- points on the diagonal = identity.
  4. Spatial GPU-CPU difference maps at h24/h48/h72 for the main prognostic variables,
     symmetric diverging colormap, tight honest scale, real max_abs annotated.
  5. ONE polished summary dashboard that tells the identity story at a glance.

Honesty contract:
  - Differences are shown at true scale. Nothing is clipped to hide error.
  - The variable subset is the predeclared *focused-writer* hard-gate scope; that scope
    is printed on every artifact.
  - Fields that are bounded-not-exact (e.g. RAINNC precipitation) are labelled, and a
    field that breaches its limit is drawn RED, never painted green.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

# --- Reuse build_grid_delta_atlas as a library (no code duplication) -----------------
_ATLAS_PATH = Path(__file__).resolve().parent / "build_grid_delta_atlas.py"
_spec = importlib.util.spec_from_file_location("build_grid_delta_atlas", _ATLAS_PATH)
assert _spec and _spec.loader, f"cannot load atlas module from {_ATLAS_PATH}"
atlas = importlib.util.module_from_spec(_spec)
sys.modules["build_grid_delta_atlas"] = atlas
_spec.loader.exec_module(atlas)

# Default focused-writer hard-gate scope (the honest identity-proof variable subset).
DEFAULT_IDENTITY_FIELDS = ("T", "U", "V", "W", "QVAPOR", "T2", "U10", "V10", "PSFC", "RAINNC")
# Main 3D/2D prognostic variables for spatial diff maps.
DEFAULT_SPATIAL_FIELDS = ("T", "U", "V", "W", "QVAPOR", "PSFC")
DEFAULT_SCATTER_FIELDS = ("T", "U", "V", "W", "QVAPOR", "T2", "U10", "V10", "PSFC", "RAINNC")
DEFAULT_PROOF_DIR = Path("proofs/v014/identity_proof")
DEFAULT_ASSET_DIR = Path("docs/assets/v014/identity_proof")
GREEN = "#1a9850"
RED = "#d73027"


def import_pyplot() -> tuple[Any | None, Any | None, str | None]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams["svg.hashsalt"] = "identity-proof-v014"
        matplotlib.rcParams["figure.max_open_warning"] = 0
        import matplotlib.pyplot as plt

        return matplotlib, plt, None
    except Exception as exc:  # pragma: no cover - optional environment
        return None, None, f"{type(exc).__name__}: {exc}"


def primary_limit(spec: dict[str, float] | None) -> tuple[str, float] | None:
    """Pick the single scalar limit a curve/scoreboard is scored against."""
    if not spec:
        return None
    for key in ("rmse", "mae", "max_abs", "p99_abs", "p95_abs"):
        if key in spec and spec[key] is not None:
            return key, float(spec[key])
    return None


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        x = float(value)
        if x == 0.0:
            return "0"
        if abs(x) >= 1e4 or abs(x) < 1e-3:
            return f"{x:.2e}"
        return f"{x:.{digits}f}"
    return "NA"


def by_lead_series(summary_field: dict[str, Any], metric: str) -> tuple[list[int], list[float]]:
    leads: list[int] = []
    vals: list[float] = []
    for row in summary_field.get("by_lead", []):
        if row.get(metric) is None:
            continue
        leads.append(int(row["lead_h"]))
        vals.append(float(row[metric]))
    order = np.argsort(leads)
    return [leads[i] for i in order], [vals[i] for i in order]


def field_scored_metric(name: str, tolerances: dict[str, dict[str, float]]) -> tuple[str, str, float | None]:
    """Return (metric_for_curve, limit_label, limit_value). limit None => report-only."""
    spec = tolerances.get(name)
    lim = primary_limit(spec)
    if lim is None:
        return "rmse", "no frozen limit (report-only)", None
    return lim[0], lim[0], lim[1]


# ------------------------------------------------------------------------------------
# Plot 1: per-variable RMSE + bias time series with tolerance bound drawn.
# ------------------------------------------------------------------------------------
def plot_timeseries_panels(plt, fields, field_metrics, tolerances, scope_label, title, path):
    fields = [f for f in fields if f in field_metrics and field_metrics[f].get("status") == "compared"]
    if not fields:
        return None
    ncol = 3
    nrow = math.ceil(len(fields) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 2.7 * nrow), dpi=130, squeeze=False)
    for idx, name in enumerate(fields):
        ax = axes[idx // ncol][idx % ncol]
        metric, _label, limit = field_scored_metric(name, tolerances)
        leads_r, rmse = by_lead_series(field_metrics[name], "rmse")
        leads_b, bias = by_lead_series(field_metrics[name], "bias")
        n_over = 0
        pooled_within = True
        if limit is not None and metric in {"rmse", "mae"}:
            n_over = sum(1 for v in rmse if v > limit)
            ov_pooled = field_metrics[name].get("overall", {}).get(metric)
            pooled_within = (ov_pooled is None) or (float(ov_pooled) <= limit)
        ax.plot(leads_r, rmse, color="#2166ac", lw=1.6, marker="o", ms=2.4, label="RMSE")
        ax.plot(leads_b, bias, color="#b2182b", lw=1.1, ls="--", label="bias")
        ax.axhline(0.0, color="0.6", lw=0.7, zorder=0)
        if limit is not None:
            ax.axhline(limit, color=GREEN if pooled_within else RED, lw=1.4, ls=":",
                       label=f"{metric} limit {fmt(limit)}")
            top = max([limit] + rmse + [abs(b) for b in bias] + [1e-12]) * 1.25
            ax.set_ylim(min(0.0, (min(bias) if bias else 0.0) * 1.25), top)
        if limit is None:
            verdict, vcolor = "report-only", "0.35"
        elif n_over == 0:
            verdict, vcolor = "all leads within", GREEN
        elif pooled_within:
            verdict, vcolor = f"pooled within; {n_over} lead(s) over", "#e08214"
        else:
            verdict, vcolor = "over limit", RED
        ax.set_title(f"{name}   [{verdict}]", fontsize=9.5, color=vcolor)
        ax.set_xlabel("lead hour", fontsize=8)
        ax.set_ylabel("RMSE / bias", fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.0, loc="upper left", framealpha=0.85)
    for j in range(len(fields), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"{title}\nper-variable RMSE & bias vs lead, with frozen tolerance bound", fontsize=13)
    fig.text(0.5, 0.005, scope_label, ha="center", fontsize=7.5, color="0.4")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


# ------------------------------------------------------------------------------------
# Plot 2: normalized scoreboard heatmap (value / limit), green-on-pass.
# ------------------------------------------------------------------------------------
def plot_scoreboard(plt, mpl, fields, field_metrics, tolerances, leads, scope_label, title, path):
    fields = [f for f in fields if f in field_metrics and field_metrics[f].get("status") == "compared"]
    if not fields or not leads:
        return None
    lead_index = {lead: i for i, lead in enumerate(leads)}
    norm = np.full((len(fields), len(leads)), np.nan)
    metrics_used: list[str] = []
    for r, name in enumerate(fields):
        metric, _label, limit = field_scored_metric(name, tolerances)
        metrics_used.append(metric if limit is not None else f"{metric}*")
        for row in field_metrics[name].get("by_lead", []):
            lead = int(row["lead_h"])
            if lead not in lead_index:
                continue
            val = row.get(metric)
            if val is None or not (limit and limit > 0):
                continue
            norm[r, lead_index[lead]] = float(val) / limit
    fig_h = max(3.0, 0.42 * len(fields) + 1.6)
    fig_w = max(8.0, 0.16 * len(leads) + 3.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=130)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "scoreboard", [(0.0, GREEN), (0.49, "#d9f0d3"), (0.5, "#ffffff"), (0.6, "#fddbc7"), (1.0, RED)])
    display = np.clip(norm / 2.0, 0.0, 1.0)  # limit-fraction 0..2 -> 0..1
    masked = np.ma.masked_invalid(display)
    im = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
    step = max(1, len(leads) // 24)
    ax.set_xticks(range(0, len(leads), step))
    ax.set_xticklabels([str(leads[i]) for i in range(0, len(leads), step)], fontsize=7)
    ax.set_yticks(range(len(fields)))
    ax.set_yticklabels([f"{f}  ({m})" for f, m in zip(fields, metrics_used)], fontsize=8)
    ax.set_xlabel("lead hour", fontsize=9)
    ax.set_title(f"{title}\nnormalized error  =  (per-lead metric) / frozen limit   "
                 f"[green < 1 within, red >= 1 over]", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.set_yticklabels(["0", "0.5", "1.0 (=limit)", "1.5", ">=2.0"], fontsize=7)
    for r, c in np.argwhere(norm >= 1.0)[:400]:
        ax.text(c, r, "x", ha="center", va="center", fontsize=5, color="black")
    fig.text(0.5, 0.005, scope_label + "   (* = report-only field, no frozen limit -> blank)",
             ha="center", fontsize=7.5, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


# ------------------------------------------------------------------------------------
# Plot 3: pooled GPU-vs-CPU 1:1 identity scatter per variable.
# ------------------------------------------------------------------------------------
def _gather_pooled_values(pairs, name, max_points, rng):
    per_pair = max(64, max_points // max(1, len(pairs)))
    cpu_vals: list[np.ndarray] = []
    gpu_vals: list[np.ndarray] = []
    for pair in pairs:
        try:
            with Dataset(pair.cpu_file, "r") as cds, Dataset(pair.gpu_file, "r") as gds:
                if name not in cds.variables or name not in gds.variables:
                    continue
                cpu = atlas.read_variable(cds, name).astype(np.float64).ravel()
                gpu = atlas.read_variable(gds, name).astype(np.float64).ravel()
        except Exception:
            continue
        if cpu.shape != gpu.shape:
            continue
        valid = np.isfinite(cpu) & np.isfinite(gpu)
        cpu, gpu = cpu[valid], gpu[valid]
        if cpu.size == 0:
            continue
        if cpu.size > per_pair:
            sel = rng.choice(cpu.size, size=per_pair, replace=False)
            cpu, gpu = cpu[sel], gpu[sel]
        cpu_vals.append(cpu)
        gpu_vals.append(gpu)
    if not cpu_vals:
        return None, None
    return np.concatenate(cpu_vals), np.concatenate(gpu_vals)


def plot_identity_scatter(plt, fields, pairs, field_metrics, tolerances, max_points,
                          scope_label, title, path, seed=20260612):
    fields = [f for f in fields if f in field_metrics and field_metrics[f].get("status") == "compared"]
    if not fields:
        return None
    rng = np.random.default_rng(seed)
    ncol = 3
    nrow = math.ceil(len(fields) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.8 * nrow), dpi=130, squeeze=False)
    for idx, name in enumerate(fields):
        ax = axes[idx // ncol][idx % ncol]
        cpu, gpu = _gather_pooled_values(pairs, name, max_points, rng)
        if cpu is None or cpu.size == 0:
            ax.axis("off")
            continue
        lo = float(min(cpu.min(), gpu.min()))
        hi = float(max(cpu.max(), gpu.max()))
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
        ax.plot([lo, hi], [lo, hi], color="0.2", lw=1.0, zorder=3, label="1:1 identity")
        ax.scatter(cpu, gpu, s=2.0, alpha=0.18, color="#2166ac", edgecolors="none",
                   rasterized=True, zorder=2)
        corr = field_metrics[name].get("overall", {}).get("correlation")
        rmse = field_metrics[name].get("overall", {}).get("rmse")
        _m, _lab, limit = field_scored_metric(name, tolerances)
        within = (limit is None) or (rmse is not None and float(rmse) <= limit)
        vcolor = "0.35" if limit is None else (GREEN if within else RED)
        ax.set_title(f"{name}   r={fmt(corr,5)}", fontsize=10, color=vcolor)
        ax.set_xlabel("CPU-WRF cell value", fontsize=8)
        ax.set_ylabel("GPU cell value", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.85)
    for j in range(len(fields), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"{title}\nGPU vs CPU cell values pooled over all leads (subsampled) -- on-diagonal = identity",
                 fontsize=12)
    fig.text(0.5, 0.005, scope_label, ha="center", fontsize=7.5, color="0.4")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


# ------------------------------------------------------------------------------------
# Plot 4: spatial GPU-CPU diff maps at chosen leads for main prognostic vars.
# ------------------------------------------------------------------------------------
def _lead_to_pair(pairs, lead):
    cand = [p for p in pairs if int(p.lead_h) == int(lead)]
    return cand[0] if cand else None


def _reduce_signed_2d(diff: np.ndarray) -> np.ndarray | None:
    """Collapse to 2D keeping the signed value where |.| is worst over collapsed axes."""
    arr = np.asarray(diff, dtype=np.float64)
    if arr.ndim < 2:
        return None
    while arr.ndim > 2:
        absmax_idx = np.nanargmax(np.abs(arr), axis=0)
        arr = np.take_along_axis(arr, absmax_idx[None, ...], axis=0)[0]
    return arr


def _vars_in(pair):
    if pair is None:
        return set()
    try:
        with Dataset(pair.cpu_file, "r") as cds, Dataset(pair.gpu_file, "r") as gds:
            return set(cds.variables) & set(gds.variables)
    except Exception:
        return set()


def plot_spatial_diffs(plt, mpl, fields, pairs, leads_wanted, scope_label, title, path):
    avail_leads = sorted({int(p.lead_h) for p in pairs})
    if not avail_leads:
        return None
    chosen: list[int] = []
    for want in leads_wanted:
        nearest = min(avail_leads, key=lambda L: abs(L - want))
        if nearest not in chosen:
            chosen.append(nearest)
    present = set()
    for L in chosen:
        present |= _vars_in(_lead_to_pair(pairs, L))
    rows = [f for f in fields if f in present]
    if not rows:
        return None
    ncol = len(chosen)
    nrow = len(rows)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), dpi=130, squeeze=False)
    cmap = mpl.colormaps["RdBu_r"].copy()
    for ri, name in enumerate(rows):
        for ci, lead in enumerate(chosen):
            ax = axes[ri][ci]
            pair = _lead_to_pair(pairs, lead)
            diff2d = None
            if pair is not None:
                try:
                    with Dataset(pair.cpu_file, "r") as cds, Dataset(pair.gpu_file, "r") as gds:
                        if name in cds.variables and name in gds.variables:
                            cpu = atlas.read_variable(cds, name).astype(np.float64)
                            gpu = atlas.read_variable(gds, name).astype(np.float64)
                            if cpu.shape == gpu.shape:
                                diff2d = _reduce_signed_2d(gpu - cpu)
                except Exception:
                    diff2d = None
            if diff2d is None:
                ax.axis("off")
                ax.set_title(f"{name} h{lead}: n/a", fontsize=8)
                continue
            mx = float(np.nanmax(np.abs(diff2d))) if np.isfinite(diff2d).any() else 0.0
            scale = mx if mx > 0 else 1.0
            im = ax.imshow(diff2d, origin="lower", cmap=cmap, vmin=-scale, vmax=scale,
                           interpolation="nearest", aspect="auto")
            ax.set_title(f"{name}  h{lead}\nmax|GPU-CPU|={fmt(mx)}", fontsize=8.5)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    fig.suptitle(f"{title}\nsigned GPU-CPU difference maps (worst level), symmetric scale per panel, true max annotated",
                 fontsize=12)
    fig.text(0.5, 0.004, scope_label, ha="center", fontsize=7.5, color="0.4")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


# ------------------------------------------------------------------------------------
# Plot 5: summary dashboard.
# ------------------------------------------------------------------------------------
def plot_dashboard(plt, mpl, region_label, init_label, fields, field_metrics, tolerances,
                   leads, scope_label, path):
    fields = [f for f in fields if f in field_metrics and field_metrics[f].get("status") == "compared"]
    rows = []
    n_within = 0
    n_scored = 0
    for name in fields:
        metric, _lab, limit = field_scored_metric(name, tolerances)
        ov = field_metrics[name].get("overall", {})
        val = ov.get(metric)
        within = None
        if limit is not None and val is not None:
            within = float(val) <= limit
            n_scored += 1
            n_within += int(within)
        rows.append({"field": name, "metric": metric, "value": val, "limit": limit,
                     "within": within, "rmse": ov.get("rmse"), "max_abs": ov.get("max_abs"),
                     "corr": ov.get("correlation"), "bias": ov.get("bias")})
    worst = None
    for r in rows:
        if r["limit"] and r["value"] is not None:
            frac = r["value"] / r["limit"]
            if worst is None or frac > worst[1]:
                worst = (r["field"], frac, r)

    fig = plt.figure(figsize=(13.6, 7.6), dpi=140)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], width_ratios=[1.15, 1.0],
                          hspace=0.32, wspace=0.22)

    axh = fig.add_subplot(gs[0, 0])
    axh.axis("off")
    all_pass = (n_scored > 0 and n_within == n_scored)
    badge = "ALL WITHIN TOLERANCE" if all_pass else f"{n_within}/{n_scored} WITHIN TOLERANCE"
    badge_color = GREEN if all_pass else "#e08214"
    axh.text(0.0, 1.0, "GPU<->CPU IDENTITY PROOF", fontsize=20, weight="bold", va="top")
    axh.text(0.0, 0.80, region_label, fontsize=14, va="top")
    axh.text(0.0, 0.66, init_label, fontsize=10, color="0.4", va="top")
    axh.text(0.0, 0.50, badge, fontsize=17, weight="bold", color=badge_color, va="top")
    lines = [
        f"variables (hard-gate scope): {len(rows)}",
        f"leads compared: {len(leads)}  (0..{max(leads) if leads else 0} h)",
        f"scored fields within frozen limit: {n_within}/{n_scored}",
    ]
    if worst is not None:
        wf, wfrac, wr = worst
        lines.append(f"worst field: {wf}  ({wr['metric']}={fmt(wr['value'])} vs limit {fmt(wr['limit'])}, "
                     f"{wfrac*100:.0f}% of limit)")
    axh.text(0.0, 0.34, "\n".join(lines), fontsize=10.5, family="monospace", va="top")
    axh.text(0.0, -0.02, scope_label, fontsize=7.5, color="0.45", va="top", wrap=True)

    axb = fig.add_subplot(gs[0, 1])
    bnames, bfrac, bcol = [], [], []
    for r in rows:
        if r["limit"] and r["value"] is not None:
            bnames.append(r["field"])
            frac = r["value"] / r["limit"]
            bfrac.append(frac)
            bcol.append(GREEN if frac <= 1.0 else RED)
    order = list(np.argsort(bfrac))
    bnames = [bnames[i] for i in order]
    bfrac = [bfrac[i] for i in order]
    bcol = [bcol[i] for i in order]
    axb.barh(bnames, bfrac, color=bcol)
    axb.axvline(1.0, color="0.2", lw=1.4, ls="--")
    axb.text(1.0, len(bnames) - 0.4, " limit", fontsize=8, color="0.2", va="top")
    axb.set_xlabel("pooled metric / frozen limit  (<1 = within)", fontsize=9)
    axb.set_title("per-variable margin vs frozen tolerance", fontsize=10)
    axb.tick_params(labelsize=8)
    axb.grid(True, axis="x", alpha=0.25)
    axb.set_xlim(0, max(2.0, (max(bfrac) * 1.1) if bfrac else 2.0))

    axs = fig.add_subplot(gs[1, 0])
    lead_index = {lead: i for i, lead in enumerate(leads)}
    norm = np.full((len(rows), len(leads)), np.nan)
    for ri, r in enumerate(rows):
        metric, _lab, limit = field_scored_metric(r["field"], tolerances)
        if not limit:
            continue
        for row in field_metrics[r["field"]].get("by_lead", []):
            lead = int(row["lead_h"])
            if lead in lead_index and row.get(metric) is not None:
                norm[ri, lead_index[lead]] = row[metric] / limit
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "sb", [(0.0, GREEN), (0.49, "#d9f0d3"), (0.5, "#ffffff"), (0.6, "#fddbc7"), (1.0, RED)])
    disp = np.clip(np.ma.masked_invalid(norm) / 2.0, 0.0, 1.0)
    im = axs.imshow(disp, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    axs.set_yticks(range(len(rows)))
    axs.set_yticklabels([r["field"] for r in rows], fontsize=8)
    step = max(1, len(leads) // 12)
    axs.set_xticks(range(0, len(leads), step))
    axs.set_xticklabels([str(leads[i]) for i in range(0, len(leads), step)], fontsize=7)
    axs.set_xlabel("lead hour", fontsize=9)
    axs.set_title("scoreboard: per-lead error / limit (green<1)", fontsize=10)
    cb = fig.colorbar(im, ax=axs, shrink=0.85, ticks=[0, 0.5, 1.0])
    cb.ax.set_yticklabels(["0", "limit", ">=2x"], fontsize=7)

    axt = fig.add_subplot(gs[1, 1])
    axt.axis("off")
    header = f"{'field':>8} {'metric':>7} {'value':>10} {'limit':>9} {'corr':>8}  ok"
    tlines = [header, "-" * len(header)]
    for r in rows:
        ok = "  -" if r["within"] is None else ("  Y" if r["within"] else "  N")
        tlines.append(f"{r['field']:>8} {r['metric']:>7} {fmt(r['value']):>10} "
                      f"{fmt(r['limit']):>9} {fmt(r['corr'],4):>8} {ok}")
    axt.text(0.0, 1.0, "\n".join(tlines), fontsize=8.2, family="monospace", va="top")
    axt.set_title("pooled all-cell / all-lead metrics", fontsize=10, loc="left")

    fig.suptitle("WRF GPU port identity proof  --  offline CPU-WRF vs GPU wrfout comparison (no model rerun)",
                 fontsize=13, weight="bold")
    fig.subplots_adjust(left=0.06, right=0.97, top=0.91, bottom=0.07)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path), {"all_within": all_pass, "n_within": n_within, "n_scored": n_scored,
                       "worst_field": worst[0] if worst else None,
                       "worst_fraction_of_limit": worst[1] if worst else None}


# ------------------------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------------------------
def compute_field_metrics(pairs, tolerances, fields, relative_floor):
    union, inventory = atlas.build_field_union(pairs)
    wanted = [f for f in fields if f in union]
    field_metrics: dict[str, Any] = {}
    for name in wanted:
        summary, _issues = atlas.compare_field(
            name, pairs, inventory["first_metadata"].get(name, {}),
            tolerances.get(name), relative_floor)
        field_metrics[name] = summary
    return field_metrics, inventory


def run(args: argparse.Namespace) -> dict[str, Any]:
    mpl, plt, perr = import_pyplot()
    if plt is None:
        raise SystemExit(f"matplotlib required for identity-proof plots: {perr}")

    case_specs = atlas.case_specs_from_args(args)
    tolerances, tol_meta = atlas.load_tolerances(args.tolerance_json)
    pairs, pairing = atlas.build_pairs(case_specs, args.min_lead, args.max_lead)
    if not pairs:
        raise SystemExit("no paired wrfout files found after domain/lead filtering")

    identity_fields = list(args.field) if args.field else list(DEFAULT_IDENTITY_FIELDS)
    field_metrics, _inventory = compute_field_metrics(pairs, tolerances, identity_fields, args.relative_floor)
    compared = [f for f in identity_fields if field_metrics.get(f, {}).get("status") == "compared"]
    leads = sorted({int(p.lead_h) for p in pairs})

    region = args.region_label or args.case_id or case_specs[0].case_id
    init_dt = case_specs[0].init_time
    init_label = f"init {init_dt.isoformat()}" if init_dt else "init: inferred"
    scope_label = ("Honest scope: differences shown at true scale; variable set = predeclared focused-writer "
                   "hard-gate fields; report-only / bounded fields labelled, breaches drawn red.")
    title = f"{region}"

    asset = args.asset_dir
    asset.mkdir(parents=True, exist_ok=True)
    plots: list[dict[str, str]] = []

    p1 = plot_timeseries_panels(plt, compared, field_metrics, tolerances, scope_label, title,
                                asset / "identity_timeseries_rmse_bias.png")
    if p1:
        plots.append({"kind": "timeseries_rmse_bias", "path": p1})

    p2 = plot_scoreboard(plt, mpl, compared, field_metrics, tolerances, leads, scope_label, title,
                         asset / "identity_scoreboard.png")
    if p2:
        plots.append({"kind": "scoreboard", "path": p2})

    scatter_fields = [f for f in (args.scatter_field or DEFAULT_SCATTER_FIELDS) if f in compared]
    p3 = plot_identity_scatter(plt, scatter_fields, pairs, field_metrics, tolerances, args.scatter_points,
                               scope_label, title, asset / "identity_scatter_1to1.png")
    if p3:
        plots.append({"kind": "identity_scatter", "path": p3})

    spatial_fields = [f for f in (args.spatial_field or DEFAULT_SPATIAL_FIELDS) if f in compared]
    leads_wanted = args.spatial_lead or [24, 48, 72]
    p4 = plot_spatial_diffs(plt, mpl, spatial_fields, pairs, leads_wanted, scope_label, title,
                            asset / "identity_spatial_diff_maps.png")
    if p4:
        plots.append({"kind": "spatial_diff_maps", "path": p4})

    p5, headline = plot_dashboard(plt, mpl, region, init_label, compared, field_metrics, tolerances,
                                  leads, scope_label, asset / "identity_dashboard.png")
    if p5:
        plots.append({"kind": "dashboard", "path": p5})

    field_rows = []
    for name in compared:
        metric, _lab, limit = field_scored_metric(name, tolerances)
        ov = field_metrics[name].get("overall", {})
        val = ov.get(metric)
        field_rows.append({
            "field": name, "scored_metric": metric, "value": val, "limit": limit,
            "within_tolerance": (None if limit is None or val is None else bool(float(val) <= limit)),
            "rmse": ov.get("rmse"), "bias": ov.get("bias"), "max_abs": ov.get("max_abs"),
            "p99_abs": ov.get("p99_abs"), "correlation": ov.get("correlation"),
            "finite_pair_fraction": ov.get("finite_pair_fraction"),
        })

    manifest = {
        "schema": "identity-proof-plots-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_only": True,
        "gpu_used": False,
        "command": " ".join(sys.argv),
        "tool": "scripts/build_identity_proof_plots.py",
        "reuses": "scripts/build_grid_delta_atlas.py (pairing, parsing, tolerances, statistics)",
        "region_label": region,
        "init_time_utc": init_dt.isoformat() if init_dt else None,
        "honest_scope": scope_label,
        "identity_fields_requested": identity_fields,
        "identity_fields_compared": compared,
        "leads_h": leads,
        "lead_count": len(leads),
        "paired_file_count": pairing["paired_file_count"],
        "tolerances": tol_meta,
        "headline": headline,
        "field_metrics": field_rows,
        "plots": plots,
        "asset_dir": str(asset),
    }
    args.proof_dir.mkdir(parents=True, exist_ok=True)
    atlas.write_json(args.proof_dir / "identity_proof_manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("inputs")
    g.add_argument("--cpu-dir", type=Path, help="CPU-WRF wrfout directory.")
    g.add_argument("--gpu-dir", type=Path, help="GPU wrfout directory.")
    g.add_argument("--case-id")
    g.add_argument("--case-json", type=Path, help="Multi-case JSON (same schema as the atlas tool).")
    g.add_argument("--domain", action="append", help="Domain filter, repeatable, e.g. d01.")
    g.add_argument("--init", help="Init time ISO UTC.")
    g.add_argument("--min-lead", type=int, default=None)
    g.add_argument("--max-lead", type=int, default=None)
    g.add_argument("--tolerance-json", type=Path, help="Tolerance manifest (frozen limits).")
    g.add_argument("--region-label", help="Human label for titles/dashboard.")
    g.add_argument("--field", action="append", help="Identity-proof field subset (default: focused-writer hard-gate set).")
    g.add_argument("--scatter-field", action="append", help="Override scatter field set.")
    g.add_argument("--spatial-field", action="append", help="Override spatial diff-map field set.")
    g.add_argument("--spatial-lead", action="append", type=int, help="Leads for spatial maps (default 24 48 72).")
    g.add_argument("--scatter-points", type=int, default=120000, help="Pooled subsample budget per scatter panel.")
    g.add_argument("--relative-floor", type=float, default=1.0e-12)

    o = p.add_argument_group("outputs")
    o.add_argument("--proof-dir", type=Path, default=DEFAULT_PROOF_DIR)
    o.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run(args)
    compact = {
        "region": manifest["region_label"],
        "fields_compared": len(manifest["identity_fields_compared"]),
        "leads": manifest["lead_count"],
        "headline": manifest["headline"],
        "plot_count": len(manifest["plots"]),
        "manifest": str(args.proof_dir / "identity_proof_manifest.json"),
        "asset_dir": manifest["asset_dir"],
    }
    print(json.dumps(compact, sort_keys=True, default=atlas.json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
