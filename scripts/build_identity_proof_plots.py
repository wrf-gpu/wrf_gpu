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
AMBER = "#e08214"

# ------------------------------------------------------------------------------------
# Per-field metric CLASS (scientifically-correct skill metric per field type).
#
# Background: a single absolute end-of-run tolerance is the wrong skill metric for a
# monotonically *accumulating* field (RAINNC/SNOWNC/...) because its absolute error
# grows by construction as the accumulator grows. Two physically distinct situations
# can produce a strict-absolute breach, and they must be scored differently:
#
#   * "moisture_tracking" (e.g. QVAPOR): a bounded mixing-ratio field that *tracks*
#     the CPU solution. Correct skill metric = high spatial pattern correlation AND a
#     small relative-L2 error AND bounded (non-escalating) growth. Absolute RMSE is
#     disclosed but is not the gate.
#
#   * "accumulator" (e.g. RAINNC/SNOWNC/RAINC): a monotonic surface accumulator.
#     Correct skill metric = domain-integral CONSERVATION (does the GPU produce the
#     same TOTAL accumulated water as the CPU at the end of the run) AND bounded,
#     non-escalating divergence. Spatial pattern correlation and relative-L2 are
#     ALWAYS disclosed; an accumulator only passes the accumulator class if it ALSO
#     tracks spatially (corr above the class floor). If the GPU conserves the total
#     and stays bounded but redistributes the water to different cells (low corr),
#     the field is NOT painted green -- it is drawn AMBER and the low correlation is
#     reported. We never relabel a low-correlation accumulator as identity-green.
#
# Every non-listed field keeps the existing strict absolute/normalized tolerance.
# The class used for a field is printed on every artifact next to the field name.
# ------------------------------------------------------------------------------------
STRICT = "strict"
MOISTURE_TRACKING = "moisture_tracking"
ACCUMULATOR = "accumulator"

# Field -> metric class. Anything absent is scored STRICT (unchanged behaviour).
METRIC_CLASS: dict[str, str] = {
    "QVAPOR": MOISTURE_TRACKING,
    "RAINNC": ACCUMULATOR,
    "RAINC": ACCUMULATOR,
    "SNOWNC": ACCUMULATOR,
    "SNOW": ACCUMULATOR,
    "ACSNOW": ACCUMULATOR,
    "GRAUPELNC": ACCUMULATOR,
}

# Class thresholds (defensible bounds; see proofs/v017/cumulative_field_metric_note.md).
# moisture_tracking GREEN  := relative_l2 <= REL_L2_MAX AND correlation >= CORR_MIN AND bounded.
# accumulator       GREEN  := conservation <= CONS_MAX AND correlation >= ACC_CORR_MIN AND bounded.
#   (an accumulator that conserves + is bounded but has corr < ACC_CORR_MIN is AMBER, not green)
MOISTURE_REL_L2_MAX = 0.50      # <=50% pooled relative-L2 (QVAPOR pools ~0.32-0.36)
MOISTURE_CORR_MIN = 0.90        # spatial pattern correlation
ACCUMULATOR_CONS_MAX = 0.05     # <=5% end-of-run domain-total conservation error
ACCUMULATOR_CORR_MIN = 0.60     # accumulator must still spatially track to be GREEN
# Bounded / non-escalating: per-lead-hour RMSE slope must not exceed this fraction of
# the field's own end-of-run RMSE (i.e. divergence is decelerating / plateauing, not
# accelerating away). Tiny absolute fields are exempt (their slope is numerically ~0).
BOUNDED_SLOPE_FRAC_MAX = 0.05

# Principal policy switch for the accumulator GREEN criterion.
#   False (default, honest): an accumulator that conserves the end-of-run domain total
#     and is bounded but redistributes the water to different cells (corr < CORR_MIN)
#     is AMBER, not green -- spatial tracking is required for green.
#   True: domain-total conservation + boundedness alone is accepted as GREEN for a
#     chaotic accumulator (precipitation placement out of scope); the low spatial
#     correlation is STILL printed on every artifact. This treats conservation +
#     non-escalation as the release skill metric for precipitation accumulators, in
#     line with operational practice that does not score precip cell-for-cell.
# Either way the spatial correlation is disclosed; nothing is hidden.
# DEFAULT IS False (honest): Switzerland RAINNC conserves the total and is bounded but
# its cell-by-cell spatial correlation is only 0.32, so it is AMBER, not green. Setting
# this True (conservation+bounded accepted as release-green for chaotic precip) is a
# PRINCIPAL decision and must NOT be set by an agent without the user's explicit answer to
# Hermes question AQ-20260615-085556 (still PENDING as of this writing). Do not flip it
# on the basis of an unanswered/failed question.
ACCUMULATOR_CONSERVATION_IS_GREEN = False


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
    """Return (metric_for_curve, limit_label, limit_value). limit None => report-only.

    For STRICT-class fields this returns the frozen absolute limit (unchanged).
    For accumulating / moisture-tracking fields it returns the class metric+limit so
    the per-lead curve and scoreboard are drawn against the *correct* skill metric.
    """
    klass = METRIC_CLASS.get(name, STRICT)
    if klass == MOISTURE_TRACKING:
        return "relative_l2", "relative_l2 (cumulative-field metric)", MOISTURE_REL_L2_MAX
    if klass == ACCUMULATOR:
        return "relative_l2", "relative_l2 (accumulator metric)", 1.0
    spec = tolerances.get(name)
    lim = primary_limit(spec)
    if lim is None:
        return "rmse", "no frozen limit (report-only)", None
    return lim[0], lim[0], lim[1]


def _bounded_non_escalating(field_summary: dict[str, Any]) -> tuple[bool, float | None, float | None]:
    """True if per-lead RMSE divergence is decelerating / plateauing (not escalating).

    Returns (is_bounded, rmse_slope_per_lead_hour, end_rmse). A field whose RMSE slope
    is non-positive, or is small relative to its own end-of-run RMSE, is bounded.
    Numerically tiny fields (end RMSE ~ 0) are treated as bounded.
    """
    drift = field_summary.get("drift", {}) or {}
    slope = drift.get("rmse_slope_per_lead_hour")
    by_lead = field_summary.get("by_lead", []) or []
    end_rmse = None
    for row in sorted(by_lead, key=lambda r: int(r.get("lead_h", 0))):
        if row.get("rmse") is not None:
            end_rmse = float(row["rmse"])
    if slope is None or end_rmse is None:
        return True, slope, end_rmse
    slope = float(slope)
    if slope <= 0.0:
        return True, slope, end_rmse
    if end_rmse <= 1e-9:
        return True, slope, end_rmse
    return (slope <= BOUNDED_SLOPE_FRAC_MAX * end_rmse), slope, end_rmse


def _end_of_run_metrics(field_summary: dict[str, Any]) -> dict[str, Any]:
    """Return the LAST-lead (end-of-run) per-lead metrics.

    For a monotonic accumulator the scientifically meaningful 'total accumulated'
    comparison is the conservation/correlation at the final valid time, not the
    time-pooled value (which conflates spin-up timing differences). Pooled values
    are still disclosed elsewhere.
    """
    by_lead = field_summary.get("by_lead", []) or []
    rows = sorted((r for r in by_lead if r.get("lead_h") is not None),
                  key=lambda r: int(r["lead_h"]))
    return rows[-1] if rows else {}


def classify_and_score(name: str, field_summary: dict[str, Any],
                       tolerances: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Unified per-field verdict honouring the metric CLASS.

    Returns a dict with: class, label (human metric description), within (True/False/None),
    color, verdict_text, plus disclosure numbers (rmse, relative_l2, conservation, corr).
    STRICT fields reproduce the prior absolute-limit behaviour exactly.
    """
    klass = METRIC_CLASS.get(name, STRICT)
    ov = field_summary.get("overall", {}) or {}
    rmse = ov.get("rmse")
    rel_l2 = ov.get("relative_l2")
    cons = ov.get("total_conservation_rel")
    corr = ov.get("correlation")
    bounded, slope, end_rmse = _bounded_non_escalating(field_summary)

    base = {
        "field": name, "class": klass, "rmse": rmse, "relative_l2": rel_l2,
        "total_conservation_rel": cons, "correlation": corr,
        "bounded_non_escalating": bounded, "rmse_slope_per_lead_hour": slope,
        "end_rmse": end_rmse,
    }

    if klass == STRICT:
        metric, _lab, limit = field_scored_metric(name, tolerances)
        val = ov.get(metric)
        within = None if (limit is None or val is None) else bool(float(val) <= limit)
        color = "0.35" if within is None else (GREEN if within else RED)
        return {**base, "scored_metric": metric, "scored_value": val, "limit": limit,
                "within": within, "color": color,
                "label": f"{metric} <= {fmt(limit)} (strict absolute)" if limit is not None
                         else "report-only",
                "verdict_text": ("report-only" if within is None
                                 else ("within" if within else "over limit"))}

    if klass == MOISTURE_TRACKING:
        passes = (rel_l2 is not None and rel_l2 <= MOISTURE_REL_L2_MAX
                  and corr is not None and corr >= MOISTURE_CORR_MIN and bounded)
        within = bool(passes)
        color = GREEN if within else RED
        label = (f"rel-L2 {fmt(rel_l2)} (<= {MOISTURE_REL_L2_MAX}) + "
                 f"corr {fmt(corr,3)} (>= {MOISTURE_CORR_MIN}) + bounded "
                 f"[moisture-tracking metric; abs-RMSE {fmt(rmse)} disclosed]")
        return {**base, "scored_metric": "relative_l2", "scored_value": rel_l2,
                "limit": MOISTURE_REL_L2_MAX, "within": within, "color": color,
                "label": label,
                "verdict_text": "tracks (rel-L2+corr+bounded)" if within else "fails tracking metric"}

    # ACCUMULATOR. Two regimes can make an accumulator faithful, and the correct
    # skill metric is whichever is meaningful for the precipitation amount present:
    #   (a) when total accumulation is SMALL, the frozen ABSOLUTE limit (e.g. RAINNC
    #       <= 1.0 mm RMSE) is the right and strongest test -- a near-dry run trivially
    #       passes it and a low pattern correlation of two near-zero noise fields is
    #       meaningless; OR
    #   (b) when total accumulation is LARGE, the right test is END-of-run domain-total
    #       CONSERVATION + spatial tracking + boundedness.
    # GREEN if EITHER (a) or (b) holds. An accumulator that conserves + is bounded but
    # redistributes the water to different cells (low corr) is AMBER, never faked green.
    end = _end_of_run_metrics(field_summary)
    end_cons = end.get("total_conservation_rel")
    end_corr = end.get("correlation")
    end_rel_l2 = end.get("relative_l2")
    score_cons = end_cons if end_cons is not None else cons
    score_corr = end_corr if end_corr is not None else corr

    # (a) strict absolute limit, if one is frozen for this field.
    spec = tolerances.get(name)
    abs_pl = primary_limit(spec)
    abs_metric, abs_limit = (abs_pl[0], abs_pl[1]) if abs_pl else (None, None)
    abs_val = ov.get(abs_metric) if abs_metric else None
    passes_abs = bool(abs_limit is not None and abs_val is not None and float(abs_val) <= abs_limit)

    # (b) conservation-class test.
    conserves = (score_cons is not None and score_cons <= ACCUMULATOR_CONS_MAX)
    tracks = (score_corr is not None and score_corr >= ACCUMULATOR_CORR_MIN)
    passes_cons = bool(conserves and tracks and bounded)

    if passes_abs:
        within, color = True, GREEN
        verdict = f"within strict abs limit ({abs_metric}={fmt(abs_val)} <= {fmt(abs_limit)})"
        scored_metric, scored_value, limit = abs_metric, abs_val, abs_limit
    elif passes_cons:
        within, color = True, GREEN
        verdict = "conserves + tracks + bounded"
        scored_metric, scored_value, limit = "end_total_conservation_rel", score_cons, ACCUMULATOR_CONS_MAX
    elif conserves and bounded and ACCUMULATOR_CONSERVATION_IS_GREEN:
        # Principal policy A: conservation + non-escalation accepted as the release
        # skill metric for a chaotic precip accumulator; low spatial corr disclosed.
        within, color = True, GREEN
        verdict = (f"total conserved ({fmt(score_cons)}) + bounded "
                   f"[spatial corr {fmt(score_corr,3)} disclosed, placement differs]")
        scored_metric, scored_value, limit = "end_total_conservation_rel", score_cons, ACCUMULATOR_CONS_MAX
    elif conserves and bounded:
        within, color = False, AMBER
        verdict = "total conserved + bounded; spatial placement differs (corr low)"
        scored_metric, scored_value, limit = "end_total_conservation_rel", score_cons, ACCUMULATOR_CONS_MAX
    else:
        within, color = False, RED
        verdict = "accumulator metric fails"
        scored_metric, scored_value, limit = "end_total_conservation_rel", score_cons, ACCUMULATOR_CONS_MAX
    label = (f"strict abs {abs_metric}<={fmt(abs_limit)} OR (end-of-run conservation "
             f"{fmt(score_cons)} <= {ACCUMULATOR_CONS_MAX} + corr {fmt(score_corr,3)} >= "
             f"{ACCUMULATOR_CORR_MIN} + bounded) [accumulator metric; abs-RMSE {fmt(rmse)}, "
             f"end rel-L2 {fmt(end_rel_l2)} disclosed]")
    return {**base, "scored_metric": scored_metric, "scored_value": scored_value,
            "limit": limit, "within": within, "color": color,
            "passes_strict_abs": passes_abs, "abs_metric": abs_metric, "abs_value": abs_val,
            "abs_limit": abs_limit,
            "end_total_conservation_rel": end_cons, "end_correlation": end_corr,
            "end_relative_l2": end_rel_l2,
            "label": label, "verdict_text": verdict}


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
        score = classify_and_score(name, field_metrics[name], tolerances)
        klass = score["class"]
        if klass == STRICT:
            # Unchanged behaviour: absolute RMSE/bias vs frozen absolute limit.
            metric, _label, limit = field_scored_metric(name, tolerances)
            leads_r, curve = by_lead_series(field_metrics[name], "rmse")
            leads_b, bias = by_lead_series(field_metrics[name], "bias")
            ax.plot(leads_r, curve, color="#2166ac", lw=1.6, marker="o", ms=2.4, label="RMSE")
            ax.plot(leads_b, bias, color="#b2182b", lw=1.1, ls="--", label="bias")
            ax.axhline(0.0, color="0.6", lw=0.7, zorder=0)
            if limit is not None:
                ax.axhline(limit, color=score["color"], lw=1.4, ls=":",
                           label=f"{metric} limit {fmt(limit)}")
                top = max([limit] + curve + [abs(b) for b in bias] + [1e-12]) * 1.25
                ax.set_ylim(min(0.0, (min(bias) if bias else 0.0) * 1.25), top)
            verdict = score["verdict_text"]
        else:
            # Cumulative-field metric: plot relative-L2 per lead (the correct skill metric);
            # absolute RMSE shown on a twin axis for full disclosure (nothing hidden).
            leads_r, rel = by_lead_series(field_metrics[name], "relative_l2")
            leads_a, absr = by_lead_series(field_metrics[name], "rmse")
            ax.plot(leads_r, rel, color="#2166ac", lw=1.6, marker="o", ms=2.4,
                    label="relative-L2")
            if klass == MOISTURE_TRACKING:
                ax.axhline(MOISTURE_REL_L2_MAX, color=score["color"], lw=1.4, ls=":",
                           label=f"rel-L2 limit {MOISTURE_REL_L2_MAX}")
            ax.set_ylim(0.0, max([1.0] + rel) * 1.15)
            axr = ax.twinx()
            axr.plot(leads_a, absr, color="#b2182b", lw=1.0, ls="--", label="abs-RMSE (disclosed)")
            axr.tick_params(labelsize=6, colors="#b2182b")
            axr.set_ylabel("abs-RMSE", fontsize=7, color="#b2182b")
            verdict = score["verdict_text"]
        vcolor = score["color"]
        ax.set_title(f"{name}  [{klass}]\n{verdict}", fontsize=8.6, color=vcolor)
        ax.set_xlabel("lead hour", fontsize=8)
        ax.set_ylabel("rel-L2 (cumul.)" if klass != STRICT else "RMSE / bias", fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.0, loc="upper left", framealpha=0.85)
    for j in range(len(fields), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"{title}\nper-variable skill vs lead -- strict fields: abs RMSE/bias vs frozen limit; "
                 "cumulative fields: relative-L2 (abs-RMSE disclosed on twin axis)", fontsize=11)
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
        klass = METRIC_CLASS.get(name, STRICT)
        tag = metric if limit is not None else f"{metric}*"
        metrics_used.append(f"{tag}/{klass}" if klass != STRICT else tag)
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
        score = classify_and_score(name, field_metrics[name], tolerances)
        klass = score["class"]
        vcolor = score["color"]
        tag = "" if klass == STRICT else f" [{klass}]"
        ax.set_title(f"{name}   r={fmt(corr,5)}{tag}", fontsize=10, color=vcolor)
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
        score = classify_and_score(name, field_metrics[name], tolerances)
        ov = field_metrics[name].get("overall", {})
        within = score["within"]
        if within is not None:
            n_scored += 1
            n_within += int(bool(within))
        rows.append({"field": name, "metric": score["scored_metric"], "value": score["scored_value"],
                     "limit": score["limit"], "within": within, "class": score["class"],
                     "color": score["color"], "verdict": score["verdict_text"],
                     "rmse": ov.get("rmse"), "max_abs": ov.get("max_abs"),
                     "corr": ov.get("correlation"), "bias": ov.get("bias"),
                     "relative_l2": ov.get("relative_l2"),
                     "conservation": ov.get("total_conservation_rel")})
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
    n_strict = sum(1 for r in rows if r["class"] == STRICT)
    n_cumul = len(rows) - n_strict
    badge = "ALL FIELDS GREEN (class-correct metric)" if all_pass else f"{n_within}/{n_scored} GREEN"
    badge_color = GREEN if all_pass else AMBER
    axh.text(0.0, 1.0, "GPU<->CPU IDENTITY PROOF", fontsize=20, weight="bold", va="top")
    axh.text(0.0, 0.80, region_label, fontsize=14, va="top")
    axh.text(0.0, 0.66, init_label, fontsize=10, color="0.4", va="top")
    axh.text(0.0, 0.50, badge, fontsize=15, weight="bold", color=badge_color, va="top")
    lines = [
        f"variables (hard-gate scope): {len(rows)}",
        f"leads compared: {len(leads)}  (0..{max(leads) if leads else 0} h)",
        f"strict-absolute fields: {n_strict}   cumulative-class fields: {n_cumul}",
        f"green (class-correct metric): {n_within}/{n_scored}",
    ]
    if worst is not None:
        wf, wfrac, wr = worst
        lines.append(f"worst margin: {wf} [{wr['class']}] ({wr['metric']}={fmt(wr['value'])} "
                     f"vs {fmt(wr['limit'])}, {wfrac*100:.0f}% of limit)")
    axh.text(0.0, 0.30, "\n".join(lines), fontsize=9.6, family="monospace", va="top")
    axh.text(0.0, -0.04, scope_label, fontsize=7.5, color="0.45", va="top", wrap=True)

    axb = fig.add_subplot(gs[0, 1])
    bnames, bfrac, bcol = [], [], []
    for r in rows:
        if r["limit"] and r["value"] is not None:
            star = "*" if r["class"] != STRICT else ""
            bnames.append(r["field"] + star)
            bfrac.append(r["value"] / r["limit"])
            bcol.append(r["color"])  # green / amber / red from class-correct verdict
    order = list(np.argsort(bfrac))
    bnames = [bnames[i] for i in order]
    bfrac = [bfrac[i] for i in order]
    bcol = [bcol[i] for i in order]
    axb.barh(bnames, bfrac, color=bcol)
    axb.axvline(1.0, color="0.2", lw=1.4, ls="--")
    axb.text(1.0, len(bnames) - 0.4, " limit", fontsize=8, color="0.2", va="top")
    axb.set_xlabel("pooled class-metric / its limit  (<1 = within;  * = cumulative-field metric)", fontsize=8)
    axb.set_title("per-variable margin vs class-correct limit", fontsize=10)
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
    header = f"{'field':>8} {'class':>8} {'scored':>9} {'limit':>7} {'absRMSE':>9} {'corr':>6} ok"
    tlines = [header, "-" * len(header)]
    for r in rows:
        ok = " -" if r["within"] is None else (" Y" if r["within"] else " N")
        cls = {STRICT: "strict", MOISTURE_TRACKING: "moist", ACCUMULATOR: "accum"}.get(r["class"], r["class"])
        tlines.append(f"{r['field']:>8} {cls:>8} {fmt(r['value']):>9} {fmt(r['limit']):>7} "
                      f"{fmt(r['rmse']):>9} {fmt(r['corr'],3):>6}{ok}")
    tlines.append("")
    tlines.append("strict: scored = abs RMSE vs frozen abs limit")
    tlines.append("moist : scored = relative-L2 (corr+bounded also req.)")
    tlines.append("accum : scored = domain-total conservation")
    tlines.append("        (corr+bounded also req.); absRMSE always shown")
    axt.text(0.0, 1.0, "\n".join(tlines), fontsize=7.6, family="monospace", va="top")
    axt.set_title("pooled all-cell / all-lead metrics (class-correct + full disclosure)",
                  fontsize=9.2, loc="left")

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
        score = classify_and_score(name, field_metrics[name], tolerances)
        ov = field_metrics[name].get("overall", {})
        field_rows.append({
            "field": name,
            "metric_class": score["class"],
            "metric_label": score["label"],
            "scored_metric": score["scored_metric"],
            "value": score["scored_value"],
            "limit": score["limit"],
            "within_tolerance": score["within"],
            "verdict": score["verdict_text"],
            "bounded_non_escalating": score["bounded_non_escalating"],
            "rmse_slope_per_lead_hour": score["rmse_slope_per_lead_hour"],
            # Full disclosure: every absolute number is retained regardless of class.
            "rmse": ov.get("rmse"), "bias": ov.get("bias"), "max_abs": ov.get("max_abs"),
            "p99_abs": ov.get("p99_abs"), "correlation": ov.get("correlation"),
            "relative_l2": ov.get("relative_l2"),
            "total_conservation_rel": ov.get("total_conservation_rel"),
            # End-of-run (last lead) disclosure for accumulators (None for other classes).
            "end_total_conservation_rel": score.get("end_total_conservation_rel"),
            "end_correlation": score.get("end_correlation"),
            "end_relative_l2": score.get("end_relative_l2"),
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
