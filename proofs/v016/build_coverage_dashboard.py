#!/usr/bin/env python
"""Build the v0.16 coupled-coverage release dashboard.

CPU-only.  Joins the recomputable v0.16 coverage map with the per-scheme
``*_gate.json`` verdicts produced by ``coupled_coverage_gate.py`` and emits a
release-ready markdown table plus plots.

Default inputs are this worktree's ``proofs/v016/coverage_map.json`` and
``proofs/v016/coverage``.  During an active sweep, point the script at the live
verdict directory:

  python proofs/v016/build_coverage_dashboard.py \
      --coverage-map <worktree>/proofs/v016/coverage_map.json \
      --coverage-dir <worktree>/proofs/v016/coverage

All paths embedded in the published artifacts are rendered repo-relative (see
``_repo_rel``) so a clean public tree never leaks a local worktree path.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


HERE = Path(__file__).resolve().parent
DEFAULT_MAP = HERE / "coverage_map.json"
DEFAULT_COVERAGE_DIR = HERE / "coverage"
DEFAULT_OUT = HERE / "dashboard"

# Repo root (…/<repo>/proofs/v016/build_coverage_dashboard.py → <repo>). Used to
# emit ONLY repo-relative paths in the published artifacts, so a clean public
# tree never leaks a local worktree absolute path (e.g. /home/<user>/…/.wt-*).
REPO_ROOT = HERE.parent.parent


def _repo_rel(path: Path) -> str:
    """Render ``path`` relative to the repo root; never leak an absolute path.

    Falls back to the basename if the input is outside the repo (e.g. a live
    sweep dir in a sibling worktree), so the published artifact stays portable.
    """
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.name


TOKEN_TO_FAMILY = {
    "mp": "mp_physics",
    "pbl": "bl_pbl_physics",
    "sfclay": "sf_sfclay_physics",
    "cu": "cu_physics",
    "sw": "ra_sw_physics",
    "lw": "ra_lw_physics",
    "lsm": "sf_surface_physics",
}
FAMILY_TO_TOKEN = {v: k for k, v in TOKEN_TO_FAMILY.items()}
FAMILY_LABEL = {
    "mp_physics": "Microphysics",
    "bl_pbl_physics": "PBL",
    "sf_sfclay_physics": "Surface layer",
    "cu_physics": "Cumulus",
    "sf_surface_physics": "Land surface",
    "ra_sw_physics": "Shortwave",
    "ra_lw_physics": "Longwave",
}
DYNAMICS_FIELDS = {"T", "U", "V", "W", "PSFC"}
REVIEW_CEILING = 3.0
GATE_RE = re.compile(r"^([a-z]+)([0-9]+)_gate\.json$")

# SCOPED_CARRY is its own status: a recognized scheme that is not coupled-runnable
# on this real case without an additional data build.  It is an honest, scoped
# carry — NOT a missing/unreadable proof — so it gets its own count and colour and
# must never be rendered grey "MISSING".
STATUS_ORDER = ["GREEN", "SCOPED_CARRY", "REVIEW", "RED", "PENDING", "MISSING"]
STATUS_COLOR = {
    "GREEN": "#2ca25f",
    "SCOPED_CARRY": "#3182bd",  # distinct blue — a scoped carry, NOT a missing proof
    "REVIEW": "#f0ad4e",
    "RED": "#d73027",
    "PENDING": "#bdbdbd",
    "MISSING": "#6c6c6c",
}
STATUS_LABEL = {
    "GREEN": "GREEN",
    "SCOPED_CARRY": "SCOPED CARRY",
    "REVIEW": "REVIEW",
    "RED": "RED",
    "PENDING": "PENDING",
    "MISSING": "MISSING",
}
# Lower-case summary labels so the headline reads "24 GREEN + 1 scoped carry",
# consistent with coverage_rollup.json (n_green / n_carry / ALL_GREEN_OR_CARRIED).
_SUMMARY_LABEL = {
    "GREEN": "GREEN",
    "SCOPED_CARRY": "scoped carry",
    "REVIEW": "REVIEW",
    "RED": "RED",
    "PENDING": "PENDING",
    "MISSING": "MISSING",
}


@dataclass
class GateRecord:
    path: Path
    payload: dict[str, Any] | None
    error: str | None = None


@dataclass
class Metric:
    leaf: str
    field: str
    rmse: float
    manifest_limit: float
    rmse_over_manifest: float
    threshold_multiplier: float

    @property
    def pass_limit(self) -> float:
        return self.manifest_limit * self.threshold_multiplier

    @property
    def rmse_over_pass_limit(self) -> float:
        if self.threshold_multiplier == 0:
            return math.inf
        return self.rmse_over_manifest / self.threshold_multiplier


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _fmt_float(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except Exception:
        return "n/a"
    if not math.isfinite(v):
        return "inf" if v > 0 else "-inf"
    av = abs(v)
    if av == 0.0:
        return "0"
    if av < 1.0e-3 or av >= 1.0e4:
        return f"{v:.3e}"
    if av < 1.0:
        return f"{v:.4g}"
    return f"{v:.4g}"


def _scheme_tag(family: str, option: int) -> str:
    return f"{FAMILY_TO_TOKEN.get(family, family)}{option}"


def _family_from_gate_token(token: str) -> str:
    return TOKEN_TO_FAMILY.get(token, token)


def _parse_gate_filename(path: Path) -> tuple[str, int] | None:
    match = GATE_RE.match(path.name)
    if not match:
        return None
    return _family_from_gate_token(match.group(1)), int(match.group(2))


def _load_gates(coverage_dir: Path) -> dict[tuple[str, int], GateRecord]:
    gates: dict[tuple[str, int], GateRecord] = {}
    if not coverage_dir.exists():
        return gates
    for path in sorted(coverage_dir.glob("*_gate.json")):
        key = _parse_gate_filename(path)
        try:
            payload = _read_json(path)
            family = _family_from_gate_token(str(payload.get("family", "")))
            option = int(payload.get("option"))
            key = (family, option)
            gates[key] = GateRecord(path=path, payload=payload)
        except Exception as exc:  # partial writes during the sweep must not kill dashboard generation
            if key is not None:
                gates[key] = GateRecord(path=path, payload=None, error=f"{type(exc).__name__}: {exc}")
    return gates


def _gate_status(gate: GateRecord | None) -> tuple[str, str]:
    if gate is None:
        return "PENDING", "not run"
    if gate.error or gate.payload is None:
        return "MISSING", gate.error or "unreadable gate"
    raw = str(gate.payload.get("verdict", "")).upper()
    if raw == "PASS":
        return "GREEN", raw
    if raw == "SCOPED_CARRY":
        # An explicit, scoped carry (a recognized scheme not coupled-runnable on
        # this real case without an additional data build) — NOT a missing proof.
        return "SCOPED_CARRY", raw
    if raw == "REVIEW":
        return "REVIEW", raw
    if raw == "FAIL":
        return "RED", raw
    return "MISSING", raw or "missing verdict"


def _is_dynamics_metric(rec: dict[str, Any]) -> bool:
    return str(rec.get("manifest_field", "")) in DYNAMICS_FIELDS


def _threshold_multiplier(payload: dict[str, Any]) -> float:
    deltas = payload.get("field_deltas_vs_baseline") or {}
    if deltas.get("family_perturbs_dynamics"):
        return REVIEW_CEILING
    return 1.0


def _extract_key_metric(gate: GateRecord | None) -> Metric | None:
    if gate is None or gate.payload is None or gate.error:
        return None
    payload = gate.payload
    deltas = payload.get("field_deltas_vs_baseline") or {}
    gated = deltas.get("gated_fields") or {}
    candidates: list[tuple[str, dict[str, Any]]] = [
        (leaf, rec) for leaf, rec in gated.items() if isinstance(rec, dict) and _is_dynamics_metric(rec)
    ]
    if not candidates:
        candidates = [(leaf, rec) for leaf, rec in gated.items() if isinstance(rec, dict)]
    best: tuple[str, dict[str, Any]] | None = None
    best_ratio = -math.inf
    for leaf, rec in candidates:
        try:
            ratio = float(rec["rmse_over_limit"])
            rmse = float(rec["rmse"])
            limit = float(rec["manifest_rmse_limit"])
        except Exception:
            continue
        if not (math.isfinite(rmse) and limit > 0):
            continue
        if ratio > best_ratio:
            best = (leaf, rec)
            best_ratio = ratio
    if best is None:
        return None
    leaf, rec = best
    return Metric(
        leaf=leaf,
        field=str(rec.get("manifest_field", leaf)),
        rmse=float(rec["rmse"]),
        manifest_limit=float(rec["manifest_rmse_limit"]),
        rmse_over_manifest=float(rec["rmse_over_limit"]),
        threshold_multiplier=_threshold_multiplier(payload),
    )


def _metric_text(metric: Metric | None, status: str, note: str) -> str:
    if metric is None:
        if status == "PENDING":
            return "not run"
        if status == "SCOPED_CARRY":
            return note
        if status == "MISSING":
            return note
        return "no gated RMSE metric"
    return (
        f"{metric.field} RMSE {_fmt_float(metric.rmse)} "
        f"({_fmt_float(metric.rmse_over_manifest)}x manifest)"
    )


def _tolerance_text(metric: Metric | None) -> str:
    if metric is None:
        return "n/a"
    if metric.threshold_multiplier == 1.0:
        return f"{metric.field} RMSE <= {_fmt_float(metric.manifest_limit)}"
    return (
        f"{metric.field} RMSE <= {_fmt_float(metric.pass_limit)} "
        f"({int(metric.threshold_multiplier)}x manifest {_fmt_float(metric.manifest_limit)})"
    )


def _build_rows(coverage_map: dict[str, Any], gates: dict[tuple[str, int], GateRecord]) -> list[dict[str, Any]]:
    rows = []
    for cov_row in coverage_map.get("rows", []):
        if not cov_row.get("l2_target"):
            continue
        family = str(cov_row["family"])
        option = int(cov_row["option"])
        gate = gates.get((family, option))
        status, note = _gate_status(gate)
        if status == "SCOPED_CARRY" and gate and gate.payload:
            # Prefer a short, honest carry summary over the raw verdict token.
            note = "scoped carry (needs real-case data build)"
        metric = _extract_key_metric(gate)
        raw = gate.payload.get("verdict") if gate and gate.payload else None
        rows.append(
            {
                "scheme": _scheme_tag(family, option),
                "family": family,
                "family_label": FAMILY_LABEL.get(family, family),
                "option": option,
                "name": str(cov_row.get("name", "")),
                "status": status,
                "raw_verdict": raw,
                "note": note,
                "gate_file": _repo_rel(gate.path) if gate else None,
                "key_metric": _metric_text(metric, status, note),
                "tolerance": _tolerance_text(metric),
                "metric": metric,
            }
        )
    return rows


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {status: sum(1 for row in rows if row["status"] == status) for status in STATUS_ORDER}


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| scheme | family | option | name | verdict | key metric vs oracle | tolerance |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {scheme} | {family} | {option} | {name} | {status} | {key_metric} | {tolerance} |".format(
                scheme=row["scheme"],
                family=row["family"],
                option=row["option"],
                name=row["name"].replace("|", "\\|"),
                status=row["status"],
                key_metric=row["key_metric"].replace("|", "\\|"),
                tolerance=row["tolerance"].replace("|", "\\|"),
            )
        )
    return "\n".join(lines)


def _summary_line(counts: dict[str, int], total: int) -> str:
    # Render e.g. "24 GREEN + 1 scoped carry" so the dashboard reads consistently
    # with coverage_rollup.json (n_green=24, n_carry=1, ALL_GREEN_OR_CARRIED).
    parts = [f"{counts[s]} {_SUMMARY_LABEL[s]}" for s in STATUS_ORDER if counts.get(s, 0)]
    return f"{total} L2 targets: " + " + ".join(parts)


def _write_markdown(
    out_dir: Path,
    rows: list[dict[str, Any]],
    coverage_map_path: Path,
    coverage_dir: Path,
    generated_at: str,
) -> None:
    counts = _counts(rows)
    total = len(rows)
    table = _markdown_table(rows)
    summary = _summary_line(counts, total)
    common_intro = (
        f"Generated: {generated_at}\n\n"
        f"Inputs: `{_repo_rel(coverage_map_path)}` and `{_repo_rel(coverage_dir)}`.\n\n"
        "Only an explicit gate verdict of `PASS` is classified as GREEN. "
        "`SCOPED_CARRY` is a distinct status — a recognized scheme that is not "
        "coupled-runnable on this real case without an additional data build (an "
        "honest, scoped carry, **not** a missing proof). `FAIL` is RED, absent "
        "gate files are PENDING, and unreadable/incomplete gate files are MISSING. "
        "`REVIEW` is kept separate and is not counted as green.\n\n"
        f"Summary: **{summary}**.\n\n"
    )
    status_md = (
        "# v0.16 Coupled-Coverage Dashboard\n\n"
        + common_intro
        + "![Coverage grid](coverage_grid.png)\n\n"
        + "![Metric headroom](metric_headroom.png)\n\n"
        + table
        + "\n"
    )
    (out_dir / "COVERAGE_STATUS.md").write_text(status_md)

    release_md = (
        "## v0.16 Coupled-Coverage Status\n\n"
        + common_intro
        + "The first plot shows GREEN/SCOPED-CARRY/RED/REVIEW/PENDING coverage "
        "across every L2 target by family. The second plot shows each completed "
        "scheme's worst dynamics RMSE divided by the applicable pass threshold on "
        "a log scale; values below 1.0 are inside the gate threshold.\n\n"
        + "![v0.16 coverage grid](coverage_grid.png)\n\n"
        + "![v0.16 metric headroom](metric_headroom.png)\n\n"
        + table
        + "\n"
    )
    (out_dir / "RELEASE_COVERAGE_SECTION.md").write_text(release_md)


def _plot_coverage_grid(rows: list[dict[str, Any]], out: Path) -> None:
    families: list[str] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        fam = row["family"]
        if fam not in grouped:
            families.append(fam)
            grouped[fam] = []
        grouped[fam].append(row)

    max_cols = max((len(v) for v in grouped.values()), default=1)
    fig_w = max(9.0, max_cols * 1.25)
    fig_h = max(3.6, len(families) * 0.72 + 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for y, fam in enumerate(families):
        for x, row in enumerate(grouped[fam]):
            status = row["status"]
            rect = Rectangle((x, y), 1, 1, facecolor=STATUS_COLOR[status], edgecolor="white", linewidth=1.5)
            ax.add_patch(rect)
            text_color = "white" if status in {"GREEN", "SCOPED_CARRY", "RED", "MISSING"} else "black"
            ax.text(x + 0.5, y + 0.5, row["scheme"], ha="center", va="center", fontsize=8, color=text_color)

    ax.set_xlim(0, max_cols)
    ax.set_ylim(0, len(families))
    ax.set_yticks([i + 0.5 for i in range(len(families))])
    ax.set_yticklabels([FAMILY_LABEL.get(fam, fam) for fam in families], fontsize=9)
    ax.set_xticks([])
    ax.invert_yaxis()
    ax.set_title("v0.16 L2 coupled-coverage targets")
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend = [Patch(facecolor=STATUS_COLOR[s], edgecolor="none", label=STATUS_LABEL[s]) for s in STATUS_ORDER]
    ax.legend(handles=legend, ncol=len(legend), loc="lower center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _plot_metric_headroom(rows: list[dict[str, Any]], out: Path) -> None:
    labels = [row["scheme"] for row in rows]
    x_all = list(range(len(rows)))
    metric_points = []
    pending_points = []
    for i, row in enumerate(rows):
        metric = row["metric"]
        if isinstance(metric, Metric):
            metric_points.append((i, metric.rmse_over_pass_limit, row["status"]))
        else:
            pending_points.append((i, row["status"]))

    values = [v for _, v, _ in metric_points if math.isfinite(v) and v > 0]
    y_min = 1.0e-4
    if values:
        y_min = 10 ** math.floor(math.log10(max(min(values) / 2.0, 1.0e-6)))
    y_max = 10.0
    if values:
        y_max = max(10.0, 10 ** math.ceil(math.log10(max(max(values) * 2.0, 1.0))))

    fig_w = max(10.0, len(rows) * 0.42)
    fig, ax = plt.subplots(figsize=(fig_w, 5.3))
    for status in STATUS_ORDER:
        xs = [i for i, v, s in metric_points if s == status]
        ys = [v for i, v, s in metric_points if s == status]
        if xs:
            ax.scatter(xs, ys, s=48, color=STATUS_COLOR[status], edgecolor="black", linewidth=0.4, label=STATUS_LABEL[status], zorder=3)
    # Rows without a gated RMSE metric (PENDING / MISSING / SCOPED_CARRY) sit at the
    # floor as markers, coloured by their own status so a scoped carry is never
    # rendered as a missing proof.
    for status in STATUS_ORDER:
        xs = [i for i, s in pending_points if s == status]
        if xs:
            ax.scatter(
                xs,
                [y_min] * len(xs),
                s=34,
                color=STATUS_COLOR[status],
                marker="x",
                label=f"{STATUS_LABEL[status]} (no RMSE metric)",
                zorder=2,
            )

    ax.axhline(1.0, color="#d73027", linestyle="--", linewidth=1.2, label="pass threshold")
    ax.set_yscale("log")
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.75, len(rows) - 0.25)
    ax.set_xticks(x_all)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("worst dynamics RMSE / applicable pass threshold")
    ax.set_title("v0.16 metric headroom for completed L2 gates")
    ax.grid(True, axis="y", which="both", linestyle=":", linewidth=0.6, alpha=0.65)
    handles, labels_seen = ax.get_legend_handles_labels()
    dedup: dict[str, Any] = {}
    for handle, label in zip(handles, labels_seen):
        dedup.setdefault(label, handle)
    ax.legend(dedup.values(), dedup.keys(), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _jsonable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        metric = row["metric"]
        rec = {k: v for k, v in row.items() if k != "metric"}
        if isinstance(metric, Metric):
            rec["metric"] = {
                "leaf": metric.leaf,
                "field": metric.field,
                "rmse": metric.rmse,
                "manifest_limit": metric.manifest_limit,
                "rmse_over_manifest": metric.rmse_over_manifest,
                "threshold_multiplier": metric.threshold_multiplier,
                "pass_limit": metric.pass_limit,
                "rmse_over_pass_limit": metric.rmse_over_pass_limit,
            }
        else:
            rec["metric"] = None
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage-map", type=Path, default=DEFAULT_MAP)
    ap.add_argument("--coverage-dir", type=Path, default=DEFAULT_COVERAGE_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    coverage_map = _read_json(args.coverage_map)
    gates = _load_gates(args.coverage_dir)
    rows = _build_rows(coverage_map, gates)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()

    _plot_coverage_grid(rows, args.out_dir / "coverage_grid.png")
    _plot_metric_headroom(rows, args.out_dir / "metric_headroom.png")
    _write_markdown(args.out_dir, rows, args.coverage_map, args.coverage_dir, generated_at)

    summary = {
        "schema": "V016CoverageDashboard",
        "generated_at": generated_at,
        "coverage_map": _repo_rel(args.coverage_map),
        "coverage_dir": _repo_rel(args.coverage_dir),
        "l2_target_count": len(rows),
        "counts": _counts(rows),
        "rows": _jsonable_rows(rows),
    }
    (args.out_dir / "coverage_dashboard.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
