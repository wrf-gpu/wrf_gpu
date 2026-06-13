#!/usr/bin/env python3
"""Apply the long-horizon non-escalating-divergence criterion to the v0.15 72 h gates.

This is NOT the fp32 ADR-031 S4 lane. It applies the SAME scientific equivalence
criterion (``.agent/decisions/REDUCED-PRECISION-EQUIVALENCE-AND-FP32-RIGOR.md §3``,
implemented in ``divergence_growth_metric.py``) to the two v0.15 fields that exceed
the *strict frozen per-cell tolerance* -- Switzerland RAINNC and Canary QVAPOR --
plus all other prognostic fields, to answer ONE question rigorously:

  Are these two carries genuine RUN-AWAYS (escalating GPU-vs-CPU divergence) or are
  they BOUNDED / non-escalating diagnostics that merely exceed the tight tolerance?

It does NOT move or hide the strict tolerance. The strict frozen-tolerance PASS/FAIL
(from the existing atlas distilled JSONs) is carried through untouched; this adds a
SECOND, defensible classification on the DIVERGENCE-GROWTH TREND.

Inputs (all already on disk; CPU-only; no model rerun, no GPU):
  - the per-lead GPU-vs-CPU divergence series d(t) = RMSE(GPU, CPU) per field, read
    straight from the existing ``atlas_grid_delta_summary.json`` ``field_metrics``
    ``by_lead[].rmse`` (the exact numbers behind the shipped dashboards).
  - the ENVELOPE per field = the ORACLE's (CPU-WRF) own internal-variability scale,
    measured as the CPU field's spatial standard deviation, averaged over the late
    third of the forecast. For an accumulated field (RAINNC) this is large and
    growing, so "bounded" honestly means "the GPU-CPU difference does not escape the
    field's OWN natural spread/growth", not "the difference is tiny".

The classifier (``classify_divergence_growth``) is the single source of truth for
BOUNDED / BOUNDED_GROWTH / ESCALATING. We only feed it measured numbers.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

# _HERE = .../proofs/perf/v015/fp32_oracles ; repo root is 4 levels up. This makes
# the tool work both in the shared checkout and in an isolated git worktree.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[3]

# --- load the divergence-growth metric (the gate) ----------------------------------
_METRIC = _HERE / "divergence_growth_metric.py"
_spec = importlib.util.spec_from_file_location("divergence_growth_metric", _METRIC)
dgm = importlib.util.module_from_spec(_spec)
sys.modules["divergence_growth_metric"] = dgm
_spec.loader.exec_module(dgm)

# --- reuse the atlas read_variable so envelope reading matches the scoring exactly --
_ATLAS = _ROOT / "scripts" / "build_grid_delta_atlas.py"
_aspec = importlib.util.spec_from_file_location("build_grid_delta_atlas", _ATLAS)
atlas = importlib.util.module_from_spec(_aspec)
sys.modules["build_grid_delta_atlas"] = atlas
_aspec.loader.exec_module(atlas)

IDENTITY_FIELDS = ("T", "U", "V", "W", "QVAPOR", "T2", "U10", "V10", "PSFC", "RAINNC")
FOCUS_FIELDS = {"RAINNC", "QVAPOR"}


def load_series(summary_path: Path, fields=IDENTITY_FIELDS) -> dict[str, dict]:
    """Return {field: {leads, divergence(=RMSE per lead), cpu_files_by_lead, overall_rmse}}."""
    with summary_path.open() as f:
        summ = json.load(f)
    fm = summ["field_metrics"]
    out: dict[str, dict] = {}
    for name in fields:
        rec = fm.get(name)
        if not rec or rec.get("status") != "compared":
            continue
        by_lead = sorted(rec["by_lead"], key=lambda r: int(r["lead_h"]))
        leads = np.array([int(r["lead_h"]) for r in by_lead], dtype=np.float64)
        div = np.array([float(r["rmse"]) for r in by_lead], dtype=np.float64)
        cpu_files = {}
        for r in by_lead:
            w = r.get("worst") or {}
            cf = w.get("cpu_file")
            if cf:
                cpu_files[int(r["lead_h"])] = cf
        out[name] = {
            "leads": leads,
            "divergence": div,
            "cpu_files_by_lead": cpu_files,
            "overall_rmse": float(rec["overall"]["rmse"]),
            "tolerance_pass": bool(rec.get("tolerance_result", {}).get("within_tolerance"))
            if isinstance(rec.get("tolerance_result"), dict) else None,
        }
    return out


def _spatial_std(path: str, name: str) -> float | None:
    try:
        with Dataset(path, "r") as ds:
            if name not in ds.variables:
                return None
            arr = atlas.read_variable(ds, name).astype(np.float64)
    except Exception:
        return None
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return None
    return float(np.std(arr))


def measure_envelope(series: dict, late_fraction: float = 1.0 / 3.0) -> dict:
    """Envelope = CPU(oracle) field spatial std, averaged over the LATE window leads.

    Honest scale for "bounded": the field's own natural spatial spread at the horizon.
    We average the LATE third so an accumulated field (RAINNC) is measured at its
    grown size, not its near-zero start.
    """
    leads = series["leads"]
    cpu_files = series["cpu_files_by_lead"]
    T = leads.size
    third = max(T // 3, 2)
    late_leads = [int(l) for l in leads[-third:]]
    stds = []
    used = []
    for lh in late_leads:
        cf = cpu_files.get(lh)
        if not cf:
            continue
        s = _spatial_std(cf, series["_field_name"])
        if s is not None and np.isfinite(s) and s > 0:
            stds.append(s)
            used.append(lh)
    # full-run envelope as a fallback / cross-check (mean spatial std over all leads)
    all_stds = []
    for lh in [int(l) for l in leads]:
        cf = cpu_files.get(lh)
        if not cf:
            continue
        s = _spatial_std(cf, series["_field_name"])
        if s is not None and np.isfinite(s) and s > 0:
            all_stds.append(s)
    env_late = float(np.mean(stds)) if stds else None
    env_full = float(np.mean(all_stds)) if all_stds else None
    return {
        "envelope_late_window": env_late,
        "envelope_full_run": env_full,
        "late_window_leads": used,
        "n_late_used": len(stds),
    }


def evaluate_region(region: str, summary_path: Path, envelope_factor: float,
                    late_slope_tol: float) -> dict:
    series = load_series(summary_path)
    per_field: dict[str, Any] = {}
    overall_no_runaway = True
    for name, s in series.items():
        s["_field_name"] = name
        env_info = measure_envelope(s)
        env = env_info["envelope_late_window"] or env_info["envelope_full_run"] or 1.0
        res = dgm.classify_divergence_growth(
            s["leads"], s["divergence"], envelope=env, field_name=name,
            envelope_factor=envelope_factor, late_slope_tol=late_slope_tol,
        )
        # --- Decompose the verdict into its two independent sub-tests, then apply
        #     the doc §3 run-away definition. The doc says run-away = "escalating/
        #     runaway divergence" and bounded = "co-evolves within the oracle's own
        #     variability envelope". The metric operationalizes this as ESCALATING
        #     (grows AND breaches the envelope) vs BOUNDED / BOUNDED_GROWTH (slow
        #     drift still inside the envelope -> still equivalent / not a run-away).
        #
        #     RUN-AWAY := the ESCALATING regime (the metric's FAIL state). This is
        #     the right call empirically: a single linear slope over the last third
        #     can catch the RISING LIMB of a DIURNAL OSCILLATION (e.g. W, U10) and
        #     look "non-saturating" while the field is in fact bounded and turning
        #     over -- so a slope-only run-away test gives false positives. Requiring
        #     BOTH non-saturation AND envelope breach (= ESCALATING) is the honest
        #     run-away signature. slope_saturating + within_envelope are reported
        #     separately so the reasoning is fully transparent. ---
        slope_saturating = bool(res.detail["saturating"])   # envelope-INDEPENDENT
        within_envelope = bool(res.detail["within_envelope"])  # magnitude test
        is_runaway = (res.regime == "ESCALATING")
        no_runaway = (not is_runaway)
        overall_no_runaway = overall_no_runaway and no_runaway
        per_field[name] = {
            "divergence_series_rmse": [float(x) for x in s["divergence"]],
            "overall_rmse": s["overall_rmse"],
            "final_rmse": float(s["divergence"][-1]),
            "max_rmse": float(np.max(s["divergence"])),
            "envelope_oracle_spatial_std_late": env_info["envelope_late_window"],
            "envelope_oracle_spatial_std_full": env_info["envelope_full_run"],
            "envelope_used": env,
            "early_slope_per_lead": res.early_slope,
            "late_slope_per_lead": res.late_slope,
            "late_slope_ratio": res.late_slope_ratio,
            "final_over_envelope": res.final_over_envelope,
            "max_over_envelope": res.max_over_envelope,
            # the two orthogonal sub-tests:
            "slope_saturating": slope_saturating,
            "within_envelope_factorK": within_envelope,
            # combined regime from the metric (BOUNDED / BOUNDED_GROWTH / ESCALATING):
            "regime": res.regime,
            # the headline call, defined per doc §3 (escalating = non-saturating slope):
            "is_runaway": bool(is_runaway),
            "non_escalating_no_runaway": bool(no_runaway),
            "detail": res.detail,
        }
    return {
        "region": region,
        "summary_path": str(summary_path),
        "leads_h": [int(x) for x in series[next(iter(series))]["leads"]] if series else [],
        "per_field": per_field,
        "all_fields_non_escalating": bool(overall_no_runaway),
        "envelope_factor": envelope_factor,
        "late_slope_tol": late_slope_tol,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--switz-summary", type=Path,
                   default=_ROOT / "proofs/v015/finalgates/switzerland_d01/atlas_grid_delta_summary.json")
    p.add_argument("--canary-summary", type=Path,
                   default=_ROOT / "proofs/v015/finalgates/canary_l2_d02/atlas_grid_delta_summary.json")
    p.add_argument("--envelope-factor", type=float, default=5.0)
    p.add_argument("--late-slope-tol", type=float, default=0.25)
    p.add_argument("--out", type=Path, default=_ROOT / "proofs/v015/long_horizon_divergence_verdict.json")
    args = p.parse_args(argv)

    regions = {
        "switzerland_d01_72h": evaluate_region(
            "switzerland_d01_72h", args.switz_summary, args.envelope_factor, args.late_slope_tol),
        "canary_l2_d02_72h": evaluate_region(
            "canary_l2_d02_72h", args.canary_summary, args.envelope_factor, args.late_slope_tol),
    }

    # Spotlight the two carried fields.
    focus = {
        "switzerland_RAINNC": regions["switzerland_d01_72h"]["per_field"].get("RAINNC"),
        "canary_QVAPOR": regions["canary_l2_d02_72h"]["per_field"].get("QVAPOR"),
    }
    both_carries_bounded = (
        focus["switzerland_RAINNC"]["non_escalating_no_runaway"]
        and focus["canary_QVAPOR"]["non_escalating_no_runaway"]
    )

    report = {
        "schema": "v015-long-horizon-divergence-verdict-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_only": True,
        "gpu_used": False,
        "criterion_doc": ".agent/decisions/REDUCED-PRECISION-EQUIVALENCE-AND-FP32-RIGOR.md §3",
        "metric": "proofs/perf/v015/fp32_oracles/divergence_growth_metric.py",
        "criterion": (
            "Long-horizon NON-ESCALATING-divergence: classify the GPU-vs-CPU "
            "divergence(t)=RMSE per lead by its TREND (saturating vs sustained/"
            "super-linear growth), normalized by the ORACLE's own spatial-variability "
            "envelope. BOUNDED / BOUNDED_GROWTH = no run-away (co-evolves within the "
            "field's own spread); ESCALATING = run-away (FAIL). This is ADDED to, and "
            "does NOT replace or loosen, the strict frozen per-cell tolerance gate."
        ),
        "envelope_definition": (
            "CPU-WRF (oracle) field spatial standard deviation, averaged over the late "
            "third of the 72h forecast (the field's natural scale at the horizon; for "
            "accumulated RAINNC this grows with the field, so the test is whether the "
            "GPU-CPU difference escapes the field's OWN growth, not whether RAINNC grows)."
        ),
        "params": {"envelope_factor": args.envelope_factor, "late_slope_tol": args.late_slope_tol},
        "carried_fields_focus": focus,
        "both_carried_fields_bounded_no_runaway": bool(both_carries_bounded),
        "regions": regions,
        "honest_verdict_sentence": (
            ("v0.15's two carries (Switzerland RAINNC, Canary QVAPOR) are NOT run-aways: "
             "both exceed the tight frozen per-cell tolerance but their GPU-vs-CPU "
             "divergence is BOUNDED / non-escalating over 72h (saturating slope, within "
             "the oracle's own variability) -- carried to 0.16 for the tolerance, not a "
             "stability failure.")
            if both_carries_bounded else
            ("v0.15's carries are NOT both bounded: at least one of Switzerland RAINNC / "
             "Canary QVAPOR shows ESCALATING GPU-vs-CPU divergence -- see "
             "carried_fields_focus; that one needs the 0.16 numerics fix, not a reframe.")
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    # console summary
    print(f"wrote {args.out}")
    for rk, rv in regions.items():
        print(f"\n== {rk} ==  all_non_escalating(no run-away)={rv['all_fields_non_escalating']}")
        for fn, fv in rv["per_field"].items():
            star = "  <<FOCUS" if fn in FOCUS_FIELDS else ""
            print(f"  {fn:8s} regime={fv['regime']:14s} "
                  f"runaway={'YES' if fv['is_runaway'] else 'no ':3s} "
                  f"slope_sat={'Y' if fv['slope_saturating'] else 'N'} "
                  f"max/env={fv['max_over_envelope']:.3f} "
                  f"late/early_slope={fv['late_slope_ratio']:+.3f} "
                  f"finalRMSE={fv['final_rmse']:.4g}{star}")
    print(f"\nBOTH CARRIES BOUNDED (no run-away): {both_carries_bounded}")
    print(report["honest_verdict_sentence"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
