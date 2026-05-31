"""CPU-ONLY case3 wind-skill regime diagnostic (NO GPU, NO forecast).

Mission STEP 1 (decisive): is the case3 L3 24h V10 below-persistence outcome a
REAL model deficiency, or a REGIME/METRIC limit (a calm/unpredictable regime
where even CPU-WRF barely beats persistence)?

This reuses ONLY existing corpus CPU-WRF wrfout history; NO new WRF runs.

DISCOVERY enabling the decisive test: case3 (init 2026-05-21 18z) has MULTIPLE
independent CPU-WRF forecasts of the SAME init on the SAME d02 (66x159) grid:
  - L3 run1 (133443Z): 0..24 h full hourly  <- the persistence-baseline truth
  - L2   run (133443Z): 0..19 h hourly      <- coarser-parent, independent
  - L3 run2 (072630Z): 0..8  h hourly       <- second independent L3
So we can run the EXACT same "WRF self-spread = irreducible uncertainty lower
bound" cross-RMSE test that proved CPU-WRF skillful on case2, now for case3.

Three lines of evidence, mirroring proofs/wind/cpu_regime_diagnostic.py:
  (1) PERSISTENCE growth vs TRUTH (L3 run1), hour by hour, U10/V10/T2/SPD.
  (2) CPU-WRF cross-RMSE (the "WRF spread"): L2-vs-L3run1 (leads 1..19h) and
      L3run1-vs-L3run2 (leads 1..8h). If CPU-WRF agrees with CPU-WRF far better
      than persistence's error, a faithful model SHOULD beat persistence ->
      model deficiency. If WRF self-spread ~ persistence error -> regime/metric
      limit (whole-domain gridded RMSE on this field is not a discriminating
      metric here).
  (3) FIELD MOTION: temporal_change_rmse vs spatial_std (does the field evolve?).

Plus a direct case2-vs-case3 side-by-side of the same numbers, since case2 is
the PROVEN-skillful reference (root-cause doc: case2 V10 24h WRF self-spread
0.13 m/s vs persistence 2.05 m/s -> WRF clearly skillful there).

USAGE
  JAX_PLATFORMS=cpu PYTHONPATH=src OMP_NUM_THREADS=4 taskset -c 0-3 \
    python proofs/wind/case3_regime_diagnostic.py \
      --out proofs/wind/case3_regime_diagnostic.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gpuwrf.io.gen2_wrfout_loader import read_wrfout_file

FIELDS = ("U10", "V10", "T2")
STATIC = ("XLAND",)

# --- case3 (2026-05-21 18z) corpus, all on the same 66x159 d02 grid ---
C3_INIT = datetime(2026, 5, 21, 18, 0, 0, tzinfo=timezone.utc)
C3_L3a = Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z")  # truth, 0..24h
C3_L2 = Path("/mnt/data/canairy_meteo/runs/wrf_l2/20260521_18z_l2_72h_20260522T133443Z")    # 0..19h
C3_L3b = Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T072630Z")    # 0..8h

# --- case2 (2026-05-09 18z) reference (proven skillful) ---
C2_INIT = datetime(2026, 5, 9, 18, 0, 0, tzinfo=timezone.utc)
C2_L2 = Path("/mnt/data/canairy_meteo/runs/wrf_l2/20260509_18z_l2_72h_20260511T190519Z")
C2_L3 = Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260509_18z_l3_24h_20260511T190519Z")


def wrfout(d: Path, valid: datetime) -> Path:
    return d / f"wrfout_d02_{valid:%Y-%m-%d_%H:%M:%S}"


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def load(d: Path, valid: datetime, fields=FIELDS) -> dict[str, np.ndarray] | None:
    p = wrfout(d, valid)
    if not p.is_file():
        return None
    return read_wrfout_file(p, fields=fields)["fields"]


def spd(d: dict[str, np.ndarray]) -> np.ndarray:
    return np.hypot(np.asarray(d["U10"], dtype=np.float64), np.asarray(d["V10"], dtype=np.float64))


def regime(init_dir: Path, init: datetime, truth_dir: Path, b_dir: Path | None,
           b_label: str, c_dir: Path | None, c_label: str, max_lead: int) -> dict[str, Any]:
    """One case: persistence growth + WRF self-spread + field motion."""
    init_f = load(init_dir, init, fields=FIELDS + STATIC)
    assert init_f is not None, f"missing init {init_dir}"
    init_spd = spd(init_f)

    rows: list[dict[str, Any]] = []
    for lead_h in range(1, max_lead + 1):
        valid = init + timedelta(hours=lead_h)
        truth = load(truth_dir, valid)
        if truth is None:
            continue
        row: dict[str, Any] = {"lead_h": lead_h}
        for f in FIELDS:
            row[f"persist_{f}"] = rmse(init_f[f], truth[f])
            row[f"truth_std_{f}"] = float(np.std(np.asarray(truth[f], dtype=np.float64)))
            row[f"truth_mean_{f}"] = float(np.mean(np.asarray(truth[f], dtype=np.float64)))
            # temporal motion of the truth field
            row[f"motion_{f}"] = rmse(init_f[f], truth[f])  # change-from-init RMSE
        row["persist_SPD"] = rmse(init_spd, spd(truth))
        row["truth_mean_SPD"] = float(np.mean(spd(truth)))
        # WRF self-spread vs the same valid time
        b = load(b_dir, valid) if b_dir is not None else None
        if b is not None:
            for f in FIELDS:
                row[f"spread_{b_label}_{f}"] = rmse(truth[f], b[f])
            row[f"spread_{b_label}_SPD"] = rmse(spd(truth), spd(b))
        c = load(c_dir, valid) if c_dir is not None else None
        if c is not None:
            for f in FIELDS:
                row[f"spread_{c_label}_{f}"] = rmse(truth[f], c[f])
            row[f"spread_{c_label}_SPD"] = rmse(spd(truth), spd(c))
        rows.append(row)
    return {
        "init_utc": init.isoformat(),
        "grid_shape": list(np.asarray(init_f["U10"]).shape),
        "init_field_stats": {
            f: {"mean": float(np.mean(np.asarray(init_f[f], dtype=np.float64))),
                "std": float(np.std(np.asarray(init_f[f], dtype=np.float64)))}
            for f in FIELDS
        },
        "init_SPD_mean": float(np.mean(init_spd)),
        "rows": rows,
    }


def at(rows: list[dict], lead: int, key: str) -> float | None:
    for r in rows:
        if r["lead_h"] == lead:
            return r.get(key)
    return None


def verdict_for_field(rows: list[dict], field: str, spread_labels: list[str],
                      probe_leads: list[int]) -> dict[str, Any]:
    """At probe leads: is WRF self-spread << persistence error? -> skillful regime."""
    out: dict[str, Any] = {"by_lead": []}
    ratios = []
    for lead in probe_leads:
        pers = at(rows, lead, f"persist_{field}")
        if pers is None:
            continue
        entry: dict[str, Any] = {"lead_h": lead, "persist_rmse": pers}
        for lbl in spread_labels:
            sp = at(rows, lead, f"spread_{lbl}_{field}")
            if sp is not None:
                entry[f"spread_{lbl}"] = sp
                entry[f"ratio_{lbl}"] = sp / (pers + 1e-12)
                ratios.append(sp / (pers + 1e-12))
        out["by_lead"].append(entry)
    if ratios:
        out["min_spread_over_persist"] = float(min(ratios))
        out["max_spread_over_persist"] = float(max(ratios))
        # If WRF agrees with itself much better than persistence error
        # (ratio well below 1) there is predictable signal a faithful model
        # should capture -> deficiency. If ratio ~>= 1, regime/metric limit.
        out["interpretation"] = (
            "MODEL_DEFICIENCY (WRF self-spread << persistence error: predictable "
            "signal exists)" if min(ratios) < 0.5
            else "REGIME_LIMIT (WRF self-spread ~ persistence error: little "
                 "discriminating signal for whole-domain gridded RMSE)"
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("proofs/wind/case3_regime_diagnostic.json"))
    args = ap.parse_args()

    case3 = regime(C3_L3a, C3_INIT, C3_L3a, C3_L2, "L2", C3_L3b, "L3b", max_lead=24)
    case2 = regime(C2_L2, C2_INIT, C2_L2, C2_L3, "L3", None, "", max_lead=24)

    # case3 self-spread probes: L2 overlaps 1..19h, L3b overlaps 1..8h.
    case3_verdict = {
        f: verdict_for_field(case3["rows"], f, ["L2", "L3b"],
                             probe_leads=[3, 6, 8, 12, 19])
        for f in FIELDS
    }
    # case2 reference: L3 overlaps to 24h.
    case2_verdict = {
        f: verdict_for_field(case2["rows"], f, ["L3"],
                             probe_leads=[6, 12, 24])
        for f in FIELDS
    }

    # Headline: does case3 V10 have predictable WRF signal at the leads we can probe?
    v10_min_ratio = case3_verdict["V10"].get("min_spread_over_persist")
    u10_min_ratio = case3_verdict["U10"].get("min_spread_over_persist")
    c2_v10_min = case2_verdict["V10"].get("min_spread_over_persist")

    headline = {
        "case3_V10_interpretation": case3_verdict["V10"].get("interpretation"),
        "case3_U10_interpretation": case3_verdict["U10"].get("interpretation"),
        "case3_V10_min_WRF_spread_over_persistence": v10_min_ratio,
        "case3_U10_min_WRF_spread_over_persistence": u10_min_ratio,
        "case2_V10_min_WRF_spread_over_persistence_REF": c2_v10_min,
        "note": (
            "ratio = (CPU-WRF self-spread RMSE) / (persistence RMSE) at the same "
            "lead. <<1 means CPU-WRF carries predictable signal a faithful model "
            "should beat persistence on (deficiency if GPU does not). ~>=1 means "
            "even independent CPU-WRF forecasts disagree about as much as "
            "persistence errs -> whole-domain gridded RMSE is regime/metric-limited."
        ),
    }

    payload = {
        "_doc": "case3 wind-skill regime diagnostic. CPU-only, reuses corpus "
                "CPU-WRF only. Answers: is case3 L3 24h V10 loss a model "
                "deficiency or a regime/metric limit, via CPU-WRF self-spread "
                "(L2-vs-L3run1 + L3run1-vs-L3run2) vs persistence error.",
        "case3": case3,
        "case2_reference": case2,
        "case3_self_spread_verdict": case3_verdict,
        "case2_self_spread_verdict": case2_verdict,
        "headline": headline,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(headline, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
