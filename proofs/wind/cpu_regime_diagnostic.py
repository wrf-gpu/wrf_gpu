"""CPU-ONLY wind-skill regime diagnostic (NO GPU, NO forecast, NO JAX).

Answers the #1 question for the V10/U10 wind-skill gap: is the LIMIT the
meteorological REGIME (the meridional wind is barely predictable here, so even a
perfect model can't beat persistence) or is it a MODEL deficiency (CPU-WRF is
clearly skillful while the GPU port is not)?

It uses ONLY the existing corpus CPU-WRF wrfout history (case2 0509 has hourly
L2 72h + L3 24h history). No GPU forecast is launched. Three independent lines of
evidence:

  (1) PERSISTENCE-vs-TRUTH growth, hour by hour, for U10/V10/T2/wind-speed.
      How fast does each field decorrelate from its t=0 value?

  (2) CPU-WRF L2-vs-L3 cross-RMSE (the "WRF spread"). L2 and L3 are TWO
      INDEPENDENT, legitimate CPU-WRF forecasts of the SAME case. Their mutual
      disagreement at d02 is a lower bound on the IRREDUCIBLE forecast
      uncertainty WRF carries for this field/regime. If CPU-WRF disagrees with
      CPU-WRF on V10 by as much as we miss persistence, the metric (gridded
      whole-domain RMSE on a near-zero-mean field) is the limit, not the model.

  (3) FIELD STATISTICS: mean, std, signal range of U10/V10 and their temporal
      evolution.

CPU-only: numpy + netCDF reads. Pin with JAX_PLATFORMS=cpu just in case.

USAGE
  JAX_PLATFORMS=cpu PYTHONPATH=src OMP_NUM_THREADS=2 taskset -c 0-3 \
    python proofs/wind/cpu_regime_diagnostic.py --out proofs/wind/cpu_regime_diagnostic.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gpuwrf.io.gen2_wrfout_loader import read_wrfout_file

L2_DIR = Path("/mnt/data/canairy_meteo/runs/wrf_l2/20260509_18z_l2_72h_20260511T190519Z")
L3_DIR = Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260509_18z_l3_24h_20260511T190519Z")
INIT = datetime(2026, 5, 9, 18, 0, 0, tzinfo=timezone.utc)
FIELDS = ("U10", "V10", "T2")
STATIC = ("XLAND", "HGT", "LANDMASK")


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("proofs/wind/cpu_regime_diagnostic.json"))
    args = ap.parse_args()

    init = read_wrfout_file(wrfout(L2_DIR, INIT), fields=FIELDS + STATIC)["fields"]
    xland = np.asarray(init["XLAND"], dtype=np.float64)   # WRF: 1=land, 2=water
    land = xland < 1.5
    water = ~land
    coast = np.zeros_like(land)
    for ax in (0, 1):
        for sh in (1, -1):
            nbr = np.roll(land, sh, axis=ax)
            coast |= (land != nbr)
    coast[0, :] = coast[-1, :] = coast[:, 0] = coast[:, -1] = False

    masks = {"all": np.ones_like(land), "land": land, "water": water, "coast": coast}

    def masked_rmse(a, b, m):
        a = np.asarray(a, dtype=np.float64)[m]
        b = np.asarray(b, dtype=np.float64)[m]
        if a.size == 0:
            return None
        return float(np.sqrt(np.mean((a - b) ** 2)))

    persistence_growth: list[dict[str, Any]] = []
    wrf_spread: list[dict[str, Any]] = []
    for lead_h in range(0, 73):
        valid = INIT + timedelta(hours=lead_h)
        l2 = load(L2_DIR, valid)
        l3 = load(L3_DIR, valid)
        if l2 is not None:
            row: dict[str, Any] = {"lead_h": lead_h}
            for f in FIELDS:
                row[f"persist_{f}"] = rmse(init[f], l2[f])
                row[f"std_{f}"] = float(np.std(np.asarray(l2[f], dtype=np.float64)))
                row[f"mean_{f}"] = float(np.mean(np.asarray(l2[f], dtype=np.float64)))
            spd0 = np.hypot(np.asarray(init["U10"]), np.asarray(init["V10"]))
            spdL = np.hypot(np.asarray(l2["U10"]), np.asarray(l2["V10"]))
            row["persist_SPD"] = rmse(spd0, spdL)
            row["mean_SPD"] = float(np.mean(spdL))
            persistence_growth.append(row)
        if l2 is not None and l3 is not None:
            row = {"lead_h": lead_h}
            for f in FIELDS:
                row[f"L2vL3_{f}"] = rmse(l2[f], l3[f])
                for mname, m in masks.items():
                    if mname == "all":
                        continue
                    mr = masked_rmse(l2[f], l3[f], m)
                    if mr is not None:
                        row[f"L2vL3_{f}_{mname}"] = mr
            wrf_spread.append(row)

    def field_motion(lead_h: int) -> dict[str, Any]:
        valid = INIT + timedelta(hours=lead_h)
        l2 = load(L2_DIR, valid)
        if l2 is None:
            return {}
        out: dict[str, Any] = {"lead_h": lead_h}
        for f in FIELDS:
            truth = np.asarray(l2[f], dtype=np.float64)
            i0 = np.asarray(init[f], dtype=np.float64)
            out[f] = {
                "spatial_std_truth": float(np.std(truth)),
                "spatial_mean_truth": float(np.mean(truth)),
                "temporal_change_rmse": rmse(i0, truth),
                "change_over_std": rmse(i0, truth) / (float(np.std(truth)) + 1e-9),
            }
        return out

    motion = [m for m in (field_motion(h) for h in (24, 48, 72)) if m]

    def at(lead_h: int, key: str, src: list[dict]) -> float | None:
        for r in src:
            if r["lead_h"] == lead_h:
                return r.get(key)
        return None

    summary: dict[str, Any] = {}
    for f in FIELDS + ("SPD",):
        if f == "SPD":
            persist24 = at(24, "persist_SPD", persistence_growth)
            persist48 = at(48, "persist_SPD", persistence_growth)
            spread24 = None
        else:
            persist24 = at(24, f"persist_{f}", persistence_growth)
            persist48 = at(48, f"persist_{f}", persistence_growth)
            spread24 = at(24, f"L2vL3_{f}", wrf_spread)
        summary[f] = {
            "persistence_rmse_24h": persist24,
            "persistence_rmse_48h": persist48,
            "wrf_L2vL3_spread_24h": spread24,
            "wrf_spread_ge_persistence_24h": (
                spread24 is not None and persist24 is not None and spread24 >= persist24
            ),
        }

    payload = {
        "_doc": "CPU-only wind-skill regime diagnostic. Uses ONLY existing corpus "
                "CPU-WRF wrfout history (case2 0509 L2 72h + L3 24h). No GPU, no "
                "forecast. (1) persistence RMSE growth, (2) CPU-WRF L2-vs-L3 "
                "cross-RMSE = WRF self-spread = irreducible-uncertainty lower "
                "bound, (3) field temporal-motion vs spatial-std.",
        "init_utc": INIT.isoformat(),
        "grid": {"land_cells": int(land.sum()), "water_cells": int(water.sum()),
                 "coast_cells": int(coast.sum()), "total": int(land.size)},
        "persistence_growth_hourly": persistence_growth,
        "wrf_L2_vs_L3_spread_hourly": wrf_spread,
        "field_motion_at_leads": motion,
        "summary_by_field": summary,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")

    print("\n=== PERSISTENCE RMSE growth (L2 truth, hold-t0) ===")
    print(f"{'lead':>4} {'pV10':>7} {'pU10':>7} {'pT2':>7} {'pSPD':>7} "
          f"{'<V10>':>7} {'sdV10':>7} {'<U10>':>7} {'sdU10':>7}")
    for r in persistence_growth:
        if r["lead_h"] % 6 == 0:
            print(f"{r['lead_h']:>4} {r['persist_V10']:>7.3f} {r['persist_U10']:>7.3f} "
                  f"{r['persist_T2']:>7.3f} {r['persist_SPD']:>7.3f} "
                  f"{r['mean_V10']:>7.3f} {r['std_V10']:>7.3f} "
                  f"{r['mean_U10']:>7.3f} {r['std_U10']:>7.3f}")

    print("\n=== CPU-WRF L2-vs-L3 spread (two indep WRF runs, same case) ===")
    print(f"{'lead':>4} {'V10':>7} {'U10':>7} {'T2':>7} | {'V10_land':>9} {'V10_water':>9} {'V10_coast':>9}")
    for r in wrf_spread:
        if r["lead_h"] % 4 == 0:
            print(f"{r['lead_h']:>4} {r.get('L2vL3_V10',0):>7.3f} {r.get('L2vL3_U10',0):>7.3f} "
                  f"{r.get('L2vL3_T2',0):>7.3f} | {r.get('L2vL3_V10_land',0):>9.3f} "
                  f"{r.get('L2vL3_V10_water',0):>9.3f} {r.get('L2vL3_V10_coast',0):>9.3f}")

    print("\n=== SUMMARY ===")
    for f, s in summary.items():
        print(f"{f}: persist24={s['persistence_rmse_24h']}, "
              f"WRF-L2vL3-spread24={s['wrf_L2vL3_spread_24h']}, "
              f"WRF_spread>=persist: {s['wrf_spread_ge_persistence_24h']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
