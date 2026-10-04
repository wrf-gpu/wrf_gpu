"""Per-domain max-Courant report for the 3-dom Canary K2 validation (TEST INFRA).

For each domain dNN, the model timestep is the root dt / (product of
parent_grid_ratio up the chain). We read each domain's wrfout, infer its dt from
the global DT attribute WRF writes per file, and compute the worst Courant; then
we compare BESTc vs R1c per domain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from courant_diag import courant_for_file  # noqa: E402

# Reuse the surface/3-D operational bands from the single-domain comparator.
sys.path.insert(0, str(HERE))
from compare_to_r1 import BANDS_RMSE, compare as compare_fields  # noqa: E402


def _domain_dt(wrfout: Path) -> float:
    """Read the model timestep WRF stamps as the global DT attribute."""
    with Dataset(str(wrfout)) as ds:
        return float(getattr(ds, "DT"))


def _latest_per_domain(out_dir: Path, maxdom: int):
    files = {}
    for d in range(1, maxdom + 1):
        dom = f"d{d:02d}"
        cands = sorted(out_dir.glob(f"wrfout_{dom}_*"))
        if cands:
            files[dom] = cands[-1]
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1-dir", required=True)
    ap.add_argument("--best-dir", required=True)
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--nsound", type=int, required=True)
    ap.add_argument("--maxdom", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    r1 = _latest_per_domain(Path(args.r1_dir), args.maxdom)
    best = _latest_per_domain(Path(args.best_dir), args.maxdom)

    report = {
        "scale_factor": args.scale,
        "best_nsound": args.nsound,
        "maxdom": args.maxdom,
        "domains": {},
        "best_within_band_all_domains": True,
        "best_cfl_safe_all_domains": True,
        "worst_C_total_best": 0.0,
    }

    for dom in sorted(set(r1) | set(best)):
        entry = {}
        if dom in r1:
            dt_r1 = _domain_dt(r1[dom])
            cr1 = courant_for_file(r1[dom], dt_r1)
            entry["r1"] = {"dt_s": dt_r1, "worst": cr1["worst"], "file": str(r1[dom])}
        if dom in best:
            dt_b = _domain_dt(best[dom])
            cb = courant_for_file(best[dom], dt_b)
            entry["best"] = {"dt_s": dt_b, "worst": cb["worst"], "file": str(best[dom])}
            report["worst_C_total_best"] = max(
                report["worst_C_total_best"], cb["worst"]["C_total"]
            )
        # field comparison best vs r1 (same valid time, same grid per domain)
        if dom in r1 and dom in best:
            try:
                cmp = compare_fields(str(best[dom]), str(r1[dom]))
                entry["compare"] = {
                    "within_operational_band": cmp["within_operational_band"],
                    "worst_rmse_over_band": cmp["worst_rmse_over_band"],
                    "fields": cmp["fields"],
                }
                if not cmp["within_operational_band"]:
                    report["best_within_band_all_domains"] = False
            except Exception as exc:  # noqa: BLE001
                entry["compare_error"] = f"{type(exc).__name__}: {exc}"
        report["domains"][dom] = entry

    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
