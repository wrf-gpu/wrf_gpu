"""Find WRF output files with wet graupel columns for Thompson cap evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def _scan_file(path: Path, threshold: float) -> dict | None:
    try:
        with Dataset(path, "r") as ds:
            if "QGRAUP" not in ds.variables:
                return None
            qg = np.asarray(ds.variables["QGRAUP"][0])
            wet3d = qg > threshold
            wet_cols = np.any(wet3d, axis=0)
            qr_max = float(np.nanmax(ds.variables["QRAIN"][0])) if "QRAIN" in ds.variables else None
            qc_max = float(np.nanmax(ds.variables["QCLOUD"][0])) if "QCLOUD" in ds.variables else None
            rain_acc_max = float(np.nanmax(ds.variables["RAINNC"][0])) if "RAINNC" in ds.variables else None
            return {
                "path": str(path),
                "qgraup_max": float(np.nanmax(qg)),
                "qgraup_wet_cells": int(np.count_nonzero(wet3d)),
                "qgraup_wet_columns": int(np.count_nonzero(wet_cols)),
                "qgraup_wet_fraction": float(np.count_nonzero(wet_cols) / max(wet_cols.size, 1)),
                "qrain_max": qr_max,
                "qcloud_max": qc_max,
                "rainnc_max": rain_acc_max,
            }
    except Exception as exc:
        return {"path": str(path), "error": repr(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="*", type=Path, default=[Path("/mnt/data/canairy_meteo/runs")])
    parser.add_argument("--threshold", type=float, default=1.0e-12)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--max-files", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=Path("proofs/v0100/graupel_wet_candidates.json"))
    args = parser.parse_args()

    paths: list[Path] = []
    for root in args.roots:
        if root.is_file():
            paths.append(root)
        else:
            paths.extend(sorted(root.rglob("wrfout_d0*_*")))
    paths = paths[: int(args.max_files)]

    records = []
    for idx, path in enumerate(paths, 1):
        rec = _scan_file(path, float(args.threshold))
        if rec is not None:
            records.append(rec)
        if idx % 250 == 0:
            print(f"scanned {idx}/{len(paths)}", flush=True)

    valid = [r for r in records if "error" not in r]
    ranked = sorted(
        valid,
        key=lambda r: (int(r["qgraup_wet_columns"]), float(r["qgraup_max"]), float(r.get("qrain_max") or 0.0)),
        reverse=True,
    )
    payload = {
        "schema": "V0100GraupelWetCandidateSearch",
        "schema_version": 1,
        "threshold": float(args.threshold),
        "scanned_files": len(paths),
        "readable_files": len(valid),
        "error_count": len(records) - len(valid),
        "top": ranked[: int(args.limit)],
        "errors": [r for r in records if "error" in r][:20],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"scanned_files": len(paths), "readable_files": len(valid), "top": ranked[:5]}, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
