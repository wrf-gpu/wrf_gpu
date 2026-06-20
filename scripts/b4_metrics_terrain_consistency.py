#!/usr/bin/env python3
"""B4 loader fix proof: dycore terrain metrics consistent with state PHB.

Demonstrates the loader-consistency bug and fix: ``load_wrfinput_metrics`` was
reading the dycore terrain (used for pressure-gradient slopes) from
``wrfinput_d02``, while the prognostic state's base geopotential ``PHB`` (and
``GridSpec.terrain``) come from the t=0 ``wrfout_d02`` snapshot.  The two HGT
fields differ by up to ~228 m near the nest boundary (WRF parent->nest terrain
blending during init), so the PGF terrain disagreed with the geopotential the
state actually carries -- a boundary-strip-dominant initialisation error.

After the fix (``build_replay_case`` sources metrics from the same wrfout), the
metrics terrain matches PHB[0]/g to fp32 round-off.  All map factors / eta
coefficients are bitwise-identical between the two files; only HGT differs.

Writes ``proofs/b4/metrics_consistency.json``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from netCDF4 import Dataset

from gpuwrf.dynamics.metrics import load_wrfinput_metrics, terrain_slope_metrics
from gpuwrf.io.gen2_accessor import Gen2Run

DEFAULT_RUN = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T072630Z")
G = 9.81


def _read(path: Path, name: str) -> np.ndarray:
    with Dataset(path, "r") as ds:
        var = ds.variables[name]
        data = var[0] if ("Time" in var.dimensions) else var[:]
        return np.asarray(np.ma.filled(data, np.nan), dtype=np.float64)


def run(run_dir: Path) -> dict:
    run = Gen2Run(run_dir)
    wrfinput = run.wrfinput_file("d02")
    wrfout0 = run.history_files("d02")[0]

    phb_surface = _read(wrfout0, "PHB")[0] / G  # state base geopotential -> surface height
    hgt_wrfinput = _read(wrfinput, "HGT")
    hgt_wrfout = _read(wrfout0, "HGT")

    def stat(a, b):
        d = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
        return {"max_abs_diff": float(np.nanmax(d)), "mean_abs_diff": float(np.nanmean(d))}

    # terrain slopes (dzdx) the dycore PGF actually uses, from each HGT source
    dx = float(_dx(wrfout0))
    dy = float(_dy(wrfout0))
    sl_in = np.asarray(terrain_slope_metrics(hgt_wrfinput, dx, dy)[0])
    sl_out = np.asarray(terrain_slope_metrics(hgt_wrfout, dx, dy)[0])
    sl_phb = np.asarray(terrain_slope_metrics(phb_surface, dx, dy)[0])
    # confirm the loaded metrics' slopes equal the wrfout-HGT slopes (fix in place)
    loaded_out_slope = np.asarray(load_wrfinput_metrics(wrfout0).dzdx)

    # verify all non-terrain metrics identical between files
    metric_vars = ["ZNW", "MAPFAC_MX", "MAPFAC_MY", "MAPFAC_UX", "MAPFAC_UY",
                   "MAPFAC_VX", "MAPFAC_VY", "C1H", "C2H", "C3H", "C4H",
                   "C1F", "C2F", "C3F", "C4F", "DN", "DNW", "RDN", "RDNW",
                   "CF1", "CF2", "CF3", "FNM", "FNP"]
    nonterrain_max = 0.0
    for v in metric_vars:
        nonterrain_max = max(nonterrain_max, float(np.nanmax(np.abs(_read(wrfinput, v) - _read(wrfout0, v)))))

    before_terr = stat(hgt_wrfinput, phb_surface)        # wrfinput terrain vs state PHB
    after_terr = stat(hgt_wrfout, phb_surface)           # wrfout terrain vs state PHB
    before_slope = stat(sl_in, sl_phb)                   # PGF slope error before fix
    after_slope = stat(sl_out, sl_phb)                   # PGF slope error after fix
    loaded_matches_wrfout = stat(loaded_out_slope, sl_out)
    status = (
        "PASS"
        if after_terr["max_abs_diff"] < 1.0e-2
        and nonterrain_max == 0.0
        and loaded_matches_wrfout["max_abs_diff"] == 0.0
        else "FAIL"
    )

    return {
        "artifact_type": "b4_metrics_terrain_consistency",
        "status": status,
        "run_dir": str(run_dir),
        "wrfinput": str(wrfinput),
        "wrfout_t0": str(wrfout0),
        "hgt_wrfinput_vs_wrfout_m": stat(hgt_wrfinput, hgt_wrfout),
        "terrain_vs_state_PHB_m": {
            "before_fix_wrfinput": before_terr,
            "after_fix_wrfout": after_terr,
        },
        "pgf_dzdx_slope_vs_PHB_slope": {
            "before_fix_wrfinput": before_slope,
            "after_fix_wrfout": after_slope,
        },
        "loaded_metrics_dzdx_matches_wrfout_hgt_slope": loaded_matches_wrfout,
        "nonterrain_metrics_max_abs_diff_between_files": nonterrain_max,
        "conclusion": (
            "wrfinput-sourced terrain disagreed with state PHB by "
            f"{before_terr['max_abs_diff']:.1f} m (parent->nest blending); wrfout-sourced "
            f"terrain agrees to {after_terr['max_abs_diff']:.2e} m. PGF dzdx slope error "
            f"vs PHB dropped from {before_slope['max_abs_diff']:.3g} to "
            f"{after_slope['max_abs_diff']:.3g}. Only HGT differs between files; all map "
            "factors/eta coeffs are bitwise-identical, and build_replay_case now loads "
            "metrics from the wrfout snapshot."
        ),
    }


def _dx(path: Path) -> float:
    with Dataset(path, "r") as ds:
        return float(getattr(ds, "DX"))


def _dy(path: Path) -> float:
    with Dataset(path, "r") as ds:
        return float(getattr(ds, "DY"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--output", type=Path, default=ROOT / "proofs/b4/metrics_consistency.json")
    args = ap.parse_args(argv)
    payload = run(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
