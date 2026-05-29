#!/usr/bin/env python3
"""B4 static-field parity: gpuwrf-loaded prescribed fields vs WRF ground truth.

Compares the static / prescribed fields that the B4 loaders place into the
operational ``State`` against the WRF ``wrfinput_d02`` (and, where the field is
a model-diagnostic, the t=0 ``wrfout_d02``) for the pinned Canary L3 run.

Ground truth: ``/mnt/data/canairy_meteo/runs/wrf_l3/<run>/wrfinput_d02`` and
``wrfout_d02_<t0>``.  No self-compares: every reference array is read straight
from the WRF NetCDF, independently of the gpuwrf loader path.

Writes ``proofs/b4/static_field_parity.json`` and prints a table.
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

from gpuwrf.integration.d02_replay import build_replay_case

DEFAULT_RUN = Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T072630Z")
EXPLAINABLE_TOL = 1.0e-5  # relative-to-range tolerance for "explainably close"


def _read(ds: Dataset, name: str) -> np.ndarray | None:
    if name not in ds.variables:
        return None
    var = ds.variables[name]
    data = var[0] if ("Time" in var.dimensions) else var[:]
    return np.asarray(np.ma.filled(data, np.nan), dtype=np.float64)


def _compare(name: str, model: np.ndarray, ref: np.ndarray, note: str = "") -> dict:
    model = np.asarray(model, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    shape_ok = model.shape == ref.shape
    if not shape_ok:
        return {
            "field": name, "verdict": "SHAPE_MISMATCH",
            "model_shape": list(model.shape), "ref_shape": list(ref.shape),
            "note": note,
        }
    diff = np.abs(model - ref)
    rng = float(np.nanmax(ref) - np.nanmin(ref))
    max_abs = float(np.nanmax(diff))
    bitwise = bool(np.array_equal(model, ref))
    rel = max_abs / rng if rng > 0 else max_abs
    if bitwise:
        verdict = "BITWISE"
    elif rel <= EXPLAINABLE_TOL:
        verdict = "EXPLAINABLE"
    else:
        verdict = "MISMATCH"
    return {
        "field": name,
        "verdict": verdict,
        "max_abs_diff": max_abs,
        "ref_range": rng,
        "rel_to_range": rel,
        "model_min": float(np.nanmin(model)),
        "model_max": float(np.nanmax(model)),
        "ref_min": float(np.nanmin(ref)),
        "ref_max": float(np.nanmax(ref)),
        "shape": list(model.shape),
        "note": note,
    }


def run(run_dir: Path) -> dict:
    case = build_replay_case(str(run_dir), domain="d02")
    st = case.state
    g = lambda f: np.asarray(getattr(st, f))

    wrfinput = run_dir / "wrfinput_d02"
    history = sorted(run_dir.glob("wrfout_d02_*"))
    wrfout0 = history[0]

    # Terrain ground truth is the t=0 wrfout HGT: the prognostic state's base
    # geopotential PHB[0]/g matches *that* file's HGT (the model integrates from
    # the post-init, terrain-blended state), and the B4 loader now sources both
    # GridSpec.terrain and the dycore metrics from this same snapshot.
    with Dataset(wrfout0, "r") as do0:
        hgt_wrfout = _read(do0, "HGT")

    rows: list[dict] = []
    with Dataset(wrfinput, "r") as di:
        # Direct prescribed fields straight from wrfinput.
        xland = _read(di, "XLAND")
        landmask = _read(di, "LANDMASK")
        lu_index = _read(di, "LU_INDEX")
        ivgtyp = _read(di, "IVGTYP")
        isltyp = _read(di, "ISLTYP")
        hgt = _read(di, "HGT")
        vegfra = _read(di, "VEGFRA")
        sst = _read(di, "SST")
        tsk = _read(di, "TSK")
        smois = _read(di, "SMOIS")  # (soil, y, x)
        tslb = _read(di, "TSLB")
        sh2o = _read(di, "SH2O")
        lakemask = _read(di, "LAKEMASK")

        rows.append(_compare("XLAND", g("xland"), xland, "wrfinput_d02 -> State.xland"))
        rows.append(_informational("LANDMASK", landmask, "consumed via xland/lakemask; not a separate State leaf"))
        rows.append(_compare("LU_INDEX", g("lu_index"), lu_index, "wrfinput_d02 -> State.lu_index (int32)"))
        rows.append(_compare("LAKEMASK", g("lakemask"), lakemask, "wrfinput_d02 -> State.lakemask"))
        rows.append(_compare("TSK->t_skin", g("t_skin"), np.clip(tsk, 180.0, 340.0),
                             "State.t_skin = clip(TSK,180,340); raw TSK range printed in ref"))
        rows.append(_compare("SMOIS[0]->soil_moisture", g("soil_moisture"), smois[0],
                             "State.soil_moisture = top SMOIS layer"))
        # IVGTYP / ISLTYP / HGT / VEGFRA / SST are prescribed inputs consumed by
        # the loader but not all kept as distinct State leaves -> informational.
        rows.append(_informational("IVGTYP", ivgtyp, "WRF land-use category (input to lu_index/Noah); not a State leaf"))
        rows.append(_informational("ISLTYP", isltyp, "WRF soil category (Noah input); not a State leaf"))
        rows.append(_compare("HGT->GridSpec.terrain_height", np.asarray(case.grid.terrain_height), hgt_wrfout,
                             "wrfout t0 HGT -> GridSpec.terrain_height (consistent with state PHB)"))
        # Document the WRF terrain-blending delta as explainable, not a bug.
        terr_delta = float(np.nanmax(np.abs(hgt - hgt_wrfout)))
        rows.append({
            "field": "HGT wrfinput-vs-wrfout delta", "verdict": "INFORMATIONAL",
            "max_abs_diff": terr_delta, "shape": list(hgt.shape),
            "note": "WRF parent->nest terrain blending at init; loader now sources terrain+metrics from wrfout to match PHB",
        })
        rows.append(_informational("VEGFRA", vegfra, "vegetation fraction (roughness/RRTMG input); not a State leaf"))
        rows.append(_informational("SST", sst, "sea-surface temperature (water t_skin source in hourly land state); not init State leaf"))

    # ALBEDO / EMISS / ZNT are model-diagnostic (RRTMG/surface) -> compare the
    # WRF wrfout t=0 presence and note they are owned by B3, not B4 init.
    with Dataset(wrfout0, "r") as do:
        albedo = _read(do, "ALBEDO")
        emiss = _read(do, "EMISS")
        znt = _read(do, "ZNT")
    rows.append(_informational("ALBEDO", albedo, "wrfout diagnostic; surface radiation prop owned by B3 (_surface_radiation_properties)"))
    rows.append(_informational("EMISS", emiss, "wrfout diagnostic; surface radiation prop owned by B3"))
    rows.append(_informational("ZNT", znt if znt is not None else np.array([np.nan]),
                               "ABSENT in this physics config; State.roughness_m derived from VEGFRA/land-water surrogate (documented)"))

    verdicts = [r["verdict"] for r in rows if r["verdict"] in ("BITWISE", "EXPLAINABLE", "MISMATCH", "SHAPE_MISMATCH")]
    compared = [r for r in rows if "max_abs_diff" in r]
    n_mismatch = sum(1 for r in compared if r["verdict"] in ("MISMATCH", "SHAPE_MISMATCH"))
    status = "PASS" if n_mismatch == 0 else "FAIL"

    return {
        "artifact_type": "b4_static_field_parity",
        "status": status,
        "run_dir": str(run_dir),
        "domain": "d02",
        "wrfinput": str(wrfinput),
        "wrfout_t0": str(wrfout0),
        "explainable_rel_tol": EXPLAINABLE_TOL,
        "grid": {"nz": int(case.grid.nz), "ny": int(case.grid.ny), "nx": int(case.grid.nx)},
        "n_compared": len(compared),
        "n_mismatch": n_mismatch,
        "rows": rows,
        "note": "State leaves compared bitwise/explainable vs WRF; informational rows are WRF inputs/diagnostics not stored as distinct State leaves.",
    }


def _informational(name: str, ref: np.ndarray, note: str) -> dict:
    ref = np.asarray(ref, dtype=np.float64)
    return {
        "field": name,
        "verdict": "INFORMATIONAL",
        "ref_min": float(np.nanmin(ref)),
        "ref_max": float(np.nanmax(ref)),
        "shape": list(ref.shape),
        "note": note,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--output", type=Path, default=ROOT / "proofs/b4/static_field_parity.json")
    args = ap.parse_args(argv)
    payload = run(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"status={payload['status']} compared={payload['n_compared']} mismatch={payload['n_mismatch']}")
    print(f"{'field':28s} {'verdict':14s} {'max_abs':>12s} {'rel_range':>12s}  note")
    for r in payload["rows"]:
        mad = r.get("max_abs_diff")
        rel = r.get("rel_to_range")
        mad_s = f"{mad:.4g}" if isinstance(mad, float) else "-"
        rel_s = f"{rel:.3g}" if isinstance(rel, float) else "-"
        print(f"{r['field']:28s} {r['verdict']:14s} {mad_s:>12s} {rel_s:>12s}  {r['note'][:60]}")
    print(f"wrote {args.output}")
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
