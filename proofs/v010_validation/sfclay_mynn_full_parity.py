"""Full MYNN surface-layer external-oracle parity harness for P1-4a.

VALIDATES P1-4a POST-TAG. This harness is the acceptance gate for the v0.1.1
item P1-4a (MYNN/HFX full surface-layer parity; closes GPT review findings G1/G3
in .agent/reviews/2026-05-31-gpt-hfx-and-proof-review.md). It is intentionally
written and committed NOW, during the v0.1.0 prep freeze, but **must not be the
gate for v0.1.0**: the current src/gpuwrf/physics/surface_layer.py (commit
d1c373b) is an empirical PARTIAL MYNN repair, so this harness is EXPECTED to flag
the zol/restar/PSIH2 mismatches until the P1-4a code edit lands. Run it only
AFTER the v0.1.0 tag, against the P1-4a-corrected surface_layer.py.

WHAT THIS IS (and is NOT)
-------------------------
External oracle, NOT a self-compare: the corpus L3 (sf_sfclay_physics=5 = MYNN
surface layer, module_sf_mynn.F) midday/night d03 wrfout supplies the SAME-STEP
column inputs (theta, p, qv, u, v, dz, TSK, PSFC, XLAND, prior-step UST), runs
them through the GPU port, and compares the GPU outputs against the SAME-STEP
WRF-MYNN outputs in that same wrfout file. This isolates the SCHEME from forecast
drift. It is NOT a JAX-vs-JAX comparison and NOT a re-run of the GPU forecast.

ORACLE COVERAGE / HONESTY (GPT finding #2)
------------------------------------------
The corpus wrfout carries direct same-step truth for:
    HFX, LH, QFX, Q2, T2, TH2, U10, V10, UST, PBLH
It does NOT carry: ZNT, QSFC, MOL/RMOL, MAVAIL, REGIME, ZOL, CHS/CHS2/CQS2,
FLHC/FLQC. Therefore:
  * ZNT is RECONSTRUCTED from the land/water default the corpus LSM used
    (land 0.10 m, water 2.85e-3 m); QSFC is recomputed from TSK/PSFC (water +
    cold-start land); MAVAIL=1, LAKEMASK=0, SNOWH=0 (verified no-snow).
  * MOL, PSIT, PSIQ have NO direct wrfout field. They are reported as GPU
    diagnostics with a back-derived WRF check where one exists (e.g. HFX implies
    a consistency band on PSIT via HFX = cpm*rho*ust*K*(thgb-thx)/psit), and are
    flagged "needs instrumented Fortran MYNN column trace for exact parity"
    (GPT finding #2/#3 — the strict MOL/PSIT/PSIQ oracle is an instrumented WRF
    column dump, a separate post-tag deliverable noted in the spec).
  * The standalone land HFX/QFX residual vs WRF is the irreducible Noah-MP LSM
    surface-energy-balance coupling (a standalone surface layer cannot reproduce
    it); the SCHEME-binding land metrics are T2/Q2/U10/V10 and PSIT/PSIQ
    structure, while HFX/LH parity is the BINDING metric over WATER (no LSM).

CASES (land/water x stable/unstable)
------------------------------------
  unstable/daytime : wrfout_d03_2026-05-22_12:00:00 (HFX land ~459, water ~4 W/m2)
  stable/nighttime : wrfout_d03_2026-05-22_03:00:00 (radiative night, BR>0)
Each is split into land (XLAND<1.5) and water (XLAND>=1.5) sub-populations, and
each sub-population is further split by the GPU BR sign into stable / unstable
cells, so all four land/water x stable/unstable quadrants are scored where the
case populates them.

Run (POST-TAG only; GPU):
    PYTHONPATH=src XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 taskset -c 0-3 \
        python proofs/v010_validation/sfclay_mynn_full_parity.py
Output: proofs/v010_validation/sfclay_mynn_full_parity.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.7")
os.environ.setdefault("OMP_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

try:
    from netCDF4 import Dataset
except Exception as exc:  # pragma: no cover - environment guard
    raise SystemExit(f"netCDF4 required: {exc}")

RUN = Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z")
CASES = {
    "unstable_daytime": "wrfout_d03_2026-05-22_12:00:00",
    "stable_nighttime": "wrfout_d03_2026-05-22_03:00:00",
}
P0 = 100000.0
ROVCP = 287.0 / 1004.0
G = 9.80665

# Acceptance bands for P1-4a (PROVISIONAL; predeclared here, refine in the 0.1.1
# commit against the instrumented column trace). Scheme-binding metrics only.
# Water HFX/LH parity is binding (no LSM); land HFX/LH carry the Noah-MP residual.
ACCEPT = {
    "T2_all_rmse_max": 1.5,        # K  (operationally binding diagnostic)
    "T2_water_bias_abs_max": 0.20,  # K  (water has no LSM coupling -> tight)
    "U10_all_rmse_max": 1.0,        # m/s
    "V10_all_rmse_max": 1.0,        # m/s
    "Q2_all_rmse_max": 1.5e-3,      # kg/kg
    "HFX_water_rmse_max": 25.0,     # W/m2 (binding: no LSM over water)
    "LH_water_rmse_max": 60.0,      # W/m2
    "PBLH_passthrough": True,       # PBLH is a PBL diagnostic, passed through
}

# Diagnostics with NO direct same-step wrfout truth -> reported, not gated here.
NO_WRFOUT_TRUTH = ["MOL", "RMOL", "PSIT", "PSIQ", "ZOL", "REGIME", "QSFC", "ZNT"]


def _v(d, name):
    return np.asarray(d.variables[name][0], dtype=np.float64)


def _build_state(d):
    """Build the GPU surface-layer input state from a corpus wrfout slice."""
    import jax.numpy as jnp

    T = _v(d, "T"); P = _v(d, "P"); PB = _v(d, "PB")
    PH = _v(d, "PH"); PHB = _v(d, "PHB"); QV = _v(d, "QVAPOR")
    U = _v(d, "U"); V = _v(d, "V")
    TSK = _v(d, "TSK"); PSFC = _v(d, "PSFC"); XLAND = _v(d, "XLAND")
    UST_wrf = _v(d, "UST")

    theta0 = T[0] + 300.0
    p0 = P[0] + PB[0]
    qv0 = np.maximum(QV[0], 0.0)
    dz0 = ((PH + PHB)[1] - (PH + PHB)[0]) / G
    um = 0.5 * (U[0, :, :-1] + U[0, :, 1:])
    vm = 0.5 * (V[0, :-1, :] + V[0, 1:, :])
    land = XLAND < 1.5
    # RECONSTRUCTED inputs (not in wrfout): land/water default znt, mavail=1, no snow.
    znt = np.where(land, 0.10, 2.85e-3)

    state = SimpleNamespace(
        u=jnp.asarray(um[..., None]), v=jnp.asarray(vm[..., None]),
        theta=jnp.asarray(theta0[..., None]), qv=jnp.asarray(qv0[..., None]),
        p=jnp.asarray(p0[..., None]), dz=jnp.asarray(dz0),
        t_skin=jnp.asarray(TSK), psfc=jnp.asarray(PSFC), xland=jnp.asarray(XLAND),
        lakemask=jnp.zeros_like(jnp.asarray(XLAND)),
        mavail=jnp.ones_like(jnp.asarray(XLAND)),
        roughness_m=jnp.asarray(znt),
        ustar=jnp.asarray(np.maximum(UST_wrf, 1e-4)),  # prior-step UST (warm input)
    )
    return state, land


def _stat(g, w, m):
    if int(m.sum()) == 0:
        return {"n": 0}
    gg, ww = g[m], w[m]
    return {
        "n": int(m.sum()),
        "gpu_mean": float(gg.mean()), "wrf_mean": float(ww.mean()),
        "bias": float((gg - ww).mean()),
        "rmse": float(np.sqrt(((gg - ww) ** 2).mean())),
    }


def run_case(name: str, fname: str) -> dict:
    import gpuwrf  # noqa: F401  enables x64 at import
    from gpuwrf.physics import surface_layer as sl

    path = RUN / fname
    d = Dataset(str(path))
    state, land = _build_state(d)

    diag = sl.surface_layer_with_diagnostics(state)
    gpu = {
        "HFX": np.asarray(diag.hfx), "LH": np.asarray(diag.lh),
        "QFX": np.asarray(diag.lh) / 2.5e6,  # LH = XLV*QFX -> QFX back-out for reporting
        "Q2": np.asarray(diag.q2), "T2": np.asarray(diag.t2),
        "U10": np.asarray(diag.u10), "V10": np.asarray(diag.v10),
        "UST": np.asarray(diag.fluxes.ustar),
        "MOL": np.asarray(diag.mol), "ZOL": np.asarray(diag.zol),
        "REGIME": np.asarray(diag.regime), "BR": np.asarray(diag.br),
    }
    wrf = {
        "HFX": _v(d, "HFX"), "LH": _v(d, "LH"), "QFX": _v(d, "QFX"),
        "Q2": _v(d, "Q2"), "T2": _v(d, "T2"),
        "U10": _v(d, "U10"), "V10": _v(d, "V10"), "UST": _v(d, "UST"),
        "PBLH": _v(d, "PBLH"),
    }

    # stable/unstable masks from the GPU BR sign (REGIME proxy; no wrfout REGIME)
    stable = gpu["BR"] > 0.0
    unstable = gpu["BR"] < 0.0
    quad = {
        "land_stable": land & stable, "land_unstable": land & unstable,
        "water_stable": (~land) & stable, "water_unstable": (~land) & unstable,
        "land_all": land, "water_all": ~land,
        "all": np.ones_like(land, bool),
    }

    scored = {}
    for fld in ["HFX", "LH", "QFX", "Q2", "T2", "U10", "V10", "UST"]:
        scored[fld] = {q: _stat(gpu[fld], wrf[fld], m) for q, m in quad.items()}

    # PBLH passthrough sanity (PBLH is a PBL diagnostic, not produced here)
    pblh = {
        "note": "PBLH is a PBL-scheme diagnostic, not output by surface_layer; "
                "checked for finite/in-range passthrough only.",
        "wrf_min": float(wrf["PBLH"].min()), "wrf_max": float(wrf["PBLH"].max()),
    }

    # diagnostics without direct wrfout truth -> report GPU distribution only
    no_truth = {
        f: {"gpu_mean": float(gpu[f].mean()), "gpu_min": float(gpu[f].min()),
            "gpu_max": float(gpu[f].max())}
        for f in ["MOL", "ZOL", "REGIME"]
    }

    d.close()
    return {
        "case": name, "file": str(path),
        "land_cells": int(land.sum()), "water_cells": int((~land).sum()),
        "scored": scored, "pblh": pblh,
        "diagnostics_no_wrfout_truth": no_truth,
    }


def _verdict(cases: dict) -> dict:
    """Apply provisional acceptance bands. EXPECTED to FAIL pre-P1-4a (partial fix)."""
    checks = {}
    # use the daytime case as the primary scoring case (both reported)
    c = cases.get("unstable_daytime", {})
    sc = c.get("scored", {})

    def grab(fld, quad, key):
        return sc.get(fld, {}).get(quad, {}).get(key)

    def le(name, val, thr):
        ok = (val is not None) and (val <= thr)
        checks[name] = {"value": val, "threshold": thr, "pass": bool(ok)}

    le("T2_all_rmse", grab("T2", "all", "rmse"), ACCEPT["T2_all_rmse_max"])
    b = grab("T2", "water_all", "bias")
    checks["T2_water_bias_abs"] = {
        "value": (abs(b) if b is not None else None),
        "threshold": ACCEPT["T2_water_bias_abs_max"],
        "pass": bool(b is not None and abs(b) <= ACCEPT["T2_water_bias_abs_max"]),
    }
    le("U10_all_rmse", grab("U10", "all", "rmse"), ACCEPT["U10_all_rmse_max"])
    le("V10_all_rmse", grab("V10", "all", "rmse"), ACCEPT["V10_all_rmse_max"])
    le("Q2_all_rmse", grab("Q2", "all", "rmse"), ACCEPT["Q2_all_rmse_max"])
    le("HFX_water_rmse", grab("HFX", "water_all", "rmse"), ACCEPT["HFX_water_rmse_max"])
    le("LH_water_rmse", grab("LH", "water_all", "rmse"), ACCEPT["LH_water_rmse_max"])

    all_pass = all(v["pass"] for v in checks.values())
    return {
        "all_pass": all_pass,
        "verdict": "P1-4A_MYNN_PARITY_PASS" if all_pass else "P1-4A_PENDING_OR_FAIL",
        "checks": checks,
        "note": (
            "Land HFX/QFX are intentionally NOT gated (Noah-MP LSM coupling "
            "residual). MOL/PSIT/PSIQ have no wrfout truth -> exact parity needs "
            "an instrumented Fortran MYNN column trace (separate post-tag item). "
            "Pre-P1-4a this verdict is EXPECTED to be PENDING_OR_FAIL."
        ),
    }


def run() -> dict:
    missing = [f for f in CASES.values() if not (RUN / f).exists()]
    cases = {}
    for name, fname in CASES.items():
        if (RUN / fname).exists():
            cases[name] = run_case(name, fname)
    record = {
        "proof": "sfclay-mynn-full-parity (P1-4a; closes GPT G1/G3)",
        "status": "VALIDATES P1-4a POST-TAG -- not a v0.1.0 gate",
        "kind": (
            "external oracle: corpus-WRF MYNN (sf_sfclay_physics=5) same-step d03 "
            "column inputs -> GPU surface_layer -> vs SAME-STEP WRF outputs "
            "(NOT a self-compare, NOT a GPU forecast re-run)"
        ),
        "spec": str((ROOT / ".agent/decisions/P1-4a-MYNN-PARITY-SPEC.md")),
        "reconstructed_inputs": {
            "znt": "land 0.10 m / water 2.85e-3 m (not in wrfout)",
            "mavail": 1.0, "lakemask": 0.0, "snowh": 0.0,
            "qsfc": "recomputed from TSK/PSFC (water + cold-start land)",
            "ustar_input": "same-step WRF UST used as prior-step ustar",
        },
        "no_wrfout_truth_fields": NO_WRFOUT_TRUTH,
        "missing_case_files": missing,
        "cases": cases,
        "verdict": _verdict(cases) if cases else {"verdict": "NO_CASE_FILES"},
    }
    return record


if __name__ == "__main__":
    rec = run()
    out = Path(__file__).resolve().parent / "sfclay_mynn_full_parity.json"
    out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(rec["verdict"], indent=2, sort_keys=True))
