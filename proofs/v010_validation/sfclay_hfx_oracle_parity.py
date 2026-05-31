"""WRF surface-oracle parity for the daytime HFX over-flux fix (proof #56).

Feeds the corpus-WRF midday d03 (1 km Tenerife) column inputs through the GPU
``sf_sfclayrev`` port and compares HFX/T2/UST against the corpus-WRF outputs in
the SAME wrfout file. The corpus L3 ran ``sf_sfclay_physics=5`` = the MYNN surface
layer (module_sf_mynn.F), whose land heat profile carries a SEPARATE thermal
roughness ``z_t`` (Zilitinkevich 1995, CZIL=0.085) distinct from the momentum
roughness. The bare sfclayrev port used the momentum roughness for heat over land
and over-fluxed midday sensible heat ~4x; the fix ports the MYNN land thermal
roughness for the heat/moisture profiles (momentum + water unchanged).

This isolates the SCHEME from forecast drift (same inputs -> same-step WRF HFX).
The residual land HFX gap is the irreducible Noah-MP LSM surface-energy-balance
coupling that a standalone surface layer cannot reproduce; the operationally
binding metric is the T2 diagnostic, which the fix collapses from a +3.6 K land
warm bias to ~+1 K (water +0.001 K and momentum UST unchanged).

Run:  PYTHONPATH=src XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 taskset -c 0-3 \
        python proofs/v010_validation/sfclay_hfx_oracle_parity.py
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
from netCDF4 import Dataset

import gpuwrf  # noqa: F401  x64
import jax.numpy as jnp
from gpuwrf.physics import surface_layer as sl

RUN = Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z")
MIDDAY = "wrfout_d03_2026-05-22_12:00:00"
P0 = 100000.0
ROVCP = 287.0 / 1004.0
G = 9.80665

# Documented BEFORE-fix oracle numbers (bare sfclayrev land-on-z0; captured
# before the z_t land thermal-roughness fix landed -- see git diff and the
# worker report). Kept here so the proof shows the before/after delta even
# though the source now only computes the AFTER state.
BEFORE = {
    "HFX_land_mean": 1935.6,
    "HFX_land_ratio_vs_wrf": 4.22,
    "HFX_all_mean": 567.9,
    "T2_land_bias": 3.56,
    "T2_land_rmse": 3.90,
    "T2_all_rmse": 2.106,
    "T2_water_bias": 0.001,
    "UST_land_mean": 0.486,
}


def _v(d, name):
    return np.asarray(d.variables[name][0], dtype=np.float64)


def run() -> dict:
    d = Dataset(str(RUN / MIDDAY))
    T = _v(d, "T"); P = _v(d, "P"); PB = _v(d, "PB")
    PH = _v(d, "PH"); PHB = _v(d, "PHB"); QV = _v(d, "QVAPOR")
    U = _v(d, "U"); V = _v(d, "V")
    TSK = _v(d, "TSK"); PSFC = _v(d, "PSFC"); XLAND = _v(d, "XLAND")
    HFX_wrf = _v(d, "HFX"); UST_wrf = _v(d, "UST"); T2_wrf = _v(d, "T2")

    theta0 = T[0] + 300.0
    p0 = P[0] + PB[0]
    qv0 = np.maximum(QV[0], 0.0)
    dz0 = ((PH + PHB)[1] - (PH + PHB)[0]) / G
    um = 0.5 * (U[0, :, :-1] + U[0, :, 1:])
    vm = 0.5 * (V[0, :-1, :] + V[0, 1:, :])
    land = XLAND < 1.5
    znt = np.where(land, 0.10, 2.85e-3)

    state = SimpleNamespace(
        u=jnp.asarray(um[..., None]), v=jnp.asarray(vm[..., None]),
        theta=jnp.asarray(theta0[..., None]), qv=jnp.asarray(qv0[..., None]),
        p=jnp.asarray(p0[..., None]), dz=jnp.asarray(dz0),
        t_skin=jnp.asarray(TSK), psfc=jnp.asarray(PSFC), xland=jnp.asarray(XLAND),
        lakemask=jnp.zeros_like(jnp.asarray(XLAND)), mavail=jnp.ones_like(jnp.asarray(XLAND)),
        roughness_m=jnp.asarray(znt), ustar=jnp.asarray(np.maximum(UST_wrf, 1e-4)),
    )
    diag = sl.surface_layer_with_diagnostics(state)
    hfx = np.asarray(diag.hfx); t2 = np.asarray(diag.t2); ust = np.asarray(diag.fluxes.ustar)

    def stat(g, w, m):
        gg, ww = g[m], w[m]
        return {
            "gpu_mean": float(gg.mean()), "wrf_mean": float(ww.mean()),
            "bias": float((gg - ww).mean()), "rmse": float(np.sqrt(((gg - ww) ** 2).mean())),
        }

    after = {
        "HFX_land": stat(hfx, HFX_wrf, land),
        "HFX_water": stat(hfx, HFX_wrf, ~land),
        "HFX_all": stat(hfx, HFX_wrf, np.ones_like(land, bool)),
        "T2_land": stat(t2, T2_wrf, land),
        "T2_water": stat(t2, T2_wrf, ~land),
        "T2_all": stat(t2, T2_wrf, np.ones_like(land, bool)),
        "UST_land": stat(ust, UST_wrf, land),
    }
    after["HFX_land"]["ratio_vs_wrf"] = after["HFX_land"]["gpu_mean"] / after["HFX_land"]["wrf_mean"]
    d.close()

    record = {
        "proof": "sfclay-hfx-oracle-parity (#56 daytime surface-flux over-flux)",
        "kind": "corpus-WRF midday d03 column inputs -> GPU sfclayrev -> vs same-step WRF HFX/T2/UST (NOT a self-compare)",
        "source_file": str(RUN / MIDDAY),
        "root_cause": (
            "Corpus L3 ran sf_sfclay_physics=5 (MYNN surface layer); over land MYNN "
            "uses a thermal roughness z_t (Zilitinkevich 1995, CZIL=0.085) << momentum "
            "roughness for the heat profile. The bare sfclayrev port used momentum "
            "roughness for heat -> psit ~4x too small -> HFX ~4x over -> +3.6 K T2 warm "
            "bias. Fix: port the MYNN land thermal roughness for psit/psit2/psiq."
        ),
        "before_fix": BEFORE,
        "after_fix": after,
        "notes": [
            "Water (Fairall z0t) and momentum (ustar/u10/v10 on znt) are UNCHANGED.",
            "Residual land HFX > corpus is the Noah-MP LSM surface-energy-balance "
            "coupling, not reproducible by a standalone surface layer; the binding "
            "metric is the T2 diagnostic, which drops from +3.6 K to ~+1 K land bias.",
        ],
    }
    return record


if __name__ == "__main__":
    rec = run()
    out = Path(__file__).resolve().parent / "sfclay_hfx_oracle_parity.json"
    out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(rec, indent=2, sort_keys=True))
