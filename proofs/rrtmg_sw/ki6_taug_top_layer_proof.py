"""KI-6 PROOF — RRTMG-SW intermediate `taug` UV-band fidelity (top model layer).

ROOT CAUSE (falsifies the original "top-layer pressure-duplication" hypothesis):
  The JAX SW extra model-top layer construction in `_shortwave_impl` /
  `compute_rrtmg_sw_intermediates` already matches the WRF RRTMG_SWRAD driver
  convention exactly:
      play(kte+1) = 0.5 * plev(kte+1)   (extra-layer midpoint pressure)
      plev(kte+2) = 1.0e-5 mb           (= 1.0e-3 Pa TOA interface)
      gas vmr / T duplicated from the top model layer
  All setcoef intermediates (jp, jt, jt1, fac00..fac11, indfor, indself,
  selffac, forfac, colmol) match the WRF oracle at the extra top layer to ~1e-6.

  The ONLY discrepancy driving the bands-9/10/12/13 `taug` failures was the
  OZONE SOURCE: the JAX kernel uses the WRF O3DATA climatology
  (`_wrf_o3_vmr`, the o3input=0 path: o3vmr = o3mmr*amdo) — exactly what the real
  RRTMG_SWRAD driver and the integrated-flux Tier-1 fixture use — while the SW
  *intermediate-oracle* harness used a constant `o3_vmr_default = 8e-8`. At the
  extra model-top layer the climatology ozone column is ~28x the 8e-8 constant,
  so the O3-dependent UV bands' top-layer `taug` disagreed by 1-2 orders of
  magnitude. (Verified: forcing the JAX kernel to the constant 8e-8 made ALL 14
  bands pass — confirming there is NO pressure/temperature top-layer bug.)

FIX (WRF-faithful, no JAX-vs-JAX): the SW intermediate-oracle harness
  (`scripts/wrf_rrtmg_harness.f90`) now uses the SAME WRF O3DATA climatology
  ozone the real driver uses. The oracle was regenerated against the independent
  pristine WRFv4 build (gfortran serial; Gen2 NVHPC objects unavailable). Only the
  four ozone-coupled SW arrays changed (sw_colmol / sw_taug / sw_sfluxzen /
  sw_per_band_flux); every other SW array and ALL LW arrays are bit-identical to
  the prior Gen2 oracle (proves the pristine harness is an equivalent oracle, not
  build noise).

This script re-runs the intermediate validation and the Tier-1 SW flux gate and
emits a proof JSON with before/after taug for the affected bands.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import logging

logging.getLogger("jax").setLevel(logging.ERROR)

import numpy as np
import jax

from gpuwrf.validation.rrtmg_intermediate_oracles import (
    _load_oracle,
    SW_BAND_GPOINTS,
    _to_wrf_band_axis,
    run_intermediate_validation,
)
from gpuwrf.validation.tier1_rrtmg import load_sw_fixture_state, run_tier1_sw, run_tier1_lw
from gpuwrf.physics.rrtmg_sw import compute_rrtmg_sw_intermediates

ROOT = Path(__file__).resolve().parents[2]
AFFECTED_BANDS = (9, 10, 12, 13)  # 1-based O3/O2 UV bands (jp~12-14)
PROOF = ROOT / "proofs" / "rrtmg_sw" / "ki6_taug_top_layer_proof.json"


def _band_top_layer(taug: np.ndarray, ref: np.ndarray, band_1based: int) -> dict:
    g = int(SW_BAND_GPOINTS[band_1based - 1])
    c = taug[:, :, :g, band_1based - 1]
    r = ref[:, :, :g, band_1based - 1]
    diff = np.abs(c - r)
    rel = diff / (np.abs(r) + np.finfo(np.float64).eps)
    allowed = 1.0e-8 + 1.0e-4 * np.abs(r)
    top_c = taug[:, 16, :g, band_1based - 1]
    top_r = ref[:, 16, :g, band_1based - 1]
    return {
        "band": band_1based,
        "gpoints": g,
        "pass": bool(np.all(np.isfinite(c)) and np.all(diff <= allowed)),
        "max_abs": float(diff.max()),
        "max_rel": float(rel.max()),
        "top_layer_col0_jax": [float(x) for x in top_c[0]],
        "top_layer_col0_ref": [float(x) for x in top_r[0]],
    }


def main() -> int:
    oracle = _load_oracle()
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        st, _ = load_sw_fixture_state()
        sw = compute_rrtmg_sw_intermediates(st)
    taug = np.asarray(_to_wrf_band_axis(sw.taug), dtype=np.float64)
    ref = np.asarray(oracle["sw_taug"], dtype=np.float64)

    after = [_band_top_layer(taug, ref, b) for b in AFFECTED_BANDS]
    all_bands = []
    for b in range(1, 15):
        g = int(SW_BAND_GPOINTS[b - 1])
        c = taug[:, :, :g, b - 1]
        r = ref[:, :, :g, b - 1]
        diff = np.abs(c - r)
        allowed = 1.0e-8 + 1.0e-4 * np.abs(r)
        all_bands.append({"band": b, "pass": bool(np.all(np.isfinite(c)) and np.all(diff <= allowed))})

    rec = run_intermediate_validation()
    tier1_sw = run_tier1_sw()
    tier1_lw = run_tier1_lw()

    proof = {
        "ki": "KI-6",
        "title": "RRTMG-SW intermediate taug UV-band fidelity (extra model-top layer)",
        "root_cause": (
            "Oracle ozone-source mismatch, NOT a top-layer pressure/temperature bug. JAX kernel "
            "uses the WRF O3DATA climatology (o3input=0); the SW intermediate-oracle harness used a "
            "constant 8e-8 VMR. Fixed by making the harness use the climatology and regenerating the "
            "oracle vs the independent pristine WRFv4 build."
        ),
        "before": {
            "failing_bands_1based": [9, 10, 12, 13],
            "constant_o3_vmr": 8.0e-8,
            "note": (
                "Against the constant-8e-8 oracle, bands 9/10/12/13 failed entirely at the extra "
                "top layer. Example band-9 col0 TOP jax_top[g=8] ~ "
                "[3.19e-3,7.52e-3,1.62e-2,2.54e-2,4.41e-2,2.19e-1,3.04e0,6.57e1] vs constant-oracle "
                "ref ~ [1.34e-4,3.54e-4,7.77e-4,1.15e-3,1.41e-2,1.90e-1,3.03e0,6.57e1] "
                "(top-layer climatology ozone ~28x the 8e-8 constant). Band 12 all-layer max_abs ~1.04e4."
            ),
        },
        "after": {
            "intermediate_oracle_pass": bool(rec["pass"]),
            "sw_setcoef_pass": bool(rec["sw"]["setcoef"]["pass"]),
            "sw_taur_pass": bool(rec["sw"]["taur"]["pass"]),
            "sw_sfluxzen_pass": bool(rec["sw"]["sfluxzen"]["pass"]),
            "sw_taug_all_bands_pass": bool(all(b["pass"] for b in all_bands)),
            "sw_taug_per_band_pass": all_bands,
            "affected_band_top_layer_detail": after,
        },
        "integrated_flux_gate": {
            "tier1_sw_pass": bool(tier1_sw["pass"]),
            "tier1_sw_flux_down_max_abs_w_m2": float(tier1_sw["per_field_max_abs_err"]["flux_down"]),
            "tier1_sw_flux_up_max_abs_w_m2": float(tier1_sw["per_field_max_abs_err"]["flux_up"]),
            "tier1_lw_pass": bool(tier1_lw["pass"]),
            "tolerance_w_m2": 1.0,
        },
    }
    PROOF.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(proof, indent=2, sort_keys=True))
    ok = (
        proof["after"]["intermediate_oracle_pass"]
        and proof["after"]["sw_taug_all_bands_pass"]
        and proof["integrated_flux_gate"]["tier1_sw_pass"]
        and proof["integrated_flux_gate"]["tier1_lw_pass"]
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
