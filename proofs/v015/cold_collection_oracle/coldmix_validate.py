"""Cold mixed-phase Thompson column oracle validator (v0.15 RAINNC lane).

Validates the JAX Thompson column scheme against a bit-exact WRF mp_gt_driver
savepoint built on a SUB-FREEZING rain+snow+graupel column (the regime that
activates rain-collecting-snow / rain-collecting-graupel / Bigg rain-freezing).
The existing precip oracle column is warm-biased and never exercises these
cold-collection lanes; this savepoint makes the missing rain sink observable.

Run (CPU only, no GPU): python proofs/v015/cold_collection_oracle/coldmix_validate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import gpuwrf.physics.thompson_column as tc  # noqa: E402
from gpuwrf.physics.thompson_column import (  # noqa: E402
    ThompsonColumnState,
    density_from_pressure_temperature,
)

ORACLE = Path("/mnt/data/wrf_gpu2/physics_oracle/microphysics_coldmix")
Q3D = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "th", "pii", "p", "dz8w")
ORACLE_TO_KERNEL = {"qv": "qv", "qc": "qc", "qr": "qr", "qi": "qi", "qs": "qs",
                    "qg": "qg", "ni": "Ni", "nr": "Nr"}


def _meta():
    side = (ORACLE / "thompson_in.sidecar.txt").read_text().splitlines()
    for ln in side:
        if ln.startswith("dims_ni_nk_nj"):
            ni, nk, nj = (int(x) for x in ln.split()[1:4])
            return ni, nk, nj
    raise RuntimeError("no dims in sidecar")


def _rd(tag, name, ni, nk, nj):
    a = np.fromfile(ORACLE / f"thompson_{tag}__{name}.f64", dtype=">f8")
    if name in ("rainnc", "rainncv", "snownc", "graupelnc", "sr"):
        return a.reshape(nj, ni)
    return a.reshape(nj, nk, ni)


def _to_cols(arr_jki):
    a = np.moveaxis(np.asarray(arr_jki, dtype=np.float64), 1, -1)  # (j,i,k)
    return jnp.asarray(np.ascontiguousarray(a.reshape(-1, a.shape[-1])))


def build_state(tag, ni, nk, nj):
    arr = {n: _rd(tag, n, ni, nk, nj) for n in Q3D}
    T = _to_cols(arr["th"] * arr["pii"])
    p = _to_cols(arr["p"])
    qv = _to_cols(arr["qv"])
    rho = density_from_pressure_temperature(p, T, qv)
    return ThompsonColumnState(
        qv=qv, qc=_to_cols(arr["qc"]), qr=_to_cols(arr["qr"]),
        qi=_to_cols(arr["qi"]), qs=_to_cols(arr["qs"]), qg=_to_cols(arr["qg"]),
        Ni=_to_cols(arr["ni"]), Nr=_to_cols(arr["nr"]),
        T=T, p=p, rho=rho, dz=_to_cols(arr["dz8w"]),
        w=jnp.zeros_like(qv),
    )


def field_errors(out_state, ref_post):
    res = {}
    for oname, kname in ORACLE_TO_KERNEL.items():
        cand = np.asarray(getattr(out_state, kname), dtype=np.float64)
        ref = np.asarray(_to_cols(ref_post[oname]), dtype=np.float64)
        floor = 1.0 if oname in ("ni", "nr") else 1e-9
        mask = np.abs(ref) > floor
        diff = np.abs(cand - ref)
        n = int(mask.sum())
        res[oname] = {
            "active_cells": n,
            "max_abs": float(diff[mask].max()) if n else 0.0,
            "max_rel": float((diff[mask] / np.abs(ref[mask])).max()) if n else 0.0,
            "mean_rel": float((diff[mask] / np.abs(ref[mask])).mean()) if n else 0.0,
        }
    return res


def surface_precip_mm(precip_dict):
    surface_keys = ("rain", "snow", "graupel", "ice")
    precip = sum(np.asarray(precip_dict[k], dtype=np.float64)
                 for k in surface_keys if k in precip_dict)
    return float(np.sum(precip)), {k: float(np.asarray(v).sum()) for k, v in precip_dict.items()}


def _column_deltas(out_state, ni, nk, nj):
    """JAX vs WRF column-integrated qr / qg / nr change (the RAINNC-relevant
    bulk rain->graupel conversion)."""
    def _wrf(name):
        return np.asarray(_rd("out", name, ni, nk, nj), dtype=np.float64) \
            - np.asarray(_rd("in", name, ni, nk, nj), dtype=np.float64)

    def _jax(kname):
        a = np.asarray(getattr(out_state, kname), dtype=np.float64).reshape(nj, ni, nk)
        a = np.moveaxis(a, -1, 1)
        return a - np.asarray(_rd("in", {"qr": "qr", "qg": "qg", "Nr": "nr"}[kname], ni, nk, nj), dtype=np.float64)

    res = {}
    for oname, kname in (("qr", "qr"), ("qg", "qg"), ("nr", "Nr")):
        w = _wrf(oname).sum()
        j = _jax(kname).sum()
        res[oname] = {"wrf_column_delta": float(w), "jax_column_delta": float(j),
                      "ratio_jax_over_wrf": float(j / w) if w != 0 else None}
    return res


def run():
    ni, nk, nj = _meta()
    state_in = build_state("in", ni, nk, nj)
    ref_post = {n: _rd("out", n, ni, nk, nj) for n in
                ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr")}

    # cold-collection ON (default)
    out_state, precip = tc._step_thompson_column_full_impl(state_in, 18.0, False)
    fe = field_errors(out_state, ref_post)
    sp, by_species = surface_precip_mm(precip)
    deltas = _column_deltas(out_state, ni, nk, nj)

    # cold-collection OFF (falsification: prove the lane is what closes the gap)
    import os
    os.environ["GPUWRF_THOMPSON_COLD_COLLECTION"] = "0"
    import importlib
    importlib.reload(tc)
    out_off, _ = tc._step_thompson_column_full_impl(state_in, 18.0, False)
    fe_off = field_errors(out_off, ref_post)
    deltas_off = _column_deltas(out_off, ni, nk, nj)
    os.environ["GPUWRF_THOMPSON_COLD_COLLECTION"] = "1"
    importlib.reload(tc)

    # WRF reference surface rain (rainncv per col, summed) and the rain-sink
    # delta the cold collection produced in the reference.
    rnv = _rd("out", "rainncv", ni, nk, nj)
    qr_in = np.asarray(_rd("in", "qr", ni, nk, nj))
    qr_out = np.asarray(_rd("out", "qr", ni, nk, nj))
    qg_in = np.asarray(_rd("in", "qg", ni, nk, nj))
    qg_out = np.asarray(_rd("out", "qg", ni, nk, nj))
    dz = np.asarray(_rd("in", "dz8w", ni, nk, nj))
    th = np.asarray(_rd("in", "th", ni, nk, nj), dtype=np.float64)
    pii = np.asarray(_rd("in", "pii", ni, nk, nj), dtype=np.float64)
    p_in = np.asarray(_rd("in", "p", ni, nk, nj), dtype=np.float64)
    qv_in = np.asarray(_rd("in", "qv", ni, nk, nj), dtype=np.float64)
    qr_in = np.asarray(qr_in, dtype=np.float64); qr_out = np.asarray(qr_out, dtype=np.float64)
    qg_in = np.asarray(qg_in, dtype=np.float64); qg_out = np.asarray(qg_out, dtype=np.float64)
    T = th * pii
    rho = np.asarray(density_from_pressure_temperature(
        jnp.asarray(p_in), jnp.asarray(T), jnp.asarray(qv_in)))

    out = {
        "savepoint": str(ORACLE),
        "n_columns": int(state_in.qv.shape[0]),
        "n_levels": nk,
        "regime": {
            "T_min_K": float(T.min()), "T_max_K": float(T.max()),
            "cold_rain_snow_levels": int(((T < 273.15) & (qr_in > 1e-6) & (np.asarray(_rd("in","qs",ni,nk,nj), dtype=np.float64) > 1e-6)).sum()),
            "cold_rain_graupel_levels": int(((T < 273.15) & (qr_in > 1e-6) & (qg_in > 1e-6)).sum()),
        },
        "wrf_reference": {
            "qr_column_delta_kg": float((qr_out - qr_in).sum()),
            "qg_column_delta_kg": float((qg_out - qg_in).sum()),
            "rainncv_mm_per_col": rnv.reshape(-1).tolist(),
            "total_rainncv_mm": float(rnv.sum()),
        },
        "jax_scheme_cold_collection_ON": {
            "surface_precip_mm": sp,
            "precip_by_species_mm": by_species,
            "per_field": fe,
            "column_deltas_vs_wrf": deltas,
        },
        "jax_scheme_cold_collection_OFF": {
            "per_field": fe_off,
            "column_deltas_vs_wrf": deltas_off,
        },
    }
    return out


if __name__ == "__main__":
    out = run()
    rep = Path(__file__).resolve().parent / "coldmix_validation_report.json"
    rep.write_text(json.dumps(out, indent=2))
    on = out["jax_scheme_cold_collection_ON"]
    off = out["jax_scheme_cold_collection_OFF"]
    print("=== COLDMIX cold-collection (rcs+rcg) validation ===")
    print(f"regime: {out['regime']}")
    print(f"qr column delta:  WRF={on['column_deltas_vs_wrf']['qr']['wrf_column_delta']:.4e}")
    print(f"  cold-collection OFF: JAX={off['column_deltas_vs_wrf']['qr']['jax_column_delta']:.4e} "
          f"(ratio {off['column_deltas_vs_wrf']['qr']['ratio_jax_over_wrf']:.2f})")
    print(f"  cold-collection ON:  JAX={on['column_deltas_vs_wrf']['qr']['jax_column_delta']:.4e} "
          f"(ratio {on['column_deltas_vs_wrf']['qr']['ratio_jax_over_wrf']:.2f})")
    print(f"qr per-cell mean_rel:  OFF={off['per_field']['qr']['mean_rel']:.3f}  ON={on['per_field']['qr']['mean_rel']:.3f}")
    print(f"qg per-cell mean_rel:  OFF={off['per_field']['qg']['mean_rel']:.3f}  ON={on['per_field']['qg']['mean_rel']:.3f}")
    print(f"\nwrote {rep}")
