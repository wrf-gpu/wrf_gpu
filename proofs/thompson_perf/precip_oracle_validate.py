"""Validate the JAX FAITHFUL-EXPLICIT Thompson kernel against a PRECIPITATING
WRF Thompson oracle, and (separately) evaluate the IMPLICIT backward-Euler
sedimentation prototype against the same oracle.

The oracle is produced by the standalone single-column harness
``/home/enric/src/wrf_pristine/precip_oracle/precip_column_oracle.exe`` driving
the REAL WRF ``mp_gt_driver`` on a deliberately precipitating, near-saturated
column with ACTIVE rain / snow / graupel / cloud-ice (nonzero fall speeds). It
is dumped via the same ``module_wrfgpu2_oracle`` raw big-endian .f64 + sidecar
format the full model uses, into
``/mnt/data/wrf_gpu2/physics_oracle/microphysics_precip/``.

Two comparisons:
  (A) JAX faithful-explicit (DEFAULT shipped) Thompson vs WRF oracle.
  (B) JAX with the IMPLICIT-sed prototype swapped into ``_sedimentation`` vs the
      SAME WRF oracle (so we can see how diffusive implicit really is on a real
      precipitating profile and whether it preserves precip + column mass).

Run:
  PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 taskset -c 0-3 \
    /home/enric/miniconda3/bin/python3 proofs/thompson_perf/precip_oracle_validate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import gpuwrf.physics.thompson_column as tc
from gpuwrf.physics.thompson_column import (
    ThompsonColumnState,
    density_from_pressure_temperature,
    step_thompson_column_with_precip,
)

ROOT = Path(__file__).resolve().parents[2]
ORACLE = Path("/mnt/data/wrf_gpu2/physics_oracle/microphysics_precip")
sys.path.insert(0, str(ROOT / "proofs" / "thompson_perf"))
from implicit_sedimentation_prototype import sedimentation_implicit  # noqa: E402

# field set the harness dumped (3D)
Q3D = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "th", "pii", "p", "dz8w")
ORACLE_TO_KERNEL = {"qv": "qv", "qc": "qc", "qr": "qr", "qi": "qi", "qs": "qs",
                    "qg": "qg", "ni": "Ni", "nr": "Nr"}
P0_PA = 100000.0
RD_CP = 287.0 / 1004.0


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
    return a.reshape(nj, nk, ni)  # (j, k, i)


def _to_cols(arr_jki):
    # (j, k, i) -> (columns=j*i, levels=k). k=0 is surface (matches kernel).
    a = np.moveaxis(np.asarray(arr_jki, dtype=np.float64), 1, -1)  # (j, i, k)
    return jnp.asarray(np.ascontiguousarray(a.reshape(-1, a.shape[-1])))


def build_state(tag, ni, nk, nj):
    arr = {n: _rd(tag, n, ni, nk, nj) for n in Q3D}
    th = arr["th"]; pii = arr["pii"]
    T = _to_cols(th * pii)
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


def field_errors(out_state, ref_post, ni, nk, nj):
    """Per-field max abs/rel on cells where the WRF post field is meaningfully
    nonzero (>1e-9 for mixing ratios, >1.0 for number conc)."""
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


def column_precip_and_mass(state_in, out_state, precip_dict):
    rho = np.asarray(state_in.rho, dtype=np.float64)
    dz = np.asarray(state_in.dz, dtype=np.float64)
    mf = ("qc", "qr", "qi", "qs", "qg")
    qin = sum(np.asarray(getattr(state_in, ORACLE_TO_KERNEL[f]), dtype=np.float64) for f in mf)
    qout = sum(np.asarray(getattr(out_state, ORACLE_TO_KERNEL[f]), dtype=np.float64) for f in mf)
    qv_in = np.asarray(state_in.qv); qv_out = np.asarray(out_state.qv)
    mass_cond_in = np.sum(qin * rho * dz, axis=-1)
    mass_cond_out = np.sum(qout * rho * dz, axis=-1)
    mass_vap_in = np.sum(qv_in * rho * dz, axis=-1)
    mass_vap_out = np.sum(qv_out * rho * dz, axis=-1)
    # SURFACE PRECIP = the four WRF precipitating channels only.  Cloud-water
    # sedimentation (``cloudw``) is NOT counted as surface precip in WRF (no
    # pptXXX accumulation; module_mp_thompson.F:3824-3837) -- it is a (small)
    # water-budget SINK, so it is excluded from the precip total but INCLUDED in
    # the closure (every water-leaving channel must close the budget).
    surface_keys = ("rain", "snow", "graupel", "ice")
    precip = sum(np.asarray(precip_dict[k], dtype=np.float64) for k in surface_keys if k in precip_dict)
    all_sinks = sum(np.asarray(v, dtype=np.float64) for v in precip_dict.values())
    cloudw = np.asarray(precip_dict.get("cloudw", 0.0), dtype=np.float64)
    total_in = mass_cond_in + mass_vap_in
    total_out = mass_cond_out + mass_vap_out
    closure = (total_out - total_in) + all_sinks
    return {
        "surface_precip_mm_per_col": precip.tolist(),
        "total_surface_precip_mm": float(precip.sum()),
        "cloudw_surface_sink_mm": float(np.sum(cloudw)),
        "water_closure_max_abs_residual_kg_m2": float(np.max(np.abs(closure))),
        "water_closure_max_rel_residual": float(np.max(np.abs(closure) / np.maximum(total_in, 1e-30))),
    }


def run_scheme(label, sed_override=None):
    """Run the FULL WRF Thompson column path (correct operator ordering). When
    ``sed_override`` is given, ``tc._sedimentation`` is monkeypatched so the
    alternative sedimentation runs at WRF's exact position (after rain-evap,
    before instant melt/freeze) -- a fair scheme swap, not a re-ordered tack-on.
    """
    ni, nk, nj = _meta()
    state_in = build_state("in", ni, nk, nj)
    ref_post = {n: _rd("out", n, ni, nk, nj) for n in ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr")}

    orig = tc._sedimentation
    try:
        if sed_override is not None:
            tc._sedimentation = sed_override
        # call the un-jitted impl so the monkeypatch is honoured (the jitted
        # entry would cache the original closure).
        out_state, precip = tc._step_thompson_column_full_impl(state_in, 18.0, False)
    finally:
        tc._sedimentation = orig

    fe = field_errors(out_state, ref_post, ni, nk, nj)
    cm = column_precip_and_mass(state_in, out_state, precip)
    return {"label": label, "n_columns": int(state_in.qv.shape[0]),
            "n_levels": nk, "per_field": fe, "precip_mass": cm,
            "precip_by_species_mm": {k: float(np.asarray(v).sum()) for k, v in precip.items()}}


def wrf_reference_precip():
    ni, nk, nj = _meta()
    rnv = _rd("out", "rainncv", ni, nk, nj)  # (nj, ni)
    return {"wrf_rainncv_mm_per_col": rnv.reshape(-1).tolist(),
            "wrf_total_rainncv_mm": float(rnv.sum())}


def main():
    print("devices:", jax.devices())
    faithful = run_scheme("faithful_explicit")
    implicit = run_scheme("implicit_be_nsub1", lambda s, dt: sedimentation_implicit(s, dt, nsub=1))
    implicit2 = run_scheme("implicit_be_nsub2", lambda s, dt: sedimentation_implicit(s, dt, nsub=2))
    implicit4 = run_scheme("implicit_be_nsub4", lambda s, dt: sedimentation_implicit(s, dt, nsub=4))
    wrf = wrf_reference_precip()

    record = {
        "oracle": str(ORACLE),
        "oracle_kind": "WRF mp_gt_driver single-column PRECIPITATING (active rain/snow/graupel/ice)",
        "dt_s": 18.0,
        "nsed_substeps": tc.NSED_SUBSTEPS,
        "wrf_reference": wrf,
        "faithful_explicit": faithful,
        "implicit_be_nsub1": implicit,
        "implicit_be_nsub2": implicit2,
        "implicit_be_nsub4": implicit4,
    }
    out = ROOT / "proofs" / "thompson_perf" / "precip_oracle_validation.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
