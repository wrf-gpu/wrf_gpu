#!/usr/bin/env python3
"""v0.15 MYNN BouLac mixing-length O(nz) oracle.

The dense ``(B, nz, nz)`` parcel search in ``mynn_pbl._boulac_length`` is the
measured non-radiation HBM hot spot (~14.5 ms / ~12% of the cond16 step at d01).
This proof validates the new O(nz)-memory ``lax.scan`` formulation against a
straight loop-for-loop NumPy transcription of WRF ``boulac_length``
(``module_bl_mynnedmf.F:2192-2338``) -- i.e. the actual WRF nested-DO-WHILE
algorithm executed exactly as written.  Because the change is structural (same
arithmetic, O(nz) memory instead of O(nz^2)), the new lengths must match the
WRF algorithm to ~fp64 roundoff, NOT a physics tolerance band.

This is a WRF-algorithm oracle, not a JAX-vs-JAX self-compare: the reference is
an independent NumPy implementation that mirrors the Fortran control flow.

Run (CPU only, no GPU lock needed):
  taskset -c 0-3 env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true GPUWRF_JAX_CACHE=0 \
    PYTHONPATH=src python proofs/perf/v015/boulac_nz_oracle.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("GPUWRF_JAX_CACHE", "0")

import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.mynn_constants import GTR, NL_BOULAC_LMAX
# Validate BOTH the O(nz) candidate (_boulac_length_onz) AND the production
# default dense form (_boulac_length_dense) against the WRF reference; the O(nz)
# form is the algorithm under test (default-OFF on an XLA compile pathology), the
# dense form is what ships.
from gpuwrf.physics.mynn_pbl import _boulac_length_dense, _boulac_length_onz

OUT = Path("proofs/perf/v015/boulac_nz_oracle.json")
BETA = GTR
LMAX = NL_BOULAC_LMAX


def wrf_boulac_length_column(zw_full, dz, qtke, theta):
    """Loop-for-loop NumPy transcription of WRF ``boulac_length`` (1 column).

    ``zw_full`` is zw(kts:kte+1) (length nz+1, the WRF interface array including
    the top); ``dz``, ``qtke``, ``theta`` are length nz (0-indexed kts..kte).
    Returns (lb1, lb2) length nz.
    """
    nz = dz.shape[0]
    kts, kte = 0, nz - 1  # 0-indexed WRF kts/kte
    dlu = np.zeros(nz)
    dld = np.zeros(nz)
    beta = BETA

    for iz in range(kts, kte + 1):
        # ---- upward ----
        zup = 0.0
        dlu[iz] = zw_full[kte + 1] - zw_full[iz] - dz[iz] * 0.5
        zzz = 0.0
        zup_inf = 0.0
        if iz < kte:
            found = 0
            izz = iz
            while found == 0:
                if izz < kte:
                    dzt = dz[izz]
                    zup = zup - beta * theta[iz] * dzt
                    zup = zup + beta * (theta[izz + 1] + theta[izz]) * dzt * 0.5
                    zzz = zzz + dzt
                    if qtke[iz] < zup and qtke[iz] >= zup_inf:
                        bbb = (theta[izz + 1] - theta[izz]) / dzt
                        if bbb != 0.0:
                            tl = (
                                -beta * (theta[izz] - theta[iz])
                                + np.sqrt(
                                    max(
                                        0.0,
                                        (beta * (theta[izz] - theta[iz])) ** 2
                                        + 2.0 * bbb * beta * (qtke[iz] - zup_inf),
                                    )
                                )
                            ) / bbb / beta
                        else:
                            if theta[izz] != theta[iz]:
                                tl = (qtke[iz] - zup_inf) / (
                                    beta * (theta[izz] - theta[iz])
                                )
                            else:
                                tl = 0.0
                        dlu[iz] = zzz - dzt + tl
                        found = 1
                    zup_inf = zup
                    izz = izz + 1
                else:
                    found = 1

        # ---- downward ----
        zdo = 0.0
        zdo_sup = 0.0
        dld[iz] = zw_full[iz]
        zzz = 0.0
        if iz > kts:
            found = 0
            izz = iz
            while found == 0:
                if izz > kts:
                    dzt = dz[izz - 1]
                    zdo = zdo + beta * theta[iz] * dzt
                    zdo = zdo - beta * (theta[izz - 1] + theta[izz]) * dzt * 0.5
                    zzz = zzz + dzt
                    if qtke[iz] < zdo and qtke[iz] >= zdo_sup:
                        bbb = (theta[izz] - theta[izz - 1]) / dzt
                        if bbb != 0.0:
                            tl = (
                                beta * (theta[izz] - theta[iz])
                                + np.sqrt(
                                    max(
                                        0.0,
                                        (beta * (theta[izz] - theta[iz])) ** 2
                                        + 2.0 * bbb * beta * (qtke[iz] - zdo_sup),
                                    )
                                )
                            ) / bbb / beta
                        else:
                            if theta[izz] != theta[iz]:
                                tl = (qtke[iz] - zdo_sup) / (
                                    beta * (theta[izz] - theta[iz])
                                )
                            else:
                                tl = 0.0
                        dld[iz] = zzz - dzt + tl
                        found = 1
                    zdo_sup = zdo
                    izz = izz - 1
                else:
                    found = 1

        # ---- combine ----
        dld[iz] = min(dld[iz], zw_full[iz + 1])
        dlu[iz] = max(0.1, dlu[iz] / (1.0 + dlu[iz] / LMAX))
        dld[iz] = max(0.1, dld[iz] / (1.0 + dld[iz] / LMAX))

    lb1 = np.minimum(dlu, dld)
    lb2 = np.sqrt(dlu * dld)
    lb1[kte] = lb1[kte - 1]
    lb2[kte] = lb2[kte - 1]
    return lb1, lb2


def make_cases():
    """A spread of stratification regimes (stable, near-neutral, unstable,
    inversion-capped, sharp jumps, ragged dz) -- the cases that exercise every
    WRF crossing branch (bbb!=0, bbb==0, no-crossing default, top/bottom)."""
    rng = np.random.default_rng(20260613)
    cases = []
    nz = 44

    def heights(dz):
        zwf = np.concatenate([[0.0], np.cumsum(dz)])
        return zwf

    # 1. uniform dz, weakly stable
    dz = np.full(nz, 250.0)
    th = 290.0 + 0.004 * np.cumsum(dz) + rng.normal(0, 0.2, nz)
    cases.append(("weakly_stable_uniform", dz, th))

    # 2. strongly stable
    dz = np.full(nz, 250.0)
    th = 285.0 + 0.012 * np.cumsum(dz)
    cases.append(("strongly_stable", dz, th))

    # 3. near-neutral
    dz = np.full(nz, 250.0)
    th = 300.0 + 0.0005 * np.cumsum(dz) + rng.normal(0, 0.05, nz)
    cases.append(("near_neutral", dz, th))

    # 4. unstable lower layer + capping inversion
    dz = np.full(nz, 200.0)
    z = np.cumsum(dz)
    th = np.where(z < 1500.0, 300.0 - 0.001 * z, 298.5 + 0.010 * (z - 1500.0))
    cases.append(("unstable_cap_inversion", dz, th))

    # 5. sharp jumps (multiple crossings)
    dz = np.full(nz, 150.0)
    th = 290.0 + 0.003 * np.cumsum(dz)
    th[10] += 4.0
    th[20] += 6.0
    th[30] += 3.0
    cases.append(("sharp_jumps", dz, th))

    # 6. ragged stretched dz
    dz = 100.0 * (1.05 ** np.arange(nz))
    th = 288.0 + 0.005 * np.cumsum(dz) + rng.normal(0, 0.3, nz)
    cases.append(("ragged_stretched", dz, th))

    # 7. isothermal (bbb==0 path everywhere)
    dz = np.full(nz, 250.0)
    th = np.full(nz, 295.0)
    cases.append(("isothermal_bbb0", dz, th))

    # 8. random rough turbulence
    dz = np.full(nz, 300.0) + rng.normal(0, 20.0, nz)
    th = 290.0 + 0.006 * np.cumsum(dz) + rng.normal(0, 0.8, nz)
    cases.append(("random_rough", dz, th))

    built = []
    for name, dz, th in cases:
        zwf = heights(dz)
        # qtke spread spanning the small/large TKE regimes
        qtke = np.clip(0.5 + rng.normal(0, 0.4, nz), 0.005, 5.0)
        built.append((name, dz.astype(np.float64), th.astype(np.float64),
                      qtke.astype(np.float64), zwf.astype(np.float64)))
    return built


def main() -> int:
    cases = make_cases()
    # batch them all (same nz) and run the JAX kernel once.
    nz = cases[0][1].shape[0]
    dz_b = np.stack([c[1] for c in cases])
    th_b = np.stack([c[2] for c in cases])
    qtke_b = np.stack([c[3] for c in cases])
    zw_b = np.stack([c[4][:-1] for c in cases])  # zw(kts:kte) subset for the kernel

    refs = []
    for (name, dz, th, qtke, zwf) in cases:
        lb1_ref, lb2_ref = wrf_boulac_length_column(zwf, dz, qtke, th)
        refs.append((lb1_ref, lb2_ref))

    zw_j = jnp.asarray(zw_b); dz_j = jnp.asarray(dz_b)
    qtke_j = jnp.asarray(qtke_b); th_j = jnp.asarray(th_b)

    impls = {
        "onz": _boulac_length_onz,   # the O(nz) algorithm under test
        "dense": _boulac_length_dense,  # the production default
    }
    abs_tol = 1.0e-9
    rel_tol = 1.0e-10
    impl_reports = {}
    overall_worst = {"impl": None, "case": None, "field": None, "max_abs": -1.0, "max_rel": -1.0}
    overall_pass = True
    for impl_name, fn in impls.items():
        lb1_jax, lb2_jax = fn(zw_j, dz_j, qtke_j, th_j)
        lb1_jax = np.asarray(lb1_jax, np.float64)
        lb2_jax = np.asarray(lb2_jax, np.float64)
        per_case = []
        worst = {"case": None, "field": None, "max_abs": -1.0, "max_rel": -1.0}
        for i, (name, dz, th, qtke, zwf) in enumerate(cases):
            lb1_ref, lb2_ref = refs[i]
            for field, ref, jx in (("lb1", lb1_ref, lb1_jax[i]), ("lb2", lb2_ref, lb2_jax[i])):
                d = np.abs(jx - ref)
                scale = np.maximum(np.abs(ref), 1e-12)
                rec = {"case": name, "field": field,
                       "max_abs": float(np.max(d)), "max_rel": float(np.max(d / scale))}
                per_case.append(rec)
                if rec["max_abs"] > worst["max_abs"]:
                    worst = {k: rec[k] for k in ("case", "field", "max_abs", "max_rel")}
        omax_abs = max(r["max_abs"] for r in per_case)
        omax_rel = max(r["max_rel"] for r in per_case)
        ipass = (omax_abs <= abs_tol or omax_rel <= rel_tol)
        overall_pass = overall_pass and ipass
        if worst["max_abs"] > overall_worst["max_abs"]:
            overall_worst = {"impl": impl_name, **worst}
        impl_reports[impl_name] = {
            "verdict": "PASS" if ipass else "FAIL",
            "overall_max_abs": omax_abs, "overall_max_rel": omax_rel,
            "worst": worst, "per_case": per_case,
        }

    verdict = "PASS" if overall_pass else "FAIL"
    report = {
        "schema": "gpuwrf.v015.boulac_nz_oracle.v2",
        "scope": "MYNN BouLac mixing-length: O(nz) unrolled (candidate) AND dense "
                 "(default) vs WRF nested-DO-WHILE NumPy reference",
        "wrf_reference": "module_bl_mynnedmf.F:2192-2338 (boulac_length), loop-for-loop NumPy transcription",
        "constants": {"beta_gtr": float(BETA), "Lmax": float(LMAX)},
        "nz": int(nz),
        "n_cases": len(cases),
        "regimes": [c[0] for c in cases],
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
        "verdict": verdict,
        "overall_worst": overall_worst,
        "implementations": impl_reports,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(
        {"verdict": verdict, "overall_worst": overall_worst,
         "onz_max_abs": impl_reports["onz"]["overall_max_abs"],
         "onz_max_rel": impl_reports["onz"]["overall_max_rel"],
         "dense_max_abs": impl_reports["dense"]["overall_max_abs"],
         "dense_max_rel": impl_reports["dense"]["overall_max_rel"]}, indent=2))
    print(f"wrote {OUT}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
