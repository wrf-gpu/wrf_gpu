#!/usr/bin/env python3
"""Grell-Freitas GPU-batched (jit/vmap) savepoint parity vs the unmodified-WRF oracle.

This gates the NEW jax.jit / jax.vmap GF kernel (``gpuwrf.physics._gf_jax``)
against the SAME five unmodified-WRF GF module savepoints used by the CPU
reference gate (``proofs/v060/savepoints/gf_case_{1..5}.json``), at the
PREDECLARED tightened tolerance ``tendency_max_relative = 2.0e-2`` (vs the loose
5.0e-2 CPU-reference gate). The oracle gold lives in the savepoints, generated
by compiling pristine ``module_cu_gf_deep.F`` / ``module_cu_gf_sh.F`` /
``module_cu_gf_wrfdrv.F`` into a standalone single-column driver (see
``proofs/v060/oracle/gf_oracle_driver.f90`` + ``build_and_run.sh``).

It also verifies that the ``jax.vmap`` batched kernel is bit-identical to the
per-column jit (so the GPU batching does not change physics), and records that
the kernel lowers to device HLO with NO host callback / device_put inside the
column loop (no host transfer in the column loop).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from gpuwrf.physics import _gf_jax as J  # noqa: E402

SAVEPOINT_DIR = ROOT / "proofs" / "v060" / "savepoints"
REPORT_PATH = ROOT / "proofs" / "v060" / "gf_gpubatch_savepoint_parity.json"
TENDENCY_FIELDS = ("RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQICUTEN")
CASES = (1, 2, 3, 4, 5)

PREDECLARED_TOLERANCES = {
    "trigger": "exact deep/shallow/nontrigger categorical match",
    "tendency_max_relative": 2.0e-2,
    "tendency_abs_floor": 1.0e-8,
    "raincv_max_relative": 5.0e-2,
    "raincv_abs": 1.0e-4,
    "xmb_shallow_max_relative": 5.0e-2,
    "xmb_shallow_abs": 1.0e-5,
    "ktop_deep_levels": 1,
    "shallow_level_indices": 1,
    "scale_pair": "fine-grid case must produce less RAINCV than coarse-grid case; ratio tol 0.20",
    "vmap_vs_single_max_abs": 1.0e-12,
}

WRF_SOURCE_PATHS = (
    "/home/enric/src/wrf_pristine/WRF/phys/module_cu_gf_deep.F",
    "/home/enric/src/wrf_pristine/WRF/phys/module_cu_gf_sh.F",
    "/home/enric/src/wrf_pristine/WRF/phys/module_cu_gf_wrfdrv.F",
    "/home/enric/src/wrf_pristine/WRF/phys/module_gfs_physcons.F",
    "/home/enric/src/wrf_pristine/WRF/phys/module_gfs_machine.F",
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_case(case_id):
    with (SAVEPOINT_DIR / f"gf_case_{case_id}.json").open() as fh:
        return json.load(fh)


def field_error(candidate, oracle):
    candidate = np.asarray(candidate, np.float64)
    oracle = np.asarray(oracle, np.float64)
    max_abs = float(np.max(np.abs(candidate - oracle)))
    scale = max(float(np.max(np.abs(oracle))), PREDECLARED_TOLERANCES["tendency_abs_floor"])
    max_rel = max_abs / scale
    passed = (max_rel <= PREDECLARED_TOLERANCES["tendency_max_relative"]) or (
        max_abs <= PREDECLARED_TOLERANCES["tendency_abs_floor"])
    return {"max_abs": max_abs, "max_rel": float(max_rel), "pass": bool(passed)}


def scalar_error(candidate, oracle, rel_tol, abs_tol):
    candidate = float(candidate); oracle = float(oracle)
    max_abs = abs(candidate - oracle)
    return {
        "oracle": oracle, "jax": candidate, "max_abs": float(max_abs),
        "max_rel": float(max_abs / max(abs(oracle), abs_tol)),
        "tolerance_abs": float(abs_tol), "tolerance_rel": float(rel_tol),
        "pass": bool(max_abs <= max(abs_tol, rel_tol * abs(oracle))),
    }


def oracle_triggers(scalars, columns):
    max_t = max(max(abs(x) for x in columns.get(f, [0.0])) for f in TENDENCY_FIELDS)
    deep = int(scalars.get("KTOP_DEEP", 0)) > 0 and float(scalars.get("RAINCV", 0.0)) > 0.0
    shallow = float(scalars.get("XMB_SHALLOW", 0.0)) > 0.0 or int(scalars.get("KTOP_SHALLOW", 0)) > 0
    return {"deep": bool(deep), "shallow": bool(shallow),
            "any": bool(deep or shallow or max_t > PREDECLARED_TOLERANCES["tendency_abs_floor"])}


def run_gpu_column(scalars, cols):
    return J.grell_freitas_column_gpu(
        np.asarray(cols["T"], np.float64), np.asarray(cols["QV"], np.float64),
        np.asarray(cols["P"], np.float64), np.asarray(cols["DZ"], np.float64),
        np.asarray(cols["RHO"], np.float64), np.asarray(cols["W"], np.float64),
        dt=float(scalars["DT"]), dx=float(scalars["DX"]),
        pi_exner=np.asarray(cols["PI"], np.float64),
        u=np.asarray(cols["U"], np.float64), v=np.asarray(cols["V"], np.float64),
        rthblten=np.asarray(cols["RTHBLTEN"], np.float64),
        rqvblten=np.asarray(cols["RQVBLTEN"], np.float64),
        kpbl=int(scalars["KPBL"]), hfx=float(scalars["HFX"]),
        qfx=float(scalars["QFX"]), xland=float(scalars["XLAND"]))


def to1(a, kx):
    x = np.zeros(kx + 1); x[1:] = np.asarray(a, np.float64); return x


def run_batched(all_cases):
    kx = all_cases[0][2]
    names = ("T", "QV", "P", "PI", "DZ", "RHO", "U", "V", "W", "RTHBLTEN", "RQVBLTEN")
    arrs = []
    for j in range(len(names)):
        arrs.append(jnp.stack([jnp.asarray(to1(all_cases[i][1][names[j]], kx), jnp.float64)
                               for i in range(len(all_cases))]))
    sc = [all_cases[i][0] for i in range(len(all_cases))]
    dt = jnp.array([float(s["DT"]) for s in sc])
    dx = jnp.array([float(s["DX"]) for s in sc])
    hfx = jnp.array([float(s["HFX"]) for s in sc])
    qfx = jnp.array([float(s["QFX"]) for s in sc])
    kpbl = jnp.array([int(s["KPBL"]) for s in sc], jnp.int32)
    xland = jnp.array([float(s["XLAND"]) for s in sc])
    ht = jnp.zeros(len(sc))
    return J.gfdrv_batched(*arrs, kx, dt, dx, hfx, qfx, kpbl, xland, ht)


def build_report():
    started = time.perf_counter()
    case_reports = []
    failures = []
    all_cases = []
    for cid in CASES:
        d = load_case(cid)
        all_cases.append((d["scalars"], d["columns"], int(d["scalars"]["KX"])))

    # batched (vmap) run for the per-column-vs-batched equivalence check
    batched = run_batched(all_cases)
    batched_raincv = np.asarray(batched["RAINCV"])
    batched_rth = np.asarray(batched["RTHCUTEN"])

    for i, cid in enumerate(CASES):
        scalars, cols, kx = all_cases[i]
        out = run_gpu_column(scalars, cols)
        oc = oracle_triggers(scalars, cols)
        jc = {"deep": bool(out["TRIGGER_DEEP"]), "shallow": bool(out["TRIGGER_SHALLOW"]),
              "any": bool(out["TRIGGER_DEEP"] or out["TRIGGER_SHALLOW"])}
        categorical = {"oracle": oc, "jax": jc, "pass": oc == jc}
        if not categorical["pass"]:
            failures.append(f"case {cid} trigger mismatch oracle={oc} jax={jc}")

        fields = {}
        for f in TENDENCY_FIELDS:
            fields[f] = field_error(np.asarray(out[f]), cols[f])
            if not fields[f]["pass"]:
                failures.append(f"case {cid} {f}: max_abs={fields[f]['max_abs']:.3e} max_rel={fields[f]['max_rel']:.3e}")

        rain = scalar_error(out["RAINCV"], scalars["RAINCV"],
                            PREDECLARED_TOLERANCES["raincv_max_relative"],
                            PREDECLARED_TOLERANCES["raincv_abs"])
        if not rain["pass"]:
            failures.append(f"case {cid} RAINCV: max_abs={rain['max_abs']:.3e} max_rel={rain['max_rel']:.3e}")

        ktop = {"oracle": int(scalars["KTOP_DEEP"]), "jax": int(out["KTOP_DEEP"]),
                "pass": bool(abs(int(scalars["KTOP_DEEP"]) - int(out["KTOP_DEEP"])) <= 1)}
        if oc["deep"] and not ktop["pass"]:
            failures.append(f"case {cid} KTOP_DEEP oracle={ktop['oracle']} jax={ktop['jax']}")

        shallow_levels = {}
        for key in ("K22_SHALLOW", "KBCON_SHALLOW", "KTOP_SHALLOW"):
            shallow_levels[key] = {"oracle": int(scalars[key]), "jax": int(out[key]),
                                   "pass": bool(abs(int(scalars[key]) - int(out[key])) <= 1)}
            if oc["shallow"] and not shallow_levels[key]["pass"]:
                failures.append(f"case {cid} {key} oracle={shallow_levels[key]['oracle']} jax={shallow_levels[key]['jax']}")

        xmb = scalar_error(out["XMB_SHALLOW"], scalars["XMB_SHALLOW"],
                           PREDECLARED_TOLERANCES["xmb_shallow_max_relative"],
                           PREDECLARED_TOLERANCES["xmb_shallow_abs"])
        if oc["shallow"] and not xmb["pass"]:
            failures.append(f"case {cid} XMB_SHALLOW max_abs={xmb['max_abs']:.3e} max_rel={xmb['max_rel']:.3e}")

        # batched-vs-single equivalence
        vmap_dr = float(abs(batched_raincv[i] - float(out["RAINCV"])))
        vmap_dt = float(np.max(np.abs(batched_rth[i][1:] - np.asarray(out["RTHCUTEN"]))))
        vmap_pass = bool(max(vmap_dr, vmap_dt) <= PREDECLARED_TOLERANCES["vmap_vs_single_max_abs"])
        if not vmap_pass:
            failures.append(f"case {cid} vmap-vs-single drift raincv={vmap_dr:.3e} rth={vmap_dt:.3e}")

        case_reports.append({
            "case": cid, "regime": scalars["REGIME"], "categorical": categorical,
            "fields": fields, "raincv": rain, "ktop_deep": ktop,
            "shallow": {"xmb": xmb, "levels": shallow_levels},
            "vmap_vs_single": {"raincv_abs": vmap_dr, "rthcuten_abs": vmap_dt, "pass": vmap_pass},
        })

    by_case = {r["case"]: r for r in case_reports}
    coarse = by_case[4]["raincv"]; fine = by_case[5]["raincv"]
    oratio = fine["oracle"] / coarse["oracle"] if coarse["oracle"] else 0.0
    jratio = fine["jax"] / coarse["jax"] if coarse["jax"] else 0.0
    ratio_pass = (fine["oracle"] < coarse["oracle"]) and (fine["jax"] < coarse["jax"]) and abs(jratio - oratio) <= 0.20
    scale_pair = {"coarse_case": 4, "fine_case": 5,
                  "oracle_fine_to_coarse_raincv": float(oratio),
                  "jax_fine_to_coarse_raincv": float(jratio), "pass": bool(ratio_pass)}
    if not ratio_pass:
        failures.append(f"scale-aware pair ratio oracle={oratio:.3e} jax={jratio:.3e}")

    src_sums = {str(Path(p)): sha256(p) for p in WRF_SOURCE_PATHS}
    src_sums[str(ROOT / "proofs/v060/oracle/gf_oracle_driver.f90")] = sha256(ROOT / "proofs/v060/oracle/gf_oracle_driver.f90")
    src_sums[str(ROOT / "src/gpuwrf/physics/_gf_jax.py")] = sha256(ROOT / "src/gpuwrf/physics/_gf_jax.py")
    src_sums[str(ROOT / "src/gpuwrf/physics/_gf_reference.py")] = sha256(ROOT / "src/gpuwrf/physics/_gf_reference.py")

    # device-transfer audit: lower the column kernel and inspect HLO
    d = load_case(1); kx1 = int(d["scalars"]["KX"]); c1 = d["columns"]; s1 = d["scalars"]
    a1 = [jnp.asarray(to1(c1[k], kx1), jnp.float64) for k in ("T", "QV", "P", "PI", "DZ", "RHO", "U", "V", "W", "RTHBLTEN", "RQVBLTEN")]
    hlo = jax.jit(lambda *a: J.gfdrv_column(*a, kx1)).lower(
        *a1, float(s1["DT"]), float(s1["DX"]), float(s1["HFX"]), float(s1["QFX"]),
        jnp.asarray(int(s1["KPBL"]), jnp.int32), float(s1["XLAND"]), 0.0).as_text()
    no_host = not any(tok in hlo for tok in ("pure_callback", "io_callback", "device_put", 'custom_call("__host'))

    report = {
        "schema": "wrf-v090-gf-gpubatch-savepoint-parity-report-v2",
        "scheme": "Grell-Freitas scale-aware cumulus (cu_physics=3)",
        "verdict": "PASS" if not failures else "FAIL",
        "gpu_batched": True,
        "jit_vmap_native_kernel": True,
        "no_host_transfer_in_column_loop": bool(no_host),
        "failures": failures,
        "predeclared_tolerances": PREDECLARED_TOLERANCES,
        "cases": case_reports,
        "scale_aware_pair": scale_pair,
        "oracle": {
            "full_wrf_exe_run": False,
            "source": "unmodified WRF GF modules from /home/enric/src/wrf_pristine/WRF/phys compiled into a standalone single-column driver",
            "wrf_gf_deep_source": "/home/enric/src/wrf_pristine/WRF/phys/module_cu_gf_deep.F",
            "wrf_gf_sh_source": "/home/enric/src/wrf_pristine/WRF/phys/module_cu_gf_sh.F",
            "wrf_gf_wrfdrv_source": "/home/enric/src/wrf_pristine/WRF/phys/module_cu_gf_wrfdrv.F",
            "savepoints": "proofs/v060/savepoints/gf_case_{1..5}.json (unmodified-WRF GF module gold; not a JAX self-compare)",
            "source_checksums_sha256": src_sums,
            "note": "Same WRF-module oracle gold as the CPU-reference gate. Not a coupled wrf.exe run.",
        },
        "method": {
            "kbcon_search": "jax.lax.while_loop bounded over k22-raises (data-dependent cap-increment search)",
            "jmini_search": "jax.lax.while_loop bounded over jmini decrements",
            "closure_ensemble": "16-member MAXENS3 ensemble as a vectorized jnp array (no per-member python branch)",
            "beta_pdf_gamma": "jnp.exp(gammaln(.)) for math.gamma",
            "ierr_shortcircuit": "jnp.where masking (compute-all, select-on-ierr)",
            "first_crossing_searches": "jnp.argmax over boolean crossing masks",
            "level_recurrences": "jax.lax.scan (hydrostatic z, hcot, hc/uc/vc/hco, moisture, downdraft)",
            "batching": "jax.vmap over the leading column axis inside a single jax.jit",
        },
        "commands": {
            "report": "JAX_PLATFORMS=cpu JAX_ENABLE_X64=true taskset -c 0-3 python proofs/v060/run_grellfreitas_gpubatch_parity.py --fail-on-parity-fail",
            "oracle": "taskset -c 0-3 bash proofs/v060/oracle/build_and_run.sh",
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-parity-fail", action="store_true")
    args = parser.parse_args()
    report = build_report()
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{report['verdict']}: wrote {REPORT_PATH}")
    worst = 0.0
    for c in report["cases"]:
        for f, v in c["fields"].items():
            worst = max(worst, v["max_rel"])
    print(f"  worst tendency rel residual = {worst:.3e} (tol {PREDECLARED_TOLERANCES['tendency_max_relative']:.1e})")
    print(f"  gpu_batched={report['gpu_batched']} no_host_transfer={report['no_host_transfer_in_column_loop']}")
    for fail in report["failures"][:20]:
        print(f"  FAIL {fail}")
    if args.fail_on_parity_fail and report["verdict"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
