#!/usr/bin/env python3
"""v0.6.0 BouLac PBL savepoint parity against the WRF module oracle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.pbl_boulac import boulac_columns


ROOT = Path(__file__).resolve().parents[2]
SAVEPOINT_DIR = ROOT / "proofs" / "v060" / "savepoints"
REPORT_PATH = ROOT / "proofs" / "v060" / "boulac_pbl_savepoint_parity.json"
# Pristine-WRF checkout root. Override with WRF_PRISTINE_ROOT; default = sibling of repo.
WRF_PRISTINE_ROOT = Path(os.environ.get("WRF_PRISTINE_ROOT", str(ROOT.parent / "wrf_pristine" / "WRF")))
WRF_SOURCE = WRF_PRISTINE_ROOT / "phys/module_bl_boulac.F"
CASES = (1, 2, 3, 4, 5, 6)

# Predeclared before the full six-case parity run. The WRF oracle is compiled
# from unmodified source with default REAL promoted to fp64, and the candidate is
# JAX x64, so this is a roundoff-level tolerance band rather than a WRF-r4 band.
TOLERANCES = {
    "tendencies": {"abs": 5.0e-10, "rel": 5.0e-10, "floor": 1.0e-12},
    "tke": {"abs": 5.0e-10, "rel": 5.0e-10, "floor": 1.0e-12},
    "exchange": {"abs": 5.0e-10, "rel": 5.0e-10, "floor": 1.0e-12},
    "pblh": {"abs": 5.0e-9, "rel": 5.0e-12, "floor": 1.0e-12},
}

FIELD_MAP = {
    "RUBLTEN": ("u", "tendencies"),
    "RVBLTEN": ("v", "tendencies"),
    "RTHBLTEN": ("theta", "tendencies"),
    "RQVBLTEN": ("qv", "tendencies"),
    "TKE": ("tke", "tke"),
    "EXCH_H": ("exch_h", "exchange"),
    "EXCH_M": ("exch_m", "exchange"),
}


def _load(case_id: int) -> dict:
    with (SAVEPOINT_DIR / f"boulac_case_{case_id}.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def _stack(cases: list[dict], name: str) -> np.ndarray:
    return np.asarray([case["columns"][name] for case in cases], dtype=np.float64)


def _scalars(cases: list[dict], name: str) -> np.ndarray:
    return np.asarray([case["scalars"][name] for case in cases], dtype=np.float64)


def _sha256(path: Path) -> str:
    # Provenance hash of the (optional) pristine-WRF source tree. The binding
    # oracle is the vendored savepoints + boulac_wrf_source_checksums.txt, so a
    # missing WRF tree (outsider without WRF_PRISTINE_ROOT) must not fail the gate.
    if not Path(path).is_file():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _metrics(actual, expected, tol: dict[str, float]) -> dict[str, float | bool]:
    actual_arr = np.asarray(actual, dtype=np.float64)
    expected_arr = np.asarray(expected, dtype=np.float64)
    diff = actual_arr - expected_arr
    max_abs = float(np.max(np.abs(diff)))
    scale = max(float(np.max(np.abs(expected_arr))), tol["floor"])
    max_rel = max_abs / scale
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "scale": scale,
        "abs_tolerance": tol["abs"],
        "rel_tolerance": tol["rel"],
        "pass": bool(max_abs <= tol["abs"] or max_rel <= tol["rel"]),
    }


def main() -> int:
    cases = [_load(case_id) for case_id in CASES]
    out = jax.jit(
        lambda u, v, th, qv, qc, rho, dz, tke, hfx, qfx, ust, dt, cp, g: boulac_columns(
            u, v, th, qv, qc, rho, dz, tke, hfx=hfx, qfx=qfx, ust=ust, dt=dt, cp=cp, g=g
        )
    )(
        jnp.asarray(_stack(cases, "U"), dtype=jnp.float64),
        jnp.asarray(_stack(cases, "V"), dtype=jnp.float64),
        jnp.asarray(_stack(cases, "TH"), dtype=jnp.float64),
        jnp.asarray(_stack(cases, "QV"), dtype=jnp.float64),
        jnp.asarray(_stack(cases, "QC"), dtype=jnp.float64),
        jnp.asarray(_stack(cases, "RHO"), dtype=jnp.float64),
        jnp.asarray(_stack(cases, "DZ"), dtype=jnp.float64),
        jnp.asarray(_stack(cases, "TKE_IN"), dtype=jnp.float64),
        jnp.asarray(_scalars(cases, "HFX"), dtype=jnp.float64),
        jnp.asarray(_scalars(cases, "QFX"), dtype=jnp.float64),
        jnp.asarray(_scalars(cases, "UST"), dtype=jnp.float64),
        float(cases[0]["scalars"]["DT"]),
        float(cases[0]["scalars"]["CP"]),
        float(cases[0]["scalars"]["G"]),
    )

    field_metrics = {}
    worst = {"field": None, "metric": "max_abs", "value": -1.0}
    for oracle_name, (jax_name, tol_name) in FIELD_MAP.items():
        metric = _metrics(out[jax_name], _stack(cases, oracle_name), TOLERANCES[tol_name])
        field_metrics[oracle_name] = metric
        if metric["max_abs"] > worst["value"]:
            worst = {"field": oracle_name, "metric": "max_abs", "value": metric["max_abs"]}

    pblh_metric = _metrics(out["pblh"], _scalars(cases, "PBLH"), TOLERANCES["pblh"])
    if pblh_metric["max_abs"] > worst["value"]:
        worst = {"field": "PBLH", "metric": "max_abs", "value": pblh_metric["max_abs"]}

    per_case = []
    for idx, case in enumerate(cases):
        per_case_fields = {}
        for oracle_name, (jax_name, tol_name) in FIELD_MAP.items():
            per_case_fields[oracle_name] = _metrics(
                np.asarray(out[jax_name])[idx], np.asarray(case["columns"][oracle_name]), TOLERANCES[tol_name]
            )
        per_case.append(
            {
                "case": int(case["scalars"]["CASE"]),
                "regime": case["scalars"]["REGIME"],
                "pblh": _metrics(np.asarray(out["pblh"])[idx], case["scalars"]["PBLH"], TOLERANCES["pblh"]),
                "fields": per_case_fields,
            }
        )

    verdict = "PASS" if pblh_metric["pass"] and all(m["pass"] for m in field_metrics.values()) else "FAIL"
    source_hash = _sha256(WRF_SOURCE)
    report = {
        "schema": "gpuwrf.v060.boulac_pbl_savepoint_parity.v1",
        "scheme": "Bougeault-Lacarrere PBL (bl_pbl_physics=8)",
        "verdict": verdict,
        "oracle": {
            "type": "single-column Fortran driver compiled against unmodified WRF module_bl_boulac.F",
            "full_wrf_exe_run": False,
            "wrf_source": str(WRF_SOURCE),
            "wrf_source_sha256": source_hash,
            "saved_checksum_file": str(SAVEPOINT_DIR / "boulac_wrf_source_checksums.txt"),
            "generation_command": "taskset -c 0-3 bash proofs/v060/oracle/boulac_build_and_run.sh",
            "compile_precision": "gfortran -fdefault-real-8 -fdefault-double-8",
        },
        "candidate": {
            "entry": "gpuwrf.physics.pbl_boulac.boulac_columns",
            "jax_platform": "cpu",
            "jax_enable_x64": True,
            "batched": True,
        },
        "predeclared_tolerances": TOLERANCES,
        "regimes_covered": [case["scalars"]["REGIME"] for case in cases],
        "field_metrics": field_metrics,
        "pblh": pblh_metric,
        "worst_residual": worst,
        "cases": per_case,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({"verdict": verdict, "worst_residual": worst}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
