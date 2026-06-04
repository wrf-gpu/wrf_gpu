"""v0.9.0 Thompson (mp_physics=8) ACTIVE-PRECIP isolated WRF-savepoint parity.

Oracle = UNMODIFIED pristine WRF v4.7.1 ``module_mp_thompson.F`` (``mp_gt_driver``),
captured by ``module_wrfgpu2_oracle.F`` around the Thompson call in
``module_microphysics_driver.F``, at a LATE timestep (WRFGPU2_ORACLE_STEP, ~5h into
the forecast) where warm-rain (qc,qr) and ice/snow (qi,qs) are ACTIVE -- unlike the
pre-existing itimestep=1 oracle (proofs/b1), which is a near-inactive step (all
hydrometeors zero in+out). This REPLACES the inactive oracle with a real precipitating
one. JAX-vs-WRF, NOT a self-compare; tolerances are the frozen Phase-B transcription
ladder (phase_b_savepoint.PHASE_B_TOLERANCES), NOT loosened to pass.

Validates ALL moist species (qv,qc,qr,qi,qs,qg) + number concentrations (Ni,Nr) +
theta + water-mass/surface-precip closure, on the inactive-physical moist mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import gpuwrf  # noqa: F401  enables jax_enable_x64 at import (fp64 throughout)

from gpuwrf.validation.tier1_thompson import run_oracle_parity_f64

ORACLE_DIR = Path("/mnt/data/wrf_gpu2/physics_oracle_v090/microphysics")
PRISTINE_SRC = Path("/home/enric/src/wrf_pristine/WRF/phys/module_mp_thompson.F")
DEFAULT_OUT = Path(__file__).resolve().parent / "thompson_savepoint_parity.json"


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hydrometeor_activity(oracle_dir: Path) -> dict[str, Any]:
    """Report which species are actually active in this oracle (honesty)."""
    mpath = oracle_dir / "manifest.json"
    if not mpath.exists():
        return {}
    m = json.loads(mpath.read_text())
    act = {}
    for f in m["fields"]:
        if f["name"] in ("qc", "qr", "qi", "qs", "qg", "ni", "nr", "qv"):
            act.setdefault(f["name"], {})[f["tag"]] = {"min": f["min"], "max": f["max"], "mean": f["mean"]}
    return act


def run(out_path: Path, oracle_dir: Path = ORACLE_DIR, dt: float | None = None) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not (oracle_dir / "manifest.json").exists():
        rec = {"proof": "v090-thompson-savepoint-parity", "status": "PENDING-ORACLE",
               "oracle_dir": str(oracle_dir),
               "reason": "active-precip oracle not yet captured (WRF re-run in flight)"}
        out_path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
        return rec

    # Reuse the PROVEN B1 fp64 oracle-parity engine (frozen tolerance ladder +
    # inactive-physical moist mask + theta float32-ULP analysis + water closure).
    record = run_oracle_parity_f64(oracle_dir=oracle_dir, out=out_path, dt=dt)
    record["proof"] = "v090-thompson-savepoint-parity"
    record["wrf_source"] = str(PRISTINE_SRC)
    record["wrf_source_sha256"] = _sha256(PRISTINE_SRC)
    record["hydrometeor_activity"] = _hydrometeor_activity(oracle_dir)
    record["oracle_note"] = (
        "ACTIVE-PRECIP oracle (late forecast step) replacing the inactive "
        "itimestep=1 oracle (proofs/b1) which had all hydrometeors zero."
    )
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--oracle-dir", default=str(ORACLE_DIR))
    ap.add_argument("--dt", type=float, default=None)
    args = ap.parse_args()
    r = run(Path(args.out), Path(args.oracle_dir), args.dt)
    summary = {
        "proof": r.get("proof"),
        "status": r.get("status"),
        "pass": r.get("pass"),
        "per_field_pass": {k: v.get("pass") for k, v in r.get("per_field", {}).items()},
        "hydrometeor_activity_out_max": {
            k: v.get("out", {}).get("max") for k, v in r.get("hydrometeor_activity", {}).items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if r.get("status") not in ("COMPARED", "ORACLE-VALIDATED") or r.get("pass") else 2)
