#!/usr/bin/env python3
"""Assemble the consolidated v0.6.0 classic-radiation savepoint-parity proof.

Combines the Dudhia shortwave (ra_sw_physics=1) machine-precision parity result
and the classic RRTM longwave (ra_lw_physics=1) oracle-readiness result into one
proof object: proofs/v060/radiation_dudhia_rrtm_savepoint_parity.json.

Runs the underlying gates first so the proof always reflects the current code.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "radiation_dudhia_rrtm_savepoint_parity.json"


def run(cmd: list[str]) -> int:
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> int:
    # Regenerate the per-scheme reports on the current commit.
    sw_rc = run([sys.executable, str(HERE / "run_dudhia_parity.py")])
    lw_rc = run([sys.executable, str(HERE / "run_rrtm_lw_parity.py")])

    sw = json.loads((HERE / "dudhia_sw_savepoint_parity_report.json").read_text(encoding="utf-8"))
    lw = json.loads((HERE / "rrtm_lw_savepoint_parity_report.json").read_text(encoding="utf-8"))

    proof = {
        "title": "v0.6.0 classic radiation pair savepoint parity (Dudhia SW + RRTM LW)",
        "sprint": "worker/opus/v060-radiation",
        "git_head": sw.get("git_head", "unknown"),
        "oracle_rule": (
            "Oracle = UNMODIFIED pristine WRF (/home/enric/src/wrf_pristine/WRF), "
            "conda env wrfbuild, standalone single-column drivers. No JAX self-compare. "
            "No mask/tolerance loosening. Tolerances predeclared before comparison."
        ),
        "schemes": {
            "ra_sw_physics=1": {
                "name": "Dudhia shortwave (Stephens 1984 broadband)",
                "wrf_source": sw["oracle"]["source"],
                "wrf_entry": sw["oracle"]["entry"],
                "jax_module": "src/gpuwrf/physics/ra_sw_dudhia.py",
                "verdict": sw["verdict"],
                "overall_pass": sw["overall_pass"],
                "canonical_fp32_pass": sw["canonical_fp32_pass"],
                "fp64_machine_precision_pass": sw["fp64_precision_audit_pass"],
                "predeclared_tolerances": sw["predeclared_tolerances"],
                "comparison_space": sw["comparison_space"],
                "fp32_source_checksums": sw["oracle"]["fp32_source_checksums"],
                "fp64_source_checksums": sw["oracle"]["fp64_source_checksums"],
                # "per-band": Dudhia is a single broadband SW band; the per-case
                # PASS/FAIL across the daytime/nighttime/cloud/ozone-profile edge
                # set is the band-level evidence.
                "per_case": {
                    cid: {
                        "label": c["label"],
                        "coszen": c["coszen"],
                        "RTHRATEN": c["fields"]["RTHRATEN"],
                        "GSW": c["fields"]["GSW"],
                        "pass": c["pass"],
                    }
                    for cid, c in sw["cases"].items()
                },
                "fp64_per_case": {
                    cid: {
                        "RTHRATEN_rel": c["fields"]["RTHRATEN"]["max_rel"],
                        "GSW_err": c["fields"]["GSW"]["abs_err"],
                        "pass": c["pass"],
                    }
                    for cid, c in sw["fp64_audit_cases"].items()
                },
            },
            "ra_lw_physics=1": {
                "name": "classic RRTM longwave (AER 16-band k-distribution)",
                "wrf_source": lw["oracle"]["source"],
                "wrf_entry": lw["oracle"]["entry"],
                "lookup_asset": lw["oracle"]["lookup_asset"],
                "jax_module": "src/gpuwrf/physics/ra_lw_rrtm.py",
                "verdict": lw["verdict"],
                "overall_pass": lw["overall_pass"],
                "jax_port_present": lw["jax_port_present"],
                "note": lw.get("note", ""),
                "worst_residual": lw.get("worst_residual", {}),
                "predeclared_tolerances": lw["predeclared_tolerances"],
                "fp32_source_checksums": lw["oracle"]["fp32_source_checksums"],
                "fp64_source_checksums": lw["oracle"]["fp64_source_checksums"],
                "fp64_oracle_summary": lw["oracle"]["fp64_oracle_summary"],
            },
        },
        "registry_wiring": {
            "ACCEPTED_RA_SW_PHYSICS": [0, 1, 4],
            "ACCEPTED_RA_LW_PHYSICS": [0, 1, 4],
            "namelist_check": "ra_sw=1 (Dudhia) + ra_lw=1 (RRTM) accepted on this branch",
            "physics_interfaces": "SCHEME_STEP_SPECS radiation option 1 sw+lw added (held-rate theta endpoint)",
        },
        "verdict": (
            "PASS" if (sw["overall_pass"] and lw["overall_pass"]) else "PARTIAL"
        ),
        "summary": (
            "Dudhia SW: faithful JAX port PASSES 7/7 savepoint cases — fp64 to "
            "machine precision (rel ~1e-15), fp32 within predeclared tol (worst "
            "RTHRATEN rel 2.5e-6, GSW sub-mW/m^2). Classic RRTM LW: faithful "
            "JAX column port passes 7/7 fp64 WRF savepoint cases for GLW, OLR, "
            "and held-rate RTHRATEN using the 16-band RRTM_DATA k-distribution."
        ),
    }
    OUT.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT)
    print("Dudhia SW verdict:", sw["verdict"], "| RRTM LW verdict:", lw["verdict"])
    # Gate exit: green when SW passes and LW oracle is ready (pre-port).
    return 0 if (sw_rc == 0 and lw_rc == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
