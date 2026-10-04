"""Direct pristine WRF v4.7.1 source proof for the PH base-operand correction.

Rechecks every citation against the pristine checkout (no inheritance from
prior reports), quoting the exact lines and hashing the cited files.  The
central invariant: WRF's ``grid%ph_2`` is INOUT through BOTH ``advance_w``
(which updates every non-excluded column in place, ring excluded by loop
bounds) AND ``spec_bdyupdate_ph`` (which rewrites only the spec-zone ring
from the field's current — i.e. pre-advance — ring value).  A functional
transcription must compose BOTH writes onto one output array.  The falsified
`60659a2e` wiring composed only the ring write onto the pre-advance array,
which provably (retained carries) froze the interior geopotential.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

WRF = Path("<USER_HOME>/src/wrf_pristine/WRF")
SCHEMA = "gpuwrf.v0234.final-ni-fable5-source-invariant.v1"
OUT_DIR = Path(__file__).resolve().parents[1] / (
    ".agent/sprints/2026-07-14-v0234-final-ni-fable5"
)

CITATIONS = (
    {
        "id": "solve-em-small-step-uv-walk",
        "file": "dyn_em/solve_em.F",
        "lines": (1345, 1366),
        "must_contain": ["spec_bdyupdate(grid%u_2, grid%ru_tend, dts_rk", "spec_bdyupdate(grid%v_2, grid%rv_tend, dts_rk"],
        "claim": "u/v ring walk inside the acoustic loop uses the large-step tendency arrays whose ring values spec_bdy_dry set to the coupled record tendencies.",
    },
    {
        "id": "solve-em-small-step-t-mu-muts-walk",
        "file": "dyn_em/solve_em.F",
        "lines": (1460, 1492),
        "must_contain": ["spec_bdyupdate(grid%t_2, t_tend, dts_rk", "spec_bdyupdate(grid%mu_2, mu_tend, dts_rk", "spec_bdyupdate(grid%muts, mu_tend, dts_rk"],
        "claim": "t/mu/muts ring walk after advance_mu_t; muts walks with the same mu_tend.",
    },
    {
        "id": "solve-em-advance-w-inout-ph",
        "file": "dyn_em/solve_em.F",
        "lines": (1498, 1521),
        "must_contain": ["CALL advance_w( grid%w_2, rw_tend, grid%ww, w_save"],
        "claim": "advance_w receives grid%w_2 and grid%ph_2 INOUT inside the acoustic loop; the interior geopotential advances here every substep.",
    },
    {
        "id": "solve-em-spec-bdyupdate-ph-after-advance-w",
        "file": "dyn_em/solve_em.F",
        "lines": (1583, 1622),
        "must_contain": ["CALL spec_bdyupdate_ph( ph_save, grid%ph_2, ph_tend", "CALL spec_bdyupdate ( grid%w_2, rw_tend, dts_rk"],
        "claim": "spec_bdyupdate_ph runs AFTER advance_w on the SAME grid%ph_2 array (ring only); nested w uses spec_bdyupdate with rw_tend.",
    },
    {
        "id": "advance-w-ring-exclusion-bounds",
        "file": "dyn_em/module_small_step_em.F",
        "lines": (1272, 1284),
        "must_contain": ["i_start = max(its,ids+1)", "j_start = max(jts,jds+1)"],
        "claim": "advance_w's specified/nested loop bounds exclude ONLY the outermost row/column; every other column (including relaxation rings) is advanced in place, so the ring's pre-update value entering spec_bdyupdate_ph is the previous substep's walked value while the interior is freshly advanced.",
    },
    {
        "id": "spec-bdyupdate-ph-in-place-equation",
        "file": "dyn_em/module_bc_em.F",
        "lines": (83, 100),
        "must_contain": ["MU_OLD(i,j) = MUTS(i,j) - dt*MU_TEND(i,j)", "field(i,k,j) = field(i,k,j)*(c1(k)*mu_old(i,j)+c2(k))/(c1(k)*muts(i,j)+c2(k)) +"],
        "claim": "spec_bdyupdate_ph reads field(i,k,j) IN PLACE (the ring's current value) and applies the mass-reweighted walk; it never rewrites non-ring cells.",
    },
    {
        "id": "spec-bdyupdate-additive-in-place",
        "file": "share/module_bc.F",
        "lines": (2015, 2022),
        "must_contain": ["field(i,k,j) = field(i,k,j) + dt*field_tend(i,k,j)"],
        "claim": "spec_bdyupdate is the plain in-place additive walk on the ring's current value.",
    },
    {
        "id": "small-step-prep-ph-work-convention",
        "file": "dyn_em/module_small_step_em.F",
        "lines": (270, 283),
        "must_contain": ["ph_save(i,k,j) = ph_2(i,k,j)", "ph_2(i,k,j) = ph_1(i,k,j)-ph_2(i,k,j)"],
        "claim": "ph work array = ph_1 - ph_save at stage start; the port's uncoupled perturbation-delta convention matches.",
    },
    {
        "id": "small-step-finish-ph-reconstruction",
        "file": "dyn_em/module_small_step_em.F",
        "lines": (398, 407),
        "must_contain": ["ph_2(i,k,j) = ph_2(i,k,j) + ph_save(i,k,j)"],
        "claim": "physical ph at stage end = work + save. With the interior work frozen (the falsified wiring) the physical interior geopotential can never change, exactly as the retained carries show.",
    },
    {
        "id": "spec-bdy-dry-nested-w-record",
        "file": "dyn_em/module_bc_em.F",
        "lines": (519, 530),
        "must_contain": ["if(config_flags%nested)", "CALL spec_bdytend (   rw_tend,"],
        "claim": "nested domains stage the w boundary-record tendency into rw_tend's ring; all six walked families (u,v,ph,t,mu,w) exist and the corrected port stages each.",
    },
)


def canonical_hash(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("proof_sha256", None)
    return hashlib.sha256(
        json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    version = (WRF / "README").read_text(errors="replace").splitlines()
    version_line = next(line for line in version if "Version" in line).strip()
    if "4.7.1" not in version_line:
        raise RuntimeError(f"pristine checkout is not 4.7.1: {version_line}")
    head = subprocess.run(
        ["git", "-C", str(WRF), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    rows = []
    for citation in CITATIONS:
        path = WRF / citation["file"]
        text = path.read_text(errors="replace").splitlines()
        lo, hi = citation["lines"]
        quoted = text[lo - 1 : hi]
        for needle in citation["must_contain"]:
            if not any(needle in line for line in quoted):
                raise RuntimeError(
                    f"{citation['id']}: '{needle}' not found in "
                    f"{citation['file']}:{lo}-{hi}"
                )
        rows.append(
            {
                "id": citation["id"],
                "file": citation["file"],
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "lines": [lo, hi],
                "quote": quoted,
                "claim": citation["claim"],
            }
        )

    proof = {
        "schema": SCHEMA,
        "pristine_root": str(WRF),
        "pristine_version_line": version_line,
        "pristine_git_head": head,
        "citations": rows,
        "falsified_element": (
            "0b4374a5/60659a2e passed the pre-advance_w work array as the output "
            "base of spec_bdyupdate_ph_tendency_inloop, so the returned ph work "
            "array kept the pre-advance interior: every interior advance_w "
            "geopotential update was discarded each acoustic substep. WRF's "
            "in-place INOUT composition (advance_w interior + spec_bdyupdate_ph "
            "ring) requires the advanced array as the base."
        ),
        "remaining_valid_elements": [
            "985f5714 nested boundary construction/stagger/corner repairs (untouched)",
            "0b4374a5 additive pre-substep ring cadence for U/V (advance_uv pin uses the advanced field as base, ring-only overwrite)",
            "0b4374a5 additive cadence for MU/MUTS/THETA (advance_mu_t outputs as base, ring-only overwrite; muave untouched, matching solve_em)",
            "0b4374a5 additive cadence for W (advance_w output as base, ring-only overwrite)",
            "0b4374a5 spec_bdyupdate_ph ring equation (algebra identical to module_bc_em.F; only the base operand was wrong)",
            "60659a2e active spec_zone threading (orthogonal; production nested spec_zone=1)",
        ],
        "verdict": "SOURCE_DISCREPANCY_PROVEN",
        "commands": [
            "PYTHONPATH=.:src python scripts/v0234_final_ni_fable5_source_proof.py"
        ],
    }
    proof["proof_sha256"] = canonical_hash(proof)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "source-invariant-proof.json"
    out_path.write_text(json.dumps(proof, indent=1, sort_keys=True) + "\n")
    print(f"verdict={proof['verdict']}")
    print(f"proof_sha256={proof['proof_sha256']}")
    print(f"written={out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
