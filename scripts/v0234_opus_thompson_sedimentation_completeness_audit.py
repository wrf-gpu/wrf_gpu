#!/usr/bin/env python3
"""Completeness audit: is the missing pre-sedimentation size balance ice-only?

Standing completeness mandate for the v0234 critical phase.  For every
hydrometeor species the port sediments, this asks the same question that
localized the ice defect:

    Can the (mass, number) pair entering the fall-speed construction imply an
    unbounded particle size, and therefore an unbounded terminal velocity and
    substep count?

For each species we drive the port's own ``_fall_speeds`` with a decorrelated
pair (mass present, number driven to zero -- the exact configuration that broke
ice at d01 step 1148) and report the resulting speed and the uncapped WRF
substep count.  A species is PROTECTED when the port rebuilds its working
number from a bounded size band before the fall speed is formed.

Run (CPU only):
    JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 \
    taskset -c 13,14,15,29,30,31 nice -n 15 ionice -c 3 \
    python scripts/v0234_opus_thompson_sedimentation_completeness_audit.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def canonical_sha256(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_audit() -> dict[str, Any]:
    import jax.numpy as jnp

    from gpuwrf.physics import thompson_column as tc

    nz = 44
    dz = np.full(nz, 60.0)
    rho = np.linspace(1.15, 0.35, nz)
    T = np.full(nz, 260.0)
    p = rho * 287.0 * T

    def make_state(**species):
        base = {name: jnp.zeros(nz) for name in ("qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr", "Ns", "Ng")}
        base["qv"] = jnp.full(nz, 1.0e-3)
        base.update({k: jnp.asarray(v) for k, v in species.items()})
        return tc.ThompsonColumnState(
            **base,
            T=jnp.asarray(T),
            p=jnp.asarray(p),
            rho=jnp.asarray(rho),
            dz=jnp.asarray(dz),
            w=jnp.zeros(nz),
        )

    # A single loaded layer high in the column, number driven to zero.
    mass = np.zeros(nz)
    mass[19] = 5.0e-6
    dt = 54.0

    cases = {
        # species -> (state kwargs, index of its mass fall speed in _fall_speeds)
        "rain": (dict(qr=mass, Nr=np.zeros(nz)), 0),
        "ice": (dict(qi=mass, Ni=np.zeros(nz)), 2),
        "snow": (dict(qs=mass, Ns=np.zeros(nz)), 4),
        "graupel": (dict(qg=mass, Ng=np.zeros(nz)), 5),
    }

    species_report: dict[str, Any] = {}
    for name, (kwargs, speed_idx) in cases.items():
        state = make_state(**kwargs)
        if name == "graupel":
            # mp=8 rebuilds the diagnostic graupel number at column entry.
            state = tc._reset_mp8_graupel_number(state)
        speeds = tc._fall_speeds(state)
        vt = np.asarray(speeds[speed_idx], dtype=np.float64)
        raw_nstep = float(np.max(np.floor(dt * vt / dz + 1.0)))
        species_report[name] = {
            "zero_number_max_fall_speed_m_s": float(vt.max()),
            "uncapped_wrf_nstep": raw_nstep,
            "exceeds_static_cap": bool(raw_nstep > tc.NSED_MAX),
            "physically_plausible_speed": bool(vt.max() < 50.0),
        }

    species_report["rain"]["protection"] = (
        "_clamp_rain_number rebuilds nr(k) inside mvd [D0r*0.75, 2.5mm] "
        "(WRF 3070-3090 / 3240-3250) before _fall_speeds and before sedimentation."
    )
    species_report["ice"]["protection"] = (
        "_balance_ice_number rebuilds ni(k) inside the 5-300 um band and under the "
        "999e3 m^-3 ceiling (WRF 3033-3055 -> 3226-3234). ADDED BY THIS SPRINT; "
        "this is the defect that produced the d01 step-1148 failure."
    )
    species_report["snow"]["protection"] = (
        "mp=8 snow is SINGLE-moment: the fall speed comes from the Field et al. "
        "two-gamma moment ratio xds = smoc/smob (WRF 3711-3721), a function of qs "
        "and temperature only. No (mass, number) pair exists to decorrelate, so "
        "this omission class cannot arise. Ns is carried but never sets the speed."
    )
    species_report["graupel"]["protection"] = (
        "_graupel_distribution clamps mvd_g into [D0R, 25.4mm] and REBUILDS ng from "
        "the clamped slope; _reset_mp8_graupel_number additionally re-derives Ng "
        "diagnostically from qg at every column entry (WRF 1265-1276), so the pair "
        "cannot decorrelate across a timestep."
    )

    unprotected = [
        name
        for name, rec in species_report.items()
        if not rec["physically_plausible_speed"] or rec["exceeds_static_cap"]
    ]

    return {
        "schema": "gpuwrf.v0234.opus-thompson-sedimentation-completeness-audit.v1",
        "question": (
            "Can any sedimenting species form an unbounded terminal fall speed from a "
            "decorrelated (mass, number) pair, as cloud ice did at d01 step 1148?"
        ),
        "method": (
            "Drive the port's own _fall_speeds with mass present at k19 and the "
            "species number identically zero, then compute the uncapped WRF substep "
            "count nstep = MAX_k INT(DT/(dz/vt)+1)."
        ),
        "static_nstep_cap": int(tc.NSED_MAX),
        "species": species_report,
        "modules_audited": {
            "gpuwrf.physics.thompson_column": "mp=8 Thompson (fixed by this sprint)",
            "gpuwrf.physics.thompson_aero_column": (
                "mp=28 aerosol-aware Thompson: imports the same _fall_speeds, so it "
                "inherited the identical ice defect; its _sedimentation_aero also "
                "advected the raw Ni. Both fixed by this sprint."
            ),
        },
        "unprotected_species_after_fix": unprotected,
        "verdict": (
            "ICE_ONLY_OMISSION_CONFIRMED_ALL_SPECIES_PROTECTED"
            if not unprotected
            else "UNPROTECTED_SPECIES_REMAIN"
        ),
    }


def main() -> int:
    out_path = Path(__file__).resolve().parents[1] / (
        ".agent/sprints/2026-07-21-v0234-opus-late-ni-fix/COMPLETENESS_AUDIT.json"
    )
    audit = build_audit()
    audit["canonical_sha256"] = canonical_sha256(audit)
    out_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(f"\nwrote {out_path}")
    return 0 if not audit["unprotected_species_after_fix"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
