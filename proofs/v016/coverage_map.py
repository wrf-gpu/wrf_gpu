#!/usr/bin/env python
"""v0.16 STABILITY — recomputable scheme coverage map.

Single source of truth = the registry's ACCEPTED_* tuples intersected with the
operational fail-closed coupled-runnable authority
(``runtime.operational_mode._SCAN_WIRED_OPTIONS`` + the surface-physics gating in
``_resolve_operational_suite``).  Emits the 32-implemented-scheme coverage table
and ``coverage_map.json`` so the v0.16 plan never drifts from the code.

Classification per (family, option):
  status        : the registry's own 'implemented' | 'accepted' field
  coupled_runnable : can ride the operational device scan (the only way to do an
                     L2 coupled real-grid run); else reference-only (L1 oracle).
  coupled_tested   : in the Switzerland+Canary 6-scheme baseline suite (the 6).
  l2_target        : coupled_runnable AND NOT coupled_tested  (the v0.16 work).

CPU-only, no GPU.  Run:
  PYTHONPATH=src python proofs/v016/coverage_map.py
"""
from __future__ import annotations

import json
from pathlib import Path

from gpuwrf.contracts.physics_registry import (
    ACCEPTED_BL_PBL_PHYSICS,
    ACCEPTED_CU_PHYSICS,
    ACCEPTED_MP_PHYSICS,
    ACCEPTED_RA_LW_PHYSICS,
    ACCEPTED_RA_SW_PHYSICS,
    ACCEPTED_SF_SFCLAY_PHYSICS,
    ACCEPTED_SF_SURFACE_PHYSICS,
    CU_SCHEMES,
    MP_SCHEMES,
    PBL_SCHEMES,
    RA_LW_SCHEMES,
    RA_SW_SCHEMES,
    SFCLAY_SCHEMES,
    SURFACE_SCHEMES,
)
from gpuwrf.runtime.operational_mode import _SCAN_WIRED_OPTIONS

HERE = Path(__file__).resolve().parent

# The Switzerland + Canary coupled suite (the 6 already-coupled-tested schemes).
# cu=1 (Kain-Fritsch) is the Canary nested parent's cumulus; the Switzerland d01
# replay runs cu=0 (convection-permitting).  Both regions run the identical
# Thompson / MYNN-PBL / MYNN-sfclay / Noah-MP / RRTMG-SW / RRTMG-LW suite.
COUPLED_BASELINE = {
    "mp_physics": 8,
    "bl_pbl_physics": 5,
    "sf_sfclay_physics": 5,
    "cu_physics": 1,  # KF (Canary parent); cu=0 in the d01 replay
    "sf_surface_physics": 4,  # Noah-MP
    "ra_sw_physics": 4,  # RRTMG SW
    "ra_lw_physics": 4,  # RRTMG LW
}

FAMILIES = [
    ("mp_physics", ACCEPTED_MP_PHYSICS, MP_SCHEMES),
    ("bl_pbl_physics", ACCEPTED_BL_PBL_PHYSICS, PBL_SCHEMES),
    ("sf_sfclay_physics", ACCEPTED_SF_SFCLAY_PHYSICS, SFCLAY_SCHEMES),
    ("cu_physics", ACCEPTED_CU_PHYSICS, CU_SCHEMES),
    ("sf_surface_physics", ACCEPTED_SF_SURFACE_PHYSICS, SURFACE_SCHEMES),
    ("ra_sw_physics", ACCEPTED_RA_SW_PHYSICS, RA_SW_SCHEMES),
    ("ra_lw_physics", ACCEPTED_RA_LW_PHYSICS, RA_LW_SCHEMES),
]


def _coupled_runnable(family: str, option: int) -> bool:
    """Mirror the fail-closed coupled-runnability authority.

    For families gated by ``_SCAN_WIRED_OPTIONS`` the membership IS the authority.
    ``sf_surface_physics`` is gated separately in ``_resolve_operational_suite``:
    Noah-MP(4) runs (via use_noahmp=True) and Noah-classic(2) runs with the
    noahclassic bundle; slab(1) is reference-only.
    """
    wired = _SCAN_WIRED_OPTIONS.get(family)
    if wired is not None:
        return option in wired
    if family == "sf_surface_physics":
        # Authority: _resolve_operational_suite. 4=NoahMP and 2=Noah-classic are
        # coupled-runnable (2 needs an explicit static+land bundle); 1=slab is
        # reference-only (TSLB carry not threaded).
        return option in (2, 4)
    return False


def build() -> dict:
    rows = []
    totals = {
        "implemented_excl0": 0,
        "coupled_runnable": 0,
        "coupled_tested": 0,
        "l2_targets": 0,
        "reference_only": 0,
    }
    for family, accepted, meta in FAMILIES:
        for opt in accepted:
            if opt == 0:
                continue  # disabled/passive sentinel is not a "scheme"
            scheme = meta[opt]
            runnable = _coupled_runnable(family, opt)
            tested = COUPLED_BASELINE.get(family) == opt
            l2 = runnable and not tested
            row = {
                "family": family,
                "option": opt,
                "name": scheme.name,
                "wrf_package": scheme.wrf_package,
                "status": scheme.status,  # registry 'implemented' | 'accepted'
                "coupled_runnable": runnable,
                "coupled_tested": tested,
                "l2_target": l2,
                "reference_only": not runnable,
            }
            rows.append(row)
            totals["implemented_excl0"] += 1
            totals["coupled_runnable"] += int(runnable)
            totals["coupled_tested"] += int(tested)
            totals["l2_targets"] += int(l2)
            totals["reference_only"] += int(not runnable)

    # Mandatory pairings the L2 harness must honour.
    pairings = {
        "bl_pbl_physics=2": "sf_sfclay_physics=2 (MYJ <-> Janjic Eta)",
        "sf_sfclay_physics=2": "bl_pbl_physics=2 (Janjic Eta <-> MYJ)",
        "cu_physics=6": "use_flux_advection=True + moist_adv_opt>=1 (Tiedtke RQVFTEN)",
    }
    return {"schema": "V016CoverageMap", "totals": totals, "rows": rows,
            "coupled_baseline": COUPLED_BASELINE, "mandatory_pairings": pairings}


def _fmt_table(payload: dict) -> str:
    lines = []
    hdr = f"{'family':<20}{'opt':>4} {'name':<33}{'status':<12}{'runnable':<10}{'tested':<8}{'L2':<5}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in payload["rows"]:
        lines.append(
            f"{r['family']:<20}{r['option']:>4} {r['name'][:32]:<33}{r['status']:<12}"
            f"{('yes' if r['coupled_runnable'] else 'REF'):<10}"
            f"{('YES' if r['coupled_tested'] else '-'):<8}"
            f"{('TARGET' if r['l2_target'] else '-'):<5}"
        )
    t = payload["totals"]
    lines.append("")
    lines.append(
        f"implemented(excl0)={t['implemented_excl0']}  coupled_runnable={t['coupled_runnable']}  "
        f"coupled_tested={t['coupled_tested']}  L2_targets={t['l2_targets']}  "
        f"reference_only={t['reference_only']}"
    )
    return "\n".join(lines)


def main() -> int:
    payload = build()
    print(_fmt_table(payload))
    out = HERE / "coverage_map.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
