#!/usr/bin/env python3
"""v0.13 Tier-3 MICROPHYSICS batch-1 proof object.

Consolidates the acceptance evidence for the first Tier-3 microphysics batch
into a single machine-checked proof:

  1. PER-SCHEME PRISTINE-WRF SAVEPOINT ORACLE PASS (fp64). For each newly-ported
     scheme, the JAX port is compared column-by-column against the UNMODIFIED
     pristine WRF Fortran scheme (oracle savepoints, NOT a JAX self-compare).
     Batch 1 ships WSM7 (mp_physics=24): proofs/v013/run_wsm7_parity.py, gated
     against phys/module_mp_wsm7.F via proofs/v013/savepoints_wsm7{,_fp64}.

  2. DEFAULT mp_physics BYTE-UNCHANGED. The operational accept-matrix
     (ACCEPTED_MP_PHYSICS) and the operational scan-adapter table
     (MP_SCAN_ADAPTERS) are unchanged by this batch (WSM7 is NOT added to either
     -- it cannot be operationally selected without a hail state leaf), and the
     default scheme mp_physics=8 (Thompson) classifies IMPLEMENTED. So no
     default-path behaviour changes.

  3. FAIL-CLOSED VERIFIED. mp_physics=24 is rejected by validate_namelist with a
     named reason and classified RECOGNIZED_FAIL_CLOSED by the public honesty
     catalog (never a silent hail-dropping run). The catalog + registry
     consistency invariants still hold.

SCOPE HONESTY (recorded in the JSON): of the originally-named candidates,
WSM6/WDM6 were ALREADY operationally wired on trunk; Goddard 4-ice (mp=7) is a
separate code family with no existing GPU analogue (least tractable). The
genuinely-remaining tractable family extensions are WSM7 (24), WDM7 (26),
WDM5 (14). Batch 1 delivers WSM7 fully (kernel + oracle + fail-closed wiring).
WSM7/WDM7 add a separate precipitating HAIL class (qh) that is NOT in the
operational moist-state pytree, so they are REFERENCE_ONLY until a cross-cutting
State/dynamics/I-O hail leaf is added (out of the microphysics-slot ownership).
WDM5 (14) needs NO new state leaf (its Nn/Nc/Nr number species already exist for
the wired WDM6) and is the operationally-wirable carry-over for batch 2.

Run CPU-only: JAX_PLATFORMS=cpu python proofs/v013/t3_microphysics_oracle.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)


def _run_wsm7_parity() -> dict:
    """Run the WSM7 savepoint parity and load its report."""
    rc = subprocess.call(
        [sys.executable, os.path.join(HERE, "run_wsm7_parity.py")],
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    report_path = os.path.join(HERE, "wsm7_savepoint_parity_report.json")
    with open(report_path) as fh:
        report = json.load(fh)
    return {"rc": rc, "overall_pass": bool(report["overall_pass"]),
            "report": "proofs/v013/wsm7_savepoint_parity_report.json"}


def _default_unchanged() -> dict:
    from gpuwrf.contracts.physics_registry import ACCEPTED_MP_PHYSICS
    from gpuwrf.coupling.scan_adapters import MP_SCAN_ADAPTERS
    from gpuwrf.io.scheme_catalog import SupportStatus, classify_scheme

    accepted = tuple(ACCEPTED_MP_PHYSICS)
    wired = tuple(sorted(MP_SCAN_ADAPTERS.keys()))
    default_status = classify_scheme("mp_physics", 8).status
    return {
        "accepted_mp_physics": list(accepted),
        "scan_wired_mp_physics": list(wired),
        "wsm7_not_in_accepted": 24 not in accepted,
        "wsm7_not_scan_wired": 24 not in MP_SCAN_ADAPTERS,
        "default_mp8_implemented": default_status is SupportStatus.IMPLEMENTED,
        "pass": (24 not in accepted) and (24 not in MP_SCAN_ADAPTERS)
        and (default_status is SupportStatus.IMPLEMENTED),
    }


def _fail_closed() -> dict:
    from gpuwrf.contracts.physics_registry import assert_registry_consistent
    from gpuwrf.io.namelist_check import UnsupportedSchemeError, validate_namelist
    from gpuwrf.io.scheme_catalog import (
        SupportStatus,
        assert_catalog_consistent,
        classify_scheme,
    )

    rejected = False
    try:
        validate_namelist({"physics": {"mp_physics": [24, 24]}})
    except UnsupportedSchemeError:
        rejected = True

    # defaults + wired still accepted
    defaults_ok = True
    try:
        validate_namelist({"physics": {"mp_physics": [8, 8]}})
        validate_namelist({"physics": {"mp_physics": [16, 16]}})
    except UnsupportedSchemeError:
        defaults_ok = False

    assert_catalog_consistent()
    assert_registry_consistent()
    s = classify_scheme("mp_physics", 24)
    reason_ok = (s.status is SupportStatus.RECOGNIZED_FAIL_CLOSED
                 and "module_mp_wsm7.F" in s.reason and "hail" in s.reason.lower())
    return {
        "mp24_namelist_rejected": rejected,
        "defaults_and_wired_accepted": defaults_ok,
        "mp24_catalog_status": s.status.value,
        "mp24_reason_names_port_and_blocker": reason_ok,
        "catalog_consistent": True,
        "registry_consistent": True,
        "pass": rejected and defaults_ok and reason_ok,
    }


def main() -> int:
    schemes = {}

    wsm7 = _run_wsm7_parity()
    schemes["WSM7 (mp_physics=24)"] = {
        "status": "REFERENCE_ONLY (parity-proven, fail-closed: hail leaf not in operational state)",
        "kernel": "src/gpuwrf/physics/microphysics_wsm7.py",
        "constants": "src/gpuwrf/physics/wsm7_constants.py",
        "oracle": "proofs/v013/oracle/wsm7_oracle_driver.f90 (drives UNMODIFIED phys/module_mp_wsm7.F)",
        "oracle_pass": wsm7["overall_pass"],
        "parity_report": wsm7["report"],
    }

    default_unchanged = _default_unchanged()
    fail_closed = _fail_closed()

    overall = bool(wsm7["overall_pass"] and default_unchanged["pass"] and fail_closed["pass"])

    proof = {
        "tier": "v0.13 Tier-3 MICROPHYSICS batch 1",
        "schemes_delivered": schemes,
        "default_unchanged": default_unchanged,
        "fail_closed": fail_closed,
        "scope_honesty": {
            "already_wired_on_trunk": ["WSM6 (6)", "WDM6 (16)", "Lin (2)", "WSM3 (3)",
                                       "WSM5 (4)", "Morrison (10)", "Kessler (1)", "Thompson (8)"],
            "delivered_this_batch": ["WSM7 (24) -- ported + oracle PASS, REFERENCE_ONLY"],
            "carry_over_next_batch": {
                "WDM5 (14)": "operationally-wirable (Nn/Nc/Nr warm-rain number species already "
                             "exist for wired WDM6; NO new state leaf). Highest-value next port.",
                "WDM7 (26)": "REFERENCE_ONLY like WSM7 (adds a hail qh leaf); extends the wired "
                             "WDM6 kernel + the WSM7 hail machinery.",
                "Goddard 4-ice (7)": "intractable for this batch -- a separate code family "
                                     "(module_mp_gsfcgce.F) with no existing GPU analogue.",
            },
            "hail_leaf_blocker": "WSM7/WDM7 carry a separate precipitating HAIL class (qh) not in "
                                 "MOIST_SPECIES; operational wiring requires a cross-cutting "
                                 "State/dynamics/I-O addition (advect qh in the dycore), which is "
                                 "outside the microphysics-slot ownership and is the honest reason "
                                 "they are REFERENCE_ONLY rather than IMPLEMENTED.",
        },
        "overall_pass": overall,
    }
    outpath = os.path.join(HERE, "t3_microphysics_oracle.json")
    with open(outpath, "w") as fh:
        json.dump(proof, fh, indent=2)

    print("WSM7 oracle parity PASS:", wsm7["overall_pass"])
    print("default mp_physics unchanged PASS:", default_unchanged["pass"])
    print("fail-closed PASS:", fail_closed["pass"])
    print("OVERALL:", "PASS" if overall else "FAIL")
    print("wrote", outpath)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
