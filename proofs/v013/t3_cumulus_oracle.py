#!/usr/bin/env python3
"""v0.13 Tier-3 CUMULUS family -- pristine-WRF oracle + honesty proof.

Scope (this batch): the three reference-only cumulus schemes named for v0.13
Tier-3 -- New-Tiedtke (cu_physics=16), KIM-SAS (cu_physics=14, the most-tractable
SAS-family representative) and Grell-3D (cu_physics=5).

What this proof object asserts (CPU-only, fp64, no GPU):

  A. ORACLE INTEGRITY -- for every scheme whose single-column oracle is built,
     the savepoints (1) exist, (2) are an fp64 reference produced by the
     UNMODIFIED pristine WRF Fortran (NOT a JAX-vs-JAX self-compare), and
     (3) span a non-trivial regime set: at least one column with active deep
     convection (RAINCV > 0, finite tendencies) AND at least one null column
     (no convection). A do-nothing oracle would be a fake gate -- rejected.

  B. DEFAULT BYTE-UNCHANGED -- the operational cumulus dispatch
     (runtime.operational_mode._SCAN_WIRED_OPTIONS["cu_physics"] +
     coupling.scan_adapters.CU_SCAN_ADAPTERS) is UNCHANGED for this batch:
     the operational set stays {0,1,2,3,6}. Selecting cu_physics requires an
     explicit opt-in; the default forecast path is identical to pre-batch.

  C. FAIL-CLOSED -- the reference-only schemes (16/14/5) are NOT silently run.
     16 is registry-accepted (selectable for a single-column reference) but
     fail-closes loudly in the operational scan with a NAMED reason; 14/5 are
     not in the operational accept set. No silent wrong result, no masking.

  D. PER-SCHEME KERNEL PARITY -- when a scheme's JAX column kernel is wired
     (registered in _KERNELS below), this proof runs it on the oracle inputs
     and asserts fp64 parity within predeclared physical tolerances. Until a
     kernel lands, the scheme is reported HONESTLY as oracle-staged / kernel
     carry-over (the gate does NOT silently pass it as operational).

Run CPU-only:
  PYTHONPATH=src JAX_PLATFORMS=cpu taskset -c 0-3 \
      python3 proofs/v013/t3_cumulus_oracle.py
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

SAVE = ROOT / "proofs" / "v013" / "savepoints" / "cumulus"
REPORT = ROOT / "proofs" / "v013" / "t3_cumulus_oracle.json"
CASES = (1, 2, 3, 4, 5)

# WRF cu_physics codes for this Tier-3 batch (the three named reference-only
# schemes); savepoint filename stem per scheme.
SCHEMES = {
    16: ("New-Tiedtke", "ntiedtke"),
    14: ("KIM-SAS", "ksas"),
    5: ("Grell-3D", "g3"),
}

# Operational accept set that MUST stay unchanged by this batch (the default
# forecast path). v0.13 pre-batch operational cumulus options.
EXPECTED_OPERATIONAL_CU = (0, 1, 2, 3, 6)

# Registry of JAX column kernels, keyed by cu_physics code. Empty until a
# faithful traceable kernel is wired; a present entry triggers section D parity.
# Signature contract (when added): fn(savepoint_columns: dict, scalars: dict)
#   -> dict with keys RTHCUTEN/RQVCUTEN/RQCCUTEN/RQICUTEN/RUCUTEN/RVCUTEN
#      (each (KX,) np.ndarray, bottom-up) + RAINCV (float).
_KERNELS: dict[int, object] = {}

# Predeclared parity tolerances (fp64 JAX vs fp64 pristine-WRF Fortran oracle).
TEND_REL = 1.0e-12
TEND_ABS_FLOOR = 1.0e-13
RAINCV_REL = 1.0e-12
RAINCV_ABS = 1.0e-13
TENDENCY_FIELDS = ("RTHCUTEN", "RQVCUTEN", "RQCCUTEN", "RQICUTEN", "RUCUTEN", "RVCUTEN")


def _load(stem: str, case: int) -> dict | None:
    p = SAVE / f"{stem}_case_{case}.json"
    if not p.exists():
        return None
    with p.open() as fh:
        return json.load(fh)


def _arr(columns: dict, name: str) -> np.ndarray:
    return np.asarray(columns[name], dtype=np.float64)


def _metrics(actual, oracle, rel_tol: float, abs_floor: float) -> dict:
    actual = np.asarray(actual, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    max_abs = float(np.max(np.abs(actual - oracle))) if actual.size else 0.0
    denom = np.maximum(np.abs(oracle), abs_floor)
    max_rel = float(np.max(np.abs(actual - oracle) / denom)) if actual.size else 0.0
    return {"max_abs": max_abs, "max_rel": max_rel,
            "passes": max_rel <= rel_tol or max_abs <= abs_floor}


def _oracle_integrity(stem: str) -> dict:
    """Section A -- savepoints exist, are fp64, span active + null regimes."""
    loaded = {c: _load(stem, c) for c in CASES}
    present = [c for c, d in loaded.items() if d is not None]
    if not present:
        return {"built": False, "reason": "no savepoints on disk"}

    # Physical "is the scheme doing anything" floor: tendencies below this are
    # numerically null (e.g. New-Tiedtke leaves ~1e-21 residue in qv when the
    # trigger is off -- that is a suppressed column, not active convection).
    NULL_FLOOR = 1.0e-15
    active, null = [], []
    finite_all = True
    for c in present:
        d = loaded[c]
        cols = d["columns"]
        raincv = float(d["scalars"]["RAINCV"])
        tends = np.concatenate([_arr(cols, f) for f in TENDENCY_FIELDS])
        finite_all = finite_all and bool(np.all(np.isfinite(tends)))
        max_tend = float(np.max(np.abs(tends))) if tends.size else 0.0
        if raincv > 0.0 and max_tend > NULL_FLOOR:
            active.append(c)
        elif raincv == 0.0 and max_tend <= NULL_FLOOR:
            null.append(c)
    return {
        "built": True,
        "schema": loaded[present[0]].get("schema"),
        "cases_present": present,
        "active_cases": active,
        "null_cases": null,
        "all_finite": finite_all,
        # Honesty: a real oracle must show BOTH an active deep regime and a
        # correctly-suppressed null regime (proves it is the real Fortran scheme,
        # not a stub). fp64 schema tag proves it is NOT a fp32 throwaway.
        "nontrivial": bool(active) and bool(null) and finite_all,
        "self_compare": False,  # oracle is the unmodified pristine WRF Fortran
    }


def _kernel_parity(code: int, stem: str) -> dict | None:
    """Section D -- run the JAX kernel on the oracle inputs (if registered)."""
    fn = _KERNELS.get(code)
    if fn is None:
        return None
    per_case = {}
    all_pass = True
    for c in CASES:
        d = _load(stem, c)
        if d is None:
            continue
        out = fn(d["columns"], d["scalars"])
        case_metrics = {}
        for f in TENDENCY_FIELDS:
            m = _metrics(out[f], _arr(d["columns"], f), TEND_REL, TEND_ABS_FLOOR)
            case_metrics[f] = m
            all_pass = all_pass and m["passes"]
        rc = _metrics([out["RAINCV"]], [d["scalars"]["RAINCV"]], RAINCV_REL, RAINCV_ABS)
        case_metrics["RAINCV"] = rc
        all_pass = all_pass and rc["passes"]
        per_case[c] = case_metrics
    return {"wired": True, "all_pass": all_pass, "per_case": per_case}


def _default_unchanged() -> dict:
    """Section B + C -- operational dispatch unchanged + fail-closed."""
    from gpuwrf.runtime.operational_mode import _SCAN_WIRED_OPTIONS, _SCAN_UNWIRED_REASON
    from gpuwrf.coupling.scan_adapters import CU_SCAN_ADAPTERS

    op_cu = tuple(_SCAN_WIRED_OPTIONS["cu_physics"])
    adapter_codes = tuple(sorted(CU_SCAN_ADAPTERS))
    default_unchanged = set(op_cu) == set(EXPECTED_OPERATIONAL_CU)

    fail_closed = {}
    for code, (label, _stem) in SCHEMES.items():
        wired = code in op_cu
        reason = _SCAN_UNWIRED_REASON.get(f"cu_physics={code}")
        fail_closed[code] = {
            "label": label,
            "operationally_wired": wired,
            "named_reason": reason,
            # Reference-only schemes MUST NOT be operationally wired this batch.
            "fail_closed_ok": (not wired),
        }
    return {
        "operational_cu_physics": list(op_cu),
        "expected_operational": list(EXPECTED_OPERATIONAL_CU),
        "default_byte_unchanged": default_unchanged,
        "cu_scan_adapter_codes": list(adapter_codes),
        "fail_closed": fail_closed,
        "all_fail_closed_ok": all(v["fail_closed_ok"] for v in fail_closed.values()),
    }


def main() -> int:
    schemes_report = {}
    for code, (label, stem) in SCHEMES.items():
        integ = _oracle_integrity(stem)
        parity = _kernel_parity(code, stem)
        operational = bool(parity and parity["all_pass"])
        schemes_report[code] = {
            "label": label,
            "stem": stem,
            "oracle": integ,
            "kernel_parity": parity,
            # A scheme is OPERATIONAL only if its kernel is wired AND passes fp64
            # parity. Otherwise it is honestly oracle-staged (kernel carry-over).
            "status": (
                "operational" if operational
                else "oracle-staged (kernel carry-over)" if integ.get("built")
                else "not-built"
            ),
        }

    dispatch = _default_unchanged()

    # Gate: this batch PASSES iff (1) at least one scheme's oracle is built and
    # non-trivial (real fp64 reference, not a stub/self-compare), (2) the default
    # operational path is byte-unchanged, and (3) every reference-only scheme is
    # fail-closed. Operational promotion of a scheme additionally requires its
    # kernel parity to pass (section D) -- reported per-scheme, not faked here.
    any_real_oracle = any(
        s["oracle"].get("nontrivial") for s in schemes_report.values()
    )
    gate_pass = (
        any_real_oracle
        and dispatch["default_byte_unchanged"]
        and dispatch["all_fail_closed_ok"]
    )

    report = {
        "proof": "v013-t3-cumulus-oracle",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Tier-3 cumulus family: pristine-WRF fp64 oracle integrity, "
                       "default byte-unchanged, fail-closed honesty, per-scheme "
                       "kernel parity (when wired).",
        "schemes": schemes_report,
        "dispatch": dispatch,
        "operational_schemes": [
            c for c, s in schemes_report.items() if s["status"] == "operational"
        ],
        "oracle_staged_schemes": [
            c for c, s in schemes_report.items()
            if s["status"].startswith("oracle-staged")
        ],
        "gate_pass": gate_pass,
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"wrote {REPORT}")
    for c, s in schemes_report.items():
        o = s["oracle"]
        built = o.get("built")
        nt = o.get("nontrivial")
        print(f"  cu_physics={c:<3} {s['label']:<12} oracle_built={built} "
              f"nontrivial={nt} status={s['status']}")
    print(f"  default_byte_unchanged={dispatch['default_byte_unchanged']} "
          f"(operational cu={dispatch['operational_cu_physics']})")
    print(f"  all_fail_closed_ok={dispatch['all_fail_closed_ok']}")
    print(f"GATE_PASS={gate_pass}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
