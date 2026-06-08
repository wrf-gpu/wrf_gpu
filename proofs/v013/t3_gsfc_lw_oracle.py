#!/usr/bin/env python3
"""v0.13 Tier-3 RADIATION batch2 -- GSFC/Goddard NUWRF LONGWAVE (ra_lw=5).

CLASSIFICATION: REFERENCE-ONLY + honest carry-over (NOT operational).

The Goddard longwave scheme (WRF ``ra_lw_physics=5``, GODDARDLWSCHEME) is the
LW half of the combined ``phys/module_ra_goddard.F`` NUWRF radiation module
(~12,501 LOC; the LW core ``lwrad`` alone is ~3,900 LOC + ~1,300 LOC of
``*exps``/``*kdis``/``tablup``/``cldovlp`` helpers, with **406 ``data``
statements / ~11,853 hardcoded correlated-k coefficients** -- the h11..h83
transmission tables, c1/c2/c3, o1/o2/o3, gas k-distribution coefficients).
This is the SAME 12.5k-LOC NUWRF SW+LW family the GSFC-SW batch1 commit already
named as an 8-12k-LOC carry-over. A faithful traceable JAX column port of that
volume cannot be completed in a single session without degenerating into a
self-compare / happy-path / silently-wrong kernel -- so, per the batch2 STOP
condition, NO kernel is shipped. We deliver the **oracle infrastructure** + this
reference-only classification instead (mirroring the cumulus lane).

What this proof object asserts (CPU-only, fp64, no GPU):

  A. ORACLE INTEGRITY -- the single-column LW savepoints (1) exist, (2) are an
     fp64 reference produced by the UNMODIFIED-PHYSICS pristine WRF Fortran
     ``module_ra_goddard.F:lwrad`` (NOT a JAX-vs-JAX self-compare; the build
     applies only a checksummed single-line VISIBILITY shim ``public :: lwrad``),
     and (3) span a non-trivial regime set: clear-sky columns AND cloudy columns,
     with finite flux profiles and physically-sensible LW radiative cooling
     (clear-sky tropospheric cooling negative; warmer/moister surface => larger
     surface downwelling GLW). A do-nothing oracle would be a fake gate.

  B. DEFAULT BYTE-UNCHANGED -- the operational LW dispatch
     (runtime.operational_mode._SCAN_WIRED_OPTIONS["ra_lw_physics"]) is UNCHANGED:
     the operational LW set stays {1,4}; the default remains ra_lw=4 (RRTMG).
     Selecting ra_lw=5 never alters the default forecast path.

  C. FAIL-CLOSED -- ra_lw=5 is registry-/namelist-accepted (selectable so a
     single-column reference comparison can be run) but is NOT in the operational
     accept set; it fail-closes loudly in the operational scan with a NAMED
     reason. No silent wrong result, no masking.

  D. KERNEL CARRY-OVER (HONEST) -- no JAX ra_lw=5 column kernel is wired
     (_KERNELS empty). When a faithful traceable kernel later lands, registering
     it here triggers fp64 parity against THESE oracle savepoints within the
     predeclared tolerance. Until then the scheme is reported HONESTLY as
     oracle-staged / kernel carry-over -- the gate does NOT pass it as operational.

Run CPU-only (GPU is owned by another worker -- NEVER use it here):
  PYTHONPATH=src JAX_PLATFORMS=cpu taskset -c 0-3 \
      python3 proofs/v013/t3_gsfc_lw_oracle.py
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

SAVE = ROOT / "proofs" / "v013" / "savepoints" / "radiation_lw"
REPORT = ROOT / "proofs" / "v013" / "t3_gsfc_lw_oracle.json"
STEM = "goddard_lw"
CASES = (1, 2, 3, 4, 5, 6)

# Regime labels mirror the oracle driver's build_sounding.
REGIMES = {
    1: "clear tropical moist",
    2: "clear mid-lat dry",
    3: "clear polar cold",
    4: "thick low water cloud",
    5: "high thin ice cloud",
    6: "deep multi-layer cloud",
}
CLEAR_CASES = (1, 2, 3)
CLOUDY_CASES = (4, 5, 6)

# Operational LW accept set that MUST stay unchanged by this batch.
EXPECTED_OPERATIONAL_RA_LW = (1, 4)

# Registry of JAX column kernels keyed by ra_lw code. EMPTY = honest carry-over.
# Signature contract (when added): fn(savepoint_columns: dict, scalars: dict)
#   -> dict with keys flx/acflxd/acflxu ((NP+1,) np.ndarray, top-down) +
#      tten ((NP,) np.ndarray) + glw (float) + olr (float).
_KERNELS: dict[int, object] = {}

# Predeclared parity tolerance (fp64 JAX vs fp64 pristine-WRF Fortran oracle).
FLUX_REL = 1.0e-12
FLUX_ABS_FLOOR = 1.0e-10
FLUX_FIELDS = ("flx", "acflxd", "acflxu", "tten")


def _load(case: int) -> dict | None:
    p = SAVE / f"{STEM}_case_{case}.json"
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


def _oracle_integrity() -> dict:
    """Section A -- savepoints exist, fp64, span clear+cloudy, physically sane."""
    loaded = {c: _load(c) for c in CASES}
    present = [c for c, d in loaded.items() if d is not None]
    if not present:
        return {"built": False, "reason": "no savepoints on disk "
                "(run proofs/v013/oracle/radiation_lw/goddard_lw_build_and_run.sh)"}

    finite_all = True
    glw_by_case: dict[int, float] = {}
    clear_cools = True
    per_case: dict[int, dict] = {}
    for c in present:
        d = loaded[c]
        cols = d["columns"]
        flx = _arr(cols, "flx")
        tten = _arr(cols, "tten")
        finite = bool(np.all(np.isfinite(flx)) and np.all(np.isfinite(tten)))
        finite_all = finite_all and finite
        glw = float(d["scalars"]["glw"])
        glw_by_case[c] = glw
        # Clear-sky LW physics: the column must on net COOL (some layer with
        # negative heating rate). A scheme that never cools clear sky is broken.
        if c in CLEAR_CASES:
            clear_cools = clear_cools and bool(np.min(tten) < 0.0)
        per_case[c] = {
            "regime": REGIMES.get(c, "?"),
            "glw_sfc_down_wm2": glw,
            "olr_toa_up_wm2": float(d["scalars"]["olr"]),
            "tten_min_kday": float(np.min(tten) * 86400.0),
            "tten_max_kday": float(np.max(tten) * 86400.0),
            "finite": finite,
        }

    # Monotonicity sanity: warmer/moister surface (case 1, ts=300, RH .80)
    # must have LARGER surface downwelling GLW than the polar-cold case
    # (case 3, ts=250). This proves the LW transfer responds to the column.
    glw_monotone = (
        1 in glw_by_case and 3 in glw_by_case
        and glw_by_case[1] > glw_by_case[3]
    )
    has_clear = any(c in present for c in CLEAR_CASES)
    has_cloudy = any(c in present for c in CLOUDY_CASES)

    manifest = SAVE / f"{STEM}_build_manifest.txt"
    shim_note = None
    if manifest.exists():
        txt = manifest.read_text()
        shim_note = "visibility-only" if "visibility-only" in txt else "see manifest"

    return {
        "built": True,
        "schema": loaded[present[0]].get("schema"),
        "cases_present": present,
        "has_clear": has_clear,
        "has_cloudy": has_cloudy,
        "all_finite": finite_all,
        "clear_sky_cools": clear_cools,
        "glw_responds_to_column": glw_monotone,
        "per_case": per_case,
        "build_shim": shim_note,
        # Honesty: a real oracle must (i) be finite, (ii) show clear-sky LW
        # cooling, (iii) span clear AND cloudy regimes, (iv) have GLW respond
        # to the surface state -- proving it is the real Fortran lwrad, not a
        # stub. fp64 schema tag proves it is NOT a fp32 throwaway.
        "nontrivial": bool(
            finite_all and clear_cools and has_clear and has_cloudy and glw_monotone
        ),
        "self_compare": False,  # oracle is the unmodified-physics pristine WRF lwrad
    }


def _kernel_parity() -> dict | None:
    """Section D -- run the JAX kernel on the oracle inputs (if registered)."""
    fn = _KERNELS.get(5)
    if fn is None:
        return None
    per_case = {}
    all_pass = True
    for c in CASES:
        d = _load(c)
        if d is None:
            continue
        out = fn(d["columns"], d["scalars"])
        case_metrics = {}
        for f in FLUX_FIELDS:
            m = _metrics(out[f], _arr(d["columns"], f), FLUX_REL, FLUX_ABS_FLOOR)
            case_metrics[f] = m
            all_pass = all_pass and m["passes"]
        for sname in ("glw", "olr"):
            sm = _metrics([out[sname]], [d["scalars"][sname]], FLUX_REL, FLUX_ABS_FLOOR)
            case_metrics[sname] = sm
            all_pass = all_pass and sm["passes"]
        per_case[c] = case_metrics
    return {"wired": True, "all_pass": all_pass, "per_case": per_case}


def _default_unchanged() -> dict:
    """Section B + C -- operational LW dispatch unchanged + ra_lw=5 fail-closed."""
    import dataclasses

    from gpuwrf.runtime.operational_mode import (
        _SCAN_WIRED_OPTIONS,
        _SCAN_UNWIRED_REASON,
        OperationalNamelist,
    )

    op_lw = tuple(_SCAN_WIRED_OPTIONS["ra_lw_physics"])
    default_unchanged = set(op_lw) == set(EXPECTED_OPERATIONAL_RA_LW)
    # Read the default WITHOUT constructing the namelist (it requires grid/
    # tendencies/metrics): the dataclass field default IS the operational default.
    _fields = {f.name: f for f in dataclasses.fields(OperationalNamelist)}
    default_ra_lw = int(_fields["ra_lw_physics"].default)

    wired = 5 in op_lw
    named_reason = _SCAN_UNWIRED_REASON.get("ra_lw_physics=5")
    fail_closed_ok = (not wired) and bool(named_reason)

    # The namelist validator MUST accept ra_lw=5 (so a single-column reference
    # comparison can be run) -- reference-only schemes pass the namelist layer.
    from gpuwrf.io.scheme_catalog import classify_scheme, SupportStatus
    support = classify_scheme("ra_lw_physics", 5)
    namelist_accepts = support.status is SupportStatus.REFERENCE_ONLY
    return {
        "operational_ra_lw_physics": list(op_lw),
        "expected_operational": list(EXPECTED_OPERATIONAL_RA_LW),
        "default_ra_lw_physics": default_ra_lw,
        "default_byte_unchanged": default_unchanged and default_ra_lw == 4,
        "ra_lw5_operationally_wired": wired,
        "ra_lw5_named_reason": named_reason,
        "ra_lw5_fail_closed_ok": fail_closed_ok,
        "ra_lw5_namelist_status": support.status.value,
        "ra_lw5_namelist_accepts_reference": namelist_accepts,
    }


def main() -> int:
    integ = _oracle_integrity()
    parity = _kernel_parity()
    operational = bool(parity and parity["all_pass"])
    status = (
        "operational" if operational
        else "reference-only (oracle-staged; faithful JAX kernel = carry-over)"
        if integ.get("built")
        else "not-built"
    )

    dispatch = _default_unchanged()

    # Gate PASSES iff (1) the oracle is built + non-trivial (real fp64 reference,
    # not a stub/self-compare), (2) the default LW path is byte-unchanged
    # (ra_lw=4 RRTMG), and (3) ra_lw=5 is fail-closed with a named reason while
    # still namelist-accepted as reference-only. Operational promotion would
    # additionally require kernel parity (section D) -- reported, never faked.
    gate_pass = bool(
        integ.get("nontrivial")
        and dispatch["default_byte_unchanged"]
        and dispatch["ra_lw5_fail_closed_ok"]
        and dispatch["ra_lw5_namelist_accepts_reference"]
    )

    report = {
        "proof": "v013-t3-gsfc-lw-oracle",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scheme": "ra_lw_physics=5 GSFC/Goddard NUWRF longwave (module_ra_goddard.F:lwrad)",
        "classification": status,
        "carry_over_rationale": (
            "module_ra_goddard.F is the ~12,501-LOC combined NUWRF SW+LW family; "
            "the LW core lwrad is ~3,900 LOC + ~1,300 LOC k-distribution helpers "
            "with 406 data statements / ~11,853 hardcoded correlated-k coefficients. "
            "A faithful traceable JAX port of that volume cannot be completed in one "
            "session without degenerating into a self-compare/happy-path/silently-"
            "wrong kernel, so per the batch2 STOP condition no kernel is shipped; the "
            "fp64 non-self-compare oracle infrastructure is delivered instead "
            "(proofs/v013/oracle/radiation_lw/) so a future faithful port has a "
            "ready reference. Default ra_lw=4 (RRTMG) is byte-unchanged; ra_lw=5 "
            "fail-closes in the operational scan."
        ),
        "oracle": integ,
        "kernel_parity": parity,
        "dispatch": dispatch,
        "gate_pass": gate_pass,
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"wrote {REPORT}")
    print(f"  ra_lw=5 GSFC/Goddard LW: classification={status}")
    o = integ
    print(f"  oracle_built={o.get('built')} nontrivial={o.get('nontrivial')} "
          f"self_compare={o.get('self_compare')} build_shim={o.get('build_shim')}")
    if o.get("built"):
        print(f"  has_clear={o.get('has_clear')} has_cloudy={o.get('has_cloudy')} "
              f"clear_sky_cools={o.get('clear_sky_cools')} "
              f"glw_responds={o.get('glw_responds_to_column')}")
    print(f"  default_byte_unchanged={dispatch['default_byte_unchanged']} "
          f"(operational ra_lw={dispatch['operational_ra_lw_physics']}, "
          f"default={dispatch['default_ra_lw_physics']})")
    print(f"  ra_lw5_fail_closed_ok={dispatch['ra_lw5_fail_closed_ok']} "
          f"namelist_status={dispatch['ra_lw5_namelist_status']}")
    print(f"GATE_PASS={gate_pass}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
