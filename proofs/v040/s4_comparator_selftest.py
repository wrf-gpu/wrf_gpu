#!/usr/bin/env python3
"""S4 proof generator — comparator self-test report (v0.4.0).

Produces ``proofs/v040/s4_comparator_selftest_report.json``: the proof object for
the S4 comparator harness. It demonstrates, on the real.exe oracle corpus
(cores 0-3, CPU, NO GPU):

  1. **SANITY (ungameable):** a real.exe-vs-itself campaign (the candidate is a
     verbatim copy of each oracle file) over >=10 cases (d01/d02/d03) + wrfbdy_d01
     — every field RMSE/maxabs == ~0 and EVERY case PASSES. A comparator that
     compared the wrong variable or always-passed could not produce exact-0 here.

  2. **TOL TABLE WIRED:** the report embeds the frozen WRFINPUT_TOLS / WRFBDY_TOLS
     each field was judged against (read straight from the frozen types.py).

  3. **STUB-CANDIDATE PASS/FAIL MECHANICS:** a deliberately perturbed candidate
     (T +5 K, MU +200 Pa, ISLTYP +1) over the SAME cases — the comparator FAILS
     exactly the perturbed fields (and the case), proving it is not a rubber
     stamp; a sub-tolerance perturbation (T +0.05 K) still PASSES.

  4. **FORECAST-GATE SCAFFOLD:** the predeclared 24h native-init forecast-gate
     plan (no GPU run; the manager executes it in S5).

RUN (the standing self-test, non-GPU):
  JAX_PLATFORM_NAME=cpu PYTHONPATH=src taskset -c 0-3 \\
      python3 proofs/v040/s4_comparator_selftest.py \\
      --out proofs/v040/s4_comparator_selftest_report.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the test helper importable when run as a script.
_REPO = Path(__file__).resolve().parents[2]
for p in (str(_REPO / "src"), str(_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from gpuwrf.init.real_init import comparator as C  # noqa: E402
from gpuwrf.init.real_init.types import (  # noqa: E402
    REAL_INIT_TYPES_VERSION,
    WRFBDY_TOLS,
    WRFINPUT_TOLS,
)
from tests.init.real_init._oracle_product import build_product_from_oracle  # noqa: E402


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(["git", "-C", str(_REPO), *args],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover
        return "UNKNOWN"


def _sanity_factory(case: C.OracleCase, domain: str):
    """Verbatim oracle->product copy (must give ~0 error vs the same oracle)."""
    return build_product_from_oracle(
        case.wrfinput[domain], domain=domain,
        wrfbdy_path=case.wrfbdy_d01 if domain == "d01" else None)


def _make_perturbed_factory(perturb: dict[str, float]):
    def factory(case: C.OracleCase, domain: str):
        return build_product_from_oracle(
            case.wrfinput[domain], domain=domain,
            wrfbdy_path=case.wrfbdy_d01 if domain == "d01" else None,
            perturb=perturb)
    return factory


def _worst_ok_error(campaign: dict) -> dict[str, float]:
    """Max RMSE / maxabs over all OK-status fields in a campaign roll-up."""
    worst_rmse = 0.0
    worst_maxabs = 0.0
    for name, s in campaign["field_failure_summary"].items():
        if s["n_scored"] > 0:
            worst_rmse = max(worst_rmse, s["worst_rmse"])
            worst_maxabs = max(worst_maxabs, s["worst_maxabs"])
    return {"worst_ok_rmse": worst_rmse, "worst_ok_maxabs": worst_maxabs}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_REPO / "proofs/v040/s4_comparator_selftest_report.json"))
    ap.add_argument("--root", default=str(C.ORACLE_ROOT_DEFAULT))
    ap.add_argument("--min-cases", type=int, default=10)
    ap.add_argument("--max-cases", type=int, default=12,
                    help="cap cases for a fast standing proof (>= min-cases)")
    args = ap.parse_args(argv)

    cases = C.discover_oracle_cases(
        args.root, require_domains=("d01", "d02", "d03"),
        require_wrfbdy=True, limit=args.max_cases)

    if len(cases) < args.min_cases:
        report = {
            "schema": "v0.4.0-S4-selftest-report-2026-06-02",
            "status": "ORACLE_UNAVAILABLE",
            "n_cases_found": len(cases),
            "min_cases_required": args.min_cases,
            "oracle_root": args.root,
            "note": "real.exe oracle corpus not available; cannot run the campaign.",
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"[S4] ORACLE_UNAVAILABLE ({len(cases)} cases) -> {args.out}")
        return 1

    print(f"[S4] discovered {len(cases)} oracle cases (d01/d02/d03 + wrfbdy_d01)")

    # 1. SANITY: real.exe vs itself -> ~0 error, all PASS.
    print("[S4] running SANITY campaign (oracle vs itself)...")
    sanity = C.run_campaign(_sanity_factory, cases=cases,
                            domains=("d01", "d02", "d03"),
                            min_cases=args.min_cases)
    sanity_worst = _worst_ok_error(sanity)

    # 3a. STUB FAIL: perturb T/MU/ISLTYP above tol -> those fields + cases FAIL.
    print("[S4] running STUB-FAIL campaign (T+5K, MU+200Pa, ISLTYP+1)...")
    fail_perturb = {"T": 5.0, "MU": 200.0, "ISLTYP": 1.0}
    stub_fail = C.run_campaign(_make_perturbed_factory(fail_perturb), cases=cases,
                               domains=("d01", "d02", "d03"),
                               min_cases=args.min_cases)

    # 3b. STUB PASS: sub-tol perturbation -> still PASS.
    print("[S4] running STUB-PASS campaign (T+0.05K, sub-tol)...")
    pass_perturb = {"T": 0.05}
    stub_pass = C.run_campaign(_make_perturbed_factory(pass_perturb), cases=cases,
                               domains=("d01", "d02", "d03"),
                               min_cases=args.min_cases)

    # 4. FORECAST-GATE scaffold (no GPU).
    forecast_plan = C.run_forecast_gate(execute=False)

    # --- assemble the verdict ---
    sanity_ok = bool(
        sanity["campaign_pass"]
        and sanity_worst["worst_ok_rmse"] < 1e-6
        and sanity_worst["worst_ok_maxabs"] < 1e-5
    )
    # FAIL campaign must NOT pass, and must fail the perturbed fields specifically.
    ff_summary = stub_fail["field_failure_summary"]
    t_failed = ff_summary.get("T", {}).get("n_fail", 0) > 0
    mu_failed = ff_summary.get("MU", {}).get("n_fail", 0) > 0
    isltyp_failed = ff_summary.get("ISLTYP", {}).get("n_fail", 0) > 0
    fail_ok = bool(
        (not stub_fail["campaign_pass"]) and t_failed and mu_failed and isltyp_failed
    )
    pass_ok = bool(stub_pass["campaign_pass"])

    mechanics_proven = bool(sanity_ok and fail_ok and pass_ok)

    report = {
        "schema": "v0.4.0-S4-selftest-report-2026-06-02",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "comparator_schema": C.COMPARATOR_SCHEMA_VERSION,
        "real_init_types_version": REAL_INIT_TYPES_VERSION,
        "git": {
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
            "head_sha": _git(["rev-parse", "HEAD"]),
        },
        "oracle_root": args.root,
        "n_cases": len(cases),
        "min_cases_required": args.min_cases,
        "domains": ["d01", "d02", "d03"],
        "case_ids": [c.case_id for c in cases],
        "resource_policy": {"cores": "0-3", "platform": "cpu", "gpu": "not used"},

        # the headline ungameability verdict
        "verdict": {
            "mechanics_proven": mechanics_proven,
            "sanity_is_zero_error": sanity_ok,
            "fail_mechanics_works": fail_ok,
            "pass_mechanics_works": pass_ok,
        },

        # 1. sanity (real.exe vs itself ~0)
        "sanity_oracle_vs_itself": {
            "campaign_pass": sanity["campaign_pass"],
            "n_cases": sanity["n_cases"],
            "wrfinput_all_pass": sanity["wrfinput_all_pass"],
            "wrfbdy_all_pass": sanity["wrfbdy_all_pass"],
            "worst_ok_rmse_over_all_fields": sanity_worst["worst_ok_rmse"],
            "worst_ok_maxabs_over_all_fields": sanity_worst["worst_ok_maxabs"],
            "field_failure_summary": sanity["field_failure_summary"],
        },

        # 2. tolerance tables wired (read from frozen types.py)
        "frozen_tolerances": {
            "wrfinput": {k: list(v) for k, v in WRFINPUT_TOLS.items()},
            "wrfbdy": {k: list(v) for k, v in WRFBDY_TOLS.items()},
        },

        # 3. stub-candidate PASS/FAIL mechanics
        "stub_fail_candidate": {
            "perturbation": fail_perturb,
            "campaign_pass": stub_fail["campaign_pass"],
            "T_failed": t_failed,
            "MU_failed": mu_failed,
            "ISLTYP_failed": isltyp_failed,
            "field_failure_summary": ff_summary,
        },
        "stub_pass_candidate": {
            "perturbation": pass_perturb,
            "campaign_pass": stub_pass["campaign_pass"],
            "T_worst_rmse": stub_pass["field_failure_summary"].get("T", {}).get("worst_rmse"),
        },

        # 4. forecast-gate scaffold (no GPU run)
        "forecast_gate_scaffold": forecast_plan,

        # harness-side notes for the manager (types.py is frozen)
        "harness_notes": {
            "ALT": ("WRFINPUT_TOLS has an ALT key but real.exe does NOT write ALT "
                    "to wrfinput (only AL+ALB+T_INIT+P_HYD). Comparator records ALT "
                    "as NOT_IN_WRFINPUT (non-blocking). Manager: consider dropping "
                    "the ALT key from WRFINPUT_TOLS on the next types.py bump, OR "
                    "the driver may emit ALT into a debug wrfinput. AL is checked."),
            "wrfbdy_coupled": ("oracle wrfbdy stores COUPLED values "
                               "(field*(mu+mub)/msf) despite the NetCDF units attr "
                               "reading 'm s-1' etc.; magnitudes confirm coupling "
                               "(U_BXS ~1e6). The frozen WRFBDY_TOLS are scaled for "
                               "coupled values, so the native LateralBC.values MUST "
                               "carry coupled quantities (per types.py docstring)."),
            "wrfbdy_frames": ("Canary wrfbdy holds 4 Time frames; the comparator "
                              "scores the FIRST frame (t0 value / first-interval "
                              "tendency). A future multi-interval gate could score "
                              "all frames."),
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    print("\n[S4] ===== SELF-TEST VERDICT =====")
    print(f"  cases scored:            {len(cases)} (>= {args.min_cases} required)")
    print(f"  SANITY (oracle vs self): pass={sanity['campaign_pass']} "
          f"worst_rmse={sanity_worst['worst_ok_rmse']:.3e} "
          f"worst_maxabs={sanity_worst['worst_ok_maxabs']:.3e}")
    print(f"  STUB-FAIL campaign_pass: {stub_fail['campaign_pass']} "
          f"(T={t_failed} MU={mu_failed} ISLTYP={isltyp_failed} all failed as expected)")
    print(f"  STUB-PASS campaign_pass: {stub_pass['campaign_pass']} (sub-tol T offset)")
    print(f"  MECHANICS PROVEN:        {mechanics_proven}")
    print(f"  -> {args.out}")
    return 0 if mechanics_proven else 2


if __name__ == "__main__":
    raise SystemExit(main())
