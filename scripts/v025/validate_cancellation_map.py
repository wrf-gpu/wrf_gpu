#!/usr/bin/env python3
"""Validate the cancellation map: 100% inventory, minimal evidence-linked islands (§10)."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _validate_lib import Report, load_json

VERDICT_CLASSES = {"SAFE_FP32", "COMPENSATED_OR_REFORMULATED", "FP64_ISLAND_CANDIDATE"}
# Statuses that explicitly decline to issue a verdict. They are valid entries --
# an operator that returned all zeros, re-normalised its own dtypes, or failed to
# trace under fp32 has NOT earned a SAFE_FP32 and must not silently receive one.
NON_VERDICT = {"NOT_EXERCISED_ON_THIS_STATE", "INCONCLUSIVE_DTYPE_NORMALISED",
               "FP32_TRACE_FAILS"}
CLASSES = VERDICT_CLASSES | NON_VERDICT
PER_ENTRY = ("input_dtype", "output_dtype", "fp64_result", "fp32_result",
             "max_rel_error", "median_rel_error", "p99_rel_error", "ulps",
             "classification", "source_algebra_identity")

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path)
    ap.add_argument("--require-complete", action="store_true")
    args = ap.parse_args()
    rep = Report("cancellation_map", args.path)
    obj = load_json(args.path, rep)
    if obj is None:
        return rep.emit()

    entries = obj.get("operators") or []
    if not entries:
        rep.missing("operators", "§10 expects the full active-path inventory (order of 30)")
        return rep.emit()

    rep.require("state_source_is_real",
                obj.get("state_source") == "20260725_18z_production_snapshot",
                f"state_source={obj.get('state_source')!r}; §10 forbids synthetic arrays",
                pass_detail="state_source is the real 20260725_18z production snapshot")

    incomplete, bad_class, inactive_unproven, unearned_safe = [], [], [], []
    for e in entries:
        name = e.get("operator", "?")
        if e.get("active") is False:
            if not e.get("inactivity_proof"):
                inactive_unproven.append(name)
            continue
        klass = e.get("classification")
        if klass not in CLASSES:
            bad_class.append(name)
        if klass in NON_VERDICT:
            # A non-verdict entry cannot carry fp32 error statistics -- for
            # FP32_TRACE_FAILS there is no fp32 result to measure at all. It must
            # instead carry the reason AND the raw arm failure, so the gap is
            # documented rather than papered over with numbers from another arm.
            arms = e.get("arms") or {}
            aggressive = arms.get("fp32_aggressive") or {}
            documented = bool(e.get("classification_reason")) and (
                klass != "FP32_TRACE_FAILS" or bool(aggressive.get("error"))
            )
            if not documented:
                incomplete.append(name)
        elif any(e.get(f) is None for f in PER_ENTRY):
            incomplete.append(name)
        # An operator that did not fire, re-normalised its dtypes, or failed to
        # trace under fp32 cannot carry a verdict class. This is the check that
        # stops a zero measured error from being read as fp32 safety.
        if e.get("classification") in VERDICT_CLASSES and (
            e.get("not_exercised_on_this_state")
            or e.get("dtype_normalising_operator")
            or e.get("fp32_arm_inputs_fully_demoted") is False
        ):
            unearned_safe.append(name)

    rep.require("inventory:no_unearned_verdict", not unearned_safe,
                f"entries claiming a verdict without exercising fp32: {unearned_safe}",
                pass_detail=(
                    "every verdict-carrying entry exercised fp32: none was unexercised, "
                    "dtype-normalised, or incompletely demoted"
                ))

    if args.require_complete:
        rep.require("inventory:every_active_entry_complete", not incomplete,
                    f"incomplete: {incomplete[:8]}{'...' if len(incomplete) > 8 else ''}")
        rep.require("inventory:inactive_entries_have_executable_proof", not inactive_unproven,
                    f"inactive entries lacking an executable proof: {inactive_unproven}",
                    pass_detail="every inactive entry carries an executable inactivity proof")
        cov = obj.get("coverage_fraction")
        if cov is None:
            rep.missing("inventory:coverage_fraction", "100% coverage required")
        else:
            rep.require("inventory:coverage_100pct", cov == 1.0, f"{cov}")
    rep.require("inventory:classifications_valid", not bad_class, f"invalid: {bad_class}")

    islands = obj.get("proposed_fp64_islands")
    if islands is None:
        rep.missing("islands:proposed", "§10 requires a minimal proposed island list")
    else:
        unlinked = [i for i in islands if not i.get("evidence")]
        rep.require("islands:every_island_evidence_linked", not unlinked,
                    f"unlinked: {[i.get('site') for i in unlinked]}")
        rep.require("islands:not_keep_all_current",
                    obj.get("rationale") != "keep all current islands for safety",
                    "'keep all current islands for safety' explicitly fails §10",
                    pass_detail="the proposal states a measured rule, not blanket retention")
        # §10 minimality: the proposal is only credible if the map also examined
        # the islands that already exist and was willing to say some are not
        # earning their keep. A proposal that rubber-stamps every current site is
        # the "keep all current islands" answer wearing different words.
        audit = obj.get("existing_island_audit") or {}
        reached = audit.get("operators_reaching_a_live_island")
        if reached is None:
            rep.missing("islands:existing_audited",
                        "§10 requires the CURRENT island sites to be assessed against measurement")
        else:
            supported = set(audit.get("supported_by_measurement") or [])
            unsupported = set(audit.get("not_supported_by_measurement") or [])
            rep.require("islands:existing_sites_all_adjudicated",
                        set(reached) == supported | unsupported,
                        f"unadjudicated: {sorted(set(reached) - supported - unsupported)}",
                        pass_detail=(
                            f"all {len(reached)} live island site(s) adjudicated: "
                            f"{len(supported)} earned retention, {len(unsupported)} did not"
                        ))
            rep.require("islands:minimality_was_tested",
                        bool(reached),
                        "no current island site was measured, so minimality is unproven",
                        pass_detail=(
                            f"{len(reached)} current island site(s) were measured by "
                            f"neutralising force_fp64_island and re-running the fp32 arm; "
                            f"{len(unsupported)} did not earn retention"
                        ))
    return rep.emit()

if __name__ == "__main__":
    raise SystemExit(main())
