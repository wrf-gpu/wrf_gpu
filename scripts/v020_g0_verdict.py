#!/usr/bin/env python3
"""v020_g0_verdict.py — emit the G0 fp32-vs-fp64 GO / KILL / REDIRECT verdict.

Implements **Gate 0** of `proofs/v020/fp32_proto/S4_S7_PRODUCTION_PLAN.md` (the
v2 production contract, D-critique items B2a/B2b/B2c) as HARDENED over the older
roadmap §8.4 split. It is a PURE reducer over the A/B run manifests the G0 shell
driver produces (warm ms/step, peak VRAM incl. transient, the fp32-able transient
split, dominant-kernel walls, did-it-fit, profiler classification flags). NO GPU,
NO model — it only reads JSON the driver wrote, so it is fully CPU-dry-runnable on
a synthetic manifest (the driver's --emit-template writes one).

WHY this is not a plain "speedup >= 2 ? GO : KILL" reducer (the v2 contract,
Gate 0 / D-critique B2):

  B2a — TWO fp32 arms, decide on PRODUCTION.
        G0 runs both the AGGRESSIVE-ceiling arm (fp32 storage for p'/ph'/mu'/w
        AND an fp32 tridiagonal/PCR solve) and the PRODUCTION arm (fp32 storage
        but cancellation brackets + tridiagonal solve fp64 in registers/scratch).
        The GO/KILL/REDIRECT decision uses the *production-config* number only;
        the aggressive ceiling is reported as CONTEXT and never decides.

  B2b — Speed-KILL requires PROFILER classification, not a wall ratio.
        G0 measures fp32 storage on the UNFUSED topology, so its speed is a lower
        bound that excludes the S7 fusion/registerization multiplier.
          * real kill: fp32 did NOT shrink live/transient storage AND the dominant
            fp32/fp64 kernel walls are identical (the "1.106x with identical
            kernel wall" signature). Both must hold.
          * NOT a kill: fp32 DID shrink storage/transient but the wall is capped
            by launch count + HBM-resident fp64 islands — i.e. the cost is in
            exactly what S7 attacks. This is GO-to-S7, NEVER a kill.
        Never KILL the fp32 path on an unfused island-resident measurement.

  B2c — "bottleneck moved" is a Speed-GO only if the new bottleneck is in scope.
        A profiler-confirmed moved bottleneck is a valid Speed-GO ONLY if the new
        dominant component is addressed by S4/S7 or by a named, scheduled
        follow-on. If it moved to a component outside the funded scope with no
        scheduled fix, the outcome is REDIRECT (P3/P4/multi-GPU/that component's
        workstream), not Speed-GO.

Decision inputs (per S4_S7_PRODUCTION_PLAN Gate 0 + D-critique B2):
  fp64: { ms_per_step, peak_vram_mib, fit (bool), dominant_kernel_ms }
  fp32_production: {                          # <- the DECIDING arm
        ms_per_step, peak_vram_mib, fit (bool), dominant_kernel_ms,
        transient_fp32_fraction (0..1),       # share of transient fp32-able
        storage_shrank (bool|None),           # B2b: did live/transient storage drop?
        walls_identical (bool|None),          # B2b: dominant fp32==fp64 wall (1.106x sig)
        wall_capped_by_launch_or_island (bool|None),  # B2b: the S7 target condition
        bottleneck_moved (bool),              # B2c: profiler shows a NAMED non-dycore dominant
        bottleneck_in_scope (bool|None),      # B2c: that component is in S4/S7 or a named follow-on
        bottleneck_component (str)            # B2c: name it (for the report)
  }
  fp32_aggressive: { ms_per_step, peak_vram_mib, dominant_kernel_ms, ... }  # CONTEXT ONLY
  grid: { cols, name, fp64_oom_target (bool: is there a useful grid fp64 OOMs on?) }

Back-compat: a legacy single `fp32` block (the Wave-1 probe manifest schema) is
accepted as the PRODUCTION arm with a recorded caveat; the older `bottleneck_moved`
semantics still resolve correctly through the B2c gate (an unqualified moved
bottleneck -> REDIRECT, never a silent Speed-GO).

Usage:
  python v020_g0_verdict.py --manifest g0_manifest.json [--out verdict.json]
  python v020_g0_verdict.py --emit-template g0_manifest.template.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DOM_FASTER_FRAC = 0.90       # fp32 dominant kernel must be <90% of fp64's to count
WALL_IDENTICAL_FRAC = 0.95   # |fp32 wall / fp64 wall| within 5% == "identical walls" (1.106x sig)
TRANSIENT_MAJORITY = 0.50    # >=50% of transient fp32-able
SPEED_GO_HARD = 2.0          # clean >=2x
SPEED_GO_SOFT = 1.5          # >=1.5x but only WITH a real dominant-kernel drop


def _f(v) -> float:
    """Coerce to float; None/missing/unparseable -> NaN (the conservative sentinel)."""
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _tribool(v):
    """Three-valued: True / False / None(unknown). Unknown stays conservative."""
    if v is None:
        return None
    return bool(v)


def _production_arm(m: dict) -> tuple[dict, bool]:
    """Return (production fp32 arm, legacy_fallback_used).

    B2a: the GO decision uses the production-config arm. Prefer the explicit
    `fp32_production` block; fall back to the legacy single `fp32` block (the
    Wave-1 probe schema) and record that the production/aggressive split was not
    provided.
    """
    prod = m.get("fp32_production")
    if isinstance(prod, dict) and prod:
        return prod, False
    legacy = m.get("fp32")
    if isinstance(legacy, dict) and legacy:
        return legacy, True
    return {}, True


def decide(m: dict) -> dict:
    fp64 = m.get("fp64") or {}
    grid = m.get("grid") or {}
    fp32, legacy_fallback = _production_arm(m)           # B2a: PRODUCTION decides
    fp32_aggr = m.get("fp32_aggressive") or {}           # B2a: CONTEXT ONLY

    ms64 = _f(fp64.get("ms_per_step"))
    ms32 = _f(fp32.get("ms_per_step"))
    speedup = (ms64 / ms32) if (ms32 == ms32 and ms32 > 0) else float("nan")

    # Aggressive-ceiling speedup — reported for context, NEVER used in the gate.
    ms32_aggr = _f(fp32_aggr.get("ms_per_step"))
    speedup_aggressive = (ms64 / ms32_aggr) if (ms32_aggr == ms32_aggr and ms32_aggr > 0) else float("nan")

    v64 = _f(fp64.get("peak_vram_mib"))
    v32 = _f(fp32.get("peak_vram_mib"))
    vram_lower = (v32 == v32 and v64 == v64 and v32 < v64)

    dk64 = _f(fp64.get("dominant_kernel_ms"))
    dk32 = _f(fp32.get("dominant_kernel_ms"))
    dom_faster = (dk32 == dk32 and dk64 == dk64 and dk64 > 0 and dk32 < DOM_FASTER_FRAC * dk64)

    tfrac = _f(fp32.get("transient_fp32_fraction"))
    transient_majority = (tfrac == tfrac and tfrac >= TRANSIENT_MAJORITY)

    # --- B2b profiler-classification inputs ---------------------------------
    # storage_shrank: did fp32 reduce live/transient storage at all?
    #   If not provided, derive a conservative proxy from VRAM (lower VRAM => some
    #   storage shrank). The explicit flag wins.
    storage_shrank = _tribool(fp32.get("storage_shrank"))
    if storage_shrank is None:
        storage_shrank = vram_lower or None  # proxy; None if VRAM also unknown/equal->treat below

    # walls_identical: dominant fp32 and fp64 kernel walls match (1.106x signature).
    #   Derive from the wall ratio when not given.
    walls_identical = _tribool(fp32.get("walls_identical"))
    if walls_identical is None and dk32 == dk32 and dk64 == dk64 and dk64 > 0:
        ratio = dk32 / dk64
        walls_identical = (WALL_IDENTICAL_FRAC <= ratio <= (1.0 / WALL_IDENTICAL_FRAC))

    # wall_capped_by_launch_or_island: the S7 target condition (GO-to-S7).
    # The EXPLICIT profiler flag is authoritative. When absent, derive it only
    # from a STRONG signature: storage shrank AND fp32 engaged the hot path
    # (transient majority fp32-able) AND the dominant kernel did NOT get faster
    # -> the wall is being held by launch count / HBM-resident fp64 islands,
    # which is exactly what S7 attacks. The transient-majority requirement keeps
    # a pure CAPABILITY win (fp32 only buys grid-fit, hot path largely fp64) from
    # being misclassified as an S7 speed target.
    wall_capped = _tribool(fp32.get("wall_capped_by_launch_or_island"))
    if wall_capped is None:
        if storage_shrank and transient_majority and not dom_faster:
            wall_capped = True

    # --- B2c bottleneck-moved gating ----------------------------------------
    bottleneck_moved = bool(fp32.get("bottleneck_moved", False))
    bottleneck_in_scope = _tribool(fp32.get("bottleneck_in_scope"))
    bottleneck_component = fp32.get("bottleneck_component", "")
    # A moved bottleneck only counts toward Speed-GO if it is addressed by S4/S7
    # or a named scheduled follow-on. Unknown (None) is NOT in-scope (conservative).
    bottleneck_moved_in_scope = bool(bottleneck_moved and bottleneck_in_scope is True)
    bottleneck_moved_out_of_scope = bool(bottleneck_moved and not bottleneck_moved_in_scope)

    fp32_fit = bool(fp32.get("fit", False))
    fp64_oom_target = bool(grid.get("fp64_oom_target", False))

    # ---- Speed gate (production arm only; B2a) ----
    # ">2x trajectory" = a clean >=2x, OR a clear >=1.5x WITH the dominant kernel
    # actually faster, OR an IN-SCOPE moved bottleneck (B2c). An out-of-scope
    # moved bottleneck does NOT satisfy the trajectory (it routes to REDIRECT).
    on_2x_trajectory = (
        (speedup == speedup and speedup >= SPEED_GO_HARD)
        or (dom_faster and speedup == speedup and speedup >= SPEED_GO_SOFT)
        or bottleneck_moved_in_scope
    )
    speed_go = bool(transient_majority and vram_lower and on_2x_trajectory)

    # ---- B2b real-KILL classification (gate, not a wall ratio) ----
    # Real kill ONLY when fp32 did NOT shrink storage AND walls are identical
    # (both must hold). Anything where storage DID shrink but the wall is
    # launch/island-capped is the S7 target condition -> GO-to-S7, never KILL.
    storage_did_not_shrink = (storage_shrank is False)
    real_kill = bool(storage_did_not_shrink and walls_identical is True)

    # GO-to-S7: fp32 helped storage/transient but the wall is launch/island-capped
    # (the unfused-topology lower bound) — i.e. the cost is in exactly what S7
    # attacks. This is a SPEED-track GO whose multiplier is pending S7, so it
    # requires evidence that fp32 actually engaged the hot path (transient
    # majority fp32-able) AND that storage shrank, distinguishing it from a pure
    # CAPABILITY win (fp32 only buys grid-fit, with no S7 speed lever). The
    # explicit launch/island-cap flag is sufficient on its own; otherwise we
    # derive it from a storage shrink + a majority-fp32 transient + a dominant
    # kernel that did NOT get faster (held by launch/island, the S7 target).
    go_to_s7 = bool(
        not speed_go
        and not real_kill
        and (storage_shrank is True or vram_lower)
        and (
            wall_capped is True
            or (transient_majority and not dom_faster)
        )
    )

    # ---- Capability gate ----
    capability_go = bool(fp64_oom_target and fp32_fit and vram_lower)

    # ---- REDIRECT (B2c): bottleneck moved out of scope, with no other GO path ----
    # If the only thing pushing toward a GO was an out-of-scope moved bottleneck,
    # the honest outcome is REDIRECT, not Speed-GO and not KILL (unless the S7
    # target condition or capability independently clears).
    redirect = bool(
        bottleneck_moved_out_of_scope
        and not speed_go
        and not go_to_s7
        and not capability_go
        and not real_kill
    )

    # ---- Resolve the verdict in priority order ----
    # Speed-GO (best) > GO-to-S7 (storage win, S7 multiplier pending) >
    # Capability-GO > REDIRECT (bottleneck out of scope) > real-KILL (last resort,
    # only on a profiler-classified real kill).
    if speed_go:
        verdict = "G0-SPEED-GO"
        action = (
            "FUND the fp32 speed+capability rewrite (S4-S8, production config). "
            "Decided on the PRODUCTION arm (fp64-in-register solve): transient is "
            "majority fp32-able, VRAM nets lower, and warm ms/step is on a >=2x "
            "trajectory"
            + (" (profiler moved the bottleneck to an IN-SCOPE component)." if bottleneck_moved_in_scope
               else ".")
        )
    elif go_to_s7:
        verdict = "G0-GO-TO-S7"
        action = (
            "GO — but the win is gated on S7. fp32 (PRODUCTION arm) DID shrink "
            "storage/transient; the wall is capped by launch count + HBM-resident "
            "fp64 islands (alt/al'/php), which is exactly the S7 target condition. "
            "This is NOT an fp32 KILL (B2b): never kill on an unfused, "
            "island-resident measurement. Proceed to S4 storage + S7 "
            "fusion/registerization; re-measure the wall only after S7."
        )
    elif capability_go:
        verdict = "G0-CAPABILITY-GO"
        action = (
            "FUND the NARROWER capability rewrite: fp32 stably fits a useful grid "
            "that fp64 OOMs on, with VRAM proof. The speed gate did NOT clear on "
            "the production arm -> REDIRECT the speed budget to P3/P4/multi-GPU "
            "but KEEP the capability rewrite live. Primary deliverable: 'fp64 "
            "can't run this useful grid; fp32 can, stably'."
        )
    elif redirect:
        verdict = "G0-REDIRECT"
        action = (
            "REDIRECT (B2c). The profiler shows the bottleneck moved to "
            f"'{bottleneck_component or 'an unnamed component'}', which is OUTSIDE "
            "the funded S4/S7 scope with no named scheduled follow-on. Funding "
            "S4/S7 would leave the new bottleneck untouched (the ~1.1x trap). "
            "Route the speed budget to P3/P4/multi-GPU or that component's "
            "workstream; this is NOT a Speed-GO and NOT an fp32 KILL."
        )
    elif real_kill:
        verdict = "G0-KILL"
        action = (
            "KILL the fp32 rewrite as a speed lever — profiler-classified REAL "
            "kill (B2b): fp32 did NOT shrink live/transient storage AND the "
            "dominant fp32/fp64 kernel walls are identical (the 1.106x signature). "
            "Write the EXACT-cause honest negative and redirect to structural "
            "P3/P4 + multi-GPU/tiling. Do NOT report a vague 'fp32 didn't help'."
        )
    else:
        # No gate cleared and we could NOT profiler-classify a real kill. Per B2b
        # we must NOT KILL on an unclassified/unfused measurement. The conservative
        # honest outcome is to withhold the GO and demand the missing profiler
        # classification before any KILL.
        verdict = "G0-INCONCLUSIVE"
        action = (
            "INCONCLUSIVE — no gate cleared, and the profiler did NOT classify a "
            "REAL kill (B2b forbids killing on an unfused/island-resident or "
            "unclassified measurement). Required to resolve: the storage-shrank "
            "flag, the walls-identical (1.106x) classification, and the "
            "launch/island-cap finding. Re-run G0 with the profiler fields "
            "populated; do NOT KILL fp32 on this evidence."
        )

    return {
        "verdict": verdict,
        "action": action,
        "decided_on": "fp32_production" + (" (legacy single-arm fallback)" if legacy_fallback else ""),
        "metrics": {
            "ms_per_step_fp64": ms64,
            "ms_per_step_fp32_production": ms32,
            "warm_speedup_x_production": speedup,
            "ms_per_step_fp32_aggressive": ms32_aggr,
            "warm_speedup_x_aggressive_CONTEXT_ONLY": speedup_aggressive,
            "peak_vram_mib_fp64": v64,
            "peak_vram_mib_fp32_production": v32,
            "vram_lower": vram_lower,
            "dominant_kernel_ms_fp64": dk64,
            "dominant_kernel_ms_fp32_production": dk32,
            "dominant_kernel_faster": dom_faster,
            "transient_fp32_fraction": tfrac,
            "transient_majority_fp32_able": transient_majority,
            "on_2x_trajectory": on_2x_trajectory,
            "fp32_fit": fp32_fit,
            "fp64_oom_target_exists": fp64_oom_target,
        },
        "b2b_classification": {
            "storage_shrank": storage_shrank,
            "walls_identical_1106_signature": walls_identical,
            "wall_capped_by_launch_or_island": wall_capped,
            "real_kill": real_kill,
            "go_to_s7_target_condition": go_to_s7,
        },
        "b2c_bottleneck": {
            "bottleneck_moved": bottleneck_moved,
            "bottleneck_component": bottleneck_component,
            "bottleneck_in_scope": bottleneck_in_scope,
            "moved_in_scope_counts_as_speed_go": bottleneck_moved_in_scope,
            "moved_out_of_scope_redirect": bottleneck_moved_out_of_scope,
        },
        "gates": {
            "G0_SPEED_GO": speed_go,
            "G0_GO_TO_S7": go_to_s7,
            "G0_CAPABILITY_GO": capability_go,
            "G0_REDIRECT": redirect,
            "G0_KILL": real_kill,
            "G0_INCONCLUSIVE": verdict == "G0-INCONCLUSIVE",
        },
        "criteria_ref": "S4_S7_PRODUCTION_PLAN.md Gate 0 (v2 contract; D-critique B2a/B2b/B2c)",
        "b2a_note": (
            "GO decided on the PRODUCTION arm (fp64-in-register solve); the "
            "aggressive-ceiling fp32-solve number is CONTEXT ONLY and never decides."
        ),
        "grid": grid,
        "notes": m.get("notes", {}),
    }


def template() -> dict:
    return {
        "_doc": "G0 manifest (v2 contract): filled by scripts/v020_g0_decision.sh from the "
                "A/B runs. Values below are PLACEHOLDERS (CPU-template only, NOT measured). "
                "B2a: provide BOTH fp32_production (decides) and fp32_aggressive (context).",
        "grid": {"name": "v017_bigswiss_460x460x44", "cols": 211600,
                 "fp64_oom_target": True,
                 "_fp64_oom_target_note": "True if a useful 1km/large grid exists that fp64 "
                                          "OOMs on but fp32 fits (the capability prize)."},
        "fp64": {"ms_per_step": None, "peak_vram_mib": None, "fit": True,
                 "dominant_kernel_ms": None},
        "fp32_production": {
            "_note": "DECIDING arm (B2a): fp32 storage for p'/ph'/mu'/w with the "
                     "cancellation brackets + tridiagonal solve fp64 in registers/scratch.",
            "ms_per_step": None, "peak_vram_mib": None, "fit": None,
            "dominant_kernel_ms": None,
            "transient_fp32_fraction": None,
            "storage_shrank": None,
            "walls_identical": None,
            "wall_capped_by_launch_or_island": None,
            "bottleneck_moved": False,
            "bottleneck_in_scope": None,
            "bottleneck_component": ""},
        "fp32_aggressive": {
            "_note": "CONTEXT ONLY (B2a): fp32 storage AND fp32 tridiagonal/PCR solve. "
                     "Reported as the ceiling; NEVER decides the verdict.",
            "ms_per_step": None, "peak_vram_mib": None,
            "dominant_kernel_ms": None},
        "notes": {"how_fp32_arm_is_driven": "JAX_ENABLE_X64=false (global fp32, incl. the "
                  "PCR solve) + GPUWRF_THOMPSON_FP32=1 == the aggressive downcast proxy; the "
                  "production arm keeps brackets/solve fp64-in-register. NO production source edit."},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--emit-template", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.emit_template:
        args.emit_template.write_text(json.dumps(template(), indent=2) + "\n")
        print(f"wrote template {args.emit_template}")
        return 0

    if not args.manifest or not args.manifest.is_file():
        print("v020_g0_verdict: --manifest FILE required (or --emit-template)", file=sys.stderr)
        return 2
    m = json.loads(args.manifest.read_text())
    res = decide(m)
    payload = json.dumps(res, indent=2) + "\n"
    if args.out:
        args.out.write_text(payload)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(payload)

    print(f"\n=== G0 VERDICT: {res['verdict']} ===", file=sys.stderr)
    mm = res["metrics"]
    print(f"  decided_on={res['decided_on']}", file=sys.stderr)
    print(f"  production speedup x={mm['warm_speedup_x_production']}  "
          f"(aggressive ctx x={mm['warm_speedup_x_aggressive_CONTEXT_ONLY']})  "
          f"vram_lower={mm['vram_lower']}  "
          f"transient_majority_fp32={mm['transient_majority_fp32_able']}  "
          f"dom_kernel_faster={mm['dominant_kernel_faster']}", file=sys.stderr)
    b = res["b2b_classification"]
    print(f"  B2b: storage_shrank={b['storage_shrank']} walls_identical={b['walls_identical_1106_signature']} "
          f"wall_capped={b['wall_capped_by_launch_or_island']} real_kill={b['real_kill']} "
          f"go_to_s7={b['go_to_s7_target_condition']}", file=sys.stderr)
    c = res["b2c_bottleneck"]
    print(f"  B2c: moved={c['bottleneck_moved']} in_scope={c['bottleneck_in_scope']} "
          f"redirect={c['moved_out_of_scope_redirect']}", file=sys.stderr)
    print(f"  capability: fp32_fit={mm['fp32_fit']} fp64_oom_target={mm['fp64_oom_target_exists']}",
          file=sys.stderr)
    print(f"  ACTION: {res['action']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
