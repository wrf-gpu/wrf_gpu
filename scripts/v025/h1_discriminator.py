#!/usr/bin/env python3
"""H1's outcome rule (CPU-only arithmetic). Two-sided, after a manager correction.

H1 is "autotuning dominates the FAST-v025 cold compile".

    removable = (T_on - T_off) / T_on = 1 - T_off / T_on

where `T_on` is the cold compile with autotuning and `T_off` without it.

What was wrong with the previous version
----------------------------------------
It measured only `T_off` and combined it with `T_on >= 601.4 s` (the W1b early
stop), giving `removable >= 1 - T_off/601.4`. That bound is valid, and the
manager confirmed it -- but it is **one-sided**, and I used it in both
directions. Two branches were unsound:

* a **small lower bound** was read as "H1 NOT_SUPPORTED". It is not. With
  `T_off = 500 s` the bound is 16.9%, yet if the true `T_on` were 5000 s the true
  removable share would be 90%. A lower bound below 30% is silent about whether
  the true share is below 30%.
* an **autotune-OFF timeout** was read as "H1 REFUTED". It is not. It shows
  `T_off` is large, but `T_on` has no upper bound, so the ratio is unconstrained.

Both are now **INCONCLUSIVE**.

The general rule, stated once
-----------------------------
`removable` is *decreasing* in `T_off` and *increasing* in `T_on`. So:

* a **lower** bound on removable needs an **upper** bound on `T_off` (an exact
  measurement or a conservatively typed upper-bound observation) and a
  **lower** bound on `T_on` (the 601.4 s early stop);
* an **upper** bound on removable needs a **lower** bound on `T_off` (a timeout
  suffices) and an **upper** bound on `T_on` -- which only a **completed**
  autotune-ON compile can supply.

Consequence, stated plainly because it constrains what Step 1 can deliver: with
the cheap arm alone H1 can only ever be **supported**, never refuted. Refuting it
requires measuring `T_on` to completion, and that is a window-size decision for
the manager, not something this module can assume away.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

# The W1b early stop. A LOWER bound on T_on: the compile had not finished.
T_ON_LOWER_BOUND_S = 601.4
T_OFF_BUDGET_S = 1100.0

# THE CLOCK. 601.4 s is process-launch-to-kill: the runner started the child and
# killed it 601.4 s later, still compiling. Any T_off compared against it must be
# measured from the same origin, or the ratio is between two different quantities.
#
# The previous version compared it with an NVTX span -- first compile range start
# to last range end -- which silently discards everything before the first range:
# interpreter start, JAX import, backend init, input load. Discarding startup
# makes T_off smaller, which makes `1 - T_off/T_on` LARGER, i.e. it manufactures
# support for H1 out of a units error. The manager caught this on main `8d27c1b3`.
REQUIRED_BASIS = "process-launch-to-executable-readiness"
UPPER_BOUND_BASIS = "nsys-session-origin-to-last-compile-range-end"
T_ON_BASIS = "process-launch-to-kill"
MEASUREMENT_EXACT = "exact"
MEASUREMENT_UPPER_BOUND = "upper_bound"

# Pre-registered bands on the removable share.
BAND_SUPPORTED = 0.70
BAND_PARTIAL = 0.30

# Band edges in seconds, derived rather than typed in beside the rule.
T_OFF_SUPPORTED_MAX_S = (1.0 - BAND_SUPPORTED) * T_ON_LOWER_BOUND_S   # 180.42 s
T_OFF_PARTIAL_MAX_S = (1.0 - BAND_PARTIAL) * T_ON_LOWER_BOUND_S       # 420.98 s


def removable_lower_bound(t_off_upper_s: float, t_on_lower_bound_s: float) -> float:
    """Valid with an exact/upper-bounded `T_off` and a lower-bounded `T_on`."""
    if t_on_lower_bound_s <= 0:
        raise ValueError("T_on lower bound must be positive")
    return max(0.0, 1.0 - (t_off_upper_s / t_on_lower_bound_s))


def removable_upper_bound(t_off_lower_bound_s: float, t_on_exact_s: float) -> float:
    """Valid only with a lower bound on `T_off` and an EXACT (or upper-bounded) `T_on`."""
    if t_on_exact_s <= 0:
        raise ValueError("T_on must be positive")
    return min(1.0, max(0.0, 1.0 - (t_off_lower_bound_s / t_on_exact_s)))


def _band_from_lower_bound(bound: float) -> tuple[str, str] | None:
    """Positive claims only. Returns None when the bound cannot conclude."""
    if bound >= BAND_SUPPORTED:
        return "SUPPORTED", (
            f"at least {bound:.1%} of the cold compile is removable by disabling autotune "
            f"(>= the {BAND_SUPPORTED:.0%} dominance bar). Autotune search is the dominant "
            f"cold-compile cost."
        )
    if bound >= BAND_PARTIAL:
        return "PARTIAL_SUPPORT", (
            f"at least {bound:.1%} is removable -- material, but below the "
            f"{BAND_SUPPORTED:.0%} dominance bar. Autotune is one real lever among others."
        )
    return None


def verdict(
    t_off_s: float | None,
    *,
    completed: bool,
    t_off_basis: str | None = None,
    measurement_kind: str = MEASUREMENT_EXACT,
    t_on_s: float | None = None,
    t_on_lower_bound_s: float = T_ON_LOWER_BOUND_S,
    t_off_budget_s: float = T_OFF_BUDGET_S,
) -> dict[str, Any]:
    """Apply the pre-registered rule.

    Exact readiness uses :data:`REQUIRED_BASIS`.  The nsys-derived observation
    is accepted only as ``measurement_kind="upper_bound"`` on
    :data:`UPPER_BOUND_BASIS`: the profiler session begins before application
    launch, so its last compile-range end is conservative for H1 but ineligible
    for the product compile gate.

    `t_on_s` is the EXACT autotune-ON cold compile time, available only from a
    run that completed. Without it no negative verdict is reachable.
    """
    if measurement_kind == MEASUREMENT_EXACT:
        required_basis = REQUIRED_BASIS
    elif measurement_kind == MEASUREMENT_UPPER_BOUND:
        required_basis = UPPER_BOUND_BASIS
    else:
        return {
            "hypothesis": "H1",
            "verdict": "BLOCKED",
            "decisive": False,
            "measurement_kind": measurement_kind,
            "product_compile_gate_eligible": False,
            "reason": (
                f"unsupported T_off measurement kind {measurement_kind!r}; expected "
                f"{MEASUREMENT_EXACT!r} or {MEASUREMENT_UPPER_BOUND!r}"
            ),
        }

    if t_off_basis != required_basis:
        return {
            "hypothesis": "H1",
            "verdict": "BLOCKED",
            "decisive": False,
            "measurement_kind": measurement_kind,
            "product_compile_gate_eligible": False,
            "t_off_seconds": t_off_s,
            "t_off_basis": t_off_basis,
            "required_basis": required_basis,
            "t_on_basis": T_ON_BASIS,
            "reason": (
                f"T_off was measured on {t_off_basis!r} but T_on={t_on_lower_bound_s} s is "
                f"{T_ON_BASIS!r}. This {measurement_kind!r} observation must use "
                f"{required_basis!r} or H1 is BLOCKED. Dropping pre-range startup shrinks "
                "T_off and inflates the apparent removable share."
            ),
        }

    result: dict[str, Any] = {
        "hypothesis": "H1",
        "measurement_kind": measurement_kind,
        "product_compile_gate_eligible": measurement_kind == MEASUREMENT_EXACT,
        "exact_executable_readiness_identified": measurement_kind == MEASUREMENT_EXACT,
        "t_off_basis": t_off_basis,
        "t_on_basis": T_ON_BASIS,
        "t_off_seconds": t_off_s,
        "measurement_complete": completed,
        "t_on_seconds_exact": t_on_s,
        "t_on_lower_bound_seconds": t_on_lower_bound_s,
        "bands": {
            "SUPPORTED": f"removable >= {BAND_SUPPORTED:.0%}",
            "PARTIAL_SUPPORT": f"{BAND_PARTIAL:.0%} <= removable < {BAND_SUPPORTED:.0%}",
            "NOT_SUPPORTED": (f"removable < {BAND_PARTIAL:.0%} -- reachable ONLY with an "
                              f"exact T_on"),
        },
        "asymmetry": (
            "removable decreases in T_off and increases in T_on. A lower bound needs exact "
            "T_off plus a T_on lower bound; an upper bound needs a T_off lower bound plus an "
            "exact T_on. With the cheap arm alone H1 can only be supported, never refuted."
        ),
    }

    if measurement_kind == MEASUREMENT_UPPER_BOUND:
        result["asymmetry"] = (
            "an upper bound on T_off plus a lower bound on T_on yields a conservative "
            "lower bound on removable share. It can support H1 when large enough, but "
            "cannot refute H1 or supply exact product compile time."
        )
        result["product_compile_gate_ineligibility_reason"] = (
            "the origin is the nsys profiling session, not the application process launch, "
            "and no executable-ready boundary was mechanically identified"
        )
        if not completed or t_off_s is None:
            result.update({
                "verdict": "INCONCLUSIVE",
                "decisive": False,
                "reason": (
                    "the diagnostic upper-bound observation did not complete, so it supplies "
                    "neither an upper bound usable for a positive H1 claim nor an exact clock"
                ),
                "what_would_decide_it": (
                    "a completed nsys upper-bound observation, or an exact launch-to-ready "
                    "measurement from the forecast process"
                ),
            })
            return result
        lower = removable_lower_bound(float(t_off_s), t_on_lower_bound_s)
        result["t_off_upper_bound_seconds"] = float(t_off_s)
        result["removable_lower_bound"] = lower
        positive = _band_from_lower_bound(lower)
        if positive is not None:
            band, reason = positive
            result.update({
                "verdict": band,
                "decisive": True,
                "reason": (
                    f"{reason} T_off itself is a conservative upper bound, so this lower "
                    "bound remains valid for H1. It is still not a §5.3 compile measurement."
                ),
            })
            return result
        result.update({
            "verdict": "INCONCLUSIVE",
            "decisive": False,
            "reason": (
                f"the conservative removable lower bound is only {lower:.1%}. A small lower "
                "bound cannot refute H1, and this upper-bound clock cannot produce an exact "
                "product compile result."
            ),
            "what_would_decide_it": (
                "an exact T_on and exact launch-to-executable-readiness T_off in a separately "
                "qualified measurement"
            ),
        })
        return result

    # --- the cheap arm did not finish: T_off > budget, no upper bound on it -----
    if not completed:
        if t_on_s is None:
            result.update({
                "verdict": "INCONCLUSIVE",
                "decisive": False,
                "reason": (
                    f"the autotune-OFF arm exceeded its {t_off_budget_s:.0f} s budget, so T_off "
                    f"is only known to be > {t_off_budget_s:.0f} s, and T_on has no upper "
                    f"bound. The ratio is unconstrained in both directions. This is NOT a "
                    f"refutation of H1 -- an earlier version of this rule wrongly called it one."
                ),
                "what_would_decide_it": (
                    "an exact T_on from a completed autotune-ON cold compile, which with "
                    "T_off > budget yields an upper bound on the removable share."
                ),
            })
            return result
        bound = removable_upper_bound(t_off_budget_s, t_on_s)
        result["removable_upper_bound"] = bound
        if bound < BAND_PARTIAL:
            result.update({
                "verdict": "NOT_SUPPORTED",
                "decisive": True,
                "reason": (
                    f"T_off exceeded {t_off_budget_s:.0f} s while T_on measured "
                    f"{t_on_s:.1f} s exactly, so removable <= {bound:.1%}, below the "
                    f"{BAND_PARTIAL:.0%} bar. Disabling autotune does not make the compile "
                    f"cheap; the cost is structural."
                ),
            })
        else:
            result.update({
                "verdict": "INCONCLUSIVE",
                "decisive": False,
                "reason": (f"removable <= {bound:.1%}, which does not fall under the "
                           f"{BAND_PARTIAL:.0%} bar, and T_off has no upper bound to "
                           f"establish a positive claim."),
            })
        return result

    # --- the cheap arm finished: T_off exact --------------------------------
    if t_off_s is None:
        raise ValueError("a completed arm must report a time")

    lower = removable_lower_bound(t_off_s, t_on_lower_bound_s)
    result["removable_lower_bound"] = lower

    if t_on_s is not None:
        exact = 1.0 - (t_off_s / t_on_s)
        result["removable_exact"] = exact
        if exact >= BAND_SUPPORTED:
            band, reason = "SUPPORTED", f"removable is exactly {exact:.1%}"
        elif exact >= BAND_PARTIAL:
            band, reason = "PARTIAL_SUPPORT", f"removable is exactly {exact:.1%}"
        else:
            band, reason = "NOT_SUPPORTED", (
                f"removable is exactly {exact:.1%}, below the {BAND_PARTIAL:.0%} bar. Both "
                f"arms completed, so this is a real negative rather than a silent bound."
            )
        result.update({"verdict": band, "decisive": True, "reason": reason})
        return result

    positive = _band_from_lower_bound(lower)
    if positive is not None:
        band, reason = positive
        result.update({"verdict": band, "decisive": True, "reason": reason})
        return result

    result.update({
        "verdict": "INCONCLUSIVE",
        "decisive": False,
        "reason": (
            f"the provable lower bound is only {lower:.1%}, under the {BAND_PARTIAL:.0%} bar. "
            f"A lower bound below the bar does NOT show the true share is below it: T_on has no "
            f"upper bound, so a much larger true T_on would make the true share much larger. "
            f"An earlier version of this rule wrongly reported this case as NOT_SUPPORTED."
        ),
        "what_would_decide_it": (
            "an exact T_on from a completed autotune-ON cold compile. T_on is known only to "
            f"exceed {t_on_lower_bound_s:.1f} s, so the window must be sized for a compile "
            "whose upper tail has never been observed, and a second timeout would again be "
            "INCONCLUSIVE rather than negative."
        ),
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t-off-seconds", type=float, default=None)
    parser.add_argument("--t-on-seconds", type=float, default=None,
                        help="EXACT autotune-ON cold compile time; only a completed run has one")
    parser.add_argument("--t-off-basis", default=None,
                        help=f"clock T_off was measured on; must be {REQUIRED_BASIS!r}")
    parser.add_argument("--did-not-complete", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--print-rule", action="store_true",
                        help="print the pre-registered rule and exit; measures nothing")
    args = parser.parse_args()

    if args.print_rule:
        print(json.dumps({
            "t_on_lower_bound_seconds": T_ON_LOWER_BOUND_S,
            "t_off_budget_seconds": T_OFF_BUDGET_S,
            "reachable_with_the_cheap_arm_alone": {
                "SUPPORTED": f"T_off <= {T_OFF_SUPPORTED_MAX_S:.2f} s",
                "PARTIAL_SUPPORT": (f"{T_OFF_SUPPORTED_MAX_S:.2f} s < T_off <= "
                                    f"{T_OFF_PARTIAL_MAX_S:.2f} s"),
                "INCONCLUSIVE": (f"T_off > {T_OFF_PARTIAL_MAX_S:.2f} s, or the arm exceeds "
                                 f"its {T_OFF_BUDGET_S:.0f} s budget"),
            },
            "requires_an_exact_T_on": ["NOT_SUPPORTED"],
            "asymmetry": ("with the cheap arm alone H1 can only be supported, never refuted"),
        }, indent=2))
        return 0

    result = verdict(args.t_off_seconds, completed=not args.did_not_complete,
                     t_off_basis=args.t_off_basis, t_on_s=args.t_on_seconds)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
