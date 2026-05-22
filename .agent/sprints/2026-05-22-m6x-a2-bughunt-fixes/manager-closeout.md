# M6.x-A2 Manager Closeout — Bug-Hunt Fix-Hint: FAIL → c1 Invoked

**Sprint**: M6.x-A2 Bug-Hunt Fix-Hint Application
**Status**: **CLOSED — Worker HONEST FAIL; c1 Klemp-Skamarock dispatched**
**Date**: 2026-05-22 ~01:05
**Worker**: codex gpt-5.5 xhigh (~30 min)

## Headline

All 3 bug-hunt fixes applied + verified with prescribed tests. **6h coupled probe failure progression**:

| Stage | sanitize firing rate | result |
|---|---|---|
| A1 baseline (PRESSURE_IMPLICIT_RELAXATION=0.05) | 76.0% | FAIL |
| A2 + FIX #1 (PH gravity factor) | 77.4% | FAIL (no change) |
| A2 + FIX #2 (remove 0.05 + asymmetric mask, alpha uncapped) | **100%** | **WORSE** |
| A2 + FIX #2b (alpha capped at 5.0) | 99.95% | WORSE |
| A2 + FIX #3 (mu inside acoustic scan) | 99.95% | NO RECOVERY |

## Critical signal

**Removing the heuristic `PRESSURE_IMPLICIT_RELAXATION=0.05` damping made the system MORE unstable, not less.** This means:
- The 0.05 damping wasn't the bug — it was MASKING a deeper acoustic-mode instability
- Bug-hunt #1's main hypothesis (asymmetric damping was the problem) was WRONG
- The PH gravity factor was a real dimensional error but apparently not dominant
- Current `_vertical_implicit_weight` heuristic damping cannot resolve the underlying acoustic-mode instability

## Manager decision (immediate)

**Two parallel dispatches**:

1. **c1 Klemp-Skamarock clean-room codex** (window 0:9, 5-9 day wall)
   - Replaces `_vertical_implicit_weight` heuristic with proper per-column tridiagonal Thomas solve per Klemp 2007 §3a-c
   - This is the planned contingency fallback per design doc + plan critic kill-gate rule
   - LOW architectural risk; preserves M4 invariants

2. **Bug-hunt #2 opus with re-framing** (window 0:10, 30-60 min wall)
   - Fresh angle: what did bug-hunt #1 MISS, given that removing damping made things worse?
   - Focus operators bug-hunt #1 skipped: `_grad_x_to_u`, `_grad_y_to_v`, `_mass_to_u_face`, `boundary_apply.py`, `physics_couplers` timing
   - Could short-circuit c1's 5-9 day grind if a cheaper fix is found

**Per user directive `[[feedback_parallel_bug_angles_and_plan_critique]]`**: Gemini orthogonal third opinion attempted via `agy` CLI — **OAuth expired** (auth URL response). Will retry on next user check-in.

## Branch state

- `worker/codex/m6x-wrf-canonical-dycore` at `676ba40` (A1+A2 WIP + worker reports) — **NOT merged to main**
- Probe artifacts on disk: `m6x_a2_fix1_6h_direct_probe.json` (FIX#1), `m6x_a2_fix2_6h_direct_probe.json` (FIX#2), `m6x_a2_fix2b_6h_direct_probe.json` (FIX#2b), `m6x_a2_fix3_6h_direct_probe.json` (FIX#3) — all in `/tmp/wrf_gpu2_m6x/artifacts/m6/performance/`

## Strategic position

- M6.x branch held; broken code stays on branch as record-of-attempt
- M7-S0 + M6-S8 still BLOCKED on M6.x close
- c1 codex grinding; bug-hunt #2 opus searching for cheaper fix
- Parallel angles maxed: 1 codex + 1 opus + (Gemini if OAuth recovered)
- If both c1 and bug-hunt #2 fail to make progress within 24h → escalate to user for architectural pivot (c2 semi-implicit, c3 ML-emulator, or end-goal re-scope)

## Honest accounting

The "verifiability triple" worked — bug-hunt #1's verification tests PASSED for FIX #1 and #2, but the COUPLED probe still failed. This is the limit of single-column unit testing: dimensional correctness verified, but multi-column coupling instability not catchable in isolation. Going forward, bug-hunt #2 + c1 worker both need a way to test coupled-mode stability cheaply (which is why c1's "shrink the problem" approach via Klemp 2007 reference equations should be more robust by construction).

— Manager (Claude Opus 4.7 1M-context), 2026-05-22 01:05
