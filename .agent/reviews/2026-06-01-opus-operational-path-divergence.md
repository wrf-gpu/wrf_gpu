# Operational per-hour vs continuous path divergence — ROOT CAUSE = load-bearing theta limiter (guards), NOT the per-hour reset and NOT the radiation clock

Date: 2026-06-01
Agent: Opus 4.8 MAX (worker/opus/final-verdict, main working tree)
Owned: `src/gpuwrf/runtime/operational_mode.py` (the guard/limiter), `scripts/diag/`.

## TL;DR (which-path-is-correct ANSWERED)

The v0.1.0 D02_VALIDATED proof's CONTINUOUS `_advance_chunk` harness is **CORRECT**:
its integrated lower-troposphere theta MATCHES CPU-WRF (theta level-0 bias ~-0.3 to
-0.4 K, slightly cool — good). The PRODUCTION per-hour `run_forecast_operational`
product is the one with the bug: it warm-drifts theta0 +1.0 -> +3.3 K over 6 h.

**The divergence is NOT the per-hour carry/step reset, and NOT the radiation diurnal
clock. It is ENTIRELY the operational theta-increment limiter (the `disable_guards
=False` guard path).** The single flag `disable_guards` accounts for the whole
+3.3 K T2 / lower-tropo warm drift. The validated harness sets `disable_guards=True`;
the production pipeline `_build_real_case` leaves it `False`. The guard's per-level
domain-min/max theta clamp + mass-conserving redistribution is LOAD-BEARING and pumps
warmth into the column every step — a direct violation of the project rule "guards
must not be load-bearing."

**The +2.6 kPa surface-pressure error is NOT the path divergence.** It is present at
~IDENTICAL magnitude on BOTH paths (CONT +2648 Pa, PERHOUR +2567 Pa at hour 1). It is
the SEPARATE dycore perturbation-geopotential equilibration issue already documented
for d03 (the free-drift ph' equilibrates ~2.6 kPa low). Fixing the path divergence
will NOT collapse the +2.6 kPa; that remains a distinct dycore item.

## The decisive experiment

`scripts/diag/d02_operational_path_divergence.py` advances the SAME d02 case3 IC
(20260521_18z, the validated anchor) 6 forecast hours under four configs, all on ONE
compiled `_advance_chunk` program. The per-hour path is emulated EXACTLY by re-seeding
the operational carry + restarting the step index at 1 each hour (radiation gating is
bit-identical: cadence 180, 360 steps/hr -> fires at the same global steps). Each
config isolates ONE difference between the two paths. State compared vs CPU-WRF corpus.

| config | reseed/hr | clock | guards | theta0 bias hr1->hr6 (K) | psfc bias hr1->hr6 (Pa) |
|---|---|---|---|---|---|
| CONT (= VALIDATED harness) | no | threaded | OFF | -0.34 -> -0.43 | +2648 -> +2571 |
| PERHOUR (= production product) | yes | None | ON | +1.05 -> +3.27 | +2567 -> +2174 |
| PH_CLOCK (isolate clock) | yes | threaded | ON | +1.04 -> +3.29 | +2567 -> +2173 |
| PH_GUARDOFF (isolate guards) | yes | None | OFF | -0.33 -> -0.08 | +2648 -> +2420 |

Reading:
- **PERHOUR vs PH_CLOCK** differ ONLY in the radiation clock. They are IDENTICAL
  (+3.27 vs +3.29 K). => the radiation diurnal-clock reset is NOT the cause.
- **PERHOUR vs PH_GUARDOFF** differ ONLY in `disable_guards`. PH_GUARDOFF collapses
  the warm drift to CONT levels (theta0 -0.08 K). => **guards are the WHOLE cause.**
- **CONT vs PH_GUARDOFF** differ ONLY in the per-hour reset (both guards-off). Nearly
  identical (-0.43 vs -0.08 K) => the per-hour carry+step reset is ~harmless.
- psfc: ~+2.6 kPa on ALL configs at hour 1 regardless of guards/clock/reset =>
  pressure error is a separate dycore equilibration issue, not the path divergence.
  (Guards-on slightly REDUCES the psfc bias drift because the warmer column lifts ph;
  not a fix, a side effect.)

CONT vs PERHOUR pairwise STATE divergence appears at hour 1 (theta0 +1.38 K mean,
max +3.87 K) and builds to hour 6 (+3.70 K mean, max +12.1 K) — concentrated in
theta, ~0 in the dynamics; first operator = the post-RK3-step theta limiter.

## Root cause (the exact operator + mechanism)

`operational_mode._physics_boundary_step_with_limiter_diagnostics` line ~1613, when
`disable_guards=False`, calls `_limit_guarded_dynamics_state_with_diagnostics(
next_state, physical_origin)` after EVERY RK3 step. That:

1. `_theta_level_monotonic_bounds(origin.theta)` (line 535): per model level, takes
   the horizontal domain-wide MIN and MAX of the PRE-step theta and uses them as tight
   `[lower, upper]` bounds for the new candidate theta.
2. `_positive_definite_theta_increment_limiter` (line 447): clamps each cell's new
   theta into `[lower, upper]`, then — to "conserve" the mass-weighted theta integral
   — REDISTRIBUTES the clamped-away increment over cells with room (lines 495-503).

Over the open ocean overnight (93% of d02 is sea), the coldest columns trying to cool
below the per-level domain minimum are clamped at that minimum; the limiter treats the
suppressed cooling as "removed mass" and pumps it back as warming across the column to
conserve the integral. The result is a systematic per-step warm injection that builds
over the first ~6 h then saturates — EXACTLY the signature the d02 T2-bias diagnosis
localized (+3-5 K lowest levels, too-shallow nighttime PBL, builds in 6 h then holds).
This is load-bearing: it is the entire difference between matching CPU-WRF and a
+3.3 K warm drift.

The wide envelope `[_THETA_LIMITER_MIN_K=0, _THETA_LIMITER_MAX_K=500]` K never fires;
only the TIGHT per-level monotonic bounds + redistribution bias the physics.

## v0.1.0 d02 PROOF-VALIDITY verdict

The D02_VALIDATED proof was measured on the CONTINUOUS guards-OFF harness, which
this experiment shows MATCHES CPU-WRF state (cool theta0, beats persistence). That
proof's PHYSICS is sound. BUT the operational PRODUCT (the wrfouts a user gets via
`daily_pipeline`) ran guards-ON and is +3.3 K warmer than the validated number — so
the shipped product did NOT match the validated harness. This is a real product bug,
not a proof-fabrication: the validated integration is honest and correct; the
production default flag diverged from it.

Resolution path: make the production operational integration behaviorally identical
to the validated guards-off path for all physical (finite, in-envelope) states, while
keeping a genuine non-finite safety net (so the guard stops being load-bearing). With
that fix, the validated d02 T2 number STANDS as the operational number, and v0.1.0 can
be promoted. (The +2.6 kPa pressure-Exner item is separate and pre-existing on BOTH
paths; it does not block the path-divergence resolution.)

## Files / proofs
- `scripts/diag/d02_operational_path_divergence.py` — the isolation experiment.
- `proofs/v010_validation/path_divergence_case3.json` — per-hour psfc/theta0 bias by config.
- analytic coszen check (inline): per-hour path pins coszen at +0.39..+0.49 for all
  24 h (sun never sets) vs the correct diurnal cycle of CONT — REAL but second-order
  vs the guard (PERHOUR==PH_CLOCK proves it).
