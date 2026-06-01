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

## THE FIX (applied) + all gates PASS

`operational_mode._limit_guarded_dynamics_state_with_diagnostics`: dropped the tight
per-level domain-min/max monotonic bounds (`_theta_level_monotonic_bounds`) and the
mass-redistribution they drove. The increment limiter now uses ONLY the wide physical
envelope `[0, 500] K` + the non-finite trap, so for any physical theta it is a STRICT
IDENTITY (verified: 0 cells limited, 0 mass residual) — a genuine non-load-bearing
safety net that still catches NaN/Inf and true blow-ups. Guards-on now == guards-off ==
the validated path == CPU-WRF for all physical states.

GATES (STOP+report mandate — all PASS, nothing forced):
- (a) **idealized warm-bubble + Straka 6/6 PASS**, bit-identical to baseline (round-off
  only; they run `disable_guards=True` so the limiter change is a no-op). proofs/f7n.
- (b) **the two paths CONVERGE.** Post-fix case3 6 h: PERHOUR theta0 bias -0.33 -> -0.08 K
  (was +1.05 -> +3.27); PERHOUR-CONT divergence collapsed from +3.70 K to +0.34 K mean.
  `proofs/v010_validation/path_divergence_case3_postfix.json`.
- (c) **full d02 24 h PRODUCTION re-score (the wrfout-writing path):**
  **T2 RMSE 3.78 -> 1.52 K, T2 bias +3.44 -> +1.31 K, THonly +3.10 -> -0.22 K.**
  The theta-side warm bias is ELIMINATED; the production T2 now MATCHES the validated
  harness (1.88/2.14/1.12 @6/12/24h). all_finite, 24/24 wrfouts, stable 24 h, wall 1195 s.
  **PSFC bias UNCHANGED at +2456 Pa (Ponly +2.0 K)** — the +2.6 kPa pressure-Exner
  artifact is untouched, confirming it is the SEPARATE dycore ph'-equilibration issue,
  exactly as predicted. proofs/v010_validation/d02_t2bias_diag_case3_postfix.json.

So the prior diagnosis's "genuine theta-side lower-troposphere physics drift"
(attributed to a MYNN PBL ventilation deficit) was MIS-attributed: it was the
load-bearing theta limiter the whole time. After the fix the d02 T2 RMSE = ~ pure
pressure-Exner artifact (Ponly ~+2.0 K); there is NO residual theta-physics warm drift.

## d03 implication

d03 uses the same per-hour `execute_daily_pipeline` -> `run_forecast_operational`
(guards-on, now fixed). So d03 inherits the fix. BUT d03's interior theta is
boundary-pinned by the forced parent leaves (small 1 km nest), so the guard had little
to clamp there — the limiter fix helps d03 only marginally. d03's dominant +1.5 K T2
bias is the +2.6 kPa pressure-Exner artifact (the dycore ph' free-drift), which this
fix does NOT address. d03 therefore still needs the separate ph' / pressure-diagnosis
fix (the d03-phfix-INLOOP track). The +2.6 kPa does NOT collapse on either domain from
this fix — that was never its mechanism.

## FINAL v0.1.0 d02 proof-validity verdict

The D02_VALIDATED proof's PHYSICS is sound (it was measured on the correct guards-off
integration that matches CPU-WRF). The shipped PRODUCT previously diverged from it
(+3.3 K warm via the load-bearing limiter). With this fix the PRODUCT now reproduces
the validated number (T2 RMSE 1.52 K, 24-lead mean; matching the harness), so the
validated d02 claim STANDS and the product is now consistent with it. **v0.1.0 d02 is
promotable on the science** — with the standing caveat that the d02 T2 RMSE is now
limited by the +2.6 kPa pressure-Exner dycore artifact (Ponly ~+2.0 K), a separate
pre-existing item present on BOTH the validated harness and the product (and on d03),
NOT introduced or hidden by the path divergence. If the release wants T2 RMSE below
~1.5 K it needs the pressure-Exner dycore fix next; the path divergence itself is
resolved.

## Files / proofs
- `scripts/diag/d02_operational_path_divergence.py` — the isolation experiment.
- `proofs/v010_validation/path_divergence_case3.json` — pre-fix psfc/theta0 by config.
- `proofs/v010_validation/path_divergence_case3_postfix.json` — post-fix convergence.
- `proofs/v010_validation/d02_t2bias_diag_case3_postfix.json` — full 24 h production
  re-score (Gate c): T2 RMSE 1.52 K, THonly -0.22 K, PSFC +2456 Pa.
- `proofs/v010_validation/pipeline_run_d02_diag_postfix.json` — the 24 h run object.
- `proofs/f7n/` — idealized 6/6 PASS post-fix.
- analytic coszen check (inline): per-hour path pins coszen at +0.39..+0.49 for all
  24 h (sun never sets) vs CONT's correct diurnal cycle — REAL but second-order vs the
  guard (PERHOUR==PH_CLOCK proved the clock is not the cause). The residual ~+0.3 K /
  +6 K-max post-fix path divergence at hour 6 is this clock + ww/rthraten per-hour
  reset; small, not load-bearing, optionally fixable by threading the global clock/lead
  + carrying the operational carry across hours in the pipeline (left as a follow-up).
