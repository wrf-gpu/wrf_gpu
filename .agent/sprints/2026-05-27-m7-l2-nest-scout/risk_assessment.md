# L2 backfill risk assessment

**Sprint**: `2026-05-27-m7-l2-nest-scout` AC5
**Question framed by the user**: top 5 risks for "publish results, start backfills today" on the L2 (9 km d01 + 3 km d02) configuration.

The assessment treats the two halves of the user's timeline independently:

- **Publishing the M7 results** (5.4 min/24 h Canary 3 km, 156.82× CPU baseline) is fundamentally a *write-up + repo-public* activity. The 27-day backfill is *not* a precondition. The risks below distinguish "publish risks" from "backfill risks" and call out which would block publishing vs. which would only delay backfills.
- **Starting backfills today** literally — i.e., kicking off Stage B / C / D on at least one L2 day before close of business — depends on the L2.1 d01 ingestor (`d01_boundary_forcing_audit.md`), which is *not yet shipped*.

## Risk table (top 5)

| # | Risk | Severity | Likelihood | Owner | Status |
|---|---|---|---|---|---|
| R1 | L2.1 d01 ingest sprint slips beyond today | High | Medium | manager + worker | Mitigated by clear ~½-day scope |
| R2 | First L2 d01 run produces unphysical fields (NaN, runaway theta, top-lid blow-up) at 9 km | Medium-High | Medium | tester (this sprint surfaces; followup sprint validates) | Mitigated by 6 h spin-up gate before claiming 72 h |
| R3 | Hydrometeor boundary inflow is zeroed (Q1 in audit), measurably degrading T+12-72 h cloud/precip skill on the d02 child | Medium | High (will happen by design in v0) | tester + worker | Documented gap; ship v0, follow up |
| R4 | Publishing the M7 result while L2.x is still in flight implies/promises something we cannot yet deliver | Medium | Medium | user (publishing call) + manager | Mitigated by language: "Canary 3 km validated, multi-day backfill in flight" |
| R5 | Operator data path: 24/28 L2 days have stripped wrfouts (AC1), so Tier-4 ground truth at d01 must use the *current-Gen2-rerun* d01 (only 4 days) OR rely on thin_gridded surface fields | Medium | Certain (already true) | tester | Mitigated by limiting Tier-4 d01 RMSE to the 4 surviving-wrfout days, scoring d02 against thin_gridded fields for the other 24 |

Severity scale: blocks publishing > delays-by-week > delays-by-day > cosmetic.
Likelihood: certain > high > medium > low.

## R1 — L2.1 d01 ingest slip

**What goes wrong**: the new code path takes longer than the ~½-day budget because (a) `grid_spec_from_wrfinput` discovers a header-attr mismatch with `Gen2GridSpec`, (b) `pack_wrfbdy_all_times` mis-orders W/E vs S/N sides, or (c) the 6 h cadence breaks an implicit `update_cadence_s=3600.0` assumption baked into the daily pipeline orchestration (`scripts/m7_daily_pipeline.py`, `gpuwrf.integration.daily_pipeline`).

**Severity**: High because everything downstream (L2.2..L2.6) blocks on this. **Not** blocking on publishing the M7 result.

**Likelihood**: Medium. The decoder and apply-side infra exist; the gluing logic is small. Risk is "small bug found in production after staging" rather than "scope wrong."

**Owner**: manager dispatches one worker; tester sprint here is read-only.

**Mitigation**:

1. Pre-write the unit test against synthetic linearly-varying wrfbdy data; ship test-first (it is small).
2. Smoke-test on one L2 day first; bail on Tier-4 if smoke fails.
3. Don't bundle L2.4/L2.5 (the dx_m fix and dt-cap relaxation) into L2.1 — keep L2.1 minimal.

## R2 — L2 d01 9 km dynamics blow up on first run

**What goes wrong**: at 9 km on the Canary parent (mostly ocean, terrain in the centre), one of the following manifests:

- Top-lid / Rayleigh damping mis-tuned for the larger domain ⇒ vertical-velocity blow-up at the model top.
- `w_damping` is on in WRF L2 namelist; if our GPU operational mode doesn't replicate it, w can drift past the WRF clamp.
- The `MAX_LIFTED_DYCORE_DT_S = 12.0` cap with 6 acoustic substeps means dt_a = 2 s; CFL_acoustic at dx=9 km with c_s≈340 m/s is 0.076 (very safe), but CFL_advection at dx=9 km with u_max≈40 m/s is 0.053 (very safe). So CFL is **not** the failure mode if the cap is honored. The risk is operator-specific (e.g. moisture-coupling tendency overflow at the broader domain).
- The d02 path's "production guards are load-bearing" finding from M6c-01 (`Sprint hypothesis (step-2 scratch divergence) DISPROVEN by evidence … production GUARDS are load-bearing`) may or may not generalize to d01. Guards are not all dx-aware.

**Severity**: Medium-High. If d01 produces nonfinites, Stage C is also blocked (d02 boundary forcing comes from d01).

**Likelihood**: Medium. The dycore, physics, and operator coverage are validated at d02 3 km; the parameter sweep is small (different dx, different domain extent, different surface fraction). Most likely outcome is "first run produces finite but somewhat off-target fields."

**Owner**: tester sprint after L2.2 ships.

**Mitigation**:

1. Tier-2 gate (finiteness + bounds + mass continuity residual) on a 6 h spin-up run before claiming 24 h. Already an M6 invariant — surface in the L2 spin-up.
2. Compare GPU d01 wrfout to L2's *own* d01 wrfout on the 4 surviving days (`20260509, 20260521 (partial 20h), 20260524, 20260525`) at T+1h, T+6h, T+24h, T+48h. Tier-4 RMSE envelope: T2 ≤ 3 K, U10/V10 ≤ 7.5 m/s same as M6 close (from `feedback_validation_philosophy`).
3. If T2 or wind RMSE blows past 3 K / 7.5 m/s but the field is finite: file a follow-up analysis sprint (parallel multi-angle bug hunt per `feedback_parallel_bug_angles_and_plan_critique`). Do not roll back; keep d02 path on its proven boundary source for the rest of the backfill.

## R3 — Hydrometeor boundary inflow zeroed at d01

**What goes wrong**: `State` carries `qv_bdy` but not `qc/qr/qi/qs/qg/qni/qnr_bdy`. Boundary inflow of cloud water, rain, ice, snow, graupel, and number concentrations is zero at d01's western inflow. For Canary days where Atlantic stratocumulus advects from the west, cloud arrives at the GPU d01 boundary as "dry air" and the cloud needs to spin up inside the domain. AIFS-driven L2 CPU WRF does carry these species in wrfbdy.

**Severity**: Medium. Affects RRTMG SW/LW radiation budget (cloud overpredicts/underpredicts), which feeds back into T2, and degrades downstream d02 island-scale precipitation skill on cloudy days.

**Likelihood**: Certain (this is what we ship by design in v0). Severity is the question, not whether it happens.

**Owner**: tester documents the gap in L2.3 Tier-4 report; worker takes followup sprint to extend `State` with the seven hydrometeor `*_bdy` leaves.

**Mitigation**:

1. Score Tier-4 vs L2 d02 thin_gridded with the gap in place. If T2/U10/V10 stay inside envelope, the gap is not load-bearing for first-rollout backfills.
2. Followup sprint (L2.7) extends `State` schema; ~30 lines of state.py + 20 lines of boundary_apply.py + tests.
3. Document the gap in the backfill manifest so the published result is honest.

## R4 — Publishing claim runs ahead of delivery

**What goes wrong**: the M7 result published today says "GPU-native regional NWP at 156.82× CPU on Canary 3 km" — which is verified. But the implicit messaging "ready for operational backfill" runs ahead of the L2.1..L2.6 sprint sequence, since L2 d01 ingest does not exist yet.

**Severity**: Medium. Reputational, not technical. The published numbers are real.

**Likelihood**: Medium. The principal's "publish + start backfills today" directive may be interpreted as "the backfill is already happening" by readers.

**Owner**: user (publication call) and manager (status reporting).

**Mitigation**:

1. Phrase the public claim as "Canary 3 km single-domain forecast validated at 5.4 min/24 h on RTX 5090, 156.82× CPU. Multi-day historical backfill of Gen2 L2 nested configuration in progress." This is true and surveyable.
2. The user can publish today; the backfill rollout can be tomorrow-or-this-week without revising the publication.
3. Do not publish a "27 days of GPU-driven forecasts" number until the backfill completes; the daily wall-clock projection is a projection, not a measurement.

## R5 — Reduced ground truth for d01 due to wrfout stripping

**What goes wrong**: from AC1, only 4/28 L2 days have raw `wrfout_d01_*` series (`20260509`, `20260521` partial 20 h, `20260524`, `20260525`). For the other 24 days, the ground truth at d01 is **only** `thin_gridded_d01_v1.nc` (21 variables, surface-level only, no DX/CEN_LAT global attrs because they were dropped during post-processing). All 28 days have full `wrfinput_d01` (the IC) and `wrfbdy_d01` (the BC) which is what Stage B needs as **input**; only the **truth** is degraded.

**Severity**: Medium. Tier-4 surface-level RMSE (T2, U10, V10) is still computable from `thin_gridded_d01_v1.nc` (verified: T2, U10, V10 are in the 21-var subset, on the mass grid 93×59 with 73 hourly time steps). What is NOT computable is upper-air RMSE, profile-by-profile temperature/wind/moisture comparison, or wrfout-vs-wrfout column structure verification, except on the 4 surviving-wrfouts days.

**Likelihood**: Certain (this is already disk state).

**Owner**: tester documents in L2.3 Tier-4 report.

**Mitigation**:

1. Limit upper-air verification of GPU d01 to the 4 surviving-wrfouts days. That is enough days for first-rollout confidence; rerun more days if doubt persists.
2. Surface-level Tier-4 (T2, U10, V10) uses thin_gridded for all 28 days — the project's headline RMSE metric is unaffected.
3. As a follow-up (post-publication), kick off a "preserve raw d01 wrfouts" patch in Gen2's post-processing for the next nightly run.

## Cross-cutting risks (mentioned but not in top 5)

- **R6 (cosmetic)**: F4 12-s dt cap means d01 GPU spends ~50 % more steps than WRF's 18 s. Wall-clock projection (~3-5 min/24 h) still well-inside budget. Lift after L2.6 only if perf matters.
- **R7 (numerical)**: F3 silent 3 km default in `surface_layer` underestimates vsgd at 9 km. Affects MM5 sfclay drag on the d01 parent. Small fix queued as L2.4.
- **R8 (operational)**: GPU is shared with the parallel L2 d02 replay sprint, the honest-speedup sprint, and any future profiling. Sequence by tmux window so only one GPU job runs at a time.
- **R9 (governance)**: This sprint is research-only; no `src/` writes from this tester. The L2.1 ingest is a worker sprint and follows the normal sprint-contract → reviewer-approval → manager-merge protocol.

## Concrete go/no-go gates

For "start backfills today":

- **Gate G1**: L2.1 d01 ingest merged on `main` with passing unit tests. **Blocking**.
- **Gate G2**: L2.2 d01 ↔ d02 chain smoke-tests one day end-to-end on a day with both d01 and d02 wrfouts present (`20260524` recommended; success=1, both wrfouts present). **Blocking**.
- **Gate G3**: L2.3 Tier-4 RMSE on the same day inside envelope (T2 ≤ 3 K, U10/V10 ≤ 7.5 m/s) on both d01 and d02. **Blocking** for the *27-day backfill*; **non-blocking** for publishing the M7 result.

For "publish results today":

- **Gate P1**: M7 result already validated (5.4 min/24 h, 156.82× CPU, CV 0.42 %, D2H=0 confirmed). **Met as of `MILESTONE-M7-CLOSEOUT.md`** (3f16ca8).
- **Gate P2**: Public claim language does not over-promise. **User-owned**.

Publishing P1+P2 today is achievable. Starting full L2 backfills today requires G1+G2+G3, which is approximately 1.5 days of focused worker time — not "today" if "today" means "before sunset", but well within the user's likely working week.
