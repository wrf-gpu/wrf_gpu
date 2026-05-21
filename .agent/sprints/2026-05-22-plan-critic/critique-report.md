# Plan Critic Report - 2026-05-22

## Verdict

The manager is right that the model-critical path is blocked on dycore validity, but the current plan is not the fastest or smartest path to the constitutional end goal as written. It serializes too much operational readiness behind M6.x, over-trusts partial scaffold sprints, and uses "operational closeout" language before the project has observation-grade operational verification.

The end goal is a GPU-native regional NWP system, WRF-compatible where useful, validated explicitly against WRF/physical/or operational evidence, with Canary 3 km and then 1 km daily forecasts as the first target (`PROJECT_CONSTITUTION.md:5`). The non-negotiables require physics correctness before speed claims (`PROJECT_CONSTITUTION.md:9`), GPU-resident high-frequency state (`PROJECT_CONSTITUTION.md:10`), and Canary operational value before general WRF replacement scope (`PROJECT_CONSTITUTION.md:13`).

My bottom line: keep M6.x as an immediate kill-gate, but stop waiting on it for file-disjoint operational work. Start an M7-S0a ops/data readiness sprint now. If M6.x cannot produce green Tier-2 and 24h finite evidence quickly, invoke c1 clean-room Klemp-Skamarock rather than stretching the WRF-canonical port.

## AC1 - End-goal alignment

The project is still aligned in intent. M6-S5 honestly exposed that throughput alone is not enough: the GPU path clears 9.70x end-to-end but the lifted-cap forecast fails Tier-2 and saturates state bounds (`.agent/sprints/2026-05-21-m6-s5-adr-007-4x-verdict/manager-closeout.md:10`, `.agent/sprints/2026-05-21-m6-s5-adr-007-4x-verdict/manager-closeout.md:17`). That is the correct physics-first posture.

The plan drifts when it calls the queued M6-S8 an "Operational Closeout" while AC1 accepts GPU-vs-Gen2 RMSE within 1.5x of Gen2-vs-AIFS RMSE, treating AIFS as deterministic truth (`.agent/sprints/2026-05-22-m6-s8-operational-closeout/sprint-contract.md:13`, `.agent/sprints/2026-05-22-m6-s8-operational-closeout/sprint-contract.md:17`). M7 itself says operational verification binds to station observations for U10, V10, and T2, and no operational validation claim is allowed if observations are unavailable or stale (`.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md:449`, `.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md:453`). The validation strategy also says normal behavior is physically valid and statistically defensible against WRF fixtures, analytic oracles, and operational verification, not AIFS-as-truth (`VALIDATION_STRATEGY.md:3`).

M6-S8 can be a model-consistency closeout. It should not be presented as operational evidence unless it includes observation-side verification or explicitly labels itself non-operational.

## AC2 - Critical-path correctness

The identified model-critical path M6.x -> M6-S8 -> M7 is directionally correct. M6.x targets the actual failure surfaced by M6-S5: M4 acoustic proxy constants and missing mu continuity (`.agent/sprints/2026-05-22-m6x-wrf-canonical-dycore/sprint-contract.md:6`, `.agent/sprints/2026-05-21-m6-s5-adr-007-4x-verdict/manager-closeout.md:23`, `.agent/sprints/2026-05-21-m6-s5-adr-007-4x-verdict/manager-closeout.md:25`). M7 dispatch remains blocked pending M6.x and M6-S8 close (`.agent/sprints/2026-05-21-m6-s6-tier3-tsc/manager-closeout.md:52`, `.agent/sprints/2026-05-22-m6-s8-operational-closeout/sprint-contract.md:52`).

But the fastest path is not fully serial. The M7 plan already allows M7-S0 to run while final M6 closeout is being assembled, provided it emits BLOCKED if M6 evidence is missing (`.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md:64`, `.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md:67`). That sprint owns operational contracts, AIFS ingest contracts, and Gen2 inventory, not dynamics code (`.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md:37`, `.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md:47`). Waiting for M6.x before doing this is avoidable latency.

M6.x itself is also over-optimistic. The active M6.x contract estimates 16-32h for WRF-canonical dycore completion (`.agent/sprints/2026-05-22-m6x-wrf-canonical-dycore/sprint-contract.md:58`). The contingency design says WRF small_step_em has 2089 LoC and substantial hybrid-coordinate, off-centering, map-scale, and sumflux complexity (`.agent/sprints/2026-05-22-m6x-contingency-option-c-scope/design.md:32`, `.agent/sprints/2026-05-22-m6x-contingency-option-c-scope/design.md:37`). The same manager closeout admits that the 16-32h estimate is optimistic for WRF-canonical fidelity and ranks c1 clean-room first if M6.x fails (`.agent/sprints/2026-05-22-m6x-contingency-option-c-scope/manager-closeout.md:39`, `.agent/sprints/2026-05-22-m6x-contingency-option-c-scope/manager-closeout.md:31`). Treat M6.x as a short kill-gate, not a sprint to keep stretching.

One more hard blocker needs to be explicit in the live path: M6-S8 pre-dispatch requires F-5 denominator acceptance (`.agent/sprints/2026-05-22-m6-s8-operational-closeout/sprint-contract.md:58`, `.agent/sprints/2026-05-22-m6-s8-operational-closeout/sprint-contract.md:59`). M6-S5 already flagged denominator shopping as a follow-up (`.agent/sprints/2026-05-21-m6-s5-adr-007-4x-verdict/manager-closeout.md:56`). That has to be completed before S8 can honestly dispatch.

## AC3 - Over-engineering audit

M6-S7 was over-built before the data existed. The worker found only 3 usable pinned-grid complete members out of the required 10, and held-out validation lacked real d02 truth (`.agent/sprints/2026-05-21-m6-s7-tier4-probtest/worker-report.md:14`, `.agent/sprints/2026-05-21-m6-s7-tier4-probtest/worker-report.md:29`). The reviewer praised 1,257 LOC of scaffold (`.agent/sprints/2026-05-21-m6-s7-tier4-probtest/reviewer-report.md:31`). That scaffold is useful, but the faster sequence was a one-hour inventory first, then code only if the 10-member premise held.

M6-S6 was too broad for a dycore that was about to be invalidated. Its d02 drift AC blocked on CUDA OOM (`.agent/sprints/2026-05-21-m6-s6-tier3-tsc/manager-closeout.md:17`) and the envelope semantics require re-validation under M6.x uncapped dynamics (`.agent/sprints/2026-05-21-m6-s6-tier3-tsc/manager-closeout.md:20`). The oracle infrastructure was not wasted, but the d02 drift measurement should have waited for the uncapped dycore.

M6-S4 did useful interface work, but its strongest-looking invariant numbers are not as strong as they read. The reviewer says water closure is Thompson-internal bookkeeping, boundary closure is an arithmetic identity, and dry-mass closure is trivially satisfied under the inherited dycore cap (`.agent/sprints/2026-05-21-m6-s4-tier2-coupled-invariants/reviewer-report.md:66`, `.agent/sprints/2026-05-21-m6-s4-tier2-coupled-invariants/reviewer-report.md:72`). The state extension and pre-sanitize tap justified the sprint; the invariant PASS language should have been more restrained.

M6.5-D1 was misnamed as "backfill." It built a loader and audit surface, but it did not backfill the missing corpus. The worker reports only 3 complete d02 24h runs, one on the old 66 x 120 grid, and explicitly says the 10-member pinned-grid corpus still does not exist (`.agent/sprints/2026-05-22-m6-5-d1-gen2-data-backfill/worker-report.md:55`, `.agent/sprints/2026-05-22-m6-5-d1-gen2-data-backfill/worker-report.md:205`). This is useful M7 plumbing, not completed data backfill.

## AC4 - Missing work

The biggest missing work is operational readiness that can run now without touching dynamics.

First, live AIFS/WPS ingest is still plan text. M7-S1 is supposed to implement a real 18Z daily d01->d02 GPU run from live AIFS (`.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md:70`, `.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md:72`). The plan says v0 should reuse Gen2 WPS/WPS-like transformation rather than invent a new regridder (`.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md:331`, `.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md:333`). That work is an M7 plan item, not evidence already produced by the M6 closeout path.

Second, observation verification is not staged. M7-S5 requires an observation source manifest and operational scores against station observations (`.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md:191`, `.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md:213`). M6-S8 instead defines its RMSE gate against Gen2/AIFS (`.agent/sprints/2026-05-22-m6-s8-operational-closeout/sprint-contract.md:17`). That creates a gap exactly where the end-goal claim becomes public-facing.

Third, output, restart, monitoring, and scheduler work are still late in M7. M7-S4 owns NetCDF-like/Zarr/station products (`.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md:160`, `.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md:182`), M7-S6 owns restart/recovery (`.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md:225`, `.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md:245`), and M7-S7 owns operational status/alerts (`.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md:255`, `.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md:270`). Skeleton schemas and file ownership can be dispatched before dycore closure.

Fourth, M6.5-D1 acceptance left required amendments unapplied. ADR-016 still says "Proposed for M6.5-D1 review" (`.agent/decisions/ADR-016-gen2-data-corpus.md:3`, `.agent/decisions/ADR-016-gen2-data-corpus.md:5`), and it still has the 1 percent boundary-replay failure threshold that the reviewer said must be changed to 3 percent (`.agent/decisions/ADR-016-gen2-data-corpus.md:55`, `.agent/decisions/ADR-016-gen2-data-corpus.md:57`; `.agent/sprints/2026-05-22-m6-5-d1-gen2-data-backfill/reviewer-report.md:189`). The code default is still 0.01 (`src/gpuwrf/validation/data_quality.py:279`, `src/gpuwrf/validation/data_quality.py:285`).

## AC5 - Architectural risk audit

The manager has identified the biggest risk: the M4 dycore was a reduced proxy. The code confirms it: `acoustic_once` uses `c2 = 1.0` and `pressure_coupling = 1.0e-3` (`src/gpuwrf/dynamics/acoustic.py:66`, `src/gpuwrf/dynamics/acoustic.py:70`), while sanitize clips huge physical ranges after the coupled step (`src/gpuwrf/coupling/driver.py:817`, `src/gpuwrf/coupling/driver.py:828`). M6.x is necessary.

The missed risk is that WRF-canonical M6.x and clean-room c1 imply different definitions of WRF compatibility. c1 explicitly trades away hybrid coordinate details, off-centering, map-scale plumbing, and sumflux time-averaging for tractability (`.agent/sprints/2026-05-22-m6x-contingency-option-c-scope/c1-klemp-skamarock-contract.md:74`, `.agent/sprints/2026-05-22-m6x-contingency-option-c-scope/c1-klemp-skamarock-contract.md:76`). That may be the right engineering decision, but it needs a clear acceptance statement: "passes operational RMSE and core invariants, but is not WRF-canonical dyn_em." Otherwise the project will keep oscillating between WRF fidelity and Canary operational value.

The contingency insurance is also incomplete. Its contract required three option contract drafts and a new ADR-017 (`.agent/sprints/2026-05-22-m6x-contingency-option-c-scope/sprint-contract.md:24`, `.agent/sprints/2026-05-22-m6x-contingency-option-c-scope/sprint-contract.md:25`), but the manager closeout says c2/c3 contracts and ADR-017 were skipped (`.agent/sprints/2026-05-22-m6x-contingency-option-c-scope/manager-closeout.md:11`, `.agent/sprints/2026-05-22-m6x-contingency-option-c-scope/manager-closeout.md:35`). That is acceptable only if c1 is the sole realistic contingency. If not, the insurance sprint did not meet its own acceptance bar.

Additional risks that should move up: AIFS boundary quality and late/missing readiness (`.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md:489`), terrain/static correctness for the 1 km nest (`RISK_REGISTER.md:15`), the 1 km RTX 5090 memory/compile gate (`.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md:118`, `.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md:123`), and observation-source availability (`RISK_REGISTER.md:16`). These are not secondary to dycore if the deadline is an operational daily forecast.

## AC6 - Recommended next 2-3 sprints

1. `M6.x decision gate - WRF-canonical dycore or c1 pivot`
   Owner: current codex M6.x worker. Wall: hard 8-12h to a decision if not already green.
   AC sketch: physical sound speed, mu-continuity, 1h Tier-2 lifted-cap pass, 24h smoke finite, sanitize <5%, speedup dry-run. If any fail, do not extend; dispatch c1 from `.agent/sprints/2026-05-22-m6x-contingency-option-c-scope/c1-klemp-skamarock-contract.md`.

2. `M7-S0a operational/data readiness prologue`
   Owner: codex, file-disjoint from dynamics. Wall: 12-18h.
   AC sketch: create the M7 operational contract, AIFS/WPS ingest manifest, station observation source manifest, output/status schema, and a concrete Gen2 corpus backfill plan. It may emit BLOCKED on M6 evidence, as allowed by the M7 plan (`.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md:66`, `.agent/sprints/2026-05-21-m7-milestone-plan-scout/m7-milestone-plan.md:68`).

3. Conditional:
   If M6.x/c1 passes, dispatch `M6-S8 + d02 drift closeout`, not a broad operational sprint. Wall: 8-12h. AC sketch: use uncapped dycore, run d02 drift/process-isolated retry, compute Gen2 RMSE U10/V10/T2, close M6-S4 water/boundary follow-ups, pin the F-5 denominator, and amend ADR-007 only if stability, transfer, and speed all pass.
   If M6.x fails, dispatch `m6x-fallback-c1-klemp-skamarock`. Wall: 5-9 days per the contingency contract (`.agent/sprints/2026-05-22-m6x-contingency-option-c-scope/c1-klemp-skamarock-contract.md:55`).

## AC7 - Decision-quality audit

M6-S5: the accept-as-fail decision was correct. It prevented a speed-only success claim despite 9.70x throughput (`.agent/sprints/2026-05-21-m6-s5-adr-007-4x-verdict/manager-closeout.md:10`, `.agent/decisions/ADR-007-precision-policy.md:5`). The weak part is the phrase "throughput established"; because M6.x changes the dycore and M6-S5 still had a transfer regression and denominator issue (`.agent/sprints/2026-05-21-m6-s5-adr-007-4x-verdict/manager-closeout.md:55`, `.agent/sprints/2026-05-21-m6-s5-adr-007-4x-verdict/manager-closeout.md:56`), throughput should stay provisional until the final dycore and denominator are in the same artifact.

M6-S6: closing as scaffold-partial was reasonable. The manager did not hide the OOM and correctly deferred d02 drift until after M6.x (`.agent/sprints/2026-05-21-m6-s6-tier3-tsc/manager-closeout.md:10`, `.agent/sprints/2026-05-21-m6-s6-tier3-tsc/manager-closeout.md:26`). The only caveat is that S8 must not close until the follow-up actually re-runs against the uncapped dycore (`.agent/sprints/2026-05-22-m6-s6-followup-d02-drift-retry/sprint-contract.md:13`, `.agent/sprints/2026-05-22-m6-s6-followup-d02-drift-retry/sprint-contract.md:20`).

M6.5-D1: this should not be called closed in this worktree. The role prompt directed me to read `.agent/sprints/2026-05-22-m6-5-d1-gen2-data-backfill/manager-closeout.md` (`.agent/sprints/2026-05-22-plan-critic/role-prompts/worker.md:19`), but the sprint evidence I could inspect is contract, worker report, and reviewer report only. ADR-016 is still Proposed, the required 3 percent threshold amendment is not applied, and the code default still uses 1 percent (`.agent/decisions/ADR-016-gen2-data-corpus.md:5`, `.agent/decisions/ADR-016-gen2-data-corpus.md:57`, `src/gpuwrf/validation/data_quality.py:285`). The reviewer says M7-S0 is unblocked (`.agent/sprints/2026-05-22-m6-5-d1-gen2-data-backfill/reviewer-report.md:258`, `.agent/sprints/2026-05-22-m6-5-d1-gen2-data-backfill/reviewer-report.md:265`), but that is only true for the loader/RMSE adapter, not for the data corpus or Tier-4 production tolerance. Manager should write the closeout, apply the ADR/code amendment, and label the corpus status as "loader ready, corpus incomplete."

## Final recommendation

Do not add more review process. Make the next moves sharper:

- Force M6.x to a fast green/red decision.
- Run M7-S0a operational/data readiness in parallel now.
- Stop using "operational" for M6-S8 unless station-observation verification is in scope.
- Treat M6.5-D1 as incomplete until ADR-016 and the threshold default are reconciled.
- If M6.x slips, invoke c1 immediately instead of continuing a WRF-canonical port whose own contingency analysis says the estimate is optimistic.
