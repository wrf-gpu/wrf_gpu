# Project Reset Plan Critique

Role: ADVERSARIAL CRITIC  
Target: `.agent/decisions/PROJECT-RESET-PLAN-DRAFT.md`

Evidence note: the contract-required `src/gpuwrf/operational_mode.py` and `tests/savepoint/` paths do not exist in this worktree. I used `src/gpuwrf/runtime/operational_mode.py`, the actual operational-mode path cited by prior RCA, and the actual B6 savepoint tests/proofs under `tests/test_m6b6_coupled_step_parity.py`, `scripts/m6b6_coupled_step_compare.py`, and `.agent/sprints/2026-05-27-testing-plan-execution-redo/`.

## C1 - Hidden Assumptions

| # | Hidden load-bearing assumption | Evidence | Break impact |
|---:|---|---|---:|
| 1 | The remaining work is a known sequence of fixes, not still an RCA problem. | Iter2 already applied theta-envelope, hourly land refresh, and 5-row boundary-strip changes, but still ended `BLOCKED`; all T2/U10/V10 skill gates failed (`worker-report.md:17-23`, `worker-report.md:89-96`, `worker-report.md:141-146`). | 5 |
| 2 | Savepoint parity rails are enough to protect operational skill. | B6 is validation-only in the comparator (`scripts/m6b6_coupled_step_compare.py:457-464`); deep savepoint proof says 100-step column parity passed but 1000/10000 gates are unmet (`savepoint_parity_deep.json:23-33`, `savepoint_deep_column100.json:64481-64493`). | 5 |
| 3 | The latest 22.26x speedup has enough margin after correctness fixes. | Iter2 speedup is 22.2558x d02-only (`post_iter2_speedup.json:22-32`), leaving only a 2.2x margin before the 10x floor; Noah-MP and larger validation runs are unmeasured. | 4 |
| 4 | A pinned 5-day Canary invariant case is available and representative. | The publication redo found only two complete 24h GPU cases against a required five and failed `CANARY-MULTIDAY-SIDE-BY-SIDE` (`canary_multiday_skill.json:678-715`). | 5 |
| 5 | Monotonic RMSE ratcheting per milestone is realistic. | Iter2 preserved invariants and speed, yet skill remained bad and wind metrics worsened versus predecessor partial fix (`worker-report.md:1-3`, `worker-report.md:141-146`). | 4 |
| 6 | M9's surface-flux/MYNN target is still the same defect as the Opus RCA. | Current code already calls `surface_adapter` before `mynn_adapter` (`operational_mode.py:587-593`) and MYNN now consumes `state.theta_flux`, `state.qv_flux`, `state.tau_u`, `state.tau_v` (`physics_couplers.py:268-292`). The remaining M9 risk is parity/units/coupled behavior, not simply "wire fluxes." | 3 |
| 7 | The theta limiter is a small low-risk cleanup. | Current guard replaces out-of-envelope theta with a clipped origin fallback across lower and upper levels (`operational_mode.py:214-227`), and test coverage only asserts the envelope constants (`test_m7_skill_fix_iter2.py:26-35`). | 4 |
| 8 | Prognostic Noah-MP can be built in 4-6 weeks and will materially fix T2. | Current land path is hourly Gen2 replay, explicitly not prognostic (`daily_pipeline.py:303-355`; `worker-report.md:143-145`). M11 asks for 24h bitwise Noah-MP parity (`PROJECT-RESET-PLAN-DRAFT.md:40`). | 4 |
| 9 | Local data can support M13's 15-30 seasonal L2/L3 corpus quickly. | Current local inventory failed even the five-complete-day gate (`canary_multiday_skill.json:681-715`). | 5 |
| 10 | `p > 0.05` is a sufficient statistical-equivalence gate. | The draft defines "not statistically distinguishable" as paired t-test `p > 0.05` (`PROJECT-RESET-PLAN-DRAFT.md:13`, `PROJECT-RESET-PLAN-DRAFT.md:43`, `PROJECT-RESET-PLAN-DRAFT.md:78`), but non-rejection is not an equivalence proof without predeclared effect margins. | 4 |
| 11 | Radiation/surface/PBL fixes explain most of the first-hour error. | Codex RCA names radiation/surface/PBL as leading but still returns `MULTIPLE_CONTRIBUTORS` because `LU_INDEX` differs and `PSFC` has a uniform bias in physics-on and physics-off runs (`worker-report.md:7-11`). | 4 |
| 12 | Boundary forcing can remain a lower-priority follow-up. | Boundary-vs-interior evidence demotes lateral BCs for first-hour winds, but PSFC is spatially uniform and not explained by physics toggles (`boundary_vs_interior.json:58-95`; `physics_on_off_bracket.json:1640-1647`). | 3 |
| 13 | Static-field parity is already covered. | `LU_INDEX` differs at lead 1 by up to 14 categories while HGT/LANDMASK match (`first_hour_diff.json:110-126`, `first_hour_diff.json:59-92`). Current `State` has `xland/lakemask/roughness_m` but no `lu_index` field (`state.py:67-90`). | 4 |
| 14 | Conservation is not on the critical path. | Publication redo failed mass and energy gates for missing closed-domain/boundary-flux/CPU-envelope evidence (`aggregate_report.md:10-16`, `conservation_mass_24h.json:232-258`, `conservation_energy_24h.json:356-380`). | 4 |
| 15 | Cross-AI review at sprint close is enough to catch plan-level statistical and physics errors. | The M7 closeout was amended after publication-readiness claims were superseded by speed and skill corrections (`MILESTONE-M7-CLOSEOUT-AMENDMENT.md:5-10`, `MILESTONE-M7-CLOSEOUT-AMENDMENT.md:51-60`). | 4 |

## C2 - Milestone Ordering

M8 first is correct. The draft needs the first bitwise divergence operator before it spends more sprints on local fixes; the current proof base is validation-only in B6 and fails publication-depth savepoint requirements (`scripts/m6b6_coupled_step_compare.py:457-464`; `savepoint_parity_deep.json:23-33`).

M11 earlier - argument for: surface-flux fixes are limited if lower-boundary state is not physically evolved. Current land refresh is hourly replay from Gen2 wrfout, not prognostic Noah-MP (`daily_pipeline.py:303-355`; `worker-report.md:143-145`). If the goal is a standalone JAX-native WRF port, M11 is not optional.

M11 earlier - argument against: the existing hourly replay at least gives time-varying `TSK/SST/SMOIS/SH2O/TSLB` fields (`land_state.py:110-165`, `daily_pipeline.py:238-261`), and iter2 still failed badly after adding it (`worker-report.md:83-96`). That means the immediate blocker is probably atmospheric coupling/parity, not absence of prognostic land alone. Full Noah-MP before M8/M9/M10 risks burying a smaller atmospheric defect under the largest new-code sprint.

M10 before M9 - argument for: the guard can mask or reshape M9 results because out-of-envelope theta falls back to clipped origin values (`operational_mode.py:214-241`) and is applied after physics/boundary (`operational_mode.py:600-617`). If M9 is judged by T2 RMSE, a guard-induced false improvement or false regression is plausible.

M10 before M9 - argument against: Codex RCA's strongest first-hour signal is radiation/surface/PBL (`worker-report.md:7-11`), and current M9 parity work can instrument `SWDOWN/GLW/HFX/LH/PBLH/TSK/T2` before accepting RMSE gains. Removing the guard first could reintroduce the runaway it currently contains without proving the physical source terms.

Ordering recommendation: keep M8 first, split M9 into "radiation/surface/PBL parity audit" and "surface/MYNN bottom-BC fix if audit proves it," run M10 as an acceptance sub-gate inside M9 rather than a standalone low-risk sprint, then do M11 only after M8/M9/M10 show 24h bounded physics without hidden clipping.

## C3 - Risk Reassessment

| Milestone | Draft risk | Critic risk | Why |
|---|---|---|---|
| M8 | Low | Medium | The intended contract path is missing, B6 is validation-only, and publication-depth savepoint parity is still `FAIL_INSUFFICIENT_SAVEPOINT_DEPTH` (`savepoint_parity_deep.json:23-33`). |
| M9 | Medium | High | The latest iter2 code already has flux-to-MYNN wiring, radiation cadence 180, hourly land refresh, and 5-row boundary strips, yet T2/U10/V10 remain far outside +20% (`physics_couplers.py:268-292`; `daily_pipeline.py:78-79`; `post_iter2_skill_diff.json:29-36`, `post_iter2_skill_diff.json:60-67`, `post_iter2_skill_diff.json:91-98`). |
| M10 | Low | High | It is not just replacing a clip; it is replacing a fail-closed guard that preserves finite 24h runs (`operational_mode.py:214-241`, `worker-report.md:17-22`). |
| M11 | High | Very High | Full prognostic Noah-MP with 24h bitwise WRF parity is the largest new-code target, and the current land path is explicitly replay, not prognostic (`worker-report.md:143-145`). |
| M12 | Low | Medium | Thompson column work exists, but removing operational admissibility guards can expose coupled moisture/energy failures not covered by "no NaN" alone (`operational_mode.py:578-590`; conservation gaps in `aggregate_report.md:10-16`). |
| M13 | Low | High | Data/evaluation availability is not proven; current local inventory cannot satisfy five complete 24h cases, much less 15-30 seasonal L2/L3 cases (`canary_multiday_skill.json:681-715`). |
| M14 | Medium | High | Statistical equivalence is currently defined as `p > 0.05`, which is not a valid equivalence test without margins (`PROJECT-RESET-PLAN-DRAFT.md:78`). |
| M15 | Low | Medium-High | Perf floor has only 22.26x margin after iter2 (`post_iter2_speedup.json:22-32`), and M11/M13/M14 add unprofiled work. Release/paper also depends on failed publication tests (`aggregate_report.md:23-31`). |

## C4 - Delta Percent Calibration

The +5/+15/+5/+20/+3/+10/+5/+5 distribution is not defensible as skill gain. It mixes implementation work, validation coverage, statistical gating, and release packaging into one progress scalar (`PROJECT-RESET-PLAN-DRAFT.md:35-46`).

My calibrated ranges:

| Milestone | Draft delta | Critic expected actual gain | Reason |
|---|---:|---:|---|
| M8 | +5 | +0 to +5 | Audit work improves knowledge, not skill, unless it directly fixes the first divergent operator. |
| M9 | +15 | +5 to +25 | Surface/radiation/PBL is a leading suspect (`worker-report.md:7-11`), but current partial fixes did not recover skill (`worker-report.md:89-96`). |
| M10 | +5 | -10 to +10 | Removing/replacing a guard can improve physicality or destabilize the run; current evidence only proves boundedness, not skill (`test_m7_skill_fix_iter2.py:26-35`). |
| M11 | +20 | +0 to +25 | Prognostic land can matter for T2, but hourly replay already failed to recover skill (`worker-report.md:19`, `worker-report.md:89-96`). |
| M12 | +3 | +0 to +5 | Microphysics has weak direct evidence for T2/U10/V10 skill in the dry Canary station gate; conservation/moisture closure is unproven. |
| M13 | +10 | +0 model skill, +15 confidence | Corpus work measures truth; it does not improve forecasts unless it feeds back into fixes. |
| M14 | +5 | +0 | Statistical equivalence is an outcome, not an implementation increment. |
| M15 | +5 | +0 skill, possible negative perf | Release/profiling can preserve or reduce performance; it should not be credited as forecast-skill progress. |

If "percent complete" is retained, separate it into model-correctness progress, validation-confidence progress, and release-readiness progress. A single scalar hides the actual risk.

## C5 - Missing Milestones

1. Static-field and land-use parity milestone. `LU_INDEX` differs by up to 14 categories at lead 1 while LANDMASK and HGT match (`first_hour_diff.json:59-126`), and current runtime state has no `lu_index` leaf (`state.py:67-90`). This affects roughness, land class, and surface diagnostics.

2. Boundary/pressure forcing completeness milestone. Current `apply_lateral_boundaries` applies U/V/theta/QVAPOR/PH/MU only (`boundary_apply.py:39-45`), while `BoundaryState` already sketches missing W/P/PB fields (`state.py:170-240`). Uniform PSFC bias persists in physics-on and physics-off brackets (`physics_on_off_bracket.json:1640-1647`), so pressure/base-state/boundary handling needs its own gate.

3. Conservation and closure milestone. Mass and energy publication tests failed for missing closed-domain, boundary-flux, and CPU-envelope evidence (`conservation_mass_24h.json:232-258`; `conservation_energy_24h.json:356-380`). Add dry-mass, moisture, column water, surface flux, and energy-budget closure before M14.

4. Validation-corpus availability and statistics-design milestone before M13/M14. The local corpus currently fails the five-complete-day gate (`canary_multiday_skill.json:681-715`), and the draft's `p > 0.05` gate needs equivalence margins before the ensemble is run (`PROJECT-RESET-PLAN-DRAFT.md:78`).

5. Idealized GPU forecast-runner milestone. Publication redo skipped warmbubble, density-current, mountain-wave, CFL, and acoustic-substep gates because reviewed GPU runners are missing (`aggregate_report.md:7-13`, `aggregate_report.md:23-31`). These are not optional if physics claims require analytic/idealized evidence under `AGENTS.md:20`.

## C6 - Invariant Ladder Gaps

INV-1 is necessary but underspecified. ADR-027 requires warmed Nsight with at least three warmups and counts inter-kernel D2H, not total D2H (`ADR-027-d2h-invariant-clarification-PROPOSED.md:50-58`). Each proof should include the `.nsys-rep` path, parser version, pre-kernel D2H count, H2D count, and profiled step count.

INV-2/INV-3 do not catch enough. B6 parity is validation-only and not a production timestep loop (`scripts/m6b6_coupled_step_compare.py:351-354`, `scripts/m6b6_coupled_step_compare.py:457-464`). Add an operational-mode parity invariant: first divergent WRF-output variable/operator for 1h and 24h runs, including `SWDOWN/GLW/HFX/LH/PBLH/TSK/T2/U10/V10/PSFC/LU_INDEX`.

INV-4 is too brittle. "RMSE must decrease or hold equal" on one pinned case (`PROJECT-RESET-PLAN-DRAFT.md:55`) will reject legitimate fixes that improve ensemble skill but worsen one variable/case. Replace with: no catastrophic regression on pinned smoke case, median RMSE non-increase over a fixed mini-ensemble, and named waiver for known tradeoffs.

INV-5 needs methodology guards. The floor should use the d02-only denominator, de-duplicated CPU timing, JIT excluded, same workstation, and transfer audit, matching the corrected timing approach (`post_iter2_speedup.json:22-32`; `MILESTONE-M7-CLOSEOUT-AMENDMENT.md:27-39`).

INV-6 is good but incomplete. Add proof-schema validation and "no proof omission": every invariant missing a proof object is a failure, not "not run."

Add these invariants:

- INV-7: mass/moisture/energy closure budget with CPU-envelope or closed-domain oracle.
- INV-8: static field parity for HGT/LANDMASK/XLAND/LU_INDEX/roughness/soil category.
- INV-9: boundary forcing completeness for U/V/W/T/QVAPOR/P/PB/PH/MU and relax-zone width.
- INV-10: guard/limiter accounting: per-step clip counts, first clipped field/cell, and zero hidden fallback in accepted physics windows.
- INV-11: evaluation sufficiency: case count, seasons, station joined rows, and predeclared equivalence margins.

## C7 - Cross-AI Verification

Codex frontrunner plus Opus tester/reviewer is enough for narrow implementation sprints with machine-checkable proof objects. It is not enough for M8-M11 and M14 as written.

Tighter oversight is justified because the prior manager closeout converted finite station scores into an operational-skill implication that was later corrected (`MILESTONE-M7-CLOSEOUT-AMENDMENT.md:51-60`), and because current failures cross physics, statistics, and data availability. For M8-M11, require a third independent reviewer at milestone close plus one blinded "proof auditor" whose only job is to reproduce the proof-object interpretation from raw artifacts. For M14, require a statistics-specific review before running the ensemble, not after.

Looser oversight is acceptable for M12/M15 only if the proof schema is already frozen and the sprint does not change model physics. M13 can use lighter code review but needs strict data-provenance review.

## C8 - Timeline Reality Check

The 17-23 week estimate is optimistic. A realistic range for the binding goal is 30-54 weeks unless scope is reduced or prior unseen evidence exists.

| Milestone | Draft weeks | Critic weeks | Reason |
|---|---:|---:|---|
| M8 | 3-4 | 4-6 | Needs production divergence maps, path cleanup, and savepoint depth clarification; current `tests/savepoint/` path is absent and B6 is validation-only. |
| M9 | 2-3 | 4-8 | Surface/radiation/PBL is likely but not isolated; current partial implementation still fails all skill gates. |
| M10 | 1-2 | 3-5 | Replacing a protective theta guard requires stability, clip-accounting, and 24h evidence. |
| M11 | 4-6 | 8-14 | Full GPU Noah-MP 24h bitwise parity is a large new physics port, not a small extension of hourly replay. |
| M12 | 1-2 | 2-4 | Coupled microphysics admissibility and closure need more than no-NaN checks. |
| M13 | 2 | 3-6 | The local corpus does not currently satisfy five complete cases, so 15-30 seasonal L2/L3 cases need data work. |
| M14 | 2 | 3-6 | Equivalence margins/statistical design must be fixed, then rerun and adjudicated. |
| M15 | 2 | 3-5 | Profiling after M11 plus release/paper cleanup is not low-risk with only 22.26x current d02 margin. |

## C9 - Strongest Objection

The draft treats the remaining roadmap as ordered implementation of known fixes, but the latest iter2 proof shows the named theta, land-refresh, boundary-strip, radiation-cadence, and surface-to-MYNN changes preserved invariants and still failed T2/U10/V10 skill by large margins, so the plan is not yet a path to parity; it is an under-instrumented RCA backlog (`worker-report.md:17-23`, `worker-report.md:83-96`, `post_iter2_skill_diff.json:29-36`, `post_iter2_skill_diff.json:60-67`, `post_iter2_skill_diff.json:91-98`).

## C10 - Most-Likely-To-Succeed Path

If only four of M8-M15 survive, keep M8, M9, M11, and M13.

- Keep M8 because the project needs the first divergent operator in the production composition before more fixes are guessed.
- Keep M9 because first-hour evidence still points hardest at radiation/surface/PBL and near-surface diagnostics (`worker-report.md:7-11`).
- Keep M11 because hourly replay is explicitly not prognostic land, and the binding goal is a JAX-native WRF port, not a replay-assisted postprocessor (`worker-report.md:143-145`).
- Keep M13 because without an ensemble corpus the project cannot know whether any fix generalized; current one-case and two-complete-case evidence is insufficient (`canary_multiday_skill.json:678-715`).

Drop M10 as a standalone milestone but fold limiter replacement/clip accounting into M9 and M11 acceptance. Drop M12 unless microphysics becomes first-divergence evidence. Drop M14/M15 as separate milestones because they are gates/outcomes; their checks should close M13 and the final release, not consume roadmap slots.

## Handoff

- objective: adversarial critique of `.agent/decisions/PROJECT-RESET-PLAN-DRAFT.md` against C1-C10.
- files changed: `.agent/sprints/2026-05-28-project-reset-critic/critique.md`.
- commands run: initial unpinned orientation commands before reading the CPU-pinning rule: `pwd && rg --files ...`, `git status --short --branch`; then pinned reads with `taskset -c 0-3` for the constitution, AGENTS, sprint contract, local review skill, manager draft, M7 amendment, iter2 skill/speed/invariant proofs, Opus and Codex RCA artifacts, operational-mode and physics-coupler source, ADR-027, actual savepoint tests/proofs, daily pipeline, land state, boundary apply, state contract, surface-layer tests/source, publication-test aggregate/conservation/corpus proofs, and final git status; attempted `taskset -c 0-3 git add .agent/sprints/2026-05-28-project-reset-critic/critique.md`, which failed because the Git worktree metadata is outside the writable sandbox.
- proof objects produced: this critique file.
- unresolved risks: no GPU/runtime validation was run by contract; critique depends on committed proof JSONs and source reads only.
- next decision needed: manager should merge this with the blinded plan and rewrite the final roadmap around first-divergence RCA, missing validation milestones, and corrected risk/timeline.

CRITIQUE_COMPLETE - The draft is directionally right to freeze publication and require statistical skill, but it underrates RCA uncertainty, validation gaps, milestone risk, and timeline by enough that the final plan should be rewritten before dispatching M8.
