# Tester Report — Testing-Plan Critique

**Sprint**: `2026-05-27-testing-plan-critique`
**Role**: tester (Claude Opus 4.7 acting as sonnet-test-engineer)
**Branch**: `tester/opus/testing-plan-critique`
**Worktree**: `/tmp/wrf_gpu2_testcrit`
**Predecessor**: `2026-05-27-publication-testing-plan-research` merged at `5f5da89`.

## Summary

Decision: **PLAN_REVISED**. The predecessor testing plan is structurally sound (priority tiering, proof-object discipline, 24 GPU-hour budget, alignment with the paper strategic-framing memo) but needs material revisions to thresholds, missing tests, redundant tests, execution feasibility, and the Open-Source Release Plan. The revised plan in `test_plan_revised.md` is what the next sprint should execute against.

## Deliverables Produced

- `plan_critique.md` — full critique covering AC1 (per-section), AC2 (threshold revision table — 17 metrics reviewed; KEEP / TIGHTEN / LOOSEN / REPLACE / REMOVE / ADD decision per row), AC3 (tests to add/remove/defer/reframe), AC4 (execution feasibility per HIGH-priority item against actual repo state), AC6 (release-plan critique with 10 gap items).
- `test_plan_revised.md` — drop-in revised plan ready for sprint #3 (execution). 13 HIGH/MEDIUM/LOW test items, revised pass/fail thresholds, anchored to published references (Bryan & Fritsch 2002, Straka 1993, Schaer 2002), reuse-before-rewrite execution order, 17.4 GPU-hour HIGH budget under the 24 GPU-hour cap.
- This report.

## Tests Added or Run

This sprint is **critique + revised plan only**, per the sprint-contract hard rule "No code changes / No new tests under `tests/`." No tests were authored or executed. The contract is satisfied by the two planning artifacts.

## Fixtures and References Used

- Code surveyed (read-only): `scripts/`, `src/gpuwrf/{dynamics,validation,io,integration,runtime,fixtures}/`, `tests/` filenames, `artifacts/m6/`.
- Data inventory verified on disk: `/mnt/data/canairy_meteo/runs/wrf_l3/` (34 wrf_l3 daily runs, 20260428 → 20260525; ~28 days continuous coverage — more than the original plan's "7–10 retained cases" assumed).
- CPU WRF binary verified: `/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe` (real-data build; **cannot run idealized namelists** — flagged as previously under-acknowledged execution cost).
- Published references cited in revised plan: Bryan & Fritsch 2002 (warm bubble), Straka et al. 1993 (density current), Schaer et al. 2002 (mountain wave), Skamarock 2004 (energy-budget framing).
- Anchor docs read: `PROJECT_CONSTITUTION.md`, `AGENTS.md`, `PROJECT_PLAN.md`, `MILESTONES.md`, `.agent/milestones/ROADMAP.md`, `.agent/goals/M1-DONE.md`, `VALIDATION_STRATEGY.md`, `PRECISION_POLICY.md`, `.agent/decisions/PAPER-STRATEGIC-FRAMING.md`.

## Key Findings

1. **Three idealized cases under-anchored to published references**: the original plan named warm-bubble / density-current / mountain-wave but did not cite a specific benchmark per case. Reviewer-grade testing requires citation per case.
2. **Mountain-wave coverage is weak**: original plan picks `em_hill2d_x` (small bell-shaped hill); the published rigorous test is Schaer 2002 sinusoidal terrain with an analytic linear-regime oracle. Revised plan promotes Schaer to primary; keeps `em_hill2d_x` as a smoke for stock-WRF-binary provenance.
3. **Warm-bubble nRMSE ≤ 0.05 at every lead is too tight**: warm-bubble convection is chaotic by ~20 min; revised plan introduces a laddered threshold (0.05 @ 5 min → 0.18 @ 30 min).
4. **Closed-domain energy threshold ≤ 0.1% absolute is not physically defensible**: ARW is not formally total-energy-conserving (Skamarock 2004). Revised plan replaces with a Tier-4-style CPU envelope (GPU drift within ±20% of CPU drift on same configuration), consistent with `VALIDATION_STRATEGY.md` and `MILESTONES.md` M6c framing.
5. **BENCHMARK-WRF-STOCK-IDEAL is redundant** with the three idealized cases; revised plan merges it into a stock-WRF provenance requirement on each idealized case.
6. **Six tests added**: STABILITY-CFL-SWEEP (HIGH), STABILITY-ACOUSTIC-SUBSTEP-SWEEP (HIGH), DETERMINISM-REPEAT (HIGH), SAVEPOINT-PARITY-DEEP (HIGH), COMPILE-COLD-START-TIME (MEDIUM), VRAM-FOOTPRINT-1KM-FRESH (MEDIUM). These cover stability margins (community-acceptance §4), full-pipeline determinism, depth of bitwise parity, and *measured* substantiation of pillar-2 claims.
7. **Canary multi-day corpus**: original plan budgets "7–10 cases"; on-disk inventory has 34 days, so revised plan tightens to "≥ 14 continuous days" at no additional GPU cost. Per-variable pass/fail introduced (was aggregate-only) to honestly surface the known T2 regression.
8. **Execution-cost gap**: WRF idealized cases need `compile em_<case>` recompile of `ideal.exe + wrf.exe` per case. Original plan said "Write `scripts/pubtest_run_wrf_reference.py`" without flagging the ~1 h compile + namelist debug cost per case. Revised plan calls this out explicitly and prefers analytic references (Schaer, Straka) where available to avoid recompile cost on 2 of 3 idealized cases.
9. **Reuse-before-rewrite is under-emphasized in original plan**: `scripts/diagnostic_conservation_tracker.py` already implements mass + KE + dry-static-energy totals; `src/gpuwrf/validation/forecast_vs_obs.py:467` already implements FSS; `m7_daily_pipeline.py` and `m7_gpu_vs_cpu_skill_diff.py` already implement the per-case pipeline. Revised plan makes the reuse explicit so the execution sprint does not rewrite.
10. **Release plan gaps (10 items)**: DOI minting via Zenodo, Software Heritage SWHID, reviewer 5-minute test drive, signed checksums, pip-audit, coverage report, AI_USE.md disclosure, per-dataset data-availability split, hardware reproducibility statement, issue triage SLA.

## Gaps Identified

- No analytic IC builders for warm bubble / density current / Schaer mountain currently exist under `src/gpuwrf/fixtures/`. Execution sprint must write them.
- No `LICENSE`, `CITATION.cff`, or `INSTALL.md` in the repo today. License choice is a **human-owner decision**; execution sprint must flag for the user, not pick.
- No Zenodo or Software Heritage integration set up; release sprint will need to perform a one-time configuration.
- The existing `m6_warm_bubble_test.py` is a failure-diagnostic operator-sanity probe and should **not** be used as the starting point for the publication warm-bubble case (the plan does not say to use it, but a naive reader of the plan might).
- No SAL implementation in `forecast_vs_obs.py` (FSS is there, SAL is not). Revised plan marks SAL as deferred if implementation cost exceeds remaining MEDIUM budget.

## Risks and Cautions for the Execution Sprint

- **WRF idealized recompile risk**: build environment drift could block the warm-bubble CPU reference. Mitigation: revised plan recommends pinning the WRF source commit + compile command in `proof.wrf_provenance` for each case; consider a containerised WRF build environment.
- **Skill regression is real**: the iter2 Canary skill is materially worse than CPU per the strategic-framing memo's honest admission. The revised per-variable Canary threshold (±20% of CPU RMSE) **will fail** on T2 today. The paper's honest framing accepts this; the execution sprint must not paper over it.
- **24 GPU-hour cap is tight if stability and savepoint-deep tests both run**: revised total HIGH budget is 17.4 GPU-hours with 6.6 GPU-hours rerun reserve; the reserve is necessary because Canary 14-day reruns at 4 GPU-hours have no slack inside a single overnight window if a case fails midway.
- **Determinism**: cuDNN and some JAX/XLA kernels are not deterministic by default on Blackwell; the DETERMINISM-REPEAT test may need `XLA_FLAGS` and `JAX_DETERMINISTIC_OPS` configuration. If determinism cannot be achieved, the proof must bound the non-determinism, not silently pass.

## Files Changed in This Sprint

- `.agent/sprints/2026-05-27-testing-plan-critique/plan_critique.md` (new)
- `.agent/sprints/2026-05-27-testing-plan-critique/test_plan_revised.md` (new)
- `.agent/sprints/2026-05-27-testing-plan-critique/tester-report.md` (this file; replaces template)

No code under `src/`, `scripts/`, or `tests/` was modified. No fixtures or binary data committed. No remote push.

## Handoff

- **Objective**: critique predecessor testing plan; produce revised executable plan.
- **Files changed**: three files above, all inside the sprint directory.
- **Commands run**: read-only repo + data inventory; no GPU runtime; no test execution.
- **Proof objects produced**: `plan_critique.md`, `test_plan_revised.md`, `tester-report.md`.
- **Unresolved risks**: WRF idealized recompile cost, T2 skill regression will fail revised gate, LICENSE choice pending human-owner decision, determinism may need JAX/XLA config.
- **Next decision needed**: manager dispatches sprint #3 (execution) against `test_plan_revised.md`, scope-frozen at the revised HIGH list (10 items + release checklist). Confirm LICENSE choice and Zenodo setup before the release audit step.

Decision: **PLAN_REVISED** — proceed to execution against `test_plan_revised.md`.
