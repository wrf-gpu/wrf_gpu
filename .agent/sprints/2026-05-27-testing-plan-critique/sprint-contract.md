# Sprint Contract — Testing-Plan Critique (Opus 4.7)

**Sprint ID**: `2026-05-27-testing-plan-critique`
**Created**: 2026-05-27 (sprint #2 in publication pipeline)
**Status**: READY
**Predecessor**: `.agent/sprints/2026-05-27-publication-testing-plan-research/` (PLAN_READY merged at `5f5da89`)

## Objective

Read the testing plan that Codex produced (`test_plan.md`, `community_acceptance_criteria.md`, `gap_analysis.md`) and critique it rigorously from the perspective of (a) what the meteorological community would actually accept as proof, (b) what's executable on the user's single-workstation RTX 5090 with existing repo infrastructure, and (c) whether the proposed pass/fail thresholds are scientifically defensible.

The goal is to produce a **revised, executable test plan** that the next codex sprint will run. If the original plan is good, the critique can be minor; if it overpromises, underpromises, picks the wrong cases, sets unrealistic thresholds, or misses key tests the community expects, the critique should fix it.

Also read the strategic-framing memo (`.agent/decisions/PAPER-STRATEGIC-FRAMING.md`) — the test plan must align with what the paper will claim, no more and no less.

## Acceptance

- **AC1 — Plan critique**: produce `.agent/sprints/2026-05-27-testing-plan-critique/plan_critique.md` covering:
  - Are the proposed idealized cases (warm bubble, density current, mountain wave) the right ones? Are stricter alternatives (e.g. baroclinic wave, Schaer mountain) more rigorous?
  - Are the pass/fail thresholds (normalised RMSE ≤ 0.05/0.10, conservation residuals ≤ 1e-10/1e-6/1e-5, etc.) physically defensible for a v0 port?
  - Are the GPU-budget estimates realistic given the iter-2 wall-clock baseline (~12 min/24h on 3 km)?
  - Does the plan cover what the WRF user community actually cares about (forecast verification, restart, conservation, idealised), or does it miss anything?
  - Is the "open source release plan" portion deep enough to claim "open source" credibly?
  - Does the test plan acknowledge that some tests will need WRF Fortran reference runs (CPU WRF binary already exists at `/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe`)?

- **AC2 — Threshold revision table**: for each test that has a quantitative threshold, decide: keep as-is, tighten, loosen, or replace. Justify each change with a short rationale.

- **AC3 — Test addition/removal**: name any tests that should be added (e.g. baroclinic wave for mid-latitude regime, Schaer mountain wave for terrain-following coordinate stress, idealized supercell for moist convection) or removed (if a proposed test is redundant or doesn't add evidence).

- **AC4 — Execution feasibility**: review each HIGH-priority item against the actual repo state (codex worktree may have missed some scripts that already exist). For each test: does the codebase already have the components to run it, or does the execution sprint need to write the components? Note any scripts that should be reused from `scripts/` rather than written fresh.

- **AC5 — Revised executable plan**: produce `.agent/sprints/2026-05-27-testing-plan-critique/test_plan_revised.md` — the merged/improved plan with the revisions from AC1-AC4 applied. This is what sprint #3 (execution) will use as its scope.

- **AC6 — Public release plan critique**: critique the "open source release plan" section. Does it cover license decision, repo URL, citation, CONTRIBUTING, install, tutorials, CI? What's missing for an actual community release?

- **AC7 — Tester report**: decision `PLAN_APPROVED` (no major changes needed, proceed to execution) / `PLAN_REVISED` (revised plan attached, proceed to execution against the revision) / `PLAN_BLOCKED` (the plan has gaps too big to execute; what's needed).

## Files Tester May Read

- All of `src/gpuwrf/**` (for execution-feasibility check)
- `scripts/**` (to identify reusable scripts)
- `tests/**`
- `.agent/sprints/2026-05-27-publication-testing-plan-research/**`
- `.agent/decisions/PAPER-STRATEGIC-FRAMING.md`
- `MILESTONES.md`, `VALIDATION_STRATEGY.md`, `PRECISION_POLICY.md`
- `/mnt/data/canairy_meteo/runs/wrf_l3/**` (Gen2 CPU WRF reference files, read-only)

## Files Tester May Modify

- `.agent/sprints/2026-05-27-testing-plan-critique/**` only

## Hard Rules

1. **No code changes.** Critique + revised plan only.
2. **No new tests under `tests/`.**
3. **CPU pinning**: `taskset -c 0-3`.
4. **No GPU runtime.**
5. **Honest BLOCKED** if the plan has a fundamental gap (e.g. requires a benchmark the project doesn't have on disk).
6. **No remote push.** Local commit on `tester/opus/testing-plan-critique` only.

## Dispatch

- Tester: claude opus 4.7 xhigh
- Wall-time: 1-3 h
- Branch: `tester/opus/testing-plan-critique`
- Worktree: `/tmp/wrf_gpu2_testcrit`
- GPU usage: NONE
