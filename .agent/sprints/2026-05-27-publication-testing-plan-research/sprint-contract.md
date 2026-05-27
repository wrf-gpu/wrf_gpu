# Sprint Contract — Publication Testing Plan Research (Codex GPT-5.5)

**Sprint ID**: `2026-05-27-publication-testing-plan-research`
**Created**: 2026-05-27 (user direction: shift paper focus to "first open-source GPU port of WRF"; need community-grade testing plan)
**Status**: READY
**Predecessors**:
- `publish/paper/paper.md` (current draft, Canary-focused)
- `.agent/decisions/MILESTONE-M7-CLOSEOUT.md` + `MILESTONE-M7-CLOSEOUT-AMENDMENT.md`
- `.agent/sprints/2026-05-27-gemini-agy-review/review.md` (Gemini's recommendations)

## Strategic Context

The user has shifted the paper's center of gravity. The achievement is **the first open-source JAX/Python GPU port of WRF**, not "a fast Canary forecast". Canary is one example workload. The paper must position the port as a community asset and present rigorous testing evidence that satisfies the meteorological scientific community's acceptance criteria for new NWP implementations.

This sprint is **research and planning only** — produce the plan that the next codex sprint will execute. No model code is run, no fresh measurements taken. The plan must be specific, actionable, and based on what the meteorological community actually demands for accepting a new NWP code as proven and valued.

## Objective

Research what the meteorological / atmospheric-science community expects of a new NWP implementation to accept it as proven and useful. Synthesize that into a concrete, executable test plan for wrf_gpu — covering idealized cases, standard benchmarks, conservation laws, multi-regime evaluation, reproducibility, documentation, and public access.

The plan output must be deep enough that the execution sprint can take it and run it without needing further research. Plan must be honest about what we already have (3 V3 ICs, 1 day side-by-side AEMET, 22.26× speedup, D2H=0, restart bitwise, B6 savepoint parity) and what's missing.

## Acceptance

- **AC1 — Community acceptance criteria**: produce `.agent/sprints/2026-05-27-publication-testing-plan-research/community_acceptance_criteria.md` documenting what the WRF/NWP community looks for when reviewing a new dycore/physics implementation. Cover: idealized test cases (warm bubble, density current, mountain wave, baroclinic wave), standard benchmarks (NCAR WRF test suite, dynamical core test cases), conservation properties (mass, energy, momentum, water-vapour budget), stability margins (CFL, semi-implicit), multi-regime evaluation (continental, marine, tropical, mid-latitude, mountainous), forecast verification (operational metrics + neighbourhood/object-based for precip), reproducibility (cross-hardware, restart, ensemble), independent-review access (public repo, code documentation, tutorials, installation), licensing/citation expectations. Cite sources from the deep-research brief at `publication/research_brief/english_brief.txt`.

- **AC2 — Gap analysis vs current state**: produce `.agent/sprints/2026-05-27-publication-testing-plan-research/gap_analysis.md` listing what wrf_gpu already has on disk (proof objects + tests) vs what AC1 demands. Mark each item: HAVE, PARTIAL, MISSING, OUT_OF_SCOPE_V0.

- **AC3 — Executable test plan**: produce `.agent/sprints/2026-05-27-publication-testing-plan-research/test_plan.md` with each test scoped as a discrete actionable item:
  - **Test ID** (e.g., `IDEALIZED-WARMBUBBLE`, `CONSERVATION-MASS-24H`, `BENCHMARK-WRF-DENSITY-CURRENT`)
  - **What it proves** (one sentence)
  - **Inputs needed** (IC, BC, namelist, reference data)
  - **How to run it** (commands, scripts to write)
  - **Pass/fail criteria** (quantitative threshold)
  - **Proof object** (path under `.agent/sprints/<execution-sprint>/`)
  - **Estimated wall-time + GPU budget**
  - **Priority** (HIGH = required for arXiv submission; MEDIUM = strengthens paper; LOW = future work)

  At minimum the plan should include: 3 idealized cases (warm bubble, density current, mountain wave), 2 conservation tests (mass + energy budget over 24h), 2 standard-benchmark comparisons (vs published WRF reference), 1 multi-day Canary ensemble (extending the current 1-day side-by-side), 1 cross-hardware reproducibility test (if feasible on a single machine, otherwise note as future work), documentation checklist (README, INSTALL, CITATION, LICENSE, CONTRIBUTING, tutorial notebooks), public-release checklist.

- **AC4 — Citations**: every claim about "the community expects X" should cite a source — either from the research brief, a WRF tutorial, an NCAR tech note, a recent NWP-comparison paper, or a relevant arXiv/journal article. No fabricated citations. If a citation is needed but unverified, mark `[verify before use]`.

- **AC5 — Cost estimate**: total wall-time + GPU-hours estimate for executing the HIGH-priority items. Aim to keep total under 24h GPU-time so it can be executed overnight.

- **AC6 — Public-access plan**: distinct section in `test_plan.md` covering what the public release needs (repo structure, license, citation, documentation, example notebooks, CI) so that "open source" is a meaningful claim, not just a code dump.

- **AC7 — Worker report** with verdict `PLAN_READY` and a 5-line summary of priorities.

## Files Worker May Modify

- `.agent/sprints/2026-05-27-publication-testing-plan-research/**`

## Files Worker Must Not Modify

- `publication/draft/**` — paper untouched
- `publish/**` — staging untouched
- `src/gpuwrf/**` — no code changes
- governance files
- `/mnt/data/canairy_meteo/**`

## Hard Rules

1. **Research + plan only**. No code, no measurements, no test execution.
2. **Honest about what we have**. Read the existing proof objects + test files + closeout docs before claiming "missing".
3. **CPU pinning**: `taskset -c 0-3`.
4. **No remote push.** Local commit on `worker/gpt/publication-testing-plan-research` only.
5. **No fabricated citations.**

## Proof Objects

- `.agent/sprints/2026-05-27-publication-testing-plan-research/community_acceptance_criteria.md`
- `.agent/sprints/2026-05-27-publication-testing-plan-research/gap_analysis.md`
- `.agent/sprints/2026-05-27-publication-testing-plan-research/test_plan.md`
- `.agent/sprints/2026-05-27-publication-testing-plan-research/worker-report.md`

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Wall-time: 2-4 h
- Branch: `worker/gpt/publication-testing-plan-research`
- Worktree: `/tmp/wrf_gpu2_testplan`
- GPU usage: NONE
