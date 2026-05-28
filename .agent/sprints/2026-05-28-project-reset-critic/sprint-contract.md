# Sprint Contract — Project-Reset Plan Critic (codex GPT-5.5 xhigh)

**Sprint ID**: `2026-05-28-project-reset-critic`
**Role**: ADVERSARIAL CRITIC (frontrunner-position-attack)
**Worker**: codex gpt-5.5 xhigh
**Branch**: `worker/gpt/project-reset-critic`
**Worktree**: `/tmp/wrf_gpu2_pr_critic`
**Wall-time**: 60-90 min
**GPU usage**: NONE — pure analysis sprint

## Binding goal (top of every contract from now on)

> A JAX-native GPU port of WRF v4 that delivers Canary L2/L3 forecasts whose 24-72 h RMSE on T2, U10, V10 is not statistically distinguishable from CPU WRF v4 (paired t-test p > 0.05 on a ≥ 15-case ensemble across seasons), while preserving ≥ 10× speedup vs 28-rank CPU WRF on the same workstation.

## Objective

Attack the manager's draft project-reset plan at `.agent/decisions/PROJECT-RESET-PLAN-DRAFT.md`. Your role is **adversarial** — assume the manager has missed important things and find them. The goal is to make the merged plan better, not to validate the manager.

## Required inputs (read in order)

1. `.agent/decisions/PROJECT-RESET-PLAN-DRAFT.md` — manager's draft (the thing you are attacking)
2. `.agent/decisions/MILESTONE-M7-CLOSEOUT-AMENDMENT.md` — 156→22.26× correction context
3. `.agent/sprints/2026-05-27-m7-skill-fix-iter2/post_iter2_skill_diff.json` — last skill measurement
4. `.agent/sprints/2026-05-27-m7-skill-regression-rca-opus/top_3_suspects.md` — last RCA
5. `PROJECT_CONSTITUTION.md`, `AGENTS.md`
6. `src/gpuwrf/operational_mode.py` (head + tail), `src/gpuwrf/coupling/physics_couplers.py` (head + tail)
7. `.agent/decisions/ADR-027*.md` — D2H-zero invariant
8. `tests/savepoint/` — what B6 actually tests

## Acceptance

Produce `.agent/sprints/2026-05-28-project-reset-critic/critique.md` answering EACH of these explicitly:

- **C1 — Hidden assumptions**: list every load-bearing assumption the draft makes that isn't proven. Rate each by how badly the plan breaks if the assumption is wrong (1 = nuisance, 5 = whole roadmap collapses).

- **C2 — Milestone ordering**: is M8 → M9 → M10 → M11 → M12 → M13 → M14 → M15 the right order? Specifically: should M11 (Noah-MP) come earlier (because surface-flux fixes are pointless if land state is frozen)? Should M10 (theta limiter) come BEFORE M9 because the clip is artificially suppressing diurnal warming and skewing the surface-flux RCA? Argue both sides.

- **C3 — Risk re-assessment**: re-rate every milestone's risk. Where do you disagree with the draft? Where is "Low" actually "Medium" or "High"?

- **C4 — Δ% calibration**: is the +5/+15/+5/+20/+3/+10/+5/+5 distribution defensible? Where do you think the actual gain will land vs the draft's estimate?

- **C5 — Missing milestone**: identify at least one milestone the draft is missing (e.g. lateral BC re-verification, conservation budgets, ensemble validation, mass/energy/moisture closure, vertical-coordinate handling, p-grid baseline-state perturbation) and justify why.

- **C6 — Invariant ladder gaps**: does INV-1..6 actually catch the regressions that matter? Propose additions or modifications.

- **C7 — Cross-AI verification**: is "codex frontrunner + Opus tester+reviewer" sufficient for milestones this risky? Argue for tighter or looser oversight.

- **C8 — Timeline reality check**: is 17-23 weeks honest? Where is it under- or over-estimated? Be specific per milestone.

- **C9 — Strongest objection**: write the single sentence that, if the principal believed it, would force the manager to rewrite the plan from scratch.

- **C10 — Most-likely-to-succeed path**: if you could only keep 4 of M8-M15, which 4 and why?

## Hard rules

1. **No fluff.** Every claim cited to a file or measurement. No vague hedges.
2. **Be useful, not contrarian.** Disagree where you have grounds; agree where the draft is right (and say so).
3. **CPU pinning**: `taskset -c 0-3` on every command.
4. **No GPU runtime.** Reading + writing only.
5. **No code changes.** This is a critique sprint, not an implementation sprint.
6. **No remote push.** Local commit on `worker/gpt/project-reset-critic` only.
7. **Manager repo ONLY** — do not touch `/home/enric/src/wrf_gpu/` (the public repo).
8. **Honest verdict**: end the critique with one of `CRITIQUE_COMPLETE / CRITIQUE_PARTIAL` and a one-line summary.

## Files you may modify

- `.agent/sprints/2026-05-28-project-reset-critic/**`

## Files you must not modify

- Everything else. This is read-only outside your sprint directory.

## Dispatch

Spawn me. Read inputs. Write critique. Commit. Exit.
