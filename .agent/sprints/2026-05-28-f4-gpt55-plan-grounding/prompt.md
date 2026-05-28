# GPT-5.5 Plan Grounding — Critique the 4-Front Strategy

**Worker**: codex gpt-5.5 xhigh (critic mode)
**Wall-time**: 2-4 hours (analysis only)
**No code changes.**

## Context

Project just discovered (via Gemini agy review, see
`.agent/sprints/2026-05-28-agy-dycore-deep-review/findings.md`) that:

1. The "100-step bitwise WRF parity" test (`tests/savepoint/test_dycore_100_steps.py`
   via `scripts/m6b6_coupled_step_compare.py`) is a **JAX-vs-JAX self-compare
   tautology** — has been passing while the dycore was structurally broken.
2. THREE coordinated dycore bugs survived because of this:
   - advection deleted from operational `_rk_scan_step`
   - mu in acoustic substep set to delta not total
   - theta decouple uses running not stage-start state
3. agy's coordinated fix REGRESSED (NaN at step 12 vs baseline step 93).

Manager is now dispatching 4 PARALLEL fronts:

- **F1**: Rewrite m6b6 comparator to compare JAX vs **real WRF Fortran savepoints** (instrument WRF, dump, compare). 3-5 days. Codex GPT-5.5.
- **F2**: Add **idealized cases** (Skamarock warm bubble + Straka density current) as community-standard dycore validation. 3-5 days. Codex GPT-5.5.
- **F3**: agy follow-up consult — is M11.3 salvageable with a 4th fix, or do we need integrated RK3 cadence restructure, or restart from stencil-bakeoff? Analysis only.
- **F4**: This sprint — critique the strategy itself.

## What I need from you

Read these files NOW with care:

1. `.agent/decisions/ADR-028-PROJECT-RESET-2026-05-28.md` — the project reset binding goal
2. `.agent/decisions/PROJECT-RESET-PLAN-FINAL.md` — the multi-amended plan
3. `.agent/sprints/2026-05-28-agy-dycore-deep-review/findings.md` — agy's review
4. `.agent/sprints/2026-05-28-f1-wrf-fortran-savepoint-comparator/sprint-contract.md` — F1 contract
5. `.agent/sprints/2026-05-28-f2-idealized-cases-skamarock-straka/sprint-contract.md` — F2 contract
6. `.agent/sprints/2026-05-28-f3-agy-architecture-followup/prompt.md` — F3 prompt
7. `scripts/m6b6_coupled_step_compare.py` — the tautological comparator
8. `src/gpuwrf/runtime/operational_mode.py` (`_rk_scan_step`)
9. `src/gpuwrf/dynamics/core/acoustic.py`
10. Memory: `/home/enric/.claude/projects/-home-enric-src-wrf-gpu2/memory/project_b6_self_compare_tautology_2026_05_28.md`
11. Memory: `/home/enric/.claude/projects/-home-enric-src-wrf-gpu2/memory/project_reset_2026_05_28.md`

Then argue against the strategy — be honest, no diplomatic hedging.

### Question 1 — Is F1 the right priority?

Is rewriting the comparator the most valuable next move, or are there CHEAPER tests that would expose the same bugs?

- Could we just dump a few JAX states at e.g. step 1, 5, 25, 100 and compare to WRF Fortran wrfout output at those steps (which we already have)? Avoids WRF instrumentation.
- Could we use the existing M2 stencil-bakeoff kernel-level tests as the WRF-comparison floor?
- Are we over-engineering this?

### Question 2 — Is F2 worth it?

Skamarock and Straka are dycore-correctness floors but the bugs we have are coupling/state-management bugs, not numerical-method bugs. Will the idealized cases actually trigger our 3 bugs, or could they pass while the operational case still fails?

- Specifically: does the warm bubble use mu+mu_save in a way that would expose the `mu=mu_delta` bug?
- Does the density current excite theta-coupling in a way that would expose the decouple bug?
- Does either case test advection in a way that would expose the missing advection in operational_mode?

### Question 3 — Should F3 + F4 even run before F1+F2?

If F1+F2 take 3-5 days each, F3+F4 are 1-day analyses, should the analyses block the implementations? Or run truly in parallel?

### Question 4 — What are we MISSING?

What test/sprint/insight would you add that the 4-front list doesn't cover? Specifically:

- Is there a "first principles" check we can do in 1 day that would tell us with high confidence whether the dycore architecture is salvageable?
- Should we be reading WRF Fortran for first-principles understanding before more JAX changes?
- Is there a sanity check on the M9.C boundary placeholder that we've ignored?

### Question 5 — Honest verdict on the manager's strategy

Score the 4-front strategy 0-10 with one-sentence rationale. If <7, what would you do differently?

## Deliverable

Write `.agent/sprints/2026-05-28-f4-gpt55-plan-grounding/critique.md` with answers to Q1-Q5.

End with `PLAN_CRITIQUE_COMPLETE`.

## Hard rules

- CPU pinning: `taskset -c 0-3`.
- No model code changes — analysis only.
- No remote push.
- Manager repo only.
- Auto-notify on exit: `tmux send-keys -t 0 "AGENT REPORT: f4-critic DONE exit=$?" Enter`.
