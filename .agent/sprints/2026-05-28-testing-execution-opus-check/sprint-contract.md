# Sprint Contract — Sprint #4: Testing-Execution Opus Check + Publishability Decision

**Sprint ID**: `2026-05-28-testing-execution-opus-check`
**Created**: 2026-05-28 (sprint #4 in the publication pipeline)
**Status**: READY
**Predecessors**:
- `.agent/sprints/2026-05-27-testing-plan-execution-redo/aggregate_report.md` (EXECUTION_PARTIAL, merged at `f7222b0`)
- `.agent/sprints/2026-05-27-testing-plan-critique/test_plan_revised.md` (the reference plan)
- `.agent/sprints/2026-05-28-gpu-wrf-history-research/novelty_bounds.md` (defensible novelty bound)
- `.agent/decisions/PAPER-REWRITE-FRAMING-MEMO.md` (editorial direction)

## Objective

Opus reads all of Sprint #3 RE-DO's proof objects and renders a binding **publishability decision**:

- **PUBLISHABLE_AS_IS**: the evidence is enough for an honest v0.0.1 release and the accompanying preprint, with gaps documented openly. Sprint #5 (paper rewrite) fires next.
- **PUBLISHABLE_WITH_NARROW_PATCH**: the evidence is mostly sufficient but one or two specific gaps should be closed before release. Name them precisely and decide whether the patch goes before or after the paper rewrite.
- **DEFER_PUBLICATION**: the evidence does not justify a v0.0.1 release; specific blocker sprints must run first. Name them.

The decision is binding for the v0.0.1 timeline. The principal's stated intent is "finish this perfectly clean now" — opus should weigh that against the honest gaps without sliding into either rubber-stamp or perfectionism.

## Acceptance

- **AC1 — Per-test verdict review**: read each proof object in `.agent/sprints/2026-05-27-testing-plan-execution-redo/*.json`. For each: confirm the verdict is honestly characterised, evidence on disk matches the claim, and threshold comparison is correct. Emit `.agent/sprints/2026-05-28-testing-execution-opus-check/per_test_review.md`.

- **AC2 — Skipped/failed test triage**: for each SKIP_* or FAIL verdict, decide:
  - **MUST FIX before v0.0.1** — publication-blocking; cannot ship without
  - **DOCUMENT as known gap** — paper acknowledges; future work
  - **OUT_OF_SCOPE for v0.0.1** — never intended in this release
  Emit `.agent/sprints/2026-05-28-testing-execution-opus-check/skip_fail_triage.md` with the rationale per item.

- **AC3 — Honest-framing audit**: cross-check the verdicts against the novelty bound from `novelty_bounds.md`. The paper's claim is "first full open-source JAX/Python WRF v4 port with whole-state device residency on a consumer-grade workstation GPU". Which tests are NECESSARY to defend that claim? Which are nice-to-have? Confirm the necessary set is either PASS or honestly characterisable as known gap.

- **AC4 — Publishability decision**: write `.agent/sprints/2026-05-28-testing-execution-opus-check/publishability_decision.md` with the verdict (PUBLISHABLE_AS_IS / PUBLISHABLE_WITH_NARROW_PATCH / DEFER_PUBLICATION), the rationale, the specific must-do list if any, and the precise wording the paper Limitations section should use for each acknowledged gap.

- **AC5 — Paper-rewrite input**: produce `.agent/sprints/2026-05-28-testing-execution-opus-check/paper_rewrite_input.md` — a tight summary the sprint #5 worker can lift into the Results + Limitations sections without re-judging the evidence. Include: the exact sentences for Results (DETERMINISM PASS, GPU execution proof, Canary 3-day skill table), the exact sentences for Limitations (idealized-runner gap, conservation evidence gap, 5→3 Canary days, savepoint depth 100 vs requested 1000/10000), and the precise wording for "what this v0.0.1 release does NOT claim."

- **AC6 — Tester report**: with decision token.

## Files Tester May Modify

- `.agent/sprints/2026-05-28-testing-execution-opus-check/**` only

## Hard Rules

1. **No code changes.** Pure judgement + writing.
2. **No GPU runtime.**
3. **CPU pinning**: `taskset -c 0-3`.
4. **No fabricated evidence.** If a proof object does not exist or the verdict is wrong, say so.
5. **Honest verdict.** This sprint either green-lights v0.0.1 or names what's blocking. Wishful thinking is unacceptable.
6. **No remote push.** Local commit on `tester/opus/testing-execution-opus-check` only.

## Dispatch

- Tester: claude opus 4.7 xhigh
- Wall-time: 1-3 h
- Branch: `tester/opus/testing-execution-opus-check`
- Worktree: `/tmp/wrf_gpu2_op4check`
- GPU usage: NONE
