# Sprint Contract: V0.14 Debug Method Critic

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Reviewer: Claude Opus xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Independently critique the manager's current debug process, evidence chain, and
next-step plan for the v0.14 grid-field divergence work. The deliverable should
help accelerate the fault search and reduce risk of false conclusions.

## Non-Goals

- No production source edits.
- No proof script edits.
- No TOST.
- No Switzerland validation.
- No FP32 implementation.
- No GPU job unless explicitly needed for a one-command status check; default
  is CPU/read-only.
- No Hermes, Telegram, or `ask-hermes`.

## Inputs

- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/skills/managing-sprints/SKILL.md`
- `.agent/skills/conducting-blind-review/SKILL.md`
- `.agent/skills/resolving-cross-model-disagreements/SKILL.md`
- `.agent/decisions/V0140-VALIDATION-PLAN.md`
- `.agent/decisions/V0140-MEMORY-FIX-ROADMAP.md`
- `.agent/decisions/V0140-FP32-ACOUSTIC-ROADMAP.md`
- `proofs/v014/live_nest_base_hook.json`
- `proofs/v014/base_state_split_fix.json`
- `proofs/v014/earlier_source_bisect.json`
- latest v0.14 sprint closeouts under `.agent/sprints/2026-06-09-v014-*`

## Write Scope

- `.agent/reviews/2026-06-09-v014-debug-method-critic.md`

No other repository files may be edited.

## Required Work

1. Review whether the current conclusion is justified:
   live-nest initialization is the next source-fix target for the d02 base split.
2. Identify any weak assumptions, missing falsifiers, or cheaper discriminating
   tests that should run before or during the source sprint.
3. Critique the manager process: agent use, poll cadence, context hygiene,
   evidence quality, sprint boundaries, and performance-protection rules.
4. Suggest concrete accelerators for grid-cell parity debugging that preserve
   GPU-native performance and do not introduce CPU-WRF production shortcuts.
5. Give a compact prioritized action list: immediate, next-after-fix, and
   stop/avoid.

## Acceptance Criteria

- Findings lead the report.
- Every finding cites a local proof/doc/source file or explicitly says it is an
  inference.
- The report is concise enough for manager context use.
- It includes at least one possible alternative hypothesis or falsifier.
- It clearly distinguishes validation-oracle use of CPU-WRF from production
  dependencies.

## Validation

No code validation is required. The worker must write the review file and then
notify the manager tmux window:

```bash
tmux send-keys -t 0:2 'CLAUDE DEBUG_METHOD_CRITIC DONE - see .agent/reviews/2026-06-09-v014-debug-method-critic.md' Enter
```

## Closeout

Manager decides whether to adopt recommendations into the active source sprint,
roadmap, or a follow-up proof sprint.
