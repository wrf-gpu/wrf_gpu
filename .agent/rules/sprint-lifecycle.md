# Sprint Lifecycle

0. Manager opens the milestone with a milestone plan and review before any implementation sprint starts.
1. Manager creates sprint folder from template.
2. Manager writes sprint contract: objective, scope, file ownership, acceptance criteria, validation commands, and proof object.
3. Reviewer checks the sprint contract before code starts when the sprint changes model, validation, architecture, or performance behavior.
4. Worker implements only the contracted work.
5. Tester runs independent checks or creates missing tests.
6. Reviewer challenges correctness, performance, and scope.
7. Manager closes with decision, artifacts, risks, and memory-patch proposal.

A sprint without a proof object stays open.

## Double-AI principle (HARD RULE, added 2026-05-21 ~01:00 per user directive)

Every sprint that produces production code OR modifies governance (ADRs, skills, rules, contracts) MUST have at least one independent Claude Opus 4.7 review pass before manager close, even under bigger-steps autonomy. Manager (Claude Opus) reading the worker report directly is NOT sufficient as the second AI — the reviewer must be a separately-dispatched fresh-context Opus instance with its own role prompt.

Rationale: catches major bugs, hallucinations, and skipped work that a single-AI workflow (or self-review by the same manager Opus instance) misses systematically. Codex worker + Opus reviewer is the minimum diligence floor for important sprints. The 2026-05-20 M5-S1 cycle proved this (Opus reviewer R-1 caught Gemini's CGG11 hallucination that worker + manager both missed); the M5-S2 cycle initially skipped this and required retroactive review to catch a deferred anti-tautology gap.

Exemptions (Opus review optional, manager may close on own authority):
- Scout / research / planning sprints (no production code shipped).
- Mechanical follow-up sprints where the reviewer has already specified exact fixes in the prior cycle's report (the prior reviewer's spec IS the second-AI check).
- Pure-documentation sprints with no code or governance changes.

Non-exemptions (Opus review REQUIRED):
- Any sprint touching `src/gpuwrf/**/*.py` production code.
- Any sprint creating or amending an ADR.
- Any sprint touching `.agent/skills/**/SKILL.md` or `.agent/rules/**`.
- Any sprint claiming new physics, performance, or precision results.
- Any milestone closeout (the closeout itself, not the underlying sprints).

If the manager closes a non-exempt sprint without the Opus review pass, the close is provisional and must be reopened for retroactive review (as happened to M5-S2 on 2026-05-21 ~01:00).
