# Role: critical-review (codex gpt-5.5 xhigh)   Decision: ADR-005 First Physics Suite

## Read order (mandatory, in order)

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `CLAUDE.md`
4. `PROJECT_PLAN.md` (especially §8 M5)
5. `PROJECT_SCOPE.md`
6. `.agent/milestones/ROADMAP.md` (M5 entry)
7. `.agent/milestones/M5-first-physics-suite.md`
8. `.agent/sprints/2026-05-20-m5-s0-physics-scheme-selection/sprint-contract.md` — the contract the scout followed
9. `.agent/sprints/2026-05-20-m5-s0-physics-scheme-selection/scout-report.md` — scout's brief
10. `.agent/decisions/ADR-005-first-physics-suite.md` — the decision draft (you are reviewing this)
11. `.agent/decisions/ADR-001-backend-selection.md` — the binding backend ADR (M5 gate thresholds live here)
12. `.agent/decisions/ADR-002-state-layout.md` — state contract
13. The relevant skill: `.agent/skills/resolving-cross-model-disagreements/SKILL.md`

## Role-specific instructions

You are an **independent senior reviewer** asked by the manager for a second opinion on the scout's ADR-005 decision. Read everything above, challenge the recommendation honestly, then write your verdict.

Write to `.agent/decisions/REVIEW-codex-ADR-005/critical-review.md`:

- **Decision**: one of `Accept` | `Accept with required fixes` | `Reject`
- **Top three structural concerns** with the ADR-005 / scout-report reasoning
- **Findings** (numbered, severity-ranked blocker/major/minor/note, each citing file:line)
- **Counter-proposals** if you would have picked a different scheme — name the scheme, the operational reason, and the JAX-implementability case
- **Closing recommendation**

Hard rules:
- Write ONLY `critical-review.md` in `.agent/decisions/REVIEW-codex-ADR-005/`. Read-only everywhere else.
- Do NOT modify the ADR-005 draft, the scout-report, the contract, or any code.
- Do NOT commit anything; manager handles git operations.
- Your report must be ≥1500 bytes and include the literal `Decision:` token.
- Be honest and adversarial. The cross-model gate exists to catch reasoning a single AI would miss. If the scheme choice is wrong for Canary operational needs, say so directly with evidence.

When done, type `/exit` to end the session so the manager can continue.
