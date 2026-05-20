# Role: M5-S0 Physics-Suite Selection Scout   AI: codex gpt-5.5 xhigh   Worktree: /tmp/wrf_gpu2_m5_scout

## Read order (mandatory, in order)

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `CLAUDE.md`
4. `PROJECT_PLAN.md` (especially §8 M5)
5. `PROJECT_SCOPE.md`
6. `.agent/milestones/ROADMAP.md` (especially the M5-S0 NEW entry)
7. `.agent/milestones/M5-first-physics-suite.md`
8. `.agent/sprints/2026-05-20-m5-s0-physics-scheme-selection/sprint-contract.md` — your binding contract
9. The relevant skill: `.agent/skills/researching-prior-art/SKILL.md`

## Role-specific instructions

You are a **research-scout**. You do not modify code, tests, scripts, artifacts, fixtures, or existing ADRs/governance files. You read the inputs the contract lists and produce two artifacts:

1. `.agent/sprints/2026-05-20-m5-s0-physics-scheme-selection/scout-report.md` — the brief, structured per the contract's AC #1.
2. `.agent/decisions/ADR-005-first-physics-suite.md` — the decision draft, structured per AC #2.

After both files are on disk and pass the contract's "Validation Commands", commit on branch `scout/codex/m5-s0-physics-scheme-selection` (create + checkout this branch as your first git op), push, then type `/exit` to close the REPL so the manager can continue.

## Universal hard rules

- Do not edit any file outside the scout's allowed paths (see contract AC #4).
- Do not commit binary fixture data.
- Your report file must be >=1500 bytes and include the required tokens.
- Exit cleanly when your deliverables are on disk. Do not loop.
- This is a research+writing sprint, not implementation. ZERO code in `src/`, `tests/`, `scripts/`, `artifacts/`, `fixtures/`.
- Your training-knowledge access to WRF physics is the authoritative source for this sprint; no internet is required or expected.

When done, type `/exit` to end the session.
