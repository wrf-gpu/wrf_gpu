# `.agent/` — AI development log & operating system

This directory is **two things at once**, and the distinction matters when you read it:

1. **A historical development log.** Most of what is here — `reviews/`,
   `sprints/`, `decisions/`, `dispatches/`, `reports/`, `notes/` — is a
   point-in-time record of how this project was actually built by a team of AI
   agents (an Opus-class manager + worker/critic/debugger models). It is
   committed deliberately, for **transparency**: a reader can see the real
   process, including dead ends, refuted hypotheses, and superseded conclusions.

2. **The operating system for those agents.** `PROJECT_CONSTITUTION.md`,
   `AGENTS.md`, the active sprint contract, `skills/`, and `rules/` are the
   live rules agents follow when working in this repo.

## ⚠️ Read this before quoting anything in here as "current truth"

The log is **append-mostly and point-in-time**. A review, sprint note, or
decision records what was believed **on its date**, with the evidence available
**then**. Many entries are later **refuted, narrowed, or superseded** by a newer
entry — that is the scientific process working, not an error. A verdict here is
**not** a statement of the current state of the code.

For the **current, authoritative** status of the project, use, in order:

1. The top-level [`README.md`](../README.md) and `RELEASE_NOTES*.md` — the
   shipped, reconciled status of the release.
2. [`docs/KNOWN_ISSUES.md`](../docs/KNOWN_ISSUES.md) and
   [`KNOWN_ISSUES.md`](../KNOWN_ISSUES.md) — the honest current carry-overs.
3. The `proofs/` tree — the proof objects each headline claim links to.

If a file under `.agent/` disagrees with the top-level README / RELEASE_NOTES /
KNOWN_ISSUES, **the top-level docs win.**

## What lives where

| Path | Contents |
|---|---|
| `decisions/` | ADRs and decision records (architecture, precision, roadmap). |
| `reviews/` | Cross-model critiques and audits (Opus / GPT-5.5 / Gemini), dated. |
| `sprints/` | Per-sprint contracts, worker handoffs, tester/reviewer notes, and command/output logs. The bulk of the log. |
| `skills/` | Reusable agent skills (`<name>/SKILL.md`) — the *live* operating procedures. |
| `rules/`, `roles/`, `contracts/` | The operating rules, role definitions, and frozen interface contracts. |
| `dispatches/`, `reports/`, `notes/`, `tasks/`, `goals/`, `milestones/` | Process bookkeeping. |
| `memory/`, `references/`, `patches/` | Long-lived memory index, external references, and the memory/skill patch protocol. |

## Authoritative order for an agent working in this repo

1. [`PROJECT_CONSTITUTION.md`](../PROJECT_CONSTITUTION.md)
2. [`AGENTS.md`](../AGENTS.md)
3. the current sprint contract
4. the relevant `.agent/skills/<name>/SKILL.md`
5. on-demand references — only the ones the task needs

Do not load or follow the old global `wrf-gpu-port` skill for this repository.
It belongs to a separate, earlier legacy-port effort with different
architecture assumptions.
