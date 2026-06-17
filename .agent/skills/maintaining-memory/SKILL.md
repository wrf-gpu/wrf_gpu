---
name: maintaining-memory
description: Controls how project memory and skills are updated through evidence, review, validation, and hygiene.
---

## When to use

Use at sprint close, milestone closeout, when recurring lessons should become durable project memory, and **before every `/compact` or whenever the manager's context is at risk**.

## Pre-compaction checklist (MANDATORY before `/compact`)

Compaction discards the live context; review files and in-flight reasoning are NOT durable. Before compacting, the manager confirms ALL of the following are written AND committed (uncommitted `.agent/decisions/*` is lost on worktree teardown):

1. **Live anchor** — the current `⚑⚑ LIVE ANCHOR` auto-memory file reflects the true present state (HEAD sha, running workers + branches, what ships next), and `MEMORY.md` points to it at the top.
2. **Sprint ledger** — every major sprint closed this session has a row in `.agent/decisions/VERSION-SPRINT-LEDGER.md`.
3. **Core decisions** — every theoretical limit, what-can/can't-be-optimized verdict, closed-wontfix-with-evidence, scope cut, or roadmap change made this session lives in an authoritative in-repo doc (`.agent/decisions/*`), not only in a sub-agent review or context. Expensive cross-model findings (e.g. Fable kernel-optimization sprints) get their own FINAL doc.
4. **Commit** the decision/ledger/skill changes. A decision that cost real tokens to reach (multi-sprint, cross-model) MUST be durable before compact.

A decision is "captured" only when a future fresh context could reconstruct it from the repo + auto-memory alone.

## Inputs required

Manager closeout, tester/reviewer lessons, evidence, proposed destination, and reviewer status.

## Workflow

1. Draft memory patch.
2. Validate required fields.
3. Reviewer checks truth, generality, duplication, and usefulness.
4. Apply only approved minimal patch.
5. Run affected skill evals after skill changes.

## Hard rules

- No self-update of stable memory.
- Do not encode one-off failures as global rules.
- Remove or compress stale memory at milestone hygiene.

## Deliverables

Memory patch, validation result, approved stable-memory or skill edit.

## Validation

Run `python scripts/validate_memory_patch.py <patch>`.

## Common failure modes

Memory bloat, duplicated rules, stale lessons, and review-free updates.
