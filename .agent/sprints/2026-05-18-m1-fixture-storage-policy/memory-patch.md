# Memory Patch Proposal

## Scope

One small operating lesson from S1 worth capturing as project-local guidance. **Not a memory entry** in the manager's persistent memory store (no user/feedback/project/reference fit) — instead, a future-sprints note that the next contract author should follow.

## Evidence

- `reviewer-report.attempt1.md` Blocker #1: validator and schema disagreed on `wrf_version` non-emptiness for `source == "wrf-derived"`. Round-trip cost: one tester pass + one reviewer pass + one worker fix-cycle attempt 2 + one reviewer pass.
- `reviewer-report.md` (attempt 2) Note: contract's file-size proof command was malformed and both worker and tester worked around it independently before the manager caught it in the amendment.

## Proposed Destination

This sprint folder only. The lesson is documented in `manager-closeout.md` § Lessons. No update to the manager's auto-memory at `/home/enric/.claude/projects/-home-enric-src-wrf-gpu2/memory/` is warranted — both lessons are operational rather than spanning future conversations, and the next contract author will see the closeout when planning S2.

## Patch

None to commit. The lessons live in `manager-closeout.md` § Lessons (committed with this sprint). The S2 contract should explicitly include "schema-validator parity test required" as an acceptance criterion, and stress-test every validation command verbatim before commit. The manager will apply this when authoring S2.

## Reviewer Status

Reviewer Status: not required — no stable-memory edit proposed. The closeout lesson section is part of the sprint artifacts already accepted by the reviewer (attempt 2 Decision: Accept).
