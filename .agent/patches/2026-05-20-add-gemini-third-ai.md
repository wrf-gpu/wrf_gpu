# Patch: Add Gemini 3.5 as third AI for second-opinion / side-runner / test-tool roles

Date: 2026-05-20
Author: manager (Claude Opus 4.7 1M-context)
Status: pending reviewer approval
Policy: `.agent/rules/memory-update-policy.md`

## Scope

Adds Gemini 3.5 high-flash (via `agy` CLI or `opencode run -m google/gemini-3.5-flash`) as a third AI family this project may consult. Constrains it to non-binding roles. Updates three skill files and adds one new reference file. No deletions; all changes are additive.

## Evidence

- User authorization 2026-05-20: "Gemini 3.5 high flash available via antigravity cli ... according to the benchmarks it is equally smart as you and as gpt 5.5 in coding tasks ... at least 4x faster ... very cheap to ask for a second or third opinion when we are stuck or write test tools or reports."
- User constraint same turn: "I don't know this model yet and it should not be used as the primary or as the sole judge."
- Verified `agy` install: `/home/enric/.local/bin/agy`, supports `-p` non-interactive, `--dangerously-skip-permissions`, `--continue`.
- Verified `opencode` install: `/home/enric/.opencode/bin/opencode`, supports `run -m google/gemini-3.5-flash`, has Google API credentials at `~/.local/share/opencode/auth.json`.

## Proposed destination

If reviewer Accepts, the three `.proposed.md` siblings replace their respective live `SKILL.md` files in-place:
- `.agent/skills/resolving-cross-model-disagreements/SKILL.md`
- `.agent/skills/conducting-blind-review/SKILL.md`
- `.agent/skills/managing-sprints/SKILL.md`

The new reference `.agent/references/dispatching-gemini.md` is additive and already in place (no replacement needed).

## Proposed changes

### New file — additive, no patch protocol required, written directly

- `.agent/references/dispatching-gemini.md` — CLI patterns, allowed/forbidden roles, dispatch hygiene, track-record table.

### Modified files — proposed siblings written, awaiting reviewer

| File | Proposed sibling | Change summary |
|---|---|---|
| `.agent/skills/resolving-cross-model-disagreements/SKILL.md` | `.proposed.md` | Add Gemini quick-poll as optional step 4; hard rule preventing sole-tiebreaker role |
| `.agent/skills/conducting-blind-review/SKILL.md` | `.proposed.md` | Add hard rule: Gemini never sole reviewer; supplementary side-runner only |
| `.agent/skills/managing-sprints/SKILL.md` | `.proposed.md` | Add AI-families × roles matrix; parallelism cap (≤2 Gemini side-runners); tmux naming convention |

### Rule file — NOT modifying yet, awaiting reviewer

`.agent/rules/cross-model-review-policy.md` — needs a small addendum to acknowledge three-AI structure. Manager defers this edit to reviewer's call; if reviewer Accepts, manager will add a one-paragraph addendum.

## Why additive, not destructive

The three existing skill files already work for the two-AI (Claude + codex) case. The patches only ADD Gemini as a third option and write a CONSTRAINT rule on Gemini. No existing workflow is removed or changed. Two-AI sprints continue to be the default; Gemini is opt-in per decision.

## Why not silent edits

Per `.agent/rules/memory-update-policy.md`, stable skills require patch + evidence + reviewer approval. Even though additive, these edits change the binding-decision rules around blind review and disagreement resolution. Silent self-update would be a non-negotiable violation.

## Risk assessment

- **Low risk** — additive, constrained role.
- Failure mode: if Gemini's output quality is in fact below Claude/codex on this project's task distribution, the harm is one wasted opinion per consult (~30 s wall-clock). Track-record table at `dispatching-gemini.md` exists to catch this pattern.
- Promotion gate documented: ≥3 successful side-runner deliveries before considering wider role.

## Dispatch hygiene tightening (added 2026-05-20 per user directive)

Initial first-delivery dispatch (M5-S1 attempt-4 third opinion) used inline `agy -p` without tmux or onboarding prefix. User feedback corrected this: Gemini must always dispatch via tmux + interactive REPL + the onboarding prefix at `.agent/references/gemini-onboarding-prompt.md` so it behaves consistently with Claude / codex agents. The dispatch reference + .proposed.md siblings have been updated to encode this:

- Pattern A (canonical) — tmux new-window + agy interactive (`-i`) + onboarding-prefix-prepended prompt + pipe-pane logging + completion-handler teardown.
- Pattern D (inline `-p`) — restricted to throwaway ping / sanity tests only; never for decision-bound output.

The first-delivery dispatch is flagged in the track-record table as a pre-update Pattern D case; future Gemini dispatches will follow Pattern A.

## Role expansion (added 2026-05-20 evening per user directive)

After Gemini's first two deliveries proved high-value (1 novel reviewer check + 1 specific coefficient bug found that primary AIs missed), user expanded Gemini's role envelope. Updated:

- **Bug-fix parallel-pair (mandatory)**: every confirmed issue dispatches ≥2 AIs in parallel; one MUST be Gemini, the other codex or Claude. Manager combines candidates. Eliminates Gemini hallucination risk on consequential decisions while preserving speed advantage.
- **Large/complex reviews**: Gemini parallel side-runner is default-on alongside the primary reviewer (Claude Opus 4.7). Primary reviewer's verdict is binding; Gemini's report is supplementary.
- **Tools / sidecars / scripts / report drafts / quick probes**: unconstrained.
- **Sprint frontrunner**: codex gpt-5.5 xhigh remains the default primary worker for new sprint implementation. Gemini may run as a SECOND worker in a parallel-pair on bug-fix sprints.
- **Forbidden roles** (unchanged): never sole worker, tester, reviewer, critical-reviewer, or judge.

The promotion-gate "≥3 deliveries" criterion is now effectively retired in favor of the role-by-role expansion above. Track record at `.agent/references/dispatching-gemini.md` continues for audit purposes.

## Pending external reviewer

Codex critical-review queue is currently holding skill-patch `2026-05-19-skill-updates-m2-m3-lessons.md` (task #35). Manager will dispatch this Gemini patch to codex review immediately after the previous skill-patch closes (or in parallel if codex bandwidth permits and the two patches do not overlap — they don't).

## Reviewer status

Pending. Manager will not apply the `.proposed.md` files to live skills until reviewer Accepts.

## Validation

```bash
python scripts/validate_memory_patch.py .agent/patches/2026-05-20-add-gemini-third-ai.md
```

## Files touched by THIS patch (for the validator)

- `.agent/skills/resolving-cross-model-disagreements/SKILL.proposed.md`
- `.agent/skills/conducting-blind-review/SKILL.proposed.md`
- `.agent/skills/managing-sprints/SKILL.proposed.md`
- `.agent/references/dispatching-gemini.md` (new, additive, already written)
