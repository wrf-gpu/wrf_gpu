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
