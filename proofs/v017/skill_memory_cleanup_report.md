# Skill + Memory Cleanup Report (v0.17 release-quality pass)

Date: 2026-06-15. Engineer: Opus "Aufräumen". Scope: code/docs only, no GPU, cores 0-3.
All file changes consolidated in the MAIN repo tree `<USER_HOME>/src/wrf_gpu2`
(branch `worker/gpt/v016-fp32-s2`, same branch as the worktree). NOT committed —
left for manager review.

## Headline

- The non-managing `SKILL.md` files were **already lean** (37-49 lines each, English,
  for-AI, no anecdotes). They did **not** need rewriting — they read like crisp
  operating manuals as-is. Only one model-name micro-edit was warranted.
- The real cleanup was **ballast**: 12 stub CHANGELOGs, 3 stale `.proposed.md` drafts,
  1 stale-"M0" orphan example, and a 76-line dated CHANGELOG narrative.
- MEMORY.md (the index) was rewritten from a flat list with 5 stacked version
  LIVE-ANCHOR lines + 14 v0.14 fix-by-fix lines into a clean, sectioned, newest-first
  linked index with a single authoritative live anchor.

## TASK 1 — Skill files (per file, before → after)

| Skill SKILL.md | Before | After | Action |
|---|---|---|---|
| building-wrf-oracles | 38 | 38 | no change (already lean) |
| conducting-blind-review | 38 | 38 | no change |
| designing-gpu-state | 38 | 38 | no change |
| locking-gpu | 80 | 80 | no change (all durable protocol; kept) |
| maintaining-memory | 49 | 49 | 1-line micro-edit: genericized a model-specific parenthetical ("Fable kernel-optimization sprints" → "a multi-sprint kernel-optimization investigation") so the release skill is model-agnostic |
| profiling-nvidia-gpu | 38 | 38 | no change |
| reporting-to-human | 38 | 38 | no change |
| researching-prior-art | 38 | 38 | no change |
| resolving-cross-model-disagreements | 37 | 37 | no change |
| updating-docs-minimally | 38 | 38 | no change |
| validating-physics | 38 | 38 | no change |
| writing-execplans | 38 | 38 | no change (`codex/plans/` path verified to exist) |
| writing-gpu-kernels | 40 | 40 | no change |
| managing-sprints | 303 | 303 | NOT TOUCHED — owned by manager (their rewrite is a separate uncommitted `M` in the same tree) |

Total non-managing SKILL.md: 548 → 548 lines (already release-quality).

Scan results (markers like dates / `v0.x` / `M<n>` / model names / "bit us" across all
non-managing SKILLs): only 4 hits, all benign and durable —
`M2 ADR` / `M2 bakeoff` (real ADR-001 gate references), `codex/plans/` (path exists),
and the one model parenthetical that was genericized.

References + templates (20 files) reviewed: all clean, durable, reusable scaffolds.
No staleness. None deleted. Script references verified: `scripts/validate_memory_patch.py`
and `scripts/close_sprint.py` both exist at repo root (NOT stale).

## TASK 2 — Ballast deleted (16 files + 1 dir)

3 stale `.proposed.md` patch drafts (Gemini-centric, contradicted current model roster;
referenced only by an old historical patch file):
- `managing-sprints/SKILL.proposed.md`
- `conducting-blind-review/SKILL.proposed.md`
- `resolving-cross-model-disagreements/SKILL.proposed.md`

12 stub CHANGELOGs (5-line "## 0.1.0 — Initial X" boilerplate, zero real history):
- building-wrf-oracles, conducting-blind-review, designing-gpu-state, maintaining-memory,
  profiling-nvidia-gpu, reporting-to-human, researching-prior-art,
  resolving-cross-model-disagreements, updating-docs-minimally, validating-physics,
  writing-execplans, writing-gpu-kernels — each `CHANGELOG.md` removed.

1 orphan example (stale "M0" narrative, unreferenced by SKILL or evals):
- `writing-gpu-kernels/examples/placeholder/README.md` (+ now-empty `examples/` dir).

`managing-sprints/CHANGELOG.md` (76 → 15 lines): kept the ONE changelog that carried
real history, but replaced 76 lines of dated/model-policy-churn narrative (0.2.0–0.2.7,
Gemini/Fable/Mythos directives now contradicted by the current roster) with a `0.3.0`
entry noting the manager's release-quality SKILL rewrite + one historical-rollup line
pointing to `AGENTS.md` + `.agent/decisions/VERSION-SPRINT-LEDGER.md`.

## TASK 3 — MEMORY.md index rewrite

File: `<USER_HOME>/.claude/projects/-home-user-src-wrf-gpu2/memory/MEMORY.md`
(user's private auto-memory; not in the release tree).

Before: 76 lines / 21962 bytes — one flat bullet list; **5 stacked version LIVE-ANCHOR
lines** (v0.16, v0.15, v0.14, v0.12, v0.11) + **14 v0.14 fix-by-fix lines** + durable
feedback/decision lines, no structure, one orphan file (`feedback_gpu_scalable_positioning...`)
not indexed at all.

After: 81 lines / 12K — sectioned (Live state / Shipped versions / Roadmap+release /
Architecture / Validation philosophy / Model dispatch / Operational hygiene), newest-first,
every line links to its file with a short hook. Changes:
- ONE authoritative live anchor (the active `project_fp32_makeorbreak_overnight_2026_06_14.md`),
  merging the duplicate v0.16 anchor line + the overnight-progress line that pointed to the
  same file. The active anchor file itself was NOT edited.
- The 4 superseded version anchors collapsed to one short "Shipped versions (superseded
  anchors)" subsection with one pointer line each.
- The 14 v0.14 fix-by-fix lines collapsed into ONE "v0.14 fix-by-fix history → ledger +
  v0.14 SHIPPED anchor" pointer line. The 15 underlying `project_v014_*.md` detail files
  are RETAINED on disk (lossless) — only their individual index lines were dropped.
- Added the previously-orphaned `feedback_gpu_scalable_positioning_and_proof_2026_06_08.md`
  to the index (under Live state).
- Verified: every link in the new index resolves to an existing file; the only files no
  longer individually linked are the 15 `project_v014_*` detail files (intentional fold).

Memory directory: 71 files / 856K unchanged on disk — NO memory files deleted (conservative,
lossless). The cleanup is purely the index. The current live anchor and all 2026-06-13/14/15
files are untouched.

## Process note / risk left for the manager

- IMPORTANT: There are two working trees on the SAME branch `worker/gpt/v016-fp32-s2`:
  the MAIN repo `<USER_HOME>/src/wrf_gpu2` (where the manager's uncommitted
  managing-sprints/SKILL.md rewrite lives) and the linked worktree `.wt-v018-trunk`
  (the session's cwd). All cleanup changes were consolidated into the MAIN tree to sit
  alongside the manager's in-flight work; the worktree was reverted to clean. The manager
  reviews + commits from the MAIN tree.
- The managing-sprints/SKILL.md `M` you see is the MANAGER's rewrite (-239/+52), not mine.

## Merge/delete proposals NOT done (left for manager judgment)

- No skill was merged or deleted. All 14 skills reflect how the project works; none is
  redundant with another. `conducting-blind-review` vs `resolving-cross-model-disagreements`
  are adjacent but distinct (one-reviewer-vs-work vs N-models-disagree) — kept separate.
- The 15 `project_v014_*.md` memory detail files are kept on disk (folded out of the index
  only). If the manager wants the directory physically smaller, these are the safe
  candidates to archive — their content is captured in `VERSION-SPRINT-LEDGER.md` + the
  v0.14 SHIPPED anchor — but I left them per "prefer linking over deleting; when unsure keep".
- The two largest memory files (`project_roadmap_to_v1_and_v040_sprints` 285K,
  `project_v090_ship_and_v0100_objective` 52K) are kept verbatim; trimming their inline
  bulk is a separate judgment call I did not make.
