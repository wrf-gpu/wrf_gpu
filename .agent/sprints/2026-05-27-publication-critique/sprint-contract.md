# Sprint Contract — Publication Critique (adversarial reader)

**Sprint ID**: `2026-05-27-publication-critique`
**Created**: 2026-05-27 (user direction: GPT adversarial reader after manager review)
**Status**: READY
**Predecessor**: `publication/draft/paper.md` (first draft merged at `b4ceb7e`; updated by manager at `c9ab7c0` with post-fix verdict)

## Objective

Read the publication draft adversarially. Treat it as a reviewer would treat an arXiv preprint about a multi-agent AI-built scientific code: assume the work was rushed, the AI authors may have overclaimed, and the prior art may be miscited. Surface every problem you can find. Suggest concrete fixes inline. Add missing citations from the research brief if you spot them. Critique structure, framing, claim-evidence linkage, missing caveats, weak transitions, and prose quality.

This is the GPT (codex) critique pass. Pair with the manager's (opus) prior review pass. The output is a critique markdown + an inline-edit suggestions markdown; the manager then synthesizes and applies revisions.

## Acceptance

- **AC1 — Structural critique**: read `publication/draft/paper.md` end-to-end. Identify weak sections, missing transitions, redundant paragraphs, sections that overclaim, sections that underexplain. Emit `.agent/sprints/2026-05-27-publication-critique/structural_critique.md`.

- **AC2 — Claim-evidence audit**: for every quantitative claim in the paper (numbers, percentages, speedup ratios, RMSE values, hardware specs), verify it points to a proof object on disk or a citation. Flag claims without backing. Cross-check with `.agent/sprints/2026-05-27-publication-first-draft/honesty_audit.md`. Emit `.agent/sprints/2026-05-27-publication-critique/claim_evidence_audit.md`.

- **AC3 — Citation audit**: read `publication/draft/references.bib`. For each entry, judge:
  - Is the citation real and verifiable (DOI/arXiv ID)?
  - Is it correctly placed in the paper (cited where the right idea is being made)?
  - Are there obviously-missing citations the research brief named that we didn't include?
  Emit `.agent/sprints/2026-05-27-publication-critique/citation_audit.md`.

- **AC4 — Prior art gaps**: cross-reference the paper's Related Work section (§2.2 GPU NWP, §2.3 ML emulators, §2.4 AI agents) against the research brief at `publication/research_brief/english_brief.txt`. Did the draft omit important prior art that the brief named? Surface the gaps. Emit `.agent/sprints/2026-05-27-publication-critique/prior_art_gaps.md`.

- **AC5 — Methods section completeness**: the Methods section (§3 AI Collaboration, §4 Numerical Port, §5 Physics) is the project's primary contribution. Critique it for: explicit-enough role description, sprint-contract pattern documented, failure-mode examples concrete, ADR concept introduced, proof-object discipline named, cross-AI verification explained. Add missing detail suggestions. Emit `.agent/sprints/2026-05-27-publication-critique/methods_critique.md`.

- **AC6 — Inline edit suggestions**: produce `publication/draft/paper.critique.md` — a copy of paper.md with inline `<<<CRITIQUE: ...>>>` annotations at every suggested fix. Do NOT modify the live paper.md; produce a parallel critique file for the manager to consolidate.

- **AC7 — Tone audit**: scientific writing tone, voice consistency, hedging-vs-overclaiming balance. Are there places where the paper is too humble (selling itself short)? Too bold (overclaim)? Emit short notes in `tone_audit.md`.

- **AC8 — Top 5 must-fix list**: distill all critique into the top 5 items that, if not fixed, would weaken the paper most. Emit `top_5_must_fix.md`.

- **AC9 — Worker report**: verdict `CRITIQUE_DELIVERED` with stats (count of structural issues, claim-evidence gaps, citation issues, prior-art gaps, inline suggestions, top 5).

## Files Worker May Modify

- `publication/draft/paper.critique.md` (NEW — annotated copy, parallel to paper.md)
- `.agent/sprints/2026-05-27-publication-critique/**`

## Files Worker Must Not Modify

- `publication/draft/paper.md` — the live draft is the manager's to edit after this critique lands
- `publication/draft/references.bib` — manager adds citations after consolidating
- `publication/research_brief/**` — input only
- governance files
- `src/gpuwrf/**`, anything outside `publication/` and the sprint folder

## Hard Rules

1. **No live paper edits.** All suggestions go in the parallel `.critique.md` and standalone audit markdowns.
2. **No invented citations.** Only flag existing references or refer to papers named in the research brief.
3. **CPU pinning**: `taskset -c 0-3`.
4. **No GPU runtime.** Parallel-safe with the iter2 skill fix sprint.
5. **No remote push.** Local commit on `worker/gpt/publication-critique` only.
6. **Adversarial but honest**: surface real problems, don't manufacture imaginary ones for completeness.

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Wall-time: 2-4 h
- Branch: `worker/gpt/publication-critique`
- Worktree: `/tmp/wrf_gpu2_pubcritique`
- GPU usage: NONE
