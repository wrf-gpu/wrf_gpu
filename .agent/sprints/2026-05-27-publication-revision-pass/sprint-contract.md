# Sprint Contract — Publication Revision Pass (apply critique findings)

**Sprint ID**: `2026-05-27-publication-revision-pass`
**Created**: 2026-05-27 (user direction: GPT critiques AND adds)
**Status**: READY
**Predecessors**:
- `publication/draft/paper.md` (manager-edited at `c9ab7c0` with post-fix verdict)
- `publication/draft/paper.critique.md` (28 inline `<<<CRITIQUE:>>>` annotations from the adversarial pass)
- `.agent/sprints/2026-05-27-publication-critique/{structural_critique,claim_evidence_audit,citation_audit,prior_art_gaps,methods_critique,tone_audit,top_5_must_fix}.md`

## Objective

Apply the publication critique findings to `publication/draft/paper.md`. This is the "GPT adds" step of the user's workflow (after the draft, manager review, and GPT critique). After this, the manager (Claude Opus 4.7) does a final review and we dispatch the benchmark/tables consolidation sprint.

**Headline reframe** is the single most important change: 708.32 s / 23.02× must be the current result; 324.78 s / 50.20× is the pre-fix diagnostic/overclaim episode. Everywhere this is inverted must be corrected.

## Acceptance

- **AC1 — Abstract rewrite**: rewrite the abstract to lead with the current corrected-physics state (24h pipeline 708.32 s, 23.02× apples-to-apples speedup, all M7 invariants preserved, partial skill recovery with 6/9 metrics improved over the pre-fix baseline). The 50.20× pre-fix number should appear only as the corrected-celebration story, not as a result. 200-250 words.

- **AC2 — Introduction reframe**: contribution list (the four bullets) must use current post-fix numbers as the engineering claim. Add explicit "current limitations" framing to the introduction.

- **AC3 — Chronology / timeline**: add a short timeline subsection (probably at the end of §2 Background or in §3 Methods preface): "M6 closeout → M7 perf-measurement → pipeline integration → original celebration (156× hallucination) → honest-speedup correction → skill regression discovery → RCA convergence → algorithmic fix → SKILL_IMPROVED_PARTIAL → iter 2 in progress". Keep it brief (<10 lines), pointers to closeout docs.

- **AC4 — Results split**: §7 Results (Performance) should clearly separate pre-fix diagnostic numbers from current post-fix corrected-physics numbers in Table 1, with row groups. §8 Results (Skill) should have separate subsections for pre-fix skill regression discovery and post-fix partial recovery. The two tables (Table 1 performance, Table 2 skill) should each be labeled "pre-fix" or "post-fix" explicitly.

- **AC5 — Methods enrichment**: §3 (AI Collaboration) should include:
  - One concrete sprint-contract excerpt as a code block (use the `m7-honest-speedup-skill-diff` contract as the example)
  - A claim-type → proof-object table (performance claims need profiler proof; correctness claims need savepoint proof; operational claims need CPU/GPU/obs comparison)
  - One rejection-loop example (the BLOCKED M6c-01 sprint that disproved its own hypothesis is ideal)

- **AC6 — Citations**:
  - Remove `\cite{TODO_Mollick_AI_authorship}` — either find a suitable replacement from the brief or remove the sentence
  - Add the 2 uncited BibTeX entries that the critique flagged (`fredj2023adios2wrf`, `paredes2023gt4py`) where they fit, OR remove them from the BibTeX file
  - Verify the 16 entries the citation audit flagged for metadata review (read `.agent/sprints/2026-05-27-publication-critique/citation_audit.md` for the list)
  - Add the 5 prior-art gap groups the critique named (COSMO/CH, stronger WRF GPU acceleration context)

- **AC7 — Limitations rewrite**: §10 Limitations must use current state. Radiation cadence is no longer disabled (it's at 180). Remove stale "radiation is effectively disabled" statements. Replace with: theta guard envelope saturation, frozen land/surface state, boundary width-1 strip, small validation corpus (3 V3 ICs + 1 side-by-side day), no live AIFS ingest, no independent human numerical review.

- **AC8 — Reproducibility section build-out**: §11 Reproducibility should include:
  - Public URL placeholder `github.com/<TBD>`
  - Commit-hash table (release commit + proof-object commits)
  - Environment manifest (Python, JAX, jaxlib, CUDA, NVIDIA driver, GPU, OS)
  - Proof-object manifest (the canonical list)
  - A lightweight `scripts/m7_publication_audit.sh` (NEW) that runs: word count, BibTeX parse, proof-object existence check, agentos validate
  - Treated as a first-class section, not appendix

- **AC9 — Author contributions / authorship policy**: rewrite §12 to be policy-aware. Cite arXiv's recent AI-paper crackdown (refs from the brief). Frame Claude and Codex as AI **systems** (not human authors) with explicit AI-contribution disclosure; keep Enric R.G. as the responsible human senior corresponding author. If a venue requires only human authors, the AI systems move from author line to acknowledgements + disclosure paragraph — but for arXiv preprint, the AI-author framing with explicit disclosure is defensible. Document both options briefly.

- **AC10 — Process the 28 inline suggestions**: read each `<<<CRITIQUE: ...>>>` in `paper.critique.md` and either: (a) apply the suggested change to paper.md, or (b) explicitly defer with a comment in a `revision_decisions.md` artifact. Aim to apply at least 22 of 28.

- **AC11 — Honesty audit refresh**: update `publication/draft/honesty_audit.md` (or write a new `.agent/sprints/2026-05-27-publication-revision-pass/revision_honesty_audit.md`) listing each quantitative claim in the revised paper and its proof-object pointer.

- **AC12 — Word count + parseability**: revised paper.md should still be in the 6000-12000 main-text word range. references.bib must still parse via `bibtexparser`. No non-ASCII characters.

- **AC13 — Worker report**: verdict `REVISION_READY` with stats (lines added/removed, citations added/removed, inline suggestions applied/deferred).

## Files Worker May Modify

- `publication/draft/paper.md` (the live draft — this is the revision target)
- `publication/draft/references.bib` (add/remove/fix entries per AC6)
- `publication/draft/tables/performance_summary.md` (split pre-fix vs post-fix per AC4)
- `publication/draft/tables/skill_regression_summary.md` (split per AC4)
- `publication/draft/honesty_audit.md` (NEW — AC11)
- `scripts/m7_publication_audit.sh` (NEW — AC8)
- `.agent/sprints/2026-05-27-publication-revision-pass/**`

## Files Worker Must Not Modify

- `publication/draft/paper.critique.md` — historical record from critique sprint
- `.agent/sprints/2026-05-27-publication-critique/**` — critique sprint artifacts are read-only
- `src/gpuwrf/**`
- governance files
- `MORNING-REPORT-2026-05-27.md`
- `.agent/decisions/**`

## Hard Rules

1. **Apply critique findings, don't manufacture new content.** Use the brief and existing proof objects; do not invent citations or claims.
2. **Headline reframe is non-negotiable** (AC1, AC2, AC4). Current = 708.32 s / 23.02×; pre-fix = 324.78 s / 50.20× as the corrected-celebration story.
3. **CPU pinning**: `taskset -c 0-3`.
4. **No GPU runtime.** This is a documentation sprint. Parallel-safe with iter 2 skill fix (GPU).
5. **No remote push.** Local commit on `worker/gpt/publication-revision-pass` only.
6. **Word budget**: keep main text 6000-12000 words. If the revision pushes over, trim Discussion redundancy per critique item §35-§39 of `structural_critique.md`.

## Dependencies

- Critique sprint merged
- paper.md current at `c9ab7c0` (manager's §7+§8 update)
- BibTeX 39 entries; targeted to ~38-44 after revision

## Proof Objects

- `publication/draft/paper.md` (revised — main deliverable)
- `publication/draft/references.bib` (updated)
- `publication/draft/honesty_audit.md` (AC11)
- `scripts/m7_publication_audit.sh` (AC8)
- `.agent/sprints/2026-05-27-publication-revision-pass/revision_decisions.md` (which inline suggestions were applied vs deferred)
- `.agent/sprints/2026-05-27-publication-revision-pass/worker-report.md`

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Wall-time: 3-5 h
- Branch: `worker/gpt/publication-revision-pass`
- Worktree: `/tmp/wrf_gpu2_pubrev`
- GPU usage: NONE
