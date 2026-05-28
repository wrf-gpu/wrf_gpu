# Sprint Contract — Sprint #6: Paper Control (Opus 4.7)

**Sprint ID**: `2026-05-28-paper-control-opus`
**Created**: 2026-05-28
**Status**: READY — final quality gate before PDF + v0.0.1 push
**Predecessor**: Sprint #5 paper rewrite landed; audit script `ok: true`, 66 BibTeX, 57 cited, 7134 main words.

## Objective

Opus performs the **final manager-grade quality gate** on the v0.0.1 paper before it goes to PDF + arXiv. The principal directed: *"have opus 4.7 control the paper"*. This is that step.

The verdict drives whether the paper ships at v0.0.1. Opus has authority to either approve (paper goes to PDF + push) or block (specific patch sprint runs first).

## Specific scope

Read end-to-end (in this order):
1. `publication/draft/paper.md` (just rewritten by sprint #5, 260 lines / ~7134 words)
2. `publication/draft/references.bib`
3. `.agent/sprints/2026-05-28-testing-execution-opus-check/publishability_decision.md` (the binding verdict)
4. `.agent/sprints/2026-05-28-testing-execution-opus-check/paper_rewrite_input.md` (what sprint #5 was supposed to copy verbatim)
5. `.agent/decisions/PAPER-REWRITE-FRAMING-MEMO.md` (editorial brief)
6. `.agent/sprints/2026-05-28-gpu-wrf-history-research/novelty_bounds.md` (Option-2 wording must appear verbatim)

## Acceptance

- **AC1 — Precondition compliance**: did sprint #5 obey the opus #4 PUBLISHABLE_AS_IS preconditions?
  - Option-2 novelty wording verbatim — yes/no with line-number citation
  - Canary skill regression placed in Abstract + Results + Limitations + Discussion — yes/no with line-number citations
  - Paper title does NOT contain "Canary" — yes/no
  Emit `precondition_compliance.md`.

- **AC2 — Honesty audit**: every quantitative claim in the paper traces to a proof object listed in `publication/draft/honesty_audit.md`. Cross-check. Flag any number without backing.

- **AC3 — Citation audit**: every `\cite{...}` key in paper.md resolves in references.bib. Audit script already says yes; opus does an independent check + decides whether the 9 uncited entries (anthropic2024effective, anthropic2026claude, fredj2023adios2wrf, huang2013thermal, jakobs2024wsm7, milroy2018ensemble, roberts2008scale, schmidt2025senior, wernli2008sal) should be CITED (add them in context) or TRIMMED (remove from bib). Make the call; do not defer.

- **AC4 — Narrative flow**: read end-to-end as a reviewer. Does the introduction read well? Does the methodology section explain the frontrunner-critic-feedback pattern clearly? Does the Limitations section sound credible? Is anything redundant or contradictory? Mark concrete fixes in `narrative_critique.md`.

- **AC5 — Top-5 must-fix list**: distill all findings into a top-5 must-fix list with file:line and proposed wording. If opus would mark the paper PUBLISHABLE_AS_IS without any changes, the must-fix list is empty and the paper proceeds to PDF.

- **AC6 — Apply the fixes**: opus IS authorised to edit paper.md and references.bib directly to apply the must-fix changes. If a fix is structural (e.g. reorder a section), describe it but don't apply unless the change is small and clear-cut.

- **AC7 — Re-run audit**: `bash scripts/m7_publication_audit.sh` must still return `ok: true` after any edits.

- **AC8 — Final verdict**: `paper_control_verdict.md` with one of:
  - **APPROVED_FOR_PDF**: paper is final; manager renders PDF + tags v0.0.1
  - **APPROVED_AFTER_TRIVIAL_EDITS**: opus applied edits, paper is final
  - **BLOCKED_NEEDS_REWORK**: explicit list of what needs another sprint

## Files Opus May Modify

- `publication/draft/paper.md`
- `publication/draft/references.bib`
- `publication/draft/honesty_audit.md`
- `.agent/sprints/2026-05-28-paper-control-opus/**`

## Files Opus Must Not Modify

- `src/gpuwrf/**`, `tests/**`, `scripts/**`
- `.agent/decisions/**` (history, not editable)
- `/home/enric/src/wrf_gpu/**` (public repo — manager handles staging after this sprint)
- All proof JSONs in `.agent/sprints/*/proof_*.json` (frozen)

## Hard Rules

1. **Option-2 novelty wording is binding** — paper must use it verbatim; cannot be weakened or strengthened.
2. **No fabricated citations.**
3. **No claim without proof object.**
4. **CPU pinning**: `taskset -c 0-3`.
5. **No GPU runtime.**
6. **No remote push.**

## Dispatch

- Tester: claude opus 4.7 xhigh
- Wall-time: 1-3 h
- Branch: `tester/opus/paper-control`
- Worktree: `/tmp/wrf_gpu2_papercontrol`
- GPU usage: NONE
