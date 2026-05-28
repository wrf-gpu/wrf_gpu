# Sprint Contract — GPU WRF Port History Research (Opus 4.7)

**Sprint ID**: `2026-05-28-gpu-wrf-history-research`
**Created**: 2026-05-28
**Status**: READY — research only, no code, no GPU
**Predecessor inputs**:
- `publication/research_brief/english_brief.txt` (deep research output from earlier this week)
- `publication/research_brief/german_brief.txt`
- `publication/draft/paper.md` §2.2 GPU NWP and Regional Models
- `publication/draft/references.bib`
- `.agent/sprints/2026-05-27-publication-critique/{prior_art_gaps,citation_audit}.md`

## Objective

The principal author has stated: *"there is no full GPU port even commercially available let alone open source and not for the lack of trying. It is mathematically and physically and coding wise hard."*

Sprint #5 (paper rewrite) will use this as a central motivating claim in the introduction. Before that, this sprint produces a rigorous evidence base so the claim is defensible. Specifically:

1. Synthesize a clean narrative of WRF's history + the history of GPU porting attempts on it.
2. Catalogue every known GPU port attempt of WRF (open-source AND commercial), with verifiable status, scope, license, and result.
3. Render a judgement on whether "first full open-source GPU port of WRF" is a defensible claim for `wrf_gpu`, with the precise wording the paper should use, and the citations that support each bound.
4. Identify the technical reasons it has been hard (math, physics, software) so the introduction can give the right respectful framing of the prior attempts.

## Acceptance

- **AC1 — History narrative**: produce `gpu_wrf_port_history.md`. A ~1500-word, citation-anchored history covering:
  - WRF model origins (NCAR/NCEP, ~2000, ARW core), key technical-note citation
  - Why WRF specifically is hard to GPU-port (split-explicit RK3 + acoustic substeps, vertically implicit solves, terrain-following mass coordinate, mixed memory layouts across schemes, the C-grid staggering pattern, lateral relax zone, scheme-specific physics interfaces, the legacy Fortran-MPI host-resident control flow)
  - Each documented GPU porting attempt with: timeframe, organisation, approach (OpenACC / CUDA Fortran / CUDA C / Kokkos / DSL / clean-slate), scope of port (full / partial — which schemes ported and which left on host), peak reported speedup, open-source status, current maintenance status
  - The pattern of "5-7× ceiling" for directive-based ports (cited)
  - Commercial offerings (AceCAST, NVIDIA-related; if no peer-reviewed full benchmark exists, say so)
  - The ML-emulator parallel track (GraphCast, Pangu, NeuralGCM, AIFS, FourCastNet, GenCast, Aurora) and why they are NOT the same thing (they bypass physics)

- **AC2 — Comprehensive WRF GPU port catalogue table**: produce `gpu_wrf_port_catalogue.md` with one row per attempt:
  | Attempt | Year | Org | Approach | Scope | Reported speedup | License | Status |
  Aim for at least 8 rows. Cite source per row. If no public information exists for an attempt, mark as "no peer-reviewed citation found".

- **AC3 — Novelty bounds for the wrf_gpu claim**: produce `novelty_bounds.md` answering, with citations:
  - Is `wrf_gpu` the **first full open-source GPU port** of WRF? Yes / no / partial — and how should this be worded to be defensible?
  - Is it the **first JAX/Python full port**? (Yes is very likely based on prior research brief.)
  - Is it the **first AI-co-authored numerical-weather model**? (Likely yes, but novelty-bound and not the headline claim.)
  - What is the strongest scientifically defensible claim the paper can make? Write the precise sentence(s) the paper introduction should use, in three options ranked by aggressiveness vs defensibility.

- **AC4 — Why it has been hard**: produce `why_it_is_hard.md`. Four sections — math, physics, coding, organisational — each with concrete examples and a citation per example. The point is to frame respect for the prior attempts so the introduction does not sound dismissive.

- **AC5 — Citations to add**: list any BibTeX entries that should be added to `publication/draft/references.bib` to support AC1-AC4. Provide full BibTeX entries. Do not modify references.bib yourself — leave that for the paper-rewrite sprint.

- **AC6 — Multi-agent methodology framing**: produce `multi_agent_framing.md`. The principal author noted: *"the frontrunner-critic and feedback dialog system, I think this was key to the success here"* — plus: *"this was the second attempt, neither GPT 5.4 alone nor Opus 4.6 were able to get close in my first attempt"*. This sprint frames how the introduction should present that:
  - Why a single-model attempt failed (theoretical reasoning + the user's stated first-attempt failure)
  - What specifically the frontrunner-critic-feedback dialog adds (different blind spots, evidence-anchored cross-review, the rejection-loop mechanic that produces correction)
  - Citations the paper should use for multi-agent / actor-critic methodology in scientific software (research brief covers some)

- **AC7 — Tester report**: verdict `RESEARCH_READY` with a 5-line executive summary the paper-rewrite sprint can lift directly into the introduction's motivating paragraph.

## Files Tester May Modify

- `.agent/sprints/2026-05-28-gpu-wrf-history-research/**` only

## Files Tester Must Not Modify

- `publication/draft/**`, `publish/**`
- `src/gpuwrf/**`, `tests/**`, `scripts/**`
- governance files
- `/home/enric/src/wrf_gpu/**` (the public repo)

## Hard Rules

1. **No fabricated citations.** Use only what's in the research briefs + references.bib + verifiable academic sources. If you cannot verify a claim, mark it `[verify]`.
2. **No code, no GPU.**
3. **CPU pinning**: `taskset -c 0-3`.
4. **Honest INCONCLUSIVE**: if the user's claim of "no full open-source GPU port exists" turns out to be partially false (e.g. someone tried it 5 years ago and abandoned), report that honestly. The paper benefits more from a precisely-bounded claim than from an aspirational one.

## Dispatch

- Tester: claude opus 4.7 xhigh
- Wall-time: 1-3 h
- Branch: `tester/opus/gpu-wrf-history-research`
- Worktree: `/tmp/wrf_gpu2_history`
- GPU usage: NONE
