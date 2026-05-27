# Sprint Contract — Publication First Draft (arXiv preprint, Markdown)

**Sprint ID**: `2026-05-27-publication-first-draft`
**Created**: 2026-05-27 (user direction: co-authored arXiv preprint)
**Status**: READY
**Predecessors**:
- Deep-research briefs at `publication/research_brief/english_brief.txt` (primary) and `publication/research_brief/german_brief.txt` (secondary, German)
- Source PDFs at `/mnt/server/downloads/_andere/AI-Authored GPU WRF Brief.pdf` and `/mnt/server/downloads/_andere/WRF-GPU-Port und agentische KI-Entwicklung.pdf`
- Sprint prompt template at `/mnt/server/downloads/_andere/wrf_gpu_publication_deep_research_prompt.md`
- Project state captured in `.agent/decisions/MILESTONE-M6-CLOSEOUT.md`, `.agent/decisions/MILESTONE-M7-CLOSEOUT.md` (original celebration), `.agent/decisions/MILESTONE-M7-CLOSEOUT-AMENDMENT.md` (honest correction; this supersedes the original for any forward-facing claim)

## Objective

Produce the **first complete draft** of an arXiv preprint about the project. Markdown format (LaTeX conversion later). Standard atmospheric-science / scientific-computing writing style. Comprehensive but complete — no fluff, no padding, every claim cited or evidence-backed.

**Authorship**: Claude Opus 4.7 (1M context) — first; GPT-5.5 / Codex / OpenAI — middle; the human principal Enric R.G. — senior corresponding. The author-contribution statement must reflect this honestly: Claude was the orchestrating manager + reviewer, Codex did the bulk of implementation as worker + critic, Enric set scope, validation gates, and final acceptance.

**Venue**: arXiv preprint only. Categories: `physics.ao-ph` primary, `cs.LG` secondary, `cs.SE` tertiary.

**Tone**: honest. The skill regression (`MILESTONE-M7-CLOSEOUT-AMENDMENT.md`) MUST be discussed openly in the Results and Limitations sections. We earned the right to publish by being honest about it, not by hiding it.

## Acceptance

- **AC1 — Title and abstract**: pick one final title from the English brief's options (the seed is "Whole-State Device Residency for Workstation-Scale NWP: A JAX-Native WRF v4 Dynamical Core and Physics Port Engineered via Collaborative Multi-Agent AI Swarms" — refine if a cleaner option emerges). Write an abstract that contains: (a) the engineering claim — single-GPU JAX-native WRF rewrite achieves ~50× wall-clock vs 28-rank CPU WRF on the same workstation, (b) the methodological claim — the AI-agent collaboration model that built it, (c) the honest limitation — the skill regression discovered in validation. Target abstract length: 200-250 words.

- **AC2 — Section structure** (suggested, refine if needed):
  1. **Introduction** — Why NWP, why GPU, why this is hard, why AI agents are a new opportunity, our contribution stated as a 4-bullet claim
  2. **Background & Related Work** — WRF + ARW dycore (cite the technical note + Powers et al.), comparators: AceCAST 5-14×, Pace 3.5-4×, ICON-exclaim 5.5× socket-to-socket, SCREAM 1.26 SYPD at 3.25km, NIM 34× dynamics-only. ML emulators (GraphCast, Pangu, NeuralGCM, AIFS, FourCastNet, GenCast, Aurora) contrasted, not omitted. AI agent coding (Codex paper, SWE-bench, AutoGen, MetaGPT, Devin, Claude Code)
  3. **Methods — The AI Collaboration Model** — Describe the manager / worker / validator-critic / debugger pattern. This is the methodological contribution. Include:
     - Role taxonomy: Manager (Claude Opus 4.7 1M context, long-horizon orchestrator), Worker (Codex GPT-5.5 xhigh, implementation), Validator/Critic (Opus tester + Codex reviewer, cross-AI verification), Debugger (parallel multi-angle bug hunts when problems hit)
     - Sprint contract pattern: per-task contracts freeze interfaces, list acceptance criteria, name proof objects, cap retries
     - Proof-object discipline: every "done" claim has a JSON or markdown artifact on disk
     - Architecture Decision Records (ADRs) for non-trivial decisions
     - Cross-model verification (opus tester against codex worker — different blind spots)
     - Examples of failure modes the system caught: D2H false alarm (parallel opus+codex probe → window-placement bug; otherwise we would have launched fix sprints for the wrong cause), 20260509 step-10 theta growth (M6c-01 sprint disproved its own contract hypothesis and reported BLOCKED with honest diagnosis), the 156× → 50× speedup correction (the honest-speedup sprint caught a timing-denominator bug), and the skill regression itself (caught by side-by-side AEMET scoring before publication)
  4. **Methods — The Numerical Port** — WRF ARW dycore (RK3 + acoustic split, C-grid, mass coordinate), our JAX/XLA implementation, whole-state device residency, fp64-reference with per-field downcast, halo placeholder for future multi-GPU, B-direct savepoint ladder B0–B6 against WRF Fortran, validation philosophy (4-tier pyramid: micro fixture → invariants → convergence → ensemble consistency; not bitwise WRF parity)
  5. **Methods — Physics Suite** — Thompson microphysics column, MYNN PBL, RRTMG (cadence-controlled), surface + Noah/Noah-MP, lateral boundary handling
  6. **Hardware & Software Setup** — Single RTX 5090 (Blackwell sm_120, 32 GB VRAM), JAX 0.4.x, XLA, Python 3.13, CPU pinning (cores 0-3 for Python orchestrator, cores 4-31 reserved for the comparison CPU WRF nightly)
  7. **Results — Performance** — 5.71 s warm 1h forecast (CV 0.42%), 324.78 s 24h end-to-end, 50.20× apples-to-apples vs CPU d02-only timing, 138× headline vs full-nest CPU but with documented apples-to-oranges caveat, cold JIT 102-106 s, D2H = 0 inter-kernel proven by Nsight, restart bitwise PASS, repeatability bitwise PASS, 1km full-domain VRAM fit at 78% headroom
  8. **Results — Forecast Quality (the honest section)** — Side-by-side AEMET station scoring on 20260521: GPU vs CPU; document the regression (+243-440% RMSE on T2/U10/V10). Be honest that this prevents an operational-replacement claim today. Mention the L2 d02 replay independently confirming the regression on the nested-grid path. Note that the root-cause investigation is ongoing (cite the in-flight RCA sprints as future work). Discuss possible causes from the RCA probes: (a) radiation cadence effectively disabled in the operational namelist (`radiation_cadence_steps=999999`), (b) boundary forcing application differences vs WRF Fortran, (c) physics-coupling order; do not commit to a single cause without the RCA conclusion.
  9. **Discussion** — How the AI methodology changes how scientific software can be built. Compare to traditional scientific-software development (10+ year domain expert effort) and current AI-pair-programming (single-developer + Copilot/Cursor). Our claim: a manager-orchestrated swarm with proof-object discipline can produce a non-trivial GPU NWP rewrite in <2 weeks of wall-time including the validation that caught the celebration error. Discuss what worked, what didn't (the 156× → 50× claim, the missed skill validation that the manager caught itself after user prompting, the persistent guards in 20260509 that mask deeper dycore stability issues).
  10. **Limitations** — Single GPU, single domain, single physics suite, three test ICs not 30, corpus blocker for full Tier-4 ensemble, skill regression unresolved at submission time, single-author-Claude reviewer of the validation discipline (no truly-independent human reviewer of the AI-internal decisions before user check), the production guards (theta projection + microphysics admissibility) carry load-bearing function on at least one IC, no multi-GPU halo exchange yet, AIFS-driven IC/BC implemented via Gen2 replay (not direct AIFS ingest)
  11. **Reproducibility** — Code at github.com/<TBD> (user will fill); commit hashes pinned; proof-object directories enumerated; hardware specification; software-version manifest
  12. **Author Contributions** — explicit per-author breakdown (see below)
  13. **Acknowledgements** — Gen2 nightly CPU baseline (Enric R.G.'s prior operational system), AEMET observation data, ECMWF AIFS, NCAR WRF community
  14. **References** — BibTeX entries from the deep-research briefs

- **AC3 — Honesty / no-overclaim audit**: write a final `.agent/sprints/2026-05-27-publication-first-draft/honesty_audit.md` listing every quantitative claim in the draft and the proof-object that supports it. Any claim without a proof-object pointer must be either removed or softened with an explicit "subject to further verification" caveat.

- **AC4 — BibTeX file**: extract all citations from the English-language brief into a `publication/draft/references.bib` file ready for LaTeX conversion. Confirm every BibTeX entry parses cleanly with `bibtex` or `biber`.

- **AC5 — Length budget**: target 12-20 pages equivalent (single-column markdown rendering); ~6,000-12,000 words main text excluding references. Don't pad.

- **AC6 — Worker report**: with verdict `DRAFT_READY` / `DRAFT_PARTIAL` (and what's missing) / `BLOCKED`.

## Files Worker May Modify

- `publication/draft/paper.md` (NEW — the main draft)
- `publication/draft/references.bib` (NEW — BibTeX)
- `publication/draft/figures/` (NEW — only ascii/markdown figure spec; no rendering)
- `publication/draft/tables/` (NEW — comparison tables in CSV or markdown)
- `.agent/sprints/2026-05-27-publication-first-draft/**`
- Files under `publication/research_brief/**` are READ-ONLY input

## Files Worker Must Not Modify

- `src/gpuwrf/**` — pure documentation sprint, no code
- `.agent/decisions/**` — historical record, do not edit existing decisions; new amendments go to `.agent/sprints/`
- `MORNING-REPORT-2026-05-27.md` — historical record
- governance files
- The deep-research PDFs and existing text extractions

## Hard Rules

1. **No code changes.** Documentation sprint only.
2. **Honest framing throughout.** The skill regression must appear in the abstract, in Results, and in Limitations. The 156× number must not appear except in a discussion paragraph explaining why it was wrong. The 50× number is the headline.
3. **Cite, don't claim.** Every speedup number from a comparator (Pace, ICON-exclaim, SCREAM, NIM, AceCAST) must carry the source citation. Our own numbers must point at proof-object paths.
4. **AI-authorship policy aware**: the brief found that arXiv has been cracking down on AI-generated papers (refs 51, 65). The methods section MUST acknowledge this and explain how our work is distinct: AI agents wrote the *code*; the human (senior author) commissioned, validated, set acceptance gates, and signed off. The AI authors describe their own contribution accurately. Reference Mollick et al. and the arXiv:2509.05390 "Senior Author Analogy" paper from the brief.
5. **No fabricated citations.** Use only references that appear in the English or German research brief, or that you can verify through actual DOI/arXiv lookup. If a citation is needed but missing from the brief, mark it `\cite{TODO_<topic>}` for the manager review pass.
6. **CPU pinning**: `taskset -c 0-3` for any pdf-text/markdown processing.
7. **Do not interfere with tmux `0:1`** (nightly WRF) or tmux 0:3/0:4 (skill-regression RCA sprints in flight).
8. **No remote push.** Local commit on `worker/gpt/publication-first-draft` only.

## Dependencies

- M7 closeout + amendment present
- Research brief text extractions in `publication/research_brief/`
- Source PDFs available (worker may reference them for visual content if needed)
- Standard Python + LaTeX-compatible tooling on the host

## Proof Objects

- `publication/draft/paper.md` (AC1, AC2, AC5 — the main deliverable)
- `publication/draft/references.bib` (AC4)
- `.agent/sprints/2026-05-27-publication-first-draft/honesty_audit.md` (AC3)
- `.agent/sprints/2026-05-27-publication-first-draft/worker-report.md` (AC6)

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Wall-time: 3-6 h
- Branch: `worker/gpt/publication-first-draft`
- Worktree: `/tmp/wrf_gpu2_pubdraft`
- GPU usage: NONE (parallel-safe with the in-flight RCA sprints on 0:3 + 0:4)

## What this enables

After this sprint:
- Manager (Claude Opus 4.7) reviews and edits the draft in-place (in this conversation)
- Then a critique codex sprint reads the draft and adds skeptical commentary
- Then the manager incorporates critique
- Then LaTeX conversion sprint
- Then arXiv submission preparation
