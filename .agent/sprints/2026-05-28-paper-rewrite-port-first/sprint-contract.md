# Sprint Contract — Sprint #5: Paper Rewrite (Port-First Focus, Multi-Agent Methodology)

**Sprint ID**: `2026-05-28-paper-rewrite-port-first`
**Created**: 2026-05-28 (sprint #5)
**Status**: READY — high priority, sequential after sprint #4 PUBLISHABLE_AS_IS verdict
**Pre-conditions confirmed**: Sprint #4 (opus) issued `PUBLISHABLE_AS_IS` (commit `*`), conditional on Option-2 novelty wording + framing memo adoption + open Canary skill regression disclosure.

## Mandatory inputs the worker MUST read before writing a single sentence

In this order:
1. `.agent/decisions/PAPER-REWRITE-FRAMING-MEMO.md` — editorial brief
2. `.agent/decisions/PAPER-STRATEGIC-FRAMING.md` — earlier strategic memo
3. `.agent/sprints/2026-05-28-testing-execution-opus-check/paper_rewrite_input.md` — exact sentences for Results + Limitations + "what this release does NOT claim"
4. `.agent/sprints/2026-05-28-testing-execution-opus-check/publishability_decision.md` — verdict + precondition
5. `.agent/sprints/2026-05-28-gpu-wrf-history-research/{gpu_wrf_port_history,novelty_bounds,why_it_is_hard,multi_agent_framing,gpu_wrf_port_catalogue,citations_to_add}.md` — all six artefacts
6. `publication/draft/paper.md` (current draft, will be replaced)
7. `publication/draft/references.bib` (current bibliography)
8. `.agent/decisions/M7-PERF-MEASUREMENT-CLOSEOUT.md`, `MILESTONE-M7-CLOSEOUT.md`, `MILESTONE-M7-CLOSEOUT-AMENDMENT.md`

## Objective

Produce the final paper rewrite for arXiv preprint companion to v0.0.1 release. **Port-first focus** — the artifact (a JAX-native WRF port) is the headline, not Canary forecasting. **Multi-agent methodology** is co-equal contribution. Use the **Option-2 novelty wording verbatim** as established by the history research sprint:

> *"first full open-source JAX/Python WRF v4 port with whole-state device residency on a consumer-grade workstation GPU"* (under strict definition: prior attempts — WRFg restricted-source, AceCAST closed-proprietary, FahrenheitResearch 2026 partial OpenACC with physics/BC/IO on CPU — do not satisfy this bound).

## Acceptance

- **AC1 — Section structure** per PAPER-REWRITE-FRAMING-MEMO.md "Implied paper structure" (Title → Abstract → Introduction → Background → Code Architecture → Code Physics → Methodology Multi-Agent → Validation Strategy → Results → Canary case study → Discussion → Open Source Release Plan → Limitations → Reproducibility → Author Contributions/AI Use → Acknowledgements → References).

- **AC2 — Introduction**: ~5 paragraphs, readable historical narrative. Use `gpu_wrf_port_history.md` as anchor. Walk the reader through WRF's role in operational meteorology → history of GPU porting attempts (cite each from the 16-row catalogue) → why prior attempts stalled (math + physics + coding hardness, from `why_it_is_hard.md`) → introduce wrf_gpu under Option-2 novelty wording. Final paragraph: the multi-agent methodology was key; the first attempt (GPT 5.4 alone, Opus 4.6 alone, separately) failed; the frontrunner-critic-feedback dialog with proof-object discipline is what worked.

- **AC3 — Abstract**: 200-250 words. Lead with artifact existence + open-source + single consumer GPU. Performance numbers second. Methodology one-liner. Honest skill caveat per `paper_rewrite_input.md`.

- **AC4 — Background/Related Work**: shorter than current; use the 16-row catalogue compressed into a paragraph table or bulleted summary; lean on `multi_agent_framing.md` for the AI-agent context.

- **AC5 — Code Architecture section (centerpiece, ~6 pages)**: whole-state device residency, jax.lax.scan compiled timestep, halo placeholder, precision policy, savepoint validation strategy. Reader who reads only this section should understand the design.

- **AC6 — Multi-Agent Methodology section (~4 pages)**: elevate from current §3. Document:
  - Role taxonomy (manager / worker / tester / reviewer / debugger)
  - Sprint-contract pattern (with one concrete excerpt)
  - Proof-object discipline (with concrete example file path)
  - Cross-model review (codex + opus disagreement → on-disk reconciliation)
  - First-attempt failure context: single-model GPT 5.4 and Opus 4.6 each failed in earlier attempt (cite as principal personal communication)
  - **The 156× → 22.26× self-correction** as the strongest single piece of evidence that the method works
  - Why a single-model agent fails at this scale

- **AC7 — Results section**: copy the exact sentences from `paper_rewrite_input.md` for DETERMINISM PASS, GPU execution proof, Canary 3-day skill table, and "what this release does NOT claim." Do not re-judge the evidence.

- **AC8 — Canary subsection ONLY**: keep but de-emphasise; do NOT put Canary in title or abstract; treat as a representative case study.

- **AC9 — Limitations**: must include exactly what's listed in `skip_fail_triage.md` as DOCUMENT-as-known-gap items (9 items). Use the precise wording from `paper_rewrite_input.md`.

- **AC10 — Open Source Release Plan section**: brief, points at the GitHub repo (`github.com/wrf-gpu/wrf_gpu`), AGPL-3.0, INSTALL.md, citation, contribution model, versioning policy (0.0.x → 0.1.0 arXiv companion → 1.0.0 reserved for operational claim).

- **AC11 — Author Contributions + AI Use Disclosure**: explicit per-author; the AI byline framing per `AI_USE.md` and the framing memo; arXiv 2026 policy aware.

- **AC12 — Citations**: add the BibTeX entries from `citations_to_add.md` (~245 lines) to `publication/draft/references.bib`. Ensure every cite in the paper has a matching key.

- **AC13 — Manager repo only**: edit `publication/draft/paper.md` and `publication/draft/references.bib` in the MANAGER repo only. The manager handles staging the final files into the public repo's `paper/` after this sprint lands. The worker MUST NOT touch any file under `/home/enric/src/wrf_gpu/`.

- **AC14 — Length budget**: 7000-12000 main words; aim for ~9000-10000. Don't pad. Cut where the framing memo says cut.

- **AC15 — Audit gate**: must pass `bash scripts/m7_publication_audit.sh` in the manager repo. No fabricated citations; all cited keys must be in references.bib.

- **AC16 — Honesty audit refresh**: update `publication/draft/honesty_audit.md` (and copy to public repo's `paper/`) listing every quantitative claim in the revised paper with its proof-object path.

- **AC17 — Worker report**: with verdict `REWRITE_READY` or `REWRITE_PARTIAL` (with explicit unfinished list).

## Files Worker May Modify

- `publication/draft/paper.md`, `publication/draft/references.bib`, `publication/draft/honesty_audit.md`, `publication/draft/tables/**` (manager repo ONLY)
- `.agent/sprints/2026-05-28-paper-rewrite-port-first/**`

## Files Worker Must Not Modify

- `src/gpuwrf/**`, `tests/**`, `scripts/**`
- governance, M6/M7 closeouts, framing memos (read-only inputs)
- The previously merged proof objects in `proofs/` of either repo
- `/mnt/data/canairy_meteo/**`

## Hard Rules

1. **Option-2 novelty wording verbatim** — non-negotiable per opus #4 precondition.
2. **No "Canary" in title.** No "Canary" leading the abstract. Canary stays as a representative case study only.
3. **Manager repo ONLY** — do NOT touch `/home/enric/src/wrf_gpu/` (the public repo with the GitHub remote). Manager copies the final paper into the public repo after this sprint lands.
4. **No fabricated citations.** Use only references.bib + the entries from citations_to_add.md.
5. **CPU pinning**: `taskset -c 0-3`.
6. **No GPU runtime.** Pure writing sprint.
7. **No remote push.** Local commit on `worker/gpt/paper-rewrite-port-first` only.
8. **Honest verdict.** REWRITE_PARTIAL is acceptable if a section honestly cannot land cleanly; document what's left.

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Wall-time: 3-6 h
- Branch: `worker/gpt/paper-rewrite-port-first`
- Worktree: `/tmp/wrf_gpu2_paperrw`
- GPU usage: NONE
