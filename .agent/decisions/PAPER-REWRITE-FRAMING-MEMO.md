# Paper Rewrite Framing Memo — for Sprint #5

**Status**: editorial brief, 2026-05-28. Sprint #5 (paper rewrite, port-first focus) reads this BEFORE editing.
**Supersedes / extends**: `.agent/decisions/PAPER-STRATEGIC-FRAMING.md` (earlier brief).
**Author**: manager (Claude Opus 4.7), capturing the principal's directives 2026-05-27 and 2026-05-28.

## What the principal said (verbatim direction)

> "regarding the re-write when it gets to that step, there should be a nice good to read introduction which explains a bit on the history of wrf and gpu ports to motivate why it is quite a feat that this was done in full now first time by an ai team (at least open source). Maybe send a gpt agent to research this correctly but as far as i know there is no full gpu port even commercially available let alone open source and not for the lack of trying. It is mathematically and physically and coding wise hard. also it should be mentioned that this was the second attempt, neither gpt 5.4 alone nor opus 4.6 were able to get close in my first attempt. I think also a core point here could be the frontrunner-critic and feedback dialog system, I think this was key to the success here."

> "Finish the test and the benchmarks, finish the paper rewrite with the focus on this new port and the perspective that this should explain the port technically, present it to everyone who is interested and explain how we (you) did it with the multi agent orchestration and create a pdf version I can comment in for review."

## Five things sprint #5 must do

1. **Re-write the introduction** as a readable, scientifically-anchored historical narrative. Not a bullet list. A paragraph-prose introduction that begins with WRF's role in operational meteorology, then walks through the history of GPU porting attempts (each cited), then explains why the attempts have stalled (math + physics + coding hardness), then introduces `wrf_gpu` as the first full open-source GPU port and the first JAX-native implementation. Use the deliverables from `2026-05-28-gpu-wrf-history-research/` as the evidence base. The introduction should be enjoyable to read for an atmospheric scientist who is not a numerical-methods specialist.

2. **Treat the artifact as the headline.** Per the earlier framing memo: a JAX-native, GPU-resident, open-source code that implements WRF's ARW dycore + minimum operational physics now exists, works, and runs on a consumer GPU. Performance numbers are evidence, not the headline.

3. **Treat the methodology as a co-equal contribution.** The principal explicitly identified the frontrunner-critic-feedback dialog as "key to the success here." Sprint #5 must elevate the methodology section from its current §3 position to a proper contribution. Include:
   - The first-attempt failure (GPT 5.4 alone and Opus 4.6 alone, separately, both failed to get close) — this is the principal's stated history, must be cited as such (personal communication / experience report, not a published study; phrase accordingly)
   - The specific mechanism: a frontrunner agent (worker) generating, a critic agent (tester/reviewer) challenging, a feedback dialog anchored to proof objects on disk, and a manager orchestrating across sprints with long-context memory
   - Why a single-model agent would fail at this scale (limited context, no adversarial pressure, no proof-object discipline)
   - The documented self-correction example: the original 156× speedup claim → caught by the workflow → corrected to 22.26×. This is the strongest single piece of evidence that the method works.

4. **Make the technical port section the centerpiece.** Currently §4 (Methods: Numerical Port) and §5 (Methods: Physics Suite). Sprint #5 should rename and reorder so a reader who only reads sections 4-5 walks away understanding the design. Whole-state device residency, jax.lax.scan compiled timestep, halo placeholder, precision policy, savepoint-validated dycore.

5. **The Canary case becomes one Results example, not the framing.** Per the earlier strategic framing: title should not contain "Canary"; performance numbers should not lead the abstract; the Canary skill regression should be transparently reported but not centre-stage.

## Specific writing tasks for Sprint #5

- **Title**: pick one. Suggestions (from earlier framing memo, refined for the principal's history-narrative direction):
  - "wrf_gpu: An Open-Source JAX-Native Port of WRF's Dynamical Core to GPU"
  - "Whole-State Device Residency for WRF: A JAX-Native Open-Source GPU Port Engineered by Collaborative AI Systems"
  - "The First Full Open-Source GPU Port of WRF: A JAX-Native Implementation Engineered by Multi-Agent AI Orchestration" — only if AC3 of the history sprint confirms "first" is defensible

- **Abstract** (220 words): lead with the artifact existence, hardware ("single consumer GPU"), open-source. Performance numbers ("22.26x apples-to-apples vs 28-rank CPU WRF on the same machine; D2H = 0 inside the forecast loop; restart bitwise; 1 km fits a consumer 32 GB GPU"). Methodology one-liner ("engineered by a frontrunner-critic-feedback multi-agent process with proof-object discipline"). Honest skill caveat ("preliminary skill comparison shows the GPU forecast is currently materially less skilful than CPU WRF on a small validation corpus; remaining defects are localised to surface-flux coupling and theta-guard saturation").

- **Introduction** (~5 paragraphs): see (1) above. Anchor on the GPU-WRF history sprint's deliverables.

- **Background and Related Work**: keep existing structure, polish, lean on history sprint citations.

- **The Code: Architecture** (centerpiece, ~6 pages): JAX/XLA design, whole-state device residency, fused-scan timestep, halo placeholder, precision policy, savepoint validation strategy. The reader should be able to understand the design from this section alone.

- **The Code: Physics Suite** (~2 pages): Thompson + MYNN + RRTMG + surface; what's implemented, what's stub, what's deferred.

- **Methodology: Multi-Agent Engineering** (~4 pages): elevate from its current §3. Roles. Sprint contracts. Proof-object discipline. Cross-model review. Failure modes the method caught — especially the 156× → 22.26× self-correction. The first-attempt-failure framing (single-model GPT 5.4 + single-model Opus 4.6 separately, both failed). Why the dialog system was necessary.

- **Validation Strategy** (~2 pages): the four-tier pyramid as methodology, then evidence at each tier (savepoint parity, conservation, repeatability, ensemble consistency).

- **Results** (~3 pages): performance evidence (the table with pre-fix, iter-1, iter-2 wall-clocks and speedups, presented in chronology). Engineering invariants (D2H=0, restart bitwise, 1km headroom). Single-day AEMET skill comparison with the honest +266-440% RMSE regression characterisation.

- **The Canary Case Study** (~2 pages, as ONE Results subsection, not the framing): describe the test domain, what the Canary case tests, why it was chosen as the development workload, and what the Canary results show. Avoid making it the paper's identity.

- **Discussion** (~2 pages): what the artifact enables (differentiable physics, ML coupling, ensembles, accessibility); what the methodology says about AI-driven scientific software; limitations.

- **Open Source Release Plan**: brief section pointing at the GitHub repo, AGPL-3.0, INSTALL.md, citation, contribution model.

- **Limitations + Reproducibility + Author Contributions + AI Use Disclosure + References**: as in current draft, refreshed.

## Sources sprint #5 must cite (assembled by the history-research sprint)

- WRF technical note (Skamarock et al.)
- AceCAST documentation + any peer-reviewed comparator
- Pace (Dahm et al.) + GT4Py (Whitaker et al.) + DaCe (Ben-Nun et al.)
- ICON-exclaim (Fuhrer et al.) + ICON GPU (Lapillonne et al.)
- SCREAM (Bertagna et al.)
- NIM (Govett et al.)
- WRF OpenACC papers (the ~5-7× ceiling result)
- JAX + XLA (Bradbury, Frostig)
- GraphCast, Pangu, NeuralGCM, AIFS as ML-emulator contrast (already in references.bib)
- SWE-bench + agent-coding methodology references (Jimenez, Yang, Hong, MetaGPT, AutoGen)
- arXiv AI-authorship policy citations (Dietterich-Ginsparg 2026 if available)

## Tone notes

- Honest, declarative, not breathless. The artifact is real and impressive enough that the writing does not need to oversell.
- Treat prior GPU-port attempts with respect — they failed for hard physical and software-engineering reasons, not from lack of skill.
- Treat the AI methodology as a serious engineering contribution, not a gimmick. The 156×→22.26× self-correction is the kind of evidence that earns the right to claim it.
- Avoid "we believe" / "we think". Use "this work shows" / "the proof object at <path> records" / "validation evidence at <citation> demonstrates".

## What sprint #5 should explicitly NOT do

- Do not invent benchmark numbers. Use only what's in the proof objects.
- Do not weaken the honest skill regression. The principal accepted publishing with the honest characterisation; this is a feature of the methodology, not a bug.
- Do not put "Canary" in the title.
- Do not lead the abstract with "Canary" or "22.26×" — lead with the artifact.
- Do not name the AI co-authors as primary humans. They are AI systems with explicit AI-system disclosure in the byline.

## After sprint #5

- Sprint #6 (opus paper control) is the final quality gate.
- Then PDF render.
- Then v0.0.1 tag + push.
