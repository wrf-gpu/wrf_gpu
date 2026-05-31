# Paper v1 — Ideas & Structure Draft

Status: **thinking draft, 2026-05-31.** Not the paper. A reasoned argument for what the
paper's core messages should be and how to structure them, for the principal (first author)
to react to. Written by the manager agent — and that recursive fact (the paper about
AI-built software is itself AI-drafted) is part of the integrity story, see §6.

---

## 0. The one-sentence thesis

> **An autonomous multi-agent AI system planned, implemented, debugged, and rigorously
> validated a GPU-native reimplementation of a production atmospheric model (WRF v4) — and
> because numerical weather prediction has objective ground truth, we can *prove* the result
> is correct, not merely assert that "AI built it."**

The power is in the *conjunction*. Neither half alone is a strong paper:
- "AI builds software" — done often, usually on toy or unverifiable tasks → anecdote.
- "A JAX GPU weather model" — interesting systems work, but incremental against the broader
  GPU-NWP / ML-weather wave → a niche tools paper.
- **Together, in a domain with hard oracles → a novel, *falsifiable* claim about what
  autonomous AI can do in real, correctness-critical scientific computing.** The verifiability
  is what turns the AI story from a demo into science. That linkage is the intellectual core
  and should be foregrounded everywhere.

---

## 1. What actually makes this arXiv-worthy (my independent read)

I agree with the principal's instinct that the **AI-orchestration is the most novel
contribution** — and I'd sharpen *why*: it is the only one of our contributions that is both
(a) a genuine "first at this scale/rigor" and (b) credible *specifically because the domain is
verifiable*. Ranked:

**C1 — The methodology (primary).** A complete, hard, correctness-critical scientific artifact
was produced end-to-end by an autonomous AI swarm (Opus-class manager+frontrunner, GPT-5.5
adversarial critic, Gemini tiebreak), human as initiator/principal only. What's publishable
is not "it wrote code" but the *engineering discipline that made it trustworthy*:
  - role structure (manager / implementer / adversarial cross-model critic / tiebreak);
  - proof-object-per-claim (nothing "done" without a falsifiable artifact on disk);
  - adversarial cross-model review catching errors a single model misses;
  - **the honesty mechanisms and their receipts** — this project's own history is the evidence:
    - the "bitwise WRF parity at 100 steps" headline was a **JAX-vs-JAX self-compare**, caught
      and **publicly retracted**;
    - a **missing Coriolis force** was exposed by a *persistence baseline*, not by passing tests;
    - an inflated **"22.26× speedup"** was retracted to an honest ~5–8× after a roofline audit;
    - the **fp32-defeat bug class** (silent precision loss) found by dtype auditing the
      operational path.
  These are not embarrassments to hide — they are **reproducible demonstrations that the
  multi-agent process detects and corrects its own errors**, which is the entire credibility
  question for autonomous AI engineering. Lead with them.
  - quantify: wall-clock (~1 week for the validated core), sprint/agent counts, models used and
    where each added value (the per-model scorecard), token/compute budget.

**C2 — The artifact + the clean-rewrite-against-oracle methodology (enabling).**
  - A from-scratch **JAX/XLA GPU-native** reimplementation of the WRF dycore + physics — to our
    knowledge not previously done as a clean, oracle-validated JAX rewrite (legacy WRF is
    Fortran + OpenACC/CUDA; prior GPU efforts are incremental ports). Distinct from learned
    weather models (GraphCast/Pangu/NeuralGCM) — we reimplement the *physics-based* equations,
    not learn a surrogate.
  - The **validate-against-oracle, don't-inherit-architecture** strategy: target the GPU memory
    hierarchy from day one, prove correctness against WRF savepoints + published analytic
    benchmarks. This is a *reusable recipe* for porting legacy scientific codes — a transferable
    methodological contribution beyond WRF.
  - **Differentiability** as a structural property: a JAX model is end-to-end differentiable in
    principle, opening gradient-based DA / parameter calibration / ML-hybrid physics. State it
    honestly as a property + future-work hook; do not overclaim (we did not exercise it).

**C3 — Honest GPU performance characterization (enabling).** The publication-grade
roofline / compute-cycle analysis: the honest ~5.3× (clean) / ~7.8× (realistic) vs 28-rank
CPU-WRF on one RTX 5090, *why* fp64 split-explicit acoustic integration is bandwidth/launch-
bound and near-optimal, and **four candidate accelerations measured and refuted** (fp32
dynamics, CUDA command-buffers, fp32-Thompson, implicit sedimentation). The contribution is
the *correct number + the analysis + the refutations*, not a marketing multiplier. "How perf
claims should be made" is itself a small methodological point.

**C4 — The operational demonstration (grounding).** Near-CPU-WRF skill on real Canary Islands
3 km/1 km cases (T2, U10/V10 post-Coriolis, precip), multi-day stable, beating persistence.
This is the "it actually works on a real forecasting problem" evidence that makes C1–C3 about a
*real* artifact rather than a benchmark toy. Scientifically it's the demonstration, not the
headline novelty (regional NWP skill is well-trodden) — but without it the paper is hollow.

---

## 2. Proposed structure (efficient conveyance of C1–C4)

Working title direction (lead with the meta-message + the artifact):
*"Autonomously AI-Engineered Scientific Software: A GPU-Native JAX Reimplementation of the WRF
Atmospheric Model, Planned and Validated End-to-End by a Multi-Agent System."*

1. **Abstract** — the dual thesis (C1 × verifiability), the artifact, the honest numbers, the
   limits. One sentence each.
2. **Introduction** — the gap: (i) AI code generation is largely unverified at real scale;
   (ii) legacy scientific codes are notoriously hard to modernize; (iii) NWP is *objectively
   verifiable*. The opportunity: use a verifiable domain to falsifiably test autonomous AI
   scientific software engineering. Explicit contributions list (C1–C4).
3. **Related work** — position against four bodies: (a) learned weather models
   (GraphCast/Pangu/FourCastNet/NeuralGCM) — we reimplement physics, not learn it; (b) GPU NWP
   ports (WRF-GPU efforts, SCREAM/HOMMEXX, ICON-EXCLAIM, Pace/GT4Py) — clean JAX rewrite vs
   Kokkos/DSL; (c) JAX in scientific computing (JAX-CFD, JAX-MD, NeuralGCM's dynamical core) —
   key comparator; (d) agentic software engineering / LLM code agents (SWE-bench et al.) — we
   tackle a large, correctness-critical, oracle-verified *real* artifact, not isolated issues.
4. **Method I — the AI orchestration** *(core, not an afterthought)*. Roles, operating model,
   proof-object discipline, patch/review protocol, adversarial cross-model review, the honesty
   mechanisms. Metrics + the model scorecard. The error-detection case studies (C1 receipts).
5. **Method II — the artifact**. Architecture (SoA JAX pytree, C-grid, hybrid-eta, fp64,
   whole-state device residency, fused timestep-scale kernels). Dycore (RK3 + split-explicit
   acoustic). Physics suite. The clean-rewrite-against-oracle approach.
6. **Validation** — the tiered pyramid: micro-fixture parity → analytic invariants → idealized
   benchmarks (Skamarock warm bubble, Straka density current) vs published refs + WRF
   savepoints → operational real-case RMSE/TOST + persistence baselines (Canary 3 km/1 km). Fold
   the honesty case studies in here as *evidence the validation works*.
7. **Performance** — the roofline/compute-cycle analysis; the honest speedup; the refuted levers.
8. **Discussion** — what verifiability buys the AI claim; limitations (the gaps TODO: standalone
   init, full nesting, prognostic Noah-MP — scope precisely and honestly); reproducibility of
   *process* (transcripts, proof objects, git history) and *artifact* (code, fixtures);
   differentiability / DA / ML-hybrid future work; the release roadmap (0.1.0 → 0.2.0 → true
   port). Threats to validity + reviewer-objection preemption.
9. **Conclusion.**
- **Appendices**: proof-object index; the gap table + post-0.1.0 roadmap; operating-model docs;
  pointers to agent/role transcripts; full reproduction instructions; precision policy.

---

## 3. Reviewer objections to preempt (think like the adversary)

- *"Not a full WRF port — depends on real.exe, no live nesting, prescribed Noah-MP."* → Scope
  the claim precisely (single-domain replay forecast with WRF-faithful core); the gaps TODO is
  the preemption. **Honesty is the credibility, not a weakness.**
- *"How much was the human vs the AI?"* → Document the human role exactly (initiator; principal
  decisions at milestone boundaries; the directives), with transcripts + git history + proof
  objects as evidence. The process is reproducible.
- *"~5–8× is modest for GPU."* → The roofline analysis is the answer; the honest, near-optimal-
  faithful number + refutations *is* the contribution. Over-claiming would be the failure mode.
- *"Wind / precip skill gaps."* → Stated, bounded, roadmapped; the post-Coriolis improvement is
  itself a methodology receipt (persistence baseline → root cause → fix).
- *"Is the AISWE result general or WRF-specific?"* → The transferable claim is the *recipe*
  (clean rewrite + oracle validation + multi-agent honesty discipline) in any verifiable domain;
  WRF is the existence proof, not the boundary.

---

## 4. What evidence we already have on disk (to cite)

- Idealized gates: `proofs/f7/` (Skamarock, Straka, DYCORE_STATUS).
- Real-case skill: `proofs/m19/` (3-case verdict + persistence baselines + terrain-w resolution).
- Wind/Coriolis: `proofs/wind/` (the missing-Coriolis root-cause receipt).
- Performance: `proofs/perf/` + `publish/runtime_optimization_analysis.md` (roofline, refuted levers).
- Honesty receipts: the v0.0.1 retraction trail in README history; `proofs/thompson_perf/`.
- Gaps + roadmap: `publish/GPU_PORT_GAPS_TODO.md`, `.agent/decisions/POST-0.1.0-ROADMAP.md`.
- Process: `.agent/` (decisions, sprints, roles, milestones), git history, agent transcripts,
  the model scorecard (memory).

**Gap to fill before submission:** a clean, quantified *process dataset* (wall-clock, sprint
count, per-model contribution/error-catch tallies, token/compute budget) — currently scattered
across memory + git + transcripts. This should be assembled into one citable table for §4.

---

## 5. Open questions for the principal

- **Venue/framing.** Lead as a CS/AI-systems paper (AISWE case study) or a geoscientific-model-
  development paper (à la GMD) with the AI angle as method? My lean: **arXiv cs.SE/cs.AI primary
  with a strong geoscience validation core** — the dual-audience framing is the strength. Could
  later split into a GMD model-description paper + an AISWE methodology paper.
- **How hard to push the differentiability/ML-hybrid angle** vs keep it strictly future work
  (honest: we have not exercised it).
- **Scope of the v1 paper** — ship at v0.1.0 (single-domain replay, honest) as the existence
  proof, or wait for v0.2.0 (nesting + quality items)? My lean: **v0.1.0 paper now** — the
  methodology + verifiability thesis stands on the current artifact, and the roadmap is the
  future-work section. Waiting weakens the "what AI can do *today*" punch.

---

## 6. Integrity note

The principal is first author and initiator. The model code, the validation, the performance
analysis, and this paper draft were produced by the AI agents. The paper must **disclose this
plainly** (AI authorship of the artifact and of the draft, human as principal/initiator) — the
same honesty discipline that caught our own bugs applies to authorship. That disclosure is not a
caveat that weakens the work; it is the literal subject of the paper.
