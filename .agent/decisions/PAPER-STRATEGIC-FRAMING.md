# Paper Strategic Framing — what matters, who reads it, what they need

**Status**: manager strategic note, 2026-05-27. Feeds sprint #5 (paper rewrite).
**Author**: Claude Opus 4.7 (manager).

## The single most important sentence

> A JAX-native, GPU-resident, open-source code that implements the WRF ARW dynamical core and a minimum operational physics suite now exists, works, and runs a real regional forecast on a single consumer-grade GPU.

Everything in the paper should orbit that sentence. Performance numbers, methodology, AI-agent story, Canary skill: all of these are evidence that the sentence is true and useful. None of them is the headline.

## Three pillars, ranked

**Update 2026-06-10 for v0.14:** the validation headline should move away from
station-only TOST. The stronger paper gate is all-field CPU-WRF vs GPU-JAX
stability: Switzerland/Gotthard 72h plus Canary L2 d02 72h, scored by the
Grid-Delta Atlas over every common numeric `wrfout` field and plotted over lead
time. Powered TOST remains a useful station sanity appendix, not the main
equivalence claim.

### Pillar 1 (HEADLINE) — A new artifact exists

A JAX-native, whole-state-device-resident, WRF-compatible regional NWP code now exists. It is open source. Anyone with a single GPU can clone it, install it, and run a regional forecast. Before this, the practical options for GPU NWP were: a) commercial Fortran-OpenACC ports with a ~5-7x ceiling (AceCAST, OpenACC WRF), b) heroic clean-slate rewrites in C++/Kokkos requiring a national lab and a supercomputer (SCREAM, HOMMEXX), c) DSL-based ports requiring a stencil-DSL toolchain (Pace + GT4Py + DaCe, ICON-exclaim + GT4Py), or d) ML emulators that bypass the physics (GraphCast, Pangu-Weather, AIFS, FourCastNet, GenCast, Aurora, NeuralGCM hybrid). What did not exist: a clean-slate, physics-faithful, single-language Python/JAX port preserving the WRF Fortran dynamical-core structure at the savepoint level, runnable on commodity hardware, and openly modifiable. This paper introduces that.

### Pillar 2 (QUANTITATIVE EVIDENCE) — The artifact achieves measurable things

- 22.26x apples-to-apples speedup on the 24 h 3 km Canary d02 case vs 28-rank CPU WRF on the same workstation; well above the 4-8x exploratory target.
- Zero inter-kernel D2H transfer inside the forecast loop (ADR-027 constitutional invariant), proven by Nsight Systems; this is the architectural innovation that prior directive-based GPU NWP ports could not achieve without unported physics.
- 1 km full-domain forecast fits in 7.28 GB of 32 GB consumer VRAM.
- B6 savepoint parity bitwise 0.0 against WRF Fortran for the coupled small-step operators; 20260521 multi-step parity 0.0 bitwise; restart bitwise; repeatability bitwise.
- Compile-once, scan-the-loop: the entire timestep is one XLA program.
- Differentiability and ML-coupling are enabled by construction (JAX, not a side effect).

### Pillar 3 (METHODOLOGY) — The artifact was built by a governed AI-agent process

This is interesting but secondary. The paper should describe it honestly because (a) the user is co-author and the methodology is part of the result; (b) the methodology caught its own publication-blocking overclaim, which is itself evidence of the method's rigour. But it does not lead the paper. The reader does not need to believe in AI authorship to find the artifact useful.

## Audience matrix

| Audience | What they care about | What they need to see in the paper |
|---|---|---|
| Operational meteorologists (NWS, DWD, ECMWF, AEMET) | Can I use this? What does it gain me? Trust? | Honest skill table vs CPU WRF; same wrfout format; restart story; install path; license; documented limitations |
| Atmospheric research scientists | What can I do with this that I can't with WRF/ICON? | Differentiability; ML coupling possibility; physics module replacement interface; idealized-test validation; clear extension points |
| HPC / scientific computing | How is it built? Does the pattern generalize? | Whole-state device residency design; JAX/XLA fused-scan timestep; precision policy; halo placeholder; clear comparator table vs Pace/SCREAM/ICON-exclaim/NIM |
| Open-source / FAIR advocates | Is it really open? Reproducible? | Public repo URL; license; reproducibility manifest; audit script; tutorial notebooks; clear citation guidance |
| AI/ML methodology readers | Did agents really build this? What worked? | Honest methodology section; proof-object discipline; documented failures and self-corrections; what does and does not transfer to other scientific software |
| arXiv preprint reviewers | Rigour, honesty, novelty bounds, citations | Limitations called out in abstract; speedup denominator explicit; skill gap not hidden; AI-authorship policy addressed |

## Implied paper structure (for sprint #5)

Reorder + reweight from current draft:

1. **Abstract** — lead with artifact existence + open-source nature + 22.26x evidence + honest skill caveat + methodology one-liner. About 220 words. Don't say "Canary" before sentence 2.
2. **Introduction** — three pillars, the gap in the prior-art landscape, four contributions reordered (artifact first, performance second, validation third, methodology fourth).
3. **Background and Related Work** — keep, polish; ensure comparators table is concrete; ML-emulator distinction sharp.
4. **The Code: Architecture** (currently §4 "Methods: Numerical Port") — promote and expand. C-grid, mass coordinate, RK3 + acoustic substep, whole-state residency, fused-scan timestep, halo placeholder, precision policy, savepoint parity strategy. Make this the centerpiece. The reader should be able to understand the design from this section alone.
5. **The Code: Physics** (currently §5) — keep; Thompson + MYNN + RRTMG + surface; what's implemented and what's stub.
6. **Validation Strategy and Results** — combine current §7 (performance) and §8 (skill). Re-lead with the four-tier pyramid as a methodology; then summarize evidence at each tier; then present the Canary case study as one realisation. Move detailed pre-fix/iter-1/iter-2 tables to a sub-subsection or appendix; they matter but they are not the headline.
7. **Methodology: AI-Agent Engineering** (currently §3) — keep, but move to a later position. Title it something like "Methodology: How the Code Was Built." Be specific and concise. Sprint contracts, proof-object discipline, cross-model review, documented failures.
8. **Hardware and Software Setup** (currently §6) — keep, but compress. This is for reviewer reproducibility, not the headline. Push toward §11 area.
9. **Open Source Release Plan** (NEW) — repo layout, license, citation, contributing, tutorials, CI. This is what makes "open source" a real claim and not a buzzword.
10. **Discussion** — what the artifact enables (ML coupling, ensembles, differentiable physics, operational accessibility); what it doesn't yet do.
11. **Limitations** — keep, but tightened around the post-iter-2 state.
12. **Reproducibility** — env manifest + proof-object manifest + audit script + commit hash.
13. **Author Contributions and AI-Use Disclosure**
14. **Acknowledgements and References**

## What should NOT be in the paper as currently positioned

- "Canary Replay Prototype" should not be in the title. It is one case study. The title should claim the artifact.
- The 22.26x speedup should not be the abstract's first numeric. The first numeric should be "32 GB consumer GPU" + "open source" + "single Python language" — those communicate accessibility. Speedup is the second numeric.
- The pre-fix 50.20x cautionary tale, while honest and valuable, is a Discussion item, not a Results item. It belongs in §6 or §10. Currently it occupies too much of §7.
- The AI-authorship debate should be in §13, not §3. Reviewers should encounter the science first.

## What should be added that is currently missing

- A concrete "Open Source Release Plan" section. Without this, "open source" is a phrase.
- Idealized-case validation (warm bubble, density current, mountain wave). The community will want to see these. Currently the paper relies entirely on Canary station scoring; that is too narrow. Sprint #3 (testing-plan execution) will produce this evidence.
- Conservation budget evidence (mass, energy, water-vapour). Also currently missing.
- An explicit "what this enables" paragraph in the Discussion: differentiable physics, ensemble economics, parameterization development, hybrid ML coupling, accessibility to non-supercomputing users.

## Honest note to self

The artifact does not yet match CPU WRF on skill. That is documented and must remain documented. But the artifact is real, runs, validates against the dycore at the savepoint level, and is reachable by anyone with a single consumer GPU. That is publishable as a preprint with honest framing. The next test-plan-execution sprint will broaden the validation evidence; that strengthens but does not change the headline.

The user's instinct on the focus shift is right. The Canary forecast was the proof that the artifact works on a real problem. It is not the artifact.

## Action items this note generates

- Sprint #5 (paper rewrite) should treat this note as the editorial brief.
- Sprint #3 (testing-plan execution) should produce idealized-case + conservation evidence, because the paper's strongest version needs that.
- The "Open Source Release Plan" section needs concrete content: license decision, repo URL placeholder, citation guidance. The user has not yet committed to the license — flag this for the user's attention at submission.
- The bottleneck analyses already produced (codex + agy) inform a future optimization paper or release-notes appendix, but should not be folded into the main paper. They are a roadmap, not a result.
