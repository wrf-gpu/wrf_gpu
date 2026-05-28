# AI Use Disclosure

This document discloses the role of AI systems in the authorship of `wrf_gpu` and its accompanying preprint. It accompanies and is referenced by the preprint, [CITATION.cff](CITATION.cff), and [CONTRIBUTING.md](CONTRIBUTING.md). It is intended to satisfy the disclosure expectations of academic preprint servers and journals that require explicit declaration of generative AI use.

## Summary

`wrf_gpu` was authored substantially by AI systems supervised by a human senior author. The validation outputs in `proofs/`, the tables in `tables/`, the figures in `figures/`, and the preprint in `paper/` were all produced under a structured multi-agent workflow.

| Role | AI system | Human equivalent |
|---|---|---|
| Long-context manager + orchestrator + first reviewer | Claude Opus 4.7 (Anthropic) with 1M-token context | Principal investigator running a small lab |
| Implementation worker + critical reviewer | GPT-5.5 Codex (OpenAI) | Postdoctoral researcher executing scoped tasks |
| Independent tester | Claude Opus 4.7 (Anthropic) acting in a tester role | Internal QA |
| Third-model adversarial review | Gemini 3.5 Flash (Google) | External reviewer |
| Senior corresponding author | — | Enric R.G. (human) |

The human author defined the scientific scope, the validation gates, the milestone definitions, the licensing decision, and final acceptance authority. The AI systems wrote the code, ran the tests, produced the proof objects, drafted the manuscript, and self-corrected their own publication-blocking overclaim.

## What the AI systems did

- Drafted the dynamics, physics, runtime, validation, IO, and integration modules in `src/gpuwrf/`.
- Wrote the unit-test suite in `tests/`.
- Wrote the orchestrator scripts in `scripts/`.
- Authored the per-sprint contracts that froze interfaces, named acceptance criteria, and listed required proof objects.
- Produced the proof JSONs in `proofs/` from real measurements (Nsight Systems traces, AEMET station scoring, wall-clock timings, savepoint comparisons).
- Drafted, critiqued, and revised the preprint manuscript in `paper/paper.md`.
- Reviewed each other's work in tester / critic roles, recorded the disagreements, and resolved them on disk.

## What the human author did

- Set the scientific target (a Python/JAX-native WRF-compatible GPU port for a single workstation).
- Defined the validation gates (the four-tier pyramid; the operational T2 / U10 / V10 RMSE tolerance; the constitutional invariant that D2H inter-kernel transfers must be zero).
- Provided the CPU WRF reference operational system that served as the speedup denominator.
- Made the licensing decision (AGPL-3.0-or-later).
- Provided the strategic pivots that re-shaped the work, including the pivot from "Canary forecast prototype" to "first open-source GPU port of WRF" as the publication framing.
- Took final responsibility for the manuscript, the release tagging, and the public submission to arXiv.

## How the workflow guarded against AI failure modes

A safety-critical scientific codebase cannot be treated like an ordinary feature backlog. The workflow encoded the following structural defences:

1. **Proof objects on disk** — no AI sprint was considered complete until a machine-auditable JSON or markdown artefact existed in the repository documenting the claim. Chat-conversation summaries were not accepted as evidence.

2. **Sprint contracts** — every implementation sprint started from a contract that named file ownership, acceptance criteria, validation commands, and a decision token. Workers could not edit outside their owned paths.

3. **Cross-model review** — the worker (one AI system) was always reviewed by a tester (a different AI system, where possible) before a sprint was accepted. The disagreements were captured on disk.

4. **Self-correction loops** — when an AI sprint reported done, an independent sprint was dispatched to re-run the validation commands. This is the loop that caught the original 156× speedup claim as inflated; the corrected number is 22.26×.

5. **Bitwise-equal invariants** — five hard invariants (B6 savepoint parity, 20260521 multi-step parity step-2 bitwise, restart bitwise, repeatability bitwise, D2H = 0 inside the forecast loop) were defined early and enforced through every subsequent code change.

6. **Honesty audit** — every quantitative claim in the preprint must point to a proof object on disk. Claims without backing were either removed or softened.

## Failures and corrections that were caught by the workflow

These are documented in the preprint and in the sprint reports archived in the development repository.

- The initial M7 closeout reported a 156× speedup that was incorrect. A self-initiated honest-speedup sprint corrected the denominator and reduced the published figure to 22.26×.
- A device-to-host transfer count of 89 was initially reported as an invariant violation. A parallel architecture audit and an empirical bisection sprint discovered that the count was a profiler-window placement artefact, not a real violation; the recapture confirmed zero inter-kernel D2H.
- The original 1-day AEMET scoring produced finite numbers and was incorrectly treated as a skill claim. A side-by-side comparison sprint revealed a +243-440% RMSE regression vs CPU WRF; the limitation is documented in the preprint and in the README.

## Limitations of AI-driven authorship that the maintainer acknowledges

- The validation discipline is stronger than ordinary chat-based coding but is not equivalent to independent human numerical-methods review by a domain expert. The skill regression caught by the cross-AI audits would have been caught earlier by a human meteorologist familiar with operational verification protocols.
- The AI co-authors cannot accept legal accountability, cannot consent, and cannot hold copyright. The human senior author holds final responsibility.
- AI co-authorship is currently a contested practice. Some preprint servers (notably arXiv since 2026-05) and academic journals discourage listing AI systems as named authors. This project lists them transparently in the byline of the preprint as AI systems with an explicit disclosure, and accepts that some venues may require moving them to an acknowledgement-only role.

## How to cite the AI contributions correctly

When citing this project or the preprint, name the AI systems explicitly and disclose their use in the cited manuscript. Do not anonymise the AI co-authors. The point of this project, in part, is to document that proof-object-driven AI authorship is possible in a numerical-physics setting and to make that documentation auditable.

See [CITATION.cff](CITATION.cff) for the machine-readable citation entry.
