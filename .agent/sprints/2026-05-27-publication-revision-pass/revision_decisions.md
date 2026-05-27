# Revision Decisions

Sprint: `2026-05-27-publication-revision-pass`
Source: 28 inline `<<<CRITIQUE: ...>>>` markers in `publication/draft/paper.critique.md`.

Summary: Applied 28 of 28 critique items in the revised draft, with two items carrying explicit release-time residual work: final publisher metadata verification for brief-derived references, and final public release URL/commit pinning. No inline critique item was silently ignored.

| # | Decision | Action |
|---:|---|---|
| 1 | Applied | Replaced the title's "WRF v4 Port" framing with "WRF-Compatible Canary Replay Prototype." |
| 2 | Applied | Reworked authorship language to identify Claude and Codex as AI systems and Enric R.G. as human senior corresponding author. |
| 3 | Applied | Rewrote the abstract around 708.32 s / 23.02x as the current corrected-physics result. |
| 4 | Applied | Changed contribution 2 to use the post-fix result and treated 324.78 s / 50.20x as diagnostic history. |
| 5 | Applied | Softened the AceCAST comparison and stopped using the 5-14x range as a hard peer-reviewed comparator. |
| 6 | Applied | Added COSMO/CH regional GPU precedent and stronger WRF modernization context. |
| 7 | Applied | Removed `TODO_Mollick_AI_authorship` and the sentence that depended on it. |
| 8 | Applied | Moved authorship justification out of Methods and into Section 12 disclosure. |
| 9 | Applied | Added a concrete sprint-contract excerpt from `m7-honest-speedup-skill-diff`. |
| 10 | Applied | Added post-fix proof objects to the proof-object discussion and reproducibility manifest. |
| 11 | Applied | Added the exact contract/proof basis for grid and timestep values in the numerical-port section. |
| 12 | Applied | Clarified what bitwise evidence covers and avoided treating it as full operational skill. |
| 13 | Applied | Rewrote the radiation paragraph as pre-fix disabled, post-fix cadence 180, remaining defects still open. |
| 14 | Applied | Rephrased the 1 km memory probe as `nvidia-smi` memory-used evidence, not peak process memory. |
| 15 | Applied | Added an environment manifest in Section 11. |
| 16 | Applied | Renamed and split Table 1 by pre-fix diagnostic path and post-fix corrected-physics path. |
| 17 | Applied | Removed `same` proof-object references from Table 1. |
| 18 | Applied | Made the post-fix performance rows the current headline values. |
| 19 | Applied | Rewrote the performance interpretation around 23.02x as the current apples-to-apples result. |
| 20 | Applied | Moved the combined algorithmic fix into Methods and Results before interpreting current numbers. |
| 21 | Applied | Resolved the 700.89/708.32 conflict by using 708.32 s total and 700.73 s forecast-only. |
| 22 | Applied | Replaced "architecture is correct" with "whole-state-resident architecture remains viable under current systems evidence." |
| 23 | Applied | Updated Discussion to use 23.02x as the current result and 50.20x only as pre-fix diagnostic history. |
| 24 | Applied | Rewrote Limitations around theta guard saturation, frozen land/surface state, boundary width, corpus size, no live AIFS ingest, and no independent human numerical review. |
| 25 | Applied | Removed the unresolved TODO-citation language from limitations. |
| 26 | Applied | Built out Reproducibility with public URL placeholder, commit table, environment table, proof-object manifest, and audit command. |
| 27 | Applied | Added `scripts/m7_publication_audit.sh` as the lightweight audit command. |
| 28 | Applied | Updated the Acknowledgements and future-readiness language now that RCA and partial fix have landed. |

Citation-specific decisions:

- `fredj2023adios2wrf` is now cited in WRF modernization context.
- `paredes2023gt4py` is now cited alongside GT4Py/DaCe context.
- `yang2024sweagent` was added from the provided brief's SWE-agent reference and cited in the AI-agent related-work section.
- The unresolved Mollick citation was removed, not fabricated.
- The paper now states that several brief-derived references need release-time publisher metadata checks rather than implying all metadata is final.
