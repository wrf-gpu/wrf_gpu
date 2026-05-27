# Peer Review Report: Whole-State Device Residency for Workstation-Scale NWP: A JAX-Native WRF-Compatible Canary Replay Prototype Engineered by Collaborative AI Systems

## PART A — TOP-LEVEL REVIEW

**Executive Judgment:** This preprint presents an innovative, JAX-native, WRF-compatible regional numerical weather prediction (NWP) replay prototype developed via a structured multi-agent workflow. The paper honestly details both substantial speedups and critical meteorological skill regressions. While the work is of high technical interest to the scientific software engineering and computational physics communities, it is **NOT** acceptable for publication on arXiv in its current form. I recommend **ACCEPT WITH MAJOR REVISIONS**.

**Central Scientific Claim:** The central scientific claim is that a regional weather model can be rewritten in JAX to run entirely on a single consumer GPU (NVIDIA RTX 5090) with whole-state device residency, achieving a 22.26x (Iteration 2) speedup over a 28-rank CPU WRF baseline for a 24 h Canary Islands 3 km grid. This claim is well-supported by timing logs, Nsight profiler reports, and memory usage metrics. However, the secondary scientific claim regarding the physical validity of the forecast is severely undermined by a +402% T2 RMSE skill regression, showing that the physical coupling between the land-surface and boundary-layer dynamics is currently broken.

**Central Methodological Claim:** The central methodological claim is that a governed multi-agent engineering workflow (utilizing Claude Opus 4.7 as manager/reviewer, GPT-5.5 Codex as worker/critic, and independent tester/reviewer roles) can successfully write, compile, and debug complex physical dynamics pipelines, identify its own publication-blocking overclaims, and systematically execute recovery. This claim is well-defended by on-disk sprint contracts, proof-object files, and the documented chronology of the project.

**Honesty of Skill Regression:** The characterization of the skill regression (+402% T2 RMSE, +213% U10, +177% V10) is highly credible, scientifically rigorous, and necessary. It does not read as overhumble; rather, it is a transparent and honest disclosure that prevents scientific overclaiming. It highlights the crucial lesson that high computational speedups can easily coexist with physically invalid meteorological states.

**Key Blockers for Acceptance:**
1. **AI Byline Authorship:** Listing Claude Opus 4.7 and GPT-5.5 Codex as co-authors in the byline violates arXiv's submission standards (Dietterich & Ginsparg, 2026) and standard publisher guidelines (COPE). The AI systems must be removed from the byline and metadata, with their contributions moved to a disclosure section, leaving Enric R.G. as the sole human author.
2. **Validation Sample Size:** A single-day (24 h) verification on 73 stations is statistically insufficient to prove physical validity or characterize systematic errors. The paper must frame this limitation more prominently as a preliminary engineering case study.
3. **Audit Script Failure:** The paper fails its own audit script due to non-ASCII em-dashes on lines 245 and 257 in `paper.md`, indicating a lack of basic document quality verification before submission.

---

## PART B — DETAILED REVIEW

### 1. Section-by-Section Walkthrough

#### Title, Author List, and Metadata
1. **Strongest sentence:** "Enric R.G. (human senior corresponding author)." — Assigns clear accountability to the human principal.
2. **Weakest sentence:** "Claude Opus 4.7 (AI system); GPT-5.5 Codex / OpenAI (AI system)." — Contradicts publisher standards regarding legal and moral accountability.
3. **Unsupported claim:** The implicit claim that AI models can be listed as co-authors under current arXiv and academic policy.
4. **Missing prior art:** None.
5. **Obvious error:** Including AI systems in the author list, which violates arXiv's moderation policy.
6. **Manager Pass Suggestion:** Remove the AI systems from the byline and list Enric R.G. as the sole author.

#### Abstract
1. **Strongest sentence:** "A faster pre-fix path had completed in 324.78 s and appeared to offer 50.20x speedup, but the validation workflow identified it as an overclaim episode because the operational physics path was defective." — Lucidly explains the self-correcting contribution.
2. **Weakest sentence:** "The scientific result is therefore not an operational WRF replacement." — Too blunt; could be framed as "not yet suitable for operational forecasting due to localized surface-level skill regressions."
3. **Unsupported claim:** The utility of "retained B6 savepoint and multi-step parity evidence" is not justified, given that the model still exhibits massive station skill regressions.
4. **Missing prior art:** None.
5. **Obvious error:** Mixing Iteration 1 numbers (708.32 s / 23.02x) and Iteration 2 numbers (732.63 s / 22.26x) in a way that obscures which is the current headline.
6. **Manager Pass Suggestion:** Standardize the abstract metrics around the Iteration 2 headline (732.63 s / 22.26x), and relegate Iteration 1 to historical recovery context.

#### 1. Introduction
1. **Strongest sentence:** "The second question is methodological. Recent code models can make repository-scale edits... but safety-critical scientific software cannot be treated like an ordinary feature backlog." — Excellent contextualization of LLM risks in science.
2. **Weakest sentence:** "A model that silently invents a formula, citation, unit conversion, or validation gate can move quickly in the wrong direction." — Slightly colloquial; could be rephrased for a scientific audience.
3. **Unsupported claim:** "The warm timing window measures `run_forecast_operational` plus `block_until_ready`..." — Introduces implementation-specific code terms before the code structure has been described.
4. **Missing prior art:** Could introduce exascale C++/Kokkos or GT4Py work (e.g., SCREAM, Pace) earlier in the intro.
5. **Obvious error:** Overlaps Iteration 1 and Iteration 2 speedups without a clear transition.
6. **Manager Pass Suggestion:** Polish the tone in paragraph 3 to be more academic (e.g., "A model that silently introduces mathematical or physical errors...").

#### 2. Background and Related Work
1. **Strongest sentence:** "The validation strategy instead requires per-operator parity where it is useful, physical invariants where they are decisive, and forecast-skill evidence where the system must make external claims." — Well-scoped validation philosophy.
2. **Weakest sentence:** "...but the public citation is not a peer-reviewed end-to-end benchmark and is treated here as context rather than as a hard comparator \cite{tempoquest2025acecast}." — Dismissive of commercial efforts without demonstrating their specific technical failures.
3. **Unsupported claim:** Referring to "the provided research brief" in Section 2.2 exposes internal development assets and metadata.
4. **Missing prior art:** Missing citations on standard Canary Islands meteorology (e.g., trade-wind inversion studies) and local AEMET forecast setups.
5. **Obvious error:** The leakage of the phrase "provided research brief" on line 40.
6. **Manager Pass Suggestion:** Remove the phrase "the provided research brief and" to maintain a professional, self-contained academic format.

#### 3. Methods: AI Collaboration Model
1. **Strongest sentence:** "The method worked because both the failure and the correction were anchored to files rather than conversational confidence." — A profound insight into LLM-driven engineering.
2. **Weakest sentence:** "Separate tester and reviewer roles challenged numerical, performance, and completion claims." — Vague. Were these separate agent prompts, or different API calls of the same models?
3. **Unsupported claim:** The claim of "tester and reviewer roles" acting as independent checks lacks details on their specific configuration or temperature settings.
4. **Missing prior art:** Multi-agent frameworks in software engineering (e.g., AutoGen, ChatDev).
5. **Obvious error:** The contract excerpt in Section 3.2 uses YAML formatting but does not explain how the agent framework parses or enforces it.
6. **Manager Pass Suggestion:** Clarify the scaffolding framework (e.g., custom Python harness, AgentOS) used to execute the multi-agent taxonomy.

#### 4. Methods: Numerical Port
1. **Strongest sentence:** "Hydrostatic mass and pressure-gradient-sensitive paths use FP64 where the validation history demands it." — Demonstrates necessary precision rigor.
2. **Weakest sentence:** "The current single-GPU implementation includes a halo placeholder for future multi-GPU work." — Calling it a "placeholder" makes the architecture look unfinished.
3. **Unsupported claim:** "retained B6 savepoint and multi-step CPU parity on the 20260521 case at 0.0 bitwise for 2, 5, and 10 steps." — Does not explain the disconnect between short-step parity and long-step meteorological drift.
4. **Missing prior art:** Pace's dynamical core or SCREAM's vertical coordinate representation.
5. **Obvious error:** Achieving "bitwise parity" at step 10 but ignoring the cumulative rounding and feedback accumulation that leads to skill failure.
6. **Manager Pass Suggestion:** Explain the mathematical limits of savepoint parity in chaotic physical systems and how feedback loops amplify minor coupling discrepancies.

#### 5. Methods: Physics Suite
1. **Strongest sentence:** "The physics implementation is not presented as a validated replacement for the full WRF physics suite." — Appropriately sets scope.
2. **Weakest sentence:** "DailyPipelineConfig.radiation_cadence_steps defaulted to 999999, so RRTMG was not invoked in the 8640-step 24 h integration." — Exposes a critical verification failure in the pre-fix path.
3. **Unsupported claim:** "surface_adapter computed theta_flux, qv_flux... but those values did not feed the PBL bottom boundary in the correct order." — Vague. What was the incorrect order and how did it affect stability?
4. **Missing prior art:** Porting physics columns to GPUs (e.g., WRF-CUDA or WRF-OpenACC microphysics).
5. **Obvious error:** The choice of a 10 s timestep for a 3 km grid is extremely small compared to WRF defaults (typically 15-18 s). If the CPU WRF comparison did not use the same timestep, the speedup is artificially inflated.
6. **Manager Pass Suggestion:** Confirm that the CPU WRF baseline used the identical 10 s timestep, and clarify how fluxes were incorrectly ordered in the pre-fix adapter.

#### 6. Hardware and Software Setup
1. **Strongest sentence:** "The runtime path is Python and JAX/XLA... Python 3.13.11, JAX 0.10.0, jaxlib 0.10.0..." — Highly specific environment documentation.
2. **Weakest sentence:** "The 1 km memory audit used a derived full-domain synthetic state and reported 7278 MiB... leaving approximately 78 percent of the reported GPU memory unused..." — Synthetic state scaling does not guarantee memory layout stability during physical dynamics.
3. **Unsupported claim:** The memory headroom claim does not account for transient memory spikes during dynamic halo exchanges or multi-GPU communication.
4. **Missing prior art:** Standard benchmarking hardware specifications in scientific computing.
5. **Obvious error:** JAX 0.10.0 is cited without context. Even though verified locally, explaining this atypical version prevents confusion.
6. **Manager Pass Suggestion:** Add a footnote explaining that JAX 0.10.0 is the active, verified library version in the development environment.

#### 7. Results: Performance and Systems Evidence
1. **Strongest sentence:** "Table 1 separates the pre-fix diagnostic path from the current post-fix corrected-physics path." — Excellent tabular layout.
2. **Weakest sentence:** "Apples-to-apples d02-only speedup... 23.02x... 22.26x... 138.24x, not apples-to-apples..." — The inclusion of "not apples-to-apples" in the main results table is confusing and clutters the data.
3. **Unsupported claim:** The speedup is presented as "apples-to-apples," but JIT compilation overhead (100+ seconds) is excluded from the headline speedup calculation.
4. **Missing prior art:** Research on JIT compilation bottlenecks in operational weather systems.
5. **Obvious error:** Table 1 lists 23.02x as the "post-fix corrected-physics path" (Iteration 1) but does not include Iteration 2 (22.26x) performance metrics, creating a discrepancy with Section 8.4.
6. **Manager Pass Suggestion:** Add Iteration 2 (732.63 s, 22.26x) as a separate row group in Table 1 to align with Section 8.4 results.

#### 8. Results: Forecast Quality and Skill
1. **Strongest sentence:** "All three variables remain outside the pre-declared 20 percent tolerance... The publication therefore continues to reject any operational replacement claim." — Maintain scientific integrity.
2. **Weakest sentence:** "T2 regressed: with the envelope widened from 400 K to 450 K... surface flux magnitudes from the current surface_adapter plus MYNN coupling now over-deposit heat into the bottom level." — A qualitative description of a physical coupling error without equations.
3. **Unsupported claim:** "The diurnal warming overshoots rather than saturating." — No plot or statistical evidence is shown for this "overshoot" behavior, only the RMSE table.
4. **Missing prior art:** Standard verification protocols for regional meteorological models (e.g., AEMET or WMO standards).
5. **Obvious error:** The table in Section 8.4 labels the middle column as "GPU iter-1 RMSE", but the numbers shown (7.86, 11.31, 9.44) are the *pre-fix* numbers, not the actual Iteration 1 numbers (8.85, 6.75, 7.23).
6. **Manager Pass Suggestion:** Fix the Section 8.4 table column label or numbers to prevent critical data misrepresentation.

#### 9. Discussion
1. **Strongest sentence:** "In a field where numerical trust matters more than demonstration speed, that self-correction is part of the result." — Excellent summary of the project's engineering philosophy.
2. **Weakest sentence:** "A single human plus autocomplete workflow might have moved from the inflated closeout directly to public communication." — Speculative and sets up a strawman comparison.
3. **Unsupported claim:** "The current 23.02x corrected-physics d02 throughput ratio is large enough to matter..." — Speed does not matter if the physical forecast is invalid.
4. **Missing prior art:** Standard reviews of software engineering in scientific modeling.
5. **Obvious error:** Cites 23.02x as the current speedup instead of Iteration 2's 22.26x.
6. **Manager Pass Suggestion:** Align the discussion with the Iteration 2 results (22.26x speedup).

#### 10. Limitations
1. **Strongest sentence:** "The validation corpus is too small... one 24 h 20260521 case." — Lucidly discloses validation limits.
2. **Weakest sentence:** "The remaining blockers are now narrower..." — A +402% T2 RMSE is not a "narrow" blocker; it is a fundamental thermodynamic failure.
3. **Unsupported claim:** The choice of boundary relaxation zone size (`spec_bdy_width=5`) is stated to contribute plausible behavior without comparative sensitivity metrics.
4. **Missing prior art:** Standard boundary relaxation techniques (e.g., Davies relaxation).
5. **Obvious error:** Discloses that the land-surface state is refreshed hourly via data replay rather than run prognostically, but fails to highlight this major limitation in the abstract or introduction.
6. **Manager Pass Suggestion:** Move the land-surface data replay limitation to the Abstract and Introduction.

#### 11. Reproducibility
1. **Strongest sentence:** "The canonical proof-object manifest for this paper is: ... [lists files]" — Extremely concrete.
2. **Weakest sentence:** "The lightweight audit command is: taskset -c 0-3 bash scripts/m7_publication_audit.sh" — The script currently fails on the paper draft due to non-ASCII em-dashes.
3. **Unsupported claim:** The repository URL and commit hashes are listed as TBD, which is unacceptable for a preprint claiming immediate reproducibility.
4. **Missing prior art:** Software citation guidelines.
5. **Obvious error:** Non-ASCII em-dash characters on lines 245 and 257 break the audit script.
6. **Manager Pass Suggestion:** Replace em-dashes with standard ASCII hyphens or configure the audit script to support UTF-8.

#### 12. Author Contributions and AI Use Disclosure
1. **Strongest sentence:** "The AI systems cannot approve the final manuscript, hold legal accountability, or satisfy human-only authorship criteria." — Correctly aligns with COPE guidelines.
2. **Weakest sentence:** "This draft names Claude Opus 4.7 and GPT-5.5 Codex / OpenAI as AI systems, not as human authors." — Contradicts their listing in the main metadata byline.
3. **Unsupported claim:** The description of the cognitive labor division lacks a verifiable method of human validation.
4. **Missing prior art:** COPE and ICMJE guidelines on AI use.
5. **Obvious error:** The byline listing violates arXiv's generative AI policy.
6. **Manager Pass Suggestion:** Move the AI systems entirely out of the metadata/byline and into this section as a disclosure.

#### 13. Acknowledgements
1. **Strongest sentence:** "The authors also acknowledge the repository's internal reviewer and tester roles, which forced the correction..." — Nice recognition of the multi-agent framework.
2. **Weakest sentence:** "The project depends on the WRF and NCAR modeling community..." — Standard but could be more specific.
3. **Unsupported claim:** "retained Gen2 operational Canary forecasting system" — Mentions this system without citation or description.
4. **Missing prior art:** None.
5. **Obvious error:** Referencing the Gen2 system without defining it.
6. **Manager Pass Suggestion:** Add a brief citation or footnote describing the Gen2 baseline.

---

### 2. Consolidated Playbook of Corrections

#### Must-Fix Items (Top 6)
1. **Byline Authorship:** Remove Claude Opus 4.7 and GPT-5.5 Codex from the author byline and metadata. Register Enric R.G. as the sole human author to comply with arXiv policies (Dietterich & Ginsparg, 2026).
2. **Data Mismatch in Section 8.4 Table:** Correct the middle column of the Iteration 2 table. It is currently labeled "GPU iter-1 RMSE" but contains the *pre-fix* RMSE values (7.86 K, 11.31 m/s, 9.44 m/s) instead of the actual Iteration 1 values (8.85 K, 6.75 m/s, 7.23 m/s).
3. **Audit Script Failure:** Resolve the non-ASCII em-dash `—` characters on lines 245 and 257 in `paper.md` which cause `scripts/m7_publication_audit.sh` to fail.
4. **Table 1 Consistency:** Update Table 1 (Performance Summary) to include Iteration 2 performance numbers (732.63 s wall-clock, 22.26x speedup) so that it matches the latest state of the codebase.
5. **Internal Leak in Section 2.2:** Remove the phrase "the provided research brief and" on line 40, which exposes internal development materials.
6. **Abstract Numeric Alignment:** Align the speedup and wall-clock numbers in the abstract to reflect the Iteration 2 headline (732.63 s / 22.26x) rather than Iteration 1 (708.32 s / 23.02x), keeping Iteration 1 in the background history.

#### Should-Fix Items (Top 5)
1. **Clarify JAX Version:** Add a footnote explaining the use of JAX `0.10.0`, as it deviates from the typical public versioning history of JAX (which is in the `0.4.x` range at the time of writing).
2. **Disclose Data Replay Limitation:** Explicitly state in the Abstract and Introduction that the land-surface state is refreshed hourly via data replay from CPU WRF outputs, rather than run prognostically.
3. **Explain Short-Step vs Long-Step Disconnect:** Add a paragraph explaining why the model achieved 0.0 bitwise savepoint parity for 2, 5, and 10 steps, yet diverged to massive skill regressions after 24 hours (accumulated feedback loops, surface flux coupling order, etc.).
4. **Apples-to-Apples Timestep Check:** Disclose whether the CPU WRF baseline also used the small `dt = 10 s` timestep. If CPU WRF used a larger timestep (e.g., 15 s or 20 s), the 22.26x speedup is inflated because CPU WRF was run with sub-optimal parameters.
5. **Detail the T2 Overshoot Mechanism:** Expand Section 8.4 to physically explain the T2 overshoot regression (+402% RMSE) after widening the theta envelope. Provide the coupling equations between MYNN and the surface adapter.

#### Nice-to-Have Items
1. **Academic Tone Polish:** Replace informal sentences in Section 1 and Section 9 (e.g., "A model that silently invents... can move quickly in the wrong direction") with formal scientific language.
2. **Metadata Checks:** Perform the cited release-time publisher metadata verification checks on the 16 bib entries flagged in the critique.
3. **Multi-GPU Plan:** Expand Section 10 to include a brief architectural path for multi-GPU scaling to show that the "halo placeholder" is not just dead code.
4. **Metplus Integration:** Elaborate on the planned integration of METplus, FSS, and SAL in future milestones to show the roadmap for Tier 4 validation.

---

### 3. Broad Peer Review Concerns

#### AI-Authorship Policy at arXiv
arXiv's Administrative Board tech report (Dietterich & Ginsparg, 2026) and report in PCMag (2026) outline severe penalties for submitting papers with unverified AI-generated content or improper AI authorship. Preprints that list AI systems as co-authors in the byline will be flagged by automated moderation systems and rejected. AI models cannot assume legal or scientific responsibility for the work.

#### Reproducibility Concerns
Leaving the repository URL, release commit, and proof-object commits as "TBD" in Section 11 contradicts the claim of a "reproducible prototype." The final commit hashes must be frozen and committed before submission. The raw Nsight profiler binary (e.g., `d2h_audit_v2.json` source) must be hosted in an external repository (e.g., Zenodo) since it is missing in the local checkout.

#### Validation Rigor
The entire meteorological skill verdict relies on a single day (20260521) for a single domain. Weather models are highly sensitive to initial conditions and seasonal regimes. A single-day verification is statistically insignificant and cannot prove physical validity. The authors must frame the skill score as a "preliminary diagnostic case study" and avoid claims of "viable physics" until a multi-week ensemble is evaluated.

#### Novelty Bounds
The paper claims a "clean-slate rewrite... achieving complete, zero-in-loop-transfer device residency." This must be bounded against existing work: Pace (pyFV3) achieved Python/DSL portability; ICON-exclaim ported ICON using GT4Py; SCREAM ported the E3SM atmosphere in C++/Kokkos; NIM was an early GPU weather model; and AceCAST is a commercial GPU WRF port. The paper's novelty is not being the first GPU weather model, but being a JAX-native, compilation-fused, workstation-scale WRF-compatible prototype.

#### Meta-Question: Should AI systems be in the byline?
No. AI models do not meet the legal, moral, or professional definitions of authorship (e.g., COPE guidelines). They cannot consent, hold copyright, or be held legally liable for scientific fraud. Therefore, listing AI systems in the byline is improper. They should be moved to the Acknowledgments and a dedicated AI Use Disclosure section, while only human contributors who take responsibility are listed as authors.
