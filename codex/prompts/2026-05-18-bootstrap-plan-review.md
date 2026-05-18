# Codex Review Request — Bootstrap Plan, Roadmap, and First Sprint Contract

## Who you are in this review

You are an independent senior reviewer (`gpt-5.5`, reasoning `xhigh`) acting as the cross-model reviewer per `.agent/rules/cross-model-review-policy.md`. You did **not** write the manager plan. Your job is to find what is wrong, weak, or missing — not to rubber-stamp.

You are reviewing three artifacts written by the manager (Claude Opus 4.7 1M):

1. `PROJECT_PLAN.md` — the synthesis plan reconciling two research briefs with the bootstrap.
2. `.agent/milestones/ROADMAP.md` — milestone-by-milestone proof-object list and gates.
3. `.agent/sprints/2026-05-18-m1-fixture-storage-policy/sprint-contract.md` — the first implementation sprint contract for M1.

## What you must read first (in this order)

1. `PROJECT_CONSTITUTION.md` (immutable).
2. `AGENTS.md`.
3. `PROJECT_SCOPE.md`, `PROJECT_SPEC.md`, `ARCHITECTURE_PRINCIPLES.md`, `VALIDATION_STRATEGY.md`, `PRECISION_POLICY.md`, `PERFORMANCE_TARGETS.md`, `INTERFACE_CONTRACTS.md`, `RISK_REGISTER.md`, `MILESTONES.md`.
4. The three artifacts under review (listed above).
5. The two research briefs:
   - `v2 ai driven from scratch plan by deepthink.txt` (the deepthink "AI-native JAX/Triton" brief).
   - `wrf to gpu gpt5.5 deep research.pdf` (your own family's earlier deep-research brief that argued Kokkos/C++ over JAX).
6. The existing per-milestone files under `.agent/milestones/M*.md` and especially `.agent/milestones/M1-wrf-oracle-fixtures-plan.md`.
7. The role files under `.agent/roles/` and the rule files under `.agent/rules/`.

You do **not** need to read every skill file. Read only those relevant to a specific challenge you are raising.

## What to evaluate

For each artifact, find issues in these categories. Findings first, then your decision.

### A. Contradictions with the constitution and existing governance
Does anything in the plan, roadmap, or sprint contract violate or weaken `PROJECT_CONSTITUTION.md`, `PROJECT_SCOPE.md`, `ARCHITECTURE_PRINCIPLES.md`, `VALIDATION_STRATEGY.md`, `PRECISION_POLICY.md`, `PERFORMANCE_TARGETS.md`, or `RISK_REGISTER.md`? Cite file:line for any contradiction.

### B. Hidden architecture lock-in
The constitution and `ARCHITECTURE_PRINCIPLES.md` forbid backend lock before the M2 bakeoff. Does any phrasing in the plan or sprint contract de-facto lock a backend or a tooling choice? Examples to scrutinize:
- The §5 bakeoff candidate list (A–E). Is the exclusion of CUDA Fortran defensible, or does it pre-judge?
- The §6 validation tooling table. Does naming PyCECT / probtest / Serialbox lock tooling before M1 review?
- The §8 sprint sequence. Does it pre-assume that microphysics is the right first physics scheme?

### C. Evidence gaps
Where does the plan claim something without traceable evidence in the two research briefs or in the bootstrap docs? Cite the claim and the missing evidence.

### D. Missing milestones, missing gates, missing proof objects
Does the M0–M8 sequence cover what's actually needed for the Canary v0 operational target? Specifically:
- Initial conditions / boundary conditions handling — is it covered?
- I/O (NetCDF read/write, restart) — does M7 actually scope this?
- Surface / land-model coupling for Canary — covered or missed?
- Map projection + terrain ingestion — covered or missed?
- Operational verification (METplus or equivalent) — adequately scoped at M7?
- Multi-GPU readiness vs. v0 single-GPU scope — is the halo abstraction at M3 enough to not preclude it later?

### E. Ordering critique
Are sprints S1–S26 ordered defensibly? Specifically:
- Should S2 (analytic stencil fixture) start in parallel with S1?
- Should the M2 bakeoff include or exclude Tile-based CUDA C++ paradigms?
- Should a research-scout sprint on METplus run earlier than M7?

### F. M1 first-sprint contract critique
For `2026-05-18-m1-fixture-storage-policy/sprint-contract.md`:
- Is the file-ownership list too narrow or too wide?
- Are the acceptance criteria observable and falsifiable?
- Is the "schema overfitting to current placeholder template" risk adequately mitigated?
- Will the comparison-harness CLI skeleton actually generalize to Tier-1 fixtures used in M2?

### G. Risks not in the register
What top-risk items are missing from `PROJECT_PLAN.md §10` and `RISK_REGISTER.md`?

### H. Anything else
Free-form, but cite-or-cut: any claim you raise without a file or paragraph citation is rejected.

## Deliverable format

Write exactly one file: `.agent/decisions/REVIEW-codex-bootstrap-plan.md`.

It must contain, in this order:

1. **Decision** (one line): `Accept` | `Accept with required fixes` | `Reject`.
2. **Top three structural concerns** (≤200 words total). These are the items the manager must address before any worker dispatches.
3. **Findings** — a numbered list. For each finding:
   - severity: `blocker` | `major` | `minor` | `note`,
   - artifact + section/line citation,
   - the issue,
   - the required or recommended fix,
   - the evidence (file:line citation; if from a research brief, page reference).
4. **Required interface freezes** — any item the manager has left ambiguous that you believe must be frozen before workers dispatch.
5. **Dissent** — anything where you actively disagree with the manager's call (e.g. backend-bakeoff candidate exclusions, sprint sequencing).
6. **One-paragraph closing**: what to do next.

## Hard rules for you

- No fluff. No restating what the manager wrote. No "great plan." Findings first.
- No backend recommendation as a side channel — your job is to evaluate the manager's plan, not to author yours.
- No edits to source files. Read-only repository access. The only file you may create is `.agent/decisions/REVIEW-codex-bootstrap-plan.md`.
- Cite file paths and (where possible) line numbers for every claim.
- If you find no blockers, say so plainly and explain why the plan is shippable as-is.
- Budget: aim for a thorough review in one pass. Do not loop indefinitely.

When done, exit. The manager will read your deliverable and either incorporate it or document dissent before presenting the final version to the human arbiter.
