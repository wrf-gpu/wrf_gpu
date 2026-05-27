# Methods Critique

The Methods section is the paper's real contribution, but it needs to become more concrete and more reproducible.

## AI Collaboration Method

- The role taxonomy is clear, but it reads as a narrative of who did what. Add a process artifact: one sprint contract, one proof object, one reviewer rejection/acceptance path.
- Define ADR on first use and give two real examples: one numerical/precision decision and one release-claim decision.
- Explain cross-AI verification operationally. Who re-runs commands? What does a reviewer see? What makes a claim fail? How are disagreements resolved?
- Separate authorship disclosure from engineering method. Reviewers may reject AI-as-author claims before they evaluate the workflow.
- Add a "claim type -> required proof" table: performance -> profiler/timing, transfer -> D2H audit, skill -> side-by-side CPU/GPU/obs scoring, physics -> savepoint/invariant/equivalence evidence.

## Numerical Port Method

- The state layout needs a compact table: field groups, shape, dtype, stagger, validation-only vs operational, and proof source.
- The operator sequence should be explicit: RK3/acoustic, boundary, microphysics, surface, PBL, radiation cadence, output/restart.
- The validation-mode/operational-mode distinction is important but thin. Explain which savepoint fields are not part of operational carry and which fields remain unresolved.
- Boundary replay needs more detail: source Gen2 wrfout/wrfbdy data, side-history cadence, `bdy_width=1` limitation, and why this does not equal live AIFS ingest.
- The post-fix RCA belongs in Methods or a dedicated "repair iteration" subsection before Results, because it changes the interpretation of all performance numbers.

## Physics Method

- The physics section currently says the prototype "contains" operational physics families, but later admits radiation was disabled pre-fix and land state remains frozen. Use precise implementation status per scheme.
- Add what was validated for each physics family and what remains scaffolded. "Thompson-style" and "MYNN-style" are not enough for scientific reproducibility.
- The post-fix surface-to-PBL coupling should be described as current state, not only as a Results bullet.

## Verification Method

- Tier 4 should distinguish planned ensemble/PyCECT from currently used station scoring. The current operational failure is surface station skill, not a completed ensemble-consistency gate.
- Include sample size and temporal scope directly in Methods: one 20260521 24 h station comparison, 73 stations, 24 common valid hours, 1747 joined rows.
- Add a short statement that this corpus can reject an operational-replacement claim but cannot characterize seasonal skill.

## Reproducibility Method

- The paper needs a light verification command or manifest validator. Without it, "proof objects not chat summaries" remains philosophy rather than a reproducible method.
- Include exact environment, commit hashes, and proof-object hashes before submission.
