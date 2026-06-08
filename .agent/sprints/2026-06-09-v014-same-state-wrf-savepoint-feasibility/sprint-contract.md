# Sprint Contract: V0.14 Same-State WRF Savepoint Feasibility

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Determine the fastest reliable path to generate CPU-WRF source-derived term
savepoints for the dynamic divergence case, so the next same-state localization
sprint can compare WRF terms against JAX terms without guessing.

This is a read-only feasibility sprint. It does not patch WRF, does not edit
`src/`, and does not use the GPU.

## Inputs

- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- `proofs/v014/same_state_tendency_localization_plan.md`
- existing savepoint/test scripts under `scripts/` and `tests/`
- available WRF source/build trees under local project/data paths

## Write Scope

- `proofs/v014/same_state_wrf_savepoint_feasibility.json`
- `proofs/v014/same_state_wrf_savepoint_feasibility.md`
- `.agent/reviews/2026-06-09-v014-same-state-wrf-savepoint-feasibility.md`

No `src/` edits, no WRF source edits, no GPU.

## Required Output

The report must identify:

- exact WRF source path(s) and whether a usable build exists;
- candidate routines/files to instrument for large-step, small-step, acoustic,
  pressure, mass, PH/W, boundary, and source-tendency terms;
- whether existing savepoint scripts/tests can be reused;
- minimal patch strategy for selected h8-h14 cells and stages;
- proposed savepoint artifact schema;
- risks and estimated wall-clock;
- a concise next sprint contract outline.

## Acceptance Criteria

- CPU-only inspection completes.
- JSON and Markdown are concise and validate.
- The next sprint can start from exact paths and routines, not a broad search.
