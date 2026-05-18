# M1 Milestone Plan - WRF Oracle And Fixtures

## Milestone

M1 - WRF Oracle And Fixtures

## Objective

Create the first trusted correctness-oracle layer: fixture schema, storage policy, variable mapping seed, tolerance metadata rules, and one small validated fixture path. This milestone produces targets for M2 backend bakeoff and prevents backend experiments from optimizing against vague correctness.

## Non-Goals

- No dycore implementation.
- No physics implementation.
- No backend selection.
- No large WRF outputs committed to git.
- No broad WRF feature inventory.

## Required Proof Objects

- Fixture manifest validator output.
- At least one committed fixture manifest for an analytic or tiny WRF-derived case.
- Variable mapping table with units, staggering, and tolerance placeholders.
- Comparison command output for a tiny fixture where data is small enough to commit or externally referenced.
- Storage policy for large WRF/NetCDF/Zarr artifacts.

## Candidate Sprints

1. `m1-fixture-storage-policy`: define external artifact locations, naming, checksums, and git exclusion rules.
2. `m1-analytic-micro-fixture`: create a tiny analytic fixture and comparison command.
3. `m1-wrf-variable-map-seed`: seed the WRF-to-contract variable map for M2 stencil/column candidates.
4. `m1-tolerance-metadata-seed`: define tolerance metadata fields and review process.

## Interfaces Or Decisions Frozen

Only the fixture manifest minimum fields are frozen for M1. Backend, state layout, precision defaults, and kernel APIs remain unfrozen until later ADRs.

## Review Requirements

An independent reviewer must confirm that M1 produces enough oracle evidence for M2 without starting model implementation or over-expanding WRF scope.

## Human Decisions Needed

- Preferred external storage path for large fixtures.
- First WRF/Canary baseline run to reference.
- Whether fixture artifacts may include small JSON/CSV arrays in git for smoke tests.

## Risks

- Fixture work can expand into a full WRF instrumentation project too early.
- Tolerances may be overfit if selected after seeing candidate output.
- Large artifacts may leak into git without strict policy.
- M2 may start too early if M1 only creates schemas and no usable oracle.

## Reviewer Decision

Pending
