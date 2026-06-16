---
name: validating-physics
description: Guides validation of physics and dycore behavior against fixtures, invariants, convergence, and ensemble consistency.
---

## When to use

Use for correctness checks, tolerance design, conservation tests, edge cases, and non-bitwise model comparison.

## Inputs required

Fixture manifest, expected output, candidate output, variable tolerances, invariant definitions, and scenario metadata.

## Workflow

1. Start with tier 1 fixture or analytic oracle.
2. Add tier 2 invariant checks.
3. Use tier 3 short-run convergence for coupled behavior.
4. Use tier 4 ensemble consistency for chaotic full-model output.
5. Record tolerances before evaluating final candidate.

## Hard rules

- No bitwise requirement unless declared debug mode.
- No tolerance chosen only to pass current output.
- No physics claim without a proof object.

## Deliverables

Comparison report, invariant result, tolerance rationale, validation artifact.

## Validation

Run the relevant comparison script and record exact command and result.

## Common failure modes

Tolerances after the fact, ignoring conservation, comparing unstaggered with staggered fields, and overclaiming short tests.
