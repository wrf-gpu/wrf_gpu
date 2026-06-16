---
name: building-wrf-oracles
description: Guides creation of WRF-derived and analytic fixtures used as correctness oracles for GPU-native implementations.
---

## When to use

Use when defining fixture schemas, extracting WRF comparisons, mapping variables, or validating fixture manifests.

## Inputs required

Target routine or field, source run, variables, units, tolerances, generation command, checksums, and storage location.

## Workflow

1. Define the fixture boundary.
2. Record variables, units, shapes, staggering, and precision.
3. Generate or reference data outside git.
4. Commit manifest and checksum only.
5. Validate manifest before use.

## Hard rules

- Do not commit large WRF outputs.
- Do not use a fixture without source and tolerance metadata.
- Do not claim WRF parity from visual inspection.

## Deliverables

Fixture manifest, variable mapping, tolerance metadata, validation log.

## Validation

Run the manifest validator and at least one comparison command when data exists.

## Common failure modes

Unclear units, missing staggering, stale source run, and tolerance chosen after seeing output.
