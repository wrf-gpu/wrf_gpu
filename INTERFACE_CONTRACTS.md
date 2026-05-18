# Interface Contracts

These are placeholders, not implementation commitments. Final contracts require ADR approval once the backend bakeoff finishes.

## GridSpec

Domain dimensions, staggering, map projection, spacing, vertical coordinate metadata, halo width, and device layout policy.

## State

Prognostic and diagnostic fields, units, precision, staggering, residency state, and ownership rules.

## Tendencies

Named tendency fields, valid update windows, accumulation semantics, precision, and reset policy.

## PhysicsColumnInput

Column fields, surface data, forcing, timestep, scheme options, and fixture identifiers.

## PhysicsColumnOutput

Updated tendencies, diagnostics, conservation deltas, and validation metadata.

## DycoreStepInput

State, grid, timestep, boundary conditions, tendencies, and precision mode.

## DycoreStepOutput

Updated state, diagnostics, invariant checks, and transfer audit summary.

## FixtureManifest

Fixture id, source, variables, shapes, units, tolerances, generation command, checksum, and license notes.
