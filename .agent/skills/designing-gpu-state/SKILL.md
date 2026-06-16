---
name: designing-gpu-state
description: Guides design of device-resident state, memory layout, halos, and transfer audits for the GPU-native model.
---

## When to use

Use when defining `GridSpec`, `State`, halo contracts, memory layout, device ownership, or transfer audits.

## Inputs required

Grid dimensions, field list, staggering, backend candidates, precision mode, halo width, and target hardware.

## Workflow

1. Define interface contract before code.
2. Choose layout only through ADR or bakeoff evidence.
3. Make state residency explicit.
4. Define halo ownership and boundary updates.
5. Add transfer audit before performance claims.

## Hard rules

- No hidden host/device transfers in timestep loops.
- Do not design for multi-GPU before single-GPU proof.
- Do not optimize layout without fixture and profiler context.

## Deliverables

State/grid contract, layout rationale, halo contract, transfer audit result.

## Validation

Dummy timestep loop shows residency and transfer audit result.

## Common failure modes

CPU-owned metadata forcing copies, ambiguous staggering, layout chosen by habit, and premature multi-GPU design.
