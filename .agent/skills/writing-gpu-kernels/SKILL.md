---
name: writing-gpu-kernels
description: Guides implementation of GPU kernels and backend spikes with validation, profiler evidence, and transfer discipline.
---

## When to use

Use when writing or reviewing JAX, Triton, GT4Py/DaCe, CuPy, Numba, CUDA Tile, or explicit CUDA kernel code.

## Inputs required

Sprint contract, fixture or analytic oracle, backend target, state contract, precision policy, and profiler metrics.

## Workflow

1. Confirm interface and fixture before code.
2. Implement the smallest representative kernel.
3. Run correctness comparison.
4. Run static transfer and anti-pattern checks.
5. Profile only after correctness passes.
6. Record metrics in benchmark JSON.

## Hard rules

- No backend lock before M2 ADR.
- No host/device transfer in timestep loop.
- No performance claim without profiler artifact.
- No physics claim without validation artifact.

## Deliverables

Kernel patch, correctness report, static check result, profiler JSON when required.

## Validation

Run fixture comparison, static kernel check, and profiler command defined in sprint contract.

## Common failure modes

Overfusing before correctness, hidden CPU conversion, register spill, launch-bound tiny kernels, and unvalidated mixed precision.
