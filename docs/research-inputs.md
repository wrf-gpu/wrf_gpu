# Research Inputs Summary

This summary compresses the three seed research files. The original files remain in the repository for audit.

## Accepted Lessons

- Incremental legacy WRF OpenACC/offload is not the plan for this repository.
- Whole-state GPU residency is mandatory for serious end-to-end speedup.
- Isolated physics-kernel speedups are not operational proof.
- Validation is non-bitwise by default and must combine fixtures, invariants, convergence, and ensemble consistency.
- Profiler artifacts are required for performance claims.
- Backend choice is not locked at bootstrap.

## Backend Evidence To Test In M2

- Python orchestration is attractive for agent velocity and ML integration.
- JAX and Triton are serious candidates for an AI-native stack.
- GT4Py/DaCe and Pace-style workflows are serious candidates for stencil-heavy model code.
- Kokkos/YAKL-style C++ has strong NWP precedent but may reduce agent iteration speed.
- CUDA-family low-level paths may be justified for NVIDIA-specific kernels, but require ADR because of portability cost.

## Validation Tools To Consider

- WRF-derived savepoints or Serialbox-style fixtures for micro parity.
- probtest/PyCECT-style ensemble consistency for chaotic output.
- Timestep convergence for short coupled runs.
- Conservation, positivity, NaN, and spectral diagnostics for invariant checks.

## Project-Specific Correction

Do not use the old global `wrf-gpu-port` skill. This repository owns its local skills under `.agent/skills`, and those files are part of the git-tracked project state.
