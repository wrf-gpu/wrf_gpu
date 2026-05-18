# Project Spec

## v0 Target

Single-node, single-GPU, Canary-first prototype that can run a constrained WRF-compatible regional forecast path with documented physics fixtures, transfer audits, and profiler evidence.

## v1 Target

Professionally forkable GPU-native regional NWP core with stable interfaces for dycore, physics columns, fixtures, validation, profiling, restart, and WRF-compatible I/O mapping where useful.

## Compatibility Targets

- WRF-style variable names and units where practical.
- WRF namelist compatibility where it reduces operational friction.
- Fixture-level comparison to trusted WRF runs or analytic oracles.
- Documented deviations from WRF behavior and data layout.

## Non-Goals

- Full WRF feature parity in v0.
- Multi-GPU before the single-GPU proof is useful.
- Data assimilation in v0 unless needed for the Canary target.
- Bitwise reproducibility unless a specific debug mode declares it.
- A permanent backend decision before the architecture bakeoff.

## Success Metrics

- Every milestone has a proof object.
- M2 selects a backend with correctness, profiler, maintainability, and agent-success evidence.
- No timestep-loop host/device transfers except documented, approved exceptions.
- Canary v0 beats the existing CPU operational baseline enough to justify the rewrite path.
