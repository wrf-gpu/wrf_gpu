# Reviewer Report

## Findings

The root-cause proof is strong. WRF source anchoring shows runtime `PSFC` is
`p8w(kts)` from `grid%p_hyd_w`, not a nonhydrostatic pressure extrapolation.
The CPU formula proof reproduces WRF `PSFC` to sub-Pa RMSE across h1/h4/h10/h24.
The offline ablation removes the flat vapor-load floor without touching
`MU/P/PH/U/V/T/QVAPOR`.

## Contract Compliance

The sprint met the contract: WRF source anchors, h1/h4 plus h10/h24 budget
proofs, offline ablation before patch, production patch, focused tests, and a
compact report. It did not use the GPU or alter the active 72h run.

## Correctness Risks

The accepted fix is a WRF-exact diagnostic fix for `PSFC`, not a full 3D
pressure-state fix. The deeper dry-balanced `P/PH/W` lane is explicitly bounded
and must not be hidden by the improved `PSFC`.

## Performance Risks

The runtime diagnostic is an elementwise column reduction in the existing M9
snapshot and uses resident metrics/state leaves. No timestep-loop host transfer
or new large transient was introduced.

## Required Fixes

Before release promotion, run a short GPU h1/h4 validation of this diagnostic
fix and then open/close the moist-cqw dynamics sprint for the 3D pressure lane.

## Decision

Decision:

ACCEPT_FIX_WITH_FOLLOWUP_BLOCKER.
