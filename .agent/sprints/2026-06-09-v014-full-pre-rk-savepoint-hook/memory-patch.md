# Memory Patch Proposal

## Scope

Project memory update for the v0.14 full pre-RK savepoint hook sprint.

## Evidence

- `proofs/v014/full_pre_rk_savepoint_hook.json` records verdict
  `FULL_PRE_RK_HOOK_BLOCKED_RK_FIXED_SOURCE_UNAVAILABLE_AT_STEP_ENTRY`.
- `proofs/v014/same_input_single_rk_parity_full.json` records verdict
  `FULL_PRE_RK_JAX_LOADER_BLOCKED_RK_FIXED_SOURCE_BOUNDARY`.
- CPU-WRF successfully emitted full native state at `d02` step `6000`, with
  duplicate tile overlap max delta `0.0`.
- The final proof did not execute JAX because current-step WRF source/save
  leaves are unavailable at the exact step-entry boundary.
- `git diff -- src/gpuwrf` is empty.

## Proposed Destination

Create pending memory:

- `.agent/memory/pending/2026-06-09-v014-full-pre-rk-savepoint-hook.md`

Do not promote to stable memory until the next source/save-boundary proof
confirms the correct WRF hook location.

## Patch

Record that full pre-RK native state is now available and validated, but strict
same-input single-RK parity remains blocked by missing current-step
`DryPhysicsTendencies`/save-family leaves. The next exact proof-enabling target
is a WRF boundary after source/save generation and before state mutation.

## Reviewer Status:

Pending. Accepted as sprint-local memory only.
