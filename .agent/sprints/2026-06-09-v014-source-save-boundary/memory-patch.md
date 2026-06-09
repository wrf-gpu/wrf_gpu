# Memory Patch Proposal

## Scope

Project memory update for the v0.14 source/save-boundary sprint.

## Evidence

- `proofs/v014/source_save_boundary_hook.json` records verdict
  `SOURCE_SAVE_BOUNDARY_HOOK_READY`.
- `proofs/v014/same_input_single_rk_parity_sources.json` records verdict
  `SOURCE_SAVE_BOUNDARY_READY_NO_JAX_WRAPPER_FULL_DOMAIN_PATCH_AND_SCALAR_OLD_LIMITER`.
- CPU-WRF emitted same-boundary dry source/save leaves at `d02` step `6000`
  before the first dry/acoustic mutation.
- Native dry state preservation versus the full pre-RK savepoint is exact on
  overlap, worst max abs `0.0`.
- The strict JAX comparison did not execute because the proof-only full-domain
  wrapper/truth surface and old-field strategy are still missing.
- `git diff -- src/gpuwrf` is empty.

## Proposed Destination

Create pending memory:

- `.agent/memory/pending/2026-06-09-v014-source-save-boundary.md`

Do not promote to stable memory until a strict same-input single-RK comparison
executes or the next proof names a deeper exact blocker.

## Patch

Record that WRF current-step dry source/save leaves are now available at a
valid pre-mutation boundary. The remaining v0.14 grid-parity blocker is no
longer missing source/save leaves, but the absence of a proof-only full-domain
JAX wrapper/truth surface and a consistent `scalar_old`/old-field strategy.

## Reviewer Status:

Pending. Accepted as sprint-local memory only.
