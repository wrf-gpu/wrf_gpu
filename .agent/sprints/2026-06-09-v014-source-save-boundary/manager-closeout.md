# Manager Closeout

## Outcome

The sprint is closed as a validated blocked-proof. It closes the WRF
source/save instrumentation gap, but it does not close grid parity and does not
authorize any production source edit.

Hook-level verdict:
`SOURCE_SAVE_BOUNDARY_HOOK_READY`.

Final manager-facing verdict:
`SOURCE_SAVE_BOUNDARY_READY_NO_JAX_WRAPPER_FULL_DOMAIN_PATCH_AND_SCALAR_OLD_LIMITER`.

## Proof Objects

- `proofs/v014/source_save_boundary_hook.py`
- `proofs/v014/source_save_boundary_hook.json`
- `proofs/v014/source_save_boundary_hook.md`
- `proofs/v014/source_save_boundary_hook_wrf_patch.diff`
- `proofs/v014/same_input_single_rk_parity_sources.py`
- `proofs/v014/same_input_single_rk_parity_sources.json`
- `proofs/v014/same_input_single_rk_parity_sources.md`
- `.agent/reviews/2026-06-09-v014-source-save-boundary.md`

Key facts:

- CPU-WRF completed successfully and emitted two `d02` step-6000 source/save
  hook files.
- The accepted boundary is after WRF source generation and before the first
  dry/acoustic mutation.
- Dry source/save leaves are present.
- Native dry state preservation versus the full pre-RK savepoint is exact on
  overlap, with worst max abs `0.0`.
- The strict JAX comparison did not run because the proof-only wrapper,
  full-domain same-boundary carry/boundary leaves, full-domain/full-vertical
  truth surface, and `scalar_old` strategy are still missing.

## Merge Decision:

Merge proof/review/sprint artifacts only. Do not merge or authorize production
dycore/runtime/physics edits from this sprint.

## Scope Changes

No production `src/gpuwrf/**` code changed. No GPU, TOST, Switzerland
validation, FP32, or memory source work was run.

## Lessons

The root-cause proof chain is narrower. We no longer lack WRF source/save leaves
at a valid same-input boundary. The active blocker is now proof construction:
building a full WRF-controlled JAX input and a full truth surface without mixing
JAX carry from a previous checkpoint with WRF leaves.

## Next Sprint

Open a full-domain source/save wrapper/truth-surface sprint. The proof gate is
strict execution of one same-input RK step, or a new exact blocker after emitting
the full-domain/full-vertical WRF surfaces needed by the wrapper.
