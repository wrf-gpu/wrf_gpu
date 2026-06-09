# Manager Closeout

## Outcome

The sprint is closed as a validated blocked-boundary proof.

Hook-level verdict:
`FULL_PRE_RK_HOOK_BLOCKED_RK_FIXED_SOURCE_UNAVAILABLE_AT_STEP_ENTRY`.

Final manager-facing verdict:
`FULL_PRE_RK_JAX_LOADER_BLOCKED_RK_FIXED_SOURCE_BOUNDARY`.

The WRF hook successfully produced full native step-entry state for `d02` step
`6000`, but a strict one-step same-input JAX comparison still cannot run because
WRF has not yet computed the current-step source/save leaves at that boundary.

## Proof Objects

- `proofs/v014/full_pre_rk_savepoint_hook.py`
- `proofs/v014/full_pre_rk_savepoint_hook.json`
- `proofs/v014/full_pre_rk_savepoint_hook.md`
- `proofs/v014/full_pre_rk_savepoint_hook_wrf_patch.diff`
- `proofs/v014/same_input_single_rk_parity_full.py`
- `proofs/v014/same_input_single_rk_parity_full.json`
- `proofs/v014/same_input_single_rk_parity_full.md`
- `.agent/reviews/2026-06-09-v014-full-pre-rk-savepoint-hook.md`

Key facts:

- CPU-WRF reached `2026-05-02_04:00:00` and wrote two hook files.
- Full native dry state and active moisture/scalar records are present.
- Duplicate tile overlap max delta is `0.0`.
- Patch width is not the primary blocker; one conservative valid mass cell
  remains after an 8-cell halo.
- Missing strict same-input leaves are `ru_tendf`, `rv_tendf`, `rw_tendf`,
  `ph_tendf`, `t_tendf`, `mu_tendf`, `h_diabatic`, `u_save`, `v_save`,
  `w_save`, `ph_save`, `t_save`, `moist_old`, and `scalar_old`.

## Merge Decision:

Merge proof/review/sprint artifacts only. Do not merge or authorize production
dycore/runtime edits from this sprint.

## Scope Changes

No production `src/gpuwrf/**` code changed. No GPU, TOST, Switzerland
validation, FP32, or memory source work was run.

## Lessons

The proof boundary is now narrower. We no longer lack full pre-RK native state;
we lack the exact WRF source/save-family boundary needed to feed controlled
`DryPhysicsTendencies` into JAX. The next sprint should not hunt broadly. It
should place a second WRF hook after source/save generation and before the first
state-changing dynamics update, or prove that the comparison boundary must move.

## Next Sprint

Open a WRF source/save-boundary sprint. The target is the first point where
current-step `*_tendf`, `h_diabatic`, `*_save`, `moist_old`, and `scalar_old`
exist while the initial native state remains the same state captured by the
full pre-RK hook.
