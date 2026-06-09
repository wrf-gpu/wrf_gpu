# Manager Closeout

## Outcome

Closed as:
`STEP1_RK1_P_STATE_SOURCE_REFUTED_STALE_PROOF_LOADER_BYPASS_NEXT_T_TENDF`.

The post-Mythos RK1 `P_STATE` material-source hypothesis is refuted. The prior
fresh comparator still used a proof-local loader path that bypassed Mythos'
production `start_domain` perturbation init. Once the proof applies the
production init semantics in the capture, RK1 `P_STATE` at
`after_rk_addtend_before_small_step_prep` falls from `69.96875 Pa` to
`0.0390625 Pa`, below the `1.0 Pa` material gate.

## Proof Objects

- `proofs/v014/step1_rk1_p_state_source_split.py`
- `proofs/v014/step1_rk1_p_state_source_split.json`
- `proofs/v014/step1_rk1_p_state_source_split.md`
- `.agent/reviews/2026-06-09-v014-step1-rk1-p-state-source-split.md`

## Merge Decision:

Merge the proof artifacts and sprint closeout files. No production source patch
comes from this sprint.

## Scope Changes

No TOST, Switzerland, GPU, memory/FP32 source work, Hermes, or production
CPU-WRF dependency was used. Mythos remains active on the parallel memory/FP32
worktree.

## Lessons

After the Mythos init fix, stale proof helpers can preserve old init residuals.
Future Step-1 comparators must either call the production live-nest perturbation
init path or explicitly document the patched capture.

## Next Sprint

Open a focused tendency-contract split:

- split WRF `first_rk_step_part2` `T_TENDF`;
- split RK1 `after_rk_addtend` `T_TEND/PH_TEND/RW_TEND`;
- compare against JAX `compute_advection_tendencies` and
  `_augment_large_step_tendencies` under patched-init capture;
- do not enter acoustic substeps until this earlier tendency boundary is
  explained.
