# Manager Closeout

## Outcome

The sprint is closed as a validated fail-closed tooling proof.

Final manager-facing verdict:
`SAME_INPUT_CONTRACT_BLOCKED_NO_CANDIDATE_WRF_POST_RK_PRE_HALO_TRUTH_STEP_1`.

The important progress is that the JAX/CPU side of the same-input contract is
now ready for the initial d02 case: state, tendencies, base metrics, namelist,
parent-boundary package, and initial operational carry can be constructed under
`JAX_PLATFORMS=cpu` without calling GPU-only zero helpers. The field/staggering
schema is also frozen for the 16 required comparison fields.

## Proof Objects

- `proofs/v014/same_input_contract_builder.py`
- `proofs/v014/same_input_contract_builder.json`
- `proofs/v014/same_input_contract_builder.md`
- `.agent/reviews/2026-06-09-v014-same-input-contract-builder.md`

## Merge Decision:

Merge proof, review, sprint-closeout, roadmap, and memory artifacts only. Do
not merge or authorize production dycore/runtime/physics edits from this sprint.

## Scope Changes

No production `src/gpuwrf/**` code changed. No GPU, TOST, Switzerland
validation, FP32, or memory source work was run.

## Lessons

The fastest rigorous route remains tooling-first rather than slow runtime
chasing. One major blocker from the early-step discriminator is removed
(`State.zeros`/CPU loader). The remaining first strict-comparison blocker is
specific and small: there is no full-domain CPU-WRF d02 step-1
`post_after_all_rk_steps_pre_halo` truth surface in the accepted npz contract.

## Next Sprint

Open a CPU-WRF step-1 truth-surface sprint against a disposable WRF tree. The
gate is to produce
`/mnt/data/wrf_gpu2/v014_same_input_contract_builder/wrf_truth/same_input_post_after_all_rk_steps_pre_halo_d02_step_1.npz`,
rerun `proofs/v014/same_input_contract_builder.py`, and execute the first strict
per-field residual table if the truth contract is accepted.
