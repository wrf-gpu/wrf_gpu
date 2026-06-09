# Worker Report

Summary: The QVAPOR pre-call truth schema proof is complete and produced a
minimal WRF savepoint specification.

## Objective

Determine whether authoritative same-boundary WRF pre-call `QVAPOR` truth
already exists for the Step-1 live-nest theta proof.

## Files Changed

- `proofs/v014/step1_qvapor_precall_truth_schema.py`
- `proofs/v014/step1_qvapor_precall_truth_schema.json`
- `proofs/v014/step1_qvapor_precall_truth_schema.md`
- `.agent/reviews/2026-06-09-v014-step1-qvapor-precall-truth-schema.md`
- `.agent/sprints/2026-06-09-v014-step1-qvapor-precall-truth-schema/artifacts/proposed_wrf_savepoint.md`

No `src/gpuwrf/**` files were changed.

## Commands Run

- `python -m py_compile proofs/v014/step1_qvapor_precall_truth_schema.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_qvapor_precall_truth_schema.py`
- `python -m json.tool proofs/v014/step1_qvapor_precall_truth_schema.json >/tmp/step1_qvapor_precall_truth_schema.manager.validated.json`
- `git diff -- src/gpuwrf`

## Proof Objects

- `proofs/v014/step1_qvapor_precall_truth_schema.json`
- `proofs/v014/step1_qvapor_precall_truth_schema.md`
- `.agent/reviews/2026-06-09-v014-step1-qvapor-precall-truth-schema.md`
- `.agent/sprints/2026-06-09-v014-step1-qvapor-precall-truth-schema/artifacts/proposed_wrf_savepoint.md`

## Result

Final verdict:
`STEP1_QVAPOR_PRECALL_TRUTH_MISSING_SAVEPOINT_SPEC_READY`.

No accepted same-boundary `QVAPOR` truth exists at
`before_first_rk_step_part1_call`. Existing QVAPOR-bearing artifacts are
`post_after_all_rk_steps_pre_halo` / RK4 or otherwise different-boundary
artifacts and must not be reused for the pre-call theta proof.

## Next Decision

Run the proposed minimal CPU-WRF savepoint extension at the existing pre-call
hook in `dyn_em/solve_em.F::solve_em`.
