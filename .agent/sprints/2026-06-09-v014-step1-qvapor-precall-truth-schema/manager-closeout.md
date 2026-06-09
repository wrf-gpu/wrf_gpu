# Manager Closeout

## Outcome

The sprint is closed as a validated schema/truth-boundary proof.

Final verdict:
`STEP1_QVAPOR_PRECALL_TRUTH_MISSING_SAVEPOINT_SPEC_READY`.

No authoritative same-boundary WRF pre-call `QVAPOR` truth exists for the
Step-1 live-nest theta proof. Existing QVAPOR artifacts are post-RK/pre-halo or
different-boundary and cannot support a production theta fix.

## Proof Objects

- `proofs/v014/step1_qvapor_precall_truth_schema.py`
- `proofs/v014/step1_qvapor_precall_truth_schema.json`
- `proofs/v014/step1_qvapor_precall_truth_schema.md`
- `.agent/reviews/2026-06-09-v014-step1-qvapor-precall-truth-schema.md`
- `.agent/sprints/2026-06-09-v014-step1-qvapor-precall-truth-schema/artifacts/proposed_wrf_savepoint.md`

## Merge Decision:

Merge proof, review, savepoint spec, sprint closeout, roadmap, and memory
updates only. No model source patch is included.

## Validation

Manager ran:

- `python -m py_compile proofs/v014/step1_qvapor_precall_truth_schema.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_qvapor_precall_truth_schema.py`
- `python -m json.tool proofs/v014/step1_qvapor_precall_truth_schema.json >/tmp/step1_qvapor_precall_truth_schema.manager.validated.json`
- `git diff -- src/gpuwrf`

The rerun reproduced the verdict, validated JSON, recorded `gpu_used=false`,
and left `src/gpuwrf` unchanged.

## Key Findings

- Accepted pre-call text files under
  `/mnt/data/wrf_gpu2/v014_step1_pre_part1_handoff/wrf_truth` do not contain
  `QVAPOR`.
- Existing QVAPOR truth exists only at `post_after_all_rk_steps_pre_halo`, not at
  the pre-call boundary.
- WRF `grid%t_2` is `theta_m - 300` at this boundary when `use_theta_m=1`;
  same-boundary `QVAPOR` is required to convert or compare dry and moist theta
  conventions.

## Next Sprint

Open a CPU-only WRF savepoint sprint that extends the existing
`before_first_rk_step_part1_call` hook in `dyn_em/solve_em.F::solve_em` to emit
`QVAPOR`, then rerun the theta proof.
